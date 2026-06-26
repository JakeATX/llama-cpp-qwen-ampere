param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $MainlineBuildDir = (Join-Path (Get-Location) "..\llama.cpp-mainline\build-cuda-static-vs"),
    [string] $KvarnBuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $CaseList = "pp512:512:0,tg64:1:64",
    [int] $GpuLayers = 99,
    [ValidateSet("on", "off", "auto")] [string] $FlashAttn = "off",
    [ValidateSet("", "on", "off", "auto")] [string] $MainlineFlashAttn = "",
    [ValidateSet("", "on", "off", "auto")] [string] $KvarnFlashAttn = "",
    [string] $CacheTypeK = "q8_0",
    [string] $CacheTypeV = "q8_0",
    [string] $KvarnPreset = "kvarn_k8v4_g128",
    [int] $KvarnIters = 4,
    [double] $RtnQuantile = 1.0,
    [int] $Repetitions = 3,
    [double] $MinParityRatio = 0.95,
    [switch] $FailBelowMinParityRatio,
    [int] $MinKvarnLayerLogs = 1,
    [int] $MinKvarnBodyRecords = 0,
    [int] $MinActiveKvarnBodyRecords = 0,
    [string] $ExpectedKvarnLayers = "",
    [string] $ExpectedEffectiveKvarnBits = "",
    [string] $OutputDir = "",
    [switch] $TraceAttn,
    [int] $TraceLimit = 64,
    [switch] $TraceStore,
    [int] $TraceStoreLimit = 64,
    [switch] $TraceFwht,
    [int] $MinFwhtTaken = 0,
    [switch] $KvarnPaperFrame,
    [switch] $KvarnDirectRecordBatch,
    [switch] $RequireDirectRecordBatchPhases,
    [int] $MinBatchedStorePhaseUses = 0,
    [int] $NCpuMoe = -1,
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @(),
    [switch] $AllowKvarnFallback,
    [switch] $AllowSameBinaryBaseline,
    [ValidateSet("mainline-first", "kvarn-first")] [string] $RunOrder = "mainline-first"
)

$ErrorActionPreference = "Stop"

if ($Repetitions -le 0) { throw "Repetitions must be positive" }
if ($MinParityRatio -le 0.0 -or $MinParityRatio -gt 1.0) { throw "MinParityRatio must be in (0, 1]" }
if ($MinKvarnLayerLogs -lt 0) { throw "MinKvarnLayerLogs must be non-negative" }
if ($MinKvarnBodyRecords -lt 0 -or $MinActiveKvarnBodyRecords -lt 0) { throw "MinKvarnBodyRecords and MinActiveKvarnBodyRecords must be non-negative" }
if ($TraceLimit -le 0 -or $TraceStoreLimit -le 0) { throw "Trace limits must be positive" }
if ($MinFwhtTaken -lt 0 -or $MinBatchedStorePhaseUses -lt 0) { throw "Minimum trace counts must be non-negative" }
if ($NCpuMoe -lt -1) { throw "NCpuMoe must be -1 to disable or non-negative" }
if ($MinFwhtTaken -gt 0 -and -not $TraceFwht.IsPresent) { throw "MinFwhtTaken requires -TraceFwht" }
if ($MinBatchedStorePhaseUses -gt 0 -and -not $TraceStore.IsPresent) { throw "MinBatchedStorePhaseUses requires -TraceStore" }
if ($RequireDirectRecordBatchPhases.IsPresent -and -not $KvarnPaperFrame.IsPresent) { throw "RequireDirectRecordBatchPhases requires -KvarnPaperFrame" }
if (-not (Test-Path -LiteralPath $Model)) { throw "Model not found at $Model" }

function Resolve-BuildExe([string] $BuildDir, [string] $Name) {
    $candidates = @(
        (Join-Path $BuildDir "bin/Release/$Name"),
        (Join-Path $BuildDir "bin/$Name"),
        (Join-Path $BuildDir "bin/Release/$Name.exe"),
        (Join-Path $BuildDir "bin/$Name.exe")
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) { return (Resolve-Path -LiteralPath $path).Path }
    }
    throw "Missing $Name under $BuildDir"
}

function Convert-ToFileStem([string] $Name) {
    return ($Name -replace '[^A-Za-z0-9_.-]+', '_')
}

function Convert-ToNativeArgument([string] $Arg) {
    if ($Arg -notmatch '[\s"]') { return $Arg }
    $escaped = $Arg -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Expand-StringArrayArgs([string[]] $Args) {
    $expanded = @()
    foreach ($arg in $Args) {
        if ($arg -match ',' -and $arg -notmatch '[\\/]' -and $arg -notmatch '\s') {
            $expanded += @($arg -split ',' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        } else {
            $expanded += $arg
        }
    }
    return @($expanded)
}

function Invoke-NativeCommandCaptured {
    param([string] $Exe, [string[]] $Argv, [hashtable] $EnvSet = @{})
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Exe
    $psi.Arguments = (($Argv | ForEach-Object { Convert-ToNativeArgument $_ }) -join " ")
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $oldEnv = @{}
    foreach ($key in $EnvSet.Keys) {
        $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $EnvSet[$key], "Process")
    }
    try {
        $proc = [System.Diagnostics.Process]::Start($psi)
        $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
        $stderrTask = $proc.StandardError.ReadToEndAsync()
        $proc.WaitForExit()
        return [pscustomobject]@{
            ExitCode = $proc.ExitCode
            Text = ($stdoutTask.GetAwaiter().GetResult() + $stderrTask.GetAwaiter().GetResult())
        }
    } finally {
        foreach ($key in $EnvSet.Keys) {
            [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
        }
    }
}

function Get-FileSha256([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return "missing" }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Get-GitHead([string] $RepoRoot) {
    if (-not (Test-Path -LiteralPath $RepoRoot)) { return "unknown" }
    Push-Location $RepoRoot
    try { return (git rev-parse --short HEAD 2>$null) } finally { Pop-Location }
}

function Get-GitDirty([string] $RepoRoot) {
    if (-not (Test-Path -LiteralPath $RepoRoot)) { return "unknown" }
    Push-Location $RepoRoot
    try {
        $status = git status --porcelain 2>$null
        return ($(if ($status) { "true" } else { "false" }))
    } finally {
        Pop-Location
    }
}

function Get-GpuMemorySnapshot() {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { return "nvidia-smi unavailable" }
    try {
        $rows = & $nvidiaSmi.Path --query-gpu=memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or $null -eq $rows) { return "nvidia-smi query failed" }
        return (($rows | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }) -join "; ")
    } catch {
        return "nvidia-smi query failed: $($_.Exception.Message)"
    }
}

function Assert-NoActiveModelJobs() {
    $busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -match '^(llama|test-kvarn)' -or $_.ProcessName -match 'llama|test-kvarn'
    } | Select-Object Id, ProcessName, CPU, StartTime)
    if ($busy.Count -gt 0) {
        throw ("Refusing to start a model run while another llama/test-kvarn process is active:`n" + ($busy | Format-Table -AutoSize | Out-String))
    }
}

function Get-BenchCases([string] $RawList) {
    $cases = @()
    foreach ($raw in ($RawList -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $parts = $raw -split ":"
        if ($parts.Count -ne 3) { throw "Invalid case '$raw'; expected name:prompt:gen" }
        $cases += [pscustomobject]@{
            Name = $parts[0]
            PromptTokens = [int] $parts[1]
            GenTokens = [int] $parts[2]
        }
    }
    return $cases
}

function New-PromptFile([object] $Case, [string] $Directory) {
    $tokens = [Math]::Max(1, $Case.PromptTokens)
    $words = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $tokens; ++$i) {
        [void] $words.Add("token$i")
    }
    $path = Join-Path $Directory ("prompt.{0}.txt" -f (Convert-ToFileStem $Case.Name))
    [System.IO.File]::WriteAllText($path, (($words -join " ") + "`n"))
    return $path
}

function Get-CliThroughput([string] $Text, [object] $Case) {
    $prompt = [regex]::Match($Text, "prompt eval time\s*=\s*[0-9.]+\s*ms\s*/\s*([0-9]+)\s*tokens?.*?([0-9.]+)\s*tokens per second", [System.Text.RegularExpressions.RegexOptions]::Singleline)
    $eval = [regex]::Match($Text, "(?m)^\s*llama_perf_context_print:\s*eval time\s*=\s*[0-9.]+\s*ms\s*/\s*([0-9]+)\s*(?:runs|tokens?).*?([0-9.]+)\s*tokens per second")
    if ($Case.GenTokens -gt 0) {
        if ($eval.Success) {
            return [pscustomobject]@{ Tokens = [int] $eval.Groups[1].Value; Throughput = [double]::Parse($eval.Groups[2].Value, [System.Globalization.CultureInfo]::InvariantCulture); Metric = "eval" }
        }
        throw "Could not parse eval throughput from llama-completion output"
    }
    if ($prompt.Success) {
        return [pscustomobject]@{ Tokens = [int] $prompt.Groups[1].Value; Throughput = [double]::Parse($prompt.Groups[2].Value, [System.Globalization.CultureInfo]::InvariantCulture); Metric = "prompt_eval" }
    }
    throw "Could not parse prompt-eval throughput from llama-completion output"
}

function Get-Median([double[]] $Values) {
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { return [double]::NaN }
    $mid = [int] [Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) { return $sorted[$mid] }
    return ($sorted[$mid - 1] + $sorted[$mid]) / 2.0
}

function Get-KvarnLayerSet([string] $Text) {
    $actual = New-Object 'System.Collections.Generic.SortedSet[int]'
    foreach ($m in [regex]::Matches($Text, "llama_kv_cache_kvarn: KVarN layer\s+([0-9]+)")) {
        [void] $actual.Add([int] $m.Groups[1].Value)
    }
    return (($actual | ForEach-Object { [string] $_ }) -join ",")
}

function Get-ExpectedKvarnLayerIds([string] $Layers) {
    if ([string]::IsNullOrWhiteSpace($Layers)) { return @() }
    $ids = @()
    foreach ($raw in ($Layers -split "[,\s]+")) {
        if ([string]::IsNullOrWhiteSpace($raw)) { continue }
        if ($raw -match '^([0-9]+)-([0-9]+)(?::([0-9]+))?$') {
            $start = [int] $Matches[1]
            $end = [int] $Matches[2]
            $step = if ($Matches.ContainsKey(3) -and -not [string]::IsNullOrEmpty($Matches[3])) { [int] $Matches[3] } else { 1 }
            if ($end -lt $start -or $step -le 0) { throw "Invalid KVarN layer range '$raw'" }
            for ($id = $start; $id -le $end; $id += $step) { $ids += $id }
        } else {
            $ids += [int] $raw
        }
    }
    return @($ids | Sort-Object -Unique)
}

function Assert-ExpectedKvarnLayers([string] $Text, [int[]] $Expected, [string] $Label) {
    if ($Expected.Count -eq 0) { return }
    $actual = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($m in [regex]::Matches($Text, "llama_kv_cache_kvarn: KVarN layer\s+([0-9]+)")) {
        [void] $actual.Add([int] $m.Groups[1].Value)
    }
    $missing = @()
    foreach ($id in $Expected) {
        if (-not $actual.Contains($id)) { $missing += $id }
    }
    if ($missing.Count -gt 0) { throw "$Label missing expected KVarN layers: $($missing -join ',')" }
}

function Normalize-KvarnBits([string] $Bits) {
    return (($Bits.Trim().ToLowerInvariant()) -replace "[^0-9kv]", "")
}

function Get-KvarnEffectiveBitSet([string] $Text) {
    $actual = New-Object 'System.Collections.Generic.SortedSet[string]'
    foreach ($m in [regex]::Matches($Text, "llama_kv_cache_kvarn: KVarN layer\s+[0-9]+.*?effective k(?<k>[0-9]+)/v(?<v>[0-9]+)")) {
        [void] $actual.Add(("k{0}/v{1}" -f $m.Groups["k"].Value, $m.Groups["v"].Value))
    }
    return (($actual | ForEach-Object { [string] $_ }) -join ",")
}

function Assert-ExpectedEffectiveKvarnBits([string] $Text, [string] $Expected, [string] $Label) {
    if ([string]::IsNullOrWhiteSpace($Expected)) { return }
    $observed = Get-KvarnEffectiveBitSet $Text
    if ([string]::IsNullOrWhiteSpace($observed)) { throw "$Label did not emit effective KVarN bit logs" }
    $want = Normalize-KvarnBits $Expected
    foreach ($pair in ($observed -split "," | Where-Object { $_ })) {
        if ((Normalize-KvarnBits $pair) -ne $want) {
            throw "$Label observed effective KVarN bits '$observed', expected '$Expected'"
        }
    }
}

function Get-MaxKvarnBodyRecords([string] $Text) {
    $maxRecords = -1
    foreach ($m in [regex]::Matches($Text, "body records\s*=\s*([0-9]+)")) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) { $maxRecords = $records }
    }
    return $maxRecords
}

function Get-MaxActiveKvarnBodyRecords([string] $Text) {
    $maxRecords = -1
    foreach ($m in [regex]::Matches($Text, "KVarN CUDA mixed-attn trace:.*?\bn_records=([0-9]+)", [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) { $maxRecords = $records }
    }
    foreach ($m in [regex]::Matches($Text, "KVarN CUDA mixed-attn inner trace:.*?\brecords=([0-9]+)", [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) { $maxRecords = $records }
    }
    return $maxRecords
}

function Assert-Minimum([int] $Observed, [int] $Minimum, [string] $What) {
    if ($Minimum -le 0) { return }
    if ($Observed -lt $Minimum) { throw "$What observed $Observed, expected at least $Minimum" }
}

function Get-BatchedStorePhaseUses([string] $Text) {
    $count = 0
    foreach ($m in [regex]::Matches($Text, "KVarN CUDA store-body batched-phases trace:\s+used=1")) { ++$count }
    return $count
}

function Get-FwhtTaken([string] $Text) {
    $count = 0
    foreach ($m in [regex]::Matches($Text, "KVarN CUDA FWHT trace:\s+taken=1")) { ++$count }
    return $count
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$mainlineRepo = (Resolve-Path -LiteralPath (Join-Path $repoRoot "..\llama.cpp-mainline")).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$mainlineExe = Resolve-BuildExe $MainlineBuildDir "llama-completion.exe"
$kvarnExe = Resolve-BuildExe $KvarnBuildDir "llama-completion.exe"
$mainlineSha = Get-FileSha256 $mainlineExe
$kvarnSha = Get-FileSha256 $kvarnExe
if ($mainlineSha -eq $kvarnSha -and -not $AllowSameBinaryBaseline.IsPresent) {
    throw "Mainline and KVarN llama-completion SHA256 are identical. Refusing same-binary speed evidence without -AllowSameBinaryBaseline."
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $repoRoot "artifacts/kvarn-safe-cli-parity/$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$effectiveMainlineFlashAttn = if ([string]::IsNullOrWhiteSpace($MainlineFlashAttn)) { $FlashAttn } else { $MainlineFlashAttn }
$effectiveKvarnFlashAttn = if ([string]::IsNullOrWhiteSpace($KvarnFlashAttn)) { $FlashAttn } else { $KvarnFlashAttn }
$ExtraArgs = @(Expand-StringArrayArgs $ExtraArgs)
$KvarnExtraArgs = @(Expand-StringArrayArgs $KvarnExtraArgs)
$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$expectedLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers
$cases = @(Get-BenchCases $CaseList)

$manifest = @(
    "model=$modelPath",
    "mainline_completion=$mainlineExe",
    "mainline_completion_sha256=$mainlineSha",
    "kvarn_completion=$kvarnExe",
    "kvarn_completion_sha256=$kvarnSha",
    "same_binary_baseline=$($mainlineSha -eq $kvarnSha)",
    "mainline_repo=$mainlineRepo",
    "mainline_head=$(Get-GitHead $mainlineRepo)",
    "mainline_dirty=$(Get-GitDirty $mainlineRepo)",
    "kvarn_repo=$repoRoot",
    "kvarn_head=$(Get-GitHead $repoRoot)",
    "kvarn_dirty=$(Get-GitDirty $repoRoot)",
    "case_list=$CaseList",
    "repetitions=$Repetitions",
    "min_parity_ratio=$MinParityRatio",
    "mainline_flash_attn=$effectiveMainlineFlashAttn",
    "kvarn_flash_attn=$effectiveKvarnFlashAttn",
    "mainline_cache_type_k=$CacheTypeK",
    "mainline_cache_type_v=$CacheTypeV",
    "kvarn_preset=$KvarnPreset",
    "kvarn_iters=$KvarnIters",
    "n_cpu_moe=$NCpuMoe",
    "gpu_memory_snapshot_before=$((Get-GpuMemorySnapshot))",
    "extra_args=$($ExtraArgs -join ' ')",
    "kvarn_extra_args=$($KvarnExtraArgs -join ' ')",
    "run_order=$RunOrder"
)
[System.IO.File]::WriteAllText((Join-Path $OutputDir "manifest.txt"), ($manifest -join "`n") + "`n")

function Invoke-CliCase {
    param(
        [string] $Label,
        [string] $Exe,
        [object] $Case,
        [string] $PromptFile,
        [string[]] $Argv,
        [hashtable] $EnvSet
    )
    $stem = Convert-ToFileStem ("{0}.{1}" -f $Case.Name, $Label)
    $logPath = Join-Path $OutputDir "$stem.log.txt"
    $cmdPath = Join-Path $OutputDir "$stem.command.txt"
    $throughputs = @()
    $mergedText = ""
    for ($i = 0; $i -lt $Repetitions; ++$i) {
        Assert-NoActiveModelJobs
        $nPredict = if ($Case.GenTokens -gt 0) { $Case.GenTokens } else { 1 }
        $fullArgv = @(
            "-m", $modelPath,
            "-f", $PromptFile,
            "-n", [string] $nPredict,
            "-ngl", [string] $GpuLayers,
            "-no-cnv",
            "--seed", "1",
            "--temp", "0"
        ) + $Argv
        $commandLine = "`"$Exe`" " + (($fullArgv | ForEach-Object { Convert-ToNativeArgument $_ }) -join " ")
        if ($i -eq 0) { [System.IO.File]::WriteAllText($cmdPath, $commandLine + "`n") }
        Write-Host "== $($Case.Name) $Label rep $($i + 1)/$Repetitions"
        Write-Host $commandLine
        $result = Invoke-NativeCommandCaptured -Exe $Exe -Argv $fullArgv -EnvSet $EnvSet
        $mergedText += "`n===== rep $($i + 1) =====`n" + $result.Text
        if ($result.ExitCode -ne 0) {
            [System.IO.File]::WriteAllText($logPath, $mergedText + "`n")
            throw "$Label case '$($Case.Name)' failed with exit code $($result.ExitCode); see $logPath"
        }
        [System.IO.File]::WriteAllText($logPath, $mergedText + "`n")
        $metric = Get-CliThroughput $result.Text $Case
        $throughputs += [double] $metric.Throughput
    }
    [System.IO.File]::WriteAllText($logPath, $mergedText + "`n")
    return [pscustomobject]@{
        Throughput = Get-Median $throughputs
        Log = (Split-Path -Leaf $logPath)
        Command = (Split-Path -Leaf $cmdPath)
        Text = $mergedText
    }
}

$mainlineArgv = @("-fa", $effectiveMainlineFlashAttn, "-ctk", $CacheTypeK, "-ctv", $CacheTypeV) + $ExtraArgs
$kvarnArgv = @(
    "-fa", $effectiveKvarnFlashAttn,
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", $KvarnPreset,
    "--kvarn-iters", [string] $KvarnIters,
    "--kvarn-rtn-quantile", $rtnQuantileArg
) + $ExtraArgs + $KvarnExtraArgs
if ($NCpuMoe -ge 0) {
    $mainlineArgv += @("-ncmoe", [string] $NCpuMoe)
    $kvarnArgv += @("-ncmoe", [string] $NCpuMoe)
}

$kvarnEnv = @{}
if ($TraceAttn.IsPresent) {
    $kvarnEnv["LLAMA_KVARN_ATTN_TRACE"] = "1"
    $kvarnEnv["LLAMA_KVARN_ATTN_TRACE_LIMIT"] = [string] $TraceLimit
}
if ($TraceStore.IsPresent) {
    $kvarnEnv["LLAMA_KVARN_STORE_TRACE"] = "1"
    $kvarnEnv["LLAMA_KVARN_STORE_TRACE_LIMIT"] = [string] $TraceStoreLimit
}
if ($TraceFwht.IsPresent) {
    $kvarnEnv["LLAMA_KVARN_FWHT_TRACE"] = "1"
}
if ($KvarnPaperFrame.IsPresent) {
    $kvarnEnv["LLAMA_KVARN_ENABLE_PAPER_FRAME"] = "1"
}
if ($KvarnDirectRecordBatch.IsPresent) {
    $kvarnEnv["LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH"] = "1"
}
if ($RequireDirectRecordBatchPhases.IsPresent) {
    $kvarnEnv["LLAMA_KVARN_REQUIRE_DIRECT_RECORD_BATCH_PHASES"] = "1"
}

$summaries = @()
foreach ($case in $cases) {
    $promptFile = New-PromptFile $case $OutputDir
    if ($RunOrder -eq "kvarn-first") {
        $kvarn = Invoke-CliCase "kvarn" $kvarnExe $case $promptFile $kvarnArgv $kvarnEnv
        $mainline = Invoke-CliCase "mainline" $mainlineExe $case $promptFile $mainlineArgv @{}
    } else {
        $mainline = Invoke-CliCase "mainline" $mainlineExe $case $promptFile $mainlineArgv @{}
        $kvarn = Invoke-CliCase "kvarn" $kvarnExe $case $promptFile $kvarnArgv $kvarnEnv
    }

    $kvarnText = $kvarn.Text
    $fallbackObserved = ($kvarnText -match "KVarN fallback|normal ISWA fallback")
    if ($fallbackObserved -and -not $AllowKvarnFallback.IsPresent) {
        throw "KVarN fallback observed for case '$($case.Name)'"
    }
    if (($kvarnText -notmatch "llama_kv_cache_kvarn:") -and -not $AllowKvarnFallback.IsPresent) {
        throw "KVarN case '$($case.Name)' did not initialize KVarN cache"
    }
    $layerLogs = [regex]::Matches($kvarnText, "llama_kv_cache_kvarn: KVarN layer").Count
    if ($layerLogs -lt $MinKvarnLayerLogs -and -not $AllowKvarnFallback.IsPresent) {
        throw "KVarN case '$($case.Name)' emitted $layerLogs KVarN layer logs, expected at least $MinKvarnLayerLogs"
    }
    Assert-ExpectedKvarnLayers $kvarnText $expectedLayerIds "KVarN case '$($case.Name)'"
    Assert-ExpectedEffectiveKvarnBits $kvarnText $ExpectedEffectiveKvarnBits "KVarN case '$($case.Name)'"
    $maxBodyRecords = Get-MaxKvarnBodyRecords $kvarnText
    $maxActiveBodyRecords = Get-MaxActiveKvarnBodyRecords $kvarnText
    Assert-Minimum $maxBodyRecords $MinKvarnBodyRecords "KVarN body records for case '$($case.Name)'"
    Assert-Minimum $maxActiveBodyRecords $MinActiveKvarnBodyRecords "KVarN active body records for case '$($case.Name)'"
    $batchedStoreUses = Get-BatchedStorePhaseUses $kvarnText
    Assert-Minimum $batchedStoreUses $MinBatchedStorePhaseUses "KVarN batched store phase uses for case '$($case.Name)'"
    $fwhtTaken = Get-FwhtTaken $kvarnText
    Assert-Minimum $fwhtTaken $MinFwhtTaken "KVarN CUDA FWHT uses for case '$($case.Name)'"

    $ratio = $kvarn.Throughput / $mainline.Throughput
    $gate = if ($ratio -ge $MinParityRatio) { "PASS" } else { "FAIL" }
    if ($FailBelowMinParityRatio.IsPresent -and $ratio -lt $MinParityRatio) {
        throw "case '$($case.Name)' failed safe CLI parity gate: ratio=$($ratio.ToString('P1', [System.Globalization.CultureInfo]::InvariantCulture)), threshold=$($MinParityRatio.ToString('P1', [System.Globalization.CultureInfo]::InvariantCulture))"
    }
    $summaries += [pscustomobject]@{
        case = $case.Name
        prompt = $case.PromptTokens
        gen = $case.GenTokens
        mainline_tps = $mainline.Throughput
        kvarn_tps = $kvarn.Throughput
        ratio = $ratio
        gate = $gate
        actual_layers = Get-KvarnLayerSet $kvarnText
        effective_bits = Get-KvarnEffectiveBitSet $kvarnText
        max_body_records = $maxBodyRecords
        max_active_body_records = $maxActiveBodyRecords
        batched_store_uses = $batchedStoreUses
        fwht_taken = $fwhtTaken
        commands = "$($mainline.Command); $($kvarn.Command)"
    }
}

$csvPath = Join-Path $OutputDir "summary.csv"
$mdPath = Join-Path $OutputDir "summary.md"
$summaries | Export-Csv -NoTypeInformation -LiteralPath $csvPath
$lines = @(
    "# KVarN safe CLI parity",
    "",
    ("- Evidence mode: ``{0}``" -f "distinct-binary llama-completion timings"),
    ("- Model: ``{0}``" -f $modelPath),
    ("- Mainline SHA256: ``{0}``" -f $mainlineSha),
    ("- KVarN SHA256: ``{0}``" -f $kvarnSha),
    ("- Same binary baseline: ``{0}``" -f ($mainlineSha -eq $kvarnSha)),
    ("- GPU memory before: ``{0}``" -f (Get-GpuMemorySnapshot)),
    "",
    "| case | prompt | gen | mainline t/s | KVarN t/s | KVarN/mainline | gate | actual layers | effective bits | max body records | max active body records | batched store uses | FWHT taken | commands |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"
)
foreach ($s in $summaries) {
    $lines += "| {0} | {1} | {2} | {3:F2} | {4:F2} | {5:P1} | {6} | {7} | {8} | {9} | {10} | {11} | {12} | {13} |" -f `
        $s.case, $s.prompt, $s.gen, $s.mainline_tps, $s.kvarn_tps, $s.ratio, $s.gate, $s.actual_layers, $s.effective_bits, $s.max_body_records, $s.max_active_body_records, $s.batched_store_uses, $s.fwht_taken, $s.commands
}
[System.IO.File]::WriteAllText($mdPath, ($lines -join "`n") + "`n")
Write-Host "Safe CLI parity summary: $mdPath"
