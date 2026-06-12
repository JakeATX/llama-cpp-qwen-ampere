param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $MainlineBuildDir = (Join-Path (Get-Location) "..\llama.cpp-mainline\build-cuda-static-vs"),
    [string] $KvarnBuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $CaseList = "tg64:0:64,pp512:512:0,pp4096:4096:0,tg4096:0:4096",
    [int] $GpuLayers = 99,
    [ValidateSet("on", "off", "auto")] [string] $FlashAttn = "off",
    [string] $KvarnPreset = "kvarn_k4v2_g128",
    [int] $KvarnIters = 4,
    [double] $RtnQuantile = 1.0,
    [int] $Repetitions = 3,
    [switch] $Warmup,
    [double] $MinParityRatio = 0.90,
    [switch] $FailBelowMinParityRatio,
    [int] $MinKvarnLayerLogs = 1,
    [int] $MinKvarnBodyRecords = 0,
    [string] $ExpectedKvarnLayers = "",
    [string] $OutputDir = "",
    [switch] $TraceAttn,
    [switch] $TraceStore,
    [switch] $TraceDequantCache,
    [int] $TraceLimit = 64,
    [int] $TraceStoreLimit = 64,
    [int] $TraceDequantCacheLimit = 256,
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @(),
    [switch] $AllowKvarnFallback,
    [ValidateSet("mainline-first", "kvarn-first")] [string] $RunOrder = "mainline-first"
)

$ErrorActionPreference = "Stop"

if ($Repetitions -le 0) {
    throw "Repetitions must be positive"
}
if ($MinParityRatio -le 0.0 -or $MinParityRatio -gt 1.0) {
    throw "MinParityRatio must be in (0, 1]"
}
if ($MinKvarnLayerLogs -lt 0) {
    throw "MinKvarnLayerLogs must be non-negative"
}
if ($MinKvarnBodyRecords -lt 0) {
    throw "MinKvarnBodyRecords must be non-negative"
}
if ($TraceLimit -le 0 -or $TraceStoreLimit -le 0 -or $TraceDequantCacheLimit -le 0) {
    throw "Trace limits must be positive"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found at $Model"
}

function Resolve-BuildExe([string] $BuildDir, [string] $Name) {
    $candidates = @(
        (Join-Path $BuildDir "bin/Release/$Name"),
        (Join-Path $BuildDir "bin/$Name"),
        (Join-Path $BuildDir "bin/Release/$Name.exe"),
        (Join-Path $BuildDir "bin/$Name.exe")
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    throw "Missing $Name under $BuildDir"
}

function Get-GitHead([string] $RepoRoot) {
    if (-not (Test-Path -LiteralPath $RepoRoot)) {
        return "unknown"
    }
    Push-Location $RepoRoot
    try {
        return (git rev-parse --short HEAD 2>$null)
    } finally {
        Pop-Location
    }
}

function Get-BenchCases([string] $rawList) {
    $cases = @()
    foreach ($raw in ($rawList -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $parts = $raw -split ":"
        if ($parts.Count -ne 3) {
            throw "Invalid case '$raw'; expected name:prompt:gen"
        }
        $cases += [pscustomobject]@{
            Name = $parts[0]
            PromptTokens = [int] $parts[1]
            GenTokens = [int] $parts[2]
        }
    }
    return $cases
}

function Convert-ToFileStem([string] $name) {
    return ($name -replace '[^A-Za-z0-9_.-]+', '_')
}

function Get-BenchThroughput([string] $text) {
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -notmatch '^\|') {
            continue
        }
        $cols = $line.Split('|') | ForEach-Object { $_.Trim() }
        if ($cols.Count -lt 8) {
            continue
        }

        $nonEmptyCols = @($cols | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($nonEmptyCols.Count -lt 2) {
            continue
        }
        $throughputText = $nonEmptyCols[$nonEmptyCols.Count - 1]
        if ($throughputText -notmatch '^[0-9]') {
            continue
        }
        $m = [regex]::Match($throughputText, '([0-9]+(?:\.[0-9]+)?)')
        if ($m.Success) {
            return [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    return [double]::NaN
}

function Get-CudaDeviceSummary([string] $text) {
    $devices = @()
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -match '^\s*Device\s+[0-9]+:\s+(.+)$') {
            $devices += $Matches[1].Trim()
        }
    }
    return ($devices -join "; ")
}

function Get-MaxKvarnBodyRecords([string] $text) {
    $maxRecords = -1
    foreach ($m in [regex]::Matches($text, "body records =\s+([0-9]+)")) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }
    return $maxRecords
}

function Get-KvarnTraceSummary([string] $text) {
    $modeCounts = @{}
    $shapeCounts = @{}

    foreach ($m in [regex]::Matches(
            $text,
            "KVarN CUDA mixed-attn trace: mode=([^\s]+)\s+n_queries=([0-9]+)\s+n_head=([0-9]+)\s+n_head_kv=([0-9]+)\s+n_sink=([0-9]+)\s+n_records=([0-9]+)\s+n_pending=([0-9]+)\s+n_tail=([0-9]+).*?head_dim=([0-9]+)",
            [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $mode = $m.Groups[1].Value
        if (-not $modeCounts.ContainsKey($mode)) {
            $modeCounts[$mode] = 0
        }
        $modeCounts[$mode]++

        $shape = "{0}:q{1}/h{2}/hkv{3}/sink{4}/rec{5}/pend{6}/tail{7}/dim{8}" -f `
            $mode,
            $m.Groups[2].Value,
            $m.Groups[3].Value,
            $m.Groups[4].Value,
            $m.Groups[5].Value,
            $m.Groups[6].Value,
            $m.Groups[7].Value,
            $m.Groups[8].Value,
            $m.Groups[9].Value
        if (-not $shapeCounts.ContainsKey($shape)) {
            $shapeCounts[$shape] = 0
        }
        $shapeCounts[$shape]++
    }

    $modes = $modeCounts.GetEnumerator() |
        Sort-Object Name |
        ForEach-Object { "{0}={1}" -f $_.Key, $_.Value }
    $shapes = $shapeCounts.GetEnumerator() |
        Sort-Object @{ Expression = { -$_.Value } }, Name |
        Select-Object -First 8 |
        ForEach-Object { "{0}x {1}" -f $_.Value, $_.Key }

    return [pscustomobject]@{
        Modes = ($modes -join "; ")
        Shapes = ($shapes -join "; ")
    }
}

function Get-KvarnStoreTraceSummary([string] $text) {
    $kindCounts = @{}
    $shapeCounts = @{}

    foreach ($m in [regex]::Matches($text, "KVarN CUDA store-body trace: kind=([^\s]+)\s+head_dim=([0-9]+)\s+group_size=([0-9]+).*?scratch_floats=([0-9]+)")) {
        $kind = $m.Groups[1].Value
        if (-not $kindCounts.ContainsKey($kind)) {
            $kindCounts[$kind] = 0
        }
        $kindCounts[$kind]++

        $shape = "{0}:dim{1}/g{2}" -f $kind, $m.Groups[2].Value, $m.Groups[3].Value
        if (-not $shapeCounts.ContainsKey($shape)) {
            $shapeCounts[$shape] = 0
        }
        $shapeCounts[$shape]++
    }

    $kinds = $kindCounts.GetEnumerator() |
        Sort-Object Name |
        ForEach-Object { "{0}={1}" -f $_.Key, $_.Value }
    $shapes = $shapeCounts.GetEnumerator() |
        Sort-Object @{ Expression = { -$_.Value } }, Name |
        Select-Object -First 8 |
        ForEach-Object { "{0}x {1}" -f $_.Value, $_.Key }

    return [pscustomobject]@{
        Kinds = ($kinds -join "; ")
        Shapes = ($shapes -join "; ")
    }
}

function Get-KvarnDequantCacheTraceSummary([string] $text) {
    $counts = @{}
    foreach ($m in [regex]::Matches($text, "KVarN CUDA dequant-cache trace:\s+([^\s]+)")) {
        $kind = $m.Groups[1].Value
        if (-not $counts.ContainsKey($kind)) {
            $counts[$kind] = 0
        }
        $counts[$kind]++
    }

    return ($counts.GetEnumerator() |
        Sort-Object Name |
        ForEach-Object { "{0}={1}" -f $_.Key, $_.Value }) -join "; "
}

function Get-ExpectedKvarnLayerIds([string] $layers) {
    if ([string]::IsNullOrWhiteSpace($layers)) {
        return @()
    }

    $ids = @()
    foreach ($raw in ($layers -split "[,\s]+")) {
        if ([string]::IsNullOrWhiteSpace($raw)) {
            continue
        }
        if ($raw -match '^([0-9]+)-([0-9]+)(?::([0-9]+))?$') {
            $start = [int] $Matches[1]
            $end = [int] $Matches[2]
            $step = if ($Matches.ContainsKey(3) -and -not [string]::IsNullOrEmpty($Matches[3])) { [int] $Matches[3] } else { 1 }
            if ($end -lt $start) {
                throw "Invalid KVarN layer range '$raw' in ExpectedKvarnLayers"
            }
            if ($step -le 0) {
                throw "Invalid KVarN layer range step '$raw' in ExpectedKvarnLayers"
            }
            for ($id = $start; $id -le $end; $id += $step) {
                $ids += $id
            }
        } else {
            $id = 0
            if (-not [int]::TryParse($raw, [ref] $id) -or $id -lt 0) {
                throw "Invalid KVarN layer id '$raw' in ExpectedKvarnLayers"
            }
            $ids += $id
        }
    }
    return $ids
}

function Assert-ExpectedKvarnLayers([string] $text, [int[]] $expected, [string] $label) {
    if ($expected.Count -eq 0) {
        return
    }

    $actual = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($m in [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer\s+([0-9]+)")) {
        [void] $actual.Add([int] $m.Groups[1].Value)
    }

    $missing = @()
    foreach ($id in $expected) {
        if (-not $actual.Contains($id)) {
            $missing += $id
        }
    }
    if ($missing.Count -gt 0) {
        throw "$label missing expected KVarN layer ids: $($missing -join ',')"
    }
    Write-Host ("KVarN expected layer check: PASS, layers = {0}" -f ($expected -join ","))
}

function Assert-MinKvarnBodyRecords([string] $text, [int] $minimum, [string] $label) {
    if ($minimum -le 0) {
        return
    }

    $maxRecords = -1
    foreach ($m in [regex]::Matches($text, "body records =\s+([0-9]+)")) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }
    if ($maxRecords -lt $minimum) {
        throw "$label observed maximum KVarN body records $maxRecords, expected at least $minimum"
    }

    Write-Host ("KVarN body-record check: PASS, max body records = {0}" -f $maxRecords)
}

function Invoke-BenchRow {
    param(
        [string] $Label,
        [string] $BenchExe,
        [string] $ModelPath,
        [object] $Case,
        [string[]] $Argv,
        [hashtable] $EnvSet = @{}
    )

    $stem = Convert-ToFileStem $Case.Name
    $logPath = Join-Path $OutputDir "$stem.$Label.md.txt"
    $cmdPath = Join-Path $OutputDir "$stem.$Label.command.txt"
    $commandLine = "`"$BenchExe`" " + (($Argv | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join " ")
    [System.IO.File]::WriteAllText($cmdPath, $commandLine + "`n")

    Write-Host "== $Label case $($Case.Name) p=$($Case.PromptTokens) n=$($Case.GenTokens)"
    Write-Host $commandLine

    $oldEnv = @{}
    foreach ($key in $EnvSet.Keys) {
        $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $EnvSet[$key], "Process")
    }

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $BenchExe @Argv 2>&1
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
        foreach ($key in $EnvSet.Keys) {
            [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
        }
    }

    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    [System.IO.File]::WriteAllText($logPath, $text + "`n")
    $text | Write-Host
    if ($exit -ne 0) {
        throw "$Label llama-bench failed for case '$($Case.Name)' with exit code $exit; see $logPath"
    }

    $tps = Get-BenchThroughput $text
    if ([double]::IsNaN($tps)) {
        throw "$Label llama-bench case '$($Case.Name)' did not report throughput; see $logPath"
    }
    return [pscustomobject]@{
        Throughput = $tps
        Log = (Split-Path -Leaf $logPath)
        Text = $text
        CudaDevice = Get-CudaDeviceSummary $text
        MaxBodyRecords = Get-MaxKvarnBodyRecords $text
        AttnTrace = Get-KvarnTraceSummary $text
        StoreTrace = Get-KvarnStoreTraceSummary $text
        DequantCacheTrace = Get-KvarnDequantCacheTraceSummary $text
    }
}

$repoRoot = (Get-Location).Path
$mainlineRoot = (Resolve-Path -LiteralPath (Join-Path $repoRoot "..\llama.cpp-mainline")).Path
$mainlineBench = Resolve-BuildExe $MainlineBuildDir "llama-bench.exe"
$kvarnBench = Resolve-BuildExe $KvarnBuildDir "llama-bench.exe"
$modelPath = (Resolve-Path -LiteralPath $Model).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $repoRoot "artifacts/kvarn-mainline-parity/$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$warmupArgv = if ($Warmup.IsPresent) { @() } else { @("--no-warmup") }
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers

$manifest = @(
    "model=$modelPath",
    "mainline_repo=$mainlineRoot",
    "mainline_head=$(Get-GitHead $mainlineRoot)",
    "kvarn_repo=$repoRoot",
    "kvarn_head=$(Get-GitHead $repoRoot)",
    "mainline_build=$MainlineBuildDir",
    "kvarn_build=$KvarnBuildDir",
    "case_list=$CaseList",
    "min_parity_ratio=$MinParityRatio",
    "repetitions=$Repetitions",
    "flash_attn=$FlashAttn",
    "kvarn_preset=$KvarnPreset",
    "kvarn_iters=$KvarnIters",
    "kvarn_rtn_quantile=$rtnQuantileArg",
    "min_kvarn_layer_logs=$MinKvarnLayerLogs",
    "min_kvarn_body_records=$MinKvarnBodyRecords",
    "expected_kvarn_layers=$ExpectedKvarnLayers",
    "trace_attn=$($TraceAttn.IsPresent)",
    "trace_store=$($TraceStore.IsPresent)",
    "trace_dequant_cache=$($TraceDequantCache.IsPresent)",
    "trace_limit=$TraceLimit",
    "trace_store_limit=$TraceStoreLimit",
    "trace_dequant_cache_limit=$TraceDequantCacheLimit",
    "extra_args=$($ExtraArgs -join ' ')",
    "kvarn_extra_args=$($KvarnExtraArgs -join ' ')",
    "allow_kvarn_fallback=$($AllowKvarnFallback.IsPresent)",
    "run_order=$RunOrder"
)
[System.IO.File]::WriteAllText((Join-Path $OutputDir "manifest.txt"), ($manifest -join "`n") + "`n")

$summaries = @()
$gateFailures = @()

foreach ($case in (Get-BenchCases $CaseList)) {
    $commonArgv = @(
        "-m", $modelPath,
        "-p", [string] $case.PromptTokens,
        "-n", [string] $case.GenTokens,
        "-r", [string] $Repetitions,
        "-ngl", [string] $GpuLayers,
        "-fa", $FlashAttn,
        "-o", "md"
    ) + $warmupArgv + $ExtraArgs

    $kvarnArgv = $commonArgv + @(
        "--kv-cache-quant", "kvarn",
        "--kvarn-preset", $KvarnPreset,
        "--kvarn-iters", [string] $KvarnIters,
        "--kvarn-rtn-quantile", $rtnQuantileArg
    ) + $KvarnExtraArgs
    $kvarnEnv = @{}
    if ($TraceAttn.IsPresent) {
        $kvarnEnv["LLAMA_KVARN_ATTN_TRACE"] = "1"
        $kvarnEnv["LLAMA_KVARN_ATTN_TRACE_LIMIT"] = [string] $TraceLimit
    }
    if ($TraceStore.IsPresent) {
        $kvarnEnv["LLAMA_KVARN_STORE_TRACE"] = "1"
        $kvarnEnv["LLAMA_KVARN_STORE_TRACE_LIMIT"] = [string] $TraceStoreLimit
    }
    if ($TraceDequantCache.IsPresent) {
        $kvarnEnv["LLAMA_KVARN_DEQUANT_CACHE_TRACE"] = "1"
        $kvarnEnv["LLAMA_KVARN_DEQUANT_CACHE_TRACE_LIMIT"] = [string] $TraceDequantCacheLimit
    }

    if ($RunOrder -eq "kvarn-first") {
        $kvarnRow = Invoke-BenchRow -Label "kvarn" -BenchExe $kvarnBench -ModelPath $modelPath -Case $case -Argv $kvarnArgv -EnvSet $kvarnEnv
        $mainlineRow = Invoke-BenchRow -Label "mainline" -BenchExe $mainlineBench -ModelPath $modelPath -Case $case -Argv $commonArgv
    } else {
        $mainlineRow = Invoke-BenchRow -Label "mainline" -BenchExe $mainlineBench -ModelPath $modelPath -Case $case -Argv $commonArgv
        $kvarnRow = Invoke-BenchRow -Label "kvarn" -BenchExe $kvarnBench -ModelPath $modelPath -Case $case -Argv $kvarnArgv -EnvSet $kvarnEnv
    }

    $hasKvarnCache = $kvarnRow.Text -match "llama_kv_cache_kvarn:"
    if ($hasKvarnCache) {
        if ($MinKvarnLayerLogs -gt 0) {
            $kvarnLayerLogs = ([regex]::Matches($kvarnRow.Text, "llama_kv_cache_kvarn: KVarN layer")).Count
            if ($kvarnLayerLogs -lt $MinKvarnLayerLogs) {
                throw "KVarN case '$($case.Name)' showed only $kvarnLayerLogs layer lines, expected >= $MinKvarnLayerLogs"
            }
            Write-Host ("KVarN layer-log check: PASS, layer lines = {0}" -f $kvarnLayerLogs)
        }
        Assert-ExpectedKvarnLayers $kvarnRow.Text $expectedKvarnLayerIds "KVarN case '$($case.Name)'"
        Assert-MinKvarnBodyRecords $kvarnRow.Text $MinKvarnBodyRecords "KVarN case '$($case.Name)'"
    } elseif ($AllowKvarnFallback.IsPresent) {
        Write-Warning ("KVarN case '{0}' emitted no KVarN cache logs; treating as allowed fallback path for parity accounting" -f $case.Name)
    } else {
        throw "KVarN case '$($case.Name)' succeeded but logs did not show KVarN cache initialization. If this is the intentional Gemma fallback cell, rerun with -AllowKvarnFallback."
    }

    $ratio = $kvarnRow.Throughput / $mainlineRow.Throughput
    $gatePass = $ratio -ge $MinParityRatio
    if ($FailBelowMinParityRatio.IsPresent -and -not $gatePass) {
        $msg = "case '$($case.Name)' failed mainline parity gate: ratio={0:P1}, threshold={1:P1}" -f $ratio, $MinParityRatio
        $gateFailures += $msg
        Write-Warning $msg
    }

    Write-Host ("Mainline parity ratio: {0} = {1:P1} (kvarn {2:F2} / mainline {3:F2} t/s) [{4}]" -f `
        $case.Name, $ratio, $kvarnRow.Throughput, $mainlineRow.Throughput, ($(if ($gatePass) { "PASS" } else { "FAIL" })))

    $summaries += [pscustomobject]@{
        Case = $case.Name
        PromptTokens = $case.PromptTokens
        GenTokens = $case.GenTokens
        MainlineTps = $mainlineRow.Throughput
        KvarnTps = $kvarnRow.Throughput
        KvarnVsMainline = $ratio
        GateThreshold = $MinParityRatio
        KvarnFallbackAllowed = $AllowKvarnFallback.IsPresent -and -not $hasKvarnCache
        GatePass = $gatePass
        CudaDevice = if (-not [string]::IsNullOrWhiteSpace($kvarnRow.CudaDevice)) { $kvarnRow.CudaDevice } else { $mainlineRow.CudaDevice }
        MaxKvarnBodyRecords = $kvarnRow.MaxBodyRecords
        TraceModes = $kvarnRow.AttnTrace.Modes
        TraceShapes = $kvarnRow.AttnTrace.Shapes
        StoreTraceKinds = $kvarnRow.StoreTrace.Kinds
        StoreTraceShapes = $kvarnRow.StoreTrace.Shapes
        DequantCacheTrace = $kvarnRow.DequantCacheTrace
        MainlineLog = $mainlineRow.Log
        KvarnLog = $kvarnRow.Log
    }
}

$summaryCsv = Join-Path $OutputDir "summary.csv"
$summaryMd = Join-Path $OutputDir "summary.md"
$summaries | Export-Csv -NoTypeInformation -LiteralPath $summaryCsv

$summaryLines = @(
    "# KVarN vs mainline llama.cpp parity",
    "",
    "| case | prompt | gen | mainline t/s | KVarN t/s | KVarN/mainline | gate (>=$([int]($MinParityRatio*100))%) |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
)
foreach ($s in $summaries) {
    $gateText = if ($s.GatePass) { "PASS" } else { "**FAIL**" }
    $summaryLines += "| {0} | {1} | {2} | {3:F2} | {4:F2} | {5:P1} | {6} |" -f `
        $s.Case, $s.PromptTokens, $s.GenTokens, $s.MainlineTps, $s.KvarnTps, $s.KvarnVsMainline, $gateText
}
[System.IO.File]::WriteAllText($summaryMd, ($summaryLines -join "`n") + "`n")

Write-Host "Mainline parity summary: $summaryMd"
Write-Host "Mainline parity matrix complete: $OutputDir"

if ($gateFailures.Count -gt 0) {
    throw ("Mainline parity gate failures:`n" + ($gateFailures -join "`n"))
}
