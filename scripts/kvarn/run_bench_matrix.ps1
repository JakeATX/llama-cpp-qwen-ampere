param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [string] $CaseList = "tg64:0:64,pp128:128:0,tg384:0:384",
    [int] $GpuLayers = 99,
    [ValidateSet("on", "off", "auto")] [string] $FlashAttn = "off",
    [string] $KvCacheQuant = "none,kvarn",
    [string] $KvarnPreset = "kvarn_k4v2_g128",
    [int] $KvarnIters = 16,
    [double] $RtnQuantile = 0.95,
    [int] $Repetitions = 1,
    [int] $MinKvarnLayerLogs = 1,
    [int] $MinKvarnBodyRecords = 0,
    [string] $ExpectedKvarnLayers = "",
    [ValidateSet("csv", "json", "jsonl", "md", "sql")] [string] $OutputFormat = "md",
    [string] $OutputDir = "",
    [switch] $Warmup,
    [switch] $TraceAttn,
    [int] $TraceLimit = 64,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($Repetitions -le 0) {
    throw "Repetitions must be positive"
}
if ($GpuLayers -lt 0) {
    throw "GpuLayers must be non-negative"
}
if ($KvarnIters -le 0) {
    throw "KvarnIters must be positive"
}
if ($MinKvarnLayerLogs -lt 1) {
    throw "MinKvarnLayerLogs must be positive"
}
if ($MinKvarnBodyRecords -lt 0) {
    throw "MinKvarnBodyRecords must be non-negative"
}
if ($TraceLimit -le 0) {
    throw "TraceLimit must be positive"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found at $Model"
}

$bench = Join-Path $BuildDir "bin/Release/llama-bench.exe"
if (-not (Test-Path -LiteralPath $bench)) {
    throw "llama-bench.exe not found at $bench"
}
$bench = (Resolve-Path -LiteralPath $bench).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path (Get-Location) "artifacts/kvarn-bench/$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

function Convert-ToFileStem([string] $name) {
    return ($name -replace '[^A-Za-z0-9_.-]', '_')
}

function Get-BenchCases([string] $caseList) {
    $cases = @()
    foreach ($raw in $caseList.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)) {
        $parts = $raw.Trim().Split(":", [System.StringSplitOptions]::None)
        if ($parts.Count -ne 3) {
            throw "Invalid benchmark case '$raw'; expected name:prompt_tokens:gen_tokens"
        }

        $promptTokens = 0
        $genTokens = 0
        if (-not [int]::TryParse($parts[1], [ref] $promptTokens)) {
            throw "Invalid prompt token count in case '$raw'"
        }
        if (-not [int]::TryParse($parts[2], [ref] $genTokens)) {
            throw "Invalid generation token count in case '$raw'"
        }
        if ($promptTokens -lt 0 -or $genTokens -lt 0) {
            throw "Benchmark case '$raw' has a negative token count"
        }
        if ($promptTokens -eq 0 -and $genTokens -eq 0) {
            throw "Benchmark case '$raw' must request prompt or generation tokens"
        }

        $cases += [pscustomobject]@{
            Name = $parts[0].Trim()
            PromptTokens = $promptTokens
            GenTokens = $genTokens
        }
    }

    if ($cases.Count -eq 0) {
        throw "CaseList must include at least one benchmark case"
    }
    return $cases
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

function Get-BenchThroughputByKvq([string] $text) {
    $result = @{}
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -notmatch '^\|') {
            continue
        }
        $cols = $line.Split('|') | ForEach-Object { $_.Trim() }
        if ($cols.Count -lt 10) {
            continue
        }
        $kvq = $cols[6].ToLowerInvariant()
        if ($kvq -ne "none" -and $kvq -ne "kvarn") {
            continue
        }
        $throughputText = $cols[9]
        $m = [regex]::Match($throughputText, '([0-9]+(?:\.[0-9]+)?)')
        if (-not $m.Success) {
            continue
        }
        $result[$kvq] = [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    }
    return $result
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

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers
$requiresKvarnEvidence = ($KvCacheQuant.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { $_.Trim().ToLowerInvariant() }) -contains "kvarn"
$manifest = @(
    "model=$modelPath",
    "bench=$bench",
    "build_dir=$BuildDir",
    "cases=$CaseList",
    "kv_cache_quant=$KvCacheQuant",
    "flash_attn=$FlashAttn",
    "gpu_layers=$GpuLayers",
    "kvarn_preset=$KvarnPreset",
    "kvarn_iters=$KvarnIters",
    "kvarn_rtn_quantile=$rtnQuantileArg",
    "repetitions=$Repetitions",
    "min_kvarn_layer_logs=$MinKvarnLayerLogs",
    "min_kvarn_body_records=$MinKvarnBodyRecords",
    "expected_kvarn_layers=$ExpectedKvarnLayers",
    "output_format=$OutputFormat",
    "warmup=$($Warmup.IsPresent)",
    "trace_attn=$($TraceAttn.IsPresent)",
    "trace_limit=$TraceLimit",
    "extra_args=$($ExtraArgs -join ' ')"
)
[System.IO.File]::WriteAllText((Join-Path $OutputDir "manifest.txt"), ($manifest -join "`n") + "`n")

$summaries = @()
foreach ($case in (Get-BenchCases $CaseList)) {
    $argv = @(
        "-m", $modelPath,
        "-p", [string] $case.PromptTokens,
        "-n", [string] $case.GenTokens,
        "-r", [string] $Repetitions,
        "-ngl", [string] $GpuLayers,
        "-fa", $FlashAttn,
        "-o", $OutputFormat,
        "--kv-cache-quant", $KvCacheQuant,
        "--kvarn-preset", $KvarnPreset,
        "--kvarn-iters", [string] $KvarnIters,
        "--kvarn-rtn-quantile", $rtnQuantileArg
    )
    if (-not $Warmup.IsPresent) {
        $argv += "--no-warmup"
    }
    if ($ExtraArgs.Count -gt 0) {
        $argv += $ExtraArgs
    }

    $stem = Convert-ToFileStem $case.Name
    $logPath = Join-Path $OutputDir "$stem.$OutputFormat.txt"
    $cmdPath = Join-Path $OutputDir "$stem.command.txt"
    $commandLine = "`"$bench`" " + (($argv | ForEach-Object {
        if ($_ -match '\s') {
            "`"$_`""
        } else {
            $_
        }
    }) -join " ")
    [System.IO.File]::WriteAllText($cmdPath, $commandLine + "`n")

    Write-Host "== KVarN bench case: $($case.Name) p=$($case.PromptTokens) n=$($case.GenTokens)"
    Write-Host $commandLine

    $oldErrorActionPreference = $ErrorActionPreference
    $oldTrace = [System.Environment]::GetEnvironmentVariable("LLAMA_KVARN_ATTN_TRACE", "Process")
    $oldTraceLimit = [System.Environment]::GetEnvironmentVariable("LLAMA_KVARN_ATTN_TRACE_LIMIT", "Process")
    if ($TraceAttn.IsPresent) {
        [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_ATTN_TRACE", "1", "Process")
        [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_ATTN_TRACE_LIMIT", [string] $TraceLimit, "Process")
    }
    $ErrorActionPreference = "Continue"
    try {
        $output = & $bench @argv 2>&1
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
        if ($TraceAttn.IsPresent) {
            [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_ATTN_TRACE", $oldTrace, "Process")
            [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_ATTN_TRACE_LIMIT", $oldTraceLimit, "Process")
        }
    }

    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    [System.IO.File]::WriteAllText($logPath, $text + "`n")
    $text | Write-Host
    if ($exit -ne 0) {
        throw "llama-bench failed for case '$($case.Name)' with exit code $exit; see $logPath"
    }
    if ($requiresKvarnEvidence) {
        if ($text -notmatch "llama_kv_cache_kvarn:") {
            throw "llama-bench case '$($case.Name)' succeeded but logs did not show KVarN cache initialization; see $logPath"
        }
        $kvarnLayerLogs = ([regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer")).Count
        if ($kvarnLayerLogs -lt $MinKvarnLayerLogs) {
            throw "llama-bench case '$($case.Name)' succeeded but showed only $kvarnLayerLogs KVarN layer allocation lines, expected at least $MinKvarnLayerLogs; see $logPath"
        }
        if ($text -notmatch "(?i)\bkvarn\b") {
            throw "llama-bench case '$($case.Name)' output did not include a KVarN benchmark row; see $logPath"
        }
        Assert-ExpectedKvarnLayers $text $expectedKvarnLayerIds "llama-bench case '$($case.Name)'"
        Assert-MinKvarnBodyRecords $text $MinKvarnBodyRecords "llama-bench case '$($case.Name)'"
        Write-Host ("KVarN bench log check: PASS, KVarN layer lines = {0}" -f $kvarnLayerLogs)
    }

    $throughput = Get-BenchThroughputByKvq $text
    $noneTps = if ($throughput.ContainsKey("none")) { [Nullable[double]] $throughput["none"] } else { $null }
    $kvarnTps = if ($throughput.ContainsKey("kvarn")) { [Nullable[double]] $throughput["kvarn"] } else { $null }
    $ratio = if ($noneTps -ne $null -and $kvarnTps -ne $null -and $noneTps -gt 0.0) { [Nullable[double]] ($kvarnTps/$noneTps) } else { $null }
    $traceSummary = Get-KvarnTraceSummary $text
    $summaries += [pscustomobject]@{
        Case = $case.Name
        PromptTokens = $case.PromptTokens
        GenTokens = $case.GenTokens
        NormalTps = $noneTps
        KvarnTps = $kvarnTps
        KvarnVsNormal = $ratio
        TraceModes = $traceSummary.Modes
        TraceShapes = $traceSummary.Shapes
        Log = (Split-Path -Leaf $logPath)
    }
    if ($ratio -ne $null) {
        Write-Host ("KVarN benchmark ratio: {0} = {1:P1} of normal KV" -f $case.Name, $ratio)
    }
}

$summaryCsv = Join-Path $OutputDir "summary.csv"
$summaryMd = Join-Path $OutputDir "summary.md"
$summaries | Export-Csv -NoTypeInformation -LiteralPath $summaryCsv
$summaryLines = @(
    "| case | prompt | gen | normal t/s | KVarN t/s | KVarN/normal | trace modes | log |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"
)
foreach ($s in $summaries) {
    $normalText = if ($s.NormalTps -ne $null) { "{0:F2}" -f $s.NormalTps } else { "" }
    $kvarnText = if ($s.KvarnTps -ne $null) { "{0:F2}" -f $s.KvarnTps } else { "" }
    $ratioText = if ($s.KvarnVsNormal -ne $null) { "{0:P1}" -f $s.KvarnVsNormal } else { "" }
    $traceText = if ([string]::IsNullOrWhiteSpace($s.TraceModes)) { "" } else { $s.TraceModes }
    $summaryLines += "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f `
        $s.Case, $s.PromptTokens, $s.GenTokens, $normalText, $kvarnText, $ratioText, $traceText, $s.Log
    if (-not [string]::IsNullOrWhiteSpace($s.TraceShapes)) {
        $summaryLines += ""
        $summaryLines += "Trace shapes for `{0}`: {1}" -f $s.Case, $s.TraceShapes
        $summaryLines += ""
    }
}
[System.IO.File]::WriteAllText($summaryMd, ($summaryLines -join "`n") + "`n")
Write-Host "KVarN benchmark summary: $summaryMd"
Write-Host "KVarN benchmark matrix complete: $OutputDir"
