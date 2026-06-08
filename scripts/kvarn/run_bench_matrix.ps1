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
    [double] $MinKvarnRatio = -1.0,
    [switch] $FailBelowMinKvarnRatio,
    [switch] $AllowKvarnFallback,
    [int] $TraceLimit = 64,
    [switch] $TraceStore,
    [int] $TraceStoreLimit = 64,
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
if ($MinKvarnRatio -eq 0.0 -or $MinKvarnRatio -gt 1.0) {
    throw "MinKvarnRatio must be negative to disable the gate or in (0, 1] to enforce one"
}
if ($FailBelowMinKvarnRatio.IsPresent -and $MinKvarnRatio -lt 0.0) {
    throw "FailBelowMinKvarnRatio requires MinKvarnRatio to be set"
}
if ($TraceStoreLimit -le 0) {
    throw "TraceStoreLimit must be positive"
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

function Get-NullableDoubleValue($value) {
    if ($null -eq $value) {
        return $null
    }
    if ($value -is [Nullable[double]]) {
        if (-not $value.HasValue) {
            return $null
        }
        return [double] $value.Value
    }
    return [double] $value
}

function Format-NullableDouble($value, [string] $format) {
    $d = Get-NullableDoubleValue $value
    if ($null -eq $d) {
        return ""
    }
    return $d.ToString($format, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-RatioPercent($value) {
    $d = Get-NullableDoubleValue $value
    if ($null -eq $d) {
        return ""
    }
    return $d.ToString("P1", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Test-KvarnEvidence([string] $text) {
    return ($text -match "llama_kv_cache_kvarn:")
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

        $kvqIdx = -1
        for ($i = 0; $i -lt $cols.Count; $i++) {
            $col = $cols[$i].ToLowerInvariant()
            if ($col -eq "none" -or $col -eq "kvarn") {
                $kvqIdx = $i
                break
            }
        }
        if ($kvqIdx -lt 0) {
            continue
        }

        $kvq = $cols[$kvqIdx].ToLowerInvariant()
        $nonEmptyCols = @($cols | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($nonEmptyCols.Count -lt 2) {
            continue
        }
        $throughputText = $nonEmptyCols[$nonEmptyCols.Count - 1]
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

function Get-KvarnStoreTraceSummary([string] $text) {
    $kindCounts = @{}
    $shapeCounts = @{}

    foreach ($m in [regex]::Matches(
            $text,
            "KVarN CUDA store-body trace: kind=([kv])\s+head_dim=([0-9]+)\s+group_size=([0-9]+)\s+bits=([0-9]+)\s+sinkhorn_iters=([0-9]+)\s+rtn_quantile=([0-9.eE+-]+)\s+body_bytes=([0-9]+)\s+scale_floats=([0-9]+)\s+scratch_floats=([0-9]+)")) {
        $kind = $m.Groups[1].Value
        if (-not $kindCounts.ContainsKey($kind)) {
            $kindCounts[$kind] = 0
        }
        $kindCounts[$kind]++

        $shape = "{0}:dim{1}/g{2}/bits{3}/iters{4}/rtn{5}" -f `
            $kind,
            $m.Groups[2].Value,
            $m.Groups[3].Value,
            $m.Groups[4].Value,
            $m.Groups[5].Value,
            $m.Groups[6].Value
        if (-not $shapeCounts.ContainsKey($shape)) {
            $shapeCounts[$shape] = 0
        }
        $shapeCounts[$shape]++
    }

    foreach ($m in [regex]::Matches(
            $text,
            "KVarN CUDA store-body trace: kind=kv\s+head_dim=([0-9]+)\s+group_size=([0-9]+)\s+key_bits=([0-9]+)\s+value_bits=([0-9]+)\s+sinkhorn_iters=([0-9]+)\s+rtn_quantile=([0-9.eE+-]+)\s+k_body_bytes=([0-9]+)\s+v_body_bytes=([0-9]+)\s+k_scale_floats=([0-9]+)\s+v_scale_floats=([0-9]+)\s+scratch_floats=([0-9]+)")) {
        if (-not $kindCounts.ContainsKey("kv")) {
            $kindCounts["kv"] = 0
        }
        $kindCounts["kv"]++

        $shape = "kv:dim{0}/g{1}/bits{2}+{3}/iters{4}/rtn{5}" -f `
            $m.Groups[1].Value,
            $m.Groups[2].Value,
            $m.Groups[3].Value,
            $m.Groups[4].Value,
            $m.Groups[5].Value,
            $m.Groups[6].Value
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

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers
$requiresKvarnEvidence = ($KvCacheQuant.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { $_.Trim().ToLowerInvariant() }) -contains "kvarn"
$requiresKvarnEvidence = $requiresKvarnEvidence -and -not $AllowKvarnFallback.IsPresent
$kvqIncludesKvarn = ($KvCacheQuant.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) |
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
    "trace_store=$($TraceStore.IsPresent)",
    "trace_store_limit=$TraceStoreLimit",
    "min_kvarn_ratio=$MinKvarnRatio",
    "fail_below_min_kvarn_ratio=$($FailBelowMinKvarnRatio.IsPresent)",
    "allow_kvarn_fallback=$($AllowKvarnFallback.IsPresent)",
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
    $oldStoreTrace = [System.Environment]::GetEnvironmentVariable("LLAMA_KVARN_STORE_TRACE", "Process")
    $oldStoreTraceLimit = [System.Environment]::GetEnvironmentVariable("LLAMA_KVARN_STORE_TRACE_LIMIT", "Process")
    if ($TraceAttn.IsPresent) {
        [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_ATTN_TRACE", "1", "Process")
        [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_ATTN_TRACE_LIMIT", [string] $TraceLimit, "Process")
    }
    if ($TraceStore.IsPresent) {
        [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_STORE_TRACE", "1", "Process")
        [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_STORE_TRACE_LIMIT", [string] $TraceStoreLimit, "Process")
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
        if ($TraceStore.IsPresent) {
            [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_STORE_TRACE", $oldStoreTrace, "Process")
            [System.Environment]::SetEnvironmentVariable("LLAMA_KVARN_STORE_TRACE_LIMIT", $oldStoreTraceLimit, "Process")
        }
    }

    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    [System.IO.File]::WriteAllText($logPath, $text + "`n")
    $text | Write-Host
    if ($exit -ne 0) {
        throw "llama-bench failed for case '$($case.Name)' with exit code $exit; see $logPath"
    }
    $hasKvarnEvidence = Test-KvarnEvidence $text
    if ($requiresKvarnEvidence) {
        if (-not $hasKvarnEvidence) {
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
    } elseif ($kvqIncludesKvarn -and -not $hasKvarnEvidence) {
        Write-Warning ("llama-bench case '{0}' did not initialize KVarN tensors; treating the KVarN row as production fallback because AllowKvarnFallback was set" -f $case.Name)
    } elseif ($kvqIncludesKvarn -and $hasKvarnEvidence) {
        $kvarnLayerLogs = ([regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer")).Count
        Write-Host ("KVarN bench log check: INFO, KVarN layer lines = {0}" -f $kvarnLayerLogs)
    }

    $throughput = Get-BenchThroughputByKvq $text
    $noneVal = if ($throughput.ContainsKey("none")) { [double] $throughput["none"] } else { [double]::NaN }
    $kvarnVal = if ($throughput.ContainsKey("kvarn")) { [double] $throughput["kvarn"] } else { [double]::NaN }
    $ratioVal = if (-not [double]::IsNaN($noneVal) -and -not [double]::IsNaN($kvarnVal) -and $noneVal -gt 0.0) {
        $kvarnVal / $noneVal
    } else {
        [double]::NaN
    }
    $noneTps = if (-not [double]::IsNaN($noneVal)) { [Nullable[double]] $noneVal } else { $null }
    $kvarnTps = if (-not [double]::IsNaN($kvarnVal)) { [Nullable[double]] $kvarnVal } else { $null }
    $ratio = if (-not [double]::IsNaN($ratioVal)) { [Nullable[double]] $ratioVal } else { $null }
    $backendMode = if ($kvqIncludesKvarn -and $hasKvarnEvidence) {
        "kvarn"
    } elseif ($kvqIncludesKvarn -and $AllowKvarnFallback.IsPresent) {
        "normal-kv-fallback"
    } else {
        "normal"
    }
    $gatePassBool = $false
    if ($MinKvarnRatio -gt 0.0 -and -not [double]::IsNaN($ratioVal)) {
        $gatePassBool = ($ratioVal -ge $MinKvarnRatio)
    }
    if ($AllowKvarnFallback.IsPresent -and $backendMode -eq "normal-kv-fallback") {
        $gatePassBool = $true
    }
    $gatePass = if ($MinKvarnRatio -gt 0.0) { [Nullable[bool]] $gatePassBool } else { $null }
    if ($MinKvarnRatio -gt 0.0 -and [double]::IsNaN($ratioVal)) {
        throw "llama-bench case '$($case.Name)' could not compute KVarN/normal ratio for production gate; see $logPath"
    }
    if ($FailBelowMinKvarnRatio.IsPresent -and -not $gatePassBool) {
        throw ("llama-bench case '{0}' failed KVarN production ratio gate: ratio={1:P1}, threshold={2:P1}; see {3}" -f $case.Name, $ratioVal, $MinKvarnRatio, $logPath)
    }
    $traceSummary = Get-KvarnTraceSummary $text
    $storeTraceSummary = Get-KvarnStoreTraceSummary $text
    $summaries += [pscustomobject]@{
        Case = $case.Name
        BackendMode = $backendMode
        PromptTokens = $case.PromptTokens
        GenTokens = $case.GenTokens
        NormalTps = $noneTps
        KvarnTps = $kvarnTps
        KvarnVsNormal = $ratio
        GateThreshold = if ($MinKvarnRatio -gt 0.0) { [Nullable[double]] $MinKvarnRatio } else { $null }
        GatePass = $gatePass
        GatePassBool = $gatePassBool
        TraceModes = $traceSummary.Modes
        TraceShapes = $traceSummary.Shapes
        StoreTraceKinds = $storeTraceSummary.Kinds
        StoreTraceShapes = $storeTraceSummary.Shapes
        Log = (Split-Path -Leaf $logPath)
    }
    if (-not [double]::IsNaN($ratioVal)) {
        Write-Host ("KVarN benchmark ratio: {0} = {1:P1} of normal KV" -f $case.Name, $ratioVal)
    }
}

$summaryCsv = Join-Path $OutputDir "summary.csv"
$summaryMd = Join-Path $OutputDir "summary.md"
$summaries | Export-Csv -NoTypeInformation -LiteralPath $summaryCsv
$summaryLines = @(
    "| case | backend mode | prompt | gen | normal t/s | KVarN t/s | KVarN/normal | gate | trace modes | store traces | log |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"
)
foreach ($s in $summaries) {
    $normalText = Format-NullableDouble $s.NormalTps "F2"
    $kvarnText = Format-NullableDouble $s.KvarnTps "F2"
    $ratioText = Format-RatioPercent $s.KvarnVsNormal
    $gateText = if ($MinKvarnRatio -lt 0.0) { "" } elseif ($s.GatePassBool) { "PASS" } else { "FAIL" }
    $traceText = if ([string]::IsNullOrWhiteSpace($s.TraceModes)) { "" } else { $s.TraceModes }
    $storeTraceText = if ([string]::IsNullOrWhiteSpace($s.StoreTraceKinds)) { "" } else { $s.StoreTraceKinds }
    $summaryLines += "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} |" -f `
        $s.Case, $s.BackendMode, $s.PromptTokens, $s.GenTokens, $normalText, $kvarnText, $ratioText, $gateText, $traceText, $storeTraceText, $s.Log
    if (-not [string]::IsNullOrWhiteSpace($s.TraceShapes)) {
        $summaryLines += ""
        $summaryLines += "Trace shapes for `{0}`: {1}" -f $s.Case, $s.TraceShapes
        $summaryLines += ""
    }
    if (-not [string]::IsNullOrWhiteSpace($s.StoreTraceShapes)) {
        $summaryLines += ""
        $summaryLines += "Store trace shapes for `{0}`: {1}" -f $s.Case, $s.StoreTraceShapes
        $summaryLines += ""
    }
}
[System.IO.File]::WriteAllText($summaryMd, ($summaryLines -join "`n") + "`n")
Write-Host "KVarN benchmark summary: $summaryMd"
Write-Host "KVarN benchmark matrix complete: $OutputDir"
