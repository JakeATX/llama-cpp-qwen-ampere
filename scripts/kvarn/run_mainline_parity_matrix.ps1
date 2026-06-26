param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $MainlineBuildDir = (Join-Path (Get-Location) "..\llama.cpp-mainline\build-cuda-static-vs"),
    [string] $KvarnBuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $CaseList = "tg64:0:64,pp512:512:0,pp4096:4096:0,tg4096:0:4096",
    [int] $GpuLayers = 99,
    [ValidateSet("on", "off", "auto")] [string] $FlashAttn = "off",
    [ValidateSet("", "on", "off", "auto")] [string] $MainlineFlashAttn = "",
    [ValidateSet("", "on", "off", "auto")] [string] $KvarnFlashAttn = "",
    [string] $CacheTypeK = "q8_0",
    [string] $CacheTypeV = "q8_0",
    [string] $KvarnPreset = "kvarn_k4v2_g128",
    [int] $KvarnIters = 4,
    [double] $RtnQuantile = 1.0,
    [int] $Repetitions = 3,
    [switch] $Warmup,
    [double] $MinParityRatio = 0.90,
    [switch] $FailBelowMinParityRatio,
    [int] $MinKvarnLayerLogs = 1,
    [int] $MinKvarnBodyRecords = 0,
    [int] $MinActiveKvarnBodyRecords = 0,
    [string] $ExpectedKvarnLayers = "",
    [string] $ExpectedEffectiveKvarnBits = "",
    [string] $OutputDir = "",
    [switch] $TraceAttn,
    [switch] $TraceStore,
    [switch] $TraceFwht,
    [switch] $TraceDequantCache,
    [int] $TraceLimit = 64,
    [int] $TraceStoreLimit = 64,
    [int] $MinFwhtTaken = 0,
    [int] $TraceDequantCacheLimit = 256,
    [switch] $KvarnPaperFrame,
    [switch] $KvarnDirectRecordBatch,
    [switch] $RequireDirectRecordBatchPhases,
    [int] $MinBatchedStorePhaseUses = 0,
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @(),
    [switch] $AllowKvarnFallback,
    [switch] $AllowSameBinaryBaseline,
    [switch] $AllowUnsafeLlamaBench,
    [ValidateSet("mainline-first", "kvarn-first")] [string] $RunOrder = "mainline-first"
)

$ErrorActionPreference = "Stop"

if (-not $AllowUnsafeLlamaBench.IsPresent) {
    throw "run_mainline_parity_matrix.ps1 uses llama-bench.exe, which is disabled for this workspace because it has crashed the Codex session. Use scripts/kvarn/run_safe_cli_parity_matrix.ps1 for serial speed evidence, or pass -AllowUnsafeLlamaBench only for an explicit diagnostic run."
}

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
if ($MinActiveKvarnBodyRecords -lt 0) {
    throw "MinActiveKvarnBodyRecords must be non-negative"
}
if ($TraceLimit -le 0 -or $TraceStoreLimit -le 0 -or $TraceDequantCacheLimit -le 0) {
    throw "Trace limits must be positive"
}
if ($MinFwhtTaken -lt 0) {
    throw "MinFwhtTaken must be non-negative"
}
if ($MinFwhtTaken -gt 0 -and -not $TraceFwht.IsPresent) {
    throw "MinFwhtTaken requires -TraceFwht so the script can prove CUDA FWHT use"
}
if ($MinBatchedStorePhaseUses -lt 0) {
    throw "MinBatchedStorePhaseUses must be non-negative"
}
if ($MinBatchedStorePhaseUses -gt 0 -and -not $TraceStore.IsPresent) {
    throw "MinBatchedStorePhaseUses requires -TraceStore so the script can prove batched store use"
}
if ($RequireDirectRecordBatchPhases.IsPresent -and -not $KvarnPaperFrame.IsPresent) {
    throw "RequireDirectRecordBatchPhases requires -KvarnPaperFrame because the CUDA batched phase implementation is paper-frame only"
}
if ([string]::IsNullOrWhiteSpace($CacheTypeK) -or [string]::IsNullOrWhiteSpace($CacheTypeV)) {
    throw "CacheTypeK and CacheTypeV must be non-empty"
}
$effectiveMainlineFlashAttn = if ([string]::IsNullOrWhiteSpace($MainlineFlashAttn)) { $FlashAttn } else { $MainlineFlashAttn }
$effectiveKvarnFlashAttn = if ([string]::IsNullOrWhiteSpace($KvarnFlashAttn)) { $FlashAttn } else { $KvarnFlashAttn }
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
        if ($parts.Count -ne 3 -and $parts.Count -ne 4) {
            throw "Invalid case '$raw'; expected name:prompt:gen or name:prompt:gen:depth"
        }
        $cases += [pscustomobject]@{
            Name = $parts[0]
            PromptTokens = [int] $parts[1]
            GenTokens = [int] $parts[2]
            DepthTokens = $(if ($parts.Count -eq 4) { [int] $parts[3] } else { 0 })
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

function Get-GpuRuntimeSummary() {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return "nvidia-smi unavailable"
    }

    try {
        $rows = & $nvidiaSmi.Path --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or $null -eq $rows) {
            return "nvidia-smi query failed"
        }
        return (($rows | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }) -join "; ")
    } catch {
        return "nvidia-smi query failed: $($_.Exception.Message)"
    }
}

function Get-GpuMemorySnapshot() {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return "nvidia-smi unavailable"
    }

    try {
        $rows = & $nvidiaSmi.Path --query-gpu=memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or $null -eq $rows) {
            return "nvidia-smi query failed"
        }
        return (($rows | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }) -join "; ")
    } catch {
        return "nvidia-smi query failed: $($_.Exception.Message)"
    }
}

function Get-FileSha256([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return "missing"
    }
    try {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    } catch {
        return "hash failed: $($_.Exception.Message)"
    }
}

function Get-KvarnEnvSnapshot() {
    $names = @(
        "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA",
        "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK",
        "LLAMA_KVARN_LAYER_FILTER",
        "LLAMA_KVARN_LAYER_KEY_BITS",
        "LLAMA_KVARN_LAYER_VALUE_BITS",
        "LLAMA_KVARN_ATTN_TRACE",
        "LLAMA_KVARN_ATTN_TRACE_LIMIT",
        "LLAMA_KVARN_ATTN_REF_SCRATCH",
        "LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE",
        "LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR",
        "LLAMA_KVARN_DISABLE_PREFILL_DIRECT_STORE",
        "LLAMA_KVARN_DISABLE_PREFILL_DIRECT_ATTN",
        "LLAMA_KVARN_DISABLE_INTERNAL_CAUSAL_MASK",
        "LLAMA_KVARN_ENABLE_PAPER_FRAME",
        "LLAMA_KVARN_PAPER_MIXED_FRAME",
        "LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH",
        "LLAMA_KVARN_REQUIRE_DIRECT_RECORD_BATCH_PHASES",
        "LLAMA_KVARN_DISABLE_DIRECT_RECORD_BATCH_PHASES"
    )
    $rows = @()
    foreach ($name in $names) {
        $value = [Environment]::GetEnvironmentVariable($name, "Process")
        if (-not [string]::IsNullOrEmpty($value)) {
            $rows += ("{0}={1}" -f $name, $value)
        }
    }
    if ($rows.Count -eq 0) {
        return "(none)"
    }
    return ($rows -join "; ")
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

function Get-KvarnLayerSet([string] $text) {
    $actual = New-Object 'System.Collections.Generic.SortedSet[int]'
    foreach ($m in [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer\s+([0-9]+)")) {
        [void] $actual.Add([int] $m.Groups[1].Value)
    }
    return (($actual | ForEach-Object { [string] $_ }) -join ",")
}

function Normalize-KvarnBits([string] $bits) {
    return (($bits.Trim().ToLowerInvariant()) -replace "[^0-9kv]", "")
}

function Get-KvarnEffectiveBitSet([string] $text) {
    $actual = New-Object 'System.Collections.Generic.SortedSet[string]'
    foreach ($m in [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer\s+[0-9]+.*?effective k(?<k>[0-9]+)/v(?<v>[0-9]+)")) {
        [void] $actual.Add(("k{0}/v{1}" -f $m.Groups["k"].Value, $m.Groups["v"].Value))
    }
    return (($actual | ForEach-Object { [string] $_ }) -join ",")
}

function Get-KvarnTraceSummary([string] $text) {
    $modeCounts = @{}
    $shapeCounts = @{}
    $innerModeCounts = @{}
    $innerQtCounts = @{}
    $innerMirrorAllowedCounts = @{}
    $innerMirrorUsedCounts = @{}
    $innerQueryCounts = @{}
    $innerHeadCounts = @{}
    $innerHeadKvCounts = @{}
    $innerGqaCounts = @{}
    $innerSinkCounts = @{}
    $innerRecordCounts = @{}
    $innerPendingCounts = @{}
    $innerTailCounts = @{}
    $innerTailStartCounts = @{}
    $innerHeadDimCounts = @{}
    $innerTokenCounts = @{}
    $innerBlockCounts = @{}
    $innerGridCounts = @{}
    $innerShmemCounts = @{}
    $innerBodyRecordsCapCounts = @{}
    $innerMaskTypeCounts = @{}
    $innerMaskStrideQueryCounts = @{}
    $innerMaskStrideTokenCounts = @{}
    $innerScoresElemsCounts = @{}

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

    foreach ($m in [regex]::Matches(
            $text,
            "KVarN CUDA mixed-attn inner trace:\s+mode=([^\s]+)\s+head_dim=([0-9]+)\s+n_queries=([0-9]+)\s+n_head=([0-9]+)\s+n_head_kv=([0-9]+)\s+n_gqa=([0-9]+)\s+sink=([0-9]+)\s+records=([0-9]+)\s+pending=([0-9]+)\s+tail=([0-9]+)\s+tail_start=([0-9]+)\s+tokens=([0-9]+)\s+qt=([0-9]+)\s+block=([0-9]+)\s+grid=([0-9]+)\s+shmem=([0-9]+)\s+scores_nelems=([0-9]+)\s+body_records_cap=([0-9-]+)\s+body_mirror_allowed=([01])\s+body_mirror_used=([01])\s+kq_mask_type=([0-9]+)\s+kq_mask_stride_query_bytes=([0-9]+)\s+kq_mask_stride_token_bytes=([0-9]+)")) {
        $fields = @(
            @($innerModeCounts, $m.Groups[1].Value),
            @($innerQueryCounts, $m.Groups[3].Value),
            @($innerHeadCounts, $m.Groups[4].Value),
            @($innerHeadKvCounts, $m.Groups[5].Value),
            @($innerGqaCounts, $m.Groups[6].Value),
            @($innerSinkCounts, $m.Groups[7].Value),
            @($innerRecordCounts, $m.Groups[8].Value),
            @($innerPendingCounts, $m.Groups[9].Value),
            @($innerTailCounts, $m.Groups[10].Value),
            @($innerTailStartCounts, $m.Groups[11].Value),
            @($innerHeadDimCounts, $m.Groups[2].Value),
            @($innerTokenCounts, $m.Groups[12].Value),
            @($innerQtCounts, $m.Groups[13].Value),
            @($innerBlockCounts, $m.Groups[14].Value),
            @($innerGridCounts, $m.Groups[15].Value),
            @($innerShmemCounts, $m.Groups[16].Value),
            @($innerScoresElemsCounts, $m.Groups[17].Value),
            @($innerBodyRecordsCapCounts, $m.Groups[18].Value),
            @($innerMirrorAllowedCounts, $m.Groups[19].Value),
            @($innerMirrorUsedCounts, $m.Groups[20].Value),
            @($innerMaskTypeCounts, $m.Groups[21].Value),
            @($innerMaskStrideQueryCounts, $m.Groups[22].Value),
            @($innerMaskStrideTokenCounts, $m.Groups[23].Value)
        )
        foreach ($field in $fields) {
            $table = $field[0]
            $key = $field[1]
            if (-not $table.ContainsKey($key)) {
                $table[$key] = 0
            }
            $table[$key]++
        }
    }

    function Format-Counts([hashtable] $counts) {
        return ($counts.GetEnumerator() |
            Sort-Object Name |
            ForEach-Object { "{0}={1}" -f $_.Key, $_.Value }) -join "; "
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
        InnerModes = Format-Counts $innerModeCounts
        InnerNQueries = Format-Counts $innerQueryCounts
        InnerNHead = Format-Counts $innerHeadCounts
        InnerNHeadKv = Format-Counts $innerHeadKvCounts
        InnerNGqa = Format-Counts $innerGqaCounts
        InnerSink = Format-Counts $innerSinkCounts
        InnerRecords = Format-Counts $innerRecordCounts
        InnerPending = Format-Counts $innerPendingCounts
        InnerTail = Format-Counts $innerTailCounts
        InnerTailStart = Format-Counts $innerTailStartCounts
        InnerQT = Format-Counts $innerQtCounts
        InnerBodyMirrorAllowed = Format-Counts $innerMirrorAllowedCounts
        InnerBodyMirrorUsed = Format-Counts $innerMirrorUsedCounts
        InnerHeadDim = Format-Counts $innerHeadDimCounts
        InnerTokens = Format-Counts $innerTokenCounts
        InnerBlock = Format-Counts $innerBlockCounts
        InnerGrid = Format-Counts $innerGridCounts
        InnerShmem = Format-Counts $innerShmemCounts
        InnerBodyRecordsCap = Format-Counts $innerBodyRecordsCapCounts
        InnerMaskType = Format-Counts $innerMaskTypeCounts
        InnerMaskStrideQuery = Format-Counts $innerMaskStrideQueryCounts
        InnerMaskStrideToken = Format-Counts $innerMaskStrideTokenCounts
        InnerScoresElems = Format-Counts $innerScoresElemsCounts
    }
}

function Get-GitDirty([string] $RepoRoot) {
    if (-not (Test-Path -LiteralPath $RepoRoot)) {
        return "unknown"
    }
    Push-Location $RepoRoot
    try {
        $status = git status --porcelain 2>$null
        return ($(if ($status) { "true" } else { "false" }))
    } finally {
        Pop-Location
    }
}

function Get-KvarnStoreTraceSummary([string] $text) {
    $kindCounts = @{}
    $shapeCounts = @{}
    $batchedUsed = 0
    $batchedUnavailable = 0
    $batchedShapeCounts = @{}

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

    foreach ($m in [regex]::Matches($text, "KVarN CUDA store-body batched-phases trace:\s+used=([01])\s+head_dim=([0-9]+)\s+group_size=([0-9]+)\s+n_records=([0-9]+)\s+n_heads=([0-9]+)\s+scratch_floats=([0-9]+)")) {
        if ($m.Groups[1].Value -eq "1") {
            ++$batchedUsed
        } else {
            ++$batchedUnavailable
        }

        $shape = "used{0}:dim{1}/g{2}/rec{3}/heads{4}" -f `
            $m.Groups[1].Value, $m.Groups[2].Value, $m.Groups[3].Value, $m.Groups[4].Value, $m.Groups[5].Value
        if (-not $batchedShapeCounts.ContainsKey($shape)) {
            $batchedShapeCounts[$shape] = 0
        }
        $batchedShapeCounts[$shape]++
    }

    $kinds = $kindCounts.GetEnumerator() |
        Sort-Object Name |
        ForEach-Object { "{0}={1}" -f $_.Key, $_.Value }
    $shapes = $shapeCounts.GetEnumerator() |
        Sort-Object @{ Expression = { -$_.Value } }, Name |
        Select-Object -First 8 |
        ForEach-Object { "{0}x {1}" -f $_.Value, $_.Key }
    $batchedShapes = $batchedShapeCounts.GetEnumerator() |
        Sort-Object @{ Expression = { -$_.Value } }, Name |
        Select-Object -First 8 |
        ForEach-Object { "{0}x {1}" -f $_.Value, $_.Key }

    return [pscustomobject]@{
        Kinds = ($kinds -join "; ")
        Shapes = ($shapes -join "; ")
        BatchedPhaseUsed = $batchedUsed
        BatchedPhaseUnavailable = $batchedUnavailable
        BatchedPhaseShapes = ($batchedShapes -join "; ")
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

    $expectedSet = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($id in $expected) {
        [void] $expectedSet.Add($id)
    }
    $extra = @()
    foreach ($id in $actual) {
        if (-not $expectedSet.Contains($id)) {
            $extra += $id
        }
    }
    if ($extra.Count -gt 0) {
        throw "$label emitted unexpected extra KVarN layer ids: $($extra -join ',')"
    }

    Write-Host ("KVarN exact layer check: PASS, layers = {0}" -f ($expected -join ","))
}

function Assert-ExpectedEffectiveKvarnBits([string] $text, [string] $expected, [string] $label) {
    if ([string]::IsNullOrWhiteSpace($expected)) {
        return
    }

    $observed = Get-KvarnEffectiveBitSet $text
    if ([string]::IsNullOrWhiteSpace($observed)) {
        throw "$label did not emit effective KVarN bit logs; rebuild with effective k/v allocation logging before using -ExpectedEffectiveKvarnBits"
    }

    $want = Normalize-KvarnBits $expected
    $bad = @()
    foreach ($pair in ($observed -split "," | Where-Object { $_ })) {
        if ((Normalize-KvarnBits $pair) -ne $want) {
            $bad += $pair
        }
    }
    if ($bad.Count -gt 0) {
        throw "$label observed effective KVarN bits '$observed', expected every routed layer to be '$expected'"
    }

    Write-Host ("KVarN effective-bit check: PASS, bits = {0}" -f $observed)
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

function Assert-MinActiveKvarnBodyRecords([string] $text, [int] $minimum, [string] $label) {
    if ($minimum -le 0) {
        return
    }

    $maxRecords = -1
    foreach ($m in [regex]::Matches($text, "KVarN CUDA mixed-attn trace:.*?\bn_records=([0-9]+)",
            [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }
    foreach ($m in [regex]::Matches($text, "KVarN CUDA mixed-attn inner trace:.*?\brecords=([0-9]+)",
            [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }

    if ($maxRecords -lt 0) {
        throw "$label did not emit KVarN mixed-attn trace records evidence; rerun with -TraceAttn when using -MinActiveKvarnBodyRecords"
    }
    if ($maxRecords -lt $minimum) {
        throw "$label observed maximum active KVarN body records $maxRecords, expected at least $minimum"
    }

    Write-Host ("KVarN active body-record check: PASS, max active body records = {0}" -f $maxRecords)
}

function Assert-MinBatchedStorePhaseUses([object] $storeTrace, [int] $minimum, [string] $label) {
    if ($minimum -le 0) {
        return
    }

    if ($storeTrace.BatchedPhaseUsed -lt $minimum) {
        throw "$label used direct-record batched store phases $($storeTrace.BatchedPhaseUsed) times, expected at least $minimum. Rerun with -KvarnPaperFrame -KvarnDirectRecordBatch -TraceStore and inspect KVarN CUDA store-body batched-phases trace."
    }

    Write-Host ("KVarN batched store phase check: PASS, used = {0}" -f $storeTrace.BatchedPhaseUsed)
}

function Get-KvarnFwhtTraceSummary([string] $text) {
    $taken = 0
    $fallback = 0
    $total = 0
    foreach ($m in [regex]::Matches($text, "KVarN CUDA FWHT trace:\s+taken=([01])")) {
        $total++
        if ($m.Groups[1].Value -eq "1") {
            $taken++
        } else {
            $fallback++
        }
    }
    return [pscustomobject]@{
        Total = $total
        Taken = $taken
        Fallback = $fallback
    }
}

function Assert-MinFwhtTaken([object] $fwhtTrace, [int] $minimum, [string] $label) {
    if ($minimum -le 0) {
        return
    }
    if ($fwhtTrace.Taken -lt $minimum) {
        throw "$label used CUDA FWHT $($fwhtTrace.Taken) times, expected at least $minimum. Rerun with -TraceFwht and inspect KVarN CUDA FWHT trace lines."
    }
    Write-Host ("KVarN CUDA FWHT check: PASS, taken = {0}, fallback = {1}" -f $fwhtTrace.Taken, $fwhtTrace.Fallback)
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

    Write-Host "== $Label case $($Case.Name) p=$($Case.PromptTokens) n=$($Case.GenTokens) d=$($Case.DepthTokens)"
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
    $tail = @($text -split "`r?`n" | Select-Object -Last 40)
    if ($tail.Count -gt 0) {
        Write-Host "---- $Label case $($Case.Name) log tail ----"
        $tail | Write-Host
        Write-Host "---------------------------------------------"
    }
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
        Command = (Split-Path -Leaf $cmdPath)
        Text = $text
        CudaDevice = Get-CudaDeviceSummary $text
        MaxBodyRecords = Get-MaxKvarnBodyRecords $text
        AttnTrace = Get-KvarnTraceSummary $text
        StoreTrace = Get-KvarnStoreTraceSummary $text
        FwhtTrace = Get-KvarnFwhtTraceSummary $text
        DequantCacheTrace = Get-KvarnDequantCacheTraceSummary $text
    }
}

$repoRoot = (Get-Location).Path
$mainlineRoot = (Resolve-Path -LiteralPath (Join-Path $repoRoot "..\llama.cpp-mainline")).Path
$mainlineBench = Resolve-BuildExe $MainlineBuildDir "llama-bench.exe"
$kvarnBench = Resolve-BuildExe $KvarnBuildDir "llama-bench.exe"
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$mainlineBenchSha256 = Get-FileSha256 $mainlineBench
$kvarnBenchSha256 = Get-FileSha256 $kvarnBench
$sameBinaryBaseline = ([System.StringComparer]::OrdinalIgnoreCase.Equals($mainlineBench, $kvarnBench) -or
    ([string] $mainlineBenchSha256) -eq ([string] $kvarnBenchSha256))

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $repoRoot "artifacts/kvarn-mainline-parity/$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$warmupArgv = if ($Warmup.IsPresent) { @() } else { @("--no-warmup") }
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers
$evidenceMode = if ($sameBinaryBaseline) { "same-binary diagnostic" } else { "mainline-vs-kvarn production candidate" }

$manifest = @(
    "model=$modelPath",
    "mainline_repo=$mainlineRoot",
    "mainline_head=$(Get-GitHead $mainlineRoot)",
    "mainline_dirty=$(Get-GitDirty $mainlineRoot)",
    "kvarn_repo=$repoRoot",
    "kvarn_head=$(Get-GitHead $repoRoot)",
    "kvarn_dirty=$(Get-GitDirty $repoRoot)",
    "mainline_build=$MainlineBuildDir",
    "kvarn_build=$KvarnBuildDir",
    "mainline_bench=$mainlineBench",
    "mainline_bench_sha256=$mainlineBenchSha256",
    "kvarn_bench=$kvarnBench",
    "kvarn_bench_sha256=$kvarnBenchSha256",
    "same_binary_baseline=$sameBinaryBaseline",
    "allow_same_binary_baseline=$($AllowSameBinaryBaseline.IsPresent)",
    "evidence_mode=$evidenceMode",
    "gpu_runtime=$(Get-GpuRuntimeSummary)",
    "gpu_memory_snapshot=used,total,free,utilization.gpu -> $(Get-GpuMemorySnapshot)",
    "case_list=$CaseList",
    "min_parity_ratio=$MinParityRatio",
    "repetitions=$Repetitions",
    "warmup=$($Warmup.IsPresent)",
    "flash_attn=$FlashAttn",
    "mainline_flash_attn=$effectiveMainlineFlashAttn",
    "kvarn_flash_attn=$effectiveKvarnFlashAttn",
    "mainline_cache_type_k=$CacheTypeK",
    "mainline_cache_type_v=$CacheTypeV",
    "kvarn_preset=$KvarnPreset",
    "kvarn_iters=$KvarnIters",
    "kvarn_rtn_quantile=$rtnQuantileArg",
    "kvarn_env_snapshot=$(Get-KvarnEnvSnapshot)",
    "env_LLAMA_KVARN_ENABLE_PAPER_FRAME=$([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ENABLE_PAPER_FRAME', 'Process'))",
    "env_LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=$([Environment]::GetEnvironmentVariable('LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA', 'Process'))",
    "env_LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK=$([Environment]::GetEnvironmentVariable('LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK', 'Process'))",
    "min_kvarn_layer_logs=$MinKvarnLayerLogs",
    "min_kvarn_body_records=$MinKvarnBodyRecords",
    "min_active_kvarn_body_records=$MinActiveKvarnBodyRecords",
    "min_batched_store_phase_uses=$MinBatchedStorePhaseUses",
    "expected_effective_kvarn_bits=$ExpectedEffectiveKvarnBits",
    "kvarn_paper_frame=$($KvarnPaperFrame.IsPresent)",
    "kvarn_direct_record_batch=$($KvarnDirectRecordBatch.IsPresent)",
    "require_direct_record_batch_phases=$($RequireDirectRecordBatchPhases.IsPresent)",
    "expected_kvarn_layers=$ExpectedKvarnLayers",
    "trace_attn=$($TraceAttn.IsPresent)",
    "trace_store=$($TraceStore.IsPresent)",
    "trace_fwht=$($TraceFwht.IsPresent)",
    "min_fwht_taken=$MinFwhtTaken",
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

if ($sameBinaryBaseline -and -not $AllowSameBinaryBaseline.IsPresent) {
    $blockedSummary = Join-Path $OutputDir "summary.md"
    $blockedLines = @(
        "# KVarN mainline parity - BLOCKED",
        "",
        "The requested run would compare KVarN against the same binary used for the baseline.",
        "",
        "- mainline bench: ``$mainlineBench``",
        "- KVarN bench: ``$kvarnBench``",
        "- sha256: ``$mainlineBenchSha256``",
        "",
        "Use a real non-KVarN mainline build for production evidence. Pass ``-AllowSameBinaryBaseline`` only for diagnostic same-binary runs."
    )
    [System.IO.File]::WriteAllText($blockedSummary, ($blockedLines -join "`n") + "`n")
    throw "Refusing same-binary baseline; see $blockedSummary. Pass -AllowSameBinaryBaseline only for diagnostic runs."
}

$summaries = @()
$gateFailures = @()

foreach ($case in (Get-BenchCases $CaseList)) {
    $commonArgv = @(
        "-m", $modelPath,
        "-p", [string] $case.PromptTokens,
        "-n", [string] $case.GenTokens,
        "-d", [string] $case.DepthTokens,
        "-r", [string] $Repetitions,
        "-ngl", [string] $GpuLayers,
        "-o", "md"
    ) + $warmupArgv + $ExtraArgs

    $mainlineArgv = $commonArgv + @(
        "-fa", $effectiveMainlineFlashAttn,
        "-ctk", $CacheTypeK,
        "-ctv", $CacheTypeV
    )

    $kvarnArgv = $commonArgv + @(
        "-fa", $effectiveKvarnFlashAttn,
        "--kv-cache-quant", "kvarn",
        "--kvarn-preset", $KvarnPreset,
        "--kvarn-iters", [string] $KvarnIters,
        "--kvarn-rtn-quantile", $rtnQuantileArg
    ) + $KvarnExtraArgs
    $kvarnEnv = @{}
    if ($KvarnPaperFrame.IsPresent) {
        $kvarnEnv["LLAMA_KVARN_ENABLE_PAPER_FRAME"] = "1"
    }
    if ($KvarnDirectRecordBatch.IsPresent) {
        $kvarnEnv["LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH"] = "1"
    }
    if ($RequireDirectRecordBatchPhases.IsPresent) {
        $kvarnEnv["LLAMA_KVARN_REQUIRE_DIRECT_RECORD_BATCH_PHASES"] = "1"
        $kvarnEnv["LLAMA_KVARN_ENABLE_PAPER_FRAME"] = "1"
        $kvarnEnv["LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH"] = "1"
    }
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
    if ($TraceDequantCache.IsPresent) {
        $kvarnEnv["LLAMA_KVARN_DEQUANT_CACHE_TRACE"] = "1"
        $kvarnEnv["LLAMA_KVARN_DEQUANT_CACHE_TRACE_LIMIT"] = [string] $TraceDequantCacheLimit
    }

    if ($RunOrder -eq "kvarn-first") {
        $kvarnRow = Invoke-BenchRow -Label "kvarn" -BenchExe $kvarnBench -ModelPath $modelPath -Case $case -Argv $kvarnArgv -EnvSet $kvarnEnv
        $mainlineRow = Invoke-BenchRow -Label "mainline" -BenchExe $mainlineBench -ModelPath $modelPath -Case $case -Argv $mainlineArgv
    } else {
        $mainlineRow = Invoke-BenchRow -Label "mainline" -BenchExe $mainlineBench -ModelPath $modelPath -Case $case -Argv $mainlineArgv
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
        Assert-ExpectedEffectiveKvarnBits $kvarnRow.Text $ExpectedEffectiveKvarnBits "KVarN case '$($case.Name)'"
        Assert-MinKvarnBodyRecords $kvarnRow.Text $MinKvarnBodyRecords "KVarN case '$($case.Name)'"
        Assert-MinActiveKvarnBodyRecords $kvarnRow.Text $MinActiveKvarnBodyRecords "KVarN case '$($case.Name)'"
        Assert-MinBatchedStorePhaseUses $kvarnRow.StoreTrace $MinBatchedStorePhaseUses "KVarN case '$($case.Name)'"
        Assert-MinFwhtTaken $kvarnRow.FwhtTrace $MinFwhtTaken "KVarN case '$($case.Name)'"
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
        DepthTokens = $case.DepthTokens
        MainlineTps = $mainlineRow.Throughput
        KvarnTps = $kvarnRow.Throughput
        KvarnVsMainline = $ratio
        GateThreshold = $MinParityRatio
        KvarnFallbackAllowed = $AllowKvarnFallback.IsPresent
        KvarnFallbackObserved = -not $hasKvarnCache
        GatePass = $gatePass
        CudaDevice = if (-not [string]::IsNullOrWhiteSpace($kvarnRow.CudaDevice)) { $kvarnRow.CudaDevice } else { $mainlineRow.CudaDevice }
        MaxKvarnBodyRecords = $kvarnRow.MaxBodyRecords
        ExpectedKvarnLayers = $ExpectedKvarnLayers
        ActualKvarnLayers = Get-KvarnLayerSet $kvarnRow.Text
        ActualEffectiveKvarnBits = Get-KvarnEffectiveBitSet $kvarnRow.Text
        TraceModes = $kvarnRow.AttnTrace.Modes
        TraceShapes = $kvarnRow.AttnTrace.Shapes
        InnerTraceModes = $kvarnRow.AttnTrace.InnerModes
        InnerTraceNQueries = $kvarnRow.AttnTrace.InnerNQueries
        InnerTraceNHead = $kvarnRow.AttnTrace.InnerNHead
        InnerTraceNHeadKv = $kvarnRow.AttnTrace.InnerNHeadKv
        InnerTraceNGqa = $kvarnRow.AttnTrace.InnerNGqa
        InnerTraceSink = $kvarnRow.AttnTrace.InnerSink
        InnerTraceRecords = $kvarnRow.AttnTrace.InnerRecords
        InnerTracePending = $kvarnRow.AttnTrace.InnerPending
        InnerTraceTail = $kvarnRow.AttnTrace.InnerTail
        InnerTraceTailStart = $kvarnRow.AttnTrace.InnerTailStart
        InnerTraceQT = $kvarnRow.AttnTrace.InnerQT
        InnerTraceBodyMirrorAllowed = $kvarnRow.AttnTrace.InnerBodyMirrorAllowed
        InnerTraceBodyMirrorUsed = $kvarnRow.AttnTrace.InnerBodyMirrorUsed
        InnerTraceHeadDim = $kvarnRow.AttnTrace.InnerHeadDim
        InnerTraceTokens = $kvarnRow.AttnTrace.InnerTokens
        InnerTraceBlock = $kvarnRow.AttnTrace.InnerBlock
        InnerTraceGrid = $kvarnRow.AttnTrace.InnerGrid
        InnerTraceShmem = $kvarnRow.AttnTrace.InnerShmem
        InnerTraceBodyRecordsCap = $kvarnRow.AttnTrace.InnerBodyRecordsCap
        InnerTraceMaskType = $kvarnRow.AttnTrace.InnerMaskType
        InnerTraceMaskStrideQuery = $kvarnRow.AttnTrace.InnerMaskStrideQuery
        InnerTraceMaskStrideToken = $kvarnRow.AttnTrace.InnerMaskStrideToken
        InnerTraceScoresElems = $kvarnRow.AttnTrace.InnerScoresElems
        StoreTraceKinds = $kvarnRow.StoreTrace.Kinds
        StoreTraceShapes = $kvarnRow.StoreTrace.Shapes
        StoreTraceBatchedPhaseUsed = $kvarnRow.StoreTrace.BatchedPhaseUsed
        StoreTraceBatchedPhaseUnavailable = $kvarnRow.StoreTrace.BatchedPhaseUnavailable
        StoreTraceBatchedPhaseShapes = $kvarnRow.StoreTrace.BatchedPhaseShapes
        FwhtTraceTotal = $kvarnRow.FwhtTrace.Total
        FwhtTraceTaken = $kvarnRow.FwhtTrace.Taken
        FwhtTraceFallback = $kvarnRow.FwhtTrace.Fallback
        DequantCacheTrace = $kvarnRow.DequantCacheTrace
        MainlineLog = $mainlineRow.Log
        KvarnLog = $kvarnRow.Log
        MainlineCommand = $mainlineRow.Command
        KvarnCommand = $kvarnRow.Command
    }
}

$summaryCsv = Join-Path $OutputDir "summary.csv"
$summaryMd = Join-Path $OutputDir "summary.md"
$summaries | Export-Csv -NoTypeInformation -LiteralPath $summaryCsv

$summaryLines = @(
    "# KVarN parity",
    "",
    ("- Evidence mode: ``{0}``" -f $evidenceMode),
    ("- Same binary baseline: ``{0}``" -f $sameBinaryBaseline),
    ("- Model: ``{0}``" -f $modelPath),
    ("- Mainline SHA: ``{0}`` dirty=``{1}``" -f (Get-GitHead $mainlineRoot), (Get-GitDirty $mainlineRoot)),
    ("- KVarN SHA: ``{0}`` dirty=``{1}``" -f (Get-GitHead $repoRoot), (Get-GitDirty $repoRoot)),
    ("- Run order: ``{0}``; warmup: ``{1}``; flash-attn: ``{2}``" -f $RunOrder, $Warmup.IsPresent, $FlashAttn),
    ("- KVarN preset: ``{0}``; iters: ``{1}``; RTN quantile: ``{2}``" -f $KvarnPreset, $KvarnIters, $rtnQuantileArg),
    ("- KVarN paper frame: ``{0}``; direct record batch: ``{1}``; require batched phases: ``{2}``; min batched phase uses: ``{3}``" -f `
        $KvarnPaperFrame.IsPresent, $KvarnDirectRecordBatch.IsPresent, $RequireDirectRecordBatchPhases.IsPresent, $MinBatchedStorePhaseUses),
    ("- Expected KVarN layers: ``{0}``; fallback allowed: ``{1}``" -f $ExpectedKvarnLayers, $AllowKvarnFallback.IsPresent),
    ("- GPU/runtime: ``{0}``" -f (Get-GpuRuntimeSummary)),
    ("- Extra args: ``{0}``; KVarN extra args: ``{1}``" -f ($ExtraArgs -join ' '), ($KvarnExtraArgs -join ' ')),
    "",
    "| case | prompt | gen | depth | mainline t/s | KVarN t/s | KVarN/mainline | gate (>=$([int]($MinParityRatio*100))%) | fallback observed | actual layers | effective bits | max body records | batched store used | FWHT taken | commands |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |"
)
foreach ($s in $summaries) {
    $gateText = if ($s.GatePass) { "PASS" } else { "**FAIL**" }
    $commands = "{0}; {1}" -f $s.MainlineCommand, $s.KvarnCommand
    $actualLayers = if ([string]::IsNullOrWhiteSpace($s.ActualKvarnLayers)) { "(none)" } else { $s.ActualKvarnLayers }
    $actualBits = if ([string]::IsNullOrWhiteSpace($s.ActualEffectiveKvarnBits)) { "(unknown)" } else { $s.ActualEffectiveKvarnBits }
    $summaryLines += "| {0} | {1} | {2} | {3} | {4:F2} | {5:F2} | {6:P1} | {7} | {8} | {9} | {10} | {11} | {12} | {13} | {14} |" -f `
        $s.Case, $s.PromptTokens, $s.GenTokens, $s.DepthTokens, $s.MainlineTps, $s.KvarnTps, $s.KvarnVsMainline, $gateText,
        $s.KvarnFallbackObserved, $actualLayers, $actualBits, $s.MaxKvarnBodyRecords, $s.StoreTraceBatchedPhaseUsed, $s.FwhtTraceTaken, $commands
}
[System.IO.File]::WriteAllText($summaryMd, ($summaryLines -join "`n") + "`n")

Write-Host "Mainline parity summary: $summaryMd"
Write-Host "Mainline parity matrix complete: $OutputDir"

if ($gateFailures.Count -gt 0) {
    throw ("Mainline parity gate failures:`n" + ($gateFailures -join "`n"))
}
