param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [int] $Context = 512,
    [int] $GpuLayers = 99,
    [int] $Batch = 0,
    [ValidateSet("on", "off", "auto")] [string] $FlashAttn = "on",
    [double] $RtnQuantile = 0.95,
    [int] $Repeat = 32,
    [string] $PromptPhrase = "The quick brown fox studies attention kernels and cache layouts carefully. ",
    [string] $OutputFile = "",
    [string] $PromptFile = "",
    [int] $DebugUbatch = 0,
    [int] $MinKvarnLayerLogs = 1,
    [int] $MinKvarnBodyRecords = 0,
    [string] $ExpectedKvarnLayers = "",
    [switch] $PackedFusedBatch,
    [switch] $PackedSerialFused,
    [switch] $PackedSplitKernels,
    [switch] $CheckPackedRepeat,
    [switch] $CheckPackedSplit,
    [switch] $CheckNormalBaseline,
    [double] $NormalBaselineMaxNmse = -1.0,
    [switch] $TraceAttn,
    [int] $TraceLimit = 4,
    [string] $ExpectedPackedTraceMode = "",
    [switch] $SkipScratchCheck,
    [switch] $KeepArtifacts,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($Repeat -le 0) {
    throw "Repeat must be positive"
}
if ($Batch -lt 0) {
    throw "Batch must be non-negative"
}
if ($DebugUbatch -lt 0) {
    throw "DebugUbatch must be non-negative"
}
if ($MinKvarnLayerLogs -lt 1) {
    throw "MinKvarnLayerLogs must be positive"
}
if ($MinKvarnBodyRecords -lt 0) {
    throw "MinKvarnBodyRecords must be non-negative"
}
if ($NormalBaselineMaxNmse -eq 0.0) {
    throw "NormalBaselineMaxNmse must be negative to disable the threshold or positive to enforce one"
}
if ($TraceLimit -lt 0) {
    throw "TraceLimit must be non-negative"
}
if ([string]::IsNullOrWhiteSpace($OutputFile)) {
    $OutputFile = Join-Path $env:TEMP ("kvarn-packed-logits-{0}.gguf" -f ([guid]::NewGuid().ToString("N")))
}
if ([string]::IsNullOrWhiteSpace($PromptFile)) {
    $PromptFile = Join-Path $env:TEMP ("kvarn-logits-prompt-{0}.txt" -f ([guid]::NewGuid().ToString("N")))
}
if (-not [string]::IsNullOrWhiteSpace($ExpectedPackedTraceMode) -and -not $TraceAttn) {
    throw "ExpectedPackedTraceMode requires TraceAttn so the packed CUDA mode is emitted"
}
$packedModeCount = (@($PackedFusedBatch.IsPresent, $PackedSerialFused.IsPresent, $PackedSplitKernels.IsPresent) | Where-Object { $_ }).Count
if ($packedModeCount -gt 1) {
    throw "PackedFusedBatch, PackedSerialFused, and PackedSplitKernels are mutually exclusive"
}

function Add-TraceEnv([hashtable] $envSet) {
    if ($TraceAttn) {
        $envSet["LLAMA_KVARN_ATTN_TRACE"] = "1"
        $envSet["LLAMA_KVARN_ATTN_TRACE_LIMIT"] = [string] $TraceLimit
    }
}

function Get-ExePath([string] $buildDir, [string] $name) {
    $path = Join-Path $buildDir "bin/Release/$name"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "$name not found at $path"
    }
    return (Resolve-Path -LiteralPath $path).Path
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

function Assert-ExpectedPackedTraceMode([string] $text, [string] $expected, [string] $label) {
    if ([string]::IsNullOrWhiteSpace($expected)) {
        return
    }

    $modes = @()
    foreach ($m in [regex]::Matches($text, "KVarN CUDA mixed-attn trace: mode=([^\s]+)")) {
        $modes += $m.Groups[1].Value
    }
    if ($modes.Count -eq 0) {
        throw "$label did not emit a KVarN CUDA mixed-attn trace mode"
    }
    if ($modes -notcontains $expected) {
        throw "$label did not emit expected KVarN CUDA mode '$expected'; observed: $($modes -join ',')"
    }

    Write-Host ("KVarN packed trace mode check: PASS, mode = {0}" -f $expected)
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

function Invoke-Results(
        [string] $exe,
        [string[]] $argv,
        [hashtable] $envSet,
        [bool] $RequireKvarn = $true,
        [bool] $AllowFailureWithNmse = $false) {
    $oldEnv = @{}
    foreach ($key in $envSet.Keys) {
        $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $envSet[$key], "Process")
    }

    try {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & $exe @argv 2>&1
            $exit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
    } finally {
        foreach ($key in $envSet.Keys) {
            [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
        }
    }

    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    $hasNmse = [regex]::IsMatch($text, "NMSE=([0-9.eE+-]+|nan|inf|-inf)", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($exit -ne 0 -and -not ($AllowFailureWithNmse -and $hasNmse)) {
        throw "llama-results failed with exit code $exit`n$text"
    }
    if (-not $RequireKvarn) {
        return $text
    }
    if ($text -notmatch "llama_kv_cache_kvarn:") {
        throw "llama-results succeeded but logs did not show KVarN cache initialization"
    }
    $kvarnLayerLogs = ([regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer")).Count
    if ($kvarnLayerLogs -lt $MinKvarnLayerLogs) {
        throw "llama-results succeeded but showed only $kvarnLayerLogs KVarN layer allocation lines, expected at least $MinKvarnLayerLogs"
    }
    Assert-ExpectedKvarnLayers $text $expectedKvarnLayerIds "llama-results"
    Assert-MinKvarnBodyRecords $text $MinKvarnBodyRecords "llama-results"
    Write-Host ("KVarN llama-results log check: PASS, KVarN layer lines = {0}" -f $kvarnLayerLogs)

    return $text
}

function Get-Nmse([string] $text, [string] $label) {
    $match = [regex]::Match($text, "NMSE=([0-9.eE+-]+|nan|inf|-inf)", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) {
        throw "Could not find $label NMSE in llama-results output"
    }

    $value = $match.Groups[1].Value
    if ($value -ieq "nan") {
        return [double]::NaN
    }
    if ($value -ieq "inf") {
        return [double]::PositiveInfinity
    }
    if ($value -ieq "-inf") {
        return [double]::NegativeInfinity
    }

    return [double]::Parse($value, [System.Globalization.CultureInfo]::InvariantCulture)
}

$results = Get-ExePath $BuildDir "llama-results.exe"
$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers

[System.IO.File]::WriteAllText($PromptFile, ($PromptPhrase * $Repeat))
Remove-Item -LiteralPath $OutputFile -ErrorAction SilentlyContinue

$baseArgs = @(
    "-m", $Model,
    "-f", $PromptFile,
    "-o", $OutputFile,
    "-c", [string] $Context,
    "-ngl", [string] $GpuLayers,
    "-fa", $FlashAttn
)
$normalArgs = $baseArgs + @(
    "--kv-cache-quant", "none"
)
$commonArgs = $baseArgs + @(
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", "kvarn_k4v2_g128",
    "--kvarn-rtn-quantile", $rtnQuantileArg
)
if ($Batch -gt 0) {
    $commonArgs += @("-b", [string] $Batch)
}
if ($ExtraArgs.Count -gt 0) {
    $commonArgs += $ExtraArgs
}

$packedEnv = @{}
$scratchEnv = @{ "LLAMA_KVARN_ATTN_REF_SCRATCH" = "1" }
$splitEnv = @{ "LLAMA_KVARN_ATTN_SPLIT_KERNELS" = "1" }
if ($DebugUbatch -gt 0) {
    $packedEnv["LLAMA_KVARN_DEBUG_UBATCH"] = [string] $DebugUbatch
    $scratchEnv["LLAMA_KVARN_DEBUG_UBATCH"] = [string] $DebugUbatch
    $splitEnv["LLAMA_KVARN_DEBUG_UBATCH"] = [string] $DebugUbatch
}
if ($PackedSerialFused) {
    $packedEnv["LLAMA_KVARN_ATTN_SERIAL_FUSED"] = "1"
}
if ($PackedFusedBatch) {
    $packedEnv["LLAMA_KVARN_ATTN_FUSED_BATCH"] = "1"
    $packedEnv["LLAMA_KVARN_UNSAFE_ALLOW_FUSED_BATCH"] = "1"
}
if ($PackedSplitKernels) {
    $packedEnv["LLAMA_KVARN_ATTN_SPLIT_KERNELS"] = "1"
}
Add-TraceEnv $packedEnv
Add-TraceEnv $scratchEnv
Add-TraceEnv $splitEnv

try {
    if ($CheckNormalBaseline) {
        Write-Host "== Saving normal-KV baseline logits"
        [void] (Invoke-Results $results $normalArgs @{} $false)
        Write-Host "== Checking packed KVarN logits against normal KV"
        $normalCheck = Invoke-Results $results ($commonArgs + @("--check")) $packedEnv $true $true
        $normalNmse = Get-Nmse $normalCheck "packed-vs-normal"
        if ($NormalBaselineMaxNmse -gt 0.0 -and $normalNmse -gt $NormalBaselineMaxNmse) {
            throw ("KVarN packed-vs-normal logits exceeded threshold: NMSE = {0:E3}, threshold = {1:E3}" -f $normalNmse, $NormalBaselineMaxNmse)
        }
        Write-Host ("KVarN packed-vs-normal logits: INFO, NMSE = {0:E3}" -f $normalNmse)
        Remove-Item -LiteralPath $OutputFile -ErrorAction SilentlyContinue
    }

    Write-Host "== Saving packed KVarN logits"
    $packedText = Invoke-Results $results $commonArgs $packedEnv
    Assert-ExpectedPackedTraceMode $packedText $ExpectedPackedTraceMode "packed KVarN logits"
    if ($TraceAttn) {
        Write-Host $packedText
    }

    if ($CheckPackedRepeat) {
        Write-Host "== Checking packed KVarN repeat determinism"
        $repeatCheck = Invoke-Results $results ($commonArgs + @("--check")) $packedEnv
        $repeatNmse = Get-Nmse $repeatCheck "packed-repeat"
        Write-Host ("KVarN packed repeat logits: PASS, NMSE = {0:E3}" -f $repeatNmse)
    }

    if ($CheckPackedSplit) {
        Write-Host "== Checking split-kernel packed KVarN logits"
        $splitCheck = Invoke-Results $results ($commonArgs + @("--check")) $splitEnv
        if ($TraceAttn) {
            Write-Host $splitCheck
        }
        $splitNmse = Get-Nmse $splitCheck "packed-vs-split"
        Write-Host ("KVarN packed-vs-split logits: PASS, NMSE = {0:E3}" -f $splitNmse)
    }

    if (-not $SkipScratchCheck) {
        Write-Host "== Checking scratch-reference KVarN logits"
        $check = Invoke-Results $results ($commonArgs + @("--check")) $scratchEnv
        if ($TraceAttn) {
            Write-Host $check
        }
        $nmse = Get-Nmse $check "packed-vs-scratch"
        Write-Host ("KVarN packed-vs-scratch logits: PASS, NMSE = {0:E3}" -f $nmse)
    }
} finally {
    if (-not $KeepArtifacts) {
        Remove-Item -LiteralPath $OutputFile -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $PromptFile -ErrorAction SilentlyContinue
    }
}
