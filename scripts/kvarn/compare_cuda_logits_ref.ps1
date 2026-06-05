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
    [string] $OutputFile = (Join-Path $env:TEMP "kvarn-packed-logits.gguf"),
    [string] $PromptFile = (Join-Path $env:TEMP "kvarn-logits-prompt.txt"),
    [int] $DebugUbatch = 0,
    [switch] $PackedFusedBatch,
    [switch] $PackedSerialFused,
    [switch] $PackedSplitKernels,
    [switch] $CheckPackedRepeat,
    [switch] $CheckPackedSplit,
    [switch] $TraceAttn,
    [int] $TraceLimit = 4,
    [switch] $KeepArtifacts
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
if ($TraceLimit -lt 0) {
    throw "TraceLimit must be non-negative"
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

function Invoke-Results([string] $exe, [string[]] $argv, [hashtable] $envSet) {
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
    if ($exit -ne 0) {
        throw "llama-results failed with exit code $exit`n$text"
    }
    if ($text -notmatch "llama_kv_cache_kvarn:") {
        throw "llama-results succeeded but logs did not show KVarN cache initialization"
    }
    $kvarnLayerLogs = ([regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer")).Count
    if ($kvarnLayerLogs -lt 1) {
        throw "llama-results succeeded but did not show any KVarN layer allocation lines"
    }
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

[System.IO.File]::WriteAllText($PromptFile, ($PromptPhrase * $Repeat))
Remove-Item -LiteralPath $OutputFile -ErrorAction SilentlyContinue

$commonArgs = @(
    "-m", $Model,
    "-f", $PromptFile,
    "-o", $OutputFile,
    "-c", [string] $Context,
    "-ngl", [string] $GpuLayers,
    "-fa", $FlashAttn,
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", "kvarn_k4v2_g128",
    "--kvarn-rtn-quantile", $rtnQuantileArg
)
if ($Batch -gt 0) {
    $commonArgs += @("-b", [string] $Batch)
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
    Write-Host "== Saving packed KVarN logits"
    $packedText = Invoke-Results $results $commonArgs $packedEnv
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

    Write-Host "== Checking scratch-reference KVarN logits"
    $check = Invoke-Results $results ($commonArgs + @("--check")) $scratchEnv
    if ($TraceAttn) {
        Write-Host $check
    }
    $nmse = Get-Nmse $check "packed-vs-scratch"
    Write-Host ("KVarN packed-vs-scratch logits: PASS, NMSE = {0:E3}" -f $nmse)
} finally {
    if (-not $KeepArtifacts) {
        Remove-Item -LiteralPath $OutputFile -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $PromptFile -ErrorAction SilentlyContinue
    }
}
