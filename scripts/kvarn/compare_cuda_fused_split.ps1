param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [int] $Context = 512,
    [int] $Predict = 270,
    [int] $GpuLayers = 99,
    [int] $Seed = 1234,
    [double] $RtnQuantile = 0.95,
    [string] $Prompt = "Hello"
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)

function Get-ExePath([string] $buildDir, [string] $name) {
    $path = Join-Path $buildDir "bin/Release/$name"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "$name not found at $path"
    }
    return (Resolve-Path -LiteralPath $path).Path
}

function Invoke-Completion([string] $exe, [string[]] $argv, [hashtable] $envSet) {
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
        throw "llama-completion failed with exit code $exit`n$text"
    }

    return $text
}

function Get-GeneratedText([string] $output) {
    $lines = $output -split "`r?`n"
    $generated = New-Object System.Collections.Generic.List[string]
    $inGeneration = $false

    foreach ($line in $lines) {
        if ($line -match "^\d+\.\d+\.\d+\s+I\s+generate:") {
            $inGeneration = $true
            continue
        }

        if (-not $inGeneration) {
            continue
        }

        if ($line -match "^\d+\.\d+\.\d+\s+[A-Z]\s+common_perf_print:") {
            break
        }

        if ($line -match "^\d+\.\d+\.\d+\s+[A-Z]\s+") {
            continue
        }
        if ($line -match "^llama_kv_cache_kvarn:") {
            continue
        }

        [void] $generated.Add($line)
    }

    return (($generated -join "`n").Trim())
}

function Get-GraphsReused([string] $output) {
    $m = [regex]::Match($output, "graphs reused\s+=\s+([0-9]+)")
    if (-not $m.Success) {
        throw "Could not find graph reuse counter in output"
    }
    return [int] $m.Groups[1].Value
}

function Get-EvalTokensPerSecond([string] $output) {
    $m = [regex]::Match($output, "eval time\s+=\s+[0-9.]+\s+ms\s+/\s+[0-9]+\s+runs\s+\([^)]+\s+([0-9.]+)\s+tokens per second\)")
    if (-not $m.Success) {
        return $null
    }
    return [double] $m.Groups[1].Value
}

$completion = Get-ExePath $BuildDir "llama-completion.exe"
$args = @(
    "-m", $Model,
    "-p", $Prompt,
    "-n", [string] $Predict,
    "-c", [string] $Context,
    "-ngl", [string] $GpuLayers,
    "--no-warmup",
    "--simple-io",
    "-no-cnv",
    "--no-display-prompt",
    "--ignore-eos",
    "-fa", "on",
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", "kvarn_k4v2_g128",
    "--kvarn-rtn-quantile", $rtnQuantileArg,
    "-s", [string] $Seed,
    "--temp", "0"
)

Write-Host "== KVarN batched fused packed attention"
$batched = Invoke-Completion $completion $args @{}
$batchedText = Get-GeneratedText $batched
$batchedGraphs = Get-GraphsReused $batched
$batchedTps = Get-EvalTokensPerSecond $batched

Write-Host "== KVarN serial fused packed attention"
$serial = Invoke-Completion $completion $args @{ "LLAMA_KVARN_ATTN_SERIAL_FUSED" = "1" }
$serialText = Get-GeneratedText $serial
$serialGraphs = Get-GraphsReused $serial
$serialTps = Get-EvalTokensPerSecond $serial

Write-Host "== KVarN split packed attention"
$split = Invoke-Completion $completion $args @{ "LLAMA_KVARN_ATTN_SPLIT_KERNELS" = "1" }
$splitText = Get-GeneratedText $split
$splitGraphs = Get-GraphsReused $split
$splitTps = Get-EvalTokensPerSecond $split

if ($batchedText -ne $serialText -or $batchedText -ne $splitText) {
    Write-Host "== batched fused generated text =="
    Write-Host $batchedText
    Write-Host "== serial fused generated text =="
    Write-Host $serialText
    Write-Host "== split generated text =="
    Write-Host $splitText
    throw "KVarN packed mixed attention dispatch mode changed deterministic generated text"
}

Write-Host "KVarN packed attention dispatch parity: PASS"
Write-Host ("KVarN batched fused attention: graphs reused = {0}, eval tok/s = {1}" -f $batchedGraphs, $batchedTps)
Write-Host ("KVarN serial fused attention : graphs reused = {0}, eval tok/s = {1}" -f $serialGraphs, $serialTps)
Write-Host ("KVarN split attention        : graphs reused = {0}, eval tok/s = {1}" -f $splitGraphs, $splitTps)
