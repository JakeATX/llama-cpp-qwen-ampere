param(
    [Parameter(Mandatory = $true)] [string] $SupportedModel,
    [string] $UnsupportedDimModel = "",
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [int] $Context = 256,
    [int] $GpuLayers = 99
)

$ErrorActionPreference = "Stop"

function Get-ExePath([string] $buildDir, [string] $name) {
    $path = Join-Path $buildDir "bin/Release/$name"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "$name not found at $path"
    }
    return (Resolve-Path -LiteralPath $path).Path
}

function Invoke-ExpectFailure([string] $exe, [string[]] $argv, [hashtable] $envSet, [string] $pattern, [string] $label) {
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
    if ($exit -eq 0) {
        throw "$label unexpectedly succeeded"
    }
    if ($text -notmatch $pattern) {
        Write-Host $text
        throw "$label did not report expected failure"
    }
    Write-Host "${label}: PASS"
}

$results = Get-ExePath $BuildDir "llama-results.exe"
$cli = Get-ExePath $BuildDir "llama-cli.exe"
$tmpOut = Join-Path $env:TEMP "kvarn-unsupported-smoke.gguf"
Remove-Item -LiteralPath $tmpOut -ErrorAction SilentlyContinue

$commonKvarn = @(
    "-m", $SupportedModel,
    "-p", "hello",
    "-o", $tmpOut,
    "-c", [string] $Context,
    "-ngl", [string] $GpuLayers,
    "-fa", "off",
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", "kvarn_k4v2_g128"
)

Invoke-ExpectFailure `
    $results `
    $commonKvarn `
    @{ "LLAMA_KVARN_ATTN_FUSED_BATCH" = "1" } `
    "KVarN forced fused-batch attention is disabled because multi-query correctness is not proven" `
    "KVarN forced fused-batch rejection"

Invoke-ExpectFailure `
    $results `
    $commonKvarn `
    @{ "LLAMA_KVARN_DEBUG_UBATCH" = "129" } `
    "KVarN debug ubatch override exceeds tail-ring safety limit" `
    "KVarN unsafe debug ubatch rejection"

if ($UnsupportedDimModel -ne "") {
    $unsupportedArgs = @(
        "-m", $UnsupportedDimModel,
        "-p", "Hello",
        "-n", "1",
        "-c", [string] $Context,
        "-ngl", [string] $GpuLayers,
        "--no-warmup",
        "--simple-io",
        "--single-turn",
        "-fa", "off",
        "--kv-cache-quant", "kvarn",
        "--kvarn-preset", "kvarn_k4v2_g128"
    )

    Invoke-ExpectFailure `
        $cli `
        $unsupportedArgs `
        @{} `
        "KVarN backend currently supports only 128- or 256-dimensional K/V heads" `
        "KVarN unsupported K/V dimension rejection"
}

Remove-Item -LiteralPath $tmpOut -ErrorAction SilentlyContinue
