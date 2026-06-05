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

function Join-ProcessArgs([string[]] $argv) {
    return ($argv | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"','\"') + '"'
        } else {
            $_
        }
    }) -join ' '
}

function Invoke-ExpectProcessFailure([string] $exe, [string[]] $argv, [hashtable] $envSet, [string] $pattern, [string] $label) {
    $oldEnv = @{}
    foreach ($key in $envSet.Keys) {
        $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $envSet[$key], "Process")
    }

    $stdoutLog = Join-Path $env:TEMP ("kvarn-unsupported-{0}.out.log" -f ([guid]::NewGuid().ToString("N")))
    $stderrLog = Join-Path $env:TEMP ("kvarn-unsupported-{0}.err.log" -f ([guid]::NewGuid().ToString("N")))

    try {
        $process = Start-Process `
            -FilePath $exe `
            -ArgumentList (Join-ProcessArgs $argv) `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog `
            -PassThru `
            -WindowStyle Hidden

        if (-not $process.WaitForExit(30000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            throw "$label did not exit within timeout"
        }

        $out = Get-Content -Raw -LiteralPath $stdoutLog -ErrorAction SilentlyContinue
        $err = Get-Content -Raw -LiteralPath $stderrLog -ErrorAction SilentlyContinue
        $text = "$out`n$err"
        if ($process.ExitCode -eq 0) {
            throw "$label unexpectedly succeeded"
        }
        if ($text -notmatch $pattern) {
            Write-Host $text
            throw "$label did not report expected failure"
        }
        Write-Host "${label}: PASS"
    } finally {
        foreach ($key in $envSet.Keys) {
            [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
        }
        Remove-Item -LiteralPath $stdoutLog, $stderrLog -ErrorAction SilentlyContinue
    }
}

$results = Get-ExePath $BuildDir "llama-results.exe"
$server = Get-ExePath $BuildDir "llama-server.exe"
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
    @{ "LLAMA_KVARN_ATTN_REF_SCRATCH" = "bogus" } `
    "invalid KVarN environment flag LLAMA_KVARN_ATTN_REF_SCRATCH=bogus" `
    "KVarN invalid scratch-reference env rejection"

Invoke-ExpectFailure `
    $results `
    $commonKvarn `
    @{ "LLAMA_KVARN_DEBUG_UBATCH" = "129" } `
    "KVarN debug ubatch override exceeds tail-ring safety limit" `
    "KVarN unsafe debug ubatch rejection"

Invoke-ExpectFailure `
    $results `
    $commonKvarn `
    @{ "LLAMA_KVARN_DEBUG_UBATCH" = "0" } `
    "KVarN debug ubatch override must be a positive integer" `
    "KVarN invalid debug ubatch rejection"

Invoke-ExpectProcessFailure `
    $server `
    @(
        "-m", $SupportedModel,
        "--host", "127.0.0.1",
        "--port", "8139",
        "--parallel", "2",
        "-c", [string] $Context,
        "-ngl", [string] $GpuLayers,
        "--kv-cache-quant", "kvarn",
        "--kvarn-preset", "kvarn_k4v2_g128"
    ) `
    @{} `
    "KVarN currently supports only --parallel 1" `
    "KVarN server multi-slot rejection"

if ($UnsupportedDimModel -ne "") {
    $unsupportedArgs = @(
        "-m", $UnsupportedDimModel,
        "-p", "Hello",
        "-o", $tmpOut,
        "-c", [string] $Context,
        "-ngl", [string] $GpuLayers,
        "-fa", "off",
        "--kv-cache-quant", "kvarn",
        "--kvarn-preset", "kvarn_k4v2_g128"
    )

    Invoke-ExpectFailure `
        $results `
        $unsupportedArgs `
        @{} `
        "KVarN backend currently supports only 128-, 256-, or 512-dimensional K/V heads|KVarN backend supports SWA/ISWA only for Gemma 4 models at this stage" `
        "KVarN unsupported K/V dimension or non-Gemma SWA/ISWA rejection"
}

Remove-Item -LiteralPath $tmpOut -ErrorAction SilentlyContinue
