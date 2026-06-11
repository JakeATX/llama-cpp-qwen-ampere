param(
    [Parameter(Mandatory = $true)] [string] $SupportedModel,
    [string] $SupportedIswaModel = "",
    [string] $Supported256ActiveModel = "",
    [string] $Supported512Model = "",
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
$tmpOut = Join-Path $env:TEMP ("kvarn-unsupported-smoke-{0}.gguf" -f ([guid]::NewGuid().ToString("N")))
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

try {
    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_ATTN_FUSED_BATCH" = "bogus" } `
        "invalid KVarN environment flag LLAMA_KVARN_ATTN_FUSED_BATCH=bogus" `
        "KVarN invalid fused-batch env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_UNSAFE_ALLOW_FUSED_BATCH" = "bogus" } `
        "invalid KVarN environment flag LLAMA_KVARN_UNSAFE_ALLOW_FUSED_BATCH=bogus" `
        "KVarN invalid unsafe fused-batch env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_ATTN_REF_SCRATCH" = "bogus" } `
        "invalid KVarN( CUDA)? environment flag LLAMA_KVARN_ATTN_REF_SCRATCH=bogus" `
        "KVarN invalid scratch-reference env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_ATTN_REF_SCRATCH" = "2" } `
        "invalid KVarN( CUDA)? environment flag LLAMA_KVARN_ATTN_REF_SCRATCH=2" `
        "KVarN out-of-range scratch-reference env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_ATTN_TRACE" = "2" } `
        "invalid KVarN environment flag LLAMA_KVARN_ATTN_TRACE=2" `
        "KVarN out-of-range trace env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{
            "LLAMA_KVARN_ATTN_TRACE" = "1"
            "LLAMA_KVARN_ATTN_TRACE_LIMIT" = "bogus"
        } `
        "invalid KVarN environment integer LLAMA_KVARN_ATTN_TRACE_LIMIT=bogus" `
        "KVarN invalid trace-limit env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_STORE_TRACE" = "bogus" } `
        "invalid KVarN CUDA environment flag LLAMA_KVARN_STORE_TRACE=bogus" `
        "KVarN invalid store-trace env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{
            "LLAMA_KVARN_STORE_TRACE" = "1"
            "LLAMA_KVARN_STORE_TRACE_LIMIT" = "bogus"
        } `
        "invalid KVarN CUDA environment integer LLAMA_KVARN_STORE_TRACE_LIMIT=bogus" `
        "KVarN invalid store-trace-limit env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_ATTN_SERIAL_FUSED" = "2" } `
        "invalid KVarN CUDA environment flag LLAMA_KVARN_ATTN_SERIAL_FUSED=2" `
        "KVarN out-of-range CUDA serial-fused env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_ATTN_SPLIT_KERNELS" = "2" } `
        "invalid KVarN CUDA environment flag LLAMA_KVARN_ATTN_SPLIT_KERNELS=2" `
        "KVarN out-of-range CUDA split-kernel env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_DEQUANT_CACHE_TRACE" = "bogus" } `
        "invalid KVarN CUDA environment flag LLAMA_KVARN_DEQUANT_CACHE_TRACE=bogus" `
        "KVarN invalid dequant-cache trace env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{
            "LLAMA_KVARN_DEQUANT_CACHE_TRACE" = "1"
            "LLAMA_KVARN_DEQUANT_CACHE_TRACE_LIMIT" = "bogus"
        } `
        "invalid KVarN CUDA environment integer LLAMA_KVARN_DEQUANT_CACHE_TRACE_LIMIT=bogus" `
        "KVarN invalid dequant-cache trace-limit env rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_DEBUG_UBATCH" = "4294967296" } `
        "KVarN debug ubatch override must be a positive integer" `
        "KVarN out-of-range debug ubatch rejection"

    Invoke-ExpectFailure `
        $results `
        $commonKvarn `
        @{ "LLAMA_KVARN_DEBUG_UBATCH" = "0" } `
        "KVarN debug ubatch override must be a positive integer" `
        "KVarN invalid debug ubatch rejection"

    Invoke-ExpectFailure `
        $results `
        ($commonKvarn + @("--no-kv-offload")) `
        @{} `
        "KVarN currently requires KV cache offload because KVarN backend ops are CUDA-only" `
        "KVarN no-kv-offload rejection"

    Invoke-ExpectFailure `
        $results `
        @(
            "-fit", "off",
            "-m", $SupportedModel,
            "-p", "hello",
            "-o", $tmpOut,
            "-c", [string] $Context,
            "-ngl", "0",
            "-fa", "off",
            "--kv-cache-quant", "kvarn",
            "--kvarn-preset", "kvarn_k4v2_g128"
        ) `
        @{} `
        "KVarN backend currently requires every KV layer to run on a backend with CUDA KVarN op support; layer [0-9]+ has head_dim=[0-9]+ and is assigned to CPU" `
        "KVarN CPU layer placement rejection"

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
            "KVarN backend currently supports only 128-, 256-, or 512-dimensional K/V heads|KVarN backend requires equal K and V head dimensions|KVarN backend supports SWA/ISWA only for Gemma 4 models at this stage" `
            "KVarN unsupported K/V dimension or non-Gemma SWA/ISWA rejection"
    }

    if ($SupportedIswaModel -ne "") {
        $iswaArgs = @(
            "-m", $SupportedIswaModel,
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
            $iswaArgs `
            @{ "LLAMA_KVARN_DEBUG_UBATCH" = "bogus" } `
            "KVarN debug ubatch override must be a positive integer" `
            "KVarN+ISWA invalid debug ubatch rejection"
    }
} finally {
    Remove-Item -LiteralPath $tmpOut -ErrorAction SilentlyContinue
}
