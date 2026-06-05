param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda"),
    [string] $CtxList = "256 512 1024",
    [int] $GpuLayers = 99,
    [double] $RtnQuantile = 1.0
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)

$cli = Join-Path $BuildDir "bin/Release/llama-cli.exe"
if (-not (Test-Path -LiteralPath $cli)) {
    throw "llama-cli.exe not found at $cli"
}

foreach ($ctx in $CtxList.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)) {
    Write-Host "== FP16 KV smoke: ctx=$ctx"
    & $cli -m $Model -p "Hello" -n 8 -c $ctx -ngl $GpuLayers --no-warmup --simple-io --single-turn
    if ($LASTEXITCODE -ne 0) {
        throw "FP16 KV smoke failed for ctx=$ctx"
    }

    Write-Host "== KVarN decode smoke or explicit guard: ctx=$ctx"
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $kvarnOutput = & $cli -m $Model -p "Hello" -n 1 -c $ctx -ngl $GpuLayers --no-warmup --simple-io --single-turn `
        --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile $rtnQuantileArg 2>&1
    $kvarnExit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    $kvarnOutput | Write-Host
    if ($kvarnExit -eq 0) {
        continue
    }
    if (($kvarnOutput -join "`n") -notmatch "KVarN backend currently supports only 128- or 256-dimensional K/V heads|KVarN backend does not support MLA models yet|KVarN backend does not support hybrid SWA models yet|KVarN backend currently requires every KV layer to run on a backend with CUDA KVarN op support|KVarN graph backend does not yet support attention rotations|KVarN graph backend active body record count exceeds allocated cache capacity|KVarN graph backend is not wired yet|KVarN shift/update graph path is not wired yet") {
        throw "KVarN failure did not include the expected validation or graph-backend guard"
    }
}
