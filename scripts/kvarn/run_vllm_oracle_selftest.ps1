param(
    [string] $Python = "python",
    [string] $HeadDims = "128,256,512",
    [string] $Presets = "k4v2,k4v4,k8v2,k8v4,k8v8",
    [int] $Iters = 8,
    [int] $Queries = 7,
    [int] $Seed = 1234
)

$ErrorActionPreference = "Stop"
$script = Join-Path (Get-Location) "scripts/kvarn/kvarn_vllm_oracle.py"
if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing oracle script at $script"
}

& $Python $script --self-test --head-dims $HeadDims --presets $Presets --iters $Iters --queries $Queries --seed $Seed
if ($LASTEXITCODE -ne 0) {
    throw "KVarN vLLM oracle self-test failed with exit code $LASTEXITCODE"
}
