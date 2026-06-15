param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $Dataset,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $OutputDir = "",
    [string] $KvarnPreset = "kvarn_k8v8_g128",
    [int] $KvarnIters = 4,
    [int] $ContextSize = 4096,
    [int] $BatchSize = 4096,
    [int] $Chunks = 2,
    [int] $DumpLayer = 3,
    [int] $DumpIq = 2047,
    [int] $DumpIh = 0,
    [int] $DumpHead = 0,
    [int] $BodyRecordLimit = 30,
    [int] $BodySrcLayout = -1,
    [string[]] $ExtraArgs = @("-ncmoe", "34")
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path (Get-Location) "artifacts/kvarn-rootcause/qwen36-$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$boundaryDir = Join-Path $OutputDir "boundary"
$bodyDir = Join-Path $OutputDir "body-records"
[void] [System.IO.Directory]::CreateDirectory($boundaryDir)
[void] [System.IO.Directory]::CreateDirectory($bodyDir)

$oldEnv = @{}
$envSet = @{
    "LLAMA_KVARN_ENABLE_PAPER_FRAME" = "1"
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP" = "1"
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_DIR" = $boundaryDir
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LIMIT" = "1"
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LAYER" = [string] $DumpLayer
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_IQ" = [string] $DumpIq
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_IH" = [string] $DumpIh
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_MIN_TOKENS" = [string] $ContextSize
    "LLAMA_KVARN_DEBUG_BODY_RECORD_DUMP" = "1"
    "LLAMA_KVARN_DEBUG_BODY_RECORD_DIR" = $bodyDir
    "LLAMA_KVARN_DEBUG_BODY_RECORD_LIMIT" = [string] $BodyRecordLimit
    "LLAMA_KVARN_DEBUG_BODY_RECORD_LAYER" = [string] $DumpLayer
    "LLAMA_KVARN_DEBUG_BODY_HEAD" = [string] $DumpHead
}

if ($BodySrcLayout -ge 0) {
    $envSet["LLAMA_KVARN_DEBUG_BODY_SRC_LAYOUT"] = [string] $BodySrcLayout
}

foreach ($key in $envSet.Keys) {
    $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
    [Environment]::SetEnvironmentVariable($key, $envSet[$key], "Process")
}

try {
    scripts\kvarn\run_accuracy_gate.ps1 `
        -Model $Model `
        -Dataset $Dataset `
        -BuildDir $BuildDir `
        -OutputDir (Join-Path $OutputDir "accuracy") `
        -KvarnPreset $KvarnPreset `
        -KvarnIters $KvarnIters `
        -KvarnRtnQuantile 1.0 `
        -FlashAttn off `
        -Fit off `
        -GpuLayers 999 `
        -ContextSize $ContextSize `
        -BatchSize $BatchSize `
        -Chunks $Chunks `
        -MaxPplIncrease 1000 `
        -ExpectedKvarnLayers "3-39:4" `
        -ExtraArgs @($ExtraArgs)
} finally {
    foreach ($key in $envSet.Keys) {
        [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
    }
}

python scripts\kvarn\replay_mixed_attn_boundary.py --dump $boundaryDir --write-replay
if ($LASTEXITCODE -ne 0) {
    throw "KVarN mixed-attn self replay failed with exit code $LASTEXITCODE"
}
python scripts\kvarn\replay_f16_truth_boundary.py --boundary $boundaryDir --body-records $bodyDir --write-truth
if ($LASTEXITCODE -ne 0) {
    throw "KVarN f16-truth replay failed with exit code $LASTEXITCODE"
}

Write-Host "KVarN ctx4096 truth capture complete: $OutputDir"
