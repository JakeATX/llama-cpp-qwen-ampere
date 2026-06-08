param(
    [Parameter(Mandatory = $true)] [string] $QwenModel,
    [Parameter(Mandatory = $true)] [string] $GemmaModel,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $OutputDir = "",
    [double] $Tier1MinRatio = 0.90,
    [switch] $SkipBuild,
    [switch] $SkipTests,
    [switch] $RunGemmaExperimental,
    [switch] $RunTier2,
    [Alias("LogitsModel")]
    [string] $Tier2Model = "",
    [string[]] $QwenExtraArgs = @("-ncmoe", "34")
)

$ErrorActionPreference = "Stop"

if ($Tier1MinRatio -le 0.0 -or $Tier1MinRatio -gt 1.0) {
    throw "Tier1MinRatio must be in (0, 1]"
}
if (-not (Test-Path -LiteralPath $QwenModel)) {
    throw "QwenModel not found at $QwenModel"
}
if (-not (Test-Path -LiteralPath $GemmaModel)) {
    throw "GemmaModel not found at $GemmaModel"
}
if ($RunTier2.IsPresent -and [string]::IsNullOrWhiteSpace($Tier2Model)) {
    throw "RunTier2 requires -Tier2Model (or -LogitsModel)"
}
if ($RunTier2.IsPresent -and -not (Test-Path -LiteralPath $Tier2Model)) {
    throw "Tier2Model not found at $Tier2Model"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path (Get-Location) "artifacts/kvarn-production-gate/$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$benchScript = Join-Path (Get-Location) "scripts/kvarn/run_bench_matrix.ps1"
$logitsScript = Join-Path (Get-Location) "scripts/kvarn/compare_cuda_logits_ref.ps1"
if (-not (Test-Path -LiteralPath $benchScript)) {
    throw "Missing $benchScript"
}
if ($RunTier2.IsPresent -and -not (Test-Path -LiteralPath $logitsScript)) {
    throw "Missing $logitsScript"
}

function Invoke-Logged([string] $Name, [scriptblock] $Body) {
    $safe = ($Name -replace '[^A-Za-z0-9_.-]', '_')
    $log = Join-Path $OutputDir "$safe.log"
    Write-Host "== $Name"
    if (Test-Path -LiteralPath $log) {
        Remove-Item -LiteralPath $log -Force
    }
    try {
        & $Body 2>&1 | Tee-Object -FilePath $log | Write-Host
    } catch {
        Write-Error "$Name failed; see $log"
        throw
    }
}

$manifest = @(
    "qwen_model=$((Resolve-Path -LiteralPath $QwenModel).Path)",
    "gemma_model=$((Resolve-Path -LiteralPath $GemmaModel).Path)",
    "build_dir=$BuildDir",
    "tier1_min_ratio=$Tier1MinRatio",
    "run_gemma_experimental=$($RunGemmaExperimental.IsPresent)",
    "run_tier2=$($RunTier2.IsPresent)",
    "tier2_model=$Tier2Model",
    "qwen_extra_args=$($QwenExtraArgs -join ' ')"
)
[System.IO.File]::WriteAllText((Join-Path $OutputDir "manifest.txt"), ($manifest -join "`n") + "`n")

if (-not $SkipBuild.IsPresent) {
    Invoke-Logged "build llama-bench llama-results" {
        cmake --build $BuildDir --config Release --target llama-bench llama-results -j 8
    }
}

if (-not $SkipTests.IsPresent) {
    Invoke-Logged "ctest kvarn" {
        ctest --test-dir $BuildDir -C Release -R "test-kvarn-kv|test-kvarn-cuda|test-batch-split" --output-on-failure
    }
    Invoke-Logged "kv memory estimate self-test" {
        python scripts/kvarn/kv_memory_estimate.py --self-test
    }
}

Invoke-Logged "tier1 qwen kvarn" {
    & $benchScript `
        -Model $QwenModel `
        -BuildDir $BuildDir `
        -CaseList "pp512:512:0,tg64:0:64" `
        -FlashAttn off `
        -Repetitions 3 `
        -KvarnIters 4 `
        -MinKvarnRatio $Tier1MinRatio `
        -FailBelowMinKvarnRatio `
        -MinKvarnLayerLogs 10 `
        -ExpectedKvarnLayers "3-39:4" `
        -OutputDir (Join-Path $OutputDir "qwen-tier1") `
        -GpuLayers 99 `
        -ExtraArgs @($QwenExtraArgs)
}

$oldForceIswa = [Environment]::GetEnvironmentVariable("LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA", "Process")
try {
    [Environment]::SetEnvironmentVariable("LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA", $null, "Process")
    Invoke-Logged "tier1 gemma production fallback" {
        & $benchScript `
            -Model $GemmaModel `
            -BuildDir $BuildDir `
            -CaseList "pp512:512:0,tg64:0:64" `
            -FlashAttn off `
            -Repetitions 3 `
            -KvarnIters 4 `
            -MinKvarnRatio $Tier1MinRatio `
            -FailBelowMinKvarnRatio `
            -AllowKvarnFallback `
            -OutputDir (Join-Path $OutputDir "gemma-tier1-production-fallback") `
            -GpuLayers 99
    }
} finally {
    [Environment]::SetEnvironmentVariable("LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA", $oldForceIswa, "Process")
}

if ($RunGemmaExperimental.IsPresent) {
    $oldForceIswa = [Environment]::GetEnvironmentVariable("LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA", "Process")
    try {
        [Environment]::SetEnvironmentVariable("LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA", "1", "Process")
        Invoke-Logged "tier1 gemma experimental kvarn iswa" {
            & $benchScript `
                -Model $GemmaModel `
                -BuildDir $BuildDir `
                -CaseList "pp512:512:0,tg64:0:64" `
                -FlashAttn off `
                -Repetitions 3 `
                -KvarnIters 4 `
                -MinKvarnLayerLogs 8 `
                -ExpectedKvarnLayers "5-47:6" `
                -OutputDir (Join-Path $OutputDir "gemma-tier1-experimental-kvarn-iswa") `
                -GpuLayers 99
        }
    } finally {
        [Environment]::SetEnvironmentVariable("LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA", $oldForceIswa, "Process")
    }
}

if ($RunTier2.IsPresent) {
    Invoke-Logged "tier2 logits" {
        & $logitsScript `
            -Model $Tier2Model `
            -BuildDir $BuildDir `
            -Context 512 `
            -Batch 512 `
            -Repeat 16 `
            -KvarnIters 4 `
            -CheckPackedRepeat `
            -CheckPackedSplit `
            -ScratchMaxNmse 1e-5 `
            -SplitMaxNmse 1e-5 `
            -RepeatMaxNmse 1e-12 `
            -FlashAttn off
    }
}

Write-Host "KVarN production gate complete: $OutputDir"
