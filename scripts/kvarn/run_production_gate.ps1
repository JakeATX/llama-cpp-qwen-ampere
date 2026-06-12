param(
    [Parameter(Mandatory = $true)] [string] $QwenModel,
    [Parameter(Mandatory = $true)] [string] $GemmaModel,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $MainlineBuildDir = (Join-Path (Get-Location) "..\llama.cpp-mainline\build-cuda-static-vs"),
    [string] $OutputDir = "",
    [double] $Tier1MinRatio = 0.90,
    [int] $QwenRepetitions = 5,
    [int] $GemmaRepetitions = 3,
    [switch] $SkipBuild,
    [switch] $SkipTests,
    [switch] $RunGemmaExperimental,
    [switch] $RunTier2,
    [Alias("LogitsModel")]
    [string] $Tier2Model = "",
    [string[]] $Tier2ExtraArgs = @(),
    [int] $Tier2MinKvarnLayerLogs = -1,
    [string] $Tier2ExpectedKvarnLayers = "",
    [string[]] $QwenExtraArgs = @("-ncmoe", "34"),
    [int] $Tier2MinKvarnBodyRecords = 1,
    [switch] $AllowDiagnosticEnv
)

$ErrorActionPreference = "Stop"

if ($Tier1MinRatio -le 0.0 -or $Tier1MinRatio -gt 1.0) {
    throw "Tier1MinRatio must be in (0, 1]"
}
if ($Tier1MinRatio -lt 0.90) {
    throw "Tier1MinRatio is the production gate threshold and must be >= 0.90. Use run_bench_matrix.ps1 directly for low-threshold diagnostics."
}
if ($QwenRepetitions -le 0) {
    throw "QwenRepetitions must be positive"
}
if ($GemmaRepetitions -le 0) {
    throw "GemmaRepetitions must be positive"
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
if ($Tier2MinKvarnBodyRecords -lt 0) {
    throw "Tier2MinKvarnBodyRecords must be non-negative"
}
if ($Tier2MinKvarnLayerLogs -lt -1) {
    throw "Tier2MinKvarnLayerLogs must be -1 for auto or non-negative"
}

$unsafeDiagnosticEnv = @(
    "LLAMA_KVARN_ATTN_ENABLE_256D_WARPQK",
    "LLAMA_KVARN_ATTN_ENABLE_256D_BODY_MIRROR",
    "LLAMA_KVARN_ATTN_WARPQK_FORCE_QT",
    "LLAMA_KVARN_ATTN_SPLIT_KERNELS",
    "LLAMA_KVARN_ATTN_SERIAL_FUSED",
    "LLAMA_KVARN_ATTN_REF_SCRATCH",
    "LLAMA_KVARN_ATTN_FUSED_BATCH",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_DIR",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_CALL",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LAYER",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_IQ",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_IH",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LIMIT",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_FIRST_256D",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_MIN_TOKENS",
    "LLAMA_KVARN_ATTN_DECODE_PER_HEAD",
    "LLAMA_KVARN_ATTN_TRACE",
    "LLAMA_KVARN_ATTN_TRACE_LIMIT",
    "LLAMA_KVARN_STORE_TRACE",
    "LLAMA_KVARN_STORE_TRACE_LIMIT",
    "LLAMA_KVARN_DEQUANT_CACHE_TRACE",
    "LLAMA_KVARN_DEQUANT_CACHE_TRACE_LIMIT",
    "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK",
    "LLAMA_KVARN_UNSAFE_ALLOW_FUSED_BATCH",
    "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA",
    "LLAMA_KVARN_DEBUG_UBATCH"
)
if (-not $AllowDiagnosticEnv.IsPresent) {
    $leaked = @()
    foreach ($name in $unsafeDiagnosticEnv) {
        $value = [Environment]::GetEnvironmentVariable($name, "Process")
        if (-not [string]::IsNullOrEmpty($value)) {
            $leaked += ("{0}={1}" -f $name, $value)
        }
    }
    if ($leaked.Count -gt 0) {
        throw ("Production gate refuses to run with diagnostic env vars set. Clear them or pass -AllowDiagnosticEnv for an explicit diagnostic run:`n" + ($leaked -join "`n"))
    }
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path (Get-Location) "artifacts/kvarn-production-gate/$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$benchScript = Join-Path (Get-Location) "scripts/kvarn/run_bench_matrix.ps1"
$mainlineParityScript = Join-Path (Get-Location) "scripts/kvarn/run_mainline_parity_matrix.ps1"
$logitsScript = Join-Path (Get-Location) "scripts/kvarn/compare_cuda_logits_ref.ps1"
if (-not (Test-Path -LiteralPath $benchScript)) {
    throw "Missing $benchScript"
}
if (-not (Test-Path -LiteralPath $mainlineParityScript)) {
    throw "Missing $mainlineParityScript"
}
if ($RunTier2.IsPresent -and -not (Test-Path -LiteralPath $logitsScript)) {
    throw "Missing $logitsScript"
}

function Invoke-WithProcessEnvironment([hashtable] $Env, [scriptblock] $Body) {
    $oldEnv = @{}
    foreach ($key in $Env.Keys) {
        $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $Env[$key], "Process")
    }

    try {
        & $Body
    } finally {
        foreach ($key in $Env.Keys) {
            [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
        }
    }
}

function Invoke-Logged([string] $Name, [scriptblock] $Body) {
    $safe = ($Name -replace '[^A-Za-z0-9_.-]', '_')
    $log = Join-Path $OutputDir "$safe.log"
    Write-Host "== $Name"
    if (Test-Path -LiteralPath $log) {
        Remove-Item -LiteralPath $log -Force
    }
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $global:LASTEXITCODE = 0
        & $Body 2>&1 | Tee-Object -FilePath $log | Write-Host
        if ($global:LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $global:LASTEXITCODE"
        }
    } catch {
        Write-Error "$Name failed; see $log"
        throw
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
}

$qwenEffectiveExtraArgs = @($QwenExtraArgs)
$qwenNcmoeIdx = [Array]::IndexOf($qwenEffectiveExtraArgs, "-ncmoe")
if ($qwenNcmoeIdx -lt 0) {
    $qwenEffectiveExtraArgs += @("-ncmoe", "34")
} elseif ($qwenNcmoeIdx -ge ($qwenEffectiveExtraArgs.Count - 1) -or $qwenEffectiveExtraArgs[$qwenNcmoeIdx + 1] -ne "34") {
    throw "Qwen production gate requires -ncmoe 34 so the expected KVarN layer set remains 3-39:4"
}

$tier2EffectiveExtraArgs = @($Tier2ExtraArgs)
$tier2EffectiveExpectedLayers = $Tier2ExpectedKvarnLayers
$tier2EffectiveMinLayerLogs = $Tier2MinKvarnLayerLogs
$tier2ResolvedModel = if ($RunTier2.IsPresent) { (Resolve-Path -LiteralPath $Tier2Model).Path } else { "" }
$qwenResolvedModel = (Resolve-Path -LiteralPath $QwenModel).Path
if ($RunTier2.IsPresent -and $tier2ResolvedModel -eq $qwenResolvedModel) {
    if ($tier2EffectiveExtraArgs.Count -eq 0) {
        $tier2EffectiveExtraArgs = @($qwenEffectiveExtraArgs)
    }
    if ([string]::IsNullOrWhiteSpace($tier2EffectiveExpectedLayers)) {
        $tier2EffectiveExpectedLayers = "3-39:4"
    }
    if ($tier2EffectiveMinLayerLogs -lt 0) {
        $tier2EffectiveMinLayerLogs = 10
    }
} elseif ($tier2EffectiveMinLayerLogs -lt 0) {
    $tier2EffectiveMinLayerLogs = 1
}

$manifest = @(
    "qwen_model=$((Resolve-Path -LiteralPath $QwenModel).Path)",
    "gemma_model=$((Resolve-Path -LiteralPath $GemmaModel).Path)",
    "build_dir=$BuildDir",
    "mainline_build_dir=$MainlineBuildDir",
    "tier1_min_ratio=$Tier1MinRatio",
    "qwen_repetitions=$QwenRepetitions",
    "gemma_repetitions=$GemmaRepetitions",
    "run_gemma_experimental=$($RunGemmaExperimental.IsPresent)",
    "run_tier2=$($RunTier2.IsPresent)",
    "tier2_model=$Tier2Model",
    "tier2_extra_args=$($tier2EffectiveExtraArgs -join ' ')",
    "tier2_expected_kvarn_layers=$tier2EffectiveExpectedLayers",
    "tier2_min_kvarn_layer_logs=$tier2EffectiveMinLayerLogs",
    "tier2_min_kvarn_body_records=$Tier2MinKvarnBodyRecords",
    "allow_diagnostic_env=$($AllowDiagnosticEnv.IsPresent)",
    "qwen_extra_args=$($qwenEffectiveExtraArgs -join ' ')",
    "qwen_expected_kvarn_layers=3-39:4",
    "gemma_production_mode=normal-iswa-fallback",
    "gemma_experimental_mode=true-kvarn-iswa-diagnostic"
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

Invoke-Logged "tier1 qwen mainline parity" {
    & $mainlineParityScript `
        -Model $QwenModel `
        -MainlineBuildDir $MainlineBuildDir `
        -KvarnBuildDir $BuildDir `
        -CaseList "pp512:512:0,tg64:0:64" `
        -FlashAttn off `
        -Repetitions $QwenRepetitions `
        -KvarnIters 4 `
        -MinParityRatio $Tier1MinRatio `
        -FailBelowMinParityRatio `
        -MinKvarnLayerLogs 10 `
        -ExpectedKvarnLayers "3-39:4" `
        -OutputDir (Join-Path $OutputDir "qwen-tier1-mainline-parity") `
        -GpuLayers 99 `
        -ExtraArgs @($qwenEffectiveExtraArgs)
}

Invoke-Logged "tier1 gemma production iswa fallback" {
    Invoke-WithProcessEnvironment @{
        "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA" = $null
        "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK" = "1"
    } {
        & $mainlineParityScript `
            -Model $GemmaModel `
            -MainlineBuildDir $MainlineBuildDir `
            -KvarnBuildDir $BuildDir `
            -CaseList "pp512:512:0,tg64:0:64" `
            -FlashAttn off `
            -Repetitions $GemmaRepetitions `
            -KvarnIters 4 `
            -MinParityRatio $Tier1MinRatio `
            -FailBelowMinParityRatio `
            -AllowKvarnFallback `
            -OutputDir (Join-Path $OutputDir "gemma-tier1-production-iswa-fallback") `
            -GpuLayers 99
    }
}

if ($RunGemmaExperimental.IsPresent) {
    Invoke-Logged "tier1 gemma experimental true kvarn iswa diagnostic" {
        Invoke-WithProcessEnvironment @{
            "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA" = "1"
            "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK" = $null
        } {
            & $benchScript `
                -Model $GemmaModel `
                -BuildDir $BuildDir `
                -CaseList "pp512:512:0,tg64:0:64" `
                -FlashAttn off `
                -Repetitions $GemmaRepetitions `
                -KvarnIters 4 `
                -MinKvarnRatio $Tier1MinRatio `
                -FailBelowMinKvarnRatio `
                -MinKvarnLayerLogs 8 `
                -ExpectedKvarnLayers "5-47:6" `
                -OutputDir (Join-Path $OutputDir "gemma-tier1-experimental-true-kvarn-iswa-diagnostic") `
                -GpuLayers 99
        }
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
            -FlashAttn off `
            -MinKvarnLayerLogs $tier2EffectiveMinLayerLogs `
            -MinKvarnBodyRecords $Tier2MinKvarnBodyRecords `
            -ExpectedKvarnLayers $tier2EffectiveExpectedLayers `
            -ExtraArgs @($tier2EffectiveExtraArgs)
    }
}

Write-Host "KVarN production gate complete: $OutputDir"
