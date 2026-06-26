param(
    [Parameter(Mandatory = $true)] [string] $QwenModel,
    [Parameter(Mandatory = $true)] [string] $GemmaModel,
    [string] $QwenDataset = "",
    [string] $GemmaDataset = "",
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $MainlineBuildDir = (Join-Path (Get-Location) "..\llama.cpp-mainline\build-cuda-static-vs"),
    [string] $OutputDir = "",
    [double] $Tier1MinRatio = 0.95,
    [int] $QwenRepetitions = 5,
    [int] $GemmaRepetitions = 3,
    [string] $AccuracyPreset = "kvarn_k4v4_g128",
    [int] $AccuracyContextSize = 16384,
    [int] $AccuracyBatchSize = 512,
    [int] $AccuracyChunks = 1,
    [string] $SpeedPreset = "kvarn_k8v4_g128",
    [string] $SpeedEvidenceCaseList = "pp512:512:0",
    [double] $AccuracyMaxMeanKL = 0.02,
    [double] $AccuracyMaxKLD99 = 1.0,
    [double] $AccuracyMaxKLD999 = 1.0,
    [double] $AccuracyMaxKLDMax = 1.0,
    [double] $AccuracyMaxKLPplIncrease = -1.0,
    [double] $AccuracyMaxKLPDiffRms = -1.0,
    [double] $AccuracyMinKLSameTopP = -1.0,
    [switch] $SkipBuild,
    [switch] $SkipTests,
    [switch] $SkipAccuracy,
    [switch] $RunGemmaExperimental,
    [switch] $RunTier2,
    [Alias("LogitsModel")]
    [string] $Tier2Model = "",
    [string[]] $Tier2ExtraArgs = @(),
    [int] $Tier2MinKvarnLayerLogs = -1,
    [string] $Tier2ExpectedKvarnLayers = "",
    [string[]] $QwenExtraArgs = @("-ncmoe", "34"),
    [int] $Tier2MinKvarnBodyRecords = 1,
    [int] $Tier2MinActiveKvarnBodyRecords = 0,
    [string] $QwenLayerKeyBits = "",
    [string] $QwenLayerValueBits = "",
    [string] $GemmaLayerKeyBits = "",
    [string] $GemmaLayerValueBits = "",
    [string] $QwenExpectedEffectiveKvarnBits = "auto",
    [string] $GemmaExpectedEffectiveKvarnBits = "auto",
    [switch] $TraceFwhtEvidence,
    [int] $MinFwhtTaken = 0,
    [int] $MinBatchedStorePhaseUses = 1,
    [switch] $RunGemmaFallbackDiagnostic,
    [switch] $AllowDiagnosticEnv
)

$ErrorActionPreference = "Stop"

if ($Tier1MinRatio -le 0.0 -or $Tier1MinRatio -gt 1.0) {
    throw "Tier1MinRatio must be in (0, 1]"
}
if ($Tier1MinRatio -lt 0.95) {
    throw "Tier1MinRatio is the production gate threshold and must be >= 0.95. Use run_bench_matrix.ps1 directly for low-threshold diagnostics."
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
if (-not $SkipAccuracy.IsPresent) {
    if ([string]::IsNullOrWhiteSpace($QwenDataset) -or -not (Test-Path -LiteralPath $QwenDataset)) {
        throw "Production gate requires -QwenDataset for KL accuracy, or pass -SkipAccuracy for an explicit speed-only diagnostic run"
    }
    if ([string]::IsNullOrWhiteSpace($GemmaDataset) -or -not (Test-Path -LiteralPath $GemmaDataset)) {
        throw "Production gate requires -GemmaDataset for KL accuracy, or pass -SkipAccuracy for an explicit speed-only diagnostic run"
    }
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
if ($Tier2MinActiveKvarnBodyRecords -lt 0) {
    throw "Tier2MinActiveKvarnBodyRecords must be non-negative"
}
if ($Tier2MinKvarnLayerLogs -lt -1) {
    throw "Tier2MinKvarnLayerLogs must be -1 for auto or non-negative"
}
if ($MinFwhtTaken -lt 0) {
    throw "MinFwhtTaken must be non-negative"
}
if ($MinFwhtTaken -gt 0 -and -not $TraceFwhtEvidence.IsPresent) {
    throw "MinFwhtTaken requires -TraceFwhtEvidence so the production evidence run can prove CUDA FWHT use"
}
if ($MinBatchedStorePhaseUses -lt 0) {
    throw "MinBatchedStorePhaseUses must be non-negative"
}
if ($AccuracyMaxKLPplIncrease -lt -1.0) {
    throw "AccuracyMaxKLPplIncrease must be -1 to disable or non-negative"
}
if ($AccuracyMaxKLPDiffRms -lt -1.0) {
    throw "AccuracyMaxKLPDiffRms must be -1 to disable or non-negative"
}
if ($AccuracyMinKLSameTopP -lt -1.0 -or $AccuracyMinKLSameTopP -gt 1.0) {
    throw "AccuracyMinKLSameTopP must be -1 to disable or in [0, 1]"
}
function Test-KvarnPresetRequestsV2([string] $Preset) {
    $m = [regex]::Match($Preset, "(?i)^kvarn_k[0-9]+v([0-9]+)_g128$")
    return $m.Success -and ([int] $m.Groups[1].Value) -eq 2
}
if ((Test-KvarnPresetRequestsV2 $AccuracyPreset) -or (Test-KvarnPresetRequestsV2 $SpeedPreset)) {
    throw "V2 KVarN presets are measurement-only in this tree. Use run_accuracy_gate.ps1, run_safe_full_gate.ps1 -RunExperimentalAccuracy, or run_bench_matrix.ps1 for V2 diagnostics."
}
if (-not [string]::IsNullOrWhiteSpace($QwenLayerValueBits) -or -not [string]::IsNullOrWhiteSpace($GemmaLayerValueBits)) {
    throw "Layer value-bit overrides are diagnostic-only for this production gate. Use run_accuracy_gate.ps1 or run_safe_full_gate.ps1 experimental lanes for V2/Boundary-V measurements."
}

function Resolve-ExpectedEffectiveBitsForHighGqa([string] $Preset, [string] $Override, [string] $LayerKeyBits, [string] $LayerValueBits) {
    if ($Override -ne "auto") {
        return $Override
    }
    if (-not [string]::IsNullOrWhiteSpace($LayerKeyBits) -or -not [string]::IsNullOrWhiteSpace($LayerValueBits)) {
        return ""
    }
    $m = [regex]::Match($Preset, "(?i)kvarn_k([0-9]+)v([0-9]+)")
    if (-not $m.Success) {
        return ""
    }
    return ("k8/v{0}" -f ([int] $m.Groups[2].Value))
}

$unsafeDiagnosticEnv = @(
    "LLAMA_KVARN_ATTN_ENABLE_256D_WARPQK",
    "LLAMA_KVARN_ATTN_ENABLE_256D_BODY_MIRROR",
    "LLAMA_KVARN_ATTN_WARPQK_FORCE_QT",
    "LLAMA_KVARN_ATTN_SPLIT_KERNELS",
    "LLAMA_KVARN_ATTN_SERIAL_FUSED",
    "LLAMA_KVARN_ATTN_REF_SCRATCH",
    "LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE",
    "LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR",
    "LLAMA_KVARN_ATTN_DISABLE_BODY_F32_MIRROR",
    "LLAMA_KVARN_ATTN_FUSED_BATCH",
    "LLAMA_KVARN_ATTN_DISABLE_Q1_GQA_SCALAR",
    "LLAMA_KVARN_ATTN_REQUIRE_Q1_GQA_SCALAR",
    "LLAMA_KVARN_DISABLE_FUSED_FWHT",
    "LLAMA_KVARN_LAYER_KEY_BITS",
    "LLAMA_KVARN_LAYER_VALUE_BITS",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_DIR",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_CALL",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LAYER",
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_HEAD_DIM",
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
    "LLAMA_KVARN_EXPERIMENTAL_TURBO_V",
    "LLAMA_KVARN_EXPERIMENTAL_TURBO_V_LAYOUT",
    "LLAMA_KVARN_DISABLE_DIRECT_RECORD_BATCH_PHASES",
    "LLAMA_KVARN_REQUIRE_DIRECT_RECORD_BATCH_PHASES",
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

$safeCliParityScript = Join-Path (Get-Location) "scripts/kvarn/run_safe_cli_parity_matrix.ps1"
$accuracyScript = Join-Path (Get-Location) "scripts/kvarn/run_accuracy_gate.ps1"
$logitsScript = Join-Path (Get-Location) "scripts/kvarn/compare_cuda_logits_ref.ps1"
if (-not (Test-Path -LiteralPath $safeCliParityScript)) {
    throw "Missing $safeCliParityScript"
}
if (-not $SkipAccuracy.IsPresent -and -not (Test-Path -LiteralPath $accuracyScript)) {
    throw "Missing $accuracyScript"
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
$qwenAccuracyExpectedEffectiveBits = Resolve-ExpectedEffectiveBitsForHighGqa $AccuracyPreset $QwenExpectedEffectiveKvarnBits $QwenLayerKeyBits $QwenLayerValueBits
$gemmaAccuracyExpectedEffectiveBits = Resolve-ExpectedEffectiveBitsForHighGqa $AccuracyPreset $GemmaExpectedEffectiveKvarnBits $GemmaLayerKeyBits $GemmaLayerValueBits
$qwenSpeedExpectedEffectiveBits = Resolve-ExpectedEffectiveBitsForHighGqa $SpeedPreset $QwenExpectedEffectiveKvarnBits $QwenLayerKeyBits $QwenLayerValueBits
$gemmaSpeedExpectedEffectiveBits = Resolve-ExpectedEffectiveBitsForHighGqa $SpeedPreset $GemmaExpectedEffectiveKvarnBits $GemmaLayerKeyBits $GemmaLayerValueBits
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
    "tier2_min_active_kvarn_body_records=$Tier2MinActiveKvarnBodyRecords",
    "allow_diagnostic_env=$($AllowDiagnosticEnv.IsPresent)",
    "qwen_extra_args=$($qwenEffectiveExtraArgs -join ' ')",
    "qwen_expected_kvarn_layers=3-39:4",
    "qwen_kvarn_env_LLAMA_KVARN_ENABLE_PAPER_FRAME=1",
    "qwen_layer_key_bits=$QwenLayerKeyBits",
    "qwen_layer_value_bits=$QwenLayerValueBits",
    "qwen_accuracy_expected_effective_kvarn_bits=$qwenAccuracyExpectedEffectiveBits",
    "qwen_speed_expected_effective_kvarn_bits=$qwenSpeedExpectedEffectiveBits",
    "gemma_production_mode=true-kvarn-iswa",
    "gemma_layer_key_bits=$GemmaLayerKeyBits",
    "gemma_layer_value_bits=$GemmaLayerValueBits",
    "gemma_accuracy_expected_effective_kvarn_bits=$gemmaAccuracyExpectedEffectiveBits",
    "gemma_speed_expected_effective_kvarn_bits=$gemmaSpeedExpectedEffectiveBits",
    "speed_preset=$SpeedPreset",
    "speed_evidence_case_list=$SpeedEvidenceCaseList",
    "trace_fwht_evidence=$($TraceFwhtEvidence.IsPresent)",
    "min_fwht_taken=$MinFwhtTaken",
    "min_batched_store_phase_uses=$MinBatchedStorePhaseUses",
    "run_gemma_fallback_diagnostic=$($RunGemmaFallbackDiagnostic.IsPresent)",
    "gemma_experimental_mode=true-kvarn-iswa-diagnostic"
)
[System.IO.File]::WriteAllText((Join-Path $OutputDir "manifest.txt"), ($manifest -join "`n") + "`n")

if (-not $SkipBuild.IsPresent) {
    Invoke-Logged "build production and KVarN test targets" {
        cmake --build $BuildDir --config Release --target `
            llama-cli `
            llama-results `
            llama-perplexity `
            llama-tokenize `
            test-batch-split `
            test-kvarn-kv `
            test-kvarn-cuda-scratch-ref `
            test-kvarn-server-load-failure `
            test-arg-parser `
            -j 1
    }
}

if (-not $SkipTests.IsPresent) {
    Invoke-Logged "ctest kvarn" {
        ctest --test-dir $BuildDir -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda-scratch-ref|test-kvarn-server-load-failure|test-arg-parser" --output-on-failure
    }
    Invoke-Logged "kv memory estimate self-test" {
        python scripts/kvarn/kv_memory_estimate.py --self-test
    }
}

if (-not $SkipAccuracy.IsPresent) {
    Invoke-Logged "accuracy qwen kl" {
        Invoke-WithProcessEnvironment @{
            "LLAMA_KVARN_ENABLE_PAPER_FRAME" = "1"
            "LLAMA_KVARN_LAYER_KEY_BITS" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerKeyBits)) { $null } else { $QwenLayerKeyBits })
            "LLAMA_KVARN_LAYER_VALUE_BITS" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerValueBits)) { $null } else { $QwenLayerValueBits })
        } {
            & $accuracyScript `
                -Model $QwenModel `
                -Dataset $QwenDataset `
                -BuildDir $BuildDir `
                -OutputDir (Join-Path $OutputDir "qwen-accuracy-kl") `
                -KvarnPreset $AccuracyPreset `
                -KvarnIters 4 `
                -ContextSize $AccuracyContextSize `
                -BatchSize $AccuracyBatchSize `
                -Chunks $AccuracyChunks `
                -UseKLDivergence `
                -MaxMeanKL $AccuracyMaxMeanKL `
                -MaxKLD99 $AccuracyMaxKLD99 `
                -MaxKLD999 $AccuracyMaxKLD999 `
                -MaxKLDMax $AccuracyMaxKLDMax `
                -MaxKLPplIncrease $AccuracyMaxKLPplIncrease `
                -MaxKLPDiffRms $AccuracyMaxKLPDiffRms `
                -MinKLSameTopP $AccuracyMinKLSameTopP `
                -ExpectedKvarnLayers "3-39:4" `
                -ExpectedEffectiveKvarnBits $qwenAccuracyExpectedEffectiveBits `
                -MinKvarnBodyRecords 1 `
                -ExtraArgs @($qwenEffectiveExtraArgs)
        }
    }

    Invoke-Logged "accuracy gemma kl" {
        Invoke-WithProcessEnvironment @{
            "LLAMA_KVARN_ENABLE_PAPER_FRAME" = "1"
            "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA" = "1"
            "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK" = $null
            "LLAMA_KVARN_LAYER_KEY_BITS" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerKeyBits)) { $null } else { $GemmaLayerKeyBits })
            "LLAMA_KVARN_LAYER_VALUE_BITS" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerValueBits)) { $null } else { $GemmaLayerValueBits })
        } {
            & $accuracyScript `
                -Model $GemmaModel `
                -Dataset $GemmaDataset `
                -BuildDir $BuildDir `
                -OutputDir (Join-Path $OutputDir "gemma-accuracy-kl") `
                -KvarnPreset $AccuracyPreset `
                -KvarnIters 4 `
                -ContextSize $AccuracyContextSize `
                -BatchSize $AccuracyBatchSize `
                -Chunks $AccuracyChunks `
                -UseKLDivergence `
                -ParseSpecial `
                -MaxMeanKL $AccuracyMaxMeanKL `
                -MaxKLD99 $AccuracyMaxKLD99 `
                -MaxKLD999 $AccuracyMaxKLD999 `
                -MaxKLDMax $AccuracyMaxKLDMax `
                -MaxKLPplIncrease $AccuracyMaxKLPplIncrease `
                -MaxKLPDiffRms $AccuracyMaxKLPDiffRms `
                -MinKLSameTopP $AccuracyMinKLSameTopP `
                -ExpectedKvarnLayers "5-47:6" `
                -ExpectedEffectiveKvarnBits $gemmaAccuracyExpectedEffectiveBits `
                -MinKvarnBodyRecords 1
        }
    }
}

Invoke-Logged "tier1 qwen mainline parity" {
    Invoke-WithProcessEnvironment @{
        "LLAMA_KVARN_ENABLE_PAPER_FRAME" = "1"
        "LLAMA_KVARN_LAYER_KEY_BITS" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerKeyBits)) { $null } else { $QwenLayerKeyBits })
        "LLAMA_KVARN_LAYER_VALUE_BITS" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerValueBits)) { $null } else { $QwenLayerValueBits })
    } {
        & $safeCliParityScript `
            -Model $QwenModel `
            -MainlineBuildDir $MainlineBuildDir `
            -KvarnBuildDir $BuildDir `
            -CaseList "pp512:512:0,tg64:0:64" `
            -FlashAttn off `
            -MainlineFlashAttn on `
            -KvarnFlashAttn off `
            -CacheTypeK q8_0 `
            -CacheTypeV q8_0 `
            -KvarnPreset $SpeedPreset `
            -Repetitions $QwenRepetitions `
            -KvarnIters 4 `
            -MinParityRatio $Tier1MinRatio `
            -FailBelowMinParityRatio `
            -MinKvarnLayerLogs 10 `
            -ExpectedKvarnLayers "3-39:4" `
            -ExpectedEffectiveKvarnBits $qwenSpeedExpectedEffectiveBits `
            -KvarnPaperFrame `
            -KvarnDirectRecordBatch `
            -NCpuMoe 34 `
            -OutputDir (Join-Path $OutputDir "qwen-tier1-mainline-parity-timed") `
            -GpuLayers 99 `
            -ExtraArgs @($qwenEffectiveExtraArgs)
    }
}

Invoke-Logged "tier1 qwen path evidence" {
    Invoke-WithProcessEnvironment @{
        "LLAMA_KVARN_ENABLE_PAPER_FRAME" = "1"
        "LLAMA_KVARN_LAYER_KEY_BITS" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerKeyBits)) { $null } else { $QwenLayerKeyBits })
        "LLAMA_KVARN_LAYER_VALUE_BITS" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerValueBits)) { $null } else { $QwenLayerValueBits })
    } {
        $params = @{
            Model                         = $QwenModel
            MainlineBuildDir              = $MainlineBuildDir
            KvarnBuildDir                 = $BuildDir
            CaseList                      = $SpeedEvidenceCaseList
            FlashAttn                     = "off"
            KvarnPreset                   = $SpeedPreset
            Repetitions                   = 1
            KvarnIters                    = 4
            MinParityRatio                = 0.01
            MinKvarnLayerLogs             = 10
            ExpectedKvarnLayers           = "3-39:4"
            ExpectedEffectiveKvarnBits    = $qwenSpeedExpectedEffectiveBits
            KvarnPaperFrame               = $true
            KvarnDirectRecordBatch        = $true
            RequireDirectRecordBatchPhases = $true
            NCpuMoe                       = 34
            TraceStore                    = $true
            MinBatchedStorePhaseUses      = $MinBatchedStorePhaseUses
            OutputDir                     = (Join-Path $OutputDir "qwen-tier1-kvarn-path-evidence")
            GpuLayers                     = 99
            ExtraArgs                     = @($qwenEffectiveExtraArgs)
        }
        if ($TraceFwhtEvidence.IsPresent) {
            $params["TraceFwht"] = $true
            $params["MinFwhtTaken"] = $MinFwhtTaken
        }
        & $safeCliParityScript @params
    }
}

Invoke-Logged "tier1 gemma production true kvarn iswa" {
    Invoke-WithProcessEnvironment @{
        "LLAMA_KVARN_ENABLE_PAPER_FRAME" = "1"
        "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA" = "1"
        "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK" = $null
        "LLAMA_KVARN_LAYER_KEY_BITS" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerKeyBits)) { $null } else { $GemmaLayerKeyBits })
        "LLAMA_KVARN_LAYER_VALUE_BITS" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerValueBits)) { $null } else { $GemmaLayerValueBits })
    } {
        & $safeCliParityScript `
            -Model $GemmaModel `
            -MainlineBuildDir $MainlineBuildDir `
            -KvarnBuildDir $BuildDir `
            -CaseList "pp512:512:0,tg64:0:64" `
            -FlashAttn off `
            -MainlineFlashAttn on `
            -KvarnFlashAttn off `
            -CacheTypeK q8_0 `
            -CacheTypeV q8_0 `
            -KvarnPreset $SpeedPreset `
            -Repetitions $GemmaRepetitions `
            -KvarnIters 4 `
            -MinParityRatio $Tier1MinRatio `
            -FailBelowMinParityRatio `
            -MinKvarnLayerLogs 8 `
            -ExpectedKvarnLayers "5-47:6" `
            -ExpectedEffectiveKvarnBits $gemmaSpeedExpectedEffectiveBits `
            -KvarnPaperFrame `
            -KvarnDirectRecordBatch `
            -OutputDir (Join-Path $OutputDir "gemma-tier1-production-true-kvarn-iswa-timed") `
            -GpuLayers 99
    }
}

Invoke-Logged "tier1 gemma path evidence" {
    Invoke-WithProcessEnvironment @{
        "LLAMA_KVARN_ENABLE_PAPER_FRAME" = "1"
        "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA" = "1"
        "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK" = $null
        "LLAMA_KVARN_LAYER_KEY_BITS" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerKeyBits)) { $null } else { $GemmaLayerKeyBits })
        "LLAMA_KVARN_LAYER_VALUE_BITS" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerValueBits)) { $null } else { $GemmaLayerValueBits })
    } {
        $params = @{
            Model                         = $GemmaModel
            MainlineBuildDir              = $MainlineBuildDir
            KvarnBuildDir                 = $BuildDir
            CaseList                      = $SpeedEvidenceCaseList
            FlashAttn                     = "off"
            KvarnPreset                   = $SpeedPreset
            Repetitions                   = 1
            KvarnIters                    = 4
            MinParityRatio                = 0.01
            MinKvarnLayerLogs             = 8
            ExpectedKvarnLayers           = "5-47:6"
            ExpectedEffectiveKvarnBits    = $gemmaSpeedExpectedEffectiveBits
            KvarnPaperFrame               = $true
            KvarnDirectRecordBatch        = $true
            RequireDirectRecordBatchPhases = $true
            TraceStore                    = $true
            MinBatchedStorePhaseUses      = $MinBatchedStorePhaseUses
            OutputDir                     = (Join-Path $OutputDir "gemma-tier1-kvarn-path-evidence")
            GpuLayers                     = 99
        }
        if ($TraceFwhtEvidence.IsPresent) {
            $params["TraceFwht"] = $true
            $params["MinFwhtTaken"] = $MinFwhtTaken
        }
        & $safeCliParityScript @params
    }
}

if ($RunGemmaFallbackDiagnostic.IsPresent) {
    Invoke-Logged "tier1 gemma diagnostic normal iswa fallback" {
        Invoke-WithProcessEnvironment @{
            "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA" = $null
            "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK" = "1"
        } {
            & $safeCliParityScript `
                -Model $GemmaModel `
                -MainlineBuildDir $MainlineBuildDir `
                -KvarnBuildDir $BuildDir `
                -CaseList "pp512:512:0,tg64:0:64" `
                -FlashAttn off `
                -MainlineFlashAttn on `
                -KvarnFlashAttn off `
                -CacheTypeK q8_0 `
                -CacheTypeV q8_0 `
                -Repetitions $GemmaRepetitions `
                -KvarnIters 4 `
                -MinParityRatio $Tier1MinRatio `
                -FailBelowMinParityRatio `
                -AllowKvarnFallback `
                -OutputDir (Join-Path $OutputDir "gemma-tier1-diagnostic-iswa-fallback") `
                -GpuLayers 99
        }
    }
}

if ($RunGemmaExperimental.IsPresent) {
    Invoke-Logged "tier1 gemma experimental true kvarn iswa diagnostic" {
        Invoke-WithProcessEnvironment @{
            "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA" = "1"
            "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK" = $null
        } {
            & $safeCliParityScript `
                -Model $GemmaModel `
                -MainlineBuildDir $MainlineBuildDir `
                -KvarnBuildDir $BuildDir `
                -CaseList "pp512:512:0,tg64:0:64" `
                -FlashAttn off `
                -Repetitions $GemmaRepetitions `
                -KvarnIters 4 `
                -MinParityRatio $Tier1MinRatio `
                -FailBelowMinParityRatio `
                -MinKvarnLayerLogs 8 `
                -ExpectedKvarnLayers "5-47:6" `
                -OutputDir (Join-Path $OutputDir "gemma-tier1-experimental-true-kvarn-iswa-diagnostic") `
                -GpuLayers 99
        }
    }
}

if ($RunTier2.IsPresent) {
    Invoke-Logged "tier2 logits" {
        $tier2LogitsArgs = @(
            "-Model", $Tier2Model,
            "-BuildDir", $BuildDir,
            "-Context", "512",
            "-Batch", "512",
            "-Repeat", "16",
            "-KvarnIters", "4",
            "-CheckPackedRepeat",
            "-CheckPackedSplit",
            "-ScratchMaxNmse", "1e-5",
            "-SplitMaxNmse", "1e-5",
            "-RepeatMaxNmse", "1e-12",
            "-FlashAttn", "off",
            "-MinKvarnLayerLogs", [string] $tier2EffectiveMinLayerLogs,
            "-MinKvarnBodyRecords", [string] $Tier2MinKvarnBodyRecords,
            "-MinActiveKvarnBodyRecords", [string] $Tier2MinActiveKvarnBodyRecords,
            "-ExpectedKvarnLayers", $tier2EffectiveExpectedLayers
        )
        if ($Tier2MinActiveKvarnBodyRecords -gt 0) {
            $tier2LogitsArgs += "-TraceAttn"
        }
        $tier2LogitsArgs += "-ExtraArgs"
        $tier2LogitsArgs += @($tier2EffectiveExtraArgs)
        & $logitsScript @tier2LogitsArgs
    }
}

Write-Host "KVarN production gate complete: $OutputDir"
