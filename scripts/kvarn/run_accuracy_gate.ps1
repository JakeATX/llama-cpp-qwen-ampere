<#
.SYNOPSIS
    KVarN end-to-end accuracy gate: prove KVarN attention is numerically
    faithful to a non-KVarN baseline reference on the SAME build/model/dataset.

.DESCRIPTION
    Every existing KVarN gate (packed-vs-split NMSE, mainline throughput
    parity) compares two KVarN code paths to each other. None of them
    compares KVarN against a non-KVarN reference model, so a *systematic* KVarN
    numerical error (e.g. a rotation that is applied to the stored body but
    not to Q / sink-tail / output) is invisible to the current suite while
    still degrading generation quality - exactly the "errors accumulate over
    decoding" failure mode the KVarN paper describes.

    This script closes that hole. It runs `llama-perplexity` twice from the
    SAME binary on the SAME model and dataset:
      * baseline: non-KVarN KV cache, default f16 unless -ExtraArgs overrides it
      * kvarn:    --kv-cache-quant kvarn ...
    and fails if KVarN perplexity rises more than -MaxPplIncrease above baseline.
    Using one binary isolates the KV-cache backend from build differences.
    Both runs force `--parallel 1`, `--fit off`, and a batch size no larger
    than the context size because the current KVarN backend supports only one
    active sequence and should not be measured through hidden retry or
    multi-chunk context paths.

    With -UseKLDivergence it instead measures the per-token KL divergence of
    the KVarN logit distribution against the baseline base, which is far more
    sensitive to systematic attention errors than raw PPL.

    A passing run here is the precondition for trusting any KVarN throughput
    number. A large PPL/KL gap is direct evidence of a correctness bug, not a
    tuning problem - investigate the rotation handling first (Q is not
    Hadamard-rotated in the KVarN graph path; sink/tail K are stored
    un-rotated while the body is rotated).

.NOTES
    Mirrors the conventions of run_mainline_parity_matrix.ps1 /
    run_production_gate.ps1 (Resolve-BuildExe, artifact layout, git SHA and
    CUDA device capture, KVarN cache-engagement check).
#>
param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $Dataset,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $PerplexityExe = "",
    [string] $OutputDir = "",
    [string] $KvarnPreset = "kvarn_k4v2_g128",
    [int]    $KvarnIters = 4,
    [double] $KvarnRtnQuantile = 1.0,
    [string] $FlashAttn = "off",
    [string] $Fit = "off",
    [int]    $GpuLayers = 999,
    [int]    $ContextSize = 0,
    [int]    $BatchSize = 0,
    [int]    $Parallel = 1,
    [int]    $Chunks = 0,
    [double] $MaxPplIncrease = 0.05,
    [double] $MaxBaselinePpl = 100.0,
    [double] $MaxFixtureUnkRate = 0.001,
    [double] $MaxFixtureTokenUnkRate = 0.001,
    [double] $MaxFixtureSuppressedTokenRate = 0.0,
    [double] $MaxMeanKL = 0.02,
    [double] $MaxKLD99 = 1.0,
    [double] $MaxKLD999 = 1.0,
    [double] $MaxKLDMax = 1.0,
    [double] $MaxKLPplIncrease = -1.0,
    [double] $MaxKLPDiffRms = -1.0,
    [double] $MinKLSameTopP = -1.0,
    [switch] $UseKLDivergence,
    [switch] $ParseSpecial,
    [switch] $AllowChatMarkers,
    [switch] $SkipFixtureCheck,
    [switch] $AllowKvarnFallback,
    [switch] $AllowDiagnosticEnv,
    [string] $ExpectedKvarnLayers = "",
    [string] $ExpectedEffectiveKvarnBits = "",
    [int] $MinKvarnBodyRecords = 0,
    [int] $MinActiveKvarnBodyRecords = 0,
    [switch] $TraceAttn,
    [int] $TraceLimit = 64,
    [switch] $TraceFwht,
    [int] $MinFwhtTaken = 0,
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @()
)

$ErrorActionPreference = "Stop"

if ($MaxPplIncrease -lt 0.0) {
    throw "MaxPplIncrease must be non-negative"
}
if ($MaxBaselinePpl -lt 0.0) {
    throw "MaxBaselinePpl must be non-negative; use 0 to disable baseline-PPL sanity checking"
}
if ($MaxFixtureUnkRate -lt 0.0) {
    throw "MaxFixtureUnkRate must be non-negative"
}
if ($MaxFixtureTokenUnkRate -lt 0.0) {
    throw "MaxFixtureTokenUnkRate must be non-negative"
}
if ($MaxFixtureSuppressedTokenRate -lt 0.0) {
    throw "MaxFixtureSuppressedTokenRate must be non-negative"
}
if ($MaxMeanKL -lt 0.0) {
    throw "MaxMeanKL must be non-negative"
}
if ($MaxKLD99 -lt 0.0 -or $MaxKLD999 -lt 0.0 -or $MaxKLDMax -lt 0.0) {
    throw "MaxKLD99, MaxKLD999, and MaxKLDMax must be non-negative; use 0 to disable a tail gate"
}
if ($MaxKLPplIncrease -lt -1.0) {
    throw "MaxKLPplIncrease must be -1 to disable or non-negative"
}
if ($MaxKLPDiffRms -lt -1.0) {
    throw "MaxKLPDiffRms must be -1 to disable or non-negative"
}
if ($MinKLSameTopP -lt -1.0 -or $MinKLSameTopP -gt 1.0) {
    throw "MinKLSameTopP must be -1 to disable or in [0, 1]"
}
if ($Parallel -ne 1) {
    throw "KVarN accuracy gate requires -Parallel 1 because KVarN currently supports only n_seq_max = 1"
}
if ($BatchSize -lt 0) {
    throw "BatchSize must be non-negative"
}
if ($Chunks -lt 0) {
    throw "Chunks must be non-negative"
}
if ($MinKvarnBodyRecords -lt 0 -or $MinActiveKvarnBodyRecords -lt 0) {
    throw "MinKvarnBodyRecords and MinActiveKvarnBodyRecords must be non-negative"
}
if ($TraceLimit -le 0) {
    throw "TraceLimit must be positive"
}
if ($MinFwhtTaken -lt 0) {
    throw "MinFwhtTaken must be non-negative"
}
if ($MinActiveKvarnBodyRecords -gt 0 -and -not $TraceAttn.IsPresent) {
    throw "MinActiveKvarnBodyRecords requires -TraceAttn so the script can prove active body use"
}
if ($MinFwhtTaken -gt 0 -and -not $TraceFwht.IsPresent) {
    throw "MinFwhtTaken requires -TraceFwht so the script can prove CUDA FWHT use"
}
if ($Fit -ne "off") {
    throw "KVarN accuracy gate requires -Fit off because the auto-fit path can retry unsupported multi-sequence settings"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found at $Model"
}
if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "Dataset not found at $Dataset"
}

function Resolve-BuildExe([string] $Dir, [string] $Name) {
    $candidates = @(
        (Join-Path $Dir "bin/Release/$Name"),
        (Join-Path $Dir "bin/$Name"),
        (Join-Path $Dir "bin/Release/$Name.exe"),
        (Join-Path $Dir "bin/$Name.exe")
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    throw "Missing $Name under $Dir"
}

function Get-GitHead([string] $RepoRoot) {
    if (-not (Test-Path -LiteralPath $RepoRoot)) { return "unknown" }
    Push-Location $RepoRoot
    try { return (git rev-parse --short HEAD 2>$null) } finally { Pop-Location }
}

function Get-FileSha256([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return "missing"
    }
    try {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    } catch {
        return "hash failed: $($_.Exception.Message)"
    }
}

function Get-GpuMemorySnapshot() {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return "nvidia-smi unavailable"
    }
    try {
        $rows = & $nvidiaSmi.Path --query-gpu=memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or $null -eq $rows) {
            return "nvidia-smi query failed"
        }
        return (($rows | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }) -join "; ")
    } catch {
        return "nvidia-smi query failed: $($_.Exception.Message)"
    }
}

function Get-ArgOptionValue([string[]] $Argv, [string] $ShortName, [string] $LongName, [string] $DefaultValue) {
    for ($i = 0; $i -lt $Argv.Count; ++$i) {
        $arg = $Argv[$i]
        if (($arg -eq $ShortName -or $arg -eq $LongName) -and ($i + 1) -lt $Argv.Count) {
            return $Argv[$i + 1]
        }
        if (-not [string]::IsNullOrWhiteSpace($LongName) -and $arg.StartsWith("$LongName=")) {
            return $arg.Substring($LongName.Length + 1)
        }
    }
    return $DefaultValue
}

function Get-CudaDeviceSummary([string] $Text) {
    $m = [regex]::Match($Text, "ggml_cuda_init: found .*")
    if ($m.Success) { return $m.Value.Trim() }
    $d = [regex]::Match($Text, "Device \d+: .*")
    if ($d.Success) { return $d.Value.Trim() }
    return ""
}

function Get-FinalPpl([string] $Text) {
    # llama-perplexity prints "Final estimate: PPL = 6.2345 +/- 0.03456"
    $m = [regex]::Match($Text, "Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return [double]::NaN
}

function Get-MeanKL([string] $Text) {
    # llama-perplexity --kl-divergence prints "Mean    KLD:   0.012345 ..."
    $m = [regex]::Match($Text, "Mean\s+KLD:\s*(-?[0-9]+(?:\.[0-9]+)?)")
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return [double]::NaN
}

function Get-KLMeanPplBase([string] $Text) {
    $m = [regex]::Match($Text, "Mean\s+PPL\(base\)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return [double]::NaN
}

function Get-KLMeanPplRatio([string] $Text) {
    $m = [regex]::Match($Text, "Mean\s+PPL\(Q\)/PPL\(base\)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return [double]::NaN
}

function Get-KLPDiffRms([string] $Text) {
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -notmatch "RMS" -or $line -notmatch "p\s*:") {
            continue
        }
        $m = [regex]::Match($line, ":\s*([0-9]+(?:\.[0-9]+)?)")
        if ($m.Success) { return ([double]$m.Groups[1].Value)/100.0 }
    }
    return [double]::NaN
}

function Get-KLSameTopP([string] $Text) {
    $m = [regex]::Match($Text, "Same top p:\s*([0-9]+(?:\.[0-9]+)?)")
    if ($m.Success) { return ([double]$m.Groups[1].Value)/100.0 }
    return [double]::NaN
}

function Get-KLMetric([string] $Text, [string] $LabelRegex) {
    $m = [regex]::Match($Text, "${LabelRegex}\s+KLD:\s*(-?[0-9]+(?:\.[0-9]+)?)")
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return [double]::NaN
}

function Convert-ToFileStem([string] $Name) {
    return ($Name -replace "[^A-Za-z0-9_.-]", "_")
}

function Test-KvarnPresetRequestsV2([string] $Preset) {
    $m = [regex]::Match($Preset, "(?i)^kvarn_k[0-9]+v([0-9]+)_g128$")
    return $m.Success -and ([int] $m.Groups[1].Value) -eq 2
}

function Convert-ToNativeArgument([string] $Arg) {
    if ($Arg -notmatch '[\s"]') {
        return $Arg
    }
    $escaped = $Arg -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Invoke-NativeCommandCaptured {
    param(
        [string] $Exe,
        [string[]] $Argv
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Exe
    $psi.Arguments = (($Argv | ForEach-Object { Convert-ToNativeArgument $_ }) -join " ")
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit()

    return [pscustomobject]@{
        ExitCode = $proc.ExitCode
        Text     = ($stdoutTask.GetAwaiter().GetResult() + $stderrTask.GetAwaiter().GetResult())
    }
}

function Get-BaselinePplSanityFailure([double] $PplBase, [double] $MaxPpl, [string] $LogName) {
    if ($MaxPpl -le 0.0) {
        return ""
    }
    if ([double]::IsNaN($PplBase)) {
        return ("baseline reference PPL could not be parsed; the corpus/model pairing is not a trustworthy reference for KVarN correctness. Fix the run or pass -MaxBaselinePpl 0 intentionally. See {0}" -f $LogName)
    }
    if ($PplBase -gt $MaxPpl) {
        return ("baseline reference PPL {0:N4} exceeds -MaxBaselinePpl {1:N4}; the corpus/model pairing is not a trustworthy reference for KVarN correctness. Fix the fixture or pass -MaxBaselinePpl 0 intentionally. See {2}" -f `
            $PplBase, $MaxPpl, $LogName)
    }
    return ""
}

function Get-ProcessEnv([string] $Name) {
    return [Environment]::GetEnvironmentVariable($Name, "Process")
}

function Get-SetKvarnDiagnosticEnvs {
    $diagnosticNames = @(
        "LLAMA_KVARN_DEBUG_RAW_BODY_K",
        "LLAMA_KVARN_DEBUG_RAW_BODY_V",
        "LLAMA_KVARN_DEBUG_CAPTURE_RAW_BODY_MIRROR",
        "LLAMA_KVARN_ATTN_REF_SCRATCH",
        "LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR",
        "LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE",
        "LLAMA_KVARN_ATTN_DISABLE_BODY_F32_MIRROR",
        "LLAMA_KVARN_DEBUG_RAW_BODY_SCALAR_QT",
        "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK",
        "LLAMA_KVARN_ISWA_DEBUG_FULL_NORMAL_ATTN",
        "LLAMA_KVARN_ISWA_DEBUG_MATERIALIZE_MHA",
        "LLAMA_KVARN_ISWA_DEBUG_DUAL_MHA_COMPARE",
        "LLAMA_KVARN_ISWA_DEBUG_RAW_MHA_COMPARE",
        "LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME",
        "LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME_IDENTITY",
        "LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME_REVERSE",
        "LLAMA_KVARN_EXPERIMENTAL_TURBO_V",
        "LLAMA_KVARN_EXPERIMENTAL_TURBO_V_LAYOUT",
        "LLAMA_KVARN_UNSAFE_ENABLE_ISWA_PREFILL_DIRECT_ATTN",
        "LLAMA_KVARN_PREFILL_DIRECT_TRACE",
        "LLAMA_KVARN_PREFILL_DIRECT_TRACE_LIMIT",
        "LLAMA_KVARN_PAPER_MIXED_FRAME",
        "LLAMA_KVARN_UNSAFE_ALLOW_PAPER_MIXED_FRAME",
        "LLAMA_KVARN_GEMMA4_PROTECT_DONORS",
        "LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE",
        "LLAMA_KVARN_DISABLE_PREFILL_DIRECT_ATTN",
        "LLAMA_KVARN_DISABLE_PREFILL_DIRECT_STORE",
        "LLAMA_KVARN_DISABLE_INTERNAL_CAUSAL_MASK",
        "LLAMA_KVARN_ATTN_FORCE_COMPACT_CAUSAL_MASK",
        "LLAMA_KVARN_DISABLE_ISWA_SINKTAIL_MHA",
        "LLAMA_KVARN_DISABLE_PREFILL_PINGPONG",
        "LLAMA_KVARN_ATTN_TRACE",
        "LLAMA_KVARN_ATTN_TRACE_LIMIT",
        "LLAMA_KVARN_ATTN_DISABLE_256D_SCALAR_QT",
        "LLAMA_KVARN_ATTN_DISABLE_GQA_SCALAR_QT",
        "LLAMA_KVARN_ATTN_WARPQK_FORCE_QT",
        "LLAMA_KVARN_FWHT_TRACE",
        "LLAMA_KVARN_STORE_TRACE",
        "LLAMA_KVARN_STORE_TRACE_LIMIT",
        "LLAMA_KVARN_TENSOR_DUMP_DIR",
        "LLAMA_KVARN_TENSOR_DUMP_FILTER",
        "LLAMA_KVARN_TENSOR_DUMP_LIMIT",
        "LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME_KEEP_F32",
        "LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME_DENSE",
        "LLAMA_KVARN_LAYER_FILTER",
        "LLAMA_KVARN_LAYER_KEY_BITS",
        "LLAMA_KVARN_LAYER_VALUE_BITS",
        "LLAMA_KVARN_DISABLE_HIGH_GQA_K8"
    )
    $set = @()
    foreach ($name in $diagnosticNames) {
        $value = Get-ProcessEnv $name
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            $set += ("{0}={1}" -f $name, $value)
        }
    }
    return $set
}

function Assert-NoBaselineOnlyArgsInKvarnExtraArgs([string[]] $Argv) {
    $baselineOnlyFlags = @(
        "-ncmoe",
        "--n-cpu-moe",
        "--cpu-moe"
    )

    foreach ($arg in $Argv) {
        $name = $arg
        $eq = $name.IndexOf("=")
        if ($eq -ge 0) {
            $name = $name.Substring(0, $eq)
        }

        if ($baselineOnlyFlags -contains $name) {
            throw ("{0} must be passed through -ExtraArgs, not -KvarnExtraArgs, because it changes the non-KVarN baseline/KVarN model execution comparison." -f $name)
        }
    }
}

function Get-ExpectedKvarnLayerIds([string] $layers) {
    if ([string]::IsNullOrWhiteSpace($layers)) {
        return @()
    }

    $ids = @()
    foreach ($raw in ($layers -split "[,\s]+")) {
        if ([string]::IsNullOrWhiteSpace($raw)) {
            continue
        }
        if ($raw -match '^([0-9]+)-([0-9]+)(?::([0-9]+))?$') {
            $start = [int] $Matches[1]
            $end = [int] $Matches[2]
            $step = if ($Matches.ContainsKey(3) -and -not [string]::IsNullOrEmpty($Matches[3])) { [int] $Matches[3] } else { 1 }
            if ($end -lt $start -or $step -le 0) {
                throw "Invalid KVarN layer range '$raw' in ExpectedKvarnLayers"
            }
            for ($id = $start; $id -le $end; $id += $step) {
                $ids += $id
            }
        } else {
            $id = 0
            if (-not [int]::TryParse($raw, [ref] $id) -or $id -lt 0) {
                throw "Invalid KVarN layer id '$raw' in ExpectedKvarnLayers"
            }
            $ids += $id
        }
    }
    return @($ids | Sort-Object -Unique)
}

function Get-ObservedKvarnLayerIds([string] $text) {
    $actual = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($m in [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer\s+([0-9]+)")) {
        [void] $actual.Add([int] $m.Groups[1].Value)
    }
    return ,$actual
}

function Normalize-KvarnBits([string] $bits) {
    return (($bits.Trim().ToLowerInvariant()) -replace "[^0-9kv]", "")
}

function Get-KvarnEffectiveBitSet([string] $text) {
    $actual = New-Object 'System.Collections.Generic.SortedSet[string]'
    foreach ($m in [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer\s+[0-9]+.*?effective k(?<k>[0-9]+)/v(?<v>[0-9]+)")) {
        [void] $actual.Add(("k{0}/v{1}" -f $m.Groups["k"].Value, $m.Groups["v"].Value))
    }
    return (($actual | ForEach-Object { [string] $_ }) -join ",")
}

function Get-MaxKvarnBodyRecords([string] $text) {
    $maxRecords = -1
    foreach ($m in [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer\s+[0-9]+.*?\bbody records\s*=\s*([0-9]+)")) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }
    return $maxRecords
}

function Get-MaxActiveKvarnBodyRecords([string] $text) {
    $maxRecords = -1
    foreach ($m in [regex]::Matches($text, "KVarN CUDA mixed-attn trace:.*?\bn_records=([0-9]+)",
            [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }
    foreach ($m in [regex]::Matches($text, "KVarN CUDA mixed-attn inner trace:.*?\brecords=([0-9]+)",
            [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }
    foreach ($m in [regex]::Matches($text, "KVarN graph materialized-MHA trace:.*?\bn_records=([0-9]+)",
            [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }
    return $maxRecords
}

function Get-KvarnFwhtTraceSummary([string] $text) {
    $taken = 0
    $fallback = 0
    $total = 0
    foreach ($m in [regex]::Matches($text, "KVarN CUDA FWHT trace:\s+taken=([01])")) {
        $total++
        if ($m.Groups[1].Value -eq "1") {
            $taken++
        } else {
            $fallback++
        }
    }
    return [pscustomobject]@{
        Total = $total
        Taken = $taken
        Fallback = $fallback
    }
}

function Invoke-PplRun {
    param(
        [string] $Label,
        [string] $Exe,
        [string[]] $Argv,
        [hashtable] $EnvSet = @{}
    )
    $stem = Convert-ToFileStem $Label
    $logPath = Join-Path $OutputDir "$stem.log.txt"
    $cmdPath = Join-Path $OutputDir "$stem.command.txt"
    $commandLine = "`"$Exe`" " + (($Argv | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join " ")
    [System.IO.File]::WriteAllText($cmdPath, $commandLine + "`n")

    Write-Host "== $Label"
    Write-Host $commandLine

    $oldEnv = @{}
    foreach ($key in $EnvSet.Keys) {
        $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $EnvSet[$key], "Process")
    }
    try {
        $result = Invoke-NativeCommandCaptured -Exe $Exe -Argv $Argv
        $exit = $result.ExitCode
        $text = $result.Text
    } finally {
        foreach ($key in $EnvSet.Keys) {
            [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
        }
    }
    [System.IO.File]::WriteAllText($logPath, $text + "`n")
    if ($exit -ne 0) {
        throw "$Label llama-perplexity failed with exit code $exit; see $logPath"
    }
    return [pscustomobject]@{
        Text       = $text
        Log        = (Split-Path -Leaf $logPath)
        Command    = (Split-Path -Leaf $cmdPath)
        CudaDevice = Get-CudaDeviceSummary $text
    }
}

function Invoke-FixturePreflight {
    param(
        [string] $Script,
        [string] $ModelPath,
        [string] $DataPath,
        [string] $TokenizerExe,
        [string] $OutputDirectory,
        [double] $MaxUnkRate,
        [double] $MaxTokenUnkRate,
        [double] $MaxSuppressedTokenRate,
        [int] $MinTokens,
        [bool] $ParseSpecialTokens,
        [bool] $AllowLiteralChatMarkers
    )

    $logPath = Join-Path $OutputDirectory "fixture-preflight.log.txt"
    $cmdPath = Join-Path $OutputDirectory "fixture-preflight.command.txt"
    $argv = @(
        $Script,
        "--dataset", $DataPath,
        "--model", $ModelPath,
        "--tokenizer-exe", $TokenizerExe,
        "--max-unk-rate", ("{0}" -f $MaxUnkRate),
        "--max-token-unk-rate", ("{0}" -f $MaxTokenUnkRate),
        "--max-suppressed-token-rate", ("{0}" -f $MaxSuppressedTokenRate),
        "--min-tokens", [string] $MinTokens,
        "--fail-on-template-mismatch"
    )
    if ($ParseSpecialTokens) {
        $argv += "--parse-special"
    }
    if ($AllowLiteralChatMarkers) {
        $argv += "--allow-chat-markers"
    }
    $commandLine = "python " + (($argv | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join " ")
    [System.IO.File]::WriteAllText($cmdPath, $commandLine + "`n")

    Write-Host "== fixture-preflight"
    Write-Host $commandLine

    $result = Invoke-NativeCommandCaptured -Exe "python" -Argv $argv
    $exit = $result.ExitCode
    $text = $result.Text
    [System.IO.File]::WriteAllText($logPath, $text + "`n")
    if ($exit -ne 0) {
        throw "fixture preflight failed; see $logPath"
    }
    return (Split-Path -Leaf $logPath)
}

function Test-DatasetContainsSpecialMarkers([string] $DataPath) {
    $sample = [System.IO.File]::ReadAllText($DataPath)
    return [regex]::IsMatch($sample, "<\|[^>\s|]+\|>|<\|[^>\s|]+>|<[^>\s|]+\|>|<[^>\s|]+_of_[^>\s]+>")
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($PerplexityExe)) {
    $pplExe = Resolve-BuildExe $BuildDir "llama-perplexity.exe"
} else {
    if (-not (Test-Path -LiteralPath $PerplexityExe)) {
        throw "PerplexityExe not found at $PerplexityExe"
    }
    $pplExe = (Resolve-Path -LiteralPath $PerplexityExe).Path
}
$tokenizeExe = Resolve-BuildExe $BuildDir "llama-tokenize.exe"
$gitHead  = Get-GitHead $repoRoot
$expectedLayerIds = @(Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers)
$expectedLayerSet = New-Object 'System.Collections.Generic.HashSet[int]'
foreach ($id in $expectedLayerIds) {
    [void] $expectedLayerSet.Add([int] $id)
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "artifacts/kvarn-accuracy/$stamp"
} else {
    $OutputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$modelPath = (Resolve-Path -LiteralPath $Model).Path
$dataPath  = (Resolve-Path -LiteralPath $Dataset).Path

$diagnosticEnvValues = @(Get-SetKvarnDiagnosticEnvs)
if ($diagnosticEnvValues.Count -gt 0 -and -not $AllowDiagnosticEnv.IsPresent) {
    throw "KVarN accuracy gate refuses diagnostic/fallback KVarN environment state: $($diagnosticEnvValues -join '; '). Pass -AllowDiagnosticEnv only for diagnostic ablations."
}

if (-not $ParseSpecial.IsPresent -and (Test-DatasetContainsSpecialMarkers $dataPath)) {
    throw "Dataset contains special-token markers; pass -ParseSpecial so llama-perplexity tokenizes them as special tokens instead of ordinary text"
}

$ctxArgv = @()
if ($ContextSize -gt 0) { $ctxArgv = @("-c", [string] $ContextSize) }
$chunkArgv = @()
if ($Chunks -gt 0) { $chunkArgv = @("--chunks", [string] $Chunks) }
$specialArgv = @()
if ($ParseSpecial.IsPresent) { $specialArgv = @("--parse-special") }
$effectiveCtx = if ($ContextSize -gt 0) { $ContextSize } else { 512 }
$effectiveBatch = if ($BatchSize -gt 0) { $BatchSize } else { $effectiveCtx }
if ($effectiveBatch -gt $effectiveCtx) {
    throw "KVarN accuracy gate requires BatchSize <= ContextSize so llama-perplexity keeps n_seq_max = 1"
}
$minFixtureTokens = 0
if ($Chunks -gt 0) {
    # llama-perplexity needs one full context for warmup plus the requested
    # scored chunks; otherwise it exits before writing a valid KL base.
    $minFixtureTokens = $effectiveCtx * ($Chunks + 1)
}

Assert-NoBaselineOnlyArgsInKvarnExtraArgs $KvarnExtraArgs

$fixturePreflightLog = ""
if (-not $SkipFixtureCheck.IsPresent) {
    $fixtureChecker = Join-Path $PSScriptRoot "check_perplexity_fixture.py"
    if (-not (Test-Path -LiteralPath $fixtureChecker)) {
        throw "Missing fixture checker at $fixtureChecker"
    }
    $fixturePreflightLog = Invoke-FixturePreflight -Script $fixtureChecker -ModelPath $modelPath -DataPath $dataPath `
        -TokenizerExe $tokenizeExe -OutputDirectory $OutputDir -MaxUnkRate $MaxFixtureUnkRate `
        -MaxTokenUnkRate $MaxFixtureTokenUnkRate -MaxSuppressedTokenRate $MaxFixtureSuppressedTokenRate `
        -MinTokens $minFixtureTokens `
        -ParseSpecialTokens $ParseSpecial.IsPresent -AllowLiteralChatMarkers $AllowChatMarkers.IsPresent
}

$commonArgv = @(
    "-m", $modelPath,
    "-f", $dataPath,
    "-ngl", [string] $GpuLayers,
    "-np", [string] $Parallel,
    "-b", [string] $effectiveBatch,
    "-fit", $Fit,
    "-fa", $FlashAttn
) + $ctxArgv + $chunkArgv + $specialArgv + $ExtraArgs

$baselineCacheK = Get-ArgOptionValue $commonArgv "-ctk" "--cache-type-k" "f16"
$baselineCacheV = Get-ArgOptionValue $commonArgv "-ctv" "--cache-type-v" "f16"
$baselineFlashAttn = Get-ArgOptionValue $commonArgv "-fa" "--flash-attn" $FlashAttn
$baselineLabel = "baseline-ref"
$baselineDescriptor = ("non-KVarN ctk={0} ctv={1} fa={2}" -f $baselineCacheK, $baselineCacheV, $baselineFlashAttn)

$rtnQuantileArg = ("{0}" -f $KvarnRtnQuantile)
$kvarnCacheArgv = @(
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", $KvarnPreset,
    "--kvarn-iters", [string] $KvarnIters,
    "--kvarn-rtn-quantile", $rtnQuantileArg
) + $KvarnExtraArgs

$kvarnTraceEnv = @{}
if ($TraceAttn.IsPresent) {
    $kvarnTraceEnv["LLAMA_KVARN_ATTN_TRACE"] = "1"
    $kvarnTraceEnv["LLAMA_KVARN_ATTN_TRACE_LIMIT"] = [string] $TraceLimit
}
if ($TraceFwht.IsPresent) {
    $kvarnTraceEnv["LLAMA_KVARN_FWHT_TRACE"] = "1"
}

$gateFailures = @()
$rows = @()
$ranKvarn = $false
$kvarnText = ""
$cudaSummary = ""
$rows += [pscustomobject]@{ Metric = "diagnostic_env_count"; Value = $diagnosticEnvValues.Count; Threshold = $(if ($AllowDiagnosticEnv.IsPresent) { "allowed" } else { "0" }) }

if ($UseKLDivergence.IsPresent) {
    # Sensitive mode: the baseline writes a logit base, KVarN is scored against it.
    $basePath = Join-Path $OutputDir "baseline-logits.base.bin"
    $baseRun = Invoke-PplRun -Label "$baselineLabel-klbase" -Exe $pplExe `
        -Argv ($commonArgv + @("--kl-divergence-base", $basePath))
    $cudaSummary = $baseRun.CudaDevice
    $pplBase = Get-FinalPpl $baseRun.Text
    if ([double]::IsNaN($pplBase) -or $pplBase -le 0.0) {
        throw "baseline reference KL-base run did not report a valid 'Final estimate: PPL'; see $($baseRun.Log)"
    }
    $klBaseCheck = Join-Path $PSScriptRoot "check_kl_base_file.py"
    if (-not (Test-Path -LiteralPath $klBaseCheck)) {
        throw "Missing KL base checker at $klBaseCheck"
    }
    $klBaseCheckLog = Join-Path $OutputDir "kl-base-check.log.txt"
    $klBaseCheckJson = Join-Path $OutputDir "kl-base-check.json"
    $klCheckResult = Invoke-NativeCommandCaptured -Exe "python" -Argv @(
        $klBaseCheck,
        "--base", $basePath,
        "--expected-ppl", ("{0:R}" -f $pplBase),
        "--json", $klBaseCheckJson
    )
    [System.IO.File]::WriteAllText($klBaseCheckLog, $klCheckResult.Text + "`n")
    if ($klCheckResult.ExitCode -ne 0) {
        throw "KL base file sanity check failed; see $klBaseCheckLog"
    }
    $enforceBaselinePpl = $MaxBaselinePpl -gt 0.0
    if ($enforceBaselinePpl) {
        $baselineFailure = Get-BaselinePplSanityFailure $pplBase $MaxBaselinePpl $baseRun.Log
        if (-not [string]::IsNullOrWhiteSpace($baselineFailure)) {
            $gateFailures += $baselineFailure
        }
    }

    $meanKL = [double]::NaN
    $kld99 = [double]::NaN
    $kld999 = [double]::NaN
    $kldMax = [double]::NaN
    $klMeanPplBase = [double]::NaN
    if ($gateFailures.Count -eq 0) {
        $klDumpCsv = Join-Path $OutputDir "kl.csv"
        $oldKlDumpCsv = Get-ProcessEnv "LLAMA_KVARN_KL_DUMP_CSV"
        [Environment]::SetEnvironmentVariable("LLAMA_KVARN_KL_DUMP_CSV", $klDumpCsv, "Process")
        try {
            $kvarnRun = Invoke-PplRun -Label "kvarn-kl" -Exe $pplExe `
                -Argv ($commonArgv + $kvarnCacheArgv + @("--kl-divergence-base", $basePath, "--kl-divergence")) `
                -EnvSet $kvarnTraceEnv
        } finally {
            [Environment]::SetEnvironmentVariable("LLAMA_KVARN_KL_DUMP_CSV", $oldKlDumpCsv, "Process")
        }
        $ranKvarn = $true
        $cudaSummary = $kvarnRun.CudaDevice
        $kvarnText = $kvarnRun.Text
        $hasKlDumpCsv = Test-Path -LiteralPath $klDumpCsv
        $rows += [pscustomobject]@{ Metric = "kl_dump_csv"; Value = $(if ($hasKlDumpCsv) { "kl.csv" } else { "" }); Threshold = "present" }
        if (-not $hasKlDumpCsv) {
            $gateFailures += "KVarN KL run did not produce kl.csv at $klDumpCsv"
        }
        $klDrilldown = Join-Path $PSScriptRoot "analyze_kl_spikes.py"
        if ($hasKlDumpCsv -and (Test-Path -LiteralPath $klDrilldown)) {
            $drilldownLog = Join-Path $OutputDir "kl-drilldown.log.txt"
            $drilldownArgv = @(
                $klDrilldown,
                "--kl-csv", $klDumpCsv,
                "--base-file", $basePath,
                "--dataset", $dataPath,
                "--model", $modelPath,
                "--tokenizer-exe", $tokenizeExe,
                "--top-n", "20",
                "--context-tokens", "12",
                "--json-out", (Join-Path $OutputDir "kl-drilldown.json"),
                "--md-out", (Join-Path $OutputDir "kl-drilldown.md")
            )
            if ($ParseSpecial.IsPresent) {
                $drilldownArgv += "--parse-special"
            }
            $drilldownResult = Invoke-NativeCommandCaptured -Exe "python" -Argv $drilldownArgv
            [System.IO.File]::WriteAllText($drilldownLog, $drilldownResult.Text + "`n")
            if ($drilldownResult.ExitCode -eq 0) {
                $rows += [pscustomobject]@{ Metric = "kl_drilldown"; Value = "kl-drilldown.md"; Threshold = "generated" }
            } else {
                $rows += [pscustomobject]@{ Metric = "kl_drilldown"; Value = "failed"; Threshold = "generated" }
                $gateFailures += "KL drilldown failed; see $drilldownLog"
            }
        } elseif ($hasKlDumpCsv) {
            $rows += [pscustomobject]@{ Metric = "kl_drilldown"; Value = "missing analyzer"; Threshold = "generated" }
            $gateFailures += "Missing KL drilldown analyzer at $klDrilldown"
        }
        $klBoundaries = Join-Path $PSScriptRoot "analyze_kl_boundaries.py"
        if ($hasKlDumpCsv -and (Test-Path -LiteralPath $klBoundaries)) {
            $boundaryLog = Join-Path $OutputDir "kl-boundaries.log.txt"
            $boundaryArgv = @(
                $klBoundaries,
                "--kl-csv", $klDumpCsv,
                "--base-file", $basePath,
                "--n-ctx", [string] $effectiveCtx,
                "--sink", "128",
                "--tail", "128",
                "--group", "128",
                "--top-n", "40",
                "--json-out", (Join-Path $OutputDir "kl-boundaries.json"),
                "--md-out", (Join-Path $OutputDir "kl-boundaries.md")
            )
            $boundaryResult = Invoke-NativeCommandCaptured -Exe "python" -Argv $boundaryArgv
            [System.IO.File]::WriteAllText($boundaryLog, $boundaryResult.Text + "`n")
            if ($boundaryResult.ExitCode -eq 0) {
                $rows += [pscustomobject]@{ Metric = "kl_boundaries"; Value = "kl-boundaries.md"; Threshold = "generated" }
            } else {
                $rows += [pscustomobject]@{ Metric = "kl_boundaries"; Value = "failed"; Threshold = "generated" }
                $gateFailures += "KL boundary analysis failed; see $boundaryLog"
            }
        } elseif ($hasKlDumpCsv) {
            $rows += [pscustomobject]@{ Metric = "kl_boundaries"; Value = "missing analyzer"; Threshold = "generated" }
            $gateFailures += "Missing KL boundary analyzer at $klBoundaries"
        }

        $meanKL = Get-MeanKL $kvarnRun.Text
        $klMeanPplBase = Get-KLMeanPplBase $kvarnRun.Text
        $klPplRatio = Get-KLMeanPplRatio $kvarnRun.Text
        $klPplIncrease = if ([double]::IsNaN($klPplRatio)) { [double]::NaN } else { $klPplRatio - 1.0 }
        $klPDiffRms = Get-KLPDiffRms $kvarnRun.Text
        $klSameTopP = Get-KLSameTopP $kvarnRun.Text
        $kld99 = Get-KLMetric $kvarnRun.Text "99\.0%"
        $kld999 = Get-KLMetric $kvarnRun.Text "99\.9%"
        $kldMax = Get-KLMetric $kvarnRun.Text "Maximum"
        if ([double]::IsNaN($klMeanPplBase)) {
            $gateFailures += "KVarN KL run did not report 'Mean PPL(base)'; see $($kvarnRun.Log)"
        } elseif (-not [double]::IsNaN($pplBase) -and $pplBase -gt 0.0 -and $klMeanPplBase -gt 0.0) {
            $baseLogDiff = [Math]::Abs([Math]::Log($klMeanPplBase) - [Math]::Log($pplBase))
            if ($baseLogDiff -gt 0.005) {
                $gateFailures += ("KL reader Mean PPL(base) {0:N6} disagrees with baseline writer PPL {1:N6}; log-diff {2:E6}. Do not trust KL metrics from this run." -f `
                        $klMeanPplBase, $pplBase, $baseLogDiff)
            }
        }
        if ([double]::IsNaN($meanKL)) {
            $gateFailures += "KVarN run did not report 'Mean KLD'; see $($kvarnRun.Log)"
        }

        if ($MaxKLPplIncrease -ge 0.0 -and [double]::IsNaN($klPplIncrease)) {
            $gateFailures += "KVarN KL run did not report 'Mean PPL(Q)/PPL(base)'; see $($kvarnRun.Log)"
        }
        if ($MaxKLPDiffRms -ge 0.0 -and [double]::IsNaN($klPDiffRms)) {
            $gateFailures += "KVarN KL run did not report RMS target-probability delta; see $($kvarnRun.Log)"
        }
        if ($MinKLSameTopP -ge 0.0 -and [double]::IsNaN($klSameTopP)) {
            $gateFailures += "KVarN KL run did not report same-top probability; see $($kvarnRun.Log)"
        }
        if ($MaxKLPplIncrease -ge 0.0 -and -not [double]::IsNaN($klPplIncrease) -and $klPplIncrease -gt $MaxKLPplIncrease) {
            $gateFailures += ("Mean KL-mode PPL increase {0:P2} exceeds -MaxKLPplIncrease {1:P2}" -f $klPplIncrease, $MaxKLPplIncrease)
        }
        if ($MaxKLPDiffRms -ge 0.0 -and -not [double]::IsNaN($klPDiffRms) -and $klPDiffRms -gt $MaxKLPDiffRms) {
            $gateFailures += ("KL RMS target probability delta {0:P3} exceeds -MaxKLPDiffRms {1:P3}" -f $klPDiffRms, $MaxKLPDiffRms)
        }
        if ($MinKLSameTopP -ge 0.0 -and -not [double]::IsNaN($klSameTopP) -and $klSameTopP -lt $MinKLSameTopP) {
            $gateFailures += ("KL same-top probability {0:P3} is below -MinKLSameTopP {1:P3}" -f $klSameTopP, $MinKLSameTopP)
        }

        if (-not [double]::IsNaN($meanKL) -and $meanKL -gt $MaxMeanKL) {
            $gateFailures += ("Mean KLD {0:N6} exceeds -MaxMeanKL {1:N6}" -f $meanKL, $MaxMeanKL)
        }
        if ($MaxKLD99 -gt 0.0 -and [double]::IsNaN($kld99)) {
            $gateFailures += "KVarN run did not report '99.0% KLD'; see $($kvarnRun.Log)"
        }
        if ($MaxKLD999 -gt 0.0 -and [double]::IsNaN($kld999)) {
            $gateFailures += "KVarN run did not report '99.9% KLD'; see $($kvarnRun.Log)"
        }
        if ($MaxKLDMax -gt 0.0 -and [double]::IsNaN($kldMax)) {
            $gateFailures += "KVarN run did not report 'Maximum KLD'; see $($kvarnRun.Log)"
        }
        if ($MaxKLD99 -gt 0.0 -and -not [double]::IsNaN($kld99) -and $kld99 -gt $MaxKLD99) {
            $gateFailures += ("99.0%% KLD {0:N6} exceeds -MaxKLD99 {1:N6}" -f $kld99, $MaxKLD99)
        }
        if ($MaxKLD999 -gt 0.0 -and -not [double]::IsNaN($kld999) -and $kld999 -gt $MaxKLD999) {
            $gateFailures += ("99.9%% KLD {0:N6} exceeds -MaxKLD999 {1:N6}" -f $kld999, $MaxKLD999)
        }
        if ($MaxKLDMax -gt 0.0 -and -not [double]::IsNaN($kldMax) -and $kldMax -gt $MaxKLDMax) {
            $gateFailures += ("Maximum KLD {0:N6} exceeds -MaxKLDMax {1:N6}" -f $kldMax, $MaxKLDMax)
        }
    }
    $rows += [pscustomobject]@{ Metric = "baseline_reference"; Value = $baselineDescriptor; Threshold = "recorded" }
    $rows += [pscustomobject]@{ Metric = "ppl_baseline_ref";  Value = $pplBase; Threshold = $(if ($enforceBaselinePpl) { $MaxBaselinePpl } else { "" }) }
    $rows += [pscustomobject]@{ Metric = "kl_mean_ppl_base";  Value = $klMeanPplBase; Threshold = "log-diff <= 0.005 vs ppl_baseline_ref" }
    $rows += [pscustomobject]@{ Metric = "kl_ppl_ratio";      Value = $klPplRatio; Threshold = $(if ($MaxKLPplIncrease -ge 0.0) { 1.0 + $MaxKLPplIncrease } else { "" }) }
    $rows += [pscustomobject]@{ Metric = "kl_ppl_increase";   Value = $klPplIncrease; Threshold = $(if ($MaxKLPplIncrease -ge 0.0) { $MaxKLPplIncrease } else { "" }) }
    $rows += [pscustomobject]@{ Metric = "kl_pdiff_rms";      Value = $klPDiffRms; Threshold = $(if ($MaxKLPDiffRms -ge 0.0) { $MaxKLPDiffRms } else { "" }) }
    $rows += [pscustomobject]@{ Metric = "kl_same_top_p";     Value = $klSameTopP; Threshold = $(if ($MinKLSameTopP -ge 0.0) { $MinKLSameTopP } else { "" }) }
    $rows += [pscustomobject]@{ Metric = "mean_kld"; Value = $meanKL; Threshold = $MaxMeanKL }
    $rows += [pscustomobject]@{ Metric = "kld_99p0"; Value = $kld99; Threshold = $(if ($MaxKLD99 -gt 0.0) { $MaxKLD99 } else { "" }) }
    $rows += [pscustomobject]@{ Metric = "kld_99p9"; Value = $kld999; Threshold = $(if ($MaxKLD999 -gt 0.0) { $MaxKLD999 } else { "" }) }
    $rows += [pscustomobject]@{ Metric = "kld_max";  Value = $kldMax; Threshold = $(if ($MaxKLDMax -gt 0.0) { $MaxKLDMax } else { "" }) }
    $primarySummary = ("Mean KLD = {0:N6} (max {1:N6}); 99.0% KLD = {2:N6}; 99.9% KLD = {3:N6}; Maximum KLD = {4:N6}; PPL ratio = {5:N6}; RMS dp = {6:P3}; same-top = {7:P3}" -f `
        $meanKL, $MaxMeanKL, $kld99, $kld999, $kldMax, $klPplRatio, $klPDiffRms, $klSameTopP)
} else {
    # Default mode: compare final perplexity from the same binary.
    $baseRun = Invoke-PplRun -Label $baselineLabel -Exe $pplExe -Argv $commonArgv
    $cudaSummary = $baseRun.CudaDevice

    $pplBase = Get-FinalPpl $baseRun.Text
    $pplKvarn = [double]::NaN
    if ([double]::IsNaN($pplBase)) {
        $gateFailures += "baseline reference run did not report 'Final estimate: PPL'; see $($baseRun.Log)"
    }
    $baselineFailure = Get-BaselinePplSanityFailure $pplBase $MaxBaselinePpl $baseRun.Log
    if (-not [string]::IsNullOrWhiteSpace($baselineFailure)) {
        $gateFailures += $baselineFailure
    }

    if ($gateFailures.Count -eq 0) {
        $kvarnRun = Invoke-PplRun -Label "kvarn" -Exe $pplExe `
            -Argv ($commonArgv + $kvarnCacheArgv) `
            -EnvSet $kvarnTraceEnv
        $ranKvarn = $true
        $cudaSummary = $kvarnRun.CudaDevice
        $kvarnText = $kvarnRun.Text

        $pplKvarn = Get-FinalPpl $kvarnRun.Text
        if ([double]::IsNaN($pplKvarn)) {
            $gateFailures += "KVarN run did not report 'Final estimate: PPL'; see $($kvarnRun.Log)"
        }
    }

    $ratio = [double]::NaN
    $increase = [double]::NaN
    if (-not [double]::IsNaN($pplBase) -and -not [double]::IsNaN($pplKvarn) -and $pplBase -gt 0.0) {
        $ratio = $pplKvarn / $pplBase
        $increase = $ratio - 1.0
        if ($increase -gt $MaxPplIncrease) {
            $gateFailures += ("KVarN PPL {0:N4} is {1:P2} above baseline reference PPL {2:N4} (max {3:P2})" -f `
                $pplKvarn, $increase, $pplBase, $MaxPplIncrease)
        }
    }
    $rows += [pscustomobject]@{ Metric = "baseline_reference"; Value = $baselineDescriptor; Threshold = "recorded" }
    $rows += [pscustomobject]@{ Metric = "ppl_baseline_ref"; Value = $pplBase;  Threshold = $(if ($MaxBaselinePpl -gt 0.0) { $MaxBaselinePpl } else { "" }) }
    $rows += [pscustomobject]@{ Metric = "ppl_kvarn";      Value = $pplKvarn; Threshold = "" }
    $rows += [pscustomobject]@{ Metric = "ppl_increase";   Value = $increase; Threshold = $MaxPplIncrease }
    $primarySummary = ("PPL baseline reference = {0:N4}, PPL KVarN = {1:N4}, increase = {2:P2} (max {3:P2})" -f `
        $pplBase, $pplKvarn, $increase, $MaxPplIncrease)
}

# KVarN cache must actually be engaged, else the gate is meaningless.
$hasKvarnCache = $ranKvarn -and ($kvarnText -match "llama_kv_cache_kvarn:")
if ($ranKvarn -and -not $hasKvarnCache -and -not $AllowKvarnFallback.IsPresent) {
    $gateFailures += "KVarN run produced no 'llama_kv_cache_kvarn:' log (fell back to normal KV). Pass -AllowKvarnFallback only if this is intended."
}
if ($ranKvarn -and $hasKvarnCache -and $expectedLayerIds.Count -gt 0) {
    $observedLayerIds = Get-ObservedKvarnLayerIds $kvarnText
    $missing = @()
    foreach ($id in $expectedLayerIds) {
        if (-not $observedLayerIds.Contains($id)) {
            $missing += $id
        }
    }
    $extra = @()
    foreach ($id in $observedLayerIds) {
        if (-not $expectedLayerSet.Contains($id)) {
            $extra += $id
        }
    }
    if ($missing.Count -gt 0) {
        $gateFailures += "KVarN layer log missed expected layer ids: $($missing -join ',')"
    }
    if ($extra.Count -gt 0) {
        $gateFailures += "KVarN layer log included unexpected layer ids: $($extra -join ',')"
    }
}
if ($ranKvarn -and $hasKvarnCache -and -not [string]::IsNullOrWhiteSpace($ExpectedEffectiveKvarnBits)) {
    $observedBits = Get-KvarnEffectiveBitSet $kvarnText
    if ([string]::IsNullOrWhiteSpace($observedBits)) {
        $gateFailures += "KVarN effective-bit gate requested, but no effective k/v allocation logs were found."
    } else {
        $wantBits = Normalize-KvarnBits $ExpectedEffectiveKvarnBits
        $badBits = @()
        foreach ($pair in ($observedBits -split "," | Where-Object { $_ })) {
            if ((Normalize-KvarnBits $pair) -ne $wantBits) {
                $badBits += $pair
            }
        }
        if ($badBits.Count -gt 0) {
            $gateFailures += "KVarN effective-bit gate observed '$observedBits', expected every routed layer to be '$ExpectedEffectiveKvarnBits'."
        }
    }
}
if ($ranKvarn -and $hasKvarnCache -and $MinKvarnBodyRecords -gt 0) {
    $maxBodyRecords = Get-MaxKvarnBodyRecords $kvarnText
    if ($maxBodyRecords -lt 0) {
        $gateFailures += "KVarN body-record gate requested, but no cache body-record logs were found."
    } elseif ($maxBodyRecords -lt $MinKvarnBodyRecords) {
        $gateFailures += "KVarN cache observed maximum body records $maxBodyRecords, expected at least $MinKvarnBodyRecords."
    }
}
if ($ranKvarn -and $hasKvarnCache -and $MinActiveKvarnBodyRecords -gt 0) {
    $maxActiveBodyRecords = Get-MaxActiveKvarnBodyRecords $kvarnText
    if ($maxActiveBodyRecords -lt 0) {
        $gateFailures += "KVarN active body-record gate requested, but no mixed-attn trace records were found. Enable LLAMA_KVARN_ATTN_TRACE=1 for this gate."
    } elseif ($maxActiveBodyRecords -lt $MinActiveKvarnBodyRecords) {
        $gateFailures += "KVarN mixed attention observed maximum active body records $maxActiveBodyRecords, expected at least $MinActiveKvarnBodyRecords."
    }
}
if ($ranKvarn -and $hasKvarnCache -and $MinFwhtTaken -gt 0) {
    $fwhtTrace = Get-KvarnFwhtTraceSummary $kvarnText
    if ($fwhtTrace.Taken -lt $MinFwhtTaken) {
        $gateFailures += "KVarN CUDA FWHT gate observed taken=$($fwhtTrace.Taken), expected at least $MinFwhtTaken."
    }
}

# Artifacts.
$summaryCsv = Join-Path $OutputDir "summary.csv"
$rows | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation

$status = if ($gateFailures.Count -eq 0) { "PASS" } else { "FAIL" }
$observedEffectiveBitsForSummary = if ($hasKvarnCache) { Get-KvarnEffectiveBitSet $kvarnText } else { "" }
$hasObservedV2 = $observedEffectiveBitsForSummary -match "(^|,)k[0-9]+/v2(,|$)"
$presetStatus = if ((Test-KvarnPresetRequestsV2 $KvarnPreset) -or $hasObservedV2) { "measurement-only; not production readiness" } else { "quality gate" }
$summaryMd = Join-Path $OutputDir "summary.md"
$md = @()
$md += "# KVarN accuracy gate - $status"
$md += ""
$md += "- git: ``$gitHead``"
$md += "- perplexity exe: ``$pplExe``"
$md += "- perplexity exe sha256: ``$(Get-FileSha256 $pplExe)``"
$md += "- model: ``$modelPath``"
$md += "- dataset: ``$dataPath``"
$md += "- mode: $(if ($UseKLDivergence.IsPresent) { 'kl-divergence' } else { 'perplexity-delta' })"
$md += "- baseline reference: ``$baselineDescriptor``"
$md += "- preset: ``$KvarnPreset`` iters=$KvarnIters rtn-quantile=$KvarnRtnQuantile fa=$FlashAttn"
$md += "- preset status: $presetStatus"
$md += "- ctx: $effectiveCtx batch=$effectiveBatch chunks=$(if ($Chunks -gt 0) { $Chunks } else { 'all' })"
$md += "- parse special tokens: $($ParseSpecial.IsPresent)"
$md += "- allow diagnostic env: $($AllowDiagnosticEnv.IsPresent)"
$md += "- diagnostic env count: $($diagnosticEnvValues.Count)"
foreach ($diag in $diagnosticEnvValues) { $md += "- diagnostic env: ``$diag``" }
$md += "- max baseline PPL: $(if ($MaxBaselinePpl -gt 0.0) { $MaxBaselinePpl } else { 'disabled' })"
$md += "- max KL-mode PPL increase: $(if ($UseKLDivergence.IsPresent -and $MaxKLPplIncrease -ge 0.0) { ('{0:P2}' -f $MaxKLPplIncrease) } else { 'disabled' })"
$md += "- max KL RMS target probability delta: $(if ($UseKLDivergence.IsPresent -and $MaxKLPDiffRms -ge 0.0) { ('{0:P3}' -f $MaxKLPDiffRms) } else { 'disabled' })"
$md += "- min KL same-top probability: $(if ($UseKLDivergence.IsPresent -and $MinKLSameTopP -ge 0.0) { ('{0:P3}' -f $MinKLSameTopP) } else { 'disabled' })"
$md += "- env LLAMA_KVARN_ENABLE_PAPER_FRAME: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ENABLE_PAPER_FRAME', 'Process'))"
$md += "- env LLAMA_KVARN_PAPER_MIXED_FRAME: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_PAPER_MIXED_FRAME', 'Process'))"
$md += "- env LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH', 'Process'))"
$md += "- env LLAMA_KVARN_REQUIRE_DIRECT_RECORD_BATCH_PHASES: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_REQUIRE_DIRECT_RECORD_BATCH_PHASES', 'Process'))"
$md += "- env LLAMA_KVARN_DISABLE_DIRECT_RECORD_BATCH_PHASES: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DISABLE_DIRECT_RECORD_BATCH_PHASES', 'Process'))"
$md += "- env LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA', 'Process'))"
$md += "- env LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK', 'Process'))"
$md += "- env LLAMA_KVARN_UNSAFE_ENABLE_ISWA_PREFILL_DIRECT_ATTN: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_UNSAFE_ENABLE_ISWA_PREFILL_DIRECT_ATTN', 'Process'))"
$md += "- env LLAMA_KVARN_PREFILL_DIRECT_TRACE: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_PREFILL_DIRECT_TRACE', 'Process'))"
$md += "- env LLAMA_KVARN_PREFILL_DIRECT_TRACE_LIMIT: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_PREFILL_DIRECT_TRACE_LIMIT', 'Process'))"
$md += "- env LLAMA_KVARN_DISABLE_PREFILL_DIRECT_ATTN: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DISABLE_PREFILL_DIRECT_ATTN', 'Process'))"
$md += "- env LLAMA_KVARN_DISABLE_INTERNAL_CAUSAL_MASK: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DISABLE_INTERNAL_CAUSAL_MASK', 'Process'))"
$md += "- env LLAMA_KVARN_ATTN_FORCE_COMPACT_CAUSAL_MASK: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_FORCE_COMPACT_CAUSAL_MASK', 'Process'))"
$md += "- env LLAMA_KVARN_DISABLE_ISWA_SINKTAIL_MHA: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DISABLE_ISWA_SINKTAIL_MHA', 'Process'))"
$md += "- env LLAMA_KVARN_DISABLE_PREFILL_PINGPONG: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DISABLE_PREFILL_PINGPONG', 'Process'))"
$md += "- env LLAMA_KVARN_GEMMA4_PROTECT_DONORS: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_GEMMA4_PROTECT_DONORS', 'Process'))"
$md += "- env LLAMA_KVARN_LAYER_FILTER: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_LAYER_FILTER', 'Process'))"
$md += "- env LLAMA_KVARN_LAYER_KEY_BITS: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_LAYER_KEY_BITS', 'Process'))"
$md += "- env LLAMA_KVARN_LAYER_VALUE_BITS: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_LAYER_VALUE_BITS', 'Process'))"
$md += "- env LLAMA_KVARN_DISABLE_HIGH_GQA_K8: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DISABLE_HIGH_GQA_K8', 'Process'))"
$md += "- env LLAMA_KVARN_ATTN_TRACE: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_TRACE', 'Process'))"
$md += "- env LLAMA_KVARN_ATTN_TRACE_LIMIT: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_TRACE_LIMIT', 'Process'))"
$md += "- env LLAMA_KVARN_ATTN_DISABLE_256D_SCALAR_QT: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_DISABLE_256D_SCALAR_QT', 'Process'))"
$md += "- env LLAMA_KVARN_ATTN_DISABLE_GQA_SCALAR_QT: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_DISABLE_GQA_SCALAR_QT', 'Process'))"
$md += "- env LLAMA_KVARN_ATTN_WARPQK_FORCE_QT: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_WARPQK_FORCE_QT', 'Process'))"
$md += "- trace attn requested: $($TraceAttn.IsPresent) limit=$TraceLimit"
$md += "- trace FWHT requested: $($TraceFwht.IsPresent) min taken=$MinFwhtTaken"
$md += "- observed FWHT trace: $(if ($hasKvarnCache) { $fwhtTrace = Get-KvarnFwhtTraceSummary $kvarnText; ('taken={0} fallback={1} total={2}' -f $fwhtTrace.Taken, $fwhtTrace.Fallback, $fwhtTrace.Total) } else { '(none)' })"
$md += "- env LLAMA_KVARN_ATTN_REF_SCRATCH: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_REF_SCRATCH', 'Process'))"
$md += "- env LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR', 'Process'))"
$md += "- env LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE', 'Process'))"
$md += "- env LLAMA_KVARN_ATTN_DISABLE_BODY_F32_MIRROR: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_ATTN_DISABLE_BODY_F32_MIRROR', 'Process'))"
$md += "- env LLAMA_KVARN_DEBUG_RAW_BODY_K: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DEBUG_RAW_BODY_K', 'Process'))"
$md += "- env LLAMA_KVARN_DEBUG_RAW_BODY_V: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DEBUG_RAW_BODY_V', 'Process'))"
$md += "- env LLAMA_KVARN_DEBUG_RAW_BODY_SCALAR_QT: $([Environment]::GetEnvironmentVariable('LLAMA_KVARN_DEBUG_RAW_BODY_SCALAR_QT', 'Process'))"
$md += "- expected effective KVarN bits: $(if ([string]::IsNullOrWhiteSpace($ExpectedEffectiveKvarnBits)) { '(not enforced)' } else { $ExpectedEffectiveKvarnBits })"
$md += "- observed effective KVarN bits: $(if ($hasKvarnCache) { if ([string]::IsNullOrWhiteSpace($observedEffectiveBitsForSummary)) { '(unknown)' } else { $observedEffectiveBitsForSummary } } else { '(none)' })"
$md += "- fixture preflight: $(if ($SkipFixtureCheck.IsPresent) { 'skipped' } else { $fixturePreflightLog })"
$md += "- cuda: $cudaSummary"
$md += "- gpu memory snapshot: used,total,free,utilization.gpu -> $(Get-GpuMemorySnapshot)"
$md += "- kvarn run attempted: $ranKvarn"
$md += "- kvarn cache engaged: $hasKvarnCache"
$md += ""
$md += "## Result"
$md += ""
$md += $primarySummary
if ($gateFailures.Count -gt 0) {
    $md += ""
    $md += "## Failures"
    foreach ($f in $gateFailures) { $md += "- $f" }
}
[System.IO.File]::WriteAllText($summaryMd, ($md -join "`n") + "`n")

Write-Host ""
Write-Host "==== KVarN accuracy gate: $status ===="
Write-Host $primarySummary
Write-Host "artifacts: $OutputDir"

if ($gateFailures.Count -gt 0) {
    foreach ($f in $gateFailures) { Write-Host "FAIL: $f" }
    throw "KVarN accuracy gate failed; see $OutputDir"
}
exit 0
