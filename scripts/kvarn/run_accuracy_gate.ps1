<#
.SYNOPSIS
    KVarN end-to-end accuracy gate: prove KVarN attention is numerically
    faithful to the f16 KV cache on the SAME build/model/dataset.

.DESCRIPTION
    Every existing KVarN gate (packed-vs-split NMSE, mainline throughput
    parity) compares two KVarN code paths to each other. None of them
    compares KVarN against the f16 reference model, so a *systematic* KVarN
    numerical error (e.g. a rotation that is applied to the stored body but
    not to Q / sink-tail / output) is invisible to the current suite while
    still degrading generation quality - exactly the "errors accumulate over
    decoding" failure mode the KVarN paper describes.

    This script closes that hole. It runs `llama-perplexity` twice from the
    SAME binary on the SAME model and dataset:
      * baseline: default f16 KV cache
      * kvarn:    --kv-cache-quant kvarn ...
    and fails if KVarN perplexity rises more than -MaxPplIncrease above f16.
    Using one binary isolates the KV-cache backend from build differences.
    Both runs force `--parallel 1`, `--fit off`, and a batch size no larger
    than the context size because the current KVarN backend supports only one
    active sequence and should not be measured through hidden retry or
    multi-chunk context paths.

    With -UseKLDivergence it instead measures the per-token KL divergence of
    the KVarN logit distribution against the f16 base, which is far more
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
    [double] $MaxMeanKL = 0.02,
    [switch] $UseKLDivergence,
    [switch] $AllowKvarnFallback,
    [string] $ExpectedKvarnLayers = "",
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @()
)

$ErrorActionPreference = "Stop"

if ($MaxPplIncrease -lt 0.0) {
    throw "MaxPplIncrease must be non-negative"
}
if ($MaxMeanKL -lt 0.0) {
    throw "MaxMeanKL must be non-negative"
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
    $m = [regex]::Match($Text, "Mean\s+KLD:\s*([0-9]+(?:\.[0-9]+)?)")
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return [double]::NaN
}

function Convert-ToFileStem([string] $Name) {
    return ($Name -replace "[^A-Za-z0-9_.-]", "_")
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
    return $ids
}

function Get-ObservedKvarnLayerIds([string] $text) {
    $actual = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($m in [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer\s+([0-9]+)")) {
        [void] $actual.Add([int] $m.Groups[1].Value)
    }
    return $actual
}

function Invoke-PplRun {
    param(
        [string] $Label,
        [string] $Exe,
        [string[]] $Argv
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

    $oldPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Exe @Argv 2>&1
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPref
    }
    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
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

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$pplExe   = Resolve-BuildExe $BuildDir "llama-perplexity.exe"
$gitHead  = Get-GitHead $repoRoot
$expectedLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "artifacts/kvarn-accuracy/$stamp"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$modelPath = (Resolve-Path -LiteralPath $Model).Path
$dataPath  = (Resolve-Path -LiteralPath $Dataset).Path

$ctxArgv = @()
if ($ContextSize -gt 0) { $ctxArgv = @("-c", [string] $ContextSize) }
$chunkArgv = @()
if ($Chunks -gt 0) { $chunkArgv = @("--chunks", [string] $Chunks) }
$effectiveCtx = if ($ContextSize -gt 0) { $ContextSize } else { 512 }
$effectiveBatch = if ($BatchSize -gt 0) { $BatchSize } else { $effectiveCtx }
if ($effectiveBatch -gt $effectiveCtx) {
    throw "KVarN accuracy gate requires BatchSize <= ContextSize so llama-perplexity keeps n_seq_max = 1"
}

$commonArgv = @(
    "-m", $modelPath,
    "-f", $dataPath,
    "-ngl", [string] $GpuLayers,
    "-np", [string] $Parallel,
    "-b", [string] $effectiveBatch,
    "-fit", $Fit,
    "-fa", $FlashAttn
) + $ctxArgv + $chunkArgv + $ExtraArgs

$rtnQuantileArg = ("{0}" -f $KvarnRtnQuantile)
$kvarnCacheArgv = @(
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", $KvarnPreset,
    "--kvarn-iters", [string] $KvarnIters,
    "--kvarn-rtn-quantile", $rtnQuantileArg
) + $KvarnExtraArgs

$gateFailures = @()
$rows = @()

if ($UseKLDivergence.IsPresent) {
    # Sensitive mode: f16 writes a logit base, KVarN is scored against it.
    $basePath = Join-Path $OutputDir "f16-logits.base.bin"
    $baseRun = Invoke-PplRun -Label "baseline-f16-klbase" -Exe $pplExe `
        -Argv ($commonArgv + @("--kl-divergence-base", $basePath))
    $kvarnRun = Invoke-PplRun -Label "kvarn-kl" -Exe $pplExe `
        -Argv ($commonArgv + $kvarnCacheArgv + @("--kl-divergence-base", $basePath, "--kl-divergence"))

    $meanKL = Get-MeanKL $kvarnRun.Text
    if ([double]::IsNaN($meanKL)) {
        $gateFailures += "KVarN run did not report 'Mean KLD'; see $($kvarnRun.Log)"
    }
    $rows += [pscustomobject]@{ Metric = "mean_kld"; Value = $meanKL; Threshold = $MaxMeanKL }

    if (-not [double]::IsNaN($meanKL) -and $meanKL -gt $MaxMeanKL) {
        $gateFailures += ("Mean KLD {0:N6} exceeds -MaxMeanKL {1:N6}" -f $meanKL, $MaxMeanKL)
    }
    $kvarnText = $kvarnRun.Text
    $primarySummary = ("Mean KLD = {0:N6} (max {1:N6})" -f $meanKL, $MaxMeanKL)
} else {
    # Default mode: compare final perplexity from the same binary.
    $baseRun = Invoke-PplRun -Label "baseline-f16" -Exe $pplExe -Argv $commonArgv
    $kvarnRun = Invoke-PplRun -Label "kvarn" -Exe $pplExe -Argv ($commonArgv + $kvarnCacheArgv)

    $pplBase = Get-FinalPpl $baseRun.Text
    $pplKvarn = Get-FinalPpl $kvarnRun.Text
    if ([double]::IsNaN($pplBase)) {
        $gateFailures += "baseline f16 run did not report 'Final estimate: PPL'; see $($baseRun.Log)"
    }
    if ([double]::IsNaN($pplKvarn)) {
        $gateFailures += "KVarN run did not report 'Final estimate: PPL'; see $($kvarnRun.Log)"
    }

    $ratio = [double]::NaN
    $increase = [double]::NaN
    if (-not [double]::IsNaN($pplBase) -and -not [double]::IsNaN($pplKvarn) -and $pplBase -gt 0.0) {
        $ratio = $pplKvarn / $pplBase
        $increase = $ratio - 1.0
        if ($increase -gt $MaxPplIncrease) {
            $gateFailures += ("KVarN PPL {0:N4} is {1:P2} above f16 PPL {2:N4} (max {3:P2})" -f `
                $pplKvarn, $increase, $pplBase, $MaxPplIncrease)
        }
    }
    $rows += [pscustomobject]@{ Metric = "ppl_f16";        Value = $pplBase;  Threshold = "" }
    $rows += [pscustomobject]@{ Metric = "ppl_kvarn";      Value = $pplKvarn; Threshold = "" }
    $rows += [pscustomobject]@{ Metric = "ppl_increase";   Value = $increase; Threshold = $MaxPplIncrease }
    $kvarnText = $kvarnRun.Text
    $primarySummary = ("PPL f16 = {0:N4}, PPL KVarN = {1:N4}, increase = {2:P2} (max {3:P2})" -f `
        $pplBase, $pplKvarn, $increase, $MaxPplIncrease)
}

# KVarN cache must actually be engaged, else the gate is meaningless.
$hasKvarnCache = $kvarnText -match "llama_kv_cache_kvarn:"
if (-not $hasKvarnCache -and -not $AllowKvarnFallback.IsPresent) {
    $gateFailures += "KVarN run produced no 'llama_kv_cache_kvarn:' log (fell back to normal KV). Pass -AllowKvarnFallback only if this is intended."
}
if ($hasKvarnCache -and $expectedLayerIds.Count -gt 0) {
    $observedLayerIds = Get-ObservedKvarnLayerIds $kvarnText
    $missing = @()
    foreach ($id in $expectedLayerIds) {
        if (-not $observedLayerIds.Contains($id)) {
            $missing += $id
        }
    }
    if ($missing.Count -gt 0) {
        $gateFailures += "KVarN layer log missed expected layer ids: $($missing -join ',')"
    }
}

# Artifacts.
$summaryCsv = Join-Path $OutputDir "summary.csv"
$rows | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation

$status = if ($gateFailures.Count -eq 0) { "PASS" } else { "FAIL" }
$summaryMd = Join-Path $OutputDir "summary.md"
$md = @()
$md += "# KVarN accuracy gate - $status"
$md += ""
$md += "- git: ``$gitHead``"
$md += "- model: ``$modelPath``"
$md += "- dataset: ``$dataPath``"
$md += "- mode: $(if ($UseKLDivergence.IsPresent) { 'kl-divergence' } else { 'perplexity-delta' })"
$md += "- preset: ``$KvarnPreset`` iters=$KvarnIters rtn-quantile=$KvarnRtnQuantile fa=$FlashAttn"
$md += "- ctx: $effectiveCtx batch=$effectiveBatch chunks=$(if ($Chunks -gt 0) { $Chunks } else { 'all' })"
$md += "- cuda: $($kvarnRun.CudaDevice)"
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
    exit 1
}
exit 0
