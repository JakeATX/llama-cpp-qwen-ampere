<#
.SYNOPSIS
    Sweep --kvarn-iters and report the smallest value that still passes the
    KVarN accuracy gate, so production can run the fewest Sinkhorn iterations
    that hold f16-level accuracy.

.DESCRIPTION
    The KVarN body store runs `--kvarn-iters` Sinkhorn variance-normalization
    passes per (record, head) tile, and each pass is two CUDA kernel launches.
    At long context (e.g. 4096 with ~30 records/layer) the store is the
    dominant prefill cost, and iterations multiply it linearly. Sinkhorn
    converges quickly, so iters=2 or 3 frequently matches iters=4 accuracy
    while cutting store launches 25-50%.

    This driver runs scripts/kvarn/run_accuracy_gate.ps1 once per candidate
    iters value (smallest first), records pass/fail and the accuracy metric
    (PPL increase, or mean KLD with -UseKLDivergence), and recommends the
    SMALLEST iters that passes. By default it stops at the first passing
    candidate to avoid wasting production-model runtime; pass
    -ContinueAfterPass to collect the full curve. It changes no model
    behaviour itself; it only schedules gate runs and aggregates results. Set production
    `--kvarn-iters` to the recommended value, then re-run the throughput
    parity matrix to confirm the prefill speedup.

.NOTES
    Additive companion to run_accuracy_gate.ps1; all gate semantics
    (single-sequence enforcement, cache-engagement check, KL mode) are
    inherited by delegation.
#>
param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $Dataset,
    [int[]]  $IterValues = @(1, 2, 3, 4),
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $OutputDir = "",
    [string] $KvarnPreset = "kvarn_k4v2_g128",
    [double] $KvarnRtnQuantile = 1.0,
    [string] $FlashAttn = "off",
    [string] $Fit = "off",
    [int]    $GpuLayers = 999,
    [int]    $ContextSize = 0,
    [int]    $BatchSize = 0,
    [int]    $Chunks = 0,
    [double] $MaxPplIncrease = 0.05,
    [double] $MaxMeanKL = 0.02,
    [switch] $UseKLDivergence,
    [switch] $AllowKvarnFallback,
    [switch] $ContinueAfterPass,
    [string] $ExpectedKvarnLayers = "",
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @()
)

$ErrorActionPreference = "Stop"

if ($IterValues.Count -eq 0) {
    throw "IterValues must contain at least one candidate"
}
foreach ($iv in $IterValues) {
    if ($iv -le 0) { throw "IterValues must all be positive (got $iv)" }
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found at $Model"
}
if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "Dataset not found at $Dataset"
}

$gateScript = Join-Path $PSScriptRoot "run_accuracy_gate.ps1"
if (-not (Test-Path -LiteralPath $gateScript)) {
    throw "run_accuracy_gate.ps1 not found next to this script at $gateScript"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "artifacts/kvarn-iters-sweep/$stamp"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# The metric the gate writes to summary.csv depends on its mode.
$metricName = if ($UseKLDivergence.IsPresent) { "mean_kld" } else { "ppl_increase" }
$metricCap  = if ($UseKLDivergence.IsPresent) { $MaxMeanKL } else { $MaxPplIncrease }

function Get-GateMetric([string] $RunDir, [string] $Metric) {
    $csv = Join-Path $RunDir "summary.csv"
    if (-not (Test-Path -LiteralPath $csv)) { return [double]::NaN }
    $row = Import-Csv -LiteralPath $csv | Where-Object { $_.Metric -eq $Metric } | Select-Object -First 1
    if ($null -eq $row -or [string]::IsNullOrWhiteSpace($row.Value)) { return [double]::NaN }
    $val = 0.0
    if ([double]::TryParse([string] $row.Value, [ref] $val)) { return $val }
    return [double]::NaN
}

function Get-GateStatus([string] $RunDir) {
    $summary = Join-Path $RunDir "summary.md"
    if (-not (Test-Path -LiteralPath $summary)) { return "UNKNOWN" }
    $first = Get-Content -LiteralPath $summary -TotalCount 1
    if ($first -match '\bPASS\b') { return "PASS" }
    if ($first -match '\bFAIL\b') { return "FAIL" }
    return "UNKNOWN"
}

function Convert-ToPowerShellLiteral([string] $Value) {
    return "'" + ($Value -replace "'", "''") + "'"
}

function Convert-ToPowerShellArrayLiteral([string[]] $Values) {
    if ($null -eq $Values -or $Values.Count -eq 0) {
        return "@()"
    }
    return "@(" + (($Values | ForEach-Object { Convert-ToPowerShellLiteral $_ }) -join ", ") + ")"
}

function Invoke-GateChildProcess {
    param(
        [string] $Script,
        [hashtable] $Params,
        [string] $RunDir,
        [bool] $UseKL,
        [bool] $AllowFallback
    )

    $cmd = @()
    $cmd += "`$ErrorActionPreference = 'Stop'"
    $cmd += "`$extraArgs = $(Convert-ToPowerShellArrayLiteral ([string[]] $Params.ExtraArgs))"
    $cmd += "`$kvarnExtraArgs = $(Convert-ToPowerShellArrayLiteral ([string[]] $Params.KvarnExtraArgs))"
    $line = @(
        "&", (Convert-ToPowerShellLiteral $Script),
        "-Model", (Convert-ToPowerShellLiteral $Params.Model),
        "-Dataset", (Convert-ToPowerShellLiteral $Params.Dataset),
        "-BuildDir", (Convert-ToPowerShellLiteral $Params.BuildDir),
        "-OutputDir", (Convert-ToPowerShellLiteral $Params.OutputDir),
        "-KvarnPreset", (Convert-ToPowerShellLiteral $Params.KvarnPreset),
        "-KvarnIters", ([string] $Params.KvarnIters),
        "-KvarnRtnQuantile", ([string] $Params.KvarnRtnQuantile),
        "-FlashAttn", (Convert-ToPowerShellLiteral $Params.FlashAttn),
        "-Fit", (Convert-ToPowerShellLiteral $Params.Fit),
        "-GpuLayers", ([string] $Params.GpuLayers),
        "-ContextSize", ([string] $Params.ContextSize),
        "-BatchSize", ([string] $Params.BatchSize),
        "-Chunks", ([string] $Params.Chunks),
        "-MaxPplIncrease", ([string] $Params.MaxPplIncrease),
        "-MaxMeanKL", ([string] $Params.MaxMeanKL),
        "-ExpectedKvarnLayers", (Convert-ToPowerShellLiteral $Params.ExpectedKvarnLayers),
        "-ExtraArgs", "`$extraArgs",
        "-KvarnExtraArgs", "`$kvarnExtraArgs"
    )
    if ($UseKL) {
        $line += "-UseKLDivergence"
    }
    if ($AllowFallback) {
        $line += "-AllowKvarnFallback"
    }
    $cmd += ($line -join " ")

    $runnerPath = Join-Path $RunDir "accuracy_gate_runner.ps1"
    $stdoutPath = Join-Path $RunDir "accuracy_gate_stdout.log.txt"
    [System.IO.File]::WriteAllText($runnerPath, ($cmd -join "`n") + "`n")

    $oldPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & powershell -NoProfile -ExecutionPolicy Bypass -File $runnerPath 2>&1
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPref
    }
    [System.IO.File]::WriteAllText($stdoutPath, (($output | ForEach-Object { $_.ToString() }) -join "`n") + "`n")
    return $exit
}

# Smallest candidate first so the recommendation is the cheapest passing value.
$sorted = $IterValues | Sort-Object -Unique
$results = @()

foreach ($iters in $sorted) {
    $runDir = Join-Path $OutputDir ("iters-{0}" -f $iters)
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $gateParams = @{
        Model            = $Model
        Dataset          = $Dataset
        BuildDir         = $BuildDir
        OutputDir        = $runDir
        KvarnPreset      = $KvarnPreset
        KvarnIters       = $iters
        KvarnRtnQuantile = $KvarnRtnQuantile
        FlashAttn        = $FlashAttn
        Fit              = $Fit
        GpuLayers        = $GpuLayers
        ContextSize      = $ContextSize
        BatchSize        = $BatchSize
        Chunks           = $Chunks
        MaxPplIncrease   = $MaxPplIncrease
        MaxMeanKL        = $MaxMeanKL
        ExpectedKvarnLayers = $ExpectedKvarnLayers
        ExtraArgs        = $ExtraArgs
        KvarnExtraArgs   = $KvarnExtraArgs
    }
    if ($UseKLDivergence.IsPresent)  { $gateParams["UseKLDivergence"]  = $true }
    if ($AllowKvarnFallback.IsPresent) { $gateParams["AllowKvarnFallback"] = $true }

    Write-Host ""
    Write-Host "================ iters=$iters ================"

    $exit = Invoke-GateChildProcess `
        -Script $gateScript `
        -Params $gateParams `
        -RunDir $runDir `
        -UseKL $UseKLDivergence.IsPresent `
        -AllowFallback $AllowKvarnFallback.IsPresent

    $metric = Get-GateMetric $runDir $metricName
    $gateStatus = Get-GateStatus $runDir
    $metricPass = (-not [double]::IsNaN($metric)) -and ($metric -le $metricCap)
    $passed = ($gateStatus -eq "PASS") -and $metricPass
    $results += [pscustomobject]@{
        Iters     = $iters
        Status    = if ($passed) { "PASS" } else { "FAIL" }
        Metric    = $metricName
        Value     = $metric
        Threshold = $metricCap
        GateStatus = $gateStatus
        ExitCode   = $exit
        RunDir    = (Split-Path -Leaf $runDir)
    }

    if ($passed -and -not $ContinueAfterPass.IsPresent) {
        Write-Host "iters=$iters passed; stopping early. Use -ContinueAfterPass to sweep remaining candidates."
        break
    }
}

$passing = @($results | Where-Object { $_.Status -eq "PASS" } | Sort-Object Iters)
$recommended = if ($passing.Count -gt 0) { ($passing | Select-Object -First 1).Iters } else { $null }

# Artifacts.
$results | Export-Csv -LiteralPath (Join-Path $OutputDir "sweep_summary.csv") -NoTypeInformation

$md = @()
$md += "# KVarN --kvarn-iters sweep"
$md += ""
$md += "- model: ``$Model``"
$md += "- dataset: ``$Dataset``"
$md += "- metric: $metricName (max $metricCap)"
$md += "- candidates: $($sorted -join ', ')"
$md += "- chunks: $(if ($Chunks -gt 0) { $Chunks } else { 'all' })"
$md += "- stop after first pass: $(-not $ContinueAfterPass.IsPresent)"
$md += ""
$md += "| iters | status | $metricName | threshold |"
$md += "|---:|:--|---:|---:|"
foreach ($r in $results) {
    $v = if ([double]::IsNaN([double] $r.Value)) { "n/a" } else { ("{0:N6}" -f [double] $r.Value) }
    $md += ("| {0} | {1} | {2} | {3:N6} |" -f $r.Iters, $r.Status, $v, [double] $r.Threshold)
}
$md += ""
if ($null -ne $recommended) {
    $md += "## Recommendation"
    $md += ""
    $md += "Set production ``--kvarn-iters $recommended`` (smallest value that holds the accuracy gate), then re-run the throughput parity matrix to confirm the prefill store speedup."
} else {
    $md += "## Recommendation"
    $md += ""
    $md += "No candidate passed the accuracy gate. Do not reduce iters; investigate accuracy (start with the rotation handling) before tuning store cost."
}
[System.IO.File]::WriteAllText((Join-Path $OutputDir "sweep_summary.md"), ($md -join "`n") + "`n")

Write-Host ""
Write-Host "==== KVarN iters sweep ===="
$results | Format-Table -AutoSize | Out-String | Write-Host
if ($null -ne $recommended) {
    Write-Host "RECOMMENDED --kvarn-iters $recommended (smallest passing)"
    Write-Host "artifacts: $OutputDir"
    exit 0
} else {
    Write-Host "NO PASSING iters value; see $OutputDir"
    exit 1
}
