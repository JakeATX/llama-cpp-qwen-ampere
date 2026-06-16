param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $Dataset,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $OutputDir = "",
    [string[]] $Filters = @(),
    [string] $KvarnPreset = "kvarn_k8v8_g128",
    [int] $KvarnIters = 16,
    [double] $KvarnRtnQuantile = 1.0,
    [int] $GpuLayers = 999,
    [int] $ContextSize = 4096,
    [int] $BatchSize = 4096,
    [int] $Chunks = 2,
    [string] $FlashAttn = "off",
    [string] $Fit = "off",
    [switch] $PaperFrame,
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @()
)

$ErrorActionPreference = "Stop"

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

function Convert-ToFileStem([string] $s) {
    return ($s -replace '[^A-Za-z0-9_.-]+', '_').Trim('_')
}

function Get-FinalPpl([string] $text) {
    $matches = [regex]::Matches($text, "Final estimate: PPL =\s*([0-9]+(?:\.[0-9]+)?)")
    if ($matches.Count -eq 0) {
        return [double]::NaN
    }
    return [double] $matches[$matches.Count - 1].Groups[1].Value
}

function Get-ExpectedKvarnLayerIds([string] $spec) {
    $ids = @()
    foreach ($part in ($spec -split ',')) {
        $raw = $part.Trim()
        if ([string]::IsNullOrWhiteSpace($raw)) {
            continue
        }
        if ($raw -match '^([0-9]+)-([0-9]+)(?::([0-9]+))?$') {
            $start = [int] $Matches[1]
            $end = [int] $Matches[2]
            $step = if ($Matches.ContainsKey(3) -and -not [string]::IsNullOrEmpty($Matches[3])) { [int] $Matches[3] } else { 1 }
            if ($end -lt $start -or $step -le 0) {
                throw "Invalid KVarN layer range '$raw'"
            }
            for ($id = $start; $id -le $end; $id += $step) {
                $ids += $id
            }
        } else {
            $id = 0
            if (-not [int]::TryParse($raw, [ref] $id) -or $id -lt 0) {
                throw "Invalid KVarN layer id '$raw'"
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

function Test-ExactLayers([string] $expectedSpec, [string] $text) {
    $expected = @(Get-ExpectedKvarnLayerIds $expectedSpec)
    $expectedSet = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($id in $expected) {
        [void] $expectedSet.Add([int] $id)
    }

    $observed = Get-ObservedKvarnLayerIds $text
    $missing = @()
    foreach ($id in $expected) {
        if (-not $observed.Contains($id)) {
            $missing += $id
        }
    }

    $extra = @()
    foreach ($id in $observed) {
        if (-not $expectedSet.Contains($id)) {
            $extra += $id
        }
    }

    if ($missing.Count -gt 0 -or $extra.Count -gt 0) {
        throw "Layer mismatch for filter '$expectedSpec': missing=[$($missing -join ',')] extra=[$($extra -join ',')]"
    }
}

function Invoke-PplRun {
    param(
        [string] $Label,
        [string[]] $Argv,
        [hashtable] $Env = @{}
    )

    $stem = Convert-ToFileStem $Label
    $logPath = Join-Path $OutputDir "$stem.log.txt"
    $cmdPath = Join-Path $OutputDir "$stem.command.txt"
    $commandLine = "`"$pplExe`" " + (($Argv | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join " ")
    [System.IO.File]::WriteAllText($cmdPath, $commandLine + "`n")

    $oldValues = @{}
    foreach ($name in $Env.Keys) {
        $oldValues[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        [Environment]::SetEnvironmentVariable($name, [string] $Env[$name], "Process")
    }

    Write-Host "== $Label"
    Write-Host $commandLine
    $oldPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $pplExe @Argv 2>&1
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPref
        foreach ($name in $Env.Keys) {
            [Environment]::SetEnvironmentVariable($name, $oldValues[$name], "Process")
        }
    }
    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    [System.IO.File]::WriteAllText($logPath, $text + "`n")
    if ($exit -ne 0) {
        throw "$Label llama-perplexity failed with exit code $exit; see $logPath"
    }
    return [pscustomobject]@{
        Text = $text
        Log = (Split-Path -Leaf $logPath)
        Ppl = Get-FinalPpl $text
    }
}

if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found: $Model"
}
if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "Dataset not found: $Dataset"
}
if ($Filters.Count -eq 0) {
    throw "At least one -Filters entry is required"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$pplExe = Resolve-BuildExe $BuildDir "llama-perplexity.exe"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot ("artifacts/kvarn-layer-bisect/" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$commonArgv = @(
    "-m", (Resolve-Path -LiteralPath $Model).Path,
    "-f", (Resolve-Path -LiteralPath $Dataset).Path,
    "-ngl", [string] $GpuLayers,
    "-np", "1",
    "-b", [string] $BatchSize,
    "-fit", $Fit,
    "-fa", $FlashAttn,
    "-c", [string] $ContextSize
) + $ExtraArgs
if ($Chunks -gt 0) {
    $commonArgv += @("--chunks", [string] $Chunks)
}

$baseline = Invoke-PplRun -Label "baseline-f16" -Argv $commonArgv
if ([double]::IsNaN($baseline.Ppl)) {
    throw "Baseline did not report final PPL; see $($baseline.Log)"
}

$rows = @()
$failures = @()
foreach ($filter in $Filters) {
    $env = @{ "LLAMA_KVARN_LAYER_FILTER" = $filter }
    if ($PaperFrame.IsPresent) {
        $env["LLAMA_KVARN_ENABLE_PAPER_FRAME"] = "1"
    }
    $argv = $commonArgv + @(
        "--kv-cache-quant", "kvarn",
        "--kvarn-preset", $KvarnPreset,
        "--kvarn-iters", [string] $KvarnIters,
        "--kvarn-rtn-quantile", ("{0}" -f $KvarnRtnQuantile)
    ) + $KvarnExtraArgs
    $run = Invoke-PplRun -Label ("kvarn-filter-" + $filter) -Argv $argv -Env $env
    try {
        Test-ExactLayers -expectedSpec $filter -text $run.Text
    } catch {
        $failures += $_.Exception.Message
    }
    $route = [regex]::Match($run.Text, "diagnostic hybrid KVarN route map: .*")
    $increase = if (-not [double]::IsNaN($run.Ppl) -and $baseline.Ppl -gt 0.0) { ($run.Ppl / $baseline.Ppl) - 1.0 } else { [double]::NaN }
    $rows += [pscustomobject]@{
        Filter = $filter
        PplF16 = $baseline.Ppl
        PplKvarn = $run.Ppl
        PplIncrease = $increase
        Log = $run.Log
        Route = if ($route.Success) { $route.Value } else { "" }
    }
}

$summaryCsv = Join-Path $OutputDir "summary.csv"
$rows | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation

$summaryMd = Join-Path $OutputDir "summary.md"
$md = @("# KVarN Layer-Filter Bisection", "")
$md += "- model: ``$Model``"
$md += "- dataset: ``$Dataset``"
$md += "- preset: ``$KvarnPreset`` iters=$KvarnIters paper_frame=$($PaperFrame.IsPresent)"
$md += "- ctx: $ContextSize batch=$BatchSize chunks=$Chunks"
$md += "- common extra args: ``$($ExtraArgs -join ' ')``"
$md += "- KVarN extra args: ``$($KvarnExtraArgs -join ' ')``"
$md += ""
$md += "| filter | f16 PPL | KVarN PPL | increase | log |"
$md += "|---|---:|---:|---:|---|"
foreach ($r in $rows) {
    $md += ("| `{0}` | {1:N4} | {2:N4} | {3:P2} | `{4}` |" -f $r.Filter, $r.PplF16, $r.PplKvarn, $r.PplIncrease, $r.Log)
}
if ($failures.Count -gt 0) {
    $md += ""
    $md += "## Failures"
    foreach ($f in $failures) {
        $md += "- $f"
    }
}
[System.IO.File]::WriteAllText($summaryMd, ($md -join "`n") + "`n")

Write-Host ""
Write-Host "Layer-filter bisection summary: $summaryCsv"
foreach ($r in $rows) {
    Write-Host ("filter={0} f16={1:N4} kvarn={2:N4} increase={3:P2}" -f $r.Filter, $r.PplF16, $r.PplKvarn, $r.PplIncrease)
}
if ($failures.Count -gt 0) {
    foreach ($f in $failures) {
        Write-Host "FAIL: $f"
    }
    throw "Layer-filter bisection failed; see $OutputDir"
}
