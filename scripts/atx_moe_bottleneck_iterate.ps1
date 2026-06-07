param(
  [string]$Root = (Resolve-Path "$PSScriptRoot\..").Path,
  [string]$ModelPath = "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
  [string]$PolicySource = "C:\Users\sjake\OneDrive\Documents\New project\results\hf_policies_check",
  [string]$OutDir = "",
  [int]$MaxIterations = 20,
  [double]$MaxHours = 120.0,
  [int]$Context = 64000,
  [int]$MaxTokens = 64,
  [string]$KnownFastLayers = "25-28,31-39",
  [string]$AttentionLayers = "3,7,11,15,19,23,27,31,35,39",
  [int]$SwapCandidates = 6,
  [double]$TargetDecodeTps = 80.0,
  [switch]$NoMtp
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
  $Global:PSNativeCommandUseErrorActionPreference = $false
}

if (-not $OutDir) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $OutDir = Join-Path $Root "runs\atx_moe_bottleneck\iterate_$stamp"
}

$accept = Join-Path $Root "scripts\atx_moe_bottleneck_acceptance.ps1"
$compile = Join-Path $Root "scripts\atx_moe_policy_compile.py"
if (-not (Test-Path $accept)) { throw "Missing acceptance harness at $accept" }
if (-not (Test-Path $compile)) { throw "Missing policy compiler at $compile" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$summaryCsv = Join-Path $OutDir "iteration_results.csv"
$summaryJson = Join-Path $OutDir "iteration_summary.json"
$policyDir = Join-Path $OutDir "compiled_policies"
New-Item -ItemType Directory -Force -Path $policyDir | Out-Null

function Invoke-Acceptance {
  param(
    [string]$CaseOut,
    [string[]]$Extra
  )
  $argsList = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", $accept,
    "-ModelPath", $ModelPath,
    "-PolicySource", $PolicySource,
    "-Context", "$Context",
    "-MaxTokens", "$MaxTokens",
    "-KnownFastLayers", $KnownFastLayers,
    "-AttentionLayers", $AttentionLayers,
    "-OutDir", $CaseOut
  ) + $Extra
  if ($NoMtp) { $argsList += "-NoMtp" }
  & powershell @argsList
  if ($LASTEXITCODE -ne 0) { throw "acceptance failed for $CaseOut with exit code $LASTEXITCODE" }
}

function Read-FirstRow {
  param([string]$CaseOut)
  $path = Join-Path $CaseOut "acceptance_summary.json"
  if (-not (Test-Path $path)) { throw "Missing summary $path" }
  $obj = Get-Content -Raw -Path $path | ConvertFrom-Json
  return $obj.rows[0]
}

function Add-IterationRow {
  param([pscustomobject]$Row)
  $Row | Export-Csv -Path $summaryCsv -Append -NoTypeInformation
}

$deadline = (Get-Date).AddHours($MaxHours)
$rows = @()

$baselineOut = Join-Path $OutDir "baseline_known_fast"
Invoke-Acceptance -CaseOut $baselineOut -Extra @("-SkipCompile", "-Scenarios", "known-fast")
$baseline = Read-FirstRow $baselineOut
$baselineRow = [pscustomobject]@{
  iteration = 0
  candidate = "known_fast_tail_layers"
  policy = Join-Path $baselineOut "policies\known_fast_tail_layers.atx.json"
  decode_tps = $baseline.decode_tps
  prefill_tps = $baseline.prefill_tps
  host_bytes = $baseline.host_bytes
  host_ranges = $baseline.host_ranges
  single_ranges = $baseline.single_ranges
  draft_acceptance_rate = $baseline.draft_acceptance_rate
  peak_gpu_used_mib = $baseline.peak_gpu_used_mib
  min_gpu_free_mib = $baseline.min_gpu_free_mib
  pass_target = [bool]($baseline.decode_tps -ge $TargetDecodeTps)
  out_dir = $baselineOut
}
$rows += $baselineRow
Add-IterationRow $baselineRow

$seedStats = Join-Path $baselineOut "stats\known_fast_tail_layers.residency.json"
& python $compile `
  --policy-source $PolicySource `
  --out-dir $policyDir `
  --mode auto `
  --stats-json $seedStats `
  --max-promote-layers 16 `
  --host-byte-reduction-target 0.80 `
  --attention-baseline-layers $AttentionLayers `
  --base-keep-layers $KnownFastLayers `
  --swap-candidates $SwapCandidates
if ($LASTEXITCODE -ne 0) { throw "policy compiler failed with exit code $LASTEXITCODE" }

$candidateFiles = @()
$preferredNames = @(
  "bottleneck_base_preserve_13_layers.atx.json",
  "bottleneck_top_13_layers.atx.json",
  "bottleneck_top_12_layers.atx.json",
  "bottleneck_top_10_layers.atx.json"
)
foreach ($name in $preferredNames) {
  $p = Join-Path $policyDir $name
  if (Test-Path $p) { $candidateFiles += Get-Item $p }
}
$candidateFiles += Get-ChildItem -Path $policyDir -Filter "bottleneck_swap_in_*.atx.json" | Sort-Object Name
$candidateFiles = $candidateFiles | Select-Object -Unique

$iteration = 0
foreach ($policy in $candidateFiles) {
  if ($iteration -ge $MaxIterations) { break }
  if ((Get-Date) -ge $deadline) { break }
  $iteration++
  $safeName = ($policy.BaseName -replace '[^A-Za-z0-9_.-]', '_')
  $caseOut = Join-Path $OutDir ("iter_{0:D2}_{1}" -f $iteration, $safeName)
  Invoke-Acceptance -CaseOut $caseOut -Extra @(
    "-SkipCompile",
    "-Scenarios", "policy",
    "-PolicyPath", $policy.FullName,
    "-PolicyScenarioName", $safeName
  )
  $r = Read-FirstRow $caseOut
  $row = [pscustomobject]@{
    iteration = $iteration
    candidate = $safeName
    policy = $policy.FullName
    decode_tps = $r.decode_tps
    prefill_tps = $r.prefill_tps
    host_bytes = $r.host_bytes
    host_ranges = $r.host_ranges
    single_ranges = $r.single_ranges
    draft_acceptance_rate = $r.draft_acceptance_rate
    peak_gpu_used_mib = $r.peak_gpu_used_mib
    min_gpu_free_mib = $r.min_gpu_free_mib
    pass_target = [bool]($r.decode_tps -ge $TargetDecodeTps)
    out_dir = $caseOut
  }
  $rows += $row
  Add-IterationRow $row
  if ($row.pass_target) { break }
}

$best = $rows | Where-Object { $null -ne $_.decode_tps } | Sort-Object decode_tps -Descending | Select-Object -First 1
$summary = [pscustomobject]@{
  out_dir = $OutDir
  max_iterations = $MaxIterations
  max_hours = $MaxHours
  target_decode_tps = $TargetDecodeTps
  best = $best
  rows = $rows
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryJson -Encoding UTF8
$summary | ConvertTo-Json -Depth 5
