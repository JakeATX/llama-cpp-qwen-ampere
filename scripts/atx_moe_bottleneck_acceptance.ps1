param(
  [string]$Root = (Resolve-Path "$PSScriptRoot\..").Path,
  [string]$ModelPath = "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
  [string]$PolicySource = "C:\Users\sjake\OneDrive\Documents\New project\results\hf_policies_check",
  [string]$OutDir = "",
  [int]$Context = 64000,
  [int]$MaxTokens = 512,
  [int]$Threads = 8,
  [int]$BatchSize = 2048,
  [int]$UBatchSize = 512,
  [int]$NCpuMoe = 34,
  [string]$AttentionLayers = "3,7,11,15,19,23,27,31,35,39",
  [string]$KnownFastLayers = "25-28,31-39",
  [string[]]$Scenarios = @("reference", "attention", "known-fast", "bottleneck-auto"),
  [string]$BottleneckPolicyPattern = "bottleneck_base_preserve_13_layers.atx.json",
  [string]$PolicyPath = "",
  [string]$PolicyScenarioName = "policy_candidate",
  [int]$SwapCandidates = 3,
  [int]$MaxPromoteLayers = 16,
  [double]$TargetDecodeTps = 80.0,
  [double]$ReferenceDecodeTps = 72.0,
  [switch]$SkipCompile,
  [switch]$NoMtp
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
  $Global:PSNativeCommandUseErrorActionPreference = $false
}

if (-not $OutDir) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $OutDir = Join-Path $Root "runs\atx_moe_bottleneck\server_acceptance_$stamp"
}

$bin = Join-Path $Root "build-atx-cuda\bin\llama-server.exe"
if (-not (Test-Path $bin)) { throw "Missing llama-server.exe at $bin" }
if (-not (Test-Path $ModelPath)) { throw "Missing model at $ModelPath" }

$rawDir = Join-Path $OutDir "raw"
$statsDir = Join-Path $OutDir "stats"
$policyDir = Join-Path $OutDir "policies"
New-Item -ItemType Directory -Force -Path $rawDir, $statsDir, $policyDir | Out-Null

$summaryCsv = Join-Path $OutDir "acceptance_table.csv"
$summaryJson = Join-Path $OutDir "acceptance_summary.json"

function Get-FreePort {
  $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
  $listener.Start()
  $port = $listener.LocalEndpoint.Port
  $listener.Stop()
  return $port
}

function Get-SystemSample {
  $gpuLine = (& nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits) -join ""
  $parts = $gpuLine -split ",\s*"
  [pscustomobject]@{
    gpu_used_mib = [int]$parts[0]
    gpu_free_mib = [int]$parts[1]
    gpu_temp_c = [int]$parts[2]
    gpu_util_pct = [int]$parts[3]
  }
}

function Get-Peak {
  param([object[]]$Samples)
  [pscustomobject]@{
    peak_gpu_used_mib = if ($Samples.Count) { ($Samples | Measure-Object gpu_used_mib -Maximum).Maximum } else { $null }
    min_gpu_free_mib = if ($Samples.Count) { ($Samples | Measure-Object gpu_free_mib -Minimum).Minimum } else { $null }
    peak_gpu_util_pct = if ($Samples.Count) { ($Samples | Measure-Object gpu_util_pct -Maximum).Maximum } else { $null }
    peak_gpu_temp_c = if ($Samples.Count) { ($Samples | Measure-Object gpu_temp_c -Maximum).Maximum } else { $null }
  }
}

function Get-CodingPrompt {
  $promptPath = Join-Path $Root "prompts\atx_moe_coding_prompts.txt"
  if (Test-Path $promptPath) {
    $text = Get-Content -Raw -Path $promptPath
    $parts = $text -split "(\r?\n){2,}"
    foreach ($part in $parts) {
      if ($part.Trim().Length -gt 200) { return $part.Trim() }
    }
  }
  return @"
You are editing a Python service. Implement a small LRU cache class with get, put, delete, clear, and stats methods. Include edge-case handling for capacity 0, overwrites, and missing keys. Return only a unified diff.
"@
}

function Join-CommandArgs {
  param([string[]]$ArgsList)
  $quoted = foreach ($arg in $ArgsList) {
    if ($null -eq $arg) { continue }
    $s = [string]$arg
    if ($s -match '[\s"]') {
      '"' + ($s -replace '"', '\"') + '"'
    } else {
      $s
    }
  }
  return ($quoted -join " ")
}

function Read-StatsSummary {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return [pscustomobject]@{
      host_bytes = $null; host_ranges = $null; single_ranges = $null; direct_hits = $null
      hot_stage_violations = $null; fallback_dispatches = $null
    }
  }
  $obj = Get-Content -Raw -Path $Path | ConvertFrom-Json
  $c = $obj.counters
  $hostRanges = $c.host_expert_range_copy_calls
  if ($null -eq $hostRanges) { $hostRanges = $c.host_expert_copy_calls }
  [pscustomobject]@{
    host_bytes = [int64]($c.host_bytes_copied)
    host_ranges = [int64]$hostRanges
    single_ranges = [int64]($c.host_expert_single_copy_calls)
    direct_hits = [int64]($c.direct_resident_expert_hits)
    hot_stage_violations = [int64]($c.hot_stage_violations)
    fallback_dispatches = [int64]($c.direct_fallback_dispatches)
  }
}

function Read-SpecSummary {
  param([string]$Path)
  $text = if (Test-Path $Path) { Get-Content -Raw -Path $Path } else { "" }
  $generated = $null
  $accepted = $null
  $rate = $null
  if ($text -match "draft acceptance rate\s*=\s*([0-9.]+)\s*\(\s*([0-9]+)\s+accepted\s*/\s*([0-9]+)\s+generated") {
    $rate = [double]$Matches[1]
    $accepted = [int]$Matches[2]
    $generated = [int]$Matches[3]
  } elseif ($text -match "#gen tokens\s*=\s*([0-9]+),\s*#acc tokens\s*=\s*([0-9]+)") {
    $generated = [int]$Matches[1]
    $accepted = [int]$Matches[2]
    if ($generated -gt 0) { $rate = [math]::Round($accepted / $generated, 5) }
  }
  [pscustomobject]@{
    draft_generated = $generated
    draft_accepted = $accepted
    draft_acceptance_rate = $rate
  }
}

function Invoke-Scenario {
  param(
    [string]$Name,
    [string[]]$ExtraArgs,
    [bool]$UseNCpuMoe = $false
  )
  $port = Get-FreePort
  $stdout = Join-Path $rawDir "$Name.stdout.log"
  $stderr = Join-Path $rawDir "$Name.stderr.log"
  $statsPath = Join-Path $statsDir "$Name.residency.json"
  $serverArgs = @(
    "-m", $ModelPath,
    "--alias", "atx-bottleneck-$Name",
    "--host", "127.0.0.1",
    "--port", "$port",
    "-c", "$Context",
    "-t", "$Threads",
    "-tb", "$Threads",
    "-ngl", "999",
    "-fa", "on",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "-b", "$BatchSize",
    "-ub", "$UBatchSize",
    "--parallel", "1",
    "--cache-ram", "0",
    "--reasoning", "on",
    "--reasoning-format", "deepseek",
    "--temp", "0",
    "--top-p", "1",
    "--no-webui",
    "--moe-residency-stats", $statsPath
  )
  if (-not $NoMtp) {
    $serverArgs += @("--spec-type", "mtp", "--spec-draft-n-max", "2")
  }
  if ($UseNCpuMoe) {
    $serverArgs += @("-ncmoe", "$NCpuMoe")
  }
  $serverArgs += $ExtraArgs

  Remove-Item -Force -ErrorAction SilentlyContinue $stdout, $stderr, $statsPath
  $samples = @()
  $p = Start-Process -FilePath $bin -ArgumentList (Join-CommandArgs $serverArgs) -WorkingDirectory $Root -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  $ready = $false
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $errorText = ""
  $resp = $null
  try {
    while (-not $p.HasExited -and $sw.Elapsed.TotalMinutes -lt 12) {
      try {
        Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 | Out-Null
        $ready = $true
        break
      } catch {
        try { $samples += Get-SystemSample } catch {}
        Start-Sleep -Seconds 1
      }
    }
    if (-not $ready) { throw "server did not become ready" }
    $body = @{
      prompt = Get-CodingPrompt
      n_predict = $MaxTokens
      temperature = 0.0
      cache_prompt = $false
      stop = @()
    }
    $json = $body | ConvertTo-Json -Depth 8
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/completion" -Body $bytes -ContentType "application/json; charset=utf-8" -TimeoutSec 900
  } catch {
    $errorText = $_.Exception.Message
  } finally {
    if (-not $p.HasExited) {
      try {
        Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/shutdown" -TimeoutSec 5 | Out-Null
      } catch {}
    }
    try { $p.WaitForExit(15000) | Out-Null } catch {}
    if (-not $p.HasExited) {
      try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
      try { $p.WaitForExit(5000) | Out-Null } catch {}
    }
    try { $samples += Get-SystemSample } catch {}
    $sw.Stop()
  }

  $timings = if ($resp -and $resp.timings) { $resp.timings } else { $null }
  $decode = $null
  $prefill = $null
  $predicted = $null
  if ($timings) {
    if ($timings.PSObject.Properties.Name -contains "predicted_per_second") { $decode = [double]$timings.predicted_per_second }
    elseif ($timings.PSObject.Properties.Name -contains "predicted_n_per_second") { $decode = [double]$timings.predicted_n_per_second }
    if ($timings.PSObject.Properties.Name -contains "prompt_per_second") { $prefill = [double]$timings.prompt_per_second }
    elseif ($timings.PSObject.Properties.Name -contains "prompt_n_per_second") { $prefill = [double]$timings.prompt_n_per_second }
    if ($timings.PSObject.Properties.Name -contains "predicted_n") { $predicted = [int]$timings.predicted_n }
  }
  $stats = Read-StatsSummary $statsPath
  $spec = Read-SpecSummary $stderr
  $peak = Get-Peak $samples
  $row = [pscustomobject]@{
    scenario = $Name
    decode_tps = $decode
    prefill_tps = $prefill
    predicted_tokens = $predicted
    elapsed_sec = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    host_bytes = $stats.host_bytes
    host_ranges = $stats.host_ranges
    single_ranges = $stats.single_ranges
    direct_hits = $stats.direct_hits
    hot_stage_violations = $stats.hot_stage_violations
    fallback_dispatches = $stats.fallback_dispatches
    draft_generated = $spec.draft_generated
    draft_accepted = $spec.draft_accepted
    draft_acceptance_rate = $spec.draft_acceptance_rate
    peak_gpu_used_mib = $peak.peak_gpu_used_mib
    min_gpu_free_mib = $peak.min_gpu_free_mib
    error = $errorText
    stats = $statsPath
    stdout = $stdout
    stderr = $stderr
    args = ($serverArgs -join " ")
  }
  $row | Export-Csv -Path $summaryCsv -NoTypeInformation -Append
  return $row
}

$rows = @()
$attentionPolicy = Join-Path $policyDir "attention_layer_baseline.atx.json"
$knownFastPolicy = Join-Path $policyDir "known_fast_tail_layers.atx.json"
@{
  schema_version = "atx-moe-residency-policy-v1"
  policy_name = "attention_layer_baseline"
  keep_layers = @($AttentionLayers -split "," | ForEach-Object {
    if ($_ -match "-") { $a,$b = $_ -split "-"; $a..$b } else { [int]$_ }
  })
  basis = "Known attention-spaced layer baseline."
} | ConvertTo-Json -Depth 8 | Set-Content -Path $attentionPolicy -Encoding UTF8
@{
  schema_version = "atx-moe-residency-policy-v1"
  policy_name = "known_fast_tail_layers"
  keep_layers = @($KnownFastLayers -split "," | ForEach-Object {
    if ($_ -match "-") { $a,$b = $_ -split "-"; $a..$b } else { [int]$_ }
  })
  basis = "Known fast whole-layer tail residency candidate from previous ATX sweeps."
} | ConvertTo-Json -Depth 8 | Set-Content -Path $knownFastPolicy -Encoding UTF8

if ($Scenarios -contains "reference") {
  $rows += Invoke-Scenario -Name "reference_ncpu_moe_$NCpuMoe" -UseNCpuMoe $true -ExtraArgs @()
}
if ($Scenarios -contains "attention") {
  $rows += Invoke-Scenario -Name "attention_layers" -ExtraArgs @("--moe-residency-policy", $attentionPolicy, "--moe-residency-mode", "layer")
}
if ($Scenarios -contains "known-fast") {
  $rows += Invoke-Scenario -Name "known_fast_tail_layers" -ExtraArgs @("--moe-residency-policy", $knownFastPolicy, "--moe-residency-mode", "layer")
}
if ($Scenarios -contains "policy") {
  if (-not $PolicyPath) { throw "-PolicyPath is required when -Scenarios contains policy" }
  $rows += Invoke-Scenario -Name $PolicyScenarioName -ExtraArgs @(
    "--moe-residency-policy", $PolicyPath,
    "--moe-residency-mode", "auto",
    "--moe-direct-require",
    "--moe-direct-strict-hot-no-stage",
    "--moe-cold-coalesce-gap", "0"
  )
}

if ((-not $SkipCompile) -and ($Scenarios -contains "bottleneck-auto")) {
  $seedStats = ($rows | Where-Object scenario -eq "known_fast_tail_layers" | Select-Object -First 1).stats
  if (-not $seedStats) {
    $rows += Invoke-Scenario -Name "known_fast_tail_layers" -ExtraArgs @("--moe-residency-policy", $knownFastPolicy, "--moe-residency-mode", "layer")
    $seedStats = ($rows | Where-Object scenario -eq "known_fast_tail_layers" | Select-Object -First 1).stats
  }
  $compileArgs = @(
    "scripts\atx_moe_policy_compile.py",
    "--policy-source", $PolicySource,
    "--out-dir", $policyDir,
    "--mode", "auto",
    "--stats-json", $seedStats,
    "--max-promote-layers", "$MaxPromoteLayers",
    "--host-byte-reduction-target", "0.80",
    "--attention-baseline-layers", $AttentionLayers,
    "--base-keep-layers", $KnownFastLayers,
    "--swap-candidates", "$SwapCandidates"
  )
  & python @compileArgs
  if ($LASTEXITCODE -ne 0) { throw "policy compiler failed with exit code $LASTEXITCODE" }
  $bottleneckPolicy = Get-ChildItem -Path $policyDir -Filter $BottleneckPolicyPattern | Select-Object -First 1
  if (-not $bottleneckPolicy) {
    $bottleneckPolicy = Get-ChildItem -Path $policyDir -Filter "bottleneck_top_*.atx.json" | Sort-Object Length -Descending | Select-Object -First 1
  }
  if (-not $bottleneckPolicy) {
    $bottleneckPolicy = Get-ChildItem -Path $policyDir -Filter "*.bottleneck_first.atx.json" | Sort-Object Length -Descending | Select-Object -First 1
  }
  if ($bottleneckPolicy) {
    $rows += Invoke-Scenario -Name "bottleneck_auto" -ExtraArgs @(
      "--moe-residency-policy", $bottleneckPolicy.FullName,
      "--moe-residency-mode", "auto",
      "--moe-direct-require",
      "--moe-direct-strict-hot-no-stage",
      "--moe-cold-coalesce-gap", "0"
    )
  }
}

$best = $rows | Where-Object { $_.decode_tps -ne $null } | Sort-Object decode_tps -Descending | Select-Object -First 1
$reference = $rows | Where-Object scenario -eq "reference_ncpu_moe_$NCpuMoe" | Select-Object -First 1
$summary = [pscustomobject]@{
  out_dir = $OutDir
  target_decode_tps = $TargetDecodeTps
  reference_decode_tps = $ReferenceDecodeTps
  best_scenario = if ($best) { $best.scenario } else { $null }
  best_decode_tps = if ($best) { $best.decode_tps } else { $null }
  reproduced_reference_decode_tps = if ($reference) { $reference.decode_tps } else { $null }
  pass_80_tps = [bool]($best -and $best.decode_tps -ge $TargetDecodeTps)
  pass_reference_1_10x = [bool]($best -and $best.decode_tps -ge (1.10 * $ReferenceDecodeTps))
  rows = $rows
}
$summary | ConvertTo-Json -Depth 10 | Set-Content -Path $summaryJson -Encoding UTF8
$summary | ConvertTo-Json -Depth 4
