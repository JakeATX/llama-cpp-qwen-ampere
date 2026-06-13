param(
    [Parameter(Mandatory = $true)] [string] $BoundaryDir,
    [switch] $WriteReference,
    [double] $MaxOutNmse = -1
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $BoundaryDir)) {
    throw "BoundaryDir not found at $BoundaryDir"
}

$jsonPath = Join-Path $BoundaryDir "boundary.json"
if (-not (Test-Path -LiteralPath $jsonPath)) {
    $children = Get-ChildItem -LiteralPath $BoundaryDir -Directory -ErrorAction SilentlyContinue | Sort-Object Name
    if ($children.Count -eq 1) {
        $BoundaryDir = $children[0].FullName
        $jsonPath = Join-Path $BoundaryDir "boundary.json"
    }
}
if (-not (Test-Path -LiteralPath $jsonPath)) {
    throw "boundary.json not found under $BoundaryDir"
}

$meta = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
$requiredMeta = @(
    "version",
    "head_dim",
    "group_size",
    "key_bits",
    "value_bits",
    "n_queries",
    "n_head",
    "n_head_kv",
    "n_gqa",
    "selected_iq",
    "selected_ih",
    "selected_ikh",
    "n_sink",
    "n_records",
    "n_pending",
    "n_tail",
    "n_tokens",
    "scale",
    "mask_type",
    "mask_stride_query_bytes",
    "mask_stride_token_bytes",
    "q_stride_head_floats",
    "q_stride_query_floats",
    "out_stride_head_floats",
    "out_stride_query_floats",
    "sink_tail_stride_head_f16",
    "sink_tail_stride_token_f16",
    "pending_stride_head_floats",
    "pending_stride_token_floats",
    "k_body_stride_record_bytes",
    "v_body_stride_record_bytes",
    "k_body_stride_head_bytes",
    "v_body_stride_head_bytes",
    "k_scale_stride_record_floats",
    "v_scale_stride_record_floats",
    "k_scale_stride_head_floats",
    "v_scale_stride_head_floats",
    "scores_nelems",
    "body_records_cap",
    "qt",
    "body_mirror_allowed",
    "body_mirror_used",
    "call_index",
    "cuda_trace_mode"
)

$missing = @()
foreach ($name in $requiredMeta) {
    if (-not ($meta.PSObject.Properties.Name -contains $name)) {
        $missing += $name
    }
}
if ($missing.Count -gt 0) {
    throw "Boundary metadata missing fields: $($missing -join ', ')"
}

$requiredFiles = @(
    "q.bin",
    "sink_tail_k_f16.bin",
    "sink_tail_v_f16.bin",
    "body_k.bin",
    "body_v.bin",
    "scales_k.bin",
    "scales_v.bin",
    "pending_k.bin",
    "pending_v.bin",
    "warpqk_out.bin"
)
if ([int] $meta.mask_type -ne 0) {
    $requiredFiles += "mask.bin"
}

$missingFiles = @()
foreach ($name in $requiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $BoundaryDir $name))) {
        $missingFiles += $name
    }
}
if ($missingFiles.Count -gt 0) {
    throw "Boundary dump missing files: $($missingFiles -join ', ')"
}

if ([int] $meta.head_dim -ne 256) {
    throw "Expected a 256d Qwen boundary, got head_dim=$($meta.head_dim)"
}

Write-Host ("Boundary metadata validation: PASS path={0} mode={1} q={2} h={3} records={4} pending={5} tail={6} qt={7} mirror={8}/{9}" -f `
    $BoundaryDir, $meta.cuda_trace_mode, $meta.n_queries, $meta.n_head,
    $meta.n_records, $meta.n_pending, $meta.n_tail, $meta.qt,
    $meta.body_mirror_allowed, $meta.body_mirror_used)

$pythonReplay = Join-Path $PSScriptRoot "replay_qwen_boundary.py"
if (-not (Test-Path -LiteralPath $pythonReplay)) {
    throw "Missing Python replay helper at $pythonReplay"
}

$argv = @($pythonReplay, "--dump", $BoundaryDir)
if ($WriteReference.IsPresent) {
    $argv += "--write-reference"
}
if ($MaxOutNmse -ge 0) {
    $argv += @("--max-out-nmse", ([string] $MaxOutNmse))
}

& python @argv
if ($LASTEXITCODE -ne 0) {
    throw "Boundary CPU replay failed with exit code $LASTEXITCODE"
}
