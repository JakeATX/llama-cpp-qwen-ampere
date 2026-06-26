param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $Dataset,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $PerplexityExe = "",
    [string] $OutputDir = "",
    [string] $KvarnPreset = "kvarn_k8v8_g128",
    [int] $KvarnIters = 16,
    [int] $ContextSize = 4096,
    [int] $BatchSize = 4096,
    [int] $Chunks = 2,
    [int] $DumpLayer = 3,
    [int] $DumpIq = 2047,
    [int] $DumpIh = 0,
    [int] $DumpHead = 0,
    [int] $BoundaryMinTokens = -1,
    [int] $BoundaryDumpCall = -1,
    [int] $BoundaryDumpLimit = 1,
    [int] $BodyRecordLimit = 30,
    [int] $BodySrcLayout = -1,
    [ValidateSet("auto", "earliest", "latest")] [string] $RecordSet = "auto",
    [switch] $TensorDump,
    [string] $TensorDumpFilter = "",
    [int] $TensorDumpLimit = 16,
    [switch] $DumpFullQO,
    [switch] $AllowDiagnosticEnv,
    [switch] $ParseSpecial,
    [switch] $AllowChatMarkers,
    [switch] $SkipFixtureCheck,
    [switch] $ForceExperimentalIswa,
    [switch] $PaperMixedFrame,
    [switch] $DisablePaperFrame,
    [switch] $DebugRawBody,
    [string] $LayerFilter = "",
    [string] $ExpectedKvarnLayers = "",
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @()
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path (Get-Location) "artifacts/kvarn-rootcause/boundary-$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$boundaryDir = Join-Path $OutputDir "boundary"
$bodyDir = Join-Path $OutputDir "body-records"
$tensorDir = Join-Path $OutputDir "tensor-dump"
[void] [System.IO.Directory]::CreateDirectory($boundaryDir)
[void] [System.IO.Directory]::CreateDirectory($bodyDir)
if ($TensorDump) {
    [void] [System.IO.Directory]::CreateDirectory($tensorDir)
}

$oldEnv = @{}
$envSet = @{
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP" = "1"
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_DIR" = $boundaryDir
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LIMIT" = [string] $BoundaryDumpLimit
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LAYER" = [string] $DumpLayer
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_IQ" = [string] $DumpIq
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_IH" = [string] $DumpIh
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_MIN_TOKENS" = [string] $(if ($BoundaryMinTokens -ge 0) { $BoundaryMinTokens } else { $ContextSize })
    "LLAMA_KVARN_DEBUG_BODY_RECORD_DUMP" = "1"
    "LLAMA_KVARN_DEBUG_BODY_RECORD_DIR" = $bodyDir
    "LLAMA_KVARN_DEBUG_BODY_RECORD_LIMIT" = [string] $BodyRecordLimit
    "LLAMA_KVARN_DEBUG_BODY_RECORD_LAYER" = [string] $DumpLayer
    "LLAMA_KVARN_DEBUG_BODY_HEAD" = [string] $DumpHead
}

if (-not $DisablePaperFrame) {
    $envSet["LLAMA_KVARN_ENABLE_PAPER_FRAME"] = "1"
}
if ($PaperMixedFrame) {
    $envSet["LLAMA_KVARN_PAPER_MIXED_FRAME"] = "1"
}
if ($DebugRawBody) {
    $envSet["LLAMA_KVARN_DEBUG_RAW_BODY_K"] = "1"
    $envSet["LLAMA_KVARN_DEBUG_RAW_BODY_V"] = "1"
}

if ($BodySrcLayout -ge 0) {
    $envSet["LLAMA_KVARN_DEBUG_BODY_SRC_LAYOUT"] = [string] $BodySrcLayout
}
if ($BoundaryDumpCall -ge 0) {
    $envSet["LLAMA_KVARN_ATTN_BOUNDARY_DUMP_CALL"] = [string] $BoundaryDumpCall
}
if ($DumpFullQO) {
    $envSet["LLAMA_KVARN_ATTN_BOUNDARY_DUMP_FULL_QO"] = "1"
}
if ($ForceExperimentalIswa) {
    $envSet["LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA"] = "1"
}
if ($TensorDump) {
    $envSet["LLAMA_KVARN_TENSOR_DUMP_DIR"] = $tensorDir
    $envSet["LLAMA_KVARN_TENSOR_DUMP_LIMIT"] = [string] $TensorDumpLimit
    if ([string]::IsNullOrWhiteSpace($TensorDumpFilter)) {
        $TensorDumpFilter = "^(Kcur-$DumpLayer|Vcur-$DumpLayer)$"
    }
    $envSet["LLAMA_KVARN_TENSOR_DUMP_FILTER"] = $TensorDumpFilter
}
if (-not [string]::IsNullOrWhiteSpace($LayerFilter)) {
    $envSet["LLAMA_KVARN_LAYER_FILTER"] = $LayerFilter
    if ([string]::IsNullOrWhiteSpace($ExpectedKvarnLayers)) {
        $ExpectedKvarnLayers = $LayerFilter
    }
}

foreach ($key in $envSet.Keys) {
    $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
    [Environment]::SetEnvironmentVariable($key, $envSet[$key], "Process")
}

try {
    scripts\kvarn\run_accuracy_gate.ps1 `
        -Model $Model `
        -Dataset $Dataset `
        -BuildDir $BuildDir `
        -PerplexityExe $PerplexityExe `
        -OutputDir (Join-Path $OutputDir "accuracy") `
        -KvarnPreset $KvarnPreset `
        -KvarnIters $KvarnIters `
        -KvarnRtnQuantile 1.0 `
        -FlashAttn off `
        -Fit off `
        -GpuLayers 999 `
        -ContextSize $ContextSize `
        -BatchSize $BatchSize `
        -Chunks $Chunks `
        -MaxPplIncrease 1000 `
        -AllowDiagnosticEnv:$($AllowDiagnosticEnv.IsPresent) `
        -ExpectedKvarnLayers $ExpectedKvarnLayers `
        -ParseSpecial:$($ParseSpecial.IsPresent) `
        -AllowChatMarkers:$($AllowChatMarkers.IsPresent) `
        -SkipFixtureCheck:$($SkipFixtureCheck.IsPresent) `
        -ExtraArgs @($ExtraArgs) `
        -KvarnExtraArgs @($KvarnExtraArgs)
} finally {
    foreach ($key in $envSet.Keys) {
        [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
    }
}

$boundaryChildren = @(Get-ChildItem -LiteralPath $boundaryDir -Directory | Where-Object {
    Test-Path -LiteralPath (Join-Path $_.FullName "boundary.json")
} | Sort-Object Name)
if ($boundaryChildren.Count -eq 0 -and (Test-Path -LiteralPath (Join-Path $boundaryDir "boundary.json"))) {
    $boundaryChildren = @((Get-Item -LiteralPath $boundaryDir))
}
if ($boundaryChildren.Count -eq 0) {
    throw "No boundary dumps found under $boundaryDir"
}

$replayRows = @()
foreach ($boundary in $boundaryChildren) {
    $boundaryRecordSet = $RecordSet
    if ($boundaryRecordSet -eq "auto") {
        $boundaryRecordSet = "latest"
        if ($boundary.Name -match "call_0*0$") {
            $boundaryRecordSet = "earliest"
        }
    }
    python scripts\kvarn\replay_mixed_attn_boundary.py --dump $boundary.FullName --write-replay
    if ($LASTEXITCODE -ne 0) {
        throw "KVarN mixed-attn self replay failed with exit code $LASTEXITCODE for $($boundary.FullName)"
    }
    python scripts\kvarn\replay_f16_truth_boundary.py --boundary $boundary.FullName --body-records $bodyDir --record-set $boundaryRecordSet --write-truth
    if ($LASTEXITCODE -ne 0) {
        throw "KVarN f16-truth replay failed with exit code $LASTEXITCODE for $($boundary.FullName)"
    }

    $selfSummary = Join-Path $boundary.FullName "full_qo_summary.json"
    $truthSummary = Join-Path $boundary.FullName "f16_truth_full_qo_summary.json"
    if ((Test-Path -LiteralPath $selfSummary) -and (Test-Path -LiteralPath $truthSummary)) {
        $self = Get-Content -LiteralPath $selfSummary -Raw | ConvertFrom-Json
        $truth = Get-Content -LiteralPath $truthSummary -Raw | ConvertFrom-Json
        $replayRows += [pscustomobject]@{
            Boundary = $boundary.Name
            SelfMaxOutNmse = [double] $self.worst.out_nmse
            TruthMaxOutNmse = [double] $truth.worst.out_nmse
            RecordSet = $boundaryRecordSet
            TruthWorstIq = [int] $truth.worst.iq
            TruthWorstIh = [int] $truth.worst.ih
            TruthWorstD = [int] $truth.worst.out_worst_d
        }
    }
}

if ($replayRows.Count -gt 0) {
    $replayRows | Export-Csv -LiteralPath (Join-Path $OutputDir "boundary_replay_summary.csv") -NoTypeInformation
    $worst = $replayRows | Sort-Object TruthMaxOutNmse -Descending | Select-Object -First 1
    Write-Host ("Worst f16-truth boundary: {0} truth_nmse={1:E6} iq={2} ih={3}" -f `
            $worst.Boundary, $worst.TruthMaxOutNmse, $worst.TruthWorstIq, $worst.TruthWorstIh)
}

Write-Host "KVarN ctx4096 truth capture complete: $OutputDir"
