param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $OutputDir = "",
    [int] $Context = 512,
    [int] $Batch = 512,
    [int] $Repeat = 450,
    [string] $PromptPhrase = "hello ",
    [int] $TraceLimit = 16,
    [int] $DumpLimit = 1,
    [int] $DumpCall = -1,
    [int] $DumpLayer = -1,
    [int] $DumpIq = 0,
    [int] $DumpIh = 0,
    [switch] $DumpFullQO,
    [switch] $FirstBodyActive256D,
    [switch] $Enable256DWarpqk,
    [switch] $Enable256DBodyMirror,
    [ValidateSet(0, 1, 4, 8)] [int] $ForceQt = 0,
    [string[]] $ExtraArgs = @("-ncmoe", "34", "-fit", "off")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found at $Model"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path (Get-Location) "artifacts/kvarn-boundary/qwen36-$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$script = Join-Path (Get-Location) "scripts/kvarn/compare_cuda_logits_ref.ps1"
if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing $script"
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

$envSet = @{
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP" = "1"
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_DIR" = $OutputDir
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LIMIT" = [string] $DumpLimit
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_IQ" = [string] $DumpIq
    "LLAMA_KVARN_ATTN_BOUNDARY_DUMP_IH" = [string] $DumpIh
}
if ($DumpCall -ge 0) {
    $envSet["LLAMA_KVARN_ATTN_BOUNDARY_DUMP_CALL"] = [string] $DumpCall
}
if ($DumpLayer -ge 0) {
    $envSet["LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LAYER"] = [string] $DumpLayer
}
if ($FirstBodyActive256D.IsPresent) {
    $envSet["LLAMA_KVARN_ATTN_BOUNDARY_DUMP_FIRST_256D"] = "1"
}
if ($DumpFullQO.IsPresent) {
    $envSet["LLAMA_KVARN_ATTN_BOUNDARY_DUMP_FULL_QO"] = "1"
}
if ($Enable256DWarpqk.IsPresent) {
    $envSet["LLAMA_KVARN_ATTN_ENABLE_256D_WARPQK"] = "1"
}
if ($Enable256DBodyMirror.IsPresent) {
    $envSet["LLAMA_KVARN_ATTN_ENABLE_256D_BODY_MIRROR"] = "1"
}
if ($ForceQt -ne 0) {
    $envSet["LLAMA_KVARN_ATTN_WARPQK_FORCE_QT"] = [string] $ForceQt
}

$manifest = @(
    "model=$((Resolve-Path -LiteralPath $Model).Path)",
    "build_dir=$BuildDir",
    "context=$Context",
    "batch=$Batch",
    "repeat=$Repeat",
    "prompt_phrase=$PromptPhrase",
    "trace_limit=$TraceLimit",
    "dump_limit=$DumpLimit",
    "dump_call=$DumpCall",
    "dump_layer=$DumpLayer",
    "dump_iq=$DumpIq",
    "dump_ih=$DumpIh",
    "dump_full_qo=$($DumpFullQO.IsPresent)",
    "first_body_active_256d=$($FirstBodyActive256D.IsPresent)",
    "enable_256d_warpqk=$($Enable256DWarpqk.IsPresent)",
    "enable_256d_body_mirror=$($Enable256DBodyMirror.IsPresent)",
    "force_qt=$ForceQt",
    "extra_args=$($ExtraArgs -join ' ')"
)
[System.IO.File]::WriteAllText((Join-Path $OutputDir "capture_manifest.txt"), ($manifest -join "`n") + "`n")

Invoke-WithProcessEnvironment $envSet {
    & $script `
        -Model $Model `
        -BuildDir $BuildDir `
        -Context $Context `
        -Batch $Batch `
        -Repeat $Repeat `
        -PromptPhrase $PromptPhrase `
        -KvarnIters 4 `
        -CheckPackedRepeat `
        -CheckPackedSplit `
        -SkipScratchCheck `
        -TraceAttn `
        -TraceLimit $TraceLimit `
        -FlashAttn off `
        -MinKvarnLayerLogs 10 `
        -MinKvarnBodyRecords 1 `
        -ExpectedKvarnLayers "3-39:4" `
        -ExtraArgs @($ExtraArgs)
}

Write-Host "KVarN boundary capture complete: $OutputDir"
