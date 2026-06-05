param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [int] $Context = 512,
    [int] $GpuLayers = 99,
    [double] $RtnQuantile = 0.95,
    [int] $Repeat = 32,
    [string] $PromptPhrase = "The quick brown fox studies attention kernels and cache layouts carefully. ",
    [string] $OutputFile = (Join-Path $env:TEMP "kvarn-packed-logits.gguf"),
    [string] $PromptFile = (Join-Path $env:TEMP "kvarn-logits-prompt.txt"),
    [switch] $KeepArtifacts
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($Repeat -le 0) {
    throw "Repeat must be positive"
}

function Get-ExePath([string] $buildDir, [string] $name) {
    $path = Join-Path $buildDir "bin/Release/$name"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "$name not found at $path"
    }
    return (Resolve-Path -LiteralPath $path).Path
}

function Invoke-Results([string] $exe, [string[]] $argv, [hashtable] $envSet) {
    $oldEnv = @{}
    foreach ($key in $envSet.Keys) {
        $oldEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $envSet[$key], "Process")
    }

    try {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & $exe @argv 2>&1
            $exit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
    } finally {
        foreach ($key in $envSet.Keys) {
            [Environment]::SetEnvironmentVariable($key, $oldEnv[$key], "Process")
        }
    }

    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    if ($exit -ne 0) {
        throw "llama-results failed with exit code $exit`n$text"
    }

    return $text
}

$results = Get-ExePath $BuildDir "llama-results.exe"
$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)

[System.IO.File]::WriteAllText($PromptFile, ($PromptPhrase * $Repeat))
Remove-Item -LiteralPath $OutputFile -ErrorAction SilentlyContinue

$commonArgs = @(
    "-m", $Model,
    "-f", $PromptFile,
    "-o", $OutputFile,
    "-c", [string] $Context,
    "-ngl", [string] $GpuLayers,
    "-fa", "on",
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", "kvarn_k4v2_g128",
    "--kvarn-rtn-quantile", $rtnQuantileArg
)

try {
    Write-Host "== Saving packed KVarN logits"
    [void] (Invoke-Results $results $commonArgs @{})

    Write-Host "== Checking scratch-reference KVarN logits"
    $check = Invoke-Results $results ($commonArgs + @("--check")) @{ "LLAMA_KVARN_ATTN_REF_SCRATCH" = "1" }
    $m = [regex]::Match($check, "NMSE=([0-9.eE+-]+)")
    if (-not $m.Success) {
        throw "Could not find NMSE in llama-results output"
    }
    $nmse = [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ("KVarN packed-vs-scratch logits: PASS, NMSE = {0:E3}" -f $nmse)
} finally {
    if (-not $KeepArtifacts) {
        Remove-Item -LiteralPath $OutputFile -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $PromptFile -ErrorAction SilentlyContinue
    }
}
