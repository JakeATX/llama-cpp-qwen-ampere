param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [string] $CaseList = "tg64:0:64,pp128:128:0,tg384:0:384",
    [int] $GpuLayers = 99,
    [ValidateSet("on", "off", "auto")] [string] $FlashAttn = "off",
    [string] $KvCacheQuant = "none,kvarn",
    [string] $KvarnPreset = "kvarn_k4v2_g128",
    [double] $RtnQuantile = 0.95,
    [int] $Repetitions = 1,
    [ValidateSet("csv", "json", "jsonl", "md", "sql")] [string] $OutputFormat = "md",
    [string] $OutputDir = "",
    [switch] $Warmup,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($Repetitions -le 0) {
    throw "Repetitions must be positive"
}
if ($GpuLayers -lt 0) {
    throw "GpuLayers must be non-negative"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found at $Model"
}

$bench = Join-Path $BuildDir "bin/Release/llama-bench.exe"
if (-not (Test-Path -LiteralPath $bench)) {
    throw "llama-bench.exe not found at $bench"
}
$bench = (Resolve-Path -LiteralPath $bench).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path (Get-Location) "artifacts/kvarn-bench/$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

function Convert-ToFileStem([string] $name) {
    return ($name -replace '[^A-Za-z0-9_.-]', '_')
}

function Get-BenchCases([string] $caseList) {
    $cases = @()
    foreach ($raw in $caseList.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)) {
        $parts = $raw.Trim().Split(":", [System.StringSplitOptions]::None)
        if ($parts.Count -ne 3) {
            throw "Invalid benchmark case '$raw'; expected name:prompt_tokens:gen_tokens"
        }

        $promptTokens = 0
        $genTokens = 0
        if (-not [int]::TryParse($parts[1], [ref] $promptTokens)) {
            throw "Invalid prompt token count in case '$raw'"
        }
        if (-not [int]::TryParse($parts[2], [ref] $genTokens)) {
            throw "Invalid generation token count in case '$raw'"
        }
        if ($promptTokens -lt 0 -or $genTokens -lt 0) {
            throw "Benchmark case '$raw' has a negative token count"
        }
        if ($promptTokens -eq 0 -and $genTokens -eq 0) {
            throw "Benchmark case '$raw' must request prompt or generation tokens"
        }

        $cases += [pscustomobject]@{
            Name = $parts[0].Trim()
            PromptTokens = $promptTokens
            GenTokens = $genTokens
        }
    }

    if ($cases.Count -eq 0) {
        throw "CaseList must include at least one benchmark case"
    }
    return $cases
}

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$requiresKvarnEvidence = ($KvCacheQuant.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) |
    ForEach-Object { $_.Trim().ToLowerInvariant() }) -contains "kvarn"
$manifest = @(
    "model=$modelPath",
    "bench=$bench",
    "build_dir=$BuildDir",
    "cases=$CaseList",
    "kv_cache_quant=$KvCacheQuant",
    "flash_attn=$FlashAttn",
    "gpu_layers=$GpuLayers",
    "kvarn_preset=$KvarnPreset",
    "kvarn_rtn_quantile=$rtnQuantileArg",
    "repetitions=$Repetitions",
    "output_format=$OutputFormat",
    "warmup=$($Warmup.IsPresent)",
    "extra_args=$($ExtraArgs -join ' ')"
)
[System.IO.File]::WriteAllText((Join-Path $OutputDir "manifest.txt"), ($manifest -join "`n") + "`n")

foreach ($case in (Get-BenchCases $CaseList)) {
    $argv = @(
        "-m", $modelPath,
        "-p", [string] $case.PromptTokens,
        "-n", [string] $case.GenTokens,
        "-r", [string] $Repetitions,
        "-ngl", [string] $GpuLayers,
        "-fa", $FlashAttn,
        "-o", $OutputFormat,
        "--kv-cache-quant", $KvCacheQuant,
        "--kvarn-preset", $KvarnPreset,
        "--kvarn-rtn-quantile", $rtnQuantileArg
    )
    if (-not $Warmup.IsPresent) {
        $argv += "--no-warmup"
    }
    if ($ExtraArgs.Count -gt 0) {
        $argv += $ExtraArgs
    }

    $stem = Convert-ToFileStem $case.Name
    $logPath = Join-Path $OutputDir "$stem.$OutputFormat.txt"
    $cmdPath = Join-Path $OutputDir "$stem.command.txt"
    $commandLine = "`"$bench`" " + (($argv | ForEach-Object {
        if ($_ -match '\s') {
            "`"$_`""
        } else {
            $_
        }
    }) -join " ")
    [System.IO.File]::WriteAllText($cmdPath, $commandLine + "`n")

    Write-Host "== KVarN bench case: $($case.Name) p=$($case.PromptTokens) n=$($case.GenTokens)"
    Write-Host $commandLine

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $bench @argv 2>&1
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }

    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    [System.IO.File]::WriteAllText($logPath, $text + "`n")
    $text | Write-Host
    if ($exit -ne 0) {
        throw "llama-bench failed for case '$($case.Name)' with exit code $exit; see $logPath"
    }
    if ($requiresKvarnEvidence) {
        if ($text -notmatch "llama_kv_cache_kvarn:") {
            throw "llama-bench case '$($case.Name)' succeeded but logs did not show KVarN cache initialization; see $logPath"
        }
        $kvarnLayerLogs = ([regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer")).Count
        if ($kvarnLayerLogs -lt 1) {
            throw "llama-bench case '$($case.Name)' succeeded but did not show any KVarN layer allocation lines; see $logPath"
        }
        if ($text -notmatch "(?i)\bkvarn\b") {
            throw "llama-bench case '$($case.Name)' output did not include a KVarN benchmark row; see $logPath"
        }
        Write-Host ("KVarN bench log check: PASS, KVarN layer lines = {0}" -f $kvarnLayerLogs)
    }
}

Write-Host "KVarN benchmark matrix complete: $OutputDir"
