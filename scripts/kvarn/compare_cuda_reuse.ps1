param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [int] $Context = 512,
    [int] $Predict = 270,
    [int] $GpuLayers = 99,
    [int] $Seed = 1234,
    [double] $RtnQuantile = 1.0,
    [string] $Prompt = "Hello",
    [int] $MinKvarnLayerLogs = 1,
    [string] $ExpectedKvarnLayers = "",
    [switch] $SkipNormalBaseline
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($MinKvarnLayerLogs -lt 1) {
    throw "MinKvarnLayerLogs must be positive"
}

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)

function Get-ExpectedKvarnLayerIds([string] $layers) {
    if ([string]::IsNullOrWhiteSpace($layers)) {
        return @()
    }

    $ids = @()
    foreach ($raw in ($layers -split "[,\s]+")) {
        if ([string]::IsNullOrWhiteSpace($raw)) {
            continue
        }
        $id = 0
        if (-not [int]::TryParse($raw, [ref] $id) -or $id -lt 0) {
            throw "Invalid KVarN layer id '$raw' in ExpectedKvarnLayers"
        }
        $ids += $id
    }
    return $ids
}

function Assert-ExpectedKvarnLayers([string] $text, [int[]] $expected, [string] $label) {
    if ($expected.Count -eq 0) {
        return
    }

    $actual = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($m in [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer\s+([0-9]+)")) {
        [void] $actual.Add([int] $m.Groups[1].Value)
    }

    $missing = @()
    foreach ($id in $expected) {
        if (-not $actual.Contains($id)) {
            $missing += $id
        }
    }
    if ($missing.Count -gt 0) {
        throw "$label missing expected KVarN layer ids: $($missing -join ',')"
    }
    Write-Host ("KVarN expected layer check: PASS, layers = {0}" -f ($expected -join ","))
}

function Get-ExePath([string] $buildDir, [string] $name) {
    $path = Join-Path $buildDir "bin/Release/$name"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "$name not found at $path"
    }
    return (Resolve-Path -LiteralPath $path).Path
}

function Invoke-Completion([string] $exe, [string[]] $argv, [hashtable] $envSet, [bool] $ExpectKvarn = $true) {
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
        throw "llama-completion failed with exit code $exit`n$text"
    }

    if ($ExpectKvarn) {
        if ($text -notmatch "llama_kv_cache_kvarn:") {
            throw "Expected KVarN runtime logs, but none were found"
        }

        $layerLines = [regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer").Count
        if ($layerLines -lt $MinKvarnLayerLogs) {
            throw "Expected at least $MinKvarnLayerLogs KVarN layer logs, got $layerLines"
        }

        Assert-ExpectedKvarnLayers $text $expectedKvarnLayerIds "llama-completion"
        Write-Host ("KVarN completion log check: PASS, KVarN layer lines = {0}" -f $layerLines)
    }

    return $text
}

function Get-GeneratedText([string] $output) {
    $lines = $output -split "`r?`n"
    $generated = New-Object System.Collections.Generic.List[string]
    $inGeneration = $false

    foreach ($line in $lines) {
        if ($line -match "^\d+\.\d+\.\d+\s+I\s+generate:") {
            $inGeneration = $true
            continue
        }

        if (-not $inGeneration) {
            continue
        }

        if ($line -match "^\d+\.\d+\.\d+\s+[A-Z]\s+common_perf_print:") {
            break
        }

        if ($line -match "^\d+\.\d+\.\d+\s+[A-Z]\s+") {
            continue
        }
        if ($line -match "^llama_kv_cache_kvarn:") {
            continue
        }

        [void] $generated.Add($line)
    }

    return (($generated -join "`n").Trim())
}

function Get-GraphsReused([string] $output) {
    $m = [regex]::Match($output, "graphs reused\s+=\s+([0-9]+)")
    if (-not $m.Success) {
        throw "Could not find graph reuse counter in output"
    }
    return [int] $m.Groups[1].Value
}

function Get-EvalTokensPerSecond([string] $output) {
    $m = [regex]::Match($output, "eval time\s+=\s+[0-9.]+\s+ms\s+/\s+[0-9]+\s+runs\s+\([^)]+\s+([0-9.]+)\s+tokens per second\)")
    if (-not $m.Success) {
        return $null
    }
    return [double] $m.Groups[1].Value
}

$completion = Get-ExePath $BuildDir "llama-completion.exe"
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers
$commonArgs = @(
    "-m", $Model,
    "-p", $Prompt,
    "-n", [string] $Predict,
    "-c", [string] $Context,
    "-ngl", [string] $GpuLayers,
    "--no-warmup",
    "--simple-io",
    "-no-cnv",
    "--no-display-prompt",
    "--ignore-eos",
    "-s", [string] $Seed,
    "--temp", "0"
)

$kvarnArgs = $commonArgs + @(
    "-fa", "on",
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", "kvarn_k4v2_g128",
    "--kvarn-rtn-quantile", $rtnQuantileArg
)

Write-Host "== KVarN reference run: graph reuse disabled, rtn_quantile=$rtnQuantileArg"
$reuseOff = Invoke-Completion $completion $kvarnArgs @{ "LLAMA_GRAPH_REUSE_DISABLE" = "1" }
$reuseOffText = Get-GeneratedText $reuseOff
$reuseOffGraphs = Get-GraphsReused $reuseOff
$reuseOffTps = Get-EvalTokensPerSecond $reuseOff

if ($reuseOffGraphs -ne 0) {
    throw "Expected graph reuse disabled run to report 0 reused graphs, got $reuseOffGraphs"
}

Write-Host "== KVarN optimized run: graph reuse enabled, rtn_quantile=$rtnQuantileArg"
$reuseOn = Invoke-Completion $completion $kvarnArgs @{}
$reuseOnText = Get-GeneratedText $reuseOn
$reuseOnGraphs = Get-GraphsReused $reuseOn
$reuseOnTps = Get-EvalTokensPerSecond $reuseOn

if ($reuseOnGraphs -le 0) {
    throw "Expected graph reuse enabled run to reuse at least one graph, got $reuseOnGraphs"
}

if ($reuseOffText -ne $reuseOnText) {
    Write-Host "== reuse-disabled generated text =="
    Write-Host $reuseOffText
    Write-Host "== reuse-enabled generated text =="
    Write-Host $reuseOnText
    throw "KVarN graph reuse changed deterministic generated text"
}

Write-Host "KVarN reuse parity: PASS"
Write-Host ("KVarN reuse disabled: graphs reused = {0}, eval tok/s = {1}" -f $reuseOffGraphs, $reuseOffTps)
Write-Host ("KVarN reuse enabled : graphs reused = {0}, eval tok/s = {1}" -f $reuseOnGraphs, $reuseOnTps)

if (-not $SkipNormalBaseline) {
    Write-Host "== Normal KV baseline"
    $normalArgs = $commonArgs + @(
        "-fa", "off",
        "--kv-cache-quant", "none"
    )
    $normal = Invoke-Completion $completion $normalArgs @{} $false
    $normalGraphs = Get-GraphsReused $normal
    $normalTps = Get-EvalTokensPerSecond $normal
    Write-Host ("Normal KV baseline: graphs reused = {0}, eval tok/s = {1}" -f $normalGraphs, $normalTps)
}
