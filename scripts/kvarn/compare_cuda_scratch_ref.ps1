param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [int] $Context = 512,
    [int] $Predict = 270,
    [int] $GpuLayers = 99,
    [int] $Seed = 1234,
    [double] $RtnQuantile = 0.95,
    [string] $Prompt = "Hello",
    [int] $MinKvarnLayerLogs = 1,
    [int] $MinKvarnBodyRecords = 0,
    [string] $ExpectedKvarnLayers = "",
    [switch] $DisableGraphReuse
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($MinKvarnLayerLogs -lt 1) {
    throw "MinKvarnLayerLogs must be positive"
}
if ($MinKvarnBodyRecords -lt 0) {
    throw "MinKvarnBodyRecords must be non-negative"
}

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)

function Get-ExePath([string] $buildDir, [string] $name) {
    $path = Join-Path $buildDir "bin/Release/$name"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "$name not found at $path"
    }
    return (Resolve-Path -LiteralPath $path).Path
}

function Get-ExpectedKvarnLayerIds([string] $layers) {
    if ([string]::IsNullOrWhiteSpace($layers)) {
        return @()
    }

    $ids = @()
    foreach ($raw in ($layers -split "[,\s]+")) {
        if ([string]::IsNullOrWhiteSpace($raw)) {
            continue
        }
        if ($raw -match '^([0-9]+)-([0-9]+)(?::([0-9]+))?$') {
            $start = [int] $Matches[1]
            $end = [int] $Matches[2]
            $step = if ($Matches.ContainsKey(3) -and -not [string]::IsNullOrEmpty($Matches[3])) { [int] $Matches[3] } else { 1 }
            if ($end -lt $start) {
                throw "Invalid KVarN layer range '$raw' in ExpectedKvarnLayers"
            }
            if ($step -le 0) {
                throw "Invalid KVarN layer range step '$raw' in ExpectedKvarnLayers"
            }
            for ($id = $start; $id -le $end; $id += $step) {
                $ids += $id
            }
        } else {
            $id = 0
            if (-not [int]::TryParse($raw, [ref] $id) -or $id -lt 0) {
                throw "Invalid KVarN layer id '$raw' in ExpectedKvarnLayers"
            }
            $ids += $id
        }
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

function Assert-MinKvarnBodyRecords([string] $text, [int] $minimum, [string] $label) {
    if ($minimum -le 0) {
        return
    }

    $maxRecords = -1
    foreach ($m in [regex]::Matches($text, "body records =\s+([0-9]+)")) {
        $records = [int] $m.Groups[1].Value
        if ($records -gt $maxRecords) {
            $maxRecords = $records
        }
    }
    if ($maxRecords -lt $minimum) {
        throw "$label observed maximum KVarN body records $maxRecords, expected at least $minimum"
    }

    Write-Host ("KVarN body-record check: PASS, max body records = {0}" -f $maxRecords)
}

function Invoke-Completion([string] $exe, [string[]] $argv, [hashtable] $envSet) {
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
    if ($text -notmatch "llama_kv_cache_kvarn:") {
        throw "llama-completion succeeded but logs did not show KVarN cache initialization"
    }
    $kvarnLayerLogs = ([regex]::Matches($text, "llama_kv_cache_kvarn: KVarN layer")).Count
    if ($kvarnLayerLogs -lt $MinKvarnLayerLogs) {
        throw "llama-completion succeeded but showed only $kvarnLayerLogs KVarN layer allocation lines, expected at least $MinKvarnLayerLogs"
    }
    Assert-ExpectedKvarnLayers $text $expectedKvarnLayerIds "llama-completion"
    Assert-MinKvarnBodyRecords $text $MinKvarnBodyRecords "llama-completion"
    Write-Host ("KVarN completion log check: PASS, KVarN layer lines = {0}" -f $kvarnLayerLogs)

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
$args = @(
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
    "-fa", "on",
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", "kvarn_k4v2_g128",
    "--kvarn-rtn-quantile", $rtnQuantileArg,
    "-s", [string] $Seed,
    "--temp", "0"
)

$baseEnv = @{}
if ($DisableGraphReuse) {
    $baseEnv["LLAMA_GRAPH_REUSE_DISABLE"] = "1"
}

Write-Host "== KVarN packed mixed attention"
$packed = Invoke-Completion $completion $args $baseEnv
$packedText = Get-GeneratedText $packed
$packedGraphs = Get-GraphsReused $packed
$packedTps = Get-EvalTokensPerSecond $packed

Write-Host "== KVarN scratch-reference mixed attention"
$scratchEnv = @{}
foreach ($key in $baseEnv.Keys) {
    $scratchEnv[$key] = $baseEnv[$key]
}
$scratchEnv["LLAMA_KVARN_ATTN_REF_SCRATCH"] = "1"
$scratch = Invoke-Completion $completion $args $scratchEnv
$scratchText = Get-GeneratedText $scratch
$scratchGraphs = Get-GraphsReused $scratch
$scratchTps = Get-EvalTokensPerSecond $scratch

if ($packedText -ne $scratchText) {
    Write-Host "== packed generated text =="
    Write-Host $packedText
    Write-Host "== scratch-reference generated text =="
    Write-Host $scratchText
    throw "KVarN scratch-reference attention changed deterministic generated text"
}

Write-Host "KVarN packed-vs-scratch reference parity: PASS"
Write-Host ("KVarN packed attention : graphs reused = {0}, eval tok/s = {1}" -f $packedGraphs, $packedTps)
Write-Host ("KVarN scratch reference: graphs reused = {0}, eval tok/s = {1}" -f $scratchGraphs, $scratchTps)
