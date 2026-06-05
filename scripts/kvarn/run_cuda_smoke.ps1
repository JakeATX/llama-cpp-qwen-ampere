param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda"),
    [string] $CtxList = "256 512 1024",
    [int] $GpuLayers = 99,
    [double] $RtnQuantile = 1.0,
    [int] $MinKvarnLayerLogs = 1,
    [int] $MinKvarnBodyRecords = 0,
    [string] $ExpectedKvarnLayers = ""
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($MinKvarnLayerLogs -lt 0) {
    throw "MinKvarnLayerLogs must be non-negative"
}
if ($MinKvarnBodyRecords -lt 0) {
    throw "MinKvarnBodyRecords must be non-negative"
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

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers

$cli = Join-Path $BuildDir "bin/Release/llama-cli.exe"
if (-not (Test-Path -LiteralPath $cli)) {
    throw "llama-cli.exe not found at $cli"
}

foreach ($ctx in $CtxList.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)) {
    Write-Host "== FP16 KV smoke: ctx=$ctx"
    & $cli -m $Model -p "Hello" -n 8 -c $ctx -ngl $GpuLayers --no-warmup --simple-io --single-turn
    if ($LASTEXITCODE -ne 0) {
        throw "FP16 KV smoke failed for ctx=$ctx"
    }

    Write-Host "== KVarN decode smoke or explicit guard: ctx=$ctx"
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $kvarnOutput = & $cli -m $Model -p "Hello" -n 1 -c $ctx -ngl $GpuLayers --no-warmup --simple-io --single-turn `
        --kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-rtn-quantile $rtnQuantileArg 2>&1
    $kvarnExit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    $kvarnOutput | Write-Host
    $kvarnText = ($kvarnOutput | ForEach-Object { $_.ToString() }) -join "`n"
    if ($kvarnExit -eq 0) {
        if ($kvarnText -notmatch "llama_kv_cache_kvarn:") {
            throw "KVarN smoke for ctx=$ctx succeeded but logs did not show KVarN cache initialization"
        }
        $kvarnLayerLogs = ([regex]::Matches($kvarnText, "llama_kv_cache_kvarn: KVarN layer")).Count
        if ($kvarnLayerLogs -lt $MinKvarnLayerLogs) {
            throw "KVarN smoke for ctx=$ctx showed only $kvarnLayerLogs KVarN layer allocation lines, expected at least $MinKvarnLayerLogs"
        }
        Assert-ExpectedKvarnLayers $kvarnText $expectedKvarnLayerIds "KVarN smoke for ctx=$ctx"
        Assert-MinKvarnBodyRecords $kvarnText $MinKvarnBodyRecords "KVarN smoke for ctx=$ctx"
        Write-Host ("KVarN CLI log check: PASS, KVarN layer lines = {0}" -f $kvarnLayerLogs)
        continue
    }
    if ($kvarnText -notmatch "KVarN backend currently supports only 128-, 256-, or 512-dimensional K/V heads|KVarN backend requires equal K and V head dimensions|KVarN backend does not support MLA models yet|KVarN backend does not support ALiBi attention bias yet|KVarN backend does not support attention logit soft-capping yet|KVarN backend does not support Grok-style attention output scaling yet|KVarN backend does not support attention sinks yet|KVarN backend supports SWA/ISWA only for Gemma 4 models at this stage|KVarN backend currently requires every KV layer to run on a backend with CUDA KVarN op support|KVarN graph backend does not yet support attention rotations|KVarN graph backend active body record count exceeds allocated cache capacity|KVarN graph backend is not wired yet|KVarN shift/update graph path is not wired yet") {
        throw "KVarN failure did not include the expected validation or graph-backend guard"
    }
}
