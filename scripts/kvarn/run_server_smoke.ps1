param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [int] $Port = 8137,
    [int] $Context = 256,
    [int] $GpuLayers = 99,
    [double] $RtnQuantile = 0.95,
    [string] $Prompt = "Hello",
    [int] $Predict = 1,
    [int] $MinKvarnLayerLogs = 1,
    [string] $ExpectedKvarnLayers = "",
    [switch] $CheckSlotSaveRejection
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($Predict -le 0) {
    throw "Predict must be positive"
}
if ($MinKvarnLayerLogs -lt 0) {
    throw "MinKvarnLayerLogs must be non-negative"
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

function Join-ProcessArgs([string[]] $argv) {
    return ($argv | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"','\"') + '"'
        } else {
            $_
        }
    }) -join ' '
}

function Get-HttpErrorText([System.Management.Automation.ErrorRecord] $err) {
    $parts = @()
    if ($err.ErrorDetails -ne $null -and -not [string]::IsNullOrWhiteSpace($err.ErrorDetails.Message)) {
        $parts += $err.ErrorDetails.Message
    }

    $response = $err.Exception.Response
    if ($response -eq $null) {
        $parts += $err.ToString()
        return ($parts | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join "`n"
    }

    try {
        $stream = $response.GetResponseStream()
        if ($stream -eq $null) {
            $parts += $err.ToString()
            return ($parts | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join "`n"
        }
        $reader = [System.IO.StreamReader]::new($stream)
        try {
            $parts += $reader.ReadToEnd()
        } finally {
            $reader.Dispose()
        }
    } catch {
        $parts += $err.ToString()
    }

    $parts += $err.ToString()
    return ($parts | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join "`n"
}

$server = Get-ExePath $BuildDir "llama-server.exe"
$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers
$logId = [guid]::NewGuid().ToString("N")
$stdoutLog = Join-Path $env:TEMP ("kvarn-server-smoke-{0}.out.log" -f $logId)
$stderrLog = Join-Path $env:TEMP ("kvarn-server-smoke-{0}.err.log" -f $logId)
Remove-Item -LiteralPath $stdoutLog, $stderrLog -ErrorAction SilentlyContinue

$slotSaveDir = $null
if ($CheckSlotSaveRejection) {
    $slotSaveDir = Join-Path $env:TEMP ("kvarn-slot-save-{0}" -f ([guid]::NewGuid().ToString("N")))
    New-Item -ItemType Directory -Path $slotSaveDir | Out-Null
}

$argv = @(
    "-m", $Model,
    "--host", "127.0.0.1",
    "--port", [string] $Port,
    "-c", [string] $Context,
    "-ngl", [string] $GpuLayers,
    "--no-warmup",
    "-fa", "on",
    "--kv-cache-quant", "kvarn",
    "--kvarn-preset", "kvarn_k4v2_g128",
    "--kvarn-rtn-quantile", $rtnQuantileArg
)
if ($CheckSlotSaveRejection) {
    $argv += @("--slot-save-path", $slotSaveDir)
}

$process = Start-Process `
    -FilePath $server `
    -ArgumentList (Join-ProcessArgs $argv) `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru `
    -WindowStyle Hidden

try {
    $ready = $false
    for ($i = 0; $i -lt 90; ++$i) {
        Start-Sleep -Milliseconds 1000
        if ($process.HasExited) {
            $out = Get-Content -Raw -LiteralPath $stdoutLog -ErrorAction SilentlyContinue
            $err = Get-Content -Raw -LiteralPath $stderrLog -ErrorAction SilentlyContinue
            throw "llama-server exited early with code $($process.ExitCode)`n$out`n$err"
        }

        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($health.status -eq "ok" -or $health.status -eq "no slot available") {
                $ready = $true
                break
            }
        } catch {
        }
    }

    if (-not $ready) {
        $out = Get-Content -Raw -LiteralPath $stdoutLog -ErrorAction SilentlyContinue
        $err = Get-Content -Raw -LiteralPath $stderrLog -ErrorAction SilentlyContinue
        throw "llama-server did not become ready`n$out`n$err"
    }

    $body = @{
        prompt = $Prompt
        n_predict = $Predict
        temperature = 0
    } | ConvertTo-Json -Compress

    $response = Invoke-RestMethod `
        -Uri "http://127.0.0.1:$Port/completion" `
        -Method Post `
        -Body $body `
        -ContentType "application/json" `
        -TimeoutSec 60

    if (-not $response.content) {
        throw "completion response had empty content: $($response | ConvertTo-Json -Compress)"
    }

    $out = Get-Content -Raw -LiteralPath $stdoutLog -ErrorAction SilentlyContinue
    $err = Get-Content -Raw -LiteralPath $stderrLog -ErrorAction SilentlyContinue
    $serverLog = "$out`n$err"
    if ($serverLog -notmatch "llama_kv_cache_kvarn:") {
        throw "server log did not show KVarN cache initialization; refusing to accept possible normal-KV fallback`n$serverLog"
    }
    $kvarnLayerLogs = ([regex]::Matches($serverLog, "llama_kv_cache_kvarn: KVarN layer")).Count
    if ($kvarnLayerLogs -lt $MinKvarnLayerLogs) {
        throw "server log showed only $kvarnLayerLogs KVarN layer allocation lines, expected at least $MinKvarnLayerLogs`n$serverLog"
    }
    Assert-ExpectedKvarnLayers $serverLog $expectedKvarnLayerIds "llama-server"

    Write-Host ("KVarN server smoke: PASS, content = '{0}'" -f $response.content)
    Write-Host ("KVarN server log check: PASS, KVarN layer lines = {0}" -f $kvarnLayerLogs)

    if ($CheckSlotSaveRejection) {
        $slotBody = @{
            filename = "kvarn-slot.bin"
        } | ConvertTo-Json -Compress

        try {
            [void] (Invoke-RestMethod `
                -Uri "http://127.0.0.1:$Port/slots/0?action=save" `
                -Method Post `
                -Body $slotBody `
                -ContentType "application/json" `
                -TimeoutSec 60)
            throw "KVarN slot save unexpectedly succeeded"
        } catch {
            $text = Get-HttpErrorText $_
            if ($text -notmatch "KVarN state serialization is not implemented yet") {
                $out = Get-Content -Raw -LiteralPath $stdoutLog -ErrorAction SilentlyContinue
                $err = Get-Content -Raw -LiteralPath $stderrLog -ErrorAction SilentlyContinue
                $text = "$text`n$out`n$err"
            }
            if ($text -notmatch "KVarN state serialization is not implemented yet") {
                throw "KVarN slot save rejection did not report expected error: $text"
            }
            Write-Host "KVarN slot save rejection: PASS"
        }
    }
} finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
    if ($slotSaveDir -ne $null) {
        Remove-Item -LiteralPath $slotSaveDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
