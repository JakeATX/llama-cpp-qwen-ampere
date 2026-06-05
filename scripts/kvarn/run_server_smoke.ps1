param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-nofa-vs"),
    [int] $Port = 8137,
    [int] $Context = 256,
    [int] $GpuLayers = 99,
    [double] $RtnQuantile = 0.95,
    [string] $Prompt = "Hello",
    [int] $Predict = 1
)

$ErrorActionPreference = "Stop"

if (!($RtnQuantile -gt 0.0 -and $RtnQuantile -le 1.0)) {
    throw "RtnQuantile must be in (0, 1]"
}
if ($Predict -le 0) {
    throw "Predict must be positive"
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

$server = Get-ExePath $BuildDir "llama-server.exe"
$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$stdoutLog = Join-Path $env:TEMP "kvarn-server-smoke.out.log"
$stderrLog = Join-Path $env:TEMP "kvarn-server-smoke.err.log"
Remove-Item -LiteralPath $stdoutLog, $stderrLog -ErrorAction SilentlyContinue

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

    Write-Host ("KVarN server smoke: PASS, content = '{0}'" -f $response.content)
} finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
}
