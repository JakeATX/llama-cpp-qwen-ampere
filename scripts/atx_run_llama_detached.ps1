param(
    [Parameter(Mandatory=$true)][string]$LlamaCli,
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Policy = "",
    [Parameter(Mandatory=$true)][string]$Mode,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [int]$CtxSize = 512,
    [int]$Tokens = 8,
    [string]$Prompt = "Write a compact Python function that merges two sorted lists."
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stats = Join-Path $OutDir "$Mode`_stats.json"
$log = Join-Path $OutDir "$Mode.log"
$stdoutLog = Join-Path $OutDir "$Mode.stdout.log"
$stderrLog = Join-Path $OutDir "$Mode.stderr.log"
$exitFile = Join-Path $OutDir "$Mode.exitcode"

$llamaArgs = @(
    "-m", $Model,
    "-ngl", "all",
    "--ctx-size", "$CtxSize",
    "--seed", "1",
    "--temp", "0",
    "--no-warmup",
    "--no-display-prompt",
    "-p", $Prompt,
    "-n", "$Tokens",
    "--moe-residency-stats", $stats
)

if ($Policy -ne "") {
    $llamaArgs += @("--moe-residency-policy", $Policy)
}

if ($Mode -ne "off") {
    $llamaArgs += @("--moe-residency-mode", $Mode)
}
if ($Mode -eq "direct" -or $Mode -eq "hybrid" -or $Mode -eq "auto") {
    $llamaArgs += @("--moe-direct-require", "--moe-direct-strict-hot-no-stage")
}

$argLine = ($llamaArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join ' '
$proc = Start-Process -FilePath $LlamaCli -ArgumentList $argLine -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
(Get-Content -Raw $stdoutLog -ErrorAction SilentlyContinue) + (Get-Content -Raw $stderrLog -ErrorAction SilentlyContinue) | Set-Content -Encoding utf8 $log
$proc.ExitCode | Set-Content -Encoding ascii $exitFile
exit $proc.ExitCode
