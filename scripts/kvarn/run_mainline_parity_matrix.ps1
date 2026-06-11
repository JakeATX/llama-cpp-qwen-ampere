param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $MainlineBuildDir = (Join-Path (Get-Location) "..\llama.cpp-mainline\build-cuda-static-vs"),
    [string] $KvarnBuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $CaseList = "tg64:0:64,pp512:512:0,pp4096:4096:0,tg4096:0:4096",
    [int] $GpuLayers = 99,
    [ValidateSet("on", "off", "auto")] [string] $FlashAttn = "off",
    [string] $KvarnPreset = "kvarn_k4v2_g128",
    [int] $KvarnIters = 4,
    [double] $RtnQuantile = 1.0,
    [int] $Repetitions = 3,
    [switch] $Warmup,
    [double] $MinParityRatio = 0.90,
    [switch] $FailBelowMinParityRatio,
    [int] $MinKvarnLayerLogs = 1,
    [int] $MinKvarnBodyRecords = 0,
    [string] $ExpectedKvarnLayers = "",
    [string] $OutputDir = "",
    [string[]] $ExtraArgs = @(),
    [string[]] $KvarnExtraArgs = @()
)

$ErrorActionPreference = "Stop"

if ($Repetitions -le 0) {
    throw "Repetitions must be positive"
}
if ($MinParityRatio -le 0.0 -or $MinParityRatio -gt 1.0) {
    throw "MinParityRatio must be in (0, 1]"
}
if ($MinKvarnLayerLogs -lt 0) {
    throw "MinKvarnLayerLogs must be non-negative"
}
if ($MinKvarnBodyRecords -lt 0) {
    throw "MinKvarnBodyRecords must be non-negative"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found at $Model"
}

function Resolve-BuildExe([string] $BuildDir, [string] $Name) {
    $candidates = @(
        (Join-Path $BuildDir "bin/Release/$Name"),
        (Join-Path $BuildDir "bin/$Name"),
        (Join-Path $BuildDir "bin/Release/$Name.exe"),
        (Join-Path $BuildDir "bin/$Name.exe")
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    throw "Missing $Name under $BuildDir"
}

function Get-GitHead([string] $RepoRoot) {
    if (-not (Test-Path -LiteralPath $RepoRoot)) {
        return "unknown"
    }
    Push-Location $RepoRoot
    try {
        return (git rev-parse --short HEAD 2>$null)
    } finally {
        Pop-Location
    }
}

function Get-BenchCases([string] $rawList) {
    $cases = @()
    foreach ($raw in ($rawList -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $parts = $raw -split ":"
        if ($parts.Count -ne 3) {
            throw "Invalid case '$raw'; expected name:prompt:gen"
        }
        $cases += [pscustomobject]@{
            Name = $parts[0]
            PromptTokens = [int] $parts[1]
            GenTokens = [int] $parts[2]
        }
    }
    return $cases
}

function Convert-ToFileStem([string] $name) {
    return ($name -replace '[^A-Za-z0-9_.-]+', '_')
}

function Get-BenchThroughput([string] $text) {
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -notmatch '^\|') {
            continue
        }
        $cols = $line.Split('|') | ForEach-Object { $_.Trim() }
        if ($cols.Count -lt 8) {
            continue
        }

        $nonEmptyCols = @($cols | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($nonEmptyCols.Count -lt 2) {
            continue
        }
        $throughputText = $nonEmptyCols[$nonEmptyCols.Count - 1]
        if ($throughputText -notmatch '^[0-9]') {
            continue
        }
        $m = [regex]::Match($throughputText, '([0-9]+(?:\.[0-9]+)?)')
        if ($m.Success) {
            return [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    return [double]::NaN
}

function Get-CudaDeviceSummary([string] $text) {
    $devices = @()
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -match '^\s*Device\s+[0-9]+:\s+(.+)$') {
            $devices += $Matches[1].Trim()
        }
    }
    return ($devices -join "; ")
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

function Invoke-BenchRow {
    param(
        [string] $Label,
        [string] $BenchExe,
        [string] $ModelPath,
        [object] $Case,
        [string[]] $Argv
    )

    $stem = Convert-ToFileStem $Case.Name
    $logPath = Join-Path $OutputDir "$stem.$Label.md.txt"
    $cmdPath = Join-Path $OutputDir "$stem.$Label.command.txt"
    $commandLine = "`"$BenchExe`" " + (($Argv | ForEach-Object {
        if ($_ -match '\s') { "`"$_`"" } else { $_ }
    }) -join " ")
    [System.IO.File]::WriteAllText($cmdPath, $commandLine + "`n")

    Write-Host "== $Label case $($Case.Name) p=$($Case.PromptTokens) n=$($Case.GenTokens)"
    Write-Host $commandLine

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $BenchExe @Argv 2>&1
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }

    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    [System.IO.File]::WriteAllText($logPath, $text + "`n")
    $text | Write-Host
    if ($exit -ne 0) {
        throw "$Label llama-bench failed for case '$($Case.Name)' with exit code $exit; see $logPath"
    }

    $tps = Get-BenchThroughput $text
    if ([double]::IsNaN($tps)) {
        throw "$Label llama-bench case '$($Case.Name)' did not report throughput; see $logPath"
    }
    return [pscustomobject]@{
        Throughput = $tps
        Log = (Split-Path -Leaf $logPath)
        Text = $text
        CudaDevice = Get-CudaDeviceSummary $text
    }
}

$repoRoot = (Get-Location).Path
$mainlineRoot = (Resolve-Path -LiteralPath (Join-Path $repoRoot "..\llama.cpp-mainline")).Path
$mainlineBench = Resolve-BuildExe $MainlineBuildDir "llama-bench.exe"
$kvarnBench = Resolve-BuildExe $KvarnBuildDir "llama-bench.exe"
$modelPath = (Resolve-Path -LiteralPath $Model).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $repoRoot "artifacts/kvarn-mainline-parity/$stamp"
}
[void] [System.IO.Directory]::CreateDirectory($OutputDir)
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$rtnQuantileArg = $RtnQuantile.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$warmupArgv = if ($Warmup.IsPresent) { @() } else { @("--no-warmup") }
$expectedKvarnLayerIds = Get-ExpectedKvarnLayerIds $ExpectedKvarnLayers

$manifest = @(
    "model=$modelPath",
    "mainline_repo=$mainlineRoot",
    "mainline_head=$(Get-GitHead $mainlineRoot)",
    "kvarn_repo=$repoRoot",
    "kvarn_head=$(Get-GitHead $repoRoot)",
    "mainline_build=$MainlineBuildDir",
    "kvarn_build=$KvarnBuildDir",
    "case_list=$CaseList",
    "min_parity_ratio=$MinParityRatio",
    "repetitions=$Repetitions",
    "flash_attn=$FlashAttn",
    "kvarn_preset=$KvarnPreset",
    "kvarn_iters=$KvarnIters",
    "kvarn_rtn_quantile=$rtnQuantileArg",
    "min_kvarn_layer_logs=$MinKvarnLayerLogs",
    "min_kvarn_body_records=$MinKvarnBodyRecords",
    "expected_kvarn_layers=$ExpectedKvarnLayers",
    "extra_args=$($ExtraArgs -join ' ')",
    "kvarn_extra_args=$($KvarnExtraArgs -join ' ')"
)
[System.IO.File]::WriteAllText((Join-Path $OutputDir "manifest.txt"), ($manifest -join "`n") + "`n")

$summaries = @()
$gateFailures = @()

foreach ($case in (Get-BenchCases $CaseList)) {
    $commonArgv = @(
        "-m", $modelPath,
        "-p", [string] $case.PromptTokens,
        "-n", [string] $case.GenTokens,
        "-r", [string] $Repetitions,
        "-ngl", [string] $GpuLayers,
        "-fa", $FlashAttn,
        "-o", "md"
    ) + $warmupArgv + $ExtraArgs

    $mainlineRow = Invoke-BenchRow -Label "mainline" -BenchExe $mainlineBench -ModelPath $modelPath -Case $case -Argv $commonArgv

    $kvarnArgv = $commonArgv + @(
        "--kv-cache-quant", "kvarn",
        "--kvarn-preset", $KvarnPreset,
        "--kvarn-iters", [string] $KvarnIters,
        "--kvarn-rtn-quantile", $rtnQuantileArg
    ) + $KvarnExtraArgs
    $kvarnRow = Invoke-BenchRow -Label "kvarn" -BenchExe $kvarnBench -ModelPath $modelPath -Case $case -Argv $kvarnArgv

    if ($MinKvarnLayerLogs -gt 0) {
        $kvarnLayerLogs = ([regex]::Matches($kvarnRow.Text, "llama_kv_cache_kvarn: KVarN layer")).Count
        if ($kvarnLayerLogs -lt $MinKvarnLayerLogs) {
            throw "KVarN case '$($case.Name)' showed only $kvarnLayerLogs layer lines, expected >= $MinKvarnLayerLogs"
        }
        Write-Host ("KVarN layer-log check: PASS, layer lines = {0}" -f $kvarnLayerLogs)
    }
    if ($kvarnRow.Text -notmatch "llama_kv_cache_kvarn:") {
        throw "KVarN case '$($case.Name)' succeeded but logs did not show KVarN cache initialization"
    }
    Assert-ExpectedKvarnLayers $kvarnRow.Text $expectedKvarnLayerIds "KVarN case '$($case.Name)'"
    Assert-MinKvarnBodyRecords $kvarnRow.Text $MinKvarnBodyRecords "KVarN case '$($case.Name)'"

    $ratio = $kvarnRow.Throughput / $mainlineRow.Throughput
    $gatePass = $ratio -ge $MinParityRatio
    if ($FailBelowMinParityRatio.IsPresent -and -not $gatePass) {
        $msg = "case '$($case.Name)' failed mainline parity gate: ratio={0:P1}, threshold={1:P1}" -f $ratio, $MinParityRatio
        $gateFailures += $msg
        Write-Warning $msg
    }

    Write-Host ("Mainline parity ratio: {0} = {1:P1} (kvarn {2:F2} / mainline {3:F2} t/s) [{4}]" -f `
        $case.Name, $ratio, $kvarnRow.Throughput, $mainlineRow.Throughput, ($(if ($gatePass) { "PASS" } else { "FAIL" })))

    $summaries += [pscustomobject]@{
        Case = $case.Name
        PromptTokens = $case.PromptTokens
        GenTokens = $case.GenTokens
        MainlineTps = $mainlineRow.Throughput
        KvarnTps = $kvarnRow.Throughput
        KvarnVsMainline = $ratio
        GateThreshold = $MinParityRatio
        GatePass = $gatePass
        CudaDevice = if (-not [string]::IsNullOrWhiteSpace($kvarnRow.CudaDevice)) { $kvarnRow.CudaDevice } else { $mainlineRow.CudaDevice }
        MainlineLog = $mainlineRow.Log
        KvarnLog = $kvarnRow.Log
    }
}

$summaryCsv = Join-Path $OutputDir "summary.csv"
$summaryMd = Join-Path $OutputDir "summary.md"
$summaries | Export-Csv -NoTypeInformation -LiteralPath $summaryCsv

$summaryLines = @(
    "# KVarN vs mainline llama.cpp parity",
    "",
    "| case | prompt | gen | mainline t/s | KVarN t/s | KVarN/mainline | gate (>=$([int]($MinParityRatio*100))%) |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
)
foreach ($s in $summaries) {
    $gateText = if ($s.GatePass) { "PASS" } else { "**FAIL**" }
    $summaryLines += "| {0} | {1} | {2} | {3:F2} | {4:F2} | {5:P1} | {6} |" -f `
        $s.Case, $s.PromptTokens, $s.GenTokens, $s.MainlineTps, $s.KvarnTps, $s.KvarnVsMainline, $gateText
}
[System.IO.File]::WriteAllText($summaryMd, ($summaryLines -join "`n") + "`n")

Write-Host "Mainline parity summary: $summaryMd"
Write-Host "Mainline parity matrix complete: $OutputDir"

if ($gateFailures.Count -gt 0) {
    throw ("Mainline parity gate failures:`n" + ($gateFailures -join "`n"))
}
