<#
.SYNOPSIS
    App-safe serialized KVarN validation runner.

.DESCRIPTION
    This wrapper exists for long Codex/app-hosted runs. It does not stream
    model output to the app. Each step runs as one hidden child PowerShell
    process with stdout/stderr redirected to files, a timeout, a stale-process
    preflight, and a JSON state file for resume.

    The inner KVarN scripts still own the correctness and speed metrics. This
    script only supervises them so full 16K/context and parity gates cannot run
    concurrently or flood the host UI.
#>
param(
    [string] $QwenModel = "",
    [string] $GemmaModel = "",
    [string] $Dataset = "",
    [string] $QwenDataset = "",
    [string] $GemmaDataset = "",
    [string] $BuildDir = (Join-Path (Get-Location) "build-kvarn-cuda-static-vs"),
    [string] $MainlineBuildDir = (Join-Path (Get-Location) "..\llama.cpp-mainline\build-cuda-static-vs"),
    [string] $OutputDir = "",
    [ValidateSet("all", "build", "unit", "accuracy", "speed")]
    [string[]] $Stage = @("all"),
    [string[]] $AccuracyPresets = @("kvarn_k4v4_g128", "kvarn_k8v8_g128"),
    [string[]] $SpeedPresets = @("kvarn_k4v4_g128", "kvarn_k8v8_g128"),
    [string[]] $ExperimentalAccuracyPresets = @("kvarn_k4v2_g128", "kvarn_k8v2_g128"),
    [string] $SpeedCaseList = "pp512:512:0,tg64:0:64",
    [string] $SpeedEvidenceCaseList = "pp512:512:0",
    [int] $ContextSize = 16384,
    [int] $BatchSize = 512,
    [int] $Chunks = 1,
    [int] $KvarnIters = 4,
    [double] $KvarnRtnQuantile = 1.0,
    [double] $MaxPplIncrease = 0.01,
    [double] $MaxMeanKL = 0.02,
    [double] $MaxKLD99 = 1.0,
    [double] $MaxKLD999 = 1.0,
    [double] $MaxKLDMax = 1.0,
    [double] $MaxBaselinePpl = 100.0,
    [int] $MinKvarnBodyRecords = 1,
    [int] $MinActiveKvarnBodyRecords = 0,
    [double] $Tier1MinRatio = 0.95,
    [string] $GemmaExpectedKvarnLayers = "5-47:6",
    [string] $QwenLayerFilter = "",
    [string] $GemmaLayerFilter = "",
    [string] $QwenLayerKeyBits = "",
    [string] $QwenLayerValueBits = "",
    [string] $GemmaLayerKeyBits = "",
    [string] $GemmaLayerValueBits = "",
    [string] $QwenExpectedEffectiveKvarnBits = "auto",
    [string] $GemmaExpectedEffectiveKvarnBits = "auto",
    [string] $GemmaProtectedDonors = "",
    [int] $GpuLayers = 99,
    [int] $QwenRepetitions = 3,
    [int] $GemmaRepetitions = 3,
    [int] $BuildTimeoutSec = 3600,
    [int] $UnitTimeoutSec = 1200,
    [int] $AccuracyTimeoutSec = 14400,
    [int] $SpeedTimeoutSec = 7200,
    [int] $MinFreeVramMiB = 8192,
    [int] $BuildJobs = 1,
    [switch] $DryRun,
    [switch] $PreflightOnly,
    [switch] $Resume,
    [switch] $RerunAll,
    [switch] $AllowExistingLlama,
    [switch] $AllowExistingGpuCompute,
    [switch] $SkipBuild,
    [switch] $SkipUnit,
    [switch] $SkipQwenAccuracy,
    [switch] $SkipGemmaAccuracy,
    [switch] $SkipQwenSpeed,
    [switch] $SkipGemmaSpeed,
    [switch] $RunExperimentalAccuracy,
    [switch] $SkipSpeedTimedParity,
    [switch] $SkipSpeedEvidence,
    [switch] $ParseSpecial,
    [switch] $QwenParseSpecial,
    [switch] $GemmaParseSpecial,
    [switch] $QwenAllowChatMarkers,
    [switch] $GemmaAllowChatMarkers,
    [switch] $UseKLDivergence,
    [switch] $GemmaUseKLDivergence,
    [switch] $DisableKLDivergence,
    [switch] $AllowDiagnosticAccuracyEnv,
    [switch] $GemmaRouteAllFull,
    [switch] $GemmaRouteConservative,
    [switch] $GemmaNormalFallbackDiagnostic,
    [switch] $TraceAttn,
    [int] $TraceLimit = 64,
    [switch] $TraceFwht,
    [int] $MinFwhtTaken = 0,
    [int] $MinBatchedStorePhaseUses = 1,
    [switch] $AttnRefScratch,
    [switch] $AttnEnableBodyF32Mirror,
    [switch] $EnableF32DequantCache,
    [switch] $AttnDisableBodyF32Mirror,
    [switch] $DebugRawBodyK,
    [switch] $DebugRawBodyV,
    [switch] $DisablePrefillDirectAttn,
    [switch] $DisablePrefillDirectStore,
    [switch] $DisableInternalCausalMask,
    [switch] $DisablePrefillPingPong,
    [switch] $IncludeBlockedMixedTailTest,
    [switch] $PaperMixedFrame,
    [switch] $DisablePaperFrame,
    [switch] $EnableGemmaPaperFrame,
    [switch] $SkipFixtureCheck
)

$ErrorActionPreference = "Stop"

$script:KvarnGateMutex = [System.Threading.Mutex]::new($false, "Global\llama_cpp_kvarn_safe_full_gate")
$script:KvarnGateMutexHeld = $script:KvarnGateMutex.WaitOne(0)
if (-not $script:KvarnGateMutexHeld) {
    throw "Another KVarN safe full gate is already running. Refusing to start a concurrent model/test job."
}

if ($GemmaRouteAllFull.IsPresent) {
    Write-Warning "-GemmaRouteAllFull is now the default Gemma4 route; the switch is kept for compatibility and has no effect."
}

if ($MinFwhtTaken -lt 0) {
    throw "MinFwhtTaken must be non-negative"
}
if ($MinBatchedStorePhaseUses -lt 0) {
    throw "MinBatchedStorePhaseUses must be non-negative"
}

function Convert-ToFileStem([string] $Name) {
    return ($Name -replace '[^A-Za-z0-9_.-]+', '_').Trim('_')
}

function Resolve-ExpectedEffectiveBitsForHighGqa([string] $Preset, [string] $Override, [string] $LayerKeyBits, [string] $LayerValueBits) {
    if ($Override -ne "auto") {
        return $Override
    }
    if (-not [string]::IsNullOrWhiteSpace($LayerKeyBits) -or -not [string]::IsNullOrWhiteSpace($LayerValueBits)) {
        return ""
    }
    $m = [regex]::Match($Preset, "(?i)kvarn_k([0-9]+)v([0-9]+)")
    if (-not $m.Success) {
        return ""
    }
    return ("k8/v{0}" -f ([int] $m.Groups[2].Value))
}

function Convert-ToPowerShellLiteral([string] $Value) {
    return "'" + ($Value -replace "'", "''") + "'"
}

function Convert-ToPowerShellArrayLiteral([string[]] $Values) {
    if ($null -eq $Values -or $Values.Count -eq 0) {
        return "@()"
    }
    return "@(" + (($Values | ForEach-Object { Convert-ToPowerShellLiteral $_ }) -join ", ") + ")"
}

function Convert-ToPowerShellValueLiteral($Value) {
    if ($null -eq $Value) {
        return "`$null"
    }
    if ($Value -is [bool]) {
        return $(if ($Value) { "`$true" } else { "`$false" })
    }
    if ($Value -is [System.Array]) {
        return Convert-ToPowerShellArrayLiteral ([string[]] $Value)
    }
    return Convert-ToPowerShellLiteral ([string] $Value)
}

function Test-Stage([string] $Name) {
    return ($Stage -contains "all") -or ($Stage -contains $Name)
}

function Expand-List([string[]] $Values) {
    $expanded = @()
    foreach ($value in @($Values)) {
        foreach ($part in ([string] $value -split ",")) {
            $trimmed = $part.Trim()
            if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
                $expanded += $trimmed
            }
        }
    }
    return $expanded
}

function Add-KvarnDiagnosticEnv([hashtable] $Env) {
    if ($TraceAttn.IsPresent) {
        $Env["LLAMA_KVARN_ATTN_TRACE"] = "1"
        $Env["LLAMA_KVARN_ATTN_TRACE_LIMIT"] = [string] $TraceLimit
    }
    if ($AttnRefScratch.IsPresent) { $Env["LLAMA_KVARN_ATTN_REF_SCRATCH"] = "1" }
    if ($AttnEnableBodyF32Mirror.IsPresent) { $Env["LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR"] = "1" }
    if ($EnableF32DequantCache.IsPresent) { $Env["LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE"] = "1" }
    if ($AttnDisableBodyF32Mirror.IsPresent) { $Env["LLAMA_KVARN_ATTN_DISABLE_BODY_F32_MIRROR"] = "1" }
    if ($DebugRawBodyK.IsPresent) { $Env["LLAMA_KVARN_DEBUG_RAW_BODY_K"] = "1" }
    if ($DebugRawBodyV.IsPresent) { $Env["LLAMA_KVARN_DEBUG_RAW_BODY_V"] = "1" }
    if ($DisablePrefillDirectAttn.IsPresent) { $Env["LLAMA_KVARN_DISABLE_PREFILL_DIRECT_ATTN"] = "1" }
    if ($DisablePrefillDirectStore.IsPresent) { $Env["LLAMA_KVARN_DISABLE_PREFILL_DIRECT_STORE"] = "1" }
    if ($DisableInternalCausalMask.IsPresent) { $Env["LLAMA_KVARN_DISABLE_INTERNAL_CAUSAL_MASK"] = "1" }
    if ($DisablePrefillPingPong.IsPresent) { $Env["LLAMA_KVARN_DISABLE_PREFILL_PINGPONG"] = "1" }
    if ($PaperMixedFrame.IsPresent) {
        $Env["LLAMA_KVARN_PAPER_MIXED_FRAME"] = "1"
        $Env["LLAMA_KVARN_UNSAFE_ALLOW_PAPER_MIXED_FRAME"] = "1"
    }
}

function Resolve-RequiredPath([string] $Path, [string] $Label) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "$Label is required for selected stages"
    }
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found at $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Add-Step {
    param(
        [System.Collections.Generic.List[object]] $Steps,
        [string] $Id,
        [string] $Kind,
        [string[]] $Lines,
        [int] $TimeoutSec
    )

    $commandText = ($Lines -join "`n") + "`n"
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($commandText)
    $hashBytes = $sha.ComputeHash($bytes)
    $hash = -join ($hashBytes | ForEach-Object { $_.ToString("x2") })

    $Steps.Add([pscustomobject]@{
        id          = $Id
        kind        = $Kind
        lines       = $Lines
        timeout_sec = $TimeoutSec
        command_sha256 = $hash
    })
}

function Get-ProcessTreeIds([int] $RootPid) {
    $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
    $ids = New-Object 'System.Collections.Generic.List[int]'
    $queue = New-Object 'System.Collections.Generic.Queue[int]'
    $queue.Enqueue($RootPid)
    while ($queue.Count -gt 0) {
        $curPid = $queue.Dequeue()
        if (-not $ids.Contains($curPid)) {
            $ids.Add($curPid)
            foreach ($child in ($all | Where-Object { $_.ParentProcessId -eq $curPid })) {
                $queue.Enqueue([int] $child.ProcessId)
            }
        }
    }
    return @($ids.ToArray() | Sort-Object -Descending)
}

function Stop-ProcessTree([int] $RootPid) {
    foreach ($treePid in (Get-ProcessTreeIds $RootPid)) {
        Stop-Process -Id $treePid -Force -ErrorAction SilentlyContinue
    }
}

function Get-GpuSnapshot {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return [pscustomobject]@{
            available = $false
            summary   = "nvidia-smi unavailable"
            free_mib  = -1
            compute   = @()
        }
    }

    $gpuRows = & $nvidiaSmi.Path --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>$null
    $computeRows = & $nvidiaSmi.Path --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
    $freeValues = @()
    foreach ($row in @($gpuRows)) {
        $parts = $row.ToString().Split(",") | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 5) {
            $free = 0
            if ([int]::TryParse($parts[4], [ref] $free)) {
                $freeValues += $free
            }
        }
    }
    $compute = @()
    foreach ($row in @($computeRows)) {
        $text = $row.ToString().Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }
        $parts = $text.Split(",") | ForEach-Object { $_.Trim() }
        $used = 0
        if ($parts.Count -ge 3 -and [int]::TryParse($parts[2], [ref] $used) -and $used -gt 0) {
            $compute += $text
        }
    }
    return [pscustomobject]@{
        available = $true
        summary   = (($gpuRows | ForEach-Object { $_.ToString().Trim() }) -join "; ")
        free_mib  = $(if ($freeValues.Count -gt 0) { ($freeValues | Measure-Object -Maximum).Maximum } else { -1 })
        compute   = $compute
    }
}

function Invoke-Preflight {
    $blockedNames = @(
        "llama-cli",
        "llama-bench",
        "llama-results",
        "llama-perplexity",
        "llama-server",
        "llama-tokenize"
    )
    $running = @(Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $blockedNames -contains $_.ProcessName } |
        Select-Object Id, ProcessName, Path)
    if ($running.Count -gt 0 -and -not $AllowExistingLlama.IsPresent) {
        $details = ($running | ForEach-Object { "{0} pid={1} {2}" -f $_.ProcessName, $_.Id, $_.Path }) -join "`n"
        throw "Refusing to start while llama processes are already running. Re-run with -AllowExistingLlama only if this is intentional.`n$details"
    }

    $gpu = Get-GpuSnapshot
    if ($gpu.available) {
        if ($gpu.free_mib -ge 0 -and $gpu.free_mib -lt $MinFreeVramMiB) {
            throw "Refusing to start: free VRAM $($gpu.free_mib) MiB is below -MinFreeVramMiB $MinFreeVramMiB. GPU: $($gpu.summary)"
        }
        if ($gpu.compute.Count -gt 0 -and -not $AllowExistingGpuCompute.IsPresent) {
            throw ("Refusing to start while nvidia-smi reports existing compute apps:`n" + ($gpu.compute -join "`n"))
        }
    }
    return $gpu
}

function Read-State([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return @{}
    }
    $raw = Get-Content -LiteralPath $Path -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return @{}
    }
    $stateObj = $raw | ConvertFrom-Json
    $state = @{}
    foreach ($step in @($stateObj.steps)) {
        $state[$step.id] = $step
    }
    return $state
}

function Write-State([string] $Path, [object[]] $Rows) {
    $payload = [pscustomobject]@{
        updated_at = (Get-Date).ToString("o")
        steps      = $Rows
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-SafeStep {
    param(
        [object] $Step,
        [string] $StepDir,
        [string] $StatePath,
        [System.Collections.Generic.List[object]] $StateRows
    )

    New-Item -ItemType Directory -Force -Path $StepDir | Out-Null
    $runnerPath = Join-Path $StepDir "run.ps1"
    $launcherPath = Join-Path $StepDir "launch.ps1"
    $stdoutPath = Join-Path $StepDir "stdout.log.txt"
    $stderrPath = Join-Path $StepDir "stderr.log.txt"
    $commandPath = Join-Path $StepDir "command.txt"

    $scriptLines = @(
        "`$ErrorActionPreference = 'Stop'",
        "Set-Location $(Convert-ToPowerShellLiteral (Get-Location).Path)"
    ) + $Step.lines + @(
        "exit 0"
    )
    [System.IO.File]::WriteAllText($runnerPath, ($scriptLines -join "`n") + "`n")
    $launcherLines = @(
        "`$ErrorActionPreference = 'Continue'",
        "& $(Convert-ToPowerShellLiteral $runnerPath) 1> $(Convert-ToPowerShellLiteral $stdoutPath) 2> $(Convert-ToPowerShellLiteral $stderrPath)",
        "if (`$LASTEXITCODE -ne `$null) { exit `$LASTEXITCODE }",
        "if (-not `$?) { exit 1 }",
        "exit 0"
    )
    [System.IO.File]::WriteAllText($launcherPath, ($launcherLines -join "`n") + "`n")
    [System.IO.File]::WriteAllText($commandPath, ($Step.lines -join "`n") + "`n")
    [System.IO.File]::WriteAllText($stdoutPath, "")
    [System.IO.File]::WriteAllText($stderrPath, "")

    $row = [pscustomobject]@{
        id          = $Step.id
        kind        = $Step.kind
        status      = $(if ($DryRun.IsPresent) { "DRY_RUN" } else { "RUNNING" })
        start_time  = (Get-Date).ToString("o")
        end_time    = ""
        elapsed_sec = 0
        exit_code   = $null
        timeout_sec = $Step.timeout_sec
        command     = $commandPath
        command_sha256 = $Step.command_sha256
        stdout      = $stdoutPath
        stderr      = $stderrPath
    }
    $StateRows.Add($row)
    Write-State $StatePath $StateRows.ToArray()

    if ($DryRun.IsPresent) {
        Write-Host ("DRY-RUN {0}: {1}" -f $Step.kind, $Step.id)
        return
    }

    Write-Host ("RUN {0}: {1}" -f $Step.kind, $Step.id)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = "powershell"
    $escapedLauncherPath = $launcherPath -replace '"', '\"'
    $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$escapedLauncherPath`""
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $completed = $proc.WaitForExit($Step.timeout_sec * 1000)

    if (-not $completed) {
        $treeIds = @(Get-ProcessTreeIds $proc.Id)
        Stop-ProcessTree $proc.Id
        for ($sweep = 0; $sweep -lt 5; ++$sweep) {
            Start-Sleep -Milliseconds 500
            foreach ($treePid in $treeIds) {
                Stop-Process -Id $treePid -Force -ErrorAction SilentlyContinue
            }
        }
        $proc.WaitForExit(10000) | Out-Null
        $row.status = "TIMEOUT"
        $row.exit_code = $null
    } else {
        $proc.Refresh()
        $row.exit_code = $proc.ExitCode
        $row.status = $(if ($proc.ExitCode -eq 0) { "PASS" } else { "FAIL" })
    }
    $sw.Stop()

    $row.end_time = (Get-Date).ToString("o")
    $row.elapsed_sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)
    Write-State $StatePath $StateRows.ToArray()

    Write-Host ("{0} {1}: elapsed={2}s exit={3}" -f $row.status, $Step.id, $row.elapsed_sec, $row.exit_code)
    if ($row.status -ne "PASS") {
        $tail = @()
        if (Test-Path -LiteralPath $stderrPath) {
            $tail += Get-Content -LiteralPath $stderrPath -Tail 40 -ErrorAction SilentlyContinue
        }
        if ($tail.Count -eq 0 -and (Test-Path -LiteralPath $stdoutPath)) {
            $tail += Get-Content -LiteralPath $stdoutPath -Tail 40 -ErrorAction SilentlyContinue
        }
        if ($tail.Count -gt 0) {
            Write-Host "---- last output lines ----"
            $tail | Write-Host
            Write-Host "---------------------------"
        }
        throw "Step $($Step.id) ended with status $($row.status); see $StepDir"
    }
}

function New-EnvLines([hashtable] $EnvSet) {
    $lines = @()
    foreach ($key in ($EnvSet.Keys | Sort-Object)) {
        $value = $EnvSet[$key]
        if ($null -eq $value) {
            $lines += "[Environment]::SetEnvironmentVariable($(Convert-ToPowerShellLiteral $key), `$null, 'Process')"
        } else {
            $lines += "[Environment]::SetEnvironmentVariable($(Convert-ToPowerShellLiteral $key), $(Convert-ToPowerShellLiteral ([string] $value)), 'Process')"
        }
    }
    return $lines
}

function New-InvokeScriptLines([string] $Script, [string[]] $Argv, [hashtable] $EnvSet = @{}) {
    $argLit = Convert-ToPowerShellArrayLiteral $Argv
    return @((New-EnvLines $EnvSet)) + @(
        "`$argv = $argLit",
        "& $(Convert-ToPowerShellLiteral $Script) @argv",
        "if (-not `$?) { exit 1 }"
    )
}

function New-InvokeScriptSplatLines([string] $Script, [hashtable] $Params, [hashtable] $EnvSet = @{}) {
    $lines = @((New-EnvLines $EnvSet))
    $lines += "`$params = @{}"
    foreach ($key in ($Params.Keys | Sort-Object)) {
        $lines += "`$params[$(Convert-ToPowerShellLiteral $key)] = $(Convert-ToPowerShellValueLiteral $Params[$key])"
    }
    $lines += "& $(Convert-ToPowerShellLiteral $Script) @params"
    $lines += "if (-not `$?) { exit 1 }"
    return $lines
}

function New-DirectCommandLines([string] $Exe, [string[]] $Argv) {
    $argLit = Convert-ToPowerShellArrayLiteral $Argv
    return @(
        "`$argv = $argLit",
        "& $(Convert-ToPowerShellLiteral $Exe) @argv",
        "if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }"
    )
}

function Test-PythonModule([string] $Exe, [string[]] $Prefix, [string] $Module) {
    $oldPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Prefix -c "import $Module" *> $null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $oldPref
    }
}

function Test-ExecutableStarts([string] $Exe, [string] $Label) {
    if (-not (Test-Path -LiteralPath $Exe)) {
        throw "$Label executable not found at $Exe"
    }
    $tmpDir = Join-Path $OutputDir "preflight"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
    $outPath = Join-Path $tmpDir ("{0}.stdout.txt" -f (Convert-ToFileStem $Label))
    $errPath = Join-Path $tmpDir ("{0}.stderr.txt" -f (Convert-ToFileStem $Label))

    try {
        $proc = Start-Process -FilePath $Exe -ArgumentList "--help" `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $outPath `
            -RedirectStandardError $errPath
        if ($null -eq $proc) {
            throw "$Label executable did not return a process handle: $Exe"
        }
        if (-not $proc.WaitForExit(30000)) {
            Stop-ProcessTree $proc.Id
            throw "$Label executable launch preflight timed out: $Exe"
        }
        $exitCode = $proc.ExitCode
    } catch {
        throw "$Label executable is blocked or failed to start: $Exe`n$($_.Exception.Message)"
    }
    $stdoutText = if (Test-Path -LiteralPath $outPath) { Get-Content -LiteralPath $outPath -Raw -ErrorAction SilentlyContinue } else { "" }
    $stderrText = if (Test-Path -LiteralPath $errPath) { Get-Content -LiteralPath $errPath -Raw -ErrorAction SilentlyContinue } else { "" }
    $combined = "$stderrText`n$stdoutText"
    if ($combined -match "Application Control|Windows Defender Application Control|blocked by group policy|This program is blocked") {
        throw "$Label executable is blocked by Windows Application Control or group policy: $Exe`nstdout: $outPath`nstderr: $errPath"
    }
    if ($exitCode -ne 0 -and [string]::IsNullOrWhiteSpace($combined)) {
        throw "$Label executable exited during launch preflight with code $exitCode and no help output: $Exe"
    }
}

function New-ExecutablePreflightLines([hashtable] $Executables) {
    $lines = @(
        'function Test-ExecutableStartsInStep([string] $Exe, [string] $Label) {',
        '    if (-not (Test-Path -LiteralPath $Exe)) { throw "$Label executable not found at $Exe" }',
        '    $preflightDir = Join-Path (Get-Location) ''artifacts/kvarn-safe-full-gate/preflight-step''',
        '    New-Item -ItemType Directory -Force -Path $preflightDir | Out-Null',
        '    $stem = ($Label -replace ''[^A-Za-z0-9_.-]+'', ''_'').Trim(''_'')',
        '    $outPath = Join-Path $preflightDir ($stem + ''.stdout.txt'')',
        '    $errPath = Join-Path $preflightDir ($stem + ''.stderr.txt'')',
        '    try {',
        '        $proc = Start-Process -FilePath $Exe -ArgumentList ''--help'' -WindowStyle Hidden -PassThru -RedirectStandardOutput $outPath -RedirectStandardError $errPath',
        '        if ($null -eq $proc) {',
        '            throw "$Label executable did not return a process handle: $Exe"',
        '        }',
        '        if (-not $proc.WaitForExit(30000)) {',
        '            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue',
        '            throw "$Label executable launch preflight timed out: $Exe"',
        '        }',
        '        $exitCode = $proc.ExitCode',
        '    } catch {',
        '        throw "$Label executable is blocked or failed to start: $Exe`n$($_.Exception.Message)"',
        '    }',
        '    $stdoutText = if (Test-Path -LiteralPath $outPath) { Get-Content -LiteralPath $outPath -Raw -ErrorAction SilentlyContinue } else { "" }',
        '    $stderrText = if (Test-Path -LiteralPath $errPath) { Get-Content -LiteralPath $errPath -Raw -ErrorAction SilentlyContinue } else { "" }',
        '    $combined = "$stderrText`n$stdoutText"',
        '    if ($combined -match ''Application Control|Windows Defender Application Control|blocked by group policy|This program is blocked'') {',
        '        throw "$Label executable is blocked by Windows Application Control or group policy: $Exe`nstdout: $outPath`nstderr: $errPath"',
        '    }',
        '    if ($exitCode -ne 0 -and [string]::IsNullOrWhiteSpace($combined)) {',
        '        throw "$Label executable exited during launch preflight with code $exitCode and no help output: $Exe"',
        '    }',
        '}'
    )
    foreach ($label in ($Executables.Keys | Sort-Object)) {
        $lines += "Test-ExecutableStartsInStep $(Convert-ToPowerShellLiteral ([string] $Executables[$label])) $(Convert-ToPowerShellLiteral ([string] $label))"
    }
    return $lines
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot
$Stage = @(Expand-List $Stage)
$AccuracyPresets = @(Expand-List $AccuracyPresets)
$SpeedPresets = @(Expand-List $SpeedPresets)
$ExperimentalAccuracyPresets = @(Expand-List $ExperimentalAccuracyPresets)

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $repoRoot "artifacts/kvarn-safe-full-gate/$stamp"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$statePath = Join-Path $OutputDir "state.json"

$needAccuracy = Test-Stage "accuracy"
$needSpeed = Test-Stage "speed"
$willRunQwenAccuracy = $needAccuracy -and -not $SkipQwenAccuracy.IsPresent
$willRunGemmaAccuracy = $needAccuracy -and -not $SkipGemmaAccuracy.IsPresent
$willRunAnyAccuracy = $willRunQwenAccuracy -or $willRunGemmaAccuracy
$willRunQwenSpeed = $needSpeed -and -not $SkipQwenSpeed.IsPresent
$willRunGemmaSpeed = $needSpeed -and -not $SkipGemmaSpeed.IsPresent
$willRunAnySpeed = $willRunQwenSpeed -or $willRunGemmaSpeed
$willRunSpeedTimedParity = $willRunAnySpeed -and -not $SkipSpeedTimedParity.IsPresent
$willRunSpeedEvidence = $willRunAnySpeed -and -not $SkipSpeedEvidence.IsPresent
$willRunAccuracyModelJobs = $willRunAnyAccuracy -and -not $PreflightOnly.IsPresent
$willRunSpeedModelJobs = $willRunAnySpeed -and -not $PreflightOnly.IsPresent
$needModels = $willRunAccuracyModelJobs -or $willRunSpeedModelJobs

$accuracyPresetRuns = @(
    foreach ($preset in $AccuracyPresets) {
        [pscustomobject]@{
            preset      = $preset
            step_prefix = "accuracy"
            output_root = "accuracy"
        }
    }
)
if ($RunExperimentalAccuracy.IsPresent) {
    $accuracyPresetRuns += @(
        foreach ($preset in $ExperimentalAccuracyPresets) {
            [pscustomobject]@{
                preset      = $preset
                step_prefix = "accuracy-experimental"
                output_root = "accuracy-experimental"
            }
        }
    )
}
if ($needModels) {
    if ($willRunQwenAccuracy -or $willRunQwenSpeed) {
        $QwenModel = Resolve-RequiredPath $QwenModel "QwenModel"
    }
    if ($willRunGemmaAccuracy -or $willRunGemmaSpeed) {
        $GemmaModel = Resolve-RequiredPath $GemmaModel "GemmaModel"
    }
}
if ($willRunAccuracyModelJobs) {
    if (($willRunQwenAccuracy -or $willRunQwenSpeed) -and [string]::IsNullOrWhiteSpace($QwenDataset)) {
        $QwenDataset = $Dataset
    }
    if (($willRunGemmaAccuracy -or $willRunGemmaSpeed) -and [string]::IsNullOrWhiteSpace($GemmaDataset)) {
        $GemmaDataset = $Dataset
    }
    if ($willRunQwenAccuracy) {
        $QwenDataset = Resolve-RequiredPath $QwenDataset "QwenDataset"
    }
    if ($willRunGemmaAccuracy) {
        $GemmaDataset = Resolve-RequiredPath $GemmaDataset "GemmaDataset"
    }
}
if ((Test-Stage "build") -or (Test-Stage "unit") -or $needAccuracy -or $needSpeed) {
    $BuildDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BuildDir)
}
if ($needSpeed) {
    $MainlineBuildDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($MainlineBuildDir)
}

if ($ContextSize -gt 16384) {
    throw "ContextSize must be <= 16384 for safe gate runs; got $ContextSize"
}
foreach ($case in ($SpeedCaseList -split ",")) {
    $trimmedCase = $case.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmedCase)) {
        continue
    }
    $parts = $trimmedCase -split ":"
    if ($parts.Count -lt 3) {
        throw "Speed case '$trimmedCase' must use name:prompt:gen[:draft] format"
    }
    $promptTokens = [int] $parts[1]
    $genTokens = [int] $parts[2]
    $draftTokens = 0
    if ($parts.Count -gt 3) {
        $draftTokens = [int] $parts[3]
    }
    if (($promptTokens + $genTokens + $draftTokens) -gt 16384) {
        throw "Speed case '$trimmedCase' exceeds the 16k safe gate token cap"
    }
}

$gpu = Invoke-Preflight
$pythonExe = "python"
$pythonArgPrefix = @()
$pythonPathEnv = @{}
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $pyLauncher) {
    $pythonExe = $pyLauncher.Path
    $pythonArgPrefix = @("-3")
} elseif (Test-Path -LiteralPath "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe") {
    $pythonExe = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
}
$pythonPathPrefixDir = ""
if (Test-Path -LiteralPath "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe") {
    $pythonPathPrefixDir = Split-Path -Parent "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
} elseif ($pythonExe -match '\\Python[0-9]+\\python\.exe$') {
    $pythonPathPrefixDir = Split-Path -Parent $pythonExe
}
if (-not [string]::IsNullOrWhiteSpace($pythonPathPrefixDir)) {
    $pythonPathEnv["PATH"] = "$pythonPathPrefixDir;$env:PATH"
}
$numpyPythonExe = $pythonExe
$numpyPythonArgPrefix = $pythonArgPrefix
$pathPython = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $pathPython -and (Test-PythonModule $pathPython.Path @() "numpy")) {
    $numpyPythonExe = $pathPython.Path
    $numpyPythonArgPrefix = @()
} elseif (-not (Test-PythonModule $numpyPythonExe $numpyPythonArgPrefix "numpy")) {
    $numpyPythonExe = ""
    $numpyPythonArgPrefix = @()
}
if (-not [string]::IsNullOrWhiteSpace($numpyPythonExe) -and $numpyPythonExe -match '\\python\.exe$') {
    $numpyPythonDir = Split-Path -Parent $numpyPythonExe
    $pythonPathEnv["PATH"] = "$numpyPythonDir;$env:PATH"
}
$steps = [System.Collections.Generic.List[object]]::new()

if ((Test-Stage "build") -and -not $SkipBuild.IsPresent) {
    $buildTargets = @(
        "--build", $BuildDir,
        "--config", "Release",
        "--target",
        "llama-bench",
        "llama-perplexity",
        "llama-tokenize",
        "test-batch-split",
        "test-kvarn-kv",
        "test-kvarn-cuda-scratch-ref"
    )
    if ($IncludeBlockedMixedTailTest.IsPresent) {
        $buildTargets += @("test-kvarn-cuda-mixed-tail")
    }
    $buildTargets += @(
        "test-kvarn-server-load-failure",
        "test-arg-parser",
        "-j", [string] $BuildJobs
    )
    Add-Step $steps "build-kvarn-targets" "build" (New-DirectCommandLines "cmake" $buildTargets) $BuildTimeoutSec
}

if ((Test-Stage "unit") -and -not $SkipUnit.IsPresent) {
    $unitRegex = "test-batch-split|test-kvarn-kv|test-kvarn-cuda-scratch-ref|test-kvarn-server-load-failure|test-arg-parser"
    if ($IncludeBlockedMixedTailTest.IsPresent) {
        $unitRegex = "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure|test-arg-parser"
    }
    Add-Step $steps "ctest-kvarn" "unit" (New-DirectCommandLines "ctest" @(
        "--test-dir", $BuildDir,
        "-C", "Release",
        "-R", $unitRegex,
        "-j", "1",
        "--output-on-failure"
    )) $UnitTimeoutSec

    $kvMemoryArgs = @($pythonArgPrefix) + @(
        "scripts/kvarn/kv_memory_estimate.py",
        "--self-test"
    )
    Add-Step $steps "kv-memory-selftest" "unit" (New-DirectCommandLines -Exe $pythonExe -Argv $kvMemoryArgs) $UnitTimeoutSec

    if ([string]::IsNullOrWhiteSpace($numpyPythonExe)) {
        Write-Warning "Skipping vllm-oracle-selftest because no Python interpreter with numpy was found."
    } else {
        $vllmOracleArgs = @($numpyPythonArgPrefix) + @(
            "scripts/kvarn/kvarn_vllm_oracle.py",
            "--self-test",
            "--head-dims", "128,256,512",
            "--presets", "k4v2,k4v4,k8v2,k8v4,k8v8",
            "--iters", [string] $KvarnIters
        )
        Add-Step $steps "vllm-oracle-selftest" "unit" (New-DirectCommandLines -Exe $numpyPythonExe -Argv $vllmOracleArgs) $UnitTimeoutSec
    }
}

if ($willRunAnyAccuracy) {
    $accuracyPreflightExecutables = @{
        "llama-perplexity" = (Join-Path $BuildDir "bin/Release/llama-perplexity.exe")
    }
    if (-not $SkipFixtureCheck.IsPresent) {
        $accuracyPreflightExecutables["llama-tokenize"] = (Join-Path $BuildDir "bin/Release/llama-tokenize.exe")
    }
    Add-Step $steps "accuracy-executable-preflight" "preflight" (New-ExecutablePreflightLines $accuracyPreflightExecutables) 120

    if (-not $PreflightOnly.IsPresent) {
        $accuracyScript = Join-Path $repoRoot "scripts/kvarn/run_accuracy_gate.ps1"
        foreach ($presetRun in $accuracyPresetRuns) {
        $preset = $presetRun.preset
        $commonAcc = @{
            BuildDir            = $BuildDir
            KvarnPreset         = $preset
            KvarnIters          = [string] $KvarnIters
            KvarnRtnQuantile    = [string] $KvarnRtnQuantile
            FlashAttn           = "off"
            Fit                 = "off"
            GpuLayers           = [string] $GpuLayers
            ContextSize         = [string] $ContextSize
            BatchSize           = [string] $BatchSize
            Chunks              = [string] $Chunks
            MaxPplIncrease      = [string] $MaxPplIncrease
            MaxMeanKL           = [string] $MaxMeanKL
            MaxKLD99            = [string] $MaxKLD99
            MaxKLD999           = [string] $MaxKLD999
            MaxKLDMax           = [string] $MaxKLDMax
            MaxBaselinePpl      = [string] $MaxBaselinePpl
            MinKvarnBodyRecords = [string] $MinKvarnBodyRecords
        }
        $qwenAccExpectedBits = Resolve-ExpectedEffectiveBitsForHighGqa $preset $QwenExpectedEffectiveKvarnBits $QwenLayerKeyBits $QwenLayerValueBits
        $gemmaAccExpectedBits = Resolve-ExpectedEffectiveBitsForHighGqa $preset $GemmaExpectedEffectiveKvarnBits $GemmaLayerKeyBits $GemmaLayerValueBits
        if ($MinActiveKvarnBodyRecords -gt 0) {
            $commonAcc["MinActiveKvarnBodyRecords"] = [string] $MinActiveKvarnBodyRecords
        }
        if ($TraceAttn.IsPresent) {
            $commonAcc["TraceAttn"] = $true
            $commonAcc["TraceLimit"] = [string] $TraceLimit
        }
        if ($TraceFwht.IsPresent) {
            $commonAcc["TraceFwht"] = $true
            $commonAcc["MinFwhtTaken"] = [string] $MinFwhtTaken
        }
        if ($SkipFixtureCheck.IsPresent) { $commonAcc["SkipFixtureCheck"] = $true }
        $effectiveUseKLDivergence = $UseKLDivergence.IsPresent -or -not $DisableKLDivergence.IsPresent
        if ($effectiveUseKLDivergence) { $commonAcc["UseKLDivergence"] = $true }
        if ($AllowDiagnosticAccuracyEnv.IsPresent) { $commonAcc["AllowDiagnosticEnv"] = $true }

        if ($willRunQwenAccuracy) {
            $qwenAcc = $commonAcc.Clone()
            $qwenAcc["Model"] = $QwenModel
            $qwenAcc["Dataset"] = $QwenDataset
            $qwenAcc["ExpectedKvarnLayers"] = "3-39:4"
            $qwenAcc["ExpectedEffectiveKvarnBits"] = $qwenAccExpectedBits
            $qwenAcc["ExtraArgs"] = @("-ncmoe", "34")
            $qwenAcc["OutputDir"] = Join-Path $OutputDir ("{0}/qwen/{1}" -f $presetRun.output_root, (Convert-ToFileStem $preset))
            if ($ParseSpecial.IsPresent -or $QwenParseSpecial.IsPresent) { $qwenAcc["ParseSpecial"] = $true }
            if ($QwenAllowChatMarkers.IsPresent) { $qwenAcc["AllowChatMarkers"] = $true }
            $qwenAccEnv = $pythonPathEnv.Clone()
            if (-not $DisablePaperFrame.IsPresent) { $qwenAccEnv["LLAMA_KVARN_ENABLE_PAPER_FRAME"] = "1" }
            Add-KvarnDiagnosticEnv $qwenAccEnv
            if (-not [string]::IsNullOrWhiteSpace($QwenLayerFilter)) { $qwenAccEnv["LLAMA_KVARN_LAYER_FILTER"] = $QwenLayerFilter }
            if (-not [string]::IsNullOrWhiteSpace($QwenLayerKeyBits)) { $qwenAccEnv["LLAMA_KVARN_LAYER_KEY_BITS"] = $QwenLayerKeyBits }
            if (-not [string]::IsNullOrWhiteSpace($QwenLayerValueBits)) { $qwenAccEnv["LLAMA_KVARN_LAYER_VALUE_BITS"] = $QwenLayerValueBits }
            Add-Step $steps ("{0}-qwen-{1}" -f $presetRun.step_prefix, (Convert-ToFileStem $preset)) "accuracy" (New-InvokeScriptSplatLines $accuracyScript $qwenAcc $qwenAccEnv) $AccuracyTimeoutSec
        }

        if ($willRunGemmaAccuracy) {
            $gemmaAcc = $commonAcc.Clone()
            $gemmaAcc["Model"] = $GemmaModel
            $gemmaAcc["Dataset"] = $GemmaDataset
            $gemmaAcc["ExpectedKvarnLayers"] = $GemmaExpectedKvarnLayers
            $gemmaAcc["ExpectedEffectiveKvarnBits"] = $gemmaAccExpectedBits
            $gemmaAcc["OutputDir"] = Join-Path $OutputDir ("{0}/gemma/{1}" -f $presetRun.output_root, (Convert-ToFileStem $preset))
            if ($ParseSpecial.IsPresent -or $GemmaParseSpecial.IsPresent) { $gemmaAcc["ParseSpecial"] = $true }
            if ($GemmaAllowChatMarkers.IsPresent) { $gemmaAcc["AllowChatMarkers"] = $true }
            if ($GemmaUseKLDivergence.IsPresent) { $gemmaAcc["UseKLDivergence"] = $true }
            $gemmaAccEnv = $pythonPathEnv.Clone()
            if (-not $DisablePaperFrame.IsPresent) {
                $gemmaAccEnv["LLAMA_KVARN_ENABLE_PAPER_FRAME"] = "1"
            }
            if ($GemmaNormalFallbackDiagnostic.IsPresent) {
                $gemmaAccEnv["LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA"] = $null
                $gemmaAccEnv["LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK"] = "1"
                $gemmaAcc["AllowKvarnFallback"] = $true
                $gemmaAcc["ExpectedKvarnLayers"] = ""
                $gemmaAcc["ExpectedEffectiveKvarnBits"] = ""
                $gemmaAcc["MinKvarnBodyRecords"] = "0"
            } else {
                $gemmaAccEnv["LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA"] = "1"
                $gemmaAccEnv["LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK"] = $null
            }
            if (-not [string]::IsNullOrWhiteSpace($GemmaLayerFilter)) { $gemmaAccEnv["LLAMA_KVARN_LAYER_FILTER"] = $GemmaLayerFilter }
            if (-not [string]::IsNullOrWhiteSpace($GemmaLayerKeyBits)) { $gemmaAccEnv["LLAMA_KVARN_LAYER_KEY_BITS"] = $GemmaLayerKeyBits }
            if (-not [string]::IsNullOrWhiteSpace($GemmaLayerValueBits)) { $gemmaAccEnv["LLAMA_KVARN_LAYER_VALUE_BITS"] = $GemmaLayerValueBits }
            if (-not [string]::IsNullOrWhiteSpace($GemmaProtectedDonors)) { $gemmaAccEnv["LLAMA_KVARN_GEMMA4_PROTECT_DONORS"] = $GemmaProtectedDonors }
            Add-KvarnDiagnosticEnv $gemmaAccEnv
            if ($GemmaRouteConservative.IsPresent) {
                $gemmaAccEnv["LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE"] = "1"
            }
            Add-Step $steps ("{0}-gemma-{1}" -f $presetRun.step_prefix, (Convert-ToFileStem $preset)) "accuracy" (New-InvokeScriptSplatLines $accuracyScript $gemmaAcc $gemmaAccEnv) $AccuracyTimeoutSec
        }
        }
    }
}

if ($willRunAnySpeed) {
    $speedPreflightExecutables = @{
        "kvarn-llama-bench" = (Join-Path $BuildDir "bin/Release/llama-bench.exe")
    }
    if ($willRunSpeedTimedParity) {
        $speedPreflightExecutables["mainline-llama-bench"] = (Join-Path $MainlineBuildDir "bin/Release/llama-bench.exe")
    }
    Add-Step $steps "speed-executable-preflight" "preflight" (New-ExecutablePreflightLines $speedPreflightExecutables) 120

    if (-not $PreflightOnly.IsPresent) {
        $parityScript = Join-Path $repoRoot "scripts/kvarn/run_mainline_parity_matrix.ps1"
        $benchScript = Join-Path $repoRoot "scripts/kvarn/run_bench_matrix.ps1"
        foreach ($preset in $SpeedPresets) {
        $qwenSpeedExpectedBits = Resolve-ExpectedEffectiveBitsForHighGqa $preset $QwenExpectedEffectiveKvarnBits $QwenLayerKeyBits $QwenLayerValueBits
        $gemmaSpeedExpectedBits = Resolve-ExpectedEffectiveBitsForHighGqa $preset $GemmaExpectedEffectiveKvarnBits $GemmaLayerKeyBits $GemmaLayerValueBits
        if ($willRunQwenSpeed) {
            $qwenSpeed = @{
                Model                   = $QwenModel
                MainlineBuildDir        = $MainlineBuildDir
                KvarnBuildDir           = $BuildDir
                CaseList                = $SpeedCaseList
                FlashAttn               = "off"
                MainlineFlashAttn       = "on"
                KvarnFlashAttn          = "off"
                CacheTypeK              = "q8_0"
                CacheTypeV              = "q8_0"
                KvarnPreset             = $preset
                KvarnIters              = [string] $KvarnIters
                RtnQuantile             = [string] $KvarnRtnQuantile
                Repetitions             = [string] $QwenRepetitions
                MinParityRatio          = [string] $Tier1MinRatio
                FailBelowMinParityRatio = $true
                MinKvarnLayerLogs       = "10"
                ExpectedKvarnLayers     = "3-39:4"
                ExpectedEffectiveKvarnBits = $qwenSpeedExpectedBits
                KvarnPaperFrame         = $true
                KvarnDirectRecordBatch  = $true
                OutputDir               = Join-Path $OutputDir ("speed/qwen/{0}" -f (Convert-ToFileStem $preset))
                GpuLayers               = [string] $GpuLayers
                ExtraArgs               = @("-ncmoe", "34")
            }
            $qwenSpeedEvidence = @{
                Model                         = $QwenModel
                BuildDir                      = $BuildDir
                CaseList                      = $SpeedEvidenceCaseList
                FlashAttn                     = "off"
                KvCacheQuant                  = "kvarn"
                KvarnPreset                   = $preset
                KvarnIters                    = [string] $KvarnIters
                RtnQuantile                   = [string] $KvarnRtnQuantile
                Repetitions                   = "1"
                MinKvarnLayerLogs             = "10"
                ExpectedKvarnLayers           = "3-39:4"
                ExpectedEffectiveKvarnBits    = $qwenSpeedExpectedBits
                KvarnPaperFrame               = $true
                KvarnDirectRecordBatch        = $true
                RequireDirectRecordBatchPhases = $true
                TraceStore                    = $true
                MinBatchedStorePhaseUses      = [string] $MinBatchedStorePhaseUses
                OutputDir                     = Join-Path $OutputDir ("speed-evidence/qwen/{0}" -f (Convert-ToFileStem $preset))
                GpuLayers                     = [string] $GpuLayers
                ExtraArgs                     = @("-ncmoe", "34")
            }
            if ($TraceAttn.IsPresent) {
                $qwenSpeedEvidence["TraceAttn"] = $true
                $qwenSpeedEvidence["TraceLimit"] = [string] $TraceLimit
            }
            if ($TraceFwht.IsPresent) {
                $qwenSpeedEvidence["TraceFwht"] = $true
                $qwenSpeedEvidence["MinFwhtTaken"] = [string] $MinFwhtTaken
            }
            if ($MinActiveKvarnBodyRecords -gt 0) {
                $qwenSpeedEvidence["MinActiveKvarnBodyRecords"] = [string] $MinActiveKvarnBodyRecords
            }
            $qwenSpeedEnv = @{
                "LLAMA_KVARN_ENABLE_PAPER_FRAME" = $(if ($DisablePaperFrame.IsPresent) { $null } else { "1" })
                "LLAMA_KVARN_LAYER_FILTER" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerFilter)) { $null } else { $QwenLayerFilter })
                "LLAMA_KVARN_LAYER_KEY_BITS" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerKeyBits)) { $null } else { $QwenLayerKeyBits })
                "LLAMA_KVARN_LAYER_VALUE_BITS" = $(if ([string]::IsNullOrWhiteSpace($QwenLayerValueBits)) { $null } else { $QwenLayerValueBits })
            }
            Add-KvarnDiagnosticEnv $qwenSpeedEnv
            if ($willRunSpeedTimedParity) {
                Add-Step $steps ("speed-qwen-timed-{0}" -f (Convert-ToFileStem $preset)) "speed" (New-InvokeScriptSplatLines $parityScript $qwenSpeed $qwenSpeedEnv) $SpeedTimeoutSec
            }
            if ($willRunSpeedEvidence) {
                Add-Step $steps ("speed-qwen-evidence-{0}" -f (Convert-ToFileStem $preset)) "speed" (New-InvokeScriptSplatLines $benchScript $qwenSpeedEvidence $qwenSpeedEnv) $SpeedTimeoutSec
            }
        }

        if ($willRunGemmaSpeed) {
            $gemmaSpeed = @{
                Model                   = $GemmaModel
                MainlineBuildDir        = $MainlineBuildDir
                KvarnBuildDir           = $BuildDir
                CaseList                = $SpeedCaseList
                FlashAttn               = "off"
                MainlineFlashAttn       = "on"
                KvarnFlashAttn          = "off"
                CacheTypeK              = "q8_0"
                CacheTypeV              = "q8_0"
                KvarnPreset             = $preset
                KvarnIters              = [string] $KvarnIters
                RtnQuantile             = [string] $KvarnRtnQuantile
                Repetitions             = [string] $GemmaRepetitions
                MinParityRatio          = [string] $Tier1MinRatio
                FailBelowMinParityRatio = $true
                MinKvarnLayerLogs       = "8"
                ExpectedKvarnLayers     = "5-47:6"
                ExpectedEffectiveKvarnBits = $gemmaSpeedExpectedBits
                KvarnPaperFrame         = $true
                KvarnDirectRecordBatch  = $true
                OutputDir               = Join-Path $OutputDir ("speed/gemma-true-iswa/{0}" -f (Convert-ToFileStem $preset))
                GpuLayers               = [string] $GpuLayers
            }
            $gemmaSpeedEvidence = @{
                Model                         = $GemmaModel
                BuildDir                      = $BuildDir
                CaseList                      = $SpeedEvidenceCaseList
                FlashAttn                     = "off"
                KvCacheQuant                  = "kvarn"
                KvarnPreset                   = $preset
                KvarnIters                    = [string] $KvarnIters
                RtnQuantile                   = [string] $KvarnRtnQuantile
                Repetitions                   = "1"
                MinKvarnLayerLogs             = "8"
                ExpectedKvarnLayers           = "5-47:6"
                ExpectedEffectiveKvarnBits    = $gemmaSpeedExpectedBits
                KvarnPaperFrame               = $true
                KvarnDirectRecordBatch        = $true
                RequireDirectRecordBatchPhases = $true
                TraceStore                    = $true
                MinBatchedStorePhaseUses      = [string] $MinBatchedStorePhaseUses
                OutputDir                     = Join-Path $OutputDir ("speed-evidence/gemma-true-iswa/{0}" -f (Convert-ToFileStem $preset))
                GpuLayers                     = [string] $GpuLayers
            }
            if ($TraceAttn.IsPresent) {
                $gemmaSpeedEvidence["TraceAttn"] = $true
                $gemmaSpeedEvidence["TraceLimit"] = [string] $TraceLimit
            }
            if ($TraceFwht.IsPresent) {
                $gemmaSpeedEvidence["TraceFwht"] = $true
                $gemmaSpeedEvidence["MinFwhtTaken"] = [string] $MinFwhtTaken
            }
            if ($MinActiveKvarnBodyRecords -gt 0) {
                $gemmaSpeedEvidence["MinActiveKvarnBodyRecords"] = [string] $MinActiveKvarnBodyRecords
            }
            $gemmaSpeedEnv = @{
                "LLAMA_KVARN_ENABLE_PAPER_FRAME" = $(if ($DisablePaperFrame.IsPresent) { $null } else { "1" })
                "LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA" = $(if ($GemmaNormalFallbackDiagnostic.IsPresent) { $null } else { "1" })
                "LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK" = $(if ($GemmaNormalFallbackDiagnostic.IsPresent) { "1" } else { $null })
                "LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE" = $(if ($GemmaRouteConservative.IsPresent) { "1" } else { $null })
                "LLAMA_KVARN_LAYER_FILTER" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerFilter)) { $null } else { $GemmaLayerFilter })
                "LLAMA_KVARN_LAYER_KEY_BITS" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerKeyBits)) { $null } else { $GemmaLayerKeyBits })
                "LLAMA_KVARN_LAYER_VALUE_BITS" = $(if ([string]::IsNullOrWhiteSpace($GemmaLayerValueBits)) { $null } else { $GemmaLayerValueBits })
                "LLAMA_KVARN_GEMMA4_PROTECT_DONORS" = $(if ([string]::IsNullOrWhiteSpace($GemmaProtectedDonors)) { $null } else { $GemmaProtectedDonors })
            }
            Add-KvarnDiagnosticEnv $gemmaSpeedEnv
            if ($willRunSpeedTimedParity) {
                Add-Step $steps ("speed-gemma-true-iswa-timed-{0}" -f (Convert-ToFileStem $preset)) "speed" (New-InvokeScriptSplatLines $parityScript $gemmaSpeed $gemmaSpeedEnv) $SpeedTimeoutSec
            }
            if ($willRunSpeedEvidence) {
                Add-Step $steps ("speed-gemma-true-iswa-evidence-{0}" -f (Convert-ToFileStem $preset)) "speed" (New-InvokeScriptSplatLines $benchScript $gemmaSpeedEvidence $gemmaSpeedEnv) $SpeedTimeoutSec
            }
        }
        }
    }
}

$manifest = [pscustomobject]@{
    created_at               = (Get-Date).ToString("o")
    repo                     = $repoRoot
    output_dir               = $OutputDir
    dry_run                  = $DryRun.IsPresent
    preflight_only           = $PreflightOnly.IsPresent
    resume                   = $Resume.IsPresent
    stages                   = $Stage
    qwen_model               = $QwenModel
    gemma_model              = $GemmaModel
    dataset                  = $Dataset
    qwen_dataset             = $QwenDataset
    gemma_dataset            = $GemmaDataset
    qwen_expected_effective_kvarn_bits = $QwenExpectedEffectiveKvarnBits
    gemma_expected_effective_kvarn_bits = $GemmaExpectedEffectiveKvarnBits
    build_dir                = $BuildDir
    mainline_build_dir       = $MainlineBuildDir
    context_size             = $ContextSize
    batch_size               = $BatchSize
    chunks                   = $Chunks
    accuracy_presets         = $AccuracyPresets
    experimental_accuracy_presets = $ExperimentalAccuracyPresets
    run_experimental_accuracy = $RunExperimentalAccuracy.IsPresent
    speed_presets            = $SpeedPresets
    speed_case_list          = $SpeedCaseList
    speed_evidence_case_list = $SpeedEvidenceCaseList
    skip_speed_timed_parity  = $SkipSpeedTimedParity.IsPresent
    skip_speed_evidence      = $SkipSpeedEvidence.IsPresent
    max_ppl_increase         = $MaxPplIncrease
    max_mean_kl              = $MaxMeanKL
    max_kld_99               = $MaxKLD99
    max_kld_999              = $MaxKLD999
    max_kld_max              = $MaxKLDMax
    max_baseline_ppl         = $MaxBaselinePpl
    min_kvarn_body_records   = $MinKvarnBodyRecords
    min_active_kvarn_body_records = $MinActiveKvarnBodyRecords
    min_batched_store_phase_uses = $MinBatchedStorePhaseUses
    use_kl_divergence        = $UseKLDivergence.IsPresent -or -not $DisableKLDivergence.IsPresent
    gemma_use_kl_divergence  = $GemmaUseKLDivergence.IsPresent
    disable_kl_divergence    = $DisableKLDivergence.IsPresent
    allow_diagnostic_accuracy_env = $AllowDiagnosticAccuracyEnv.IsPresent
    gemma_route_conservative = $GemmaRouteConservative.IsPresent
    gemma_normal_fallback_diagnostic = $GemmaNormalFallbackDiagnostic.IsPresent
    trace_attn               = $TraceAttn.IsPresent
    trace_limit              = $TraceLimit
    trace_fwht               = $TraceFwht.IsPresent
    min_fwht_taken           = $MinFwhtTaken
    attn_ref_scratch         = $AttnRefScratch.IsPresent
    attn_enable_body_f32_mirror = $AttnEnableBodyF32Mirror.IsPresent
    enable_f32_dequant_cache = $EnableF32DequantCache.IsPresent
    attn_disable_body_f32_mirror = $AttnDisableBodyF32Mirror.IsPresent
    disable_prefill_direct_attn = $DisablePrefillDirectAttn.IsPresent
    disable_prefill_direct_store = $DisablePrefillDirectStore.IsPresent
    paper_mixed_frame        = $PaperMixedFrame.IsPresent
    disable_paper_frame      = $DisablePaperFrame.IsPresent
    enable_gemma_paper_frame = -not $DisablePaperFrame.IsPresent
    qwen_layer_filter        = $QwenLayerFilter
    gemma_layer_filter       = $GemmaLayerFilter
    qwen_layer_key_bits      = $QwenLayerKeyBits
    qwen_layer_value_bits    = $QwenLayerValueBits
    gemma_layer_key_bits     = $GemmaLayerKeyBits
    gemma_layer_value_bits   = $GemmaLayerValueBits
    tier1_min_ratio          = $Tier1MinRatio
    gpu_snapshot             = $gpu
    allow_existing_llama     = $AllowExistingLlama.IsPresent
    allow_existing_compute   = $AllowExistingGpuCompute.IsPresent
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDir "manifest.json") -Encoding UTF8

if ($steps.Count -eq 0) {
    Write-Host "No steps selected. Output: $OutputDir"
    if ($script:KvarnGateMutexHeld) {
        $script:KvarnGateMutex.ReleaseMutex()
        $script:KvarnGateMutexHeld = $false
    }
    exit 0
}

$oldState = if ($Resume.IsPresent) { Read-State $statePath } else { @{} }
$stateRows = [System.Collections.Generic.List[object]]::new()
foreach ($step in $steps) {
    $oldStep = $(if ($oldState.ContainsKey($step.id)) { $oldState[$step.id] } else { $null })
    if (-not $RerunAll.IsPresent -and $null -ne $oldStep -and $oldStep.status -eq "PASS" -and $oldStep.command_sha256 -eq $step.command_sha256) {
        $stateRows.Add($oldState[$step.id])
        Write-Host ("SKIP PASS: {0}" -f $step.id)
        continue
    }
    if (-not $RerunAll.IsPresent -and $null -ne $oldStep -and $oldStep.status -eq "PASS") {
        Write-Host ("RERUN CHANGED: {0}" -f $step.id)
    }
    if ($step.kind -eq "accuracy" -or $step.kind -eq "speed") {
        [void] (Invoke-Preflight)
    }
    $stepDir = Join-Path $OutputDir ("steps/{0}" -f (Convert-ToFileStem $step.id))
    Invoke-SafeStep -Step $step -StepDir $stepDir -StatePath $statePath -StateRows $stateRows
}

Write-Host "KVarN safe full gate complete: $OutputDir"
if ($script:KvarnGateMutexHeld) {
    $script:KvarnGateMutex.ReleaseMutex()
    $script:KvarnGateMutexHeld = $false
}
