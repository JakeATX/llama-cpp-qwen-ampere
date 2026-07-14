$ErrorActionPreference = 'Stop'
$guardPath = Join-Path $PSScriptRoot 'Assert-KvarnGpuIdle.ps1'
. $guardPath

function Assert-True([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw "ASSERTION FAILED: $Message" }
}

$emptyProvider = { @() }
$lease = Enter-KvarnGpuLease -IntendedExecutable 'llama-cli.exe' -ProcessSnapshotProvider $emptyProvider -MutexName ("Local\KVarnGuardTest_" + [guid]::NewGuid().ToString('N'))
try {
    Assert-True ($null -ne $lease.Mutex) 'lease must retain the mutex through the caller lifetime'
} finally {
    Exit-KvarnGpuLease $lease
}
Assert-True $lease.Released 'Exit-KvarnGpuLease must mark the lease released'

$conflictMessage = ''
try {
    Assert-KvarnGpuIdle -IntendedExecutable 'llama-perplexity.exe' -ProcessSnapshotProvider {
        @([pscustomobject]@{ Id = 4242; ProcessName = 'llama-cli'; IsGpuCompute = $true; UsedGpuMemoryMiB = 123; Source = 'test' })
    }
} catch { $conflictMessage = $_.Exception.Message }
Assert-True ($conflictMessage -match 'PID=4242') 'conflict error must log PID'
Assert-True ($conflictMessage -match 'gpu_memory_mib=123') 'conflict error must log GPU memory'
Assert-True ($conflictMessage -match 'source=test') 'conflict error must log source'

function Get-InjectedGpuSnapshot(
        [object[]] $Processes,
        [string[]] $Rows,
        [string[]] $AdditionalWddmDesktopBaseName = @()) {
    return @(Get-KvarnGpuProcessSnapshot `
        -AdditionalWddmDesktopBaseName $AdditionalWddmDesktopBaseName `
        -ProcessProvider { $Processes } `
        -NvidiaSmiCommandProvider { [pscustomobject]@{ Path = 'injected-nvidia-smi' } } `
        -NvidiaSmiQueryProvider { param($Path) [pscustomobject]@{ ExitCode = 0; Rows = $Rows } })
}

function Get-IdleRefusal([object[]] $Snapshot) {
    try { Assert-KvarnGpuIdle -IntendedExecutable 'injected-test.exe' -ProcessSnapshotProvider { $Snapshot } } catch { return $_.Exception.Message }
    return ''
}

$numericSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1001; ProcessName = 'dwm.exe' }) `
    @('1001, C:\Windows\System32\dwm.exe, 1')
$numericMessage = Get-IdleRefusal $numericSnapshot
Assert-True ($numericMessage -match 'reason=numeric-gpu-memory') 'numeric GPU memory must conflict even for allowlisted desktop processes'

$computeNaSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1002; ProcessName = 'LLAMA-CLI' }) `
    @('1002, C:\tools\llama-cli.exe, [N/A]')
$computeNaMessage = Get-IdleRefusal $computeNaSnapshot
Assert-True ($computeNaMessage -match 'reason=known-compute-basename') 'known model executables must conflict with WDDM N/A memory'

$desktopNaSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1003; ProcessName = 'DWM' }) `
    @('1003, C:\Windows\System32\dwm.exe, [n/a]')
Assert-True ($desktopNaSnapshot.Count -eq 0) 'matching case-insensitive allowlisted desktop WDDM N/A row must be ignored'

$chromeDefaultSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1020; ProcessName = 'Chrome.exe' }) `
    @('1020, C:\Program Files\Google\Chrome\Application\chrome.exe, [N/A]')
$chromeDefaultMessage = Get-IdleRefusal $chromeDefaultSnapshot
Assert-True ($chromeDefaultMessage -match 'reason=unallowlisted-wddm-na-process') 'local GUI clients must block unless explicitly added'

$chromeAllowedSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1020; ProcessName = 'Chrome.exe' }) `
    @('1020, C:\Program Files\Google\Chrome\Application\chrome.exe, [N/A]') `
    @('chrome', 'CHROME.EXE', 'C:\Program Files\Google\Chrome\Application\chrome.exe')
Assert-True ($chromeAllowedSnapshot.Count -eq 0) 'explicit GUI additions must normalize case, extension, path, and duplicates'

$redactedChromeAddedSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1025; ProcessName = 'chrome.exe' }) `
    @('1025, [Insufficient Permissions], [N/A]') @('chrome')
$redactedChromeAddedMessage = Get-IdleRefusal $redactedChromeAddedSnapshot
Assert-True ($redactedChromeAddedMessage -match 'reason=unavailable-reported-name-unallowlisted') 'caller additions must not authorize redacted NVIDIA process names'

$chromeNearNameSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1021; ProcessName = 'chrome-helper.exe' }) `
    @('1021, chrome-helper.exe, [N/A]') @('chrome')
$chromeNearNameMessage = Get-IdleRefusal $chromeNearNameSnapshot
Assert-True ($chromeNearNameMessage -match 'reason=unallowlisted-wddm-na-process') 'GUI additions must use exact normalized basenames'

$chromeNumericSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1022; ProcessName = 'chrome.exe' }) `
    @('1022, chrome.exe, 1') @('chrome')
$chromeNumericMessage = Get-IdleRefusal $chromeNumericSnapshot
Assert-True ($chromeNumericMessage -match 'reason=numeric-gpu-memory') 'explicit GUI additions must not override numeric residency'

$pythonAddedSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1023; ProcessName = 'python.exe' }) `
    @('1023, python.exe, [N/A]') @('python')
$pythonAddedMessage = Get-IdleRefusal $pythonAddedSnapshot
Assert-True ($pythonAddedMessage -match 'reason=known-compute-basename') 'explicit GUI additions must not override known compute runtimes'

$script:leaseAdditionalReceived = @()
$additionalLease = Enter-KvarnGpuLease -IntendedExecutable 'llama-cli.exe' `
    -AdditionalWddmDesktopBaseName @('chrome.exe', 'C:\Program Files\Google\Chrome\Application\CHROME.EXE') `
    -MutexName ("Local\KVarnGuardAdditionalTest_" + [guid]::NewGuid().ToString('N')) `
    -ProcessSnapshotProvider {
        param([string[]] $AdditionalBaseNames)
        $script:leaseAdditionalReceived = @($AdditionalBaseNames)
        Get-InjectedGpuSnapshot `
            @([pscustomobject]@{ Id = 1024; ProcessName = 'chrome.exe' }) `
            @('1024, chrome.exe, [N/A]') $AdditionalBaseNames
    }
try {
    Assert-True ($script:leaseAdditionalReceived.Count -eq 2) 'lease must propagate additional GUI basenames to the snapshot provider'
} finally {
    Exit-KvarnGpuLease $additionalLease
}

$redactedDesktopSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1012; ProcessName = 'Taskmgr.exe' }) `
    @('1012, [Insufficient Permissions], [N/A]')
Assert-True ($redactedDesktopSnapshot.Count -eq 0) 'redacted reported name may be ignored only when PID resolves to an exact allowlisted GUI basename'

$redactedLogonUiSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1014; ProcessName = 'LogonUI' }) `
    @('1014, [Insufficient Permissions], [N/A]')
Assert-True ($redactedLogonUiSnapshot.Count -eq 0) 'protected Windows LogonUI may be ignored only when PID resolution proves the exact reviewed GUI basename'

$numericLogonUiSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1014; ProcessName = 'LogonUI' }) `
    @('1014, [Insufficient Permissions], 1')
$numericLogonUiMessage = Get-IdleRefusal $numericLogonUiSnapshot
Assert-True ($numericLogonUiMessage -match 'reason=numeric-gpu-memory') 'numeric GPU memory must conflict even for protected allowlisted LogonUI'

$mismatchedLogonUiSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1014; ProcessName = 'LogonUI' }) `
    @('1014, C:\Windows\System32\dwm.exe, [N/A]')
$mismatchedLogonUiMessage = Get-IdleRefusal $mismatchedLogonUiSnapshot
Assert-True ($mismatchedLogonUiMessage -match 'reason=process-name-disagreement') 'non-redacted reported-name mismatch must block allowlisted LogonUI'

$redactedUnallowlistedSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1013; ProcessName = 'custom-gui-app.exe' }) `
    @('1013, [Insufficient Permissions], [N/A]')
$redactedUnallowlistedMessage = Get-IdleRefusal $redactedUnallowlistedSnapshot
Assert-True ($redactedUnallowlistedMessage -match 'reason=unavailable-reported-name-unallowlisted') 'redacted reported name with unallowlisted resolved PID must block'

$unknownSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1004; ProcessName = 'dwm' }) `
    @('1004, dwm.exe, [Not Supported]')
$unknownMessage = Get-IdleRefusal $unknownSnapshot
Assert-True ($unknownMessage -match 'reason=unknown-memory-status') 'unknown memory status must block rather than inherit the N/A exception'

$unresolvedSnapshot = Get-InjectedGpuSnapshot @() @('1005, dwm.exe, [N/A]')
$unresolvedMessage = Get-IdleRefusal $unresolvedSnapshot
Assert-True ($unresolvedMessage -match 'reason=unresolved-pid') 'unresolved WDDM N/A PID must block'

$disagreementSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1006; ProcessName = 'explorer' }) `
    @('1006, dwm.exe, [N/A]')
$disagreementMessage = Get-IdleRefusal $disagreementSnapshot
Assert-True ($disagreementMessage -match 'reason=process-name-disagreement') 'reported/resolved basename disagreement must block'

$unallowlistedSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1009; ProcessName = 'custom-gpu-app.exe' }) `
    @('1009, C:\tools\custom-gpu-app.exe, [N/A]')
$unallowlistedMessage = Get-IdleRefusal $unallowlistedSnapshot
Assert-True ($unallowlistedMessage -match 'reason=unallowlisted-wddm-na-process') 'matching but unallowlisted WDDM N/A process must block'

$malformedMessage = ''
try { [void] (Get-InjectedGpuSnapshot @() @('not-a-valid-row')) } catch { $malformedMessage = $_.Exception.Message }
Assert-True ($malformedMessage -match 'could not parse nvidia-smi compute row') 'malformed nvidia-smi row must fail closed'

$duplicateNvidiaMessage = ''
try {
    [void] (Get-InjectedGpuSnapshot `
        @([pscustomobject]@{ Id = 1010; ProcessName = 'dwm' }) `
        @('1010, dwm.exe, [N/A]', '1010, dwm.exe, [N/A]'))
} catch { $duplicateNvidiaMessage = $_.Exception.Message }
Assert-True ($duplicateNvidiaMessage -match 'ambiguous duplicate nvidia-smi rows') 'duplicate nvidia-smi PID rows must fail closed'

$nearNameSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1011; ProcessName = 'not-llama-cli-helper.exe' }) @()
Assert-True ($nearNameSnapshot.Count -eq 0) 'known model detection must use exact normalized basenames, not substrings'

$knownStandaloneSnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1014; ProcessName = 'C:\tools\LLAMA-PERPLEXITY.EXE' }) @()
$knownStandaloneMessage = Get-IdleRefusal $knownStandaloneSnapshot
Assert-True ($knownStandaloneMessage -match 'reason=known-compute-basename') 'known model basename must conflict even without an nvidia-smi row'

$standaloneComputeNames = @('python', 'python3', 'pythonw', 'torchrun', 'ollama', 'vllm')
$standalonePid = 1100
foreach ($computeName in $standaloneComputeNames) {
    foreach ($injectedName in @($computeName, "$computeName.exe", "C:\runtime\$($computeName.ToUpperInvariant()).EXE")) {
        $standalonePid++
        $snapshot = Get-InjectedGpuSnapshot `
            @([pscustomobject]@{ Id = $standalonePid; ProcessName = $injectedName }) @()
        $message = Get-IdleRefusal $snapshot
        Assert-True ($message -match 'reason=known-compute-basename') `
            ("standalone compute runtime must conflict without nvidia-smi row: {0}" -f $injectedName)
    }
}

$ambiguitySnapshot = Get-InjectedGpuSnapshot `
    @([pscustomobject]@{ Id = 1007; ProcessName = 'dwm' }, [pscustomobject]@{ Id = 1007; ProcessName = 'dwm' }) `
    @('1007, dwm.exe, [N/A]')
$ambiguityMessage = Get-IdleRefusal $ambiguitySnapshot
Assert-True ($ambiguityMessage -match 'reason=ambiguous-pid-resolution') 'ambiguous PID resolution must block'

$allowedOverrideMessage = ''
try {
    Assert-KvarnGpuIdle -IntendedExecutable 'injected-test.exe' -AllowedProcessId 1008 -ProcessSnapshotProvider {
        @([pscustomobject]@{ Id = 1008; ProcessName = 'python'; IsGpuCompute = $false; IsGpuConflict = $true; UsedGpuMemoryMiB = $null; Source = 'test'; ConflictReason = 'unresolved-pid' })
    }
} catch { $allowedOverrideMessage = $_.Exception.Message }
Assert-True ($allowedOverrideMessage -match 'PID=1008') 'conflict precedence must prevent AllowedProcessId from hiding a conflict'

$missingMessage = ''
try {
    Get-KvarnGpuProcessSnapshot -NvidiaSmiCommandProvider { $null }
} catch { $missingMessage = $_.Exception.Message }
Assert-True ($missingMessage -match 'nvidia-smi was not found') 'missing nvidia-smi must fail closed'

$nonzeroMessage = ''
try {
    Get-KvarnGpuProcessSnapshot -NvidiaSmiCommandProvider { [pscustomobject]@{ Path = 'injected-nvidia-smi' } } `
        -NvidiaSmiQueryProvider { param($Path) [pscustomobject]@{ ExitCode = 9; Rows = @('injected diagnostic') } }
} catch { $nonzeroMessage = $_.Exception.Message }
Assert-True ($nonzeroMessage -match 'exit code 9') 'nonzero nvidia-smi query must fail closed'
Assert-True ($nonzeroMessage -match 'injected diagnostic') 'nonzero query diagnostic must be preserved'

$throwMessage = ''
try {
    Get-KvarnGpuProcessSnapshot -NvidiaSmiCommandProvider { [pscustomobject]@{ Path = 'injected-nvidia-smi' } } `
        -NvidiaSmiQueryProvider { param($Path) throw 'injected native query exception' }
} catch { $throwMessage = $_.Exception.Message }
Assert-True ($throwMessage -match 'query threw') 'throwing nvidia-smi query must fail closed'
Assert-True ($throwMessage -match 'injected native query exception') 'throwing query diagnostic must be preserved'

$queryMessage = ''
try {
    Enter-KvarnGpuLease -IntendedExecutable 'llama-cli.exe' -MutexName ("Local\KVarnGuardTest_" + [guid]::NewGuid().ToString('N')) `
        -ProcessSnapshotProvider { throw 'injected nvidia-smi query failure' }
} catch { $queryMessage = $_.Exception.Message }
Assert-True ($queryMessage -match 'injected nvidia-smi query failure') 'snapshot/query failure must fail closed with diagnostics'

$script:snapshotCall = 0
$script:sleepCall = 0
$lease = Enter-KvarnGpuLease -IntendedExecutable 'llama-cli.exe' -WaitTimeoutSeconds 10 -PollIntervalSeconds 3 `
    -MutexName ("Local\KVarnGuardTest_" + [guid]::NewGuid().ToString('N')) `
    -ProcessSnapshotProvider {
        $script:snapshotCall++
        if ($script:snapshotCall -eq 1) {
            @([pscustomobject]@{ Id = 777; ProcessName = 'python'; IsGpuCompute = $true; Source = 'test' })
        } else { @() }
    } `
    -SleepAction {
        param([int] $Seconds)
        $script:sleepCall++
        Assert-True ($Seconds -le 60) 'each bounded-wait sleep must be at most 60 seconds'
    }
Exit-KvarnGpuLease $lease
Assert-True ($script:snapshotCall -eq 2) 'wait policy must rescan after conflict'
Assert-True ($script:sleepCall -eq 1) 'wait policy must sleep once before rescan'

# Hold a real named mutex in another PowerShell process to prove cross-process contention.
$mutexName = "Local\KVarnGuardContention_" + [guid]::NewGuid().ToString('N')
$markerPath = Join-Path $env:TEMP ("kvarn-guard-marker-" + [guid]::NewGuid().ToString('N'))
$releasePath = $markerPath + '.release'
$escapedGuard = $guardPath.Replace("'", "''")
$escapedMarker = $markerPath.Replace("'", "''")
$escapedRelease = $releasePath.Replace("'", "''")
$escapedMutex = $mutexName.Replace("'", "''")
$childCode = @"
`$ErrorActionPreference = 'Stop'
. '$escapedGuard'
`$lease = Enter-KvarnGpuLease -IntendedExecutable 'cpu-only-test' -ProcessSnapshotProvider { @() } -MutexName '$escapedMutex'
try {
    [System.IO.File]::WriteAllText('$escapedMarker', 'acquired')
    while (-not (Test-Path -LiteralPath '$escapedRelease')) { Start-Sleep -Milliseconds 25 }
} finally { Exit-KvarnGpuLease `$lease }
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCode))
$child = Start-Process -FilePath (Get-Command powershell).Path -ArgumentList '-NoProfile', '-EncodedCommand', $encoded -WindowStyle Hidden -PassThru
try {
    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    while (-not (Test-Path -LiteralPath $markerPath) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 25 }
    Assert-True (Test-Path -LiteralPath $markerPath) 'child must acquire the test mutex'

    $fastMessage = ''
    try { Enter-KvarnGpuLease -IntendedExecutable 'llama-cli.exe' -ProcessSnapshotProvider $emptyProvider -MutexName $mutexName } catch { $fastMessage = $_.Exception.Message }
    Assert-True ($fastMessage -match 'timed out after 0s') 'default lease contention must fail fast'
    Assert-True ($fastMessage -match [regex]::Escape($mutexName)) 'contention diagnostic must name the mutex'

    $watch = [Diagnostics.Stopwatch]::StartNew()
    $timeoutMessage = ''
    try { Enter-KvarnGpuLease -IntendedExecutable 'llama-bench.exe' -WaitTimeoutSeconds 1 -ProcessSnapshotProvider $emptyProvider -MutexName $mutexName } catch { $timeoutMessage = $_.Exception.Message }
    Assert-True ($timeoutMessage -match 'timed out after 1s') 'bounded lease contention must report timeout'
    Assert-True ($watch.Elapsed.TotalMilliseconds -ge 700 -and $watch.Elapsed.TotalSeconds -lt 5) 'bounded mutex wait must be finite and near configured timeout'
} finally {
    [System.IO.File]::WriteAllText($releasePath, 'release')
    if (-not $child.WaitForExit(5000)) { $child.Kill(); $child.WaitForExit() }
    Remove-Item -LiteralPath $markerPath, $releasePath -Force -ErrorAction SilentlyContinue
}
Assert-True ($child.ExitCode -eq 0) 'contention helper must exit cleanly after releasing lease'

$guardSource = [System.IO.File]::ReadAllText($guardPath)
Assert-True ($guardSource -match "nvidia-smi was not found") 'real guard must fail closed when nvidia-smi is missing'
Assert-True ($guardSource -match 'query failed with exit code') 'real guard must fail closed on nvidia-smi nonzero exit'
Assert-True ($guardSource -match 'query threw') 'real guard must fail closed when nvidia-smi throws'

foreach ($path in @($guardPath, $MyInvocation.MyCommand.Path)) {
    $tokens = $null; $errors = $null
    [void] [Management.Automation.Language.Parser]::ParseFile($path, [ref] $tokens, [ref] $errors)
    Assert-True ($errors.Count -eq 0) ("PowerShell parser errors in {0}: {1}" -f $path, (($errors | ForEach-Object Message) -join '; '))
}

Write-Host 'GPU lease guard CPU-only tests: PASS'
