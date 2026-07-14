<#
.SYNOPSIS
    Provides a process-wide GPU launch lease and refuses launches while GPU compute is active.

.DESCRIPTION
    Enter-KvarnGpuLease first acquires a cross-process named mutex, then queries
    the process/GPU snapshot. Callers must retain the returned lease until the
    launched child exits and release it in finally with Exit-KvarnGpuLease.
    The real snapshot fails closed when nvidia-smi is unavailable or its query
    fails. Tests can inject ProcessSnapshotProvider and SleepAction.
#>

$script:KvarnGpuMutexName = 'Local\KVarnGpuExclusiveLaunch_v1'

# Compare normalized executable basenames only. The built-in WDDM GUI
# allowlist is deliberately limited to Windows shell/protected UI processes.
# Callers may explicitly add local GUI clients through
# AdditionalWddmDesktopBaseName. Membership is not evidence of idleness: it applies
# only to an [N/A] row whose PID resolves to the same exact basename (or whose
# NVIDIA-reported name is the exact protected-process placeholder).  Numeric
# residency and known compute/model runtimes always take precedence.  Keep
# additions exact, documented, and limited to reviewed non-compute GUI clients.
$script:KvarnKnownComputeBaseNames = @(
    'llama-bench', 'llama-cli', 'llama-completion', 'llama-perplexity',
    'llama-results', 'llama-server', 'ollama', 'python', 'python3', 'pythonw',
    'test-kvarn', 'test-kvarn-cuda-scratch-ref', 'torchrun', 'vllm'
)
$script:KvarnWddmDesktopBaseNames = @(
    'applicationframehost', 'crossdeviceresume', 'dwm', 'explorer', 'lockapp',
    'logonui', 'searchhost', 'shellhost', 'shellexperiencehost',
    'startmenuexperiencehost', 'systemsettings', 'taskmgr', 'textinputhost'
)

function ConvertTo-KvarnProcessBaseName {
    param([AllowNull()] [string] $Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { return '' }
    $value = $Name.Trim().Trim('"') -replace '\\', '/'
    $leaf = ($value -split '/')[-1]
    if ($leaf.EndsWith('.exe', [StringComparison]::OrdinalIgnoreCase)) {
        $leaf = $leaf.Substring(0, $leaf.Length - 4)
    }
    return $leaf.ToLowerInvariant()
}

function Test-KvarnKnownComputeBaseName {
    param([AllowNull()] [string] $Name)
    return $script:KvarnKnownComputeBaseNames -contains (ConvertTo-KvarnProcessBaseName $Name)
}

function New-KvarnWddmDesktopBaseNameSet {
    param([string[]] $AdditionalWddmDesktopBaseName = @())
    $allowed = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($name in @($script:KvarnWddmDesktopBaseNames + $AdditionalWddmDesktopBaseName)) {
        $normalized = ConvertTo-KvarnProcessBaseName $name
        if (-not [string]::IsNullOrWhiteSpace($normalized)) {
            [void] $allowed.Add($normalized)
        }
    }
    return $allowed
}

function Test-KvarnWddmDesktopBaseName {
    param(
        [AllowNull()] [string] $Name,
        [Parameter(Mandatory = $true)] $AllowedBaseNames
    )
    return $AllowedBaseNames.Contains((ConvertTo-KvarnProcessBaseName $Name))
}

function Get-KvarnGpuProcessSnapshot {
    [CmdletBinding()]
    param(
        [string[]] $AdditionalWddmDesktopBaseName = @(),
        [scriptblock] $ProcessProvider = { Get-Process -ErrorAction SilentlyContinue },
        [scriptblock] $NvidiaSmiCommandProvider = { Get-Command nvidia-smi -ErrorAction SilentlyContinue },
        [scriptblock] $NvidiaSmiQueryProvider = {
            param([string] $Path)
            $queryRows = & $Path --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>&1
            [pscustomobject]@{ ExitCode = $LASTEXITCODE; Rows = @($queryRows) }
        }
    )

    $wddmBuiltInBaseNames = New-KvarnWddmDesktopBaseNameSet
    $wddmAllowedBaseNames = New-KvarnWddmDesktopBaseNameSet $AdditionalWddmDesktopBaseName

    try { $processes = @(& $ProcessProvider) } catch {
        throw ("GPU exclusivity guard process snapshot threw: {0}" -f $_.Exception.Message)
    }
    $resolvedByPid = @{}
    $ambiguousPids = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($process in $processes) {
        $pidValue = 0
        if ($null -eq $process -or $null -eq $process.Id -or
            -not [int]::TryParse(([string] $process.Id), [ref] $pidValue) -or $pidValue -lt 0) {
            throw 'GPU exclusivity guard received a process snapshot entry with an invalid PID'
        }
        # Windows exposes the System Idle Process as PID 0; it cannot be a
        # query-compute-apps PID and is irrelevant to resolution.
        if ($pidValue -eq 0) { continue }
        $name = ConvertTo-KvarnProcessBaseName ([string] $process.ProcessName)
        if ([string]::IsNullOrWhiteSpace($name)) {
            throw ("GPU exclusivity guard could not normalize process name for PID {0}" -f $pidValue)
        }
        if ($resolvedByPid.ContainsKey($pidValue)) {
            [void] $ambiguousPids.Add($pidValue)
        } else {
            $resolvedByPid[$pidValue] = $name
        }
    }

    $records = New-Object System.Collections.Generic.List[object]
    foreach ($entry in $resolvedByPid.GetEnumerator()) {
        if (Test-KvarnKnownComputeBaseName $entry.Value) {
            $records.Add([pscustomobject]@{
                Id = [int] $entry.Key; ProcessName = [string] $entry.Value
                IsGpuCompute = $true; IsGpuConflict = $true
                UsedGpuMemoryMiB = $null; Source = 'Get-Process'
                ConflictReason = 'known-compute-basename'
            })
        }
    }

    $nvidiaSmi = & $NvidiaSmiCommandProvider
    if ($null -eq $nvidiaSmi) {
        throw 'GPU exclusivity guard cannot verify CUDA idleness because nvidia-smi was not found'
    }
    try {
        $queryResult = & $NvidiaSmiQueryProvider $nvidiaSmi.Path
    } catch {
        throw ("GPU exclusivity guard nvidia-smi query threw: {0}" -f $_.Exception.Message)
    }
    if ($null -eq $queryResult -or $null -eq $queryResult.ExitCode) {
        throw 'GPU exclusivity guard nvidia-smi query returned no exit status'
    }
    $rows = @($queryResult.Rows)
    if ([int] $queryResult.ExitCode -ne 0) {
        $diagnostic = ($rows | ForEach-Object { $_.ToString() }) -join ' | '
        throw ("GPU exclusivity guard nvidia-smi query failed with exit code {0}: {1}" -f $queryResult.ExitCode, $diagnostic)
    }

    $nvidiaPids = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($row in @($rows)) {
        if ([string]::IsNullOrWhiteSpace([string] $row)) { continue }
        $parts = @([string] $row -split ',', 3)
        $pidValue = 0
        if ($parts.Count -ne 3 -or -not [int]::TryParse($parts[0].Trim(), [ref] $pidValue) -or $pidValue -le 0) {
            throw ("GPU exclusivity guard could not parse nvidia-smi compute row: {0}" -f $row)
        }
        if (-not $nvidiaPids.Add($pidValue)) {
            throw ("GPU exclusivity guard found ambiguous duplicate nvidia-smi rows for PID {0}" -f $pidValue)
        }
        $reportedNameText = $parts[1].Trim()
        $reportedNameUnavailable = [string]::Equals(
            $reportedNameText, '[Insufficient Permissions]', [StringComparison]::OrdinalIgnoreCase)
        $reportedName = ConvertTo-KvarnProcessBaseName $reportedNameText
        if ([string]::IsNullOrWhiteSpace($reportedName)) {
            throw ("GPU exclusivity guard could not normalize nvidia-smi process name in row: {0}" -f $row)
        }
        $memoryText = $parts[2].Trim()
        $parsedMemory = 0.0
        $isNumericMemory = [double]::TryParse(
            $memoryText,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref] $parsedMemory)
        if ($isNumericMemory -and ($parsedMemory -lt 0 -or [double]::IsNaN($parsedMemory) -or [double]::IsInfinity($parsedMemory))) {
            throw ("GPU exclusivity guard found invalid nvidia-smi memory in row: {0}" -f $row)
        }

        $isResolved = $resolvedByPid.ContainsKey($pidValue) -and -not $ambiguousPids.Contains($pidValue)
        $resolvedName = if ($isResolved) { [string] $resolvedByPid[$pidValue] } else { '' }
        $nameAgrees = $isResolved -and [string]::Equals($reportedName, $resolvedName, [StringComparison]::OrdinalIgnoreCase)
        $knownCompute = (Test-KvarnKnownComputeBaseName $reportedName) -or
            ($isResolved -and (Test-KvarnKnownComputeBaseName $resolvedName))

        $reason = $null
        $memory = $null
        if ($isNumericMemory) {
            # Numeric residency is positive evidence and always wins over any
            # desktop allowlist or caller-provided PID exception.
            $memory = $parsedMemory
            $reason = 'numeric-gpu-memory'
        } elseif (-not (
            [string]::Equals($memoryText, '[N/A]', [StringComparison]::OrdinalIgnoreCase) -or
            [string]::Equals($memoryText, 'N/A', [StringComparison]::OrdinalIgnoreCase)
        )) {
            $reason = 'unknown-memory-status'
        } elseif ($knownCompute) {
            $reason = 'known-compute-basename'
        } elseif (-not $isResolved) {
            $reason = if ($ambiguousPids.Contains($pidValue)) { 'ambiguous-pid-resolution' } else { 'unresolved-pid' }
        } elseif ($reportedNameUnavailable -and -not (Test-KvarnWddmDesktopBaseName $resolvedName $wddmBuiltInBaseNames)) {
            $reason = 'unavailable-reported-name-unallowlisted'
        } elseif ($reportedNameUnavailable) {
            # WDDM may redact protected GUI process names.  This is safe only
            # when PID resolution independently yields an exact allowlisted
            # basename; an unallowlisted or unresolved PID remains blocking.
        } elseif (-not $nameAgrees) {
            $reason = 'process-name-disagreement'
        } elseif (-not (Test-KvarnWddmDesktopBaseName $resolvedName $wddmAllowedBaseNames)) {
            $reason = 'unallowlisted-wddm-na-process'
        }

        if ($null -ne $reason) {
            $source = if ($isResolved) { 'Get-Process+nvidia-smi' } else { 'nvidia-smi' }
            $records.Add([pscustomobject]@{
                Id = $pidValue; ProcessName = $reportedName
                IsGpuCompute = ($isNumericMemory -or $knownCompute)
                IsGpuConflict = $true; UsedGpuMemoryMiB = $memory
                Source = $source; ConflictReason = $reason
            })
        } else {
            # The only non-conflict nvidia-smi row: a WDDM N/A row whose PID
            # and normalized basename agree with the documented allowlist.
        }
    }
    return @($records | Sort-Object Id, Source -Unique)
}

function Assert-KvarnGpuIdle {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)] [string] $IntendedExecutable,
        [ValidateRange(0, 86400)] [int] $WaitTimeoutSeconds = 0,
        [ValidateRange(1, 60)] [int] $PollIntervalSeconds = 2,
        [int[]] $AllowedProcessId = @(),
        [string[]] $AdditionalWddmDesktopBaseName = @(),
        [scriptblock] $ProcessSnapshotProvider = {
            param([string[]] $AdditionalBaseNames)
            Get-KvarnGpuProcessSnapshot -AdditionalWddmDesktopBaseName $AdditionalBaseNames
        },
        [scriptblock] $SleepAction = { param([int] $Seconds) Start-Sleep -Seconds $Seconds }
    )
    $allowed = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($allowedId in @($AllowedProcessId)) { [void] $allowed.Add([int] $allowedId) }
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        $snapshot = @(& $ProcessSnapshotProvider $AdditionalWddmDesktopBaseName)
        $conflicts = @()
        foreach ($process in $snapshot) {
            $idValue = if ($null -ne $process.Id) { [int] $process.Id } elseif ($null -ne $process.Pid) { [int] $process.Pid } else { 0 }
            $name = if ($null -ne $process.ProcessName) { [string] $process.ProcessName } else { [string] $process.Name }
            $normalizedName = ConvertTo-KvarnProcessBaseName $name
            $numericMemory = $null -ne $process.UsedGpuMemoryMiB
            $hardConflict = $process.IsGpuConflict -eq $true -or $process.IsGpuCompute -eq $true -or
                $numericMemory -or (Test-KvarnKnownComputeBaseName $normalizedName)
            if (-not $hardConflict) { continue }
            # Conflict precedence is intentional: AllowedProcessId cannot hide
            # positive or uncertain GPU/known-model evidence.
            $conflicts += [pscustomobject]@{
                Id = $idValue; ProcessName = $normalizedName
                UsedGpuMemoryMiB = $process.UsedGpuMemoryMiB; Source = $process.Source
                ConflictReason = $process.ConflictReason
            }
        }
        if ($conflicts.Count -eq 0) {
            Write-Host ("GPU exclusivity guard: idle before {0}" -f $IntendedExecutable)
            return
        }
        $details = ($conflicts | ForEach-Object {
            $memory = if ($null -eq $_.UsedGpuMemoryMiB) { 'unknown' } else { [string] $_.UsedGpuMemoryMiB }
            $reason = if ([string]::IsNullOrWhiteSpace([string] $_.ConflictReason)) { 'conflict' } else { [string] $_.ConflictReason }
            "PID=$($_.Id) name=$($_.ProcessName) gpu_memory_mib=$memory source=$($_.Source) reason=$reason"
        }) -join '; '
        $remaining = $WaitTimeoutSeconds - $watch.Elapsed.TotalSeconds
        if ($WaitTimeoutSeconds -eq 0 -or $remaining -le 0) {
            throw ("GPU exclusivity guard refused launch of {0}; active conflicting process(es): {1}" -f $IntendedExecutable, $details)
        }
        $sleepSeconds = [Math]::Min(60, [Math]::Max(1, [int] [Math]::Ceiling([Math]::Min([double] $PollIntervalSeconds, $remaining))))
        Write-Host ("GPU exclusivity guard waiting up to {0}s before {1}; {2}" -f $sleepSeconds, $IntendedExecutable, $details)
        & $SleepAction $sleepSeconds
    }
}

function Enter-KvarnGpuLease {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)] [string] $IntendedExecutable,
        [ValidateRange(0, 86400)] [int] $WaitTimeoutSeconds = 0,
        [ValidateRange(1, 60)] [int] $PollIntervalSeconds = 2,
        [int[]] $AllowedProcessId = @(),
        [string[]] $AdditionalWddmDesktopBaseName = @(),
        [scriptblock] $ProcessSnapshotProvider = {
            param([string[]] $AdditionalBaseNames)
            Get-KvarnGpuProcessSnapshot -AdditionalWddmDesktopBaseName $AdditionalBaseNames
        },
        [scriptblock] $SleepAction = { param([int] $Seconds) Start-Sleep -Seconds $Seconds },
        [string] $MutexName = $script:KvarnGpuMutexName
    )
    $mutex = [System.Threading.Mutex]::new($false, $MutexName)
    $acquired = $false
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        try {
            $waitMs = if ($WaitTimeoutSeconds -eq 0) { 0 } else { [Math]::Min([int64]::MaxValue, [int64] $WaitTimeoutSeconds * 1000) }
            $acquired = $mutex.WaitOne([int] $waitMs)
        } catch [System.Threading.AbandonedMutexException] {
            $acquired = $true
            Write-Warning ("GPU exclusivity guard recovered abandoned lease '{0}'" -f $MutexName)
        }
        if (-not $acquired) {
            throw ("GPU exclusivity guard timed out after {0}s waiting for cross-process lease '{1}' before {2}" -f $WaitTimeoutSeconds, $MutexName, $IntendedExecutable)
        }
        Write-Host ("GPU exclusivity lease acquired: name={0} intended={1}" -f $MutexName, $IntendedExecutable)
        $remaining = if ($WaitTimeoutSeconds -eq 0) { 0 } else { [Math]::Max(0, [int] [Math]::Ceiling($WaitTimeoutSeconds - $watch.Elapsed.TotalSeconds)) }
        Assert-KvarnGpuIdle -IntendedExecutable $IntendedExecutable -WaitTimeoutSeconds $remaining -PollIntervalSeconds $PollIntervalSeconds -AllowedProcessId $AllowedProcessId -AdditionalWddmDesktopBaseName $AdditionalWddmDesktopBaseName -ProcessSnapshotProvider $ProcessSnapshotProvider -SleepAction $SleepAction
        return [pscustomobject]@{ Mutex = $mutex; MutexName = $MutexName; IntendedExecutable = $IntendedExecutable; Released = $false }
    } catch {
        if ($acquired) { try { $mutex.ReleaseMutex() } catch {} }
        $mutex.Dispose()
        throw
    }
}

function Exit-KvarnGpuLease {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)] $Lease)
    if ($Lease.Released) { return }
    try { $Lease.Mutex.ReleaseMutex() } finally {
        $Lease.Mutex.Dispose()
        $Lease.Released = $true
        Write-Host ("GPU exclusivity lease released: name={0} intended={1}" -f $Lease.MutexName, $Lease.IntendedExecutable)
    }
}
