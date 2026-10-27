param(
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [switch]$NoShutdown,
    [int]$RecoveryGraceSeconds = 180
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$resolvedRun = [System.IO.Path]::GetFullPath($RunDirectory)
$resolvedProject = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd('\') + '\'
if (-not $resolvedRun.StartsWith($resolvedProject, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'The run directory must be inside this project.'
}
if (Test-Path -LiteralPath $resolvedRun) { throw 'Refusing a duplicate launch into an existing run directory.' }
if ($RecoveryGraceSeconds -lt 0) { throw 'Recovery grace must be nonnegative.' }
$launcherDirectory = $resolvedRun + '_launcher'
New-Item -ItemType Directory -Path $launcherDirectory -Force | Out-Null
$ownerLock = [System.IO.File]::Open((Join-Path $launcherDirectory 'owner.lock'),
    [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$launcherLog = Join-Path $launcherDirectory 'launcher.log'
$linuxRun = '/mnt/d/' + $resolvedRun.Substring(3).Replace('\','/')
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class OvernightPower {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$awake = [OvernightPower]::SetThreadExecutionState([uint32]2147483649)
if ($awake -eq 0) { throw 'Could not prevent system sleep; training was not launched.' }
try {
    "Started $(Get-Date -Format o); automatic shutdown authorized for success or unresolved failure." |
        Set-Content -LiteralPath $launcherLog
    $bashCommand = "cd /mnt/d/work/robotics/IsaacLabTutorial && .venv/bin/python -u -m isaaclab_tutorial.tasks.place_box.workspace_pipeline --teacher logs/rsl_rl/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/model_2196.pt --output-dir '$linuxRun' --max-blocks 8 --camera-blocks 3 --max-hours 10"
    $nativeArgs = '-d Ubuntu -- bash -lc "' + $bashCommand + '"'
    $child = Start-Process -FilePath 'wsl.exe' -ArgumentList $nativeArgs -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput (Join-Path $launcherDirectory 'pipeline_stdout.log') `
        -RedirectStandardError (Join-Path $launcherDirectory 'pipeline_stderr.log')
    "Pipeline exited $($child.ExitCode) at $(Get-Date -Format o)." | Add-Content -LiteralPath $launcherLog
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedRun 'status.json'))) {
        New-Item -ItemType Directory -Path $resolvedRun -Force | Out-Null
        @{ outcome='failed'; phase='launcher'; error="WSL exited $($child.ExitCode) before pipeline status was created";
           launcher_logs=$launcherDirectory; updated_at=(Get-Date -Format o) } |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedRun 'status.json')
        'Pipeline failed during startup. Inspect the sibling _launcher logs.' |
            Set-Content -LiteralPath (Join-Path $resolvedRun 'SUMMARY.md')
    }
    & wsl.exe -d Ubuntu -- sync
} catch {
    $_ | Out-String | Add-Content -LiteralPath $launcherLog
} finally {
    try {
        "Pipeline ended; allowing $RecoveryGraceSeconds seconds for recovery before shutdown." |
            Add-Content -LiteralPath $launcherLog
        if (-not $NoShutdown) { Start-Sleep -Seconds $RecoveryGraceSeconds }
        if (-not $NoShutdown -and -not (Test-Path -LiteralPath (Join-Path $resolvedRun 'NO_SHUTDOWN')) `
                -and -not (Test-Path -LiteralPath (Join-Path $launcherDirectory 'NO_SHUTDOWN'))) {
            "Requesting graceful Windows shutdown at $(Get-Date -Format o)." | Add-Content -LiteralPath $launcherLog
            & "$env:SystemRoot\System32\shutdown.exe" /s /t 0
        } else {
            'Shutdown suppressed by explicit override.' | Add-Content -LiteralPath $launcherLog
        }
    } finally {
        [void][OvernightPower]::SetThreadExecutionState([uint32]2147483648)
        $ownerLock.Dispose()
    }
}
