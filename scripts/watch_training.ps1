$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$trainingOut = 'outputs/bitext-4b-full-pass-windows'
$observations = [System.Collections.Generic.List[object]]::new()
while ($true) {
    Clear-Host
    Write-Host 'GEMMA TRAINING - LIVE MONITOR' -ForegroundColor Cyan
    Write-Host 'Watching the existing GPU job. Ctrl+C closes this monitor only.'
    Write-Host ''
    $progress = $null
    if (Test-Path "$trainingOut/progress.json") {
        try { $progress = Get-Content "$trainingOut/progress.json" -Raw | ConvertFrom-Json } catch {}
    }
    if ($progress) {
        $now = [DateTimeOffset]::Now
        if ($observations.Count -eq 0 -or $observations[$observations.Count-1].Step -ne $progress.global_step) {
            $observations.Add([PSCustomObject]@{Step=[int]$progress.global_step; Time=$now})
            if ($observations.Count -gt 100) { $observations.RemoveAt(0) }
        }
        $remaining = $progress.target_steps - $progress.global_step
        Write-Host "Step: $($progress.global_step) / $($progress.target_steps)   Remaining: $remaining"
        Write-Host ('Total pass completed: {0:P1}' -f ($progress.global_step / $progress.target_steps))
        if ($observations.Count -gt 1 -and ($progress.global_step - $observations[0].Step) -ge 10) {
            $secondsPerStep = ($now - $observations[0].Time).TotalSeconds / ($progress.global_step - $observations[0].Step)
            $eta = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId($now.AddSeconds($secondsPerStep * $remaining), 'Pakistan Standard Time')
            Write-Host "Estimated finish: $($eta.ToString('ddd dd MMM, h:mm tt')) Pakistan time"
            Write-Host 'Estimate uses recent wall-clock speed; future validation/saving pauses can extend it.'
        } else { Write-Host 'Estimated finish: measuring speed; available after 10 observed steps.' }
        if ($null -ne $progress.best_validation_loss) { Write-Host "Best validation loss: $($progress.best_validation_loss)" }
    } else {
        Write-Host 'Preparing model / checking validation baseline before training updates.'
        Write-Host 'The progress counter appears after the first resumed training step.'
    }
    Write-Host ''
    Write-Host 'LATEST TRAINING LOG:' -ForegroundColor Yellow
    if (Test-Path reports/full_pass_windows.log) { Get-Content reports/full_pass_windows.log -Tail 5 }
    Write-Host ''
    Write-Host 'LATEST PROGRESS / DIAGNOSTICS:' -ForegroundColor Yellow
    if (Test-Path reports/full_pass_windows_errors.log) {
        Get-Content reports/full_pass_windows_errors.log -Tail 3 | ForEach-Object { $_ -replace '\x1b\[[0-9;]*[A-Za-z]', '' }
    }
    if (Test-Path "$trainingOut/training_summary.json") {
        $summary = Get-Content "$trainingOut/training_summary.json" -Raw | ConvertFrom-Json
        Write-Host "TRAINING COMPLETE. Selected checkpoint: $($summary.best_checkpoint)" -ForegroundColor Green
        break
    }
    if (Test-Path reports/full_pass_windows.pid) {
        $trainingPid = [int](Get-Content reports/full_pass_windows.pid)
        if (-not (Get-Process -Id $trainingPid -ErrorAction SilentlyContinue)) {
            Write-Host 'Training process exited without a completion summary. Inspect the logs above.' -ForegroundColor Red
            break
        }
    }
    Start-Sleep -Seconds 5
}
