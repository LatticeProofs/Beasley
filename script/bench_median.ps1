param(
    [int]$NBits = 128,
    [int]$G = 8,
    [int]$Ell = 5,
    [int]$Nizk1 = 1,
    [int]$Runs = 50,
    [switch]$CacheSpectra
)

$exe = Join-Path $PSScriptRoot "..\target\release\phase_bench.exe"

$env:RAYON_NUM_THREADS = "1"
if ($CacheSpectra) { $env:VOPRF_CACHE_SPECTRA = "1" } else { Remove-Item Env:VOPRF_CACHE_SPECTRA -ErrorAction SilentlyContinue }

$label = "n=$NBits g=$G ell=$Ell nizk1=$Nizk1 runs=$Runs cache=$($CacheSpectra.IsPresent)"
Write-Host "== phase_bench single thread x ${Runs}: $label ==" -ForegroundColor Cyan

$prove = @(); $verify = @(); $crs = @(); $peakMB = @(); $sizes = @(); $fps = @()

function ParseMs([string]$s) {
    if ($s -match '([\d.]+)\s*ms')      { return [double]$Matches[1] }
    elseif ($s -match '([\d.]+)\s*µs')  { return [double]$Matches[1] / 1000.0 }
    elseif ($s -match '([\d.]+)\s*s')   { return [double]$Matches[1] * 1000.0 }
    return $null
}

for ($i = 1; $i -le $Runs; $i++) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = "$NBits $G $Ell $Nizk1"
    $psi.RedirectStandardOutput = $true
    $psi.UseShellExecute = $false
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $p = [System.Diagnostics.Process]::Start($psi)

    $peak = 0L
    $outTask = $p.StandardOutput.ReadToEndAsync()
    while (-not $p.HasExited) {
        try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch {}
        Start-Sleep -Milliseconds 20
    }
    try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch {}
    $out = $outTask.Result
    $p.WaitForExit()

    foreach ($line in $out -split "`r?`n") {
        if ($line -match '^prove total:\s+(.+)$')       { $prove  += (ParseMs $Matches[1]) }
        elseif ($line -match '^verify total:\s+(\S+)')  { $verify += (ParseMs $Matches[1]) }
        elseif ($line -match '^CRS precompute:\s+(\S+)'){ $crs    += (ParseMs $Matches[1]) }
        elseif ($line -match '^proof size:.*?=\s*(\d+)\s*bytes') { $sizes += [int]$Matches[1] }
        elseif ($line -match '^proof fingerprint:\s*(\S+)')      { $fps   += $Matches[1] }
    }
    $peakMB += [math]::Round($peak / 1MB, 1)
    if ($i % 10 -eq 0) { Write-Host "  ... $i/$Runs" -ForegroundColor DarkGray }
}

function Stats([string]$name, [double[]]$xs, [string]$unit) {
    if ($xs.Count -eq 0) { Write-Host ("  {0,-16} (no data)" -f $name); return }
    $s = $xs | Sort-Object
    $med = if ($s.Count % 2) { $s[[int]([math]::Floor($s.Count / 2))] }
           else { ($s[$s.Count / 2 - 1] + $s[$s.Count / 2]) / 2 }
    Write-Host ("  {0,-16} median {1,9:N2} {5}   min {2,9:N2}   max {3,9:N2}   n={4}" -f `
        $name, $med, $s[0], $s[-1], $s.Count, $unit)
}

Write-Host ""
Stats "prove"  $prove  "ms"
Stats "verify" $verify "ms"
Stats "CRS"    $crs    "ms"
Stats "peak RAM" ($peakMB | ForEach-Object { [double]$_ }) "MB"
$uniqFp = $fps | Sort-Object -Unique
Write-Host ("  proof size       {0} bytes" -f ($sizes | Select-Object -Unique))
Write-Host ("  fingerprint      {0}{1}" -f ($uniqFp -join ", "), $(if ($uniqFp.Count -eq 1) { "  (OK)" } else { "  ** No!**" }))
Write-Host ""

Remove-Item Env:RAYON_NUM_THREADS  -ErrorAction SilentlyContinue
Remove-Item Env:VOPRF_CACHE_SPECTRA -ErrorAction SilentlyContinue
