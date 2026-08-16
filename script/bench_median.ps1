<#
.SYNOPSIS
    Median-of-N benchmark for the VOPRF phase_bench binary.

.DESCRIPTION
    Runs target\release\phase_bench.exe N times in a fresh process each time and
    reports the median prove time, verify time and peak memory (peak working
    set), along with min / max / mean / stdev so you can judge stability.

    Each run is a separate process, so the peak working set is measured for that
    run alone and no allocator state carries over between samples.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File script\bench_median.ps1
    Default: 30 runs of n=128 g=8 ell=5 nizk1=1, single threaded.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File script\bench_median.ps1 -Runs 30 -G 8 -Csv out.csv
    Same, but also writes every raw sample to out.csv.
#>
[CmdletBinding()]
param(
    [int]$NBits = 128,
    [int]$G = 8,
    [int]$Ell = 5,
    [int]$Nizk1 = 1,

    # Number of measured runs. The median is taken over these.
    [int]$Runs = 30,

    # Discarded runs executed first, to warm the file cache and CPU clocks.
    [int]$Warmup = 1,

    # RAYON_NUM_THREADS for the child process. 1 = single threaded (single core).
    [int]$Threads = 1,

    # Pin the child to one logical core, e.g. -PinCore 2. -1 = no pinning.
    # Off by default: on hybrid CPUs an arbitrary core may be an E-core.
    [int]$PinCore = -1,

    [switch]$CacheSpectra,

    # Disable AVX2 in the child. AVX2 is ON by default; use this only to
    # measure the scalar fallback for comparison.
    [switch]$NoAvx2,

    # Run the child at High priority to reduce scheduler noise.
    [switch]$HighPriority,

    # cargo build --release before benchmarking.
    [switch]$Build,

    # Optional path to write per-run raw samples as CSV.
    [string]$Csv
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'target\release\phase_bench.exe'

# ---------------------------------------------------------------- build ----

if ($Build -or -not (Test-Path $exe)) {
    Write-Host "building phase_bench (release) ..." -ForegroundColor DarkGray
    Push-Location $repo
    try {
        & cargo build --release --bin phase_bench
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}
if (-not (Test-Path $exe)) {
    throw "phase_bench.exe not found at $exe -- run with -Build, or: cargo build --release"
}

# ------------------------------------------------- peak-memory probe -------
# Process.PeakWorkingSet64 throws once the child has exited, and polling it
# while the child runs both misses short peaks and steals CPU from the very
# thing being timed. GetProcessMemoryInfo works on the still-open handle of an
# exited process, so we read the true peak once, after the run, for free.

if (-not ([System.Management.Automation.PSTypeName]'VoprfMem').Type) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class VoprfMem
{
    [StructLayout(LayoutKind.Sequential)]
    public struct PMC
    {
        public uint   cb;
        public uint   PageFaultCount;
        public IntPtr PeakWorkingSetSize;
        public IntPtr WorkingSetSize;
        public IntPtr QuotaPeakPagedPoolUsage;
        public IntPtr QuotaPagedPoolUsage;
        public IntPtr QuotaPeakNonPagedPoolUsage;
        public IntPtr QuotaNonPagedPoolUsage;
        public IntPtr PagefileUsage;
        public IntPtr PeakPagefileUsage;
    }

    [DllImport("psapi.dll", SetLastError = true)]
    private static extern bool GetProcessMemoryInfo(IntPtr hProcess, ref PMC counters, uint size);

    public static long PeakWorkingSet(IntPtr handle)
    {
        PMC c = new PMC();
        c.cb = (uint)Marshal.SizeOf(typeof(PMC));
        if (GetProcessMemoryInfo(handle, ref c, c.cb)) { return (long)c.PeakWorkingSetSize; }
        return -1;
    }
}
'@
}

$script:UseProbe = $true

# ---------------------------------------------------------------- helpers --

# Rust's Duration Debug format: "81.9984ms", "585.6us", "1.2345s", "42ns".
# The unit is matched without any non-ASCII literal so the script stays
# encoding-proof; anything unrecognised is treated as microseconds.
function ConvertTo-Ms {
    param([string]$Text)

    if ($Text -match '([0-9]+(?:\.[0-9]+)?)\s*(\D*)$') {
        $v = [double]$Matches[1]
        $u = $Matches[2].Trim()
        switch -Regex ($u) {
            '^ns$' { return $v / 1e6 }
            '^ms$' { return $v }
            '^s$'  { return $v * 1000.0 }
            default { return $v / 1000.0 }
        }
    }
    return $null
}

function Get-Stats {
    param([double[]]$Values)

    if ($null -eq $Values -or $Values.Count -eq 0) { return $null }

    $s = [double[]]($Values | Sort-Object)
    $n = $s.Count

    if ($n % 2 -eq 1) {
        $median = $s[[int](($n - 1) / 2)]
    } else {
        $median = ($s[($n / 2) - 1] + $s[$n / 2]) / 2.0
    }

    $mean = ($s | Measure-Object -Average).Average
    $sd = 0.0
    if ($n -gt 1) {
        $acc = 0.0
        foreach ($x in $s) { $acc += ($x - $mean) * ($x - $mean) }
        $sd = [math]::Sqrt($acc / ($n - 1))
    }

    return [pscustomobject]@{
        N      = $n
        Median = $median
        Min    = $s[0]
        Max    = $s[$n - 1]
        Mean   = $mean
        Sd     = $sd
    }
}

function Invoke-PhaseBench {
    param([string]$Exe, [string]$BenchArgs)

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $Exe
    $psi.Arguments              = $BenchArgs
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding  = [System.Text.Encoding]::UTF8

    $p = [System.Diagnostics.Process]::Start($psi)

    if ($HighPriority) {
        try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High } catch { }
    }
    if ($PinCore -ge 0) {
        try { $p.ProcessorAffinity = [IntPtr][int64]([int64]1 -shl $PinCore) } catch {
            Write-Warning "could not pin to core ${PinCore}: $($_.Exception.Message)"
        }
    }

    # Start both readers before waiting, or a full pipe buffer deadlocks the child.
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()

    $polled = 0L
    if (-not $script:UseProbe) {
        while (-not $p.HasExited) {
            try {
                $p.Refresh()
                if ($p.PeakWorkingSet64 -gt $polled) { $polled = $p.PeakWorkingSet64 }
            } catch { }
            Start-Sleep -Milliseconds 10
        }
    }

    $p.WaitForExit()

    $peak = 0L
    if ($script:UseProbe) {
        try { $peak = [VoprfMem]::PeakWorkingSet($p.Handle) } catch { $peak = -1 }
        if ($peak -le 0) {
            # Probe unavailable on this box; every later run falls back to polling.
            Write-Warning "GetProcessMemoryInfo failed, falling back to polling PeakWorkingSet64"
            $script:UseProbe = $false
        }
    }
    if ($peak -le 0) { $peak = $polled }

    $r = [pscustomobject]@{
        ExitCode    = $p.ExitCode
        Stdout      = $outTask.Result
        Stderr      = $errTask.Result
        PeakBytes   = $peak
        ProveMs     = $null
        VerifyMs    = $null
        CrsMs       = $null
        Verdict     = $null
        Transcript  = $null
        Serialized  = $null
        Fingerprint = $null
        Avx2        = $null
        RayonThreads = $null
    }
    $p.Dispose()

    foreach ($line in ($r.Stdout -split "`r?`n")) {
        if ($line -match '^prove total:\s+(\S+)') {
            $r.ProveMs = ConvertTo-Ms $Matches[1]
        } elseif ($line -match '^verify total:\s+(\S+)\s*->\s*(\S+)') {
            $r.VerifyMs = ConvertTo-Ms $Matches[1]
            $r.Verdict = $Matches[2]
        } elseif ($line -match '^CRS precompute:\s+(\S+)') {
            $r.CrsMs = ConvertTo-Ms $Matches[1]
        } elseif ($line -match '^proof size:\s+(\d+)\s*B') {
            $r.Transcript = [int]$Matches[1]
        } elseif ($line -match '^\s*serialized total:\s+(\d+)\s*B') {
            $r.Serialized = [int]$Matches[1]
        } elseif ($line -match '^proof fingerprint:\s*(\S+)') {
            $r.Fingerprint = $Matches[1]
        } elseif ($line -match '^env:\s+AVX2\s+(\S+)\s*\|\s*rayon threads\s+(\d+)') {
            $r.Avx2 = $Matches[1]
            $r.RayonThreads = [int]$Matches[2]
        }
    }

    return $r
}

function Write-Row {
    param([string]$Name, $Stats, [string]$Unit, [string]$Format = 'N2')

    if ($null -eq $Stats) {
        Write-Host ("  {0,-12} (no data)" -f $Name) -ForegroundColor Red
        return
    }
    $rsd = 0.0
    if ($Stats.Mean -ne 0) { $rsd = $Stats.Sd / $Stats.Mean * 100.0 }

    Write-Host ("  {0,-12}{1,12:$Format}{2,12:$Format}{3,12:$Format}{4,12:$Format}{5,11:$Format}{6,8:N1}%   {7}" -f `
        $Name, $Stats.Median, $Stats.Min, $Stats.Max, $Stats.Mean, $Stats.Sd, $rsd, $Unit)
}

# ------------------------------------------------------------------ run ----

$benchArgs = "$NBits $G $Ell $Nizk1"

$savedThreads = $env:RAYON_NUM_THREADS
$savedCache   = $env:VOPRF_CACHE_SPECTRA
$savedAvx     = $env:VOPRF_NO_AVX2

$results = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

try {
    $env:RAYON_NUM_THREADS = "$Threads"
    if ($CacheSpectra) { $env:VOPRF_CACHE_SPECTRA = "1" } else { $env:VOPRF_CACHE_SPECTRA = $null }
    if ($NoAvx2)       { $env:VOPRF_NO_AVX2       = "1" } else { $env:VOPRF_NO_AVX2       = $null }

    $pinText = 'off'
    if ($PinCore -ge 0) { $pinText = "core $PinCore" }

    Write-Host ""
    Write-Host "== phase_bench: median of $Runs runs ==" -ForegroundColor Cyan
    Write-Host ("   args      n_bits=$NBits g=$G ell=$Ell nizk1=$Nizk1")
    Write-Host ("   config    threads $Threads   warmup $Warmup   pin $pinText   cache_spectra $($CacheSpectra.IsPresent)")
    Write-Host ("   exe       $exe")
    Write-Host ""

    for ($i = 1; $i -le $Warmup; $i++) {
        Write-Host ("   warmup {0}/{1} ..." -f $i, $Warmup) -ForegroundColor DarkGray
        $w = Invoke-PhaseBench -Exe $exe -BenchArgs $benchArgs
        if ($i -eq 1 -and $w.Avx2) {
            Write-Host ("   runtime   AVX2 {0}   rayon threads {1}" -f $w.Avx2, $w.RayonThreads) -ForegroundColor DarkGray
        }
    }

    for ($i = 1; $i -le $Runs; $i++) {
        $r = Invoke-PhaseBench -Exe $exe -BenchArgs $benchArgs

        if ($r.ExitCode -ne 0) {
            Write-Warning "run $i exited with code $($r.ExitCode)"
            if ($r.Stderr) { Write-Warning $r.Stderr.Trim() }
        }
        $results += $r

        if ($i % 5 -eq 0 -or $i -eq $Runs) {
            Write-Host ("   {0,3}/{1}   prove {2,8:N2} ms   verify {3,7:N2} ms   peak {4,7:N1} MB" -f `
                $i, $Runs, $r.ProveMs, $r.VerifyMs, ($r.PeakBytes / 1MB)) -ForegroundColor DarkGray
        }
    }
} finally {
    $env:RAYON_NUM_THREADS   = $savedThreads
    $env:VOPRF_CACHE_SPECTRA = $savedCache
    $env:VOPRF_NO_AVX2       = $savedAvx
}

$sw.Stop()

# --------------------------------------------------------------- report ----

$ok = @($results | Where-Object { $_.ExitCode -eq 0 -and $null -ne $_.ProveMs -and $null -ne $_.VerifyMs })

if ($ok.Count -eq 0) {
    Write-Host ""
    Write-Host "no usable runs -- check the output of: $exe $benchArgs" -ForegroundColor Red
    exit 1
}

$proveStats  = Get-Stats ([double[]]($ok | ForEach-Object { $_.ProveMs }))
$verifyStats = Get-Stats ([double[]]($ok | ForEach-Object { $_.VerifyMs }))
$peakStats   = Get-Stats ([double[]]($ok | ForEach-Object { $_.PeakBytes / 1MB }))
$crsStats    = Get-Stats ([double[]]($ok | Where-Object { $null -ne $_.CrsMs } | ForEach-Object { $_.CrsMs }))

Write-Host ""
Write-Host ("  {0,-12}{1,12}{2,12}{3,12}{4,12}{5,11}{6,9}" -f `
    'metric', 'median', 'min', 'max', 'mean', 'sd', 'rsd') -ForegroundColor Cyan
Write-Host ("  " + ('-' * 84)) -ForegroundColor DarkGray

Write-Row 'prove'    $proveStats  'ms'
Write-Row 'verify'   $verifyStats 'ms'
Write-Row 'peak mem' $peakStats   'MB'
Write-Row 'CRS setup' $crsStats   'ms'

$verdicts = @($ok | ForEach-Object { $_.Verdict } | Sort-Object -Unique)
$fps      = @($ok | Where-Object { $_.Fingerprint } | ForEach-Object { $_.Fingerprint } | Sort-Object -Unique)
$sizes    = @($ok | Where-Object { $_.Transcript } | ForEach-Object { $_.Transcript } | Sort-Object -Unique)
$sers     = @($ok | Where-Object { $_.Serialized } | ForEach-Object { $_.Serialized } | Sort-Object -Unique)
$avx      = @($ok | Where-Object { $_.Avx2 } | ForEach-Object { $_.Avx2 } | Sort-Object -Unique)
$rthreads = @($ok | Where-Object { $_.RayonThreads } | ForEach-Object { $_.RayonThreads } | Sort-Object -Unique)

Write-Host ""
Write-Host "  MEDIAN OF $($ok.Count) RUNS   (AVX2 $($avx -join '/'), $($rthreads -join '/') thread)" -ForegroundColor Green
Write-Host ("    prove       {0,10:N2} ms" -f $proveStats.Median)  -ForegroundColor Green
Write-Host ("    verify      {0,10:N2} ms" -f $verifyStats.Median) -ForegroundColor Green
Write-Host ("    peak mem    {0,10:N1} MB" -f $peakStats.Median)   -ForegroundColor Green
Write-Host ("    proof size  {0,10} B   (transcript; {1} B serialized incl. framing + commitment stub)" -f `
    ($sizes -join '/'), ($sers -join '/')) -ForegroundColor Green

# --- consistency checks: every run must have done identical, accepted work ---

Write-Host ""
if ($avx.Count -eq 1 -and $avx[0] -eq 'on') {
    Write-Host "  AVX2           on, single code path across all runs" -ForegroundColor DarkGray
} else {
    Write-Host "  AVX2           NOT CONSISTENTLY ON: $($avx -join ', ')" -ForegroundColor Red
}
if ($rthreads.Count -eq 1 -and $rthreads[0] -eq 1) {
    Write-Host "  parallelism    1 rayon thread (single core)" -ForegroundColor DarkGray
} else {
    Write-Host "  parallelism    NOT SINGLE THREADED: $($rthreads -join ', ') rayon threads" -ForegroundColor Red
}
if ($sizes.Count -gt 1) {
    Write-Host "  proof size     VARIES ACROSS RUNS: $($sizes -join ', ') B" -ForegroundColor Red
}

if ($verdicts.Count -eq 1 -and $verdicts[0] -eq 'ACCEPT') {
    Write-Host "  verify         ACCEPT on all $($ok.Count) runs" -ForegroundColor DarkGray
} else {
    Write-Host "  verify         NOT ALL ACCEPT: $($verdicts -join ', ')" -ForegroundColor Red
}

if ($fps.Count -eq 1) {
    Write-Host "  fingerprint    $($fps[0])  (stable)" -ForegroundColor DarkGray
} else {
    Write-Host "  fingerprint    UNSTABLE: $($fps -join ', ')" -ForegroundColor Red
}

if ($results.Count -ne $ok.Count) {
    Write-Host "  discarded      $($results.Count - $ok.Count) failed run(s)" -ForegroundColor Red
}
Write-Host ("  wall clock     {0:N1} s total" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray

if ($Csv) {
    $ok | Select-Object @{n = 'run'; e = { [array]::IndexOf($ok, $_) + 1 } },
        @{n = 'prove_ms';  e = { [math]::Round($_.ProveMs, 4) } },
        @{n = 'verify_ms'; e = { [math]::Round($_.VerifyMs, 4) } },
        @{n = 'peak_mb';   e = { [math]::Round($_.PeakBytes / 1MB, 3) } },
        @{n = 'crs_ms';    e = { [math]::Round($_.CrsMs, 4) } },
        @{n = 'proof_transcript_b'; e = { $_.Transcript } },
        @{n = 'proof_serialized_b'; e = { $_.Serialized } },
        @{n = 'avx2';      e = { $_.Avx2 } },
        @{n = 'rayon_threads'; e = { $_.RayonThreads } },
        @{n = 'fingerprint'; e = { $_.Fingerprint } } |
        Export-Csv -Path $Csv -NoTypeInformation -Encoding UTF8
    Write-Host "  csv            $Csv" -ForegroundColor DarkGray
}

Write-Host ""
