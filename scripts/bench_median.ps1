param(
    [int]$NBits = 128,
    [int]$W = 4,
    [int]$Ell = 5,
    [int]$Runs = 50,
    [int]$Threads = 1,
    [switch]$NoBuild
)

$exe = Join-Path $PSScriptRoot "..\target\release\poh_demo.exe"

if (-not $NoBuild) {
    Write-Host "building..." -ForegroundColor DarkGray
    & cargo build --release --bin poh_demo | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Error "cargo build failed (exit $LASTEXITCODE)"; exit 1 }
}
if (-not (Test-Path $exe)) { Write-Error "cannot find $exe"; exit 1 }

$env:RAYON_NUM_THREADS = "$Threads"

$label = "n=$NBits w=$W m=$Ell threads=$Threads runs=$Runs"
Write-Host "== poh_demo --bench x ${Runs}: $label ==" -ForegroundColor Cyan

$data = @{}
function Add([string]$k, [double]$v) { if (-not $data.ContainsKey($k)) { $data[$k] = @() }; $data[$k] += $v }
$info = @{}
$peakMB = @()

$reqParts   = @("B.client|sample_blind", "B.prove|eval_h", "B.prove|request")
$proveSc    = @("quot", "zk mask", "bin table", "merged table", "build_rows", "w_hat", "SC1 tables", "SC1", "lg build", "SC_full", "SC_bin", "SC5")
$proveStub  = @("commit")
$verifySc   = @("check+ctx", "build_rows", "SC1", "SC_full", "SC_bin", "SC5")
$verifyStub = @("PCS stub")

for ($i = 1; $i -le $Runs; $i++) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = "$NBits $W $Ell --bench"
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
    if ($p.ExitCode -ne 0) { Write-Error "poh_demo failed in run $i (exit $($p.ExitCode))"; exit 1 }

    $run = @{}
    foreach ($line in $out -split "`r?`n") {
        if ($line -match "^@bench`t([^`t]+)`t([^`t]+)`t(\S+)$") { $run["$($Matches[1])|$($Matches[2])"] = [double]$Matches[3]; continue }
        if ($line -match "^@info`t([^`t]+)`t(\S+)$") {
            if (-not $info.ContainsKey($Matches[1])) { $info[$Matches[1]] = @() }
            $info[$Matches[1]] += $Matches[2]
        }
    }
    foreach ($k in (@("B.prove|total", "B.verify|total") + $reqParts)) {
        if (-not $run.ContainsKey($k)) { Write-Error "run $i is missing @bench $k (did the poh_demo output format change?)"; exit 1 }
    }
    foreach ($k in $run.Keys) { Add $k $run[$k] }

    $req = 0.0;   foreach ($k in $reqParts)   { $req += $run[$k] }
    $pStub = 0.0; foreach ($n in $proveStub)  { if ($run.ContainsKey("B.prove|$n"))  { $pStub += $run["B.prove|$n"] } }
    $vStub = 0.0; foreach ($n in $verifyStub) { if ($run.ContainsKey("B.verify|$n")) { $vStub += $run["B.verify|$n"] } }
    $cSc = $run["B.prove|total"] - $run["B.prove|eval_h"] - $run["B.prove|request"] - $pStub
    Add "T|client.request"  $req
    Add "T|client.sumcheck" $cSc
    Add "T|client.total"    ($req + $cSc)
    Add "T|server.sumcheck" ($run["B.verify|total"] - $vStub)

    $peakMB += [math]::Round($peak / 1MB, 1)
    if ($i % 10 -eq 0) { Write-Host "  ... $i/$Runs" -ForegroundColor DarkGray }
}

function Median([double[]]$xs) {
    $s = @($xs | Sort-Object)
    if ($s.Count % 2) { return $s[[int]([math]::Floor($s.Count / 2))] }
    return ($s[$s.Count / 2 - 1] + $s[$s.Count / 2]) / 2
}
function Med([string]$key) {
    if ($data.ContainsKey($key)) { return Median ([double[]]$data[$key]) }
    return [double]::NaN
}
function Stats([string]$name, [string]$key) {
    if (-not $data.ContainsKey($key) -or $data[$key].Count -eq 0) { Write-Host ("  {0,-22} (no data)" -f $name); return }
    $xs = [double[]]$data[$key]
    $s = @($xs | Sort-Object)
    Write-Host ("  {0,-22} median {1,10:F3} ms   min {2,10:F3}   max {3,10:F3}   n={4}" -f `
        $name, (Median $xs), $s[0], $s[-1], $s.Count)
}
function Uniq([string]$key) { if ($info.ContainsKey($key)) { return @($info[$key] | Sort-Object -Unique) } return @() }

$nv = (Uniq "nv") -join ","

Write-Host ""
Write-Host "== Table 3 ($label, median, unit ms) ==" -ForegroundColor Green
$fmt = "  {0,-20} {1,22} {2,22}"
Write-Host ($fmt -f "", "Client time", "Server verification")
Write-Host ($fmt -f "Request generation", ("{0:F3}" -f (Med "T|client.request")), "-")
Write-Host ($fmt -f "Sumcheck", ("{0:F3}" -f (Med "T|client.sumcheck")), ("{0:F3}" -f (Med "T|server.sumcheck")))
Write-Host ($fmt -f "PCS", "[Akita nv=$nv]", "[Akita nv=$nv]")
Write-Host ($fmt -f "Total", ("{0:F3} + PCS" -f (Med "T|client.total")), ("{0:F3} + PCS" -f (Med "T|server.sumcheck")))

Write-Host ""
Write-Host "-- Client · Request generation breakdown --" -ForegroundColor Yellow
Stats "sample_blind" "B.client|sample_blind"
Stats "eval_h" "B.prove|eval_h"
Stats "request (C_x,c_r,d_x)" "B.prove|request"
Stats "= Request generation" "T|client.request"

Write-Host "-- Client · Sumcheck breakdown --" -ForegroundColor Yellow
foreach ($n in $proveSc) { Stats $n "B.prove|$n" }
Stats "= Sumcheck" "T|client.sumcheck"
Stats "(excluded) commit stub" "B.prove|commit"
Stats "(ref) prove total" "B.prove|total"

Write-Host "-- Server · Sumcheck breakdown --" -ForegroundColor Yellow
foreach ($n in $verifySc) { Stats $n "B.verify|$n" }
Stats "= Sumcheck" "T|server.sumcheck"
Stats "(excluded) PCS stub" "B.verify|PCS stub"
Stats "(ref) verify total" "B.verify|total"

Write-Host "-- Phase A (statement = plaintext B_x, for comparison) --" -ForegroundColor Yellow
Stats "prove total" "A.prove|total"
Stats "verify" "A.verify|total"

$knownP = @("total", "eval_h", "request") + $proveSc + $proveStub
$knownV = @("total") + $verifySc + $verifyStub
$unknown = @($data.Keys | Where-Object {
        ($_ -like "B.prove|*" -and $knownP -notcontains $_.Substring(8)) -or
        ($_ -like "B.verify|*" -and $knownV -notcontains $_.Substring(9))
    })
if ($unknown.Count -gt 0) {
    Write-Host ("  ** unregistered segments (counted into Sumcheck, check their classification): {0}" -f ($unknown -join ", ")) -ForegroundColor Red
}

Write-Host "-- other --" -ForegroundColor Yellow
$pk = [double[]]$peakMB; $ps = @($pk | Sort-Object)
Write-Host ("  {0,-22} median {1,10:F1} MB   min {2,10:F1}   max {3,10:F1}" -f "peak RAM", (Median $pk), $ps[0], $ps[-1])
Write-Host ("  {0,-22} {1}" -f "PCS commitment nv", $nv)
Write-Host ("  {0,-22} {1} B, {2} rounds" -f "sumcheck transcript", ((Uniq "transcript_bytes") -join ","), ((Uniq "rounds") -join ","))
$uniqFp = Uniq "fingerprint"
Write-Host ("  {0,-22} {1}{2}" -f "fingerprint", ($uniqFp -join ", "), $(if ($uniqFp.Count -eq 1) { "  (all identical OK)" } else { "  ** MISMATCH! **" }))

Write-Host ""
Write-Host "notes:" -ForegroundColor DarkGray
Write-Host "  * each cell sums per run first, then takes the median => the medians of Request + Sumcheck need not add up exactly to the median of Total." -ForegroundColor DarkGray
Write-Host "  * PCS: the only commitment is the merged table (nv above); the opening is 4 points + 14 linear functionals (same Hachi cost model as report.rs)." -ForegroundColor DarkGray
Write-Host "    Akita: ref/akita/bench_dense_fp64_1core.ps1 -NumVars $nv -ProveThreads $Threads -VerifyThreads $Threads" -ForegroundColor DarkGray
Write-Host "  * Client Sumcheck includes open_h_sum from the SC5 finalization (an honest MLE evaluation; a real PCS must compute this value too)." -ForegroundColor DarkGray
Write-Host "  * peak RAM is the peak working set of the whole poh_demo process (incl. the CRS and the eval_h_naive cross-check)." -ForegroundColor DarkGray

Remove-Item Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue
