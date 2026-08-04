
```
$env:RAYON_NUM_THREADS = 1
cargo run --release --no-default-features --features q64 --bin phase_bench 128 8 5 1
Remove-Item Env:RAYON_NUM_THREADS
```


```
powershell -ExecutionPolicy Bypass -File script/bench_median.ps1 -G 8 -Runs 50
```