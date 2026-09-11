
use blmr::hash::{bits_to_groups, eval_h};
use blmr::nizk1::{sample_blind, Nizk1Params, QueryTicket, COM_N, R_DIM, W_SLACK};
use blmr::params::{HashParams, ELL};
use blmr::proof;
use blmr::rng::insecure_test_secret;
use blmr::transcript::SimpleRng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn mb(b: usize) -> f64 {
    b as f64 / 1e6
}

fn mark(label: &str) {
    let live = LIVE.load(Relaxed);
    let peak = PEAK.swap(live, Relaxed);
    println!("  {label:<34} peak in scope {:>8.1} MB   still live after {:>8.1} MB", mb(peak), mb(live));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (n_bits, group_bits, ell) = (g(1, 128), g(2, 4), g(3, ELL));
    println!("memprofile: |x| = {n_bits}, w = {group_bits}, m = {ell} (live heap bytes, MB = 10^6 B)");
    mark("startup");

    let mut params = HashParams::sample(20260901, n_bits, group_bits, ell);
    let crs = params.num_matrices() * ell * ell * blmr::ring::N * 8;
    mark(&format!("CRS sampling (table itself {:.1} MB)", mb(crs)));
    if std::env::var("MEMPROFILE_PRECOMPUTE").is_ok_and(|v| v == "1") {
        params.precompute_spectra();
        mark("precompute_spectra (spectra of the whole CRS)");
    }

    let nz = Nizk1Params::sample(20260902, &params, R_DIM, COM_N, W_SLACK);
    mark("Nizk1Params (commitment key + spectra cache)");

    let mut rng = SimpleRng::new(42);
    let bits: Vec<bool> = (0..n_bits).map(|_| rng.next_bool()).collect();
    let groups = bits_to_groups(&params, &bits);

    let (_bx, _wit) = eval_h(&params, &groups);
    mark("eval_h (spectra of A + witness)");
    drop(_wit);
    drop(_bx);
    mark("drop the eval_h result");

    let bw = sample_blind(
        QueryTicket::insecure_for_tests(insecure_test_secret(20260903), 0),
        &params,
        &nz,
        &groups,
    );
    mark("sample_blind");

    let (st, pf, _tm) = proof::prove_nizk1_with_timings(&params, &nz, &groups, &bw);
    mark("prove_nizk1 (the whole prover)");

    assert!(proof::verify_nizk1(&params, &nz, &st, &pf), "verify failed");
    mark("verify_nizk1");

    drop(params);
    mark("after dropping the CRS");
    drop(nz);
    mark("after dropping the commitment key (= statement + proof + blind witness)");
}
