
use blmr::hash::{bits_to_groups, check_witness, eval_h, eval_h_naive};
use blmr::layout::dims;
use blmr::nizk1::{sample_blind, Nizk1Params, QueryTicket, R_DIM, COM_N, W_SLACK};
use blmr::params::{HashParams, ELL};
use blmr::proof::{self, N_PUB_SCALARS};
use blmr::report;
use blmr::relation::Nizk1Ctx;
use blmr::rng::insecure_test_secret;
use blmr::transcript::SimpleRng;
use std::time::{Duration, Instant};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (n_bits, group_bits, ell) = (g(1, 128), g(2, 4), g(3, ELL));
    let bench_mode = a.iter().any(|s| s == "--bench");
    let bench = |group: &str, name: &str, d: Duration| {
        if bench_mode {
            println!("@bench\t{group}\t{name}\t{:.6}", d.as_secs_f64() * 1e3);
        }
    };
    println!(
        "BLMR NIZK1 demo: |x| = {n_bits}, w = {group_bits}, m = {ell}           (G = {} blocks, 2^w = {} symbols)",
        n_bits / group_bits,
        1usize << group_bits
    );

    let t0 = Instant::now();
    let params = HashParams::sample(20260901, n_bits, group_bits, ell);
    let ring_elems = params.num_matrices() * ell * ell;
    println!(
        "CRS sampling ({} blocks × {} symbols × {ell}×{ell} binary matrices = {ring_elems} ring elements, {:.1} MB): {:?}",
        params.num_groups(),
        params.table_size(),
        (ring_elems * blmr::ring::N * 8) as f64 / 1e6,
        t0.elapsed()
    );

    let mut rng = SimpleRng::new(42);
    let bits: Vec<bool> = (0..n_bits).map(|_| rng.next_bool()).collect();
    let groups = bits_to_groups(&params, &bits);

    let t0 = Instant::now();
    let (bx, wit) = eval_h(&params, &groups);
    println!("eval_h ({} steps × {}² ring mults): {:?}", groups.len() - 1, ell, t0.elapsed());

    assert!(check_witness(&params, &bx, &groups, &wit), "the witness does not satisfy the relation");
    println!("check_witness: OK ({} b_i, quotients for {} steps)", wit.b.len(), wit.steps());

    let t0 = Instant::now();
    let naive = eval_h_naive(&params, &groups);
    assert_eq!(bx, naive, "the column trick disagrees with the chained m×m matrix product");
    println!("eval_h_naive (m³ per step): {:?} -- bit-for-bit identical to the column trick ✅", t0.elapsed());

    println!("\n=== Phase A (statement = plaintext B_x) ===");
    let (bx2, pf, tm) = proof::prove_with_timings(&params, &groups);
    assert_eq!(bx, bx2);
    for (name, d) in &tm.0 {
        println!("  {name:<16} {d:?}");
    }
    println!("  {:<16} {:?}", "prove total", tm.total());
    bench("A.prove", "total", tm.total());
    let t0 = Instant::now();
    assert!(proof::verify(&params, &bx, &pf), "Phase A verify failed");
    let t_verify = t0.elapsed();
    println!("  verify           {t_verify:?}");
    bench("A.verify", "total", t_verify);

    println!("\n=== Phase B (NIZK1: statement = C_x, c_r, d_x) ===");
    let nz = Nizk1Params::sample(20260902, &params, R_DIM, COM_N, W_SLACK);
    let t0 = Instant::now();
    let bw = sample_blind(
        QueryTicket::insecure_for_tests(insecure_test_secret(20260903), 0),
        &params,
        &nz,
        &groups,
    );
    let t_blind = t0.elapsed();
    println!("  {:<16} {t_blind:?}", "sample_blind");
    bench("B.client", "sample_blind", t_blind);
    let (st, pfb, tmb) = proof::prove_nizk1_with_timings(&params, &nz, &groups, &bw);
    for (name, d) in &tmb.0 {
        println!("  {name:<16} {d:?}");
        bench("B.prove", name, *d);
    }
    println!("  {:<16} {:?}", "prove total", tmb.total());
    bench("B.prove", "total", tmb.total());
    let t0 = Instant::now();
    let (ok, tmv) = proof::verify_nizk1_with_timings(&params, &nz, &st, &pfb);
    let t_verify = t0.elapsed();
    assert!(ok, "Phase B verify failed");
    println!("  verify           {t_verify:?}");
    for (name, d) in &tmv.0 {
        println!("    {name:<14} {d:?}");
        bench("B.verify", name, *d);
    }
    bench("B.verify", "total", t_verify);

    let ctx = Nizk1Ctx::new(&params, &nz, &st);
    let d = dims(&params, Some(&ctx));
    println!(
        "\ndimensions: nv={} (single commitment, {} rows) nv_u={} nv_bin={} nv_i={} (binary block {} rows)",
        d.nv, d.merged.rows, d.nv_u, d.nv_bin, d.nv_i, d.bin_pad
    );
    let b = pfb.size_breakdown();
    println!(
        "proof: {} sumcheck rounds, {N_PUB_SCALARS} public scalars, transcript {} B (incl. framing/stub {} B), fingerprint {:#018x}",
        pfb.num_rounds(),
        b.transcript(),
        b.total(),
        proof::proof_fingerprint(&pfb)
    );
    if bench_mode {
        println!("@info\tnv\t{}", d.nv);
        println!("@info\ttranscript_bytes\t{}", b.transcript());
        println!("@info\trounds\t{}", pfb.num_rounds());
        println!("@info\tfingerprint\t{:#018x}", proof::proof_fingerprint(&pfb));
    }

    report::report(&params, &nz, 2, 1, 16.08).print();
    if a.iter().any(|s| s == "--tradeoff") {
        report::print_w_tradeoff(n_bits, ell, 20260901);
    }
}
