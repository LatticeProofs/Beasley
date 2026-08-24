use std::time::Instant;
use voprf::ext_field::FqExt;
use voprf::hash::{bits_to_groups, eval_h};
use voprf::params::HashParams;
use voprf::nizk1::{sample_blind, Nizk1Params, QueryCounter, COM_N, R_DIM, W_SLACK};
use voprf::proof::{prove, proof_fingerprint, prove_nizk1_with_timings, prove_with_timings, verify, verify_nizk1};
use voprf::relation::{build_rows, compute_quotients};
use voprf::rng::insecure_test_secret;
use voprf::transcript::SimpleRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_bits: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let group_bits: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let ell: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(voprf::params::ELL);
    let nizk1: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    println!(
        "phase_bench: n_bits = {n_bits}, group_bits = {group_bits}, ell = {ell}, nizk1 = {nizk1}"
    );

    let t = Instant::now();
    let params = HashParams::sample(20260713, n_bits, group_bits, ell);
    println!("CRS precompute:    {:?}  ({} rows)", t.elapsed(), params.table_size());

    let mut rng = SimpleRng::new(42);
    let bits: Vec<bool> = (0..n_bits).map(|_| rng.next_bool()).collect();
    let groups = bits_to_groups(&params, &bits);

    let t = Instant::now();
    let (ch, wit) = eval_h(&params, &groups);
    println!("eval_h:            {:?}", t.elapsed());

    {
        use voprf::ntt::neg_and_quotient_rows;
        use voprf::hash::gadget_decompose_vec;
        use rayon::prelude::*;
        let t = Instant::now();
        let specs: Vec<Vec<_>> = groups.par_iter().map(|&v| params.spectra_for(v)).collect();
        let t_spec = t.elapsed();
        let ng = params.num_groups();
        let ml = params.ml();
        let t = Instant::now();
        let mut nmul = 0usize;
        let mut y: Vec<_> =
            (0..params.ell).map(|r| params.a(groups[ng - 1], r, 0).clone()).collect();
        for i in (1..=ng - 1).rev() {
            let md = gadget_decompose_vec(&y);
            nmul += params.ell * ml;
            y = neg_and_quotient_rows(&specs[i - 1], params.ell, &md)
                .into_iter()
                .map(|(r, _)| r)
                .collect();
        }
        let t_loop = t.elapsed();
        println!(
            "  eval_h split: spectra(rayon) {:?} | sequential loop {:?}  ({} {}-NTT inner-product terms)",
            t_spec,
            t_loop,
            nmul,
            2 * voprf::ring::N
        );
    }

    let t = Instant::now();
    let q = compute_quotients(&params, &wit, None, None);
    println!("compute_quotients: {:?}  ({} quotients)", t.elapsed(), q.len());

    let mut r = SimpleRng::new(7);
    let alpha = r.next_fq4();
    let t = Instant::now();
    let rows = build_rows(&params, &ch, alpha, None);
    println!(
        "build_rows(a_base):{:?}  ({} symbols x {} rows x {} cols = {} length-{} evaluations; u-side flattened width {})",
        t.elapsed(),
        rows.a_base.len(),
        rows.a_base[0].len(),
        rows.a_base[0][0].len(),
        rows.a_base.len() * rows.a_base[0].len() * rows.a_base[0][0].len(),
        voprf::ring::N,
        rows.a_base[0][0].len() * voprf::ring::DIGIT_BITS
    );
    let _ = std::hint::black_box(rows.a_base[0][0][0] + FqExt::ONE);

    let _ = prove(&params, &groups);

    let nz = Nizk1Params::sample(31337, &params, R_DIM, COM_N, W_SLACK);
    let mut ctr = QueryCounter::new(insecure_test_secret(4242));
    let bw = sample_blind(ctr.issue(), &params, &nz, &groups);

    {
        use voprf::nizk1::{blind_statement, derive_ar};
        let (ch2, _) = eval_h(&params, &groups);
        let (st, _) = blind_statement(&params, &nz, &ch2, &bw);
        let t = Instant::now();
        let ar = derive_ar(nz.r_dim, params.ell, &st.c_r);
        let d = t.elapsed();
        let fq = nz.r_dim * params.ell * voprf::ring::N;
        println!(
            "derive_ar:         {:?}  ({} F_q = {} KB XOF; r_dim={} ell={})",
            d,
            fq,
            fq * voprf::field::FQ_BYTES / 1024,
            nz.r_dim,
            params.ell
        );
        let _ = std::hint::black_box(&ar[0][0].c[0]);
    }

    let t = Instant::now();
    let (ch, proof, tm, st_b) = if nizk1 == 1 {
        let (st, p, tm) = prove_nizk1_with_timings(&params, &nz, &groups, &bw);
        (st.c_x.clone(), p, tm, Some(st))
    } else {
        let (c, p, tm) = prove_with_timings(&params, &groups);
        (c, p, tm, None)
    };
    let total = t.elapsed();
    println!("prove total:       {total:?}");
    for (name, d) in &tm.0 {
        println!(
            "  {:<20} {:>10.3?}  {:>5.1}%",
            name,
            d,
            d.as_secs_f64() / total.as_secs_f64() * 100.0
        );
    }

    let sc = [
        ("SC1", &proof.sc1_bilinear),
        ("SC_quot", &proof.sc_quotient),
        ("SC_batched", &proof.sc_batched),
        ("SC5", &proof.sc5_onehot),
    ];
    let bytes = proof.to_bytes();
    let b = proof.size_breakdown();
    assert_eq!(bytes.len(), b.total(), "serialized length disagrees with the size breakdown");
    println!(
        "proof size:        {} B transcript (sumcheck {} B + plain scalars {} B; {} rounds; {})",
        b.transcript(),
        b.sumcheck,
        b.public_scalars,
        proof.num_rounds(),
        sc.iter()
            .map(|(n, p)| format!("{n}:{}", p.rounds.len()))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  serialized total:  {} B (+ framing {} B + commitment STUB {} B)",
        bytes.len(),
        b.framing,
        b.commitments_stub
    );

    println!("proof fingerprint: {:#018x}", proof_fingerprint(&proof));

    let t = Instant::now();
    let ok = match &st_b {
        Some(st) => verify_nizk1(&params, &nz, st, &proof),
        None => verify(&params, &ch, &proof),
    };
    println!("verify total:      {:?} -> {}", t.elapsed(), if ok { "ACCEPT" } else { "REJECT" });

    if nizk1 == 1 {
        voprf::report::report(&params, &nz, 2, 1).print();
        println!(
            "  sumcheck transcript                  = {:>8.2} KB (**measured**: ser::to_bytes)",
            b.transcript() as f64 / 1024.0
        );
    }
}
