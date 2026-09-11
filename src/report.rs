
use crate::layout::{dims, Dims, MASK_ROWS, SLOTS};
use crate::nizk1::{bin_layout, BlindStatement, Nizk1Params};
use crate::params::HashParams;
use crate::relation::{num_b_rows, num_quotients, Nizk1Ctx};
use crate::ring::{RingElem, N};

pub mod hachi {
    pub const LOG_Q: usize = (64 - crate::field::Q.leading_zeros()) as usize;
    pub const D_PCS: usize = 1024;
    pub const D_GH: usize = 64;
    pub const K: usize = crate::ext_field::EXT_DEG;
    pub const B_PLUS_2: usize = 18;
    pub const RING_SWITCH_VARS: usize = 4;
    pub const FIXED_TRANSFORM_BYTES: usize = D_PCS * LOG_Q / 8;
    pub const FIXED_COMMIT_V_BYTES: usize = 4096;
    pub const GH_COMMIT_BYTES: usize = 4608;
    pub const GH_EVAL_ANCHOR_BYTES: f64 = 43.0 * 1024.0;
    pub const GH_EVAL_ANCHOR_N: f64 = 1_048_576.0;
    pub const DECOMP_BLOWUP: f64 = 5.0;
}

pub fn hachi_open_bytes(nv: usize, binary: bool, n_openings: usize) -> usize {
    use hachi::*;
    let fixed = FIXED_TRANSFORM_BYTES + FIXED_COMMIT_V_BYTES + GH_COMMIT_BYTES;
    let nv_red = nv.saturating_sub(RING_SWITCH_VARS);
    let sumcheck_bits = nv_red * K * LOG_Q * B_PLUS_2;
    let adapt_bits = (K - 1) * K * LOG_Q + D_GH * LOG_Q;
    let mut n_gh = (1u64 << nv_red) as f64 / D_GH as f64;
    if !binary {
        n_gh *= DECOMP_BLOWUP;
    }
    let gh_eval = GH_EVAL_ANCHOR_BYTES * (n_gh / GH_EVAL_ANCHOR_N).sqrt();
    fixed + n_openings * ((sumcheck_bits + adapt_bits).div_ceil(8) + gh_eval as usize)
}

pub const BP14_GADGET_LEN: usize = 8;

pub const TARGET_DELTA0: f64 = 1.0045;
pub const SIGMA: f64 = std::f64::consts::FRAC_1_SQRT_2;

pub fn min_lattice_dim(q: u64, sigma: f64, delta0: f64) -> f64 {
    let lq = (q as f64).log2();
    let lqs = (q as f64 / sigma).log2();
    lqs * lqs / (4.0 * lq * delta0.log2())
}

pub struct LatticeDims {
    pub key: usize,
    pub blind: usize,
    pub hide: usize,
    pub bind: usize,
    pub target: f64,
}

impl LatticeDims {
    pub fn all(&self) -> [(&'static str, usize); 4] {
        [
            ("key (m·N)", self.key),
            ("blind ((r_dim−m)·N)", self.blind),
            ("hide (w·N)", self.hide),
            ("bind (com_n·N)", self.bind),
        ]
    }
    pub fn shortfall(&self) -> Vec<(&'static str, usize)> {
        self.all().into_iter().filter(|&(_, n)| (n as f64) < self.target).collect()
    }
}

pub fn s1_for(log2_qx: f64) -> f64 {
    (0.5 * log2_qx + 4.0).exp2()
}

pub fn bf_gaussian(kappa: f64, h: f64, d: f64, s1: f64, beta_r: f64, s: f64, lm_d: f64) -> f64 {
    (std::f64::consts::LN_2 * (kappa + 2.0 + (h * d).log2()) / std::f64::consts::PI).sqrt()
        * (s1 + beta_r * s * lm_d.sqrt())
}

pub fn kappa_max(log_q: f64, p: f64, h: f64, d: f64, s1: f64, beta_r: f64, s: f64, lm_d: f64) -> f64 {
    let mut k = 32.0f64;
    for _ in 0..64 {
        let bf = bf_gaussian(k, h, d, s1, beta_r, s, lm_d);
        let next = log_q - p.log2() - 2.0 - (h * d).log2() - (2.0 * bf + 1.0).log2();
        if (next - k).abs() < 1e-9 {
            return next;
        }
        k = next;
    }
    k
}

pub struct ParamReport {
    pub q_bits: u32,
    pub n_ring: usize,
    pub ell: usize,
    pub n_bits: usize,
    pub group_bits: usize,
    pub r_dim: usize,
    pub com_n: usize,
    pub w_slack: usize,
    pub beta_r: u64,
    pub p_round: u64,
    pub lattice: LatticeDims,
    pub num_groups: usize,
    pub table_size: usize,
    pub rho_r_len: usize,
    pub rho_x_len: usize,
    pub hpack_len: usize,
    pub num_b_rows: usize,
    pub bin_rows: usize,
    pub bin_pad: usize,
    pub nv_bin: usize,
    pub rows: usize,
    pub nv: usize,
    pub layout: crate::layout::MergedLayout,
    pub akita_digits: usize,
    pub nv_c: usize,
    pub nv_u: usize,
    pub nv_h: usize,
    pub nv_i: usize,
    pub num_quotients: usize,
    pub crs_bytes: usize,
    pub crs_bytes_bp14: usize,
    pub s: f64,
    pub kappa_table: Vec<(f64, f64, f64, f64)>,
    pub ring_elem_bytes: usize,
    pub c_x_bytes: usize,
    pub c_r_bytes: usize,
    pub d_x_bytes: usize,
    pub statement_bytes: usize,
    pub d_x_bytes_if_raw_x: usize,
    pub pcs_batched_bytes: usize,
    pub pcs_unbatched_bytes: usize,
    pub pcs_full_penalty_bytes: usize,
    pub pcs_three_commitments_bytes: usize,
}

fn shape_only_statement(params: &HashParams, nz: &Nizk1Params) -> BlindStatement {
    BlindStatement {
        j: 0,
        c_x: vec![RingElem::zero(); params.ell],
        c_r: vec![RingElem::zero(); nz.com_r.out_len()],
        d_x: vec![RingElem::zero(); nz.com_x.out_len()],
    }
}

pub fn report(params: &HashParams, nz: &Nizk1Params, p_round: u64, beta_r: u64, s: f64) -> ParamReport {
    let q = crate::field::Q;
    let q_bits = 64 - q.leading_zeros();
    let m = params.ell;

    let st = shape_only_statement(params, nz);
    let ctx = Nizk1Ctx::new(params, nz, &st);
    let d: Dims = dims(params, Some(&ctx));
    let w = bin_layout(params, Some(nz));

    let dd = N as f64;
    let h = 1.0;
    let lm_d = (nz.r_dim + m) as f64 * dd;
    let kappa_table = [16.0f64, 32.0, 48.0, 64.0]
        .iter()
        .map(|&lq| {
            let s1 = s1_for(lq);
            let k = kappa_max(q_bits as f64, p_round as f64, h, dd, s1, beta_r as f64, s, lm_d);
            (lq, s1, bf_gaussian(k, h, dd, s1, beta_r as f64, s, lm_d), k)
        })
        .collect();

    let ring_bytes = N * 8;
    let crs_bytes = params.num_matrices() * m * m * ring_bytes;
    let crs_bytes_bp14 = params.table_size() * m * (m * BP14_GADGET_LEN) * ring_bytes;

    let ring_elem_bytes = N * (q_bits as usize).div_ceil(8);
    let c_x_bytes = m * ring_elem_bytes;
    let c_r_bytes = nz.com_r.out_len() * ring_elem_bytes;
    let d_x_bytes = nz.com_x.out_len() * ring_elem_bytes;
    let d_x_bytes_if_raw_x = (nz.com_x.com_n() + params.n_bits.div_ceil(N)) * ring_elem_bytes;

    let pcs_batched_bytes = hachi_open_bytes(d.nv, false, 5);
    let pcs_unbatched_bytes = 4 * hachi_open_bytes(d.nv, false, 1) + hachi_open_bytes(d.nv, false, 14);
    let pcs_full_penalty_bytes =
        hachi_open_bytes(d.nv, false, 5).saturating_sub(hachi_open_bytes(d.nv, true, 5));
    let old_full_rows = 2 * num_b_rows(params).next_power_of_two().max(d.c_cells);
    let old_nv_full = (old_full_rows * SLOTS).trailing_zeros() as usize;
    let pcs_three_commitments_bytes = hachi_open_bytes(d.nv_bin, true, 3)
        + hachi_open_bytes(old_nv_full, false, 1)
        + hachi_open_bytes(9, false, 1);
    let akita_digits = 7 * (1usize << d.nv);
    let _ = MASK_ROWS;

    let com_n = nz.com_r.com_n();
    let w_slack = nz.com_r.rho_len() - com_n - nz.com_r.msg_len();
    let lattice = LatticeDims {
        key: m * N,
        blind: nz.r_dim.saturating_sub(m) * N,
        hide: w_slack * N,
        bind: com_n * N,
        target: min_lattice_dim(q, SIGMA, TARGET_DELTA0),
    };

    ParamReport {
        q_bits,
        n_ring: N,
        ell: m,
        n_bits: params.n_bits,
        group_bits: params.group_bits,
        r_dim: nz.r_dim,
        com_n,
        w_slack,
        beta_r,
        p_round,
        lattice,
        num_groups: params.num_groups(),
        table_size: params.table_size(),
        rho_r_len: nz.com_r.rho_len(),
        rho_x_len: nz.com_x.rho_len(),
        hpack_len: nz.hpack_len,
        num_b_rows: num_b_rows(params),
        bin_rows: w.total,
        bin_pad: d.bin_pad,
        nv_bin: d.nv_bin,
        rows: d.merged.rows,
        nv: d.nv,
        layout: d.merged,
        akita_digits,
        nv_c: d.nv_c,
        nv_u: d.nv_u,
        nv_h: d.nv_h,
        nv_i: d.nv_i,
        num_quotients: num_quotients(params, Some(&ctx)),
        crs_bytes,
        crs_bytes_bp14,
        s,
        kappa_table,
        ring_elem_bytes,
        c_x_bytes,
        c_r_bytes,
        d_x_bytes,
        statement_bytes: c_x_bytes + c_r_bytes + d_x_bytes,
        d_x_bytes_if_raw_x,
        pcs_batched_bytes,
        pcs_unbatched_bytes,
        pcs_full_penalty_bytes,
        pcs_three_commitments_bytes,
    }
}

impl ParamReport {
    pub fn print(&self) {
        let kb = |b: usize| b as f64 / 1024.0;
        let mb = |b: usize| b as f64 / 1e6;
        println!("\n=== parameters (BLMR) ===");
        println!(
            "  q = 2^{} − {}   N = {}   m = {}   |x| = {}   w = {}   G = {}   2^w = {}",
            self.q_bits,
            (1u128 << self.q_bits) - crate::field::Q as u128,
            self.n_ring,
            self.ell,
            self.n_bits,
            self.group_bits,
            self.num_groups,
            self.table_size
        );
        println!(
            "  r_dim = {}   com_n = {}   w_slack = {}   rho_r_len = {}   rho_x_len = {}   hpack_len = {}",
            self.r_dim, self.com_n, self.w_slack, self.rho_r_len, self.rho_x_len, self.hpack_len
        );

        println!("\n=== lattice dimensions (all must be ≥ {:.0}, δ₀ = {}, σ = {:.3}) ===", self.lattice.target, TARGET_DELTA0, SIGMA);
        for (name, n) in self.lattice.all() {
            let ok = if (n as f64) >= self.lattice.target { "OK" } else { "**insufficient**" };
            println!("  {name:<22} = {n:>6}   {ok}");
        }

        println!("\n=== CRS ===");
        println!(
            "  BLMR  G·2^w·m²·N     = {:>10.1} MB   ({} blocks × {} symbols × {}×{} binary matrices, per-block)",
            mb(self.crs_bytes),
            self.num_groups,
            self.table_size,
            self.ell,
            self.ell
        );
        println!(
            "  BP14  2^w·m·(m·δ)·N  = {:>10.1} MB   ⇒ BLMR / BP14 = {:.2}× (δ = {} digits per coefficient, BP14 shares symbols)",
            mb(self.crs_bytes_bp14),
            self.crs_bytes as f64 / self.crs_bytes_bp14 as f64,
            BP14_GADGET_LEN
        );

        println!("\n=== cube dimensions (🔴 phase1v3: a single merged table, one commit) ===");
        let l = &self.layout;
        println!(
            "  binary block  row [0,{})  uses {} rows (SC_bin subcube, nv_bin {})",
            l.bin_rows, self.bin_rows, self.nv_bin
        );
        println!(
            "  b segment     row [{},{})  {} rows  | quotient segment row [{},{})  {} rows  | mask row {}",
            l.b_base, l.quot_base, self.num_b_rows, l.quot_base, l.mask_base, self.num_quotients, l.mask_base
        );
        println!(
            "  total {} rows → nv {} (one commitment, Akita digits 7·2^nv = {})",
            self.rows, self.nv, self.akita_digits
        );
        println!(
            "  nv_c {}   nv_u {}   nv_h {}   nv_i {}   quotients {}   ({} slots per row)",
            self.nv_c, self.nv_u, self.nv_h, self.nv_i, self.num_quotients, SLOTS
        );

        println!("\n=== LeOPaRd Thm 6/7 (correctness / uniqueness) -- 🔴 the Gaussian bound of eq (26) ===");
        println!(
            "  h = 1   d = N = {}   p = {}   β_r = {}   s = {:.2}   (ℓ+m)·d = {}",
            self.n_ring,
            self.p_round,
            self.beta_r,
            self.s,
            (self.r_dim + self.ell) * self.n_ring
        );
        println!("  log₂ Q_x |    s₁    |    B_f    | κ_max   (log q = {})", self.q_bits);
        for &(lq, s1, bf, k) in &self.kappa_table {
            println!("  {lq:>8.0} | 2^{:>5.1} | 2^{:>6.2} | {k:>5.1}", s1.log2(), bf.log2());
        }
        println!(
            "  ⚠️ the κ ≈ 40.7 in the BP14 report is **fictitious** (worst-case eq (25) implicitly assumes β = β₁ = 1);"
        );
        println!("     in plain OPRF mode LeOPaRd pins Q_x = 2^κ ⇒ the two are the same budget.");

        println!("\n=== communication per query (client → server) ===");
        println!("  1 ring element = N·⌈log q/8⌉ = {} B", self.ring_elem_bytes);
        println!("  C_x   {:>3} ring elements = {:>8.2} KB", self.ell, kb(self.c_x_bytes));
        println!(
            "  c_r   {:>3} ring elements = {:>8.2} KB",
            self.c_r_bytes / self.ring_elem_bytes,
            kb(self.c_r_bytes)
        );
        println!(
            "  d_x   {:>3} ring elements = {:>8.2} KB   (if the {} bits of x were committed instead: {:.2} KB)",
            self.d_x_bytes / self.ring_elem_bytes,
            kb(self.d_x_bytes),
            self.n_bits,
            kb(self.d_x_bytes_if_raw_x)
        );
        println!("  ---------------------------------------");
        println!("  statement            = {:>8.2} KB", kb(self.statement_bytes));
        println!(
            "  PCS opening (Hachi cost model, batched) = {:>6.2} KB  (unbatched: {:.2} KB)",
            kb(self.pcs_batched_bytes),
            kb(self.pcs_unbatched_bytes)
        );
        println!(
            "    ├ the whole table is costed as full-range ⇒ {:.2} KB more than all-bits (Akita has no mixed widths)",
            kb(self.pcs_full_penalty_bytes)
        );
        println!(
            "    └ compared with the three-commitment scheme (before phase1v3): {:.2} KB ⇒ merging saves {:.2} KB",
            kb(self.pcs_three_commitments_bytes),
            kb(self.pcs_three_commitments_bytes) - kb(self.pcs_batched_bytes)
        );
        println!("  ⚠️ for the size of the sumcheck transcript see `ser::ProofBytes::transcript` (measured locally)");
        println!(
            "\n  compared with LeOPaRd Table 4 (d=64, p=4, h=1, β_r=1): client online 9.59–58.75 KB (w/o NIZK)"
        );
    }
}

pub struct WRow {
    pub group_bits: usize,
    pub num_groups: usize,
    pub crs_bytes: usize,
    pub crs_sample_ms: f64,
    pub eval_h_ms: f64,
    pub prove_ms: f64,
    pub verify_ms: f64,
    pub proof_transcript_bytes: usize,
    pub proof_total_bytes: usize,
    pub rounds: usize,
    pub nv_bin: usize,
    pub nv: usize,
    pub nv_u: usize,
    pub pcs_batched_bytes: usize,
    pub e_t: usize,
    pub provable_log_q: f64,
}

pub fn measure_w(n_bits: usize, group_bits: usize, ell: usize, seed: u64) -> WRow {
    use std::time::Instant;
    let t0 = Instant::now();
    let params = HashParams::sample(seed, n_bits, group_bits, ell);
    let crs_sample_ms = t0.elapsed().as_secs_f64() * 1e3;

    let nz = Nizk1Params::sample(
        seed + 1,
        &params,
        crate::nizk1::R_DIM,
        crate::nizk1::COM_N,
        crate::nizk1::W_SLACK,
    );
    let mut rng = crate::transcript::SimpleRng::new(seed ^ 0x5EED);
    let bits: Vec<bool> = (0..n_bits).map(|_| rng.next_bool()).collect();
    let groups = crate::hash::bits_to_groups(&params, &bits);

    let t0 = Instant::now();
    let _ = crate::hash::eval_h(&params, &groups);
    let eval_h_ms = t0.elapsed().as_secs_f64() * 1e3;

    let bw = crate::nizk1::sample_blind(
        crate::nizk1::QueryTicket::insecure_for_tests(crate::rng::insecure_test_secret(seed + 2), 0),
        &params,
        &nz,
        &groups,
    );
    let t0 = Instant::now();
    let (st, proof) = crate::proof::prove_nizk1(&params, &nz, &groups, &bw);
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    assert!(crate::proof::verify_nizk1(&params, &nz, &st, &proof), "w={group_bits} failed to verify");
    let verify_ms = t0.elapsed().as_secs_f64() * 1e3;

    let r = report(&params, &nz, 2, 1, 16.08);
    let b = proof.size_breakdown();
    let ng = params.num_groups();
    let e_t = ng - 1;
    let provable_log_q = e_t as f64 * ((ell * N) as f64).log2() + (params.table_size() as f64).log2() / 2.0;

    WRow {
        group_bits,
        num_groups: ng,
        crs_bytes: r.crs_bytes,
        crs_sample_ms,
        eval_h_ms,
        prove_ms,
        verify_ms,
        proof_transcript_bytes: b.transcript(),
        proof_total_bytes: b.total(),
        rounds: proof.num_rounds(),
        nv_bin: r.nv_bin,
        nv: r.nv,
        nv_u: r.nv_u,
        pcs_batched_bytes: r.pcs_batched_bytes,
        e_t,
        provable_log_q,
    }
}

pub fn print_w_tradeoff(n_bits: usize, ell: usize, seed: u64) {
    let rows: Vec<WRow> = [4usize, 8].iter().map(|&g| measure_w(n_bits, g, ell, seed)).collect();
    println!("\n=== trade-off table for w (|x| = {n_bits}, m = {ell}) ===");
    let hdr = |name: &str| print!("  {name:<26}");
    hdr("");
    for r in &rows {
        print!("{:>14}", format!("w = {}", r.group_bits));
    }
    println!();
    macro_rules! row {
        ($name:expr, $f:expr) => {{
            hdr($name);
            for r in &rows {
                print!("{:>14}", $f(r));
            }
            println!();
        }};
    }
    row!("G (blocks)", |r: &WRow| format!("{}", r.num_groups));
    row!("2^w (symbols)", |r: &WRow| format!("{}", 1usize << r.group_bits));
    row!("CRS", |r: &WRow| format!("{:.2} MB", r.crs_bytes as f64 / 1e6));
    row!("CRS sampling", |r: &WRow| format!("{:.0} ms", r.crs_sample_ms));
    row!("eval_h", |r: &WRow| format!("{:.2} ms", r.eval_h_ms));
    row!("prove", |r: &WRow| format!("{:.1} ms", r.prove_ms));
    row!("verify", |r: &WRow| format!("{:.2} ms", r.verify_ms));
    row!("nv_u / nv / nv_bin", |r: &WRow| format!(
        "{}/{}/{}",
        r.nv_u, r.nv, r.nv_bin
    ));
    row!("sumcheck rounds", |r: &WRow| format!("{}", r.rounds));
    row!("proof (transcript)", |r: &WRow| format!("{} B", r.proof_transcript_bytes));
    row!("proof (incl. framing/stub)", |r: &WRow| format!("{} B", r.proof_total_bytes));
    row!("PCS opening (est.)", |r: &WRow| format!(
        "{:.1} KB",
        r.pcs_batched_bytes as f64 / 1024.0
    ));
    row!("e(T) = G−1", |r: &WRow| format!("{}", r.e_t));
    row!("provable log q", |r: &WRow| format!("{:.0}", r.provable_log_q));
    println!(
        "\n  🔵 the last two rows are the **provable** requirement of BP14 Thm 2.3 (`q ≥ p·r·√|T|·(m·N)^{{e(T)}}·ω(1)`)."
    );
    println!(
        "     our log q = 64 ⇒ both choices of w fall far short -- this is a deliberate heuristic stance,"
    );
    println!("     see the BP14 §2.1 quotation in the `report` module documentation for the rationale.");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(n: usize, g: usize, ell: usize, seed: u64) -> (HashParams, Nizk1Params) {
        let params = HashParams::sample(seed, n, g, ell);
        let nz = Nizk1Params::sample(
            seed + 1,
            &params,
            crate::nizk1::R_DIM,
            crate::nizk1::COM_N,
            crate::nizk1::W_SLACK,
        );
        (params, nz)
    }

    #[test]
    fn report_dims_match_the_real_prover() {
        for &(n, g, ell) in &[
            (8usize, 2usize, 1usize),
            (8, 4, 2),
            (12, 2, 3),
            (16, 4, crate::params::ELL),
        ] {
            let (params, nz) = setup(n, g, ell, 700);
            let r = report(&params, &nz, 2, 1, 16.08);
            let groups = crate::hash::bits_to_groups(&params, &vec![false; n]);
            let bw = crate::nizk1::sample_blind(
                crate::nizk1::QueryTicket::insecure_for_tests(
                    crate::rng::insecure_test_secret(702),
                    0,
                ),
                &params,
                &nz,
                &groups,
            );
            let (st, proof) = crate::proof::prove_nizk1(&params, &nz, &groups, &bw);
            assert!(crate::proof::verify_nizk1(&params, &nz, &st, &proof));
            let tag = format!("n={n} g={g} ell={ell}");
            assert_eq!(r.nv, proof.c.num_vars, "nv ({tag})");
            assert_eq!(r.akita_digits, 7 << r.nv, "akita_digits ({tag})");
            assert_eq!(r.nv_u + 1, proof.sc1.rounds.len(), "nv_u ({tag})");
            assert_eq!(r.nv_bin + 1, proof.sc_bin.rounds.len(), "nv_bin ↔ SC_bin ({tag})");
            assert_eq!(r.nv + 1, proof.sc_full.rounds.len(), "nv ↔ SC_full ({tag})");
            assert!(r.pcs_batched_bytes < r.pcs_three_commitments_bytes, "merging the tables is not cheaper ({tag})");
            assert_eq!(r.nv_h, proof.sc5.rounds.len() + g, "nv_h ({tag})");
            assert_eq!(r.statement_bytes, r.c_x_bytes + r.c_r_bytes + r.d_x_bytes, "{tag}");
            assert_eq!(
                r.crs_bytes * BP14_GADGET_LEN,
                r.crs_bytes_bp14 * r.num_groups,
                "CRS ratio ({tag})"
            );
        }
    }

    #[test]
    fn lattice_dims_meet_target() {
        let (params, nz) = setup(128, 4, crate::params::ELL, 720);
        let r = report(&params, &nz, 2, 1, 16.08);
        assert!(
            r.lattice.shortfall().is_empty(),
            "the lattice dimension is insufficient: {:?} (target {:.0})",
            r.lattice.shortfall(),
            r.lattice.target
        );
        let want = 2509.0;
        let t = min_lattice_dim(crate::field::Q, SIGMA, TARGET_DELTA0);
        assert!((t - want).abs() < 2.0, "the closed form for the target dimension drifted: {t} (expected {want})");
    }

    #[test]
    fn bf_formula_reproduces_the_leopard_row() {
        let (h, d, p, beta_r, s) = (1.0f64, 64.0f64, 4.0f64, 1.0f64, 16.1f64);
        let lm_d = (40.0 + 37.0) * d;
        let s1 = s1_for(32.0);
        assert!((s1.log2() - 20.0).abs() < 1e-9, "the fit for s₁ drifted: 2^{}", s1.log2());
        let bf = bf_gaussian(32.0, h, d, s1, beta_r, s, lm_d);
        assert!(
            (bf.log2() - 21.57).abs() < 0.1,
            "B_f = 2^{:.2}, expected 2^21.57 (the LeOPaRd κ=32 row)",
            bf.log2()
        );
        let need = 32.0 + 2.0 + (h * d).log2() + (2.0 * bf + 1.0).log2() + p.log2();
        assert!((need - 64.6).abs() < 0.2, "log q ≥ {need:.2}, expected 64.6 (64 in the table)");
    }

    #[test]
    fn kappa_max_is_a_fixed_point_of_eq_24() {
        let (h, d, p, beta_r, s) = (1.0f64, N as f64, 2.0f64, 1.0f64, 16.08f64);
        let lm_d = (crate::nizk1::R_DIM + crate::params::ELL) as f64 * d;
        for lq in [16.0f64, 32.0, 64.0] {
            let s1 = s1_for(lq);
            let k = kappa_max(64.0, p, h, d, s1, beta_r, s, lm_d);
            let bf = bf_gaussian(k, h, d, s1, beta_r, s, lm_d);
            let need = k + 2.0 + (h * d).log2() + (2.0 * bf + 1.0).log2() + p.log2();
            assert!((need - 64.0).abs() < 1e-6, "Q_x = 2^{lq}: log q back-solves to {need}, not 64");
        }
    }

    #[test]
    fn kappa_table_matches_the_phase2_plan() {
        let (params, nz) = setup(128, 4, crate::params::ELL, 710);
        let r = report(&params, &nz, 2, 1, 16.08);
        for &(lq, want) in &[(16.0f64, 36.9f64), (32.0, 29.4), (64.0, 13.8)] {
            let (_, _, _, got) = *r
                .kappa_table
                .iter()
                .find(|&&(l, _, _, _)| (l - lq).abs() < 1e-9)
                .unwrap_or_else(|| panic!("the table has no Q_x = 2^{lq}"));
            assert!(
                (got - want).abs() < 0.3,
                "Q_x = 2^{lq}: κ_max = {got:.2}, expected {want}"
            );
        }
        let lm_d = (r.r_dim + r.ell) as f64 * N as f64;
        let k16 = kappa_max(64.0, 2.0, 1.0, N as f64, 8192.0, 1.0, 16.08, lm_d);
        assert!((k16 - 36.1).abs() < 0.1, "s₁ = 2^13 of Table 2 should give 36.1, got {k16:.2}");
        assert!(
            r.kappa_table.iter().all(|&(_, _, _, k)| k < 40.0),
            "κ is back above 40 -- was it reverted to the worst-case eq (25)?"
        );
    }

    #[test]
    fn pcs_cost_model_is_monotone() {
        let (params, nz) = setup(128, 4, crate::params::ELL, 730);
        let r = report(&params, &nz, 2, 1, 16.08);
        assert!(r.pcs_full_penalty_bytes > 0, "full-range is unexpectedly not more expensive than binary");
        assert!(r.pcs_batched_bytes < r.pcs_unbatched_bytes, "batching is unexpectedly not cheaper");
        assert!(r.pcs_batched_bytes < r.pcs_three_commitments_bytes, "merging the tables is not cheaper");
        assert_eq!(r.akita_digits, 7 << r.nv);
        assert!(
            hachi_open_bytes(20, true, 1) < hachi_open_bytes(20, false, 1),
            "the cost models for binary and full-range are not separated"
        );
    }

    #[test]
    #[ignore]
    fn w_tradeoff_runs_and_is_directionally_right() {
        print_w_tradeoff(128, crate::params::ELL, 20260901);
        let a = measure_w(128, 4, crate::params::ELL, 20260901);
        let b = measure_w(128, 8, crate::params::ELL, 20260901);
        assert_eq!(a.crs_bytes * 8, b.crs_bytes, "CRS: 2^w differs by 16×, G by 2× ⇒ 8×");
        assert_eq!(a.e_t, 31);
        assert_eq!(b.e_t, 15);
        assert!(a.provable_log_q > b.provable_log_q, "the provable log q for w=4 should be larger");
        assert!(a.nv_u < b.nv_u, "the u-cube for w=4 should be smaller");
        assert_eq!(a.nv, b.nv, "the merged table is 2^18 for both choices of w (phase1v3 §2.4)");
    }
}
