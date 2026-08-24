use crate::ext_field::EXT_DEG;
use crate::nizk1::Nizk1Params;
use crate::params::HashParams;
use crate::relation::{g_pad, num_m_rows, num_quotients, phase_a_cells};
use crate::ring::{M_BIT_ROWS, N};

use crate::proof::{T_COEF_VARS, W_COEF_VARS};

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
            ("key (ℓ·N)", self.key),
            ("blind ((r_dim−ℓ)·N)", self.blind),
            ("hide (w·N)", self.hide),
            ("bind (com_n·N)", self.bind),
        ]
    }
    pub fn shortfall(&self) -> Vec<(&'static str, usize)> {
        self.all().into_iter().filter(|&(_, n)| (n as f64) < self.target).collect()
    }
}

pub struct ParamReport {
    pub q_bits: u32,
    pub n_ring: usize,
    pub delta: usize,
    pub ell: usize,
    pub n_bits: usize,
    pub group_bits: usize,
    pub r_dim: usize,
    pub com_n: usize,
    pub w_slack: usize,
    pub beta_r: u64,
    pub p_round: u64,
    pub lattice: LatticeDims,
    pub rho_r_len: usize,
    pub rho_x_len: usize,
    pub hpack_len: usize,
    pub num_m_rows: usize,
    pub extra_w_rows: usize,
    pub kw_pad: usize,
    pub nv_w: usize,
    pub nv_c: usize,
    pub nv_u: usize,
    pub nv_h: usize,
    pub nv_t: usize,
    pub num_quotients: usize,
    pub bf: f64,
    pub pu_log2: f64,
    pub kappa: f64,
    pub min_log_q: Vec<(u32, f64)>,
    pub ring_elem_bytes: usize,
    pub c_x_bytes: usize,
    pub c_r_bytes: usize,
    pub d_x_bytes: usize,
    pub statement_bytes: usize,
    pub d_x_bytes_if_raw_x: usize,
    pub pcs_batched_bytes: usize,
    pub pcs_unbatched_bytes: usize,
    pub pcs_mask_own_commitment_bytes: usize,
}

fn nv_u_of(nv_c: usize, group_bits: usize) -> usize {
    nv_c + group_bits
}

pub fn report(params: &HashParams, nz: &Nizk1Params, p_round: u64, beta_r: u64) -> ParamReport {
    let q = crate::field::Q;
    let q_bits = 64 - q.leading_zeros();

    let w = crate::nizk1::w_layout(params, Some(nz));
    let kw_pad = (w.total + crate::proof::MASK_COEF_ROWS).next_power_of_two();
    let c_cells = (phase_a_cells(params) + nz.com_r.out_len() + nz.com_x.out_len())
        .next_power_of_two();
    let nv_c = c_cells.trailing_zeros() as usize;
    let nv_i = g_pad(params).trailing_zeros() as usize;

    let d = N as f64;
    let bf = nz.r_dim as f64 * d * beta_r as f64 + 1.0;
    let pu = d * (2.0 * bf + 1.0) / (q / p_round) as f64;
    let pu_log2 = pu.log2();
    let base = (d * (2.0 * bf + 1.0) * p_round as f64).log2();
    let min_log_q = [16u32, 32, 64].iter().map(|&k| (k, base + k as f64)).collect();

    let ring_elem_bytes = N * (q_bits as usize).div_ceil(8);
    let c_x_bytes = params.ell * ring_elem_bytes;
    let c_r_bytes = nz.com_r.out_len() * ring_elem_bytes;
    let d_x_bytes = nz.com_x.out_len() * ring_elem_bytes;
    let d_x_bytes_if_raw_x =
        (nz.com_x.com_n() + params.n_bits.div_ceil(N)) * ring_elem_bytes;

    let nv_w = (kw_pad << W_COEF_VARS).trailing_zeros() as usize;
    let nv_h = nv_i + params.group_bits;
    let nv_t = (c_cells << T_COEF_VARS).trailing_zeros() as usize;
    let pcs_batched_bytes = hachi_open_bytes(nv_w, true, 4) + hachi_open_bytes(nv_t, false, 1);
    let pcs_unbatched_bytes =
        4 * hachi_open_bytes(nv_w, true, 1) + hachi_open_bytes(nv_t, false, 1);
    let mask_fq = EXT_DEG
        * ((nv_u_of(nv_c, params.group_bits) * 4) + (nv_t * 3) + (nv_w * 4) + (nv_i * 3));
    let pcs_mask_own_commitment_bytes =
        hachi_open_bytes(mask_fq.next_power_of_two().trailing_zeros() as usize, false, 2);

    let com_n = nz.com_r.com_n();
    let w_slack = nz.com_r.rho_len() - com_n - nz.com_r.msg_len();
    let lattice = LatticeDims {
        key: params.ell * N,
        blind: nz.r_dim.saturating_sub(params.ell) * N,
        hide: w_slack * N,
        bind: com_n * N,
        target: min_lattice_dim(q, SIGMA, TARGET_DELTA0),
    };

    ParamReport {
        q_bits,
        n_ring: N,
        delta: M_BIT_ROWS,
        ell: params.ell,
        n_bits: params.n_bits,
        group_bits: params.group_bits,
        r_dim: nz.r_dim,
        com_n,
        w_slack,
        beta_r,
        p_round,
        lattice,
        rho_r_len: nz.com_r.rho_len(),
        rho_x_len: nz.com_x.rho_len(),
        hpack_len: nz.hpack_len,
        num_m_rows: num_m_rows(params),
        extra_w_rows: crate::nizk1::extra_w_rows(params, Some(nz)),
        kw_pad,
        nv_w,
        nv_c,
        nv_u: nv_u_of(nv_c, params.group_bits),
        nv_h,
        nv_t,
        num_quotients: num_quotients_of(params, nz),
        bf,
        pu_log2,
        kappa: -pu_log2,
        min_log_q,
        ring_elem_bytes,
        c_x_bytes,
        c_r_bytes,
        d_x_bytes,
        statement_bytes: c_x_bytes + c_r_bytes + d_x_bytes,
        d_x_bytes_if_raw_x,
        pcs_batched_bytes,
        pcs_unbatched_bytes,
        pcs_mask_own_commitment_bytes,
    }
}

fn num_quotients_of(params: &HashParams, nz: &Nizk1Params) -> usize {
    num_quotients(params, None) + nz.com_r.out_len() + nz.com_x.out_len()
}

impl ParamReport {
    pub fn print(&self) {
        let kb = |b: usize| b as f64 / 1024.0;
        println!("\n=== parameters ===");
        println!(
            "  q = 2^{} − {}   N = {}   δ = {}   ℓ = {}   |x| = {}   g = {}",
            self.q_bits,
            (1u128 << self.q_bits) - crate::field::Q as u128,
            self.n_ring,
            self.delta,
            self.ell,
            self.n_bits,
            self.group_bits
        );
        println!(
            "  r_dim = {}   com_n = {}   w = {}   rho_r_len = {}   rho_x_len = {}   hpack_len = {}",
            self.r_dim, self.com_n, self.w_slack, self.rho_r_len, self.rho_x_len, self.hpack_len
        );
        println!("\n=== lattice dimensions (all must be >= {:.0}, delta_0 = {}, sigma = {:.3}) ===", self.lattice.target, TARGET_DELTA0, SIGMA);
        for (name, n) in self.lattice.all() {
            let ok = if (n as f64) >= self.lattice.target { "OK" } else { "**TOO LOW**" };
            println!("  {name:<16} = {n:>6}   {ok}");
        }
        println!("\n=== cube dimensions ===");
        println!(
            "  num_m_rows {} + extra {} → kw_pad {} → nv_w {}",
            self.num_m_rows, self.extra_w_rows, self.kw_pad, self.nv_w
        );
        println!(
            "  nv_c {}   nv_u {}   nv_h {}   nv_t {}   {} quotients",
            self.nv_c, self.nv_u, self.nv_h, self.nv_t, self.num_quotients
        );
        println!("\n=== LeOPaRd Thm 6/7 (correctness / uniqueness) ===");
        println!(
            "  β_r = {}   p = {}   Bf = (m+ℓ)·d·β_r + 1 = {:.0}",
            self.beta_r, self.p_round, self.bf
        );
        println!("  pu = h·d·(2Bf+1)/⌊q/p⌋ = 2^{:.1}  ⇒  κ ≈ {:.1}", self.pu_log2, self.kappa);
        for (k, lq) in &self.min_log_q {
            println!("    kappa = {:<3} requires log q >= {:.1}", k, lq);
        }
        println!("\n=== communication per query (client -> server) ===");
        println!("  1 ring element = N*ceil(log q/8) = {} B", self.ring_elem_bytes);
        println!("  C_x   {:>3} ring elements = {:>8.2} KB", self.ell, kb(self.c_x_bytes));
        println!(
            "  c_r   {:>3} ring elements = {:>8.2} KB   <- decision B1 (no preprocessing) => sent online",
            self.c_r_bytes / self.ring_elem_bytes,
            kb(self.c_r_bytes)
        );
        println!(
            "  d_x   {:>3} ring elements = {:>8.2} KB   (committing to the {} bits of x instead: {:.2} KB)",
            self.d_x_bytes / self.ring_elem_bytes,
            kb(self.d_x_bytes),
            self.n_bits,
            kb(self.d_x_bytes_if_raw_x)
        );
        println!("  ---------------------------------------");
        println!("  statement            = {:>8.2} KB", kb(self.statement_bytes));
        println!(
            "  PCS opening (Hachi cost model, batched)= {:>6.2} KB  (unbatched: {:.2} KB)",
            kb(self.pcs_batched_bytes),
            kb(self.pcs_unbatched_bytes)
        );
        println!(
            "    +- ZK masks folded into the linear opening of c_w; a fourth commitment would cost {:.2} KB more",
            kb(self.pcs_mask_own_commitment_bytes)
        );
        println!("  [!] for the sumcheck transcript size see the `proof size` line (measured locally)");
        println!(
            "\n  compare LeOPaRd Table 4 (d=64, h=1, beta_r=1): client online 9.59-58.75 KB (w/o NIZK)"
        );
        println!("  + NIZK estimated at 45 KB (their LaBRADOR+LNP22 estimate)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_dims_match_the_real_prover() {
        for &(n, g, ell) in
            &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 8, 1), (128, 8, crate::params::ELL)]
        {
            let params = HashParams::sample(700, n, g, ell);
            let nz = Nizk1Params::sample(
                701,
                &params,
                crate::nizk1::R_DIM,
                crate::nizk1::COM_N,
                crate::nizk1::W_SLACK,
            );
            let r = report(&params, &nz, 2, 1);
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
            assert_eq!(r.nv_w, proof.c_w.num_vars, "nv_w  (n={n} g={g} ell={ell})");
            assert_eq!(
                r.nv_h,
                proof.sc5_onehot.rounds.len() + g,
                "nv_h  (n={n} g={g} ell={ell})"
            );
            assert_eq!(r.nv_t, proof.c_t.num_vars, "nv_t  (n={n} g={g} ell={ell})");
            assert_eq!(r.nv_u + 1, proof.sc1_bilinear.rounds.len(), "nv_u");
            assert_eq!(r.statement_bytes, r.c_x_bytes + r.c_r_bytes + r.d_x_bytes);
        }
    }

    #[test]
    fn lattice_dims_meet_target() {
        let params = HashParams::sample(720, 128, 8, crate::params::ELL);
        let nz = Nizk1Params::sample(721, &params, crate::nizk1::R_DIM, crate::nizk1::COM_N, crate::nizk1::W_SLACK);
        let r = report(&params, &nz, 2, 1);
        assert!(
            r.lattice.shortfall().is_empty(),
            "lattice dimension too low: {:?} (target {:.0})",
            r.lattice.shortfall(),
            r.lattice.target
        );
        let want = 2509.0;
        let t = min_lattice_dim(crate::field::Q, SIGMA, TARGET_DELTA0);
        assert!((t - want).abs() < 2.0, "the closed form for the target dimension drifted: {t} (expected {want})");
    }

    #[test]
    fn kappa_and_min_log_q_are_consistent() {
        let params = HashParams::sample(710, 8, 2, crate::params::ELL);
        let nz = Nizk1Params::sample(
            711,
            &params,
            crate::nizk1::R_DIM,
            crate::nizk1::COM_N,
            crate::nizk1::W_SLACK,
        );
        let r = report(&params, &nz, 2, 1);
        let base = (r.n_ring as f64 * (2.0 * r.bf + 1.0) * r.p_round as f64).log2();
        for (k, lq) in &r.min_log_q {
            assert!((lq - base - *k as f64).abs() < 1e-9, "the inversion for kappa={k} is inconsistent");
        }
        let (lo, hi) = (40.0, 41.0);
        assert!(r.kappa > lo && r.kappa < hi, "kappa = {} is outside the expected range ({lo}, {hi})", r.kappa);
    }
}
