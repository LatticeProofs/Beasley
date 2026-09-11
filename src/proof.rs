
use crate::ext_field::{FqExt, LazyExtSum, EXT_DEG};
use crate::field::Fq;
use crate::hash::eval_h;
use crate::layout::{
    alpha_tensor_eval, bin_prefix, build_bin_table, build_merged_table, dims, h_bit, h_cell,
    half_point, hpack_point, Dims, ALPHA_LIMIT, MASK_ROWS, SLOTS, SLOT_VARS,
};
use crate::mle::{eq_at_index, eq_eval, eq_table};
use crate::nizk1::{blind_statement, BlindStatement, BlindWitness, Nizk1Params};
use crate::params::HashParams;
use crate::pcs;
use crate::relation::{
    b_row, build_rows, compute_quotients, constraints, u_src, u_width, CKind, ConstraintMeta,
    LinRow, Nizk1Ctx, Rows,
};
use crate::ring::{RingElem, N};
use crate::sumcheck::{self, ClaimMask, LinMask, SumcheckProof};
use crate::transcript::Transcript;
use rayon::prelude::*;

pub const SIG_COEFS: usize = sumcheck::LIN_MASK_COEFS;
pub const R_COEFS: usize = 3;

pub(crate) const SIG_BIN: usize = 0;
pub(crate) const SIG_FULL: usize = SIG_COEFS;
pub(crate) const SIG_H: usize = 2 * SIG_COEFS;
pub(crate) const R_B: usize = 3 * SIG_COEFS;
pub(crate) const R_Q: usize = R_B + R_COEFS;
pub(crate) const R_U: usize = R_Q + R_COEFS;
pub(crate) const R_N: usize = R_U + R_COEFS;
pub const OPENZK_VALS: usize = 3 * SIG_COEFS + 4 * R_COEFS;
const _: () = assert!(R_N + R_COEFS == OPENZK_VALS);

const W_ROUND_DEG_3: usize = 3;
const W_ROUND_DEG_2: usize = 2;

fn z_eval(pt: &[FqExt]) -> FqExt {
    pt.iter().fold(FqExt::ONE, |a, &z| a * z * (FqExt::ONE - z))
}

fn ind_eval(pt: &[FqExt]) -> FqExt {
    pt.iter().fold(FqExt::ONE, |a, &z| a * (FqExt::ONE - z))
}

fn mask_shapes(d: &Dims) -> [Vec<usize>; 4] {
    [
        sumcheck::round_degs(d.nv_u, 3, 6, W_ROUND_DEG_3),
        sumcheck::round_degs(d.nv, 2, 4, W_ROUND_DEG_2),
        sumcheck::round_degs(d.nv_bin, 3, 7, W_ROUND_DEG_3),
        vec![2; d.nv_i],
    ]
}

fn mask_bases(d: &Dims) -> [usize; 5] {
    let sh = mask_shapes(d);
    let mut out = [0usize; 5];
    let mut acc = d.merged.mask_base * SLOTS;
    for i in 0..4 {
        out[i] = acc;
        acc += EXT_DEG * sumcheck::Masker::n_coefs(&sh[i]);
    }
    out[4] = acc;
    out
}

fn mask_start(d: &Dims) -> usize {
    d.merged.mask_base * SLOTS
}

#[inline]
fn openzk_base(d: &Dims, i: usize) -> usize {
    debug_assert!(i < OPENZK_VALS);
    mask_bases(d)[4] + EXT_DEG * i
}

pub fn mask_table_len(d: &Dims) -> usize {
    mask_bases(d)[4] + EXT_DEG * OPENZK_VALS - mask_start(d)
}

fn assert_mask_fits(d: &Dims) {
    assert!(
        mask_table_len(d) <= MASK_ROWS * SLOTS,
        "masking needs {} slots, MASK_ROWS = {MASK_ROWS} only provides {} (increase layout::MASK_ROWS)",
        mask_table_len(d),
        MASK_ROWS * SLOTS
    );
}

fn mask_weights(base: usize, w: &[FqExt]) -> Vec<(usize, FqExt)> {
    let mut out = Vec::with_capacity(w.len() * EXT_DEG);
    for (j, &wj) in w.iter().enumerate() {
        for a in 0..EXT_DEG {
            let mut basis = FqExt::ZERO;
            basis.0[a] = Fq::ONE;
            out.push((base + EXT_DEG * j + a, basis * wj));
        }
    }
    out
}

fn write_mask_coefs(zm: &mut [Fq], ms: &[sumcheck::Masker], bases: &[usize; 5]) {
    for (i, m) in ms.iter().enumerate() {
        for (j, c) in m.coeffs_flat().enumerate() {
            for a in 0..EXT_DEG {
                zm[bases[i] + EXT_DEG * j + a] = c.0[a];
            }
        }
    }
}

fn write_openzk_vals(d: &Dims, zm: &mut [Fq], vals: &[FqExt; OPENZK_VALS]) {
    let base = mask_bases(d)[4];
    for (j, v) in vals.iter().enumerate() {
        for a in 0..EXT_DEG {
            zm[base + EXT_DEG * j + a] = v.0[a];
        }
    }
}

fn n_bin_weights(lambda_b: FqExt, c: FqExt) -> Vec<FqExt> {
    ClaimMask::weights_eval(R_COEFS - 1, c).iter().map(|&p| lambda_b * p).collect()
}

fn n_full_weights(
    lambda_f: FqExt,
    gamma: FqExt,
    z_u: FqExt,
    xn1: FqExt,
    c: FqExt,
) -> Vec<FqExt> {
    let pw = ClaimMask::weights_eval(R_COEFS - 1, c);
    let mut out = Vec::with_capacity(4 * R_COEFS);
    out.extend(pw.iter().map(|&p| lambda_f * p));
    out.extend(pw.iter().map(|&p| (lambda_f * xn1 + FqExt::ONE) * p));
    out.extend(pw.iter().map(|&p| lambda_f * gamma * z_u * p));
    out.extend(pw.iter().map(|&p| -lambda_f * p));
    out
}

fn n_full_coefs(
    ozk: &[FqExt; OPENZK_VALS],
    lambda_f: FqExt,
    gamma: FqExt,
    z_u: FqExt,
    xn1: FqExt,
) -> Vec<FqExt> {
    (0..R_COEFS)
        .map(|k| {
            lambda_f * (ozk[R_B + k] + gamma * z_u * ozk[R_U + k] + xn1 * ozk[R_Q + k]
                - ozk[R_N + k])
                + ozk[R_Q + k]
        })
        .collect()
}

pub struct Proof {
    pub c: pcs::Commitment,

    pub s1: FqExt,
    pub q_claim: FqExt,
    pub claim_bin: FqExt,
    pub u_final: FqExt,
    pub open_bin: FqExt,
    pub open_full: FqExt,
    pub open_h_sc1: FqExt,
    pub open_h_sum: FqExt,

    pub mask_r_evals: [FqExt; 3],

    pub sc1: SumcheckProof,
    pub sc_full: SumcheckProof,
    pub sc_bin: SumcheckProof,
    pub sc5: SumcheckProof,

    pub mask_totals: [FqExt; 4],
    pub mask_evals: [FqExt; 4],
}

impl Proof {
    pub fn num_rounds(&self) -> usize {
        self.sc1.rounds.len()
            + self.sc_full.rounds.len()
            + self.sc_bin.rounds.len()
            + self.sc5.rounds.len()
    }
}

pub fn proof_fingerprint(p: &Proof) -> u64 {
    #[inline]
    fn mix(d: &mut u64, x: u64) {
        *d ^= x;
        *d = d.wrapping_mul(0x100000001b3);
    }
    #[inline]
    fn mix4(d: &mut u64, v: FqExt) {
        for c in v.0 {
            mix(d, c.0);
        }
    }
    let d = &mut 0xcbf29ce484222325u64;
    mix(d, p.c.digest);
    mix(d, p.c.num_vars as u64);
    for v in pub_scalars(p) {
        mix4(d, v);
    }
    for sc in [&p.sc1, &p.sc_full, &p.sc_bin, &p.sc5] {
        mix(d, sc.rounds.len() as u64);
        for r in &sc.rounds {
            mix(d, r.len() as u64);
            for &v in r {
                mix4(d, v);
            }
        }
    }
    *d
}

pub fn pub_scalars(p: &Proof) -> Vec<FqExt> {
    let mut v = vec![
        p.s1,
        p.q_claim,
        p.claim_bin,
        p.u_final,
        p.open_bin,
        p.open_full,
        p.open_h_sc1,
        p.open_h_sum,
    ];
    v.extend_from_slice(&p.mask_r_evals);
    v.extend_from_slice(&p.mask_totals);
    v.extend_from_slice(&p.mask_evals);
    v
}

pub const N_PUB_SCALARS: usize = 8 + 3 + 4 + 4;

fn bit_poly(z: FqExt) -> FqExt {
    z * (z - FqExt::ONE)
}

fn challenge_vec(tr: &mut Transcript, n: usize) -> Vec<FqExt> {
    (0..n).map(|_| tr.challenge_fq4()).collect()
}

fn transcript_init(params: &HashParams, stmt: &[RingElem]) -> Transcript {
    let mut tr = Transcript::new("blmr-nizk1-v1");
    tr.absorb_u64(params.n_bits as u64);
    tr.absorb_u64(params.group_bits as u64);
    tr.absorb_u64(params.ell as u64);
    tr.absorb_digest(&params.crs_digest);
    for c in stmt {
        tr.absorb_fqs(&c.c);
    }
    tr
}

fn contract_a_hat(rows: &Rows, r_v: &[FqExt]) -> Vec<Vec<Vec<FqExt>>> {
    let eqv = eq_table(r_v);
    rows.a_base
        .iter()
        .map(|blk| {
            let (m, k) = (blk[0].len(), blk[0][0].len());
            let mut cb = vec![vec![FqExt::ZERO; k]; m];
            for (v, &e) in eqv.iter().take(blk.len()).enumerate() {
                let av = &blk[v];
                for j in 0..m {
                    for t in 0..k {
                        cb[j][t] = cb[j][t] + e * av[j][t];
                    }
                }
            }
            cb
        })
        .collect()
}

fn c_hat_row<'a>(c_hat: &'a [Vec<Vec<FqExt>>], row: &LinRow) -> &'a [FqExt] {
    debug_assert!(row.step_i >= 1, "the commitment row has no u side");
    &c_hat[row.step_i - 1][row.chain]
}

fn ppub_sum(rows: &Rows, point: &[FqExt]) -> FqExt {
    rows.lin.iter().fold(FqExt::ZERO, |a, r| a + eq_at_index(point, r.cell) * r.p_pub)
}

fn pu_sum(rows: &Rows, r_c: &[FqExt], r_v: &[FqExt]) -> FqExt {
    let eqv = eq_table(r_v);
    let mut acc = FqExt::ZERO;
    for row in rows.lin.iter().filter(|r| r.kind == CKind::Init) {
        let inner = eqv
            .iter()
            .take(rows.a_base[0].len())
            .enumerate()
            .fold(FqExt::ZERO, |a, (v, &e)| a + e * rows.u_pub(row, v));
        acc = acc + eq_at_index(r_c, row.cell) * inner;
    }
    acc
}

fn lg_bin_eval(rows: &Rows, tau: &[FqExt], r_k: &[FqExt]) -> FqExt {
    let mut acc = FqExt::ZERO;
    for row in &rows.lin {
        if row.bin_entries.is_empty() {
            continue;
        }
        let w = eq_at_index(tau, row.cell);
        for &(k, coef) in &row.bin_entries {
            acc = acc + w * coef * eq_at_index(r_k, k);
        }
    }
    acc
}

fn lg_full_eval(
    params: &HashParams,
    rows: &Rows,
    tau: &[FqExt],
    r_c: &[FqExt],
    c_hat: &[Vec<Vec<FqExt>>],
    gamma: FqExt,
    r_k: &[FqExt],
) -> FqExt {
    let mut acc = FqExt::ZERO;
    for row in &rows.lin {
        let w_m = eq_at_index(tau, row.cell);
        for &(k, coef) in &row.full_entries {
            acc = acc + w_m * coef * eq_at_index(r_k, k);
        }
        if let Some(src) = u_src(params, row.kind) {
            let w_n = gamma * eq_at_index(r_c, row.cell);
            let cr = c_hat_row(c_hat, row);
            for t in 0..u_width(params) {
                acc = acc + w_n * cr[t] * eq_at_index(r_k, rows.b_base + b_row(params, src, t));
            }
        }
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn merged_weight_eval(
    d: &Dims,
    params: &HashParams,
    rows: &Rows,
    metas: &[ConstraintMeta],
    tau: &[FqExt],
    r_c: &[FqExt],
    c_hat: &[Vec<Vec<FqExt>>],
    gamma: FqExt,
    alpha: FqExt,
    lambda_f: FqExt,
    r: &[FqExt],
) -> FqExt {
    let nv_row = d.nv - SLOT_VARS;
    let (r_row, r_slot) = r.split_at(nv_row);
    let w_b = lambda_f * lg_full_eval(params, rows, tau, r_c, c_hat, gamma, r_row);
    let w_q = metas.iter().filter_map(|m| m.t_index.map(|t| (m.cell, t))).fold(
        FqExt::ZERO,
        |acc, (cell, t)| {
            acc + eq_at_index(tau, cell) * eq_at_index(r_row, d.merged.quot_row_of(t))
        },
    );
    (w_b + w_q) * alpha_tensor_eval(r_slot, alpha, ALPHA_LIMIT)
}

#[derive(Default)]
pub struct Timings(pub Vec<(&'static str, std::time::Duration)>);

impl Timings {
    #[inline]
    fn mark(&mut self, name: &'static str, t: &mut std::time::Instant) {
        let now = std::time::Instant::now();
        self.0.push((name, now - *t));
        *t = now;
    }
    pub fn total(&self) -> std::time::Duration {
        self.0.iter().map(|(_, d)| *d).sum()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum Sabotage {
    #[default]
    None,
    ExtraOneHot { step: usize, v: usize },
    ZeroOneHotRow { step: usize },
    FlipBinBit { row: usize, coef: usize },
    PerturbFull { row: usize, coef: usize },
    WrongQuotient { idx: usize },
    WrongCxBlinding,
}

pub fn prove(params: &HashParams, groups: &[usize]) -> (Vec<RingElem>, Proof) {
    let (bx, _, p, _) = prove_impl(params, groups, Sabotage::None, None);
    (bx, p)
}

pub fn prove_with_timings(
    params: &HashParams,
    groups: &[usize],
) -> (Vec<RingElem>, Proof, Timings) {
    let (bx, _, p, tm) = prove_impl(params, groups, Sabotage::None, None);
    (bx, p, tm)
}

pub fn prove_nizk1(
    params: &HashParams,
    nz: &Nizk1Params,
    groups: &[usize],
    bw: &BlindWitness,
) -> (BlindStatement, Proof) {
    let (_, st, p, _) = prove_impl(params, groups, Sabotage::None, Some((nz, bw)));
    (st.expect("Phase B"), p)
}

pub fn prove_nizk1_with_timings(
    params: &HashParams,
    nz: &Nizk1Params,
    groups: &[usize],
    bw: &BlindWitness,
) -> (BlindStatement, Proof, Timings) {
    let (_, st, p, tm) = prove_impl(params, groups, Sabotage::None, Some((nz, bw)));
    (st.expect("Phase B"), p, tm)
}

pub(crate) fn prove_impl(
    params: &HashParams,
    groups: &[usize],
    sab: Sabotage,
    nzin: Option<(&Nizk1Params, &BlindWitness)>,
) -> (Vec<RingElem>, Option<BlindStatement>, Proof, Timings) {
    let mut tm = Timings::default();
    let mut clk = std::time::Instant::now();

    let (bx, wit) = eval_h(params, groups);
    debug_assert!(crate::hash::check_witness(params, &bx, groups, &wit));
    tm.mark("eval_h", &mut clk);

    let mut blind = nzin.map(|(nz, bw)| (nz, bw, blind_statement(params, nz, &bx, bw)));
    if let (Sabotage::WrongCxBlinding, Some((_, _, (st, _)))) = (sab, blind.as_mut()) {
        st.c_x[0].c[0] = st.c_x[0].c[0] + Fq::ONE;
    }
    tm.mark("request", &mut clk);
    let ctx_store = blind.as_ref().map(|(nz, _, (st, _))| Nizk1Ctx::new(params, nz, st));
    let ctx = ctx_store.as_ref();
    let bq = blind.as_ref().map(|(_, _, (_, q))| q);
    let d = dims(params, ctx);
    let stmt: Vec<RingElem> = match &blind {
        Some((_, _, (st, _))) => st.c_x.clone(),
        None => bx.clone(),
    };
    let metas = constraints(params, ctx);
    let mut quotients = compute_quotients(params, &wit, ctx, bq);
    if let Sabotage::WrongQuotient { idx } = sab {
        quotients[idx][0] = quotients[idx][0] + Fq::ONE;
    }
    tm.mark("quot", &mut clk);

    let zk_seed: [u8; 32] = match nzin {
        Some((_, bw)) => bw.zk_seed,
        None => {
            let mut h = crate::keccak::Shake128::new();
            h.absorb_bytes(b"blmr-zk-mask-phaseA-NOT-HIDING");
            h.absorb_bytes(&params.crs_digest);
            for &g in groups {
                h.absorb_u64(g as u64);
            }
            let mut sd = [0u8; 32];
            h.squeeze(&mut sd);
            sd
        }
    };
    let mut mrng = crate::rng::CsRng::from_parts("blmr-zk-mask-v1", &[&zk_seed]);
    let shapes = mask_shapes(&d);
    let bases = mask_bases(&d);
    let mut maskers: Vec<sumcheck::Masker> =
        shapes.iter().map(|sh| sumcheck::Masker::with_degs(&mut mrng, sh)).collect();
    let ozk: [FqExt; OPENZK_VALS] = std::array::from_fn(|_| mrng.next_fq4());
    tm.mark("zk mask", &mut clk);

    let mut hb = crate::relation::h_bits(params, groups);
    match sab {
        Sabotage::ZeroOneHotRow { step } => {
            hb[crate::relation::h_index(params, step, groups[step])] = false
        }
        Sabotage::ExtraOneHot { step, v } => hb[crate::relation::h_index(params, step, v)] = true,
        _ => {}
    }
    if let (Some((nz, bw)), Sabotage::None) = (nzin, sab) {
        assert_eq!(crate::nizk1::pack_h(nz, &hb), bw.h_pack, "the H segment of c_bin is out of sync with the message of d_x");
    }

    assert_mask_fits(&d);
    let mut zb = build_bin_table(&d, &hb, blind.as_ref().map(|(_, bw, _)| *bw), ctx);
    if let Sabotage::FlipBinBit { row, coef } = sab {
        zb.flip(row * SLOTS + coef);
    }
    tm.mark("bin table", &mut clk);
    let mut zf = build_merged_table(params, &d, &wit, &quotients, &metas, &zb);
    write_mask_coefs(&mut zf, &maskers, &bases);
    write_openzk_vals(&d, &mut zf, &ozk);
    match sab {
        Sabotage::PerturbFull { row, coef } => {
            assert!(!d.merged.is_bin_row(row), "PerturbFull must not point into the binary block");
            zf[row * SLOTS + coef] = zf[row * SLOTS + coef] + Fq::ONE
        }
        _ => {}
    }

    tm.mark("merged table", &mut clk);
    let c = pcs::commit_fq(&zf);
    tm.mark("commit", &mut clk);

    let mut tr = transcript_init(params, &stmt);
    if let Some(c) = ctx {
        tr.absorb_digest(&c.nz.crs_digest);
        tr.absorb_u64(c.st.j);
        for e in c.st.c_r.iter().chain(&c.st.d_x) {
            tr.absorb_fqs(&e.c);
        }
    }
    tr.absorb_u64(c.digest);

    let rho = tr.challenge_fq4();
    for mk in maskers.iter_mut() {
        mk.set_rho(rho);
    }

    let alpha = tr.challenge_fq4();
    let rows = build_rows(params, &stmt, alpha, ctx);
    tm.mark("build_rows", &mut clk);

    let mut apow = Vec::with_capacity(SLOTS);
    let mut p = FqExt::ONE;
    for _ in 0..SLOTS {
        apow.push(p);
        p = p * alpha;
    }
    let xn1 = alpha.pow(N as u128) + FqExt::ONE;
    let tau = challenge_vec(&mut tr, d.nv_c);

    let w_bin: Vec<FqExt> = (0..d.bin_pad)
        .into_par_iter()
        .map(|k| {
            let base = k * SLOTS;
            let mut acc = LazyExtSum::new();
            for wi in 0..SLOTS / 64 {
                let mut word = zb.word(base / 64 + wi);
                while word != 0 {
                    let b = word.trailing_zeros() as usize;
                    acc.add(&apow[wi * 64 + b]);
                    word &= word - 1;
                }
            }
            acc.finish()
        })
        .collect();
    let w_full: Vec<FqExt> = (0..d.merged.rows)
        .into_par_iter()
        .map(|k| crate::ring::poly_eval_pows(&zf[k * SLOTS..(k + 1) * SLOTS], &apow))
        .collect();
    tm.mark("w_hat", &mut clk);

    let eq_tau = eq_table(&tau);
    let mut eq_ext = vec![FqExt::ZERO; d.u_cells];
    let mut h_ext = vec![FqExt::ZERO; d.u_cells];
    for cell in 0..d.c_cells {
        for v in 0..d.hv {
            eq_ext[cell * d.hv + v] = eq_tau[cell];
            h_ext[cell * d.hv + v] =
                if h_bit(&zb, &d, h_cell(&d, cell, v)) { FqExt::ONE } else { FqExt::ZERO };
        }
    }
    let mut u_table = vec![FqExt::ZERO; d.u_cells];
    for row in &rows.lin {
        let base = row.cell * d.hv;
        match u_src(params, row.kind) {
            Some(src) => {
                let ws: Vec<FqExt> = (0..u_width(params))
                    .map(|t| w_full[d.merged.b_row_of(params, src, t)])
                    .collect();
                for v in 0..d.tsz {
                    let ab = rows.a_row(row, v);
                    u_table[base + v] =
                        (0..ws.len()).fold(FqExt::ZERO, |a, t| a + ab[t] * ws[t]);
                }
            }
            None => {
                for v in 0..d.tsz {
                    u_table[base + v] = rows.u_pub(row, v);
                }
            }
        }
    }
    let s1_raw = eq_ext
        .iter()
        .zip(&h_ext)
        .zip(&u_table)
        .fold(FqExt::ZERO, |a, ((&e, &h), &u)| a + e * h * u);
    tm.mark("SC1 tables", &mut clk);

    let r_b_mask = ClaimMask { coef: &ozk[R_B..R_B + R_COEFS] };
    let sigma_u = ClaimMask { coef: &ozk[R_U..R_U + R_COEFS] }.total();
    let s1 = s1_raw + r_b_mask.total();
    let mt0 = maskers[0].total_plain();
    tr.absorb_fq4(s1);
    tr.absorb_fq4(mt0);
    let (sc1, r_u_full, open_h_sc1, u_final, open_r_b) = sumcheck::prove_bilinear_zk(
        eq_ext,
        h_ext,
        u_table,
        LinMask([ozk[SIG_H], ozk[SIG_H + 1]]),
        sigma_u,
        r_b_mask,
        Some(&mut maskers[0]),
        &mut tr,
    );
    let (r_u, c1) = r_u_full.split_at(d.nv_u);
    let c1 = c1[0];
    let me0 = maskers[0].eval();
    tr.absorb_fq4(open_h_sc1);
    tr.absorb_fq4(u_final);
    tr.absorb_fq4(me0);
    tm.mark("SC1", &mut clk);

    let gamma = tr.challenge_fq4();
    let (r_c_part, r_v) = r_u.split_at(d.nv_c);
    let c_hat = contract_a_hat(&rows, r_v);
    let mut lg_bin = vec![FqExt::ZERO; d.bin_pad];
    let mut lg_full = vec![FqExt::ZERO; d.merged.rows];
    for row in &rows.lin {
        let w_m = eq_at_index(&tau, row.cell);
        for &(k, coef) in &row.bin_entries {
            lg_bin[k] = lg_bin[k] + w_m * coef;
        }
        for &(k, coef) in &row.full_entries {
            lg_full[k] = lg_full[k] + w_m * coef;
        }
        if let Some(src) = u_src(params, row.kind) {
            let w_n = gamma * eq_at_index(r_c_part, row.cell);
            let cr = c_hat_row(&c_hat, row);
            for t in 0..u_width(params) {
                let k = d.merged.b_row_of(params, src, t);
                lg_full[k] = lg_full[k] + w_n * cr[t];
            }
        }
    }
    let claim_bin_raw = (0..d.bin_pad).fold(FqExt::ZERO, |a, k| a + lg_bin[k] * w_bin[k]);
    let q_raw = metas.iter().filter_map(|m| m.t_index.map(|t| (m.cell, t))).fold(
        FqExt::ZERO,
        |a, (cell, t)| {
            let base = d.merged.quot_row_of(t) * SLOTS;
            a + eq_tau[cell] * crate::ring::poly_eval_pows(&zf[base..base + SLOTS], &apow)
        },
    );
    tm.mark("lg build", &mut clk);

    let q_claim = q_raw + ClaimMask { coef: &ozk[R_Q..R_Q + R_COEFS] }.total();
    let claim_bin = claim_bin_raw + ClaimMask { coef: &ozk[R_N..R_N + R_COEFS] }.total();
    let mt1 = maskers[1].total_plain();
    tr.absorb_fq4(q_claim);
    tr.absorb_fq4(claim_bin);
    tr.absorb_fq4(mt1);
    let lambda_f = tr.challenge_fq4();
    let z_u = z_eval(r_u);

    let mut w_row: Vec<FqExt> = lg_full.iter().map(|&v| lambda_f * v).collect();
    for m in &metas {
        if let Some(t) = m.t_index {
            let r = d.merged.quot_row_of(t);
            w_row[r] = w_row[r] + eq_tau[m.cell];
        }
    }
    let mut wfull = vec![FqExt::ZERO; d.merged.rows * SLOTS];
    for (row, &w) in w_row.iter().enumerate() {
        if w != FqExt::ZERO {
            let base = row * SLOTS;
            for (s, &ap) in apow.iter().enumerate() {
                wfull[base + s] = w * ap;
            }
        }
    }
    let nf_coefs = n_full_coefs(&ozk, lambda_f, gamma, z_u, xn1);
    let (sc_full, r_f_full, open_full, open_r_full) = sumcheck::prove_product2(
        &zf,
        wfull,
        LinMask([ozk[SIG_FULL], ozk[SIG_FULL + 1]]),
        ClaimMask { coef: &nf_coefs },
        Some(&mut maskers[1]),
        &mut tr,
    );
    let (r_f, cf) = r_f_full.split_at(d.nv);
    let cf = cf[0];
    let me1 = maskers[1].eval();
    tr.absorb_fq4(open_full);
    tr.absorb_fq4(me1);
    tm.mark("SC_full", &mut clk);

    let tau0 = challenge_vec(&mut tr, d.nv_bin);
    let lambda_b = tr.challenge_fq4();
    let mt2 = maskers[2].total_plain();
    tr.absorb_fq4(mt2);
    let nb_coefs: Vec<FqExt> = (0..R_COEFS).map(|k| lambda_b * ozk[R_N + k]).collect();
    let mut scratch: Vec<FqExt> = Vec::new();
    let (sc_bin, r_b_full, open_bin, open_r_bin) = sumcheck::prove_batched_w(
        &zb,
        lg_bin,
        &apow,
        &tau0,
        lambda_b,
        LinMask([ozk[SIG_BIN], ozk[SIG_BIN + 1]]),
        ClaimMask { coef: &nb_coefs },
        &mut scratch,
        Some(&mut maskers[2]),
        &mut tr,
    );
    let (r_bin, cb) = r_b_full.split_at(d.nv_bin);
    let cb = cb[0];
    let me2 = maskers[2].eval();
    tr.absorb_fq4(open_bin);
    tr.absorb_fq4(me2);
    tm.mark("SC_bin", &mut clk);

    let tau3 = challenge_vec(&mut tr, d.nv_i);
    let eq_i = eq_table(&tau3);
    let p_tbl: Vec<FqExt> = (0..d.g_pad)
        .map(|i| {
            let cnt = (0..d.tsz).filter(|&v| h_bit(&zb, &d, i * d.hv + v)).count();
            FqExt::from_u64(cnt as u64)
        })
        .collect();
    let mt3 = maskers[3].total_plain();
    tr.absorb_fq4(mt3);
    let (sc5, r5, _) =
        sumcheck::prove(vec![eq_i, p_tbl], 2, &|v| v[0] * v[1], Some(&mut maskers[3]), &mut tr);
    let open_h_sum = pcs::open(&zb, &hpack_point(&d, &half_point(&d, &r5)));
    tm.mark("SC5", &mut clk);

    let mask_totals = [mt0, mt1, mt2, mt3];
    let mask_evals = [me0, me1, me2, maskers[3].eval()];

    debug_assert!(
        {
            let pts: [&[FqExt]; 4] = [&r_u_full, &r_f_full, &r_b_full, &r5];
            (0..4).all(|i| {
                let wt = sumcheck::Masker::weights_total_plain_degs(&shapes[i]);
                let we = sumcheck::Masker::weights_eval_degs(&shapes[i], pts[i]);
                pcs::open_linear_fq(&zf, &mask_weights(bases[i], &wt)) == mask_totals[i]
                    && pcs::open_linear_fq(&zf, &mask_weights(bases[i], &we))
                        == mask_evals[i]
            })
        },
        "mask_weights disagrees with the mask coefficients written into the mask row"
    );

    debug_assert!(
        {
            let hpt = hpack_point(&d, &r_u[d.nv_u - d.nv_h..]);
            let h5 = hpack_point(&d, &half_point(&d, &r5));
            [r_bin, &hpt[..], &h5[..]]
                .iter()
                .all(|pt| pcs::open_fq(&zf, &bin_prefix(&d, pt)) == pcs::open(&zb, pt))
        },
        "bin_prefix is misaligned: the opening of the shadow zb disagrees with the opening of the merged table at the prefix point"
    );

    debug_assert!(
        {
            let hpt = hpack_point(&d, &r_u[d.nv_u - d.nv_h..]);
            let lin = |i: usize, w: &[FqExt]| {
                pcs::open_linear_fq(&zf, &mask_weights(openzk_base(&d, i), w))
            };
            let wt = |pt: &[FqExt]| LinMask::weights(*pt.last().unwrap());
            pcs::open(&zb, r_bin) + z_eval(r_bin) * lin(SIG_BIN, &wt(r_bin)) == open_bin
                && pcs::open_fq(&zf, r_f) + z_eval(r_f) * lin(SIG_FULL, &wt(r_f)) == open_full
                && pcs::open(&zb, &hpt) + z_u * lin(SIG_H, &wt(r_u)) == open_h_sc1
                && lin(R_B, &ClaimMask::weights_eval(R_COEFS - 1, c1)) == open_r_b
                && pcs::open_linear_fq(
                    &zf,
                    &mask_weights(openzk_base(&d, R_B), &n_full_weights(lambda_f, gamma, z_u, xn1, cf)),
                ) == open_r_full
                && lin(R_N, &n_bin_weights(lambda_b, cb)) == open_r_bin
        },
        "the masked value of open-ZK disagrees with the homomorphic combination of the commitment"
    );

    let proof = Proof {
        c,
        s1,
        q_claim,
        claim_bin,
        u_final,
        open_bin,
        open_full,
        open_h_sc1,
        open_h_sum,
        mask_r_evals: [open_r_b, open_r_full, open_r_bin],
        sc1,
        sc_full,
        sc_bin,
        sc5,
        mask_totals,
        mask_evals,
    };
    let st = blind.map(|(_, _, (st, _))| st);
    (bx, st, proof, tm)
}

pub fn verify(params: &HashParams, bx: &[RingElem], proof: &Proof) -> bool {
    let mut clk = std::time::Instant::now();
    verify_impl(params, bx, None, proof, &mut Timings::default(), &mut clk)
}

pub fn verify_nizk1(
    params: &HashParams,
    nz: &Nizk1Params,
    st: &BlindStatement,
    proof: &Proof,
) -> bool {
    verify_nizk1_with_timings(params, nz, st, proof).0
}

pub fn verify_nizk1_with_timings(
    params: &HashParams,
    nz: &Nizk1Params,
    st: &BlindStatement,
    proof: &Proof,
) -> (bool, Timings) {
    let mut tm = Timings::default();
    let mut clk = std::time::Instant::now();
    if st.c_r.len() != nz.com_r.out_len() || st.d_x.len() != nz.com_x.out_len() {
        return (false, tm);
    }
    if st.c_x.len() != params.ell
        || st
            .c_x
            .iter()
            .chain(&st.c_r)
            .chain(&st.d_x)
            .any(|e| e.c.len() != N || e.c.iter().any(|c| c.0 >= crate::field::Q))
    {
        return (false, tm);
    }
    let ctx = Nizk1Ctx::new(params, nz, st);
    tm.mark("check+ctx", &mut clk);
    let ok = verify_impl(params, &st.c_x.clone(), Some(&ctx), proof, &mut tm, &mut clk);
    (ok, tm)
}

fn verify_impl(
    params: &HashParams,
    stmt: &[RingElem],
    ctx: Option<&Nizk1Ctx>,
    proof: &Proof,
    tm: &mut Timings,
    clk: &mut std::time::Instant,
) -> bool {
    if stmt.len() != params.ell
        || stmt.iter().any(|e| e.c.len() != N || e.c.iter().any(|c| c.0 >= crate::field::Q))
    {
        return false;
    }
    let ng = params.num_groups();
    let d = dims(params, ctx);

    if proof.c.num_vars != d.nv {
        return false;
    }
    assert_mask_fits(&d);
    let metas = constraints(params, ctx);

    let mut tr = transcript_init(params, stmt);
    if let Some(c) = ctx {
        tr.absorb_digest(&c.nz.crs_digest);
        tr.absorb_u64(c.st.j);
        for e in c.st.c_r.iter().chain(&c.st.d_x) {
            tr.absorb_fqs(&e.c);
        }
    }
    tr.absorb_u64(proof.c.digest);
    let rho = tr.challenge_fq4();
    let bases = mask_bases(&d);
    let shapes = mask_shapes(&d);
    let alpha = tr.challenge_fq4();
    let rows = build_rows(params, stmt, alpha, ctx);
    let xn1 = alpha.pow(N as u128) + FqExt::ONE;
    let tau = challenge_vec(&mut tr, d.nv_c);
    tm.mark("build_rows", clk);

    tr.absorb_fq4(proof.s1);
    tr.absorb_fq4(proof.mask_totals[0]);
    let claim1 = proof.s1 + rho * proof.mask_totals[0];
    let degs1 = &shapes[0];
    let Some((e1, r_u_full)) = sumcheck::verify_degs(claim1, degs1, &proof.sc1, &mut tr) else {
        return false;
    };
    let (r_u, c1) = r_u_full.split_at(d.nv_u);
    let c1 = c1[0];
    tr.absorb_fq4(proof.open_h_sc1);
    tr.absorb_fq4(proof.u_final);
    tr.absorb_fq4(proof.mask_evals[0]);
    if e1 - rho * proof.mask_evals[0]
        != (FqExt::ONE - c1) * eq_eval(&tau, &r_u[..d.nv_c]) * proof.open_h_sc1 * proof.u_final
            + ind_eval(r_u) * proof.mask_r_evals[0]
    {
        return false;
    }
    tm.mark("SC1", clk);

    let gamma = tr.challenge_fq4();
    let (r_c_part, r_v) = r_u.split_at(d.nv_c);
    let c_hat = contract_a_hat(&rows, r_v);
    let ppub = ppub_sum(&rows, &tau);
    let pu = pu_sum(&rows, r_c_part, r_v);
    let z_u = z_eval(r_u);

    tr.absorb_fq4(proof.q_claim);
    tr.absorb_fq4(proof.claim_bin);
    tr.absorb_fq4(proof.mask_totals[1]);
    let lambda_f = tr.challenge_fq4();

    let claim2_m =
        (proof.s1 - ppub) + gamma * (proof.u_final - pu) + xn1 * proof.q_claim;
    let claim_full_derived = claim2_m - proof.claim_bin;

    let claim_f = lambda_f * claim_full_derived + proof.q_claim + rho * proof.mask_totals[1];
    let degs_f = &shapes[1];
    let Some((e_f, r_f_full)) =
        sumcheck::verify_product2(claim_f, degs_f, &proof.sc_full, &mut tr)
    else {
        return false;
    };
    let (r_f, cf) = r_f_full.split_at(d.nv);
    let cf = cf[0];
    tr.absorb_fq4(proof.open_full);
    tr.absorb_fq4(proof.mask_evals[1]);
    let b_at = merged_weight_eval(
        &d, params, &rows, &metas, &tau, r_c_part, &c_hat, gamma, alpha, lambda_f, r_f,
    );
    if e_f - rho * proof.mask_evals[1]
        != (FqExt::ONE - cf) * b_at * proof.open_full + ind_eval(r_f) * proof.mask_r_evals[1]
    {
        return false;
    }
    tm.mark("SC_full", clk);

    let tau0 = challenge_vec(&mut tr, d.nv_bin);
    let lambda_b = tr.challenge_fq4();
    tr.absorb_fq4(proof.mask_totals[2]);
    let claim_b = lambda_b * proof.claim_bin + rho * proof.mask_totals[2];
    let degs_b = &shapes[2];
    let Some((e_b, r_b_full)) =
        sumcheck::verify_batched_w(claim_b, degs_b, &proof.sc_bin, &mut tr)
    else {
        return false;
    };
    let (r_bin, cb) = r_b_full.split_at(d.nv_bin);
    let cb = cb[0];
    tr.absorb_fq4(proof.open_bin);
    tr.absorb_fq4(proof.mask_evals[2]);
    let nv_k = d.bin_pad.trailing_zeros() as usize;
    let lg_bin_at = lg_bin_eval(&rows, &tau, &r_bin[..nv_k]);
    let f2 = lambda_b
        * lg_bin_at
        * alpha_tensor_eval(&r_bin[nv_k..], alpha, ALPHA_LIMIT)
        * proof.open_bin;
    let f3 = eq_eval(&tau0, r_bin) * bit_poly(proof.open_bin);
    if e_b - rho * proof.mask_evals[2]
        != (FqExt::ONE - cb) * (f2 + f3) + ind_eval(r_bin) * proof.mask_r_evals[2]
    {
        return false;
    }
    tm.mark("SC_bin", clk);

    let tau3 = challenge_vec(&mut tr, d.nv_i);
    let claim5 = (0..ng).fold(FqExt::ZERO, |a, i| a + eq_at_index(&tau3, i));
    tr.absorb_fq4(proof.mask_totals[3]);
    let Some((e5, r5)) = sumcheck::verify(
        claim5 + rho * proof.mask_totals[3],
        d.nv_i,
        2,
        &proof.sc5,
        &mut tr,
    ) else {
        return false;
    };
    let two_w = FqExt::from_u64(2).pow(d.tsz.trailing_zeros() as u128);
    if e5 - rho * proof.mask_evals[3] != eq_eval(&tau3, &r5) * two_w * proof.open_h_sum {
        return false;
    }
    tm.mark("SC5", clk);

    let mask_points: [&[FqExt]; 4] = [&r_u_full, &r_f_full, &r_b_full, &r5];
    for i in 0..4 {
        let wt = sumcheck::Masker::weights_total_plain_degs(&shapes[i]);
        if !pcs::verify_linear(&proof.c, &mask_weights(bases[i], &wt), proof.mask_totals[i]) {
            return false;
        }
        let we = sumcheck::Masker::weights_eval_degs(&shapes[i], mask_points[i]);
        if !pcs::verify_linear(&proof.c, &mask_weights(bases[i], &we), proof.mask_evals[i]) {
            return false;
        }
    }

    let wt = |pt: &[FqExt]| LinMask::weights(*pt.last().unwrap());
    let sig_bin = mask_weights(openzk_base(&d, SIG_BIN), &wt(r_bin));
    let sig_full = mask_weights(openzk_base(&d, SIG_FULL), &wt(r_f));
    let sig_h = mask_weights(openzk_base(&d, SIG_H), &wt(r_u));
    let p_bin = bin_prefix(&d, r_bin);
    let p_h = bin_prefix(&d, &hpack_point(&d, &r_u[d.nv_u - d.nv_h..]));
    let p_h5 = bin_prefix(&d, &hpack_point(&d, &half_point(&d, &r5)));
    let one = FqExt::ONE;
    let c = &proof.c;
    let ok = pcs::verify_combined(
        &[
            (one, pcs::Term::FqPoint { c, point: &p_bin }),
            (z_eval(r_bin), pcs::Term::Linear { c, weights: &sig_bin }),
        ],
        proof.open_bin,
    ) && pcs::verify_combined(
        &[
            (one, pcs::Term::FqPoint { c, point: r_f }),
            (z_eval(r_f), pcs::Term::Linear { c, weights: &sig_full }),
        ],
        proof.open_full,
    ) && pcs::verify_combined(
        &[
            (one, pcs::Term::FqPoint { c, point: &p_h }),
            (z_u, pcs::Term::Linear { c, weights: &sig_h }),
        ],
        proof.open_h_sc1,
    )
    && pcs::verify(c, &p_h5, proof.open_h_sum)
    && pcs::verify_linear(
        c,
        &mask_weights(openzk_base(&d, R_B), &ClaimMask::weights_eval(R_COEFS - 1, c1)),
        proof.mask_r_evals[0],
    )
    && pcs::verify_linear(
        c,
        &mask_weights(openzk_base(&d, R_B), &n_full_weights(lambda_f, gamma, z_u, xn1, cf)),
        proof.mask_r_evals[1],
    )
    && pcs::verify_linear(
        c,
        &mask_weights(openzk_base(&d, R_N), &n_bin_weights(lambda_b, cb)),
        proof.mask_r_evals[2],
    );
    tm.mark("PCS stub", clk);
    ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext_field::EXT_DEG;
    use crate::hash::bits_to_groups;
    use crate::mle::eq_table;
    use crate::nizk1::{sample_blind, BlindStatement, QueryTicket};
    use crate::relation::num_quotients;
    use crate::rng::insecure_test_secret;
    use crate::transcript::SimpleRng;

    fn setup(n: usize, g: usize, ell: usize, seed: u64) -> (HashParams, Vec<usize>) {
        let params = HashParams::sample(seed, n, g, ell);
        let mut rng = SimpleRng::new(seed ^ 0x5EED);
        let bits: Vec<bool> = (0..n).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        (params, groups)
    }

    fn setup_b(
        n: usize,
        g: usize,
        ell: usize,
        seed: u64,
    ) -> (HashParams, Nizk1Params, Vec<usize>, BlindWitness) {
        let (params, groups) = setup(n, g, ell, seed);
        let nz = Nizk1Params::sample(seed + 1, &params, 3, 2, 2);
        let bw = sample_blind(
            QueryTicket::insecure_for_tests(insecure_test_secret(seed + 2), 0),
            &params,
            &nz,
            &groups,
        );
        (params, nz, groups, bw)
    }

    #[test]
    fn prove_verify_roundtrip() {
        for &(n, g, ell, seed) in &[
            (8usize, 2usize, 1usize, 101u64),
            (8, 2, 3, 102),
            (8, 4, 2, 103),
            (12, 2, 5, 104),
            (8, 1, 2, 105),
            (16, 4, 5, 106),
        ] {
            let (params, nz, groups, bw) = setup_b(n, g, ell, seed);
            let (bx, pa) = prove(&params, &groups);
            assert!(verify(&params, &bx, &pa), "Phase A roundtrip failed (n={n} g={g} ell={ell})");
            let (st, pb) = prove_nizk1(&params, &nz, &groups, &bw);
            assert!(
                verify_nizk1(&params, &nz, &st, &pb),
                "Phase B roundtrip failed (n={n} g={g} ell={ell})"
            );
        }
    }

    #[test]
    fn prove_is_deterministic() {
        let (params, nz, groups, bw) = setup_b(8, 2, 2, 201);
        let a = prove(&params, &groups).1;
        let b = prove(&params, &groups).1;
        assert_eq!(proof_fingerprint(&a), proof_fingerprint(&b));
        let x = prove_nizk1(&params, &nz, &groups, &bw).1;
        let y = prove_nizk1(&params, &nz, &groups, &bw).1;
        assert_eq!(proof_fingerprint(&x), proof_fingerprint(&y));
    }

    #[test]
    fn proofs_do_not_transfer() {
        let (p1, g1) = setup(8, 2, 2, 501);
        let (p2, _) = setup(8, 2, 2, 999);
        assert_ne!(p1.crs_digest, p2.crs_digest);
        let (bx, pf) = prove(&p1, &g1);
        assert!(verify(&p1, &bx, &pf));
        assert!(!verify(&p2, &bx, &pf), "verification unexpectedly still succeeded after swapping the CRS");

        let mut bx2 = bx.clone();
        bx2[0].c[0] = bx2[0].c[0] + Fq::ONE;
        assert!(!verify(&p1, &bx2, &pf), "verification unexpectedly still succeeded after swapping the statement");
    }

    #[test]
    fn commitments_are_bound_before_the_first_challenge() {
        let (params, groups) = setup(8, 2, 1, 601);
        let (bx, mut p) = prove(&params, &groups);
        p.c.digest ^= 1;
        assert!(!verify(&params, &bx, &p), "before the commitment is bound into the first challenge");
    }

    #[test]
    fn mismatched_commitment_arity_is_rejected() {
        let (params, groups) = setup(8, 2, 1, 701);
        for delta in [1usize, 2] {
            let (bx, mut p) = prove(&params, &groups);
            p.c.num_vars += delta;
            assert!(!verify(&params, &bx, &p), "commitment arity +{delta} unexpectedly passed");
        }
        let (bx, mut p) = prove(&params, &groups);
        p.c.num_vars -= 1;
        assert!(!verify(&params, &bx, &p), "commitment arity −1 unexpectedly passed");
    }

    #[test]
    fn masker_degrees_cover_every_round_degree() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (16, 4, 5), (12, 2, 3)] {
            let (params, nz, groups, bw) = setup_b(n, g, ell, 777);
            let (st, p) = prove_nizk1(&params, &nz, &groups, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));
            let shapes = mask_shapes(&d);
            let verifier_degs = [
                sumcheck::round_degs(d.nv_u, 3, 6, W_ROUND_DEG_3),
                sumcheck::round_degs(d.nv, 2, 4, W_ROUND_DEG_2),
                sumcheck::round_degs(d.nv_bin, 3, 7, W_ROUND_DEG_3),
            ];
            let scs = [&p.sc1, &p.sc_full, &p.sc_bin];
            for i in 0..3 {
                assert_eq!(shapes[i].len(), verifier_degs[i].len(), "SC{i} round count");
                for (j, (m, v)) in shapes[i].iter().zip(&verifier_degs[i]).enumerate() {
                    assert!(m >= v, "SC{i} round {j}: masker count {m} < round count {v}");
                }
                let last_cube = shapes[i].len() - 2;
                assert_eq!(shapes[i][last_cube], [6, 4, 7][i], "mask count of the Z round of SC{i}");
                let sent = scs[i].rounds[last_cube].len();
                assert_eq!(sent, if i == 0 { 7 } else { [0, 4, 7][i] }, "number of values sent in the Z round of SC{i}");
            }
            assert!(verify_nizk1(&params, &nz, &st, &p));
        }
    }

    #[test]
    fn round_counts_match_the_cubes() {
        let (params, nz, groups, bw) = setup_b(16, 4, 5, 801);
        let (st, p) = prove_nizk1(&params, &nz, &groups, &bw);
        let ctx = Nizk1Ctx::new(&params, &nz, &st);
        let d = dims(&params, Some(&ctx));
        assert_eq!(p.sc1.rounds.len(), d.nv_u + 1, "SC1");
        assert_eq!(p.sc_full.rounds.len(), d.nv + 1, "SC_full (the whole merged table)");
        assert_eq!(p.sc_bin.rounds.len(), d.nv_bin + 1, "SC_bin (binary subcube)");
        assert_eq!(p.sc5.rounds.len(), d.nv_i, "SC5");
        assert_eq!(p.c.num_vars, d.nv, "arity of the single commitment");
        assert!(p.sc1.rounds[..d.nv_u - 1].iter().all(|r| r.len() == 4));
        assert_eq!(p.sc1.rounds[d.nv_u - 1].len(), 7, "Z round of SC1 = deg 6");
        assert!(p.sc_full.rounds[..d.nv - 1].iter().all(|r| r.len() == 2));
        assert_eq!(p.sc_full.rounds[d.nv - 1].len(), 4, "Z round of SC_full = deg 4");
        assert!(p.sc_bin.rounds[..d.nv_bin - 1].iter().all(|r| r.len() == 3));
        assert_eq!(p.sc_bin.rounds[d.nv_bin - 1].len(), 7, "Z round of SC_bin = deg 7");
        assert!(p.sc5.rounds.iter().all(|r| r.len() == 3));
    }

    #[test]
    fn the_zk_masks_are_wired_in() {
        let (params, groups) = setup(8, 2, 2, 901);
        let nz = Nizk1Params::sample(902, &params, 3, 2, 2);
        let mk = || {
            sample_blind(
                QueryTicket::insecure_for_tests(insecure_test_secret(903), 0),
                &params,
                &nz,
                &groups,
            )
        };
        let bw = mk();
        let mut bw2 = mk();
        assert_eq!(bw.zk_seed, bw2.zk_seed, "the zk_seed should be the same for the same ticket");
        bw2.zk_seed[0] ^= 0xA5;
        let (st1, p1) = prove_nizk1(&params, &nz, &groups, &bw);
        let (st2, p2) = prove_nizk1(&params, &nz, &groups, &bw2);
        assert_eq!(st1.c_x, st2.c_x, "zk_seed must not affect the statement");
        assert!(verify_nizk1(&params, &nz, &st1, &p1));
        assert!(verify_nizk1(&params, &nz, &st2, &p2));
        assert_ne!(proof_fingerprint(&p1), proof_fingerprint(&p2));
        for (i, (a, b)) in pub_scalars(&p1).iter().zip(pub_scalars(&p2)).enumerate() {
            if i == 7 {
                assert_eq!(*a, b, "open_h_sum must not be masked");
                continue;
            }
            assert_ne!(*a, b, "public scalar {i} did not change with zk_seed (masking not wired up?)");
        }
    }

    fn sweep_one(n: usize, g: usize, ell: usize, seed: u64) {
        let (params, nz, groups, bw) = setup_b(n, g, ell, seed);
        let ng = params.num_groups();
        let tsz = params.table_size();
        let tag = format!("n={n} g={g} ell={ell}");

        let (bx, proof) = prove(&params, &groups);
        assert!(verify(&params, &bx, &proof), "Phase A roundtrip failed ({tag})");
        let (st, pb) = prove_nizk1(&params, &nz, &groups, &bw);
        assert!(verify_nizk1(&params, &nz, &st, &pb), "Phase B roundtrip failed ({tag})");

        let da = dims(&params, None);
        let ctx = Nizk1Ctx::new(&params, &nz, &st);
        let db = dims(&params, Some(&ctx));
        assert_eq!(mask_shapes(&da).len(), 4, "masker count ({tag})");
        assert_eq!(pb.mask_totals.len(), 4, "mask_totals count ({tag})");
        assert_eq!(pb.mask_evals.len(), 4, "mask_evals count ({tag})");
        for (d, p) in [(&da, &proof), (&db, &pb)] {
            assert_eq!(p.sc1.rounds.len(), d.nv_u + 1, "SC1 round count ({tag})");
            assert_eq!(p.sc_full.rounds.len(), d.nv + 1, "SC_full round count ({tag})");
            assert_eq!(p.sc_bin.rounds.len(), d.nv_bin + 1, "SC_bin round count ({tag})");
            assert_eq!(p.sc5.rounds.len(), d.nv_i, "SC5 round count ({tag})");
            assert_eq!(p.c.num_vars, d.nv, "commitment arity ({tag})");
            assert_eq!(p.to_bytes().len(), p.size_breakdown().total(), "accounting ({tag})");
        }

        let bad_v = (groups[0] + 1) % tsz;
        let nq_a = num_quotients(&params, None);
        let mut sabs_a = vec![
            Sabotage::ExtraOneHot { step: 0, v: bad_v },
            Sabotage::ExtraOneHot { step: ng - 1, v: (groups[ng - 1] + 1) % tsz },
            Sabotage::ZeroOneHotRow { step: 0 },
            Sabotage::ZeroOneHotRow { step: ng - 1 },
            Sabotage::PerturbFull { row: da.merged.b_base, coef: 0 },
            Sabotage::PerturbFull { row: da.merged.b_row_of(&params, ng - 2, ell - 1), coef: N - 1 },
            Sabotage::PerturbFull { row: da.merged.quot_row_of(0), coef: 0 },
            Sabotage::PerturbFull { row: da.merged.quot_row_of(0), coef: N - 1 },
            Sabotage::WrongQuotient { idx: 0 },
            Sabotage::WrongQuotient { idx: nq_a - 1 },
            Sabotage::FlipBinBit { row: da.h_row, coef: 0 },
        ];
        if ng > 2 {
            sabs_a.push(Sabotage::ZeroOneHotRow { step: 1 });
        }
        for sab in sabs_a {
            let (bx, _, p, _) = prove_impl(&params, &groups, sab, None);
            assert!(!verify(&params, &bx, &p), "Phase A: {sab:?} unexpectedly passed ({tag})");
        }

        let nq_b = num_quotients(&params, Some(&ctx));
        for sab in [
            Sabotage::WrongCxBlinding,
            Sabotage::FlipBinBit { row: ctx.w.rho_r_pos, coef: 0 },
            Sabotage::FlipBinBit { row: ctx.w.h_pack, coef: 0 },
            Sabotage::FlipBinBit { row: ctx.w.r_pos, coef: 0 },
            Sabotage::FlipBinBit { row: ctx.w.rho_x_pos, coef: 0 },
            Sabotage::WrongQuotient { idx: nq_b - 1 },
            Sabotage::PerturbFull { row: db.merged.quot_row_of(0), coef: 0 },
            Sabotage::ExtraOneHot { step: 0, v: bad_v },
            Sabotage::ZeroOneHotRow { step: ng - 1 },
        ] {
            let (_, st2, p, _) = prove_impl(&params, &groups, sab, Some((&nz, &bw)));
            assert!(
                !verify_nizk1(&params, &nz, &st2.expect("Phase B"), &p),
                "Phase B: {sab:?} unexpectedly passed ({tag})"
            );
        }

        assert_eq!(pub_scalars(&proof).len(), N_PUB_SCALARS);
        for which in 0..N_PUB_SCALARS {
            let mut p = prove(&params, &groups).1;
            tweak_pub_scalar(&mut p, which);
            assert!(!verify(&params, &bx, &p), "Phase A: field {which} is not bound ({tag})");
            let mut q = prove_nizk1(&params, &nz, &groups, &bw).1;
            tweak_pub_scalar(&mut q, which);
            assert!(!verify_nizk1(&params, &nz, &st, &q), "Phase B: field {which} is not bound ({tag})");
        }
    }

    fn tweak_pub_scalar(p: &mut Proof, which: usize) {
        let f: &mut FqExt = match which {
            0 => &mut p.s1,
            1 => &mut p.q_claim,
            2 => &mut p.claim_bin,
            3 => &mut p.u_final,
            4 => &mut p.open_bin,
            5 => &mut p.open_full,
            6 => &mut p.open_h_sc1,
            7 => &mut p.open_h_sum,
            8..=10 => &mut p.mask_r_evals[which - 8],
            11..=14 => &mut p.mask_totals[which - 11],
            _ => &mut p.mask_evals[which - 15],
        };
        *f = *f + FqExt::ONE;
    }

    #[test]
    fn sweep_moderate() {
        for &(n, g, ell, seed) in &[
            (8usize, 1usize, 1usize, 940u64),
            (8, 2, 1, 941),
            (8, 2, 2, 942),
            (8, 4, 2, 943),
            (12, 2, 1, 944),
            (12, 1, 3, 945),
            (16, 4, 5, 946),
        ] {
            sweep_one(n, g, ell, seed);
        }
    }

    #[test]
    #[ignore]
    fn sweep_heavy() {
        for &(n, g, ell, seed) in &[
            (16usize, 8usize, 1usize, 950u64),
            (16, 8, 3, 951),
            (24, 8, 2, 952),
            (32, 8, 1, 953),
            (128, 4, crate::params::ELL, 954),
            (128, 8, crate::params::ELL, 955),
        ] {
            sweep_one(n, g, ell, seed);
        }
    }

    #[test]
    fn commitment_arity_matches_opening_points() {
        for &(n, g, ell, seed) in &[(8usize, 2usize, 1usize, 611u64), (16, 4, 2, 612), (12, 2, 3, 613)]
        {
            let (params, groups) = setup(n, g, ell, seed);
            let d = dims(&params, None);
            let (_, _, p, _) = prove_impl(&params, &groups, Sabotage::None, None);
            let tag = format!("n={n} g={g} ell={ell}");

            assert_eq!(p.c.num_vars, d.nv, "c arity ({tag})");
            assert_eq!(p.c.num_vars, (d.merged.rows * SLOTS).trailing_zeros() as usize);
            assert_eq!(p.sc_full.rounds.len(), d.nv + 1, "SC_full ↔ r_f ({tag})");
            assert_eq!(p.sc_bin.rounds.len(), d.nv_bin + 1, "SC_bin ↔ r_bin ({tag})");
            assert_eq!(p.sc1.rounds.len(), d.nv_u + 1, "SC1 ↔ nv_u + w ({tag})");
            assert_eq!(d.nv_h, p.sc5.rounds.len() + g, "nv_h ↔ r₅‖½^w ({tag})");

            let r5 = vec![FqExt::ZERO; p.sc5.rounds.len()];
            for pt in [half_point(&d, &r5), vec![FqExt::ONE; d.nv_h]] {
                assert_eq!(pt.len(), d.nv_h, "length of half_point ({tag})");
                let hp = hpack_point(&d, &pt);
                assert_eq!(hp.len(), d.nv_bin, "length of hpack_point ({tag})");
                assert_eq!(bin_prefix(&d, &hp).len(), p.c.num_vars, "length of bin_prefix ({tag})");
            }
            let rb = vec![FqExt::ONE; d.nv_bin];
            assert_eq!(bin_prefix(&d, &rb).len(), p.c.num_vars);
            let bases = mask_bases(&d);
            let shapes = mask_shapes(&d);
            let (lo, hi) = (mask_start(&d), mask_start(&d) + MASK_ROWS * SLOTS);
            for i in 0..4 {
                let wt = sumcheck::Masker::weights_total_plain_degs(&shapes[i]);
                let w = mask_weights(bases[i], &wt);
                assert!(
                    w.iter().all(|&(k, _)| (lo..hi).contains(&k)),
                    "the weight of masker {i} is not in the mask row ({tag})"
                );
            }
        }
    }

    #[test]
    fn bin_subcube_opening_matches_shadow_bits() {
        for &(n, g, ell, seed) in &[(8usize, 2usize, 1usize, 1301u64), (16, 4, 5, 1302), (12, 2, 3, 1303)]
        {
            let (params, nz, groups, bw) = setup_b(n, g, ell, seed);
            let (bx, wit) = eval_h(&params, &groups);
            let (st, bq) = blind_statement(&params, &nz, &bx, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let hb = crate::relation::h_bits(&params, &groups);
            let mut rng = SimpleRng::new(seed ^ 0xB1);
            for phase_b in [false, true] {
                let c = phase_b.then_some(&ctx);
                let d = dims(&params, c);
                let metas = constraints(&params, c);
                let qs = compute_quotients(&params, &wit, c, phase_b.then_some(&bq));
                let zb = build_bin_table(&d, &hb, phase_b.then_some(&bw), c);
                let mut zf = build_merged_table(&params, &d, &wit, &qs, &metas, &zb);
                let ms = mask_start(&d);
                for i in ms..ms + MASK_ROWS * SLOTS {
                    zf[i] = rng.next_fq();
                }
                for _ in 0..3 {
                    let r: Vec<FqExt> = (0..d.nv_bin).map(|_| rng.next_fq4()).collect();
                    assert_eq!(
                        pcs::open_fq(&zf, &bin_prefix(&d, &r)),
                        pcs::open(&zb, &r),
                        "r_bin (n={n} g={g} phase_b={phase_b})"
                    );
                    let ru: Vec<FqExt> = (0..d.nv_h).map(|_| rng.next_fq4()).collect();
                    let hp = hpack_point(&d, &ru);
                    assert_eq!(pcs::open_fq(&zf, &bin_prefix(&d, &hp)), pcs::open(&zb, &hp), "H slice");
                    let ri: Vec<FqExt> = (0..d.nv_i).map(|_| rng.next_fq4()).collect();
                    let h5 = hpack_point(&d, &half_point(&d, &ri));
                    assert_eq!(pcs::open_fq(&zf, &bin_prefix(&d, &h5)), pcs::open(&zb, &h5), "half_point");
                }
            }
        }
    }

    #[test]
    fn merged_weight_closed_form_matches_materialized() {
        for &(n, g, ell, seed) in &[(8usize, 2usize, 2usize, 1401u64), (16, 4, 5, 1402), (12, 2, 3, 1403)]
        {
            let (params, nz, groups, bw) = setup_b(n, g, ell, seed);
            let (bx, _) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &bx, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));
            let metas = constraints(&params, Some(&ctx));
            let mut rng = SimpleRng::new(seed ^ 0xC1);
            let alpha = rng.next_fq4();
            let rows = build_rows(&params, &st.c_x, alpha, Some(&ctx));
            let rv = |rng: &mut SimpleRng, k: usize| -> Vec<FqExt> {
                (0..k).map(|_| rng.next_fq4()).collect()
            };
            let tau = rv(&mut rng, d.nv_c);
            let r_c = rv(&mut rng, d.nv_c);
            let r_v = rv(&mut rng, g);
            let gamma = rng.next_fq4();
            let lambda_f = rng.next_fq4();

            let eq_tau = eq_table(&tau);
            let c_hat = contract_a_hat(&rows, &r_v);
            let mut lg_full = vec![FqExt::ZERO; d.merged.rows];
            for row in &rows.lin {
                let w_m = eq_at_index(&tau, row.cell);
                for &(k, coef) in &row.full_entries {
                    lg_full[k] = lg_full[k] + w_m * coef;
                }
                if let Some(src) = u_src(&params, row.kind) {
                    let w_n = gamma * eq_at_index(&r_c, row.cell);
                    let cr = c_hat_row(&c_hat, row);
                    for t in 0..u_width(&params) {
                        let k = d.merged.b_row_of(&params, src, t);
                        lg_full[k] = lg_full[k] + w_n * cr[t];
                    }
                }
            }
            let mut w_row: Vec<FqExt> = lg_full.iter().map(|&v| lambda_f * v).collect();
            for m in &metas {
                if let Some(t) = m.t_index {
                    let r = d.merged.quot_row_of(t);
                    w_row[r] = w_row[r] + eq_tau[m.cell];
                }
            }
            let mut apow = Vec::with_capacity(SLOTS);
            let mut p = FqExt::ONE;
            for _ in 0..SLOTS {
                apow.push(p);
                p = p * alpha;
            }
            let mut wfull = vec![FqExt::ZERO; d.merged.rows * SLOTS];
            for (row, &w) in w_row.iter().enumerate() {
                for (s, &ap) in apow.iter().enumerate() {
                    wfull[row * SLOTS + s] = w * ap;
                }
            }
            for row in 0..d.merged.bin_rows {
                assert_eq!(w_row[row], FqExt::ZERO, "binary block row {row} carries weight");
            }
            for row in d.merged.mask_base..d.merged.rows {
                assert_eq!(w_row[row], FqExt::ZERO, "mask/empty row {row} carries weight");
            }

            for _ in 0..3 {
                let r = rv(&mut rng, d.nv);
                assert_eq!(
                    crate::mle::mle_eval(&wfull, &r),
                    merged_weight_eval(
                        &d, &params, &rows, &metas, &tau, &r_c, &c_hat, gamma, alpha, lambda_f, &r
                    ),
                    "n={n} g={g} ell={ell}"
                );
            }
        }
    }

    #[test]
    fn quotient_alpha_limit_n_is_sound() {
        for &(n, g, ell, seed) in &[(8usize, 2usize, 1usize, 1501u64), (16, 4, 5, 1502)] {
            let (params, nz, groups, bw) = setup_b(n, g, ell, seed);
            let (st0, _) = prove_nizk1(&params, &nz, &groups, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st0);
            let d = dims(&params, Some(&ctx));
            let nq = num_quotients(&params, Some(&ctx));
            for t in [0usize, nq / 2, nq - 1] {
                let sab = Sabotage::PerturbFull { row: d.merged.quot_row_of(t), coef: N - 1 };
                let (_, st, p, _) = prove_impl(&params, &groups, sab, Some((&nz, &bw)));
                assert!(
                    !verify_nizk1(&params, &nz, &st.expect("Phase B"), &p),
                    "a non-zero in slot N−1 of quotient {t} unexpectedly passed (n={n} g={g})"
                );
            }
        }
    }

    #[test]
    fn constraints_only_touch_witness_rows() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (12, 2, 3), (16, 4, 5)] {
            let (params, nz, groups, bw) = setup_b(n, g, ell, 902);
            let (bx, _) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &bx, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let mut rng = SimpleRng::new(7);
            let rows = build_rows(&params, &st.c_x, rng.next_fq4(), Some(&ctx));
            let d = dims(&params, Some(&ctx));
            assert_eq!(d.bin_rows, ctx.w.total);
            assert!(d.bin_pad >= d.bin_rows && d.bin_pad.is_power_of_two());
            for row in &rows.lin {
                for &(k, _) in &row.bin_entries {
                    assert!(k < d.bin_rows, "constraint {:?} points outside the witness of c_bin {k}", row.kind);
                }
                for &(k, _) in &row.full_entries {
                    assert!(
                        (d.merged.b_base..d.merged.quot_base).contains(&k),
                        "constraint {:?} points outside the b segment {k}",
                        row.kind
                    );
                }
            }
        }
    }

    #[test]
    fn mask_table_embedding_is_recoverable() {
        use crate::rng::CsRng;
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 4, 5), (16, 8, 1)] {
            let params = HashParams::sample(800, n, g, ell);
            let nz = Nizk1Params::sample(801, &params, 3, 2, 2);
            let st = BlindStatement {
                j: 0,
                c_x: vec![RingElem::zero(); params.ell],
                c_r: vec![RingElem::zero(); nz.com_r.out_len()],
                d_x: vec![RingElem::zero(); nz.com_x.out_len()],
            };
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));

            let mut rng = CsRng::from_parts("embed-test", &[&insecure_test_secret(802)]);
            let shapes = mask_shapes(&d);
            let bases = mask_bases(&d);
            let mut ms: Vec<sumcheck::Masker> =
                shapes.iter().map(|sh| sumcheck::Masker::with_degs(&mut rng, sh)).collect();
            let ozk: [FqExt; OPENZK_VALS] = std::array::from_fn(|_| rng.next_fq4());

            let mut zm = vec![Fq::ZERO; d.merged.rows * SLOTS];
            write_mask_coefs(&mut zm, &ms, &bases);
            write_openzk_vals(&d, &mut zm, &ozk);
            assert_mask_fits(&d);
            let (lo, hi) = (mask_start(&d), mask_start(&d) + mask_table_len(&d));
            assert!(zm[..lo].iter().all(|v| *v == Fq::ZERO), "the mask was written before the mask row");
            assert!(zm[hi..].iter().all(|v| *v == Fq::ZERO), "the mask was written after the mask row");
            assert!(hi <= (d.merged.mask_base + MASK_ROWS) * SLOTS, "the mask overflows the reserved rows");

            let mut tau_rng = CsRng::from_parts("tau", &[&insecure_test_secret(803)]);
            for i in 0..4 {
                let nv = shapes[i].len();
                assert_eq!(
                    pcs::open_linear_fq(
                        &zm,
                        &mask_weights(bases[i], &sumcheck::Masker::weights_total_plain_degs(&shapes[i]))
                    ),
                    ms[i].total_plain(),
                    "the total of masker {i} cannot be recovered (n={n} g={g} ell={ell})"
                );
                let r: Vec<FqExt> = (0..nv).map(|_| tau_rng.next_fq4()).collect();
                for (j, &rj) in r.iter().enumerate() {
                    ms[i].fold(j, rj);
                }
                assert_eq!(
                    pcs::open_linear_fq(
                        &zm,
                        &mask_weights(bases[i], &sumcheck::Masker::weights_eval_degs(&shapes[i], &r))
                    ),
                    ms[i].eval(),
                    "g(r) of masker {i} cannot be recovered"
                );
            }
            for i in 0..OPENZK_VALS {
                let w = [FqExt::ONE; 1];
                assert_eq!(
                    pcs::open_linear_fq(&zm, &mask_weights(openzk_base(&d, i), &w)),
                    ozk[i],
                    "open-ZK value {i} cannot be recovered"
                );
            }
            let (lf, gm, zu, xn1, c) = (
                tau_rng.next_fq4(),
                tau_rng.next_fq4(),
                tau_rng.next_fq4(),
                tau_rng.next_fq4(),
                tau_rng.next_fq4(),
            );
            let nf = n_full_coefs(&ozk, lf, gm, zu, xn1);
            assert_eq!(
                pcs::open_linear_fq(
                    &zm,
                    &mask_weights(openzk_base(&d, R_B), &n_full_weights(lf, gm, zu, xn1, c))
                ),
                ClaimMask { coef: &nf }.eval(c),
                "the weights and coefficients of N_full disagree (prove and verify would each compute their own)"
            );
            let nb: Vec<FqExt> = (0..R_COEFS).map(|k| lf * ozk[R_N + k]).collect();
            assert_eq!(
                pcs::open_linear_fq(&zm, &mask_weights(openzk_base(&d, R_N), &n_bin_weights(lf, c))),
                ClaimMask { coef: &nb }.eval(c),
                "the weights and coefficients of N_bin disagree"
            );
        }
    }

    #[test]
    fn mask_row_layout_is_contiguous_and_fits() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 4, 5), (128, 8, 3)] {
            let (params, nz, groups, bw) = setup_b(n, g, ell, 940);
            let (bx, _) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &bx, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            for d in [dims(&params, Some(&ctx)), dims(&params, None)] {
                let bases = mask_bases(&d);
                let shapes = mask_shapes(&d);
                let mut acc = mask_start(&d);
                assert_eq!(acc, d.merged.mask_base * SLOTS);
                for i in 0..4 {
                    assert_eq!(bases[i], acc, "the start offset of masker {i} is misaligned");
                    acc += EXT_DEG * sumcheck::Masker::n_coefs(&shapes[i]);
                }
                assert_eq!(bases[4], acc, "the open-ZK region does not follow the maskers");
                assert_eq!(mask_table_len(&d), acc + EXT_DEG * OPENZK_VALS - mask_start(&d));
                assert!(mask_table_len(&d) <= MASK_ROWS * SLOTS, "the mask row does not fit");
                assert!(mask_start(&d) >= d.merged.quot_base * SLOTS, "the mask row comes before the quotient segment");
                assert_eq!(R_Q, R_B + R_COEFS);
                assert_eq!(R_U, R_Q + R_COEFS);
                assert_eq!(R_N, R_U + R_COEFS);
                let w = mask_weights(openzk_base(&d, R_B), &[FqExt::ONE; 4 * R_COEFS]);
                assert_eq!(w.len(), 4 * R_COEFS * EXT_DEG);
                assert!(w.iter().all(|&(i, _)| i < d.merged.rows * SLOTS));
                assert_eq!(d.bin_pad, d.bin_rows.next_power_of_two().max(2));
            }
        }
    }

    #[test]
    fn lg_full_eval_matches_naive_u_cube_expansion() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (8, 1, 1), (16, 4, 5)] {
            let (params, _) = setup(n, g, ell, 900 + (n * 10 + g) as u64);
            let mut rng = SimpleRng::new(0xA11CE ^ (n as u64) << 8 ^ g as u64);
            let alpha = rng.next_fq4();
            let stmt: Vec<RingElem> =
                (0..ell).map(|_| RingElem { c: (0..N).map(|_| rng.next_fq()).collect() }).collect();
            let rows = build_rows(&params, &stmt, alpha, None);
            let d = dims(&params, None);
            let nv_k = d.nv - SLOT_VARS;
            let rv = |rng: &mut SimpleRng, k: usize| -> Vec<FqExt> {
                (0..k).map(|_| rng.next_fq4()).collect()
            };
            let tau = rv(&mut rng, d.nv_c);
            let r_c = rv(&mut rng, d.nv_c);
            let r_v = rv(&mut rng, params.group_bits);
            let gamma = rng.next_fq4();
            let r_k = rv(&mut rng, nv_k);

            let c_hat = contract_a_hat(&rows, &r_v);
            let fast = lg_full_eval(&params, &rows, &tau, &r_c, &c_hat, gamma, &r_k);

            let r_u: Vec<FqExt> = r_c.iter().chain(&r_v).copied().collect();
            let mut naive = FqExt::ZERO;
            for row in &rows.lin {
                let w_m = eq_at_index(&tau, row.cell);
                for &(k, coef) in &row.full_entries {
                    naive = naive + w_m * coef * eq_at_index(&r_k, k);
                }
                if let Some(src) = u_src(&params, row.kind) {
                    for v in 0..d.tsz {
                        let e = gamma * eq_at_index(&r_u, row.cell * d.tsz + v);
                        for t in 0..u_width(&params) {
                            naive = naive
                                + e * rows.a_base[row.step_i - 1][v][row.chain][t]
                                    * eq_at_index(&r_k, rows.b_base + b_row(&params, src, t));
                        }
                    }
                }
            }
            assert_eq!(fast, naive, "n={n} g={g} ell={ell}");
        }
    }

    #[test]
    fn lg_bin_eval_matches_the_sparse_rows() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 4, 5)] {
            let (params, nz, groups, bw) = setup_b(n, g, ell, 910);
            let (bx, _) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &bx, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));
            let mut rng = SimpleRng::new(0xB11 ^ n as u64);
            let rows = build_rows(&params, &st.c_x, rng.next_fq4(), Some(&ctx));
            let tau: Vec<FqExt> = (0..d.nv_c).map(|_| rng.next_fq4()).collect();
            let nv_k = d.bin_pad.trailing_zeros() as usize;
            let r_k: Vec<FqExt> = (0..nv_k).map(|_| rng.next_fq4()).collect();

            let mut lg = vec![FqExt::ZERO; d.bin_pad];
            for row in &rows.lin {
                let w = eq_at_index(&tau, row.cell);
                for &(k, coef) in &row.bin_entries {
                    lg[k] = lg[k] + w * coef;
                }
            }
            let eqk = eq_table(&r_k);
            let dense = (0..d.bin_pad).fold(FqExt::ZERO, |a, k| a + eqk[k] * lg[k]);
            assert_eq!(lg_bin_eval(&rows, &tau, &r_k), dense, "n={n} g={g} ell={ell}");
        }
    }

    const GOLDEN_ROUNDS_W4: usize = 54;
    const GOLDEN_TRANSCRIPT_W4: usize = 2944;
    const GOLDEN_TOTAL_W4: usize = 3188;
    const GOLDEN_FP_W4: u64 = 0x2999_3386_3add_cf25;

    #[test]
    #[ignore]
    fn golden_fingerprint_and_sizes_at_the_production_parameters() {
        let params = HashParams::sample(20260901, 128, 4, crate::params::ELL);
        let nz = Nizk1Params::sample(
            20260902,
            &params,
            crate::nizk1::R_DIM,
            crate::nizk1::COM_N,
            crate::nizk1::W_SLACK,
        );
        let mut rng = SimpleRng::new(42);
        let bits: Vec<bool> = (0..128).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        let bw = sample_blind(
            QueryTicket::insecure_for_tests(insecure_test_secret(20260903), 0),
            &params,
            &nz,
            &groups,
        );
        let (st, p) = prove_nizk1(&params, &nz, &groups, &bw);
        assert!(verify_nizk1(&params, &nz, &st, &p));
        let b = p.size_breakdown();
        println!(
            "GOLDEN w=4: fp={:#018x} transcript={} total={} rounds={}",
            proof_fingerprint(&p),
            b.transcript(),
            b.total(),
            p.num_rounds()
        );
        assert_eq!(p.num_rounds(), GOLDEN_ROUNDS_W4, "round count");
        assert_eq!(b.transcript(), GOLDEN_TRANSCRIPT_W4, "transcript bytes");
        assert_eq!(b.total(), GOLDEN_TOTAL_W4, "total bytes");
        assert_eq!(proof_fingerprint(&p), GOLDEN_FP_W4, "proof fingerprint");
    }

    #[test]
    fn verify_never_panics_on_malformed_proofs() {
        let (params, nz, groups, bw) = setup_b(8, 2, 2, 1201);
        let (bx, base_a) = prove(&params, &groups);
        let (st, base_b) = prove_nizk1(&params, &nz, &groups, &bw);
        let mut rng = SimpleRng::new(0xDEAD);

        let empty = || Proof {
            c: pcs::Commitment { digest: 0, num_vars: 0 },
            s1: FqExt::ZERO,
            q_claim: FqExt::ZERO,
            claim_bin: FqExt::ZERO,
            u_final: FqExt::ZERO,
            open_bin: FqExt::ZERO,
            open_full: FqExt::ZERO,
            open_h_sc1: FqExt::ZERO,
            open_h_sum: FqExt::ZERO,
            mask_r_evals: [FqExt::ZERO; 3],
            sc1: SumcheckProof { rounds: vec![] },
            sc_full: SumcheckProof { rounds: vec![] },
            sc_bin: SumcheckProof { rounds: vec![] },
            sc5: SumcheckProof { rounds: vec![] },
            mask_totals: [FqExt::ZERO; 4],
            mask_evals: [FqExt::ZERO; 4],
        };
        assert!(!verify(&params, &bx, &empty()));
        assert!(!verify_nizk1(&params, &nz, &st, &empty()));

        for trial in 0..200 {
            let mut p = if trial % 2 == 0 {
                prove(&params, &groups).1
            } else {
                prove_nizk1(&params, &nz, &groups, &bw).1
            };
            let scs: [&mut SumcheckProof; 4] =
                [&mut p.sc1, &mut p.sc_full, &mut p.sc_bin, &mut p.sc5];
            let sc = scs.into_iter().nth((rng.next_u64() % 4) as usize).unwrap();
            match rng.next_u64() % 5 {
                0 => {
                    sc.rounds.pop();
                }
                1 => sc.rounds.push(vec![FqExt::ZERO; 3]),
                2 => {
                    if let Some(r) = sc.rounds.first_mut() {
                        r.pop();
                    }
                }
                3 => {
                    if let Some(r) = sc.rounds.last_mut() {
                        r.push(FqExt::ONE);
                    }
                }
                _ => {
                    sc.rounds.clear();
                }
            }
            if rng.next_bool() {
                p.c.num_vars = (rng.next_u64() % 40) as usize;
            }
            let _ = verify(&params, &bx, &p);
            let _ = verify_nizk1(&params, &nz, &st, &p);
        }
        assert!(verify(&params, &bx, &base_a));
        assert!(verify_nizk1(&params, &nz, &st, &base_b));
    }

    #[test]
    fn malformed_statements_are_rejected() {
        let (params, nz, groups, bw) = setup_b(8, 2, 2, 1101);
        let (st, p) = prove_nizk1(&params, &nz, &groups, &bw);
        assert!(verify_nizk1(&params, &nz, &st, &p));

        let rebuild = |f: &dyn Fn(&mut BlindStatement)| {
            let mut s = BlindStatement {
                j: st.j,
                c_x: st.c_x.clone(),
                c_r: st.c_r.clone(),
                d_x: st.d_x.clone(),
            };
            f(&mut s);
            s
        };
        for bad in [
            rebuild(&|s| {
                s.c_x[0].c.pop();
            }),
            rebuild(&|s| {
                s.c_r[0].c.push(Fq::ZERO);
            }),
            rebuild(&|s| {
                s.d_x[0].c.truncate(1);
            }),
            rebuild(&|s| {
                s.c_x.pop();
            }),
            rebuild(&|s| {
                s.c_r.pop();
            }),
            rebuild(&|s| {
                s.d_x.push(RingElem::zero());
            }),
        ] {
            assert!(!verify_nizk1(&params, &nz, &bad, &p), "a malformed length unexpectedly passed");
        }
        for bad in [
            rebuild(&|s| s.c_x[0].c[0] = Fq(crate::field::Q)),
            rebuild(&|s| s.c_r[0].c[N - 1] = Fq(u64::MAX)),
            rebuild(&|s| s.d_x[0].c[7] = Fq(crate::field::Q + 1)),
        ] {
            assert!(!verify_nizk1(&params, &nz, &bad, &p), "a non-reduced Fq unexpectedly passed");
        }
        let (bx, pa) = prove(&params, &groups);
        assert!(verify(&params, &bx, &pa));
        let mut bad = bx.clone();
        bad[0].c[0] = Fq(crate::field::Q);
        assert!(!verify(&params, &bad, &pa), "Phase A: a non-reduced Fq unexpectedly passed");
        let mut short = bx.clone();
        short[0].c.pop();
        assert!(!verify(&params, &short, &pa), "Phase A: a wrong length unexpectedly passed");
    }

    #[test]
    #[ignore]
    fn verify_breakdown() {
        use std::time::Instant;
        const REPS: u32 = 20;
        for &w in &[4usize, 8] {
            let (params, nz, groups, bw) = setup_b(128, w, crate::params::ELL, 4242);
            let (st, p) = prove_nizk1(&params, &nz, &groups, &bw);
            assert!(verify_nizk1(&params, &nz, &st, &p));
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));
            let mut rng = SimpleRng::new(7);
            let alpha = rng.next_fq4();

            let t = Instant::now();
            for _ in 0..REPS {
                std::hint::black_box(build_rows(&params, &st.c_x, alpha, Some(&ctx)));
            }
            let t_rows = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;

            let rows = build_rows(&params, &st.c_x, alpha, Some(&ctx));
            let rv = |rng: &mut SimpleRng, k: usize| -> Vec<FqExt> {
                (0..k).map(|_| rng.next_fq4()).collect()
            };
            let tau = rv(&mut rng, d.nv_c);
            let r_c = rv(&mut rng, d.nv_c);
            let r_v = rv(&mut rng, w);
            let gamma = rng.next_fq4();
            let nv_k = d.nv - SLOT_VARS;
            let r_k = rv(&mut rng, nv_k);
            let r_kb = rv(&mut rng, d.bin_pad.trailing_zeros() as usize);

            let t = Instant::now();
            for _ in 0..REPS {
                let ch = contract_a_hat(&rows, &r_v);
                std::hint::black_box(ppub_sum(&rows, &tau));
                std::hint::black_box(pu_sum(&rows, &r_c, &r_v));
                std::hint::black_box(lg_full_eval(&params, &rows, &tau, &r_c, &ch, gamma, &r_k));
                std::hint::black_box(lg_bin_eval(&rows, &tau, &r_kb));
            }
            let t_sparse = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;

            let t = Instant::now();
            for _ in 0..REPS {
                assert!(verify_nizk1(&params, &nz, &st, &p));
            }
            let t_total = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;

            let table = params.num_matrices() * params.ell * params.ell;
            println!(
                "w={w}: verify {t_total:.3} ms | build_rows {t_rows:.3} ms ({:.0}%, {table} ring-element evaluations)        | sparse eval {t_sparse:.3} ms ({:.0}%) | rest {:.3} ms ({:.0}%)",
                100.0 * t_rows / t_total,
                100.0 * t_sparse / t_total,
                t_total - t_rows - t_sparse,
                100.0 * (t_total - t_rows - t_sparse) / t_total
            );
            println!(
                "     compared with BP14 at the same (w,m) (shared symbols): public table 2^w·m·(m·8) = {} ring elements (BLMR per-block is G/8 = {:.2}× of it)",
                params.table_size() * params.ell * params.ell * crate::report::BP14_GADGET_LEN,
                params.num_groups() as f64 / crate::report::BP14_GADGET_LEN as f64
            );
        }
    }

}
