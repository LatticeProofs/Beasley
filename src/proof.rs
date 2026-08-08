use crate::bits::PackedBits;
use crate::ext_field::{FqExt, LazyExtSum, EXT_DEG};
use crate::field::Fq;
use crate::hash::eval_h;
use crate::mle::{eq_at_index, eq_eval, eq_table};
use crate::params::HashParams;
use crate::pcs;
use crate::nizk1::{blind_statement, BlindStatement, BlindWitness, Nizk1Params};
use crate::relation::{
    bit_scale, build_rows, check_witness, compute_quotients, constraints, g_pad, m_row,
    m_row_flat, ml_bits,
    phase_a_cells, u_src, CKind, LinRow, Nizk1Ctx, Rows,
};
use crate::ring::{
    RingElem, DIGIT_BITS, GADGET_BASE, GADGET_LEN, M_BIT_ROWS, N, W_RANGE_BASE,
};
use crate::sumcheck::{self, SumcheckProof};
use crate::transcript::Transcript;
use rayon::prelude::*;

pub const W_COEF_VARS: usize = N.trailing_zeros() as usize;
pub const W_COEF_SLOTS: usize = 1 << W_COEF_VARS;

pub const T_COEF_VARS: usize = W_COEF_VARS;
pub const T_COEF_SLOTS: usize = 1 << T_COEF_VARS;

const _: () = assert!(W_COEF_SLOTS == N, "the number of W slots must be exactly N");
const _: () = assert!(T_COEF_SLOTS == N, "the number of T slots must be exactly N (masking moved to W)");

pub const T_ALPHA_LIMIT: usize = N - 1;

const _: () = assert!(T_ALPHA_LIMIT < T_COEF_SLOTS, "T must leave at least one mask slot whose α weight is 0");

pub const MASK_COEF_ROWS: usize = 128;

const MASK_FQ_BITS: usize = M_BIT_ROWS;
const _: () = assert!(W_COEF_SLOTS % MASK_FQ_BITS == 0, "a row must divide evenly into a whole number of F_q elements");

pub const R_COEFS: usize = 3;
pub const SIG_COEFS: usize = sumcheck::LIN_MASK_COEFS;
pub const OPENZK_VALS: usize = 3 * SIG_COEFS + 3 * R_COEFS;

pub(crate) const SIG_W: usize = 0;
pub(crate) const SIG_H: usize = SIG_COEFS;
pub(crate) const SIG_T: usize = 2 * SIG_COEFS;
pub(crate) const R_B: usize = 3 * SIG_COEFS;
pub(crate) const R_Q: usize = R_B + R_COEFS;
pub(crate) const R_U: usize = R_Q + R_COEFS;

const W_ROUND_DEG_3: usize = 3;
const W_ROUND_DEG_2: usize = 2;

fn z_eval(pt: &[FqExt]) -> FqExt {
    pt.iter().fold(FqExt::ONE, |a, &z| a * z * (FqExt::ONE - z))
}

fn ind_eval(pt: &[FqExt]) -> FqExt {
    pt.iter().fold(FqExt::ONE, |a, &z| a * (FqExt::ONE - z))
}

fn m_weights(gamma: FqExt, z_u: FqExt, xn1: FqExt, lambda: FqExt, c: FqExt) -> Vec<FqExt> {
    let pw = sumcheck::ClaimMask::weights_eval(R_COEFS - 1, c);
    let mut out = Vec::with_capacity(3 * R_COEFS);
    out.extend(pw.iter().map(|&p| lambda * p));
    out.extend(pw.iter().map(|&p| lambda * xn1 * p));
    out.extend(pw.iter().map(|&p| lambda * gamma * z_u * p));
    out
}

pub struct Proof {
    pub c_w: pcs::Commitment,
    pub c_t: pcs::Commitment,

    pub s1: FqExt,
    pub q_claim: FqExt,
    pub u_final: FqExt,
    pub open_w: FqExt,
    pub open_t: FqExt,
    pub open_h_sc1: FqExt,
    pub open_h_sum: FqExt,
    pub mask_r_evals: [FqExt; 3],

    pub sc1_bilinear: SumcheckProof,
    pub sc_quotient: SumcheckProof,
    pub sc_batched: SumcheckProof,
    pub sc5_onehot: SumcheckProof,

    pub mask_totals: [FqExt; 4],
    pub mask_evals: [FqExt; 4],
}

pub fn proof_fingerprint(p: &Proof) -> u64 {
    #[inline]
    fn mix(d: &mut u64, x: u64) {
        *d ^= x;
        *d = d.wrapping_mul(0x100000001b3);
    }
    #[inline]
    fn mix_fq4(d: &mut u64, v: FqExt) {
        for c in v.0 {
            mix(d, c.0 as u64);
        }
    }
    let d = &mut 0xcbf29ce484222325u64;
    for c in [&p.c_w, &p.c_t] {
        mix(d, c.digest);
        mix(d, c.num_vars as u64);
    }
    for v in [
        p.s1,
        p.q_claim,
        p.u_final,
        p.open_w,
        p.open_t,
        p.open_h_sc1,
        p.open_h_sum,
    ] {
        mix_fq4(d, v);
    }
    for v in p.mask_r_evals.iter().chain(p.mask_totals.iter()).chain(p.mask_evals.iter()) {
        mix_fq4(d, *v);
    }
    for sc in [&p.sc1_bilinear, &p.sc_quotient, &p.sc_batched, &p.sc5_onehot] {
        mix(d, sc.rounds.len() as u64);
        for r in &sc.rounds {
            mix(d, r.len() as u64);
            for &v in r {
                mix_fq4(d, v);
            }
        }
    }
    *d
}

fn bit_poly(z: FqExt) -> FqExt {
    z * (z - FqExt::ONE)
}

fn challenge_vec(tr: &mut Transcript, n: usize) -> Vec<FqExt> {
    (0..n).map(|_| tr.challenge_fq4()).collect()
}

fn mask_bit(d: &Dims, flat: usize, b: usize) -> usize {
    debug_assert!(b < MASK_FQ_BITS);
    let idx = flat * MASK_FQ_BITS + b;
    assert!(
        idx < MASK_COEF_ROWS * W_COEF_SLOTS,
        "the ZK mask coefficients do not fit in the W mask region: {} F_q elements needed, only {} available (increase MASK_COEF_ROWS)",
        flat + 1,
        MASK_COEF_ROWS * W_COEF_SLOTS / MASK_FQ_BITS
    );
    (d.w_mask_coef + idx / W_COEF_SLOTS) * W_COEF_SLOTS + idx % W_COEF_SLOTS
}

fn mask_weights(d: &Dims, base: usize, w: &[FqExt]) -> Vec<(usize, FqExt)> {
    let mut out = Vec::with_capacity(w.len() * EXT_DEG * MASK_FQ_BITS);
    for (j, &wj) in w.iter().enumerate() {
        for a in 0..EXT_DEG {
            let mut basis = FqExt::ZERO;
            basis.0[a] = Fq::ONE;
            let bw = basis * wj;
            let flat = base + EXT_DEG * j + a;
            for b in 0..MASK_FQ_BITS {
                out.push((mask_bit(d, flat, b), bw * FqExt::from_fq(crate::ring::bit_weight(b))));
            }
        }
    }
    out
}

fn write_mask_bits(d: &Dims, zw: &mut PackedBits, maskers: &[sumcheck::Masker], bases: &[usize; 5]) {
    for (i, m) in maskers.iter().enumerate() {
        for (j, c) in m.coeffs_flat().enumerate() {
            for a in 0..EXT_DEG {
                let v = c.0[a].0 as u64;
                let flat = bases[i] + EXT_DEG * j + a;
                for b in 0..MASK_FQ_BITS {
                    zw.or_bit(mask_bit(d, flat, b), ((v >> b) & 1) as u32);
                }
            }
        }
    }
}

fn mask_shapes(d: &Dims) -> [(usize, usize); 4] {
    [(d.nv_u + 1, 3), (d.nv_t() + 1, 2), (d.nv_w + 1, 3), (d.nv_i, 2)]
}

fn mask_bases(d: &Dims) -> [usize; 5] {
    let sh = mask_shapes(d);
    let mut out = [0usize; 5];
    let mut acc = 0;
    for i in 0..4 {
        out[i] = acc;
        acc += EXT_DEG * sh[i].0 * (sh[i].1 + 1);
    }
    out[4] = acc;
    out
}

#[inline]
fn openzk_base(d: &Dims, i: usize) -> usize {
    debug_assert!(i < OPENZK_VALS);
    mask_bases(d)[4] + EXT_DEG * i
}

fn write_openzk_vals(d: &Dims, zw: &mut PackedBits, vals: &[FqExt; OPENZK_VALS]) {
    let base = mask_bases(d)[4];
    for (j, v) in vals.iter().enumerate() {
        for a in 0..EXT_DEG {
            let x = v.0[a].0 as u64;
            let flat = base + EXT_DEG * j + a;
            for b in 0..MASK_FQ_BITS {
                zw.or_bit(mask_bit(d, flat, b), ((x >> b) & 1) as u32);
            }
        }
    }
}

fn alpha_tensor_eval(r_l: &[FqExt], alpha: FqExt, limit: usize) -> FqExt {
    let nv = r_l.len();
    let mut pre = Vec::with_capacity(nv + 1);
    pre.push(FqExt::ONE);
    let mut ap = alpha;
    for j in 0..nv {
        let r = r_l[nv - 1 - j];
        pre.push(pre[j] * (FqExt::ONE - r + r * ap));
        ap = ap * ap;
    }
    if limit >= (1usize << nv) {
        return pre[nv];
    }
    let mut acc = FqExt::ZERO;
    let mut prefix_eq = FqExt::ONE;
    for i in (0..nv).rev() {
        let r = r_l[nv - 1 - i];
        if (limit >> i) & 1 == 1 {
            let hi = (limit >> (i + 1)) << (i + 1);
            acc = acc + prefix_eq * (FqExt::ONE - r) * alpha.pow(hi as u128) * pre[i];
            prefix_eq = prefix_eq * r;
        } else {
            prefix_eq = prefix_eq * (FqExt::ONE - r);
        }
    }
    acc
}

fn transcript_init(params: &HashParams, ch: &[RingElem]) -> Transcript {
    let mut tr = Transcript::new("proof-of-hash-v3-onehot");
    tr.absorb_u64(params.n_bits as u64);
    tr.absorb_u64(params.group_bits as u64);
    tr.absorb_u64(params.ell as u64);
    tr.absorb_digest(&params.crs_digest);
    for c in ch {
        tr.absorb_fqs(&c.c);
    }
    tr
}

fn contract_a_hat(rows: &Rows, r_v: &[FqExt]) -> Vec<Vec<FqExt>> {
    let eqv = eq_table(r_v);
    let (m, ml) = (rows.a_base[0].len(), rows.a_base[0][0].len());
    let mut cb = vec![vec![FqExt::ZERO; ml]; m];
    for (v, &e) in eqv.iter().take(rows.a_base.len()).enumerate() {
        let av = &rows.a_base[v];
        for r in 0..m {
            let (dst, src) = (&mut cb[r], &av[r]);
            for d in 0..ml {
                dst[d] = dst[d] + e * src[d];
            }
        }
    }
    let pow2 = bit_scale();
    cb.iter()
        .map(|row| {
            (0..ml * DIGIT_BITS)
                .map(|e| {
                    #[allow(clippy::modulo_one)]
                    let b = e % DIGIT_BITS;
                    pow2[b] * row[e / DIGIT_BITS]
                })
                .collect()
        })
        .collect()
}

fn lg_mle_eval(
    params: &HashParams,
    rows: &Rows,
    tau: &[FqExt],
    r_c: &[FqExt],
    c_hat: &[Vec<FqExt>],
    gamma: FqExt,
    r_k: &[FqExt],
) -> FqExt {
    let mlb = ml_bits(params);
    let mut acc = FqExt::ZERO;
    for row in &rows.lin {
        let w_m = eq_at_index(tau, row.cell);
        for &(k, coef) in &row.m_entries {
            acc = acc + w_m * coef * eq_at_index(r_k, k);
        }
        if let Some(src) = u_src(row.kind) {
            let w_n = gamma * eq_at_index(r_c, row.cell);
            let cr = &c_hat[row.chain];
            for e in 0..mlb {
                acc = acc + w_n * cr[e] * eq_at_index(r_k, m_row_flat(params, src, e));
            }
        }
    }
    acc
}

fn ppub_sum(rows: &Rows, point: &[FqExt]) -> FqExt {
    rows.lin.iter().fold(FqExt::ZERO, |a, row| a + eq_at_index(point, row.cell) * row.p_pub)
}

fn pu_sum(rows: &Rows, r_c: &[FqExt], r_v: &[FqExt]) -> FqExt {
    let eqv = eq_table(r_v);
    let mut acc = FqExt::ZERO;
    for row in &rows.lin {
        if row.kind != CKind::Base {
            continue;
        }
        let inner = eqv
            .iter()
            .take(rows.a_base.len())
            .enumerate()
            .fold(FqExt::ZERO, |a, (v, &e)| a + e * rows.u_pub(row, v));
        acc = acc + eq_at_index(r_c, row.cell) * inner;
    }
    acc
}

struct Dims {
    g_pad: usize,
    tsz: usize,
    hv: usize,
    #[allow(dead_code)]
    nv_j: usize,
    nv_c: usize,
    nv_u: usize,
    nv_i: usize,
    nv_h: usize,
    kw_pad: usize,
    nv_w: usize,
    w_mask_coef: usize,
    c_cells: usize,
    u_cells: usize,
    h_cells: usize,
    h_row: usize,
    hpack_rows: usize,
    hpack_per: usize,
    hpack_per_bits: u32,
}

impl Dims {
    fn nv_t(&self) -> usize {
        (self.c_cells * T_COEF_SLOTS).trailing_zeros() as usize
    }
}

fn dims(params: &HashParams, nz: Option<&Nizk1Ctx>) -> Dims {
    let ell_pad = params.ell.next_power_of_two();
    let g_pad = g_pad(params);
    let tsz = params.table_size();
    let w = crate::nizk1::w_layout(params, nz.map(|c| c.nz));
    let base_rows = w.total;
    let w_rows = base_rows + MASK_COEF_ROWS;
    let kw_pad = w_rows.next_power_of_two();
    let nv_j = ell_pad.trailing_zeros() as usize;
    let nv_i = g_pad.trailing_zeros() as usize;
    let gb = params.group_bits;
    let c_cells = (phase_a_cells(params) + nz.map_or(0, |c| c.num_rows())).next_power_of_two();
    let nv_c = c_cells.trailing_zeros() as usize;
    let hv = crate::relation::hv_len(params);
    let hvb = hv.trailing_zeros() as usize;
    debug_assert_eq!(hvb, gb);
    debug_assert!(
        c_cells >= g_pad,
        "c_cells ({c_cells}) < g_pad ({g_pad}) -- the τ_lo slice in omega_eval would go out of bounds"
    );
    Dims {
        g_pad,
        tsz,
        hv,
        nv_j,
        nv_c,
        nv_u: nv_c + hvb,
        nv_i,
        nv_h: nv_i + hvb,
        kw_pad,
        nv_w: (kw_pad * W_COEF_SLOTS).trailing_zeros() as usize,
        w_mask_coef: base_rows,
        c_cells,
        u_cells: c_cells * hv,
        h_cells: crate::relation::h_cells(params),
        h_row: w.h_pack,
        hpack_rows: crate::relation::hpack_rows(params),
        hpack_per: crate::relation::hpack_per(params),
        hpack_per_bits: crate::relation::hpack_per(params).trailing_zeros(),
    }
}

fn put_bits(zw: &mut PackedBits, k: usize, poly: &[Fq]) {
    let acc = poly.iter().fold(0u64, |a, &v| a | v.0 as u64);
    assert!(acc < W_RANGE_BASE, "W may only hold bits, got a row containing {acc} (did the digit skip its base-2 decomposition? see q64.md §4b)");
    let base = k * W_COEF_SLOTS;
    for (c, &v) in poly.iter().enumerate() {
        zw.or_bit(base + c, v.0 as u32);
    }
}

fn put_digit_bit(zw: &mut PackedBits, k: usize, digit: &[Fq], b: usize) {
    debug_assert!(b < DIGIT_BITS);
    let base = k * W_COEF_SLOTS;
    for (c, &v) in digit.iter().enumerate() {
        debug_assert!((v.0 as u64) < GADGET_BASE, "digit out of range [0,B)");
        zw.or_bit(base + c, (((v.0 as u64) >> b) & 1) as u32);
    }
}

#[inline]
fn h_cell(d: &Dims, cell: usize, v: usize) -> usize {
    (cell % d.g_pad) * d.hv + v
}

#[inline]
fn h_slot(d: &Dims, idx: usize) -> usize {
    debug_assert!(idx < d.h_cells);
    debug_assert!(d.hpack_per.is_power_of_two());
    (d.h_row + (idx >> d.hpack_per_bits)) * W_COEF_SLOTS + (idx & (d.hpack_per - 1))
}

#[inline]
fn h_bit(zw: &PackedBits, d: &Dims, idx: usize) -> bool {
    zw.get(h_slot(d, idx))
}

fn half_point(d: &Dims, r_i: &[FqExt]) -> Vec<FqExt> {
    let half = FqExt::from_u64(2).inv();
    let g = d.tsz.trailing_zeros() as usize;
    r_i.iter().copied().chain(std::iter::repeat(half).take(g)).collect()
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

pub fn prove(params: &HashParams, groups: &[usize]) -> (Vec<RingElem>, Proof) {
    let (ch, _, proof, _) = prove_impl(params, groups, Sabotage::None, None);
    (ch, proof)
}

pub fn prove_with_timings(params: &HashParams, groups: &[usize]) -> (Vec<RingElem>, Proof, Timings) {
    let (ch, _, proof, tm) = prove_impl(params, groups, Sabotage::None, None);
    (ch, proof, tm)
}

pub fn prove_nizk1(
    params: &HashParams,
    nz: &Nizk1Params,
    groups: &[usize],
    bw: &BlindWitness,
) -> (BlindStatement, Proof) {
    let (_, st, proof, _) = prove_impl(params, groups, Sabotage::None, Some((nz, bw)));
    (st.expect("Phase B"), proof)
}

pub fn prove_nizk1_with_timings(
    params: &HashParams,
    nz: &Nizk1Params,
    groups: &[usize],
    bw: &BlindWitness,
) -> (BlindStatement, Proof, Timings) {
    let (_, st, proof, tm) = prove_impl(params, groups, Sabotage::None, Some((nz, bw)));
    (st.expect("Phase B"), proof, tm)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum Sabotage {
    #[default]
    None,
    ExtraOneHot { step: usize, v: usize },
    ZeroOneHotRow { step: usize },
    FlipMBit { row: usize, coef: usize },
    WrongQuotient { idx: usize },
    WrongRho { idx: usize },
    WrongHpack { idx: usize },
    WrongCxBlinding,
    NonCanonicalDigit { chain: usize, coef: usize },
}

fn build_w_table(
    params: &HashParams,
    d: &Dims,
    wit: &crate::hash::HashWitness,
    nzctx: Option<&Nizk1Ctx>,
    bw: Option<&BlindWitness>,
    h_bits: &[bool],
) -> PackedBits {
    let ng = params.num_groups();
    let mut zw = PackedBits::zeros(d.kw_pad * W_COEF_SLOTS);
    assert_eq!(h_bits.len(), d.h_cells, "h_bits must have length h_cells");
    for (idx, &b) in h_bits.iter().enumerate() {
        if b {
            zw.set(h_slot(d, idx));
        }
    }
    for i in 2..=ng {
        let md = wit.column(i);
        for (dd, digit) in md.iter().enumerate() {
            let (t, c) = (dd / GADGET_LEN, dd % GADGET_LEN);
            for b in 0..DIGIT_BITS {
                put_digit_bit(&mut zw, m_row(params, t, i, c * DIGIT_BITS + b), &digit.c, b);
            }
        }
    }
    if let (Some(ctx), Some(bw)) = (nzctx, bw) {
        let w = &ctx.w;
        for (start, src) in [
            (w.r_pos, &bw.r_pos),
            (w.r_neg, &bw.r_neg),
            (w.rho_r_pos, &bw.rho_r_pos),
            (w.rho_r_neg, &bw.rho_r_neg),
            (w.rho_x_pos, &bw.rho_x_pos),
            (w.rho_x_neg, &bw.rho_x_neg),
        ] {
            for (i, e) in src.iter().enumerate() {
                put_bits(&mut zw, start + i, &e.c);
            }
        }
    }
    zw
}

fn build_h_bits(params: &HashParams, groups: &[usize], sab: Sabotage) -> Vec<bool> {
    let mut out = crate::relation::h_bits(params, groups);
    let tsz = params.table_size();
    match sab {
        Sabotage::ZeroOneHotRow { step } => {
            out[crate::relation::h_index(params, step, groups[step])] = false;
        }
        Sabotage::ExtraOneHot { step, v } => {
            debug_assert!(v < tsz, "ExtraOneHot must poke the real half in order to violate (H2)");
            out[crate::relation::h_index(params, step, v)] = true;
        }
        _ => {}
    }
    out
}

fn h_open(zw: &PackedBits, d: &Dims, pt: &[FqExt]) -> FqExt {
    debug_assert_eq!(pt.len(), d.nv_h);
    let mut tbl = vec![FqExt::ZERO; d.h_cells];
    for (idx, slot) in tbl.iter_mut().enumerate() {
        if h_bit(zw, d, idx) {
            *slot = FqExt::ONE;
        }
    }
    crate::mle::mle_eval(&tbl, pt)
}

pub(crate) fn prove_impl(
    params: &HashParams,
    groups: &[usize],
    sab: Sabotage,
    nzin: Option<(&Nizk1Params, &BlindWitness)>,
) -> (Vec<RingElem>, Option<BlindStatement>, Proof, Timings) {
    let mut tm = Timings::default();
    let mut clk = std::time::Instant::now();

    let (mut ch, mut wit) = eval_h(params, groups);
    debug_assert!(check_witness(params, &ch, groups, &wit));
    if let Sabotage::NonCanonicalDigit { chain, coef } = sab {
        let ml = params.ml();
        let blk = &mut wit.m[0][chain * GADGET_LEN..(chain + 1) * GADGET_LEN];
        let v = blk.iter().rev().fold(0u128, |acc, e| (acc << DIGIT_BITS) | e.c[coef].0 as u128);
        let alt = v + crate::field::Q as u128;
        assert!(alt < 1u128 << M_BIT_ROWS, "coefficient {v} >= 2^{M_BIT_ROWS}−q has no second bit representation");
        for (dg, e) in blk.iter_mut().enumerate() {
            e.c[coef] = Fq(((alt >> (dg * DIGIT_BITS)) & (GADGET_BASE as u128 - 1)) as _);
        }
        let specs = params.spectra_for(groups[0]);
        for r in 0..params.ell {
            let (red, t) = crate::ntt::neg_and_quotient(&specs[r * ml..(r + 1) * ml], &wit.m[0]);
            ch[r] = red;
            wit.t[0][r] = t;
        }
    }

    let mut blind = nzin.map(|(nz, bw)| (nz, bw, blind_statement(params, nz, &ch, bw)));
    if let (Sabotage::WrongCxBlinding, Some((_, _, (st, _)))) = (sab, blind.as_mut()) {
        st.c_x[0].c[0] = st.c_x[0].c[0] + Fq::ONE;
    }
    let ctx_store = blind.as_ref().map(|(nz, _, (st, _))| Nizk1Ctx::new(params, nz, st));
    let nzctx = ctx_store.as_ref();
    let bq = blind.as_ref().map(|(_, _, (_, q))| q);
    let d = dims(params, nzctx);
    let stmt: Vec<RingElem> = match &blind {
        Some((_, _, (st, _))) => st.c_x.clone(),
        None => ch.clone(),
    };
    let mut quotients = compute_quotients(params, &wit, nzctx, bq);
    if let Sabotage::WrongQuotient { idx } = sab {
        quotients[idx][0] = quotients[idx][0] + Fq::ONE;
    }

    let zk_seed: [u8; 32] = match nzin {
        Some((_, bw)) => bw.zk_seed,
        None => {
            let mut h = crate::keccak::Shake128::new();
            h.absorb_bytes(b"voprf-zk-mask-phaseA-NOT-HIDING");
            h.absorb_bytes(&params.crs_digest);
            for &g in groups {
                h.absorb_u64(g as u64);
            }
            let mut sd = [0u8; 32];
            h.squeeze(&mut sd);
            sd
        }
    };
    let mut mrng = crate::rng::CsRng::from_parts("voprf-zk-mask-v1", &[&zk_seed]);
    let shapes = mask_shapes(&d);
    let bases = mask_bases(&d);
    let mut maskers: Vec<sumcheck::Masker> =
        shapes.iter().map(|&(nv, deg)| sumcheck::Masker::new(&mut mrng, nv, deg)).collect();
    let ozk: [FqExt; OPENZK_VALS] = std::array::from_fn(|_| mrng.next_fq4());
    let h_bits = build_h_bits(params, groups, sab);
    if let (Some((nz, bw)), Sabotage::None) = (nzin, sab) {
        assert_eq!(crate::nizk1::pack_h(nz, &h_bits), bw.h_pack, "the H segment of W is out of sync with the message of d_x");
    }

    let mut zw = build_w_table(
        params,
        &d,
        &wit,
        nzctx,
        blind.as_ref().map(|(_, bw, _)| *bw),
        &h_bits,
    );
    write_mask_bits(&d, &mut zw, &maskers, &bases);
    write_openzk_vals(&d, &mut zw, &ozk);
    match sab {
        Sabotage::FlipMBit { row, coef } => zw.flip(row * W_COEF_SLOTS + coef),
        Sabotage::WrongRho { idx } => {
            zw.flip((nzctx.expect("Phase B").w.rho_r_pos + idx) * W_COEF_SLOTS)
        }
        Sabotage::WrongHpack { idx } => {
            zw.flip((nzctx.expect("Phase B").w.h_pack + idx) * W_COEF_SLOTS)
        }
        _ => {}
    }

    let mut t_full = vec![Fq::ZERO; d.c_cells * T_COEF_SLOTS];
    for meta in constraints(params, nzctx) {
        if let Some(idx) = meta.t_index {
            let base = meta.cell * T_COEF_SLOTS;
            assert!(
                quotients[idx].len() <= T_ALPHA_LIMIT,
                "the quotient has {} coefficients, exceeding T_ALPHA_LIMIT = {T_ALPHA_LIMIT} (the last slot is reserved for the ZK mask)",
                quotients[idx].len()
            );
            for (c, &v) in quotients[idx].iter().enumerate() {
                t_full[base + c] = v;
            }
        }
    }
    for cell in 0..d.c_cells {
        t_full[cell * T_COEF_SLOTS + T_ALPHA_LIMIT] = mrng.next_fq();
    }

    tm.mark("witness+tables", &mut clk);
    let c_w = pcs::commit(&zw);
    let c_t = pcs::commit_fq(&t_full);

    tm.mark("commit", &mut clk);

    let mut tr = transcript_init(params, &stmt);
    if let Some(ctx) = nzctx {
        tr.absorb_digest(&ctx.nz.crs_digest);
        tr.absorb_u64(ctx.st.j);
        for e in ctx.st.c_r.iter().chain(&ctx.st.d_x) {
            tr.absorb_fqs(&e.c);
        }
    }
    tr.absorb_u64(c_w.digest);
    tr.absorb_u64(c_t.digest);

    let rho = tr.challenge_fq4();
    for m in maskers.iter_mut() {
        m.set_rho(rho);
    }

    let alpha = tr.challenge_fq4();
    let rows = build_rows(params, &stmt, alpha, nzctx);

    tm.mark("build_rows(a_hat)", &mut clk);

    let mut alpha_pows = Vec::with_capacity(T_COEF_SLOTS);
    let mut p = FqExt::ONE;
    for s in 0..T_COEF_SLOTS {
        alpha_pows.push(if s < N { p } else { FqExt::ZERO });
        p = p * alpha;
    }
    let alpha_pows_w = &alpha_pows[..W_COEF_SLOTS];

    let tau = challenge_vec(&mut tr, d.nv_c);

    let w_rows = crate::nizk1::w_layout(params, nzctx.map(|c| c.nz)).total;
    let w_hat: Vec<FqExt> = (0..w_rows)
        .into_par_iter()
        .map(|k| {
            let base = k * W_COEF_SLOTS;
            let mut acc = LazyExtSum::new();
            for wi in 0..W_COEF_SLOTS / 64 {
                let mut word = zw.word(base / 64 + wi);
                while word != 0 {
                    let b = word.trailing_zeros() as usize;
                    acc.add(&alpha_pows_w[wi * 64 + b]);
                    word &= word - 1;
                }
            }
            acc.finish()
        })
        .collect();

    tm.mark("w_hat", &mut clk);

    let eq_tau = eq_table(&tau);
    let mut eq_ext = vec![FqExt::ZERO; d.u_cells];
    let mut h_ext = vec![FqExt::ZERO; d.u_cells];
    for cell in 0..d.c_cells {
        for v in 0..d.hv {
            eq_ext[cell * d.hv + v] = eq_tau[cell];
            h_ext[cell * d.hv + v] =
                if h_bit(&zw, &d, h_cell(&d, cell, v)) { FqExt::ONE } else { FqExt::ZERO };
        }
    }
    let pow2 = bit_scale();
    let mut u_table = vec![FqExt::ZERO; d.u_cells];
    for row in &rows.lin {
        let base = row.cell * d.hv;
        match u_src(row.kind) {
            Some(src) => {
                let ws: Vec<FqExt> = (0..ml_bits(params))
                    .map(|e| w_hat[m_row_flat(params, src, e)])
                    .collect();
                let wsum: Vec<FqExt> = ws
                    .chunks(DIGIT_BITS)
                    .map(|ch| {
                        ch.iter().enumerate().fold(FqExt::ZERO, |a, (b, &w)| a + pow2[b] * w)
                    })
                    .collect();
                for v in 0..d.tsz {
                    let ab = &rows.a_base[v][row.chain];
                    let mut acc = FqExt::ZERO;
                    for (dd, &w) in wsum.iter().enumerate() {
                        acc = acc + ab[dd] * w;
                    }
                    u_table[base + v] = acc;
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
        .fold(FqExt::ZERO, |acc, ((&e, &h), &u)| acc + e * h * u);

    tm.mark("SC1 tables (U)", &mut clk);

    let r_b = sumcheck::ClaimMask { coef: &ozk[R_B..R_B + R_COEFS] };
    let sigma_u = sumcheck::ClaimMask { coef: &ozk[R_U..R_U + R_COEFS] }.total();
    let s1 = s1_raw + r_b.total();
    let mt0 = maskers[0].total_plain();
    tr.absorb_fq4(s1);
    tr.absorb_fq4(mt0);
    let (sc1, r_u_full, open_h_sc1, u_final, open_r_b) = sumcheck::prove_bilinear_zk(
        eq_ext,
        h_ext,
        u_table,
        sumcheck::LinMask([ozk[SIG_H], ozk[SIG_H + 1]]),
        sigma_u,
        r_b,
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

    let eq_tau_q = eq_table(&tau);
    let mut wq = vec![FqExt::ZERO; t_full.len()];
    for cell in 0..d.c_cells {
        let e = eq_tau_q[cell];
        for (s, &ap) in alpha_pows.iter().take(T_ALPHA_LIMIT).enumerate() {
            wq[cell * T_COEF_SLOTS + s] = e * ap;
        }
    }
    let r_q_mask = sumcheck::ClaimMask { coef: &ozk[R_Q..R_Q + R_COEFS] };
    let q_claim = t_full
        .iter()
        .zip(&wq)
        .fold(FqExt::ZERO, |a, (&t, &w)| a + w * FqExt::from_fq(t))
        + r_q_mask.total();
    let mt1 = maskers[1].total_plain();
    tr.absorb_fq4(q_claim);
    tr.absorb_fq4(mt1);
    let (sc_quotient, r_q_full, open_t, open_r_q) = sumcheck::prove_product2(
        &t_full,
        wq,
        sumcheck::LinMask([ozk[SIG_T], ozk[SIG_T + 1]]),
        r_q_mask,
        Some(&mut maskers[1]),
        &mut tr,
    );
    let (r_q, cq) = r_q_full.split_at(d.nv_t());
    let cq = cq[0];
    let me1 = maskers[1].eval();
    tr.absorb_fq4(open_t);
    tr.absorb_fq4(me1);

    tm.mark("SC_quotient", &mut clk);

    let gamma = tr.challenge_fq4();
    let (r_c_part, r_v) = r_u.split_at(d.nv_c);
    let c_hat = contract_a_hat(&rows, r_v);
    let mut lg = vec![FqExt::ZERO; d.kw_pad];
    for row in &rows.lin {
        let w_m = eq_at_index(&tau, row.cell);
        for &(k, coef) in &row.m_entries {
            lg[k] = lg[k] + w_m * coef;
        }
        if let Some(src) = u_src(row.kind) {
            let w_n = gamma * eq_at_index(r_c_part, row.cell);
            let cr = &c_hat[row.chain];
            for (e, &c) in cr.iter().enumerate() {
                let k = m_row_flat(params, src, e);
                lg[k] = lg[k] + w_n * c;
            }
        }
    }
    tm.mark("lg build", &mut clk);

    let mut scratch: Vec<FqExt> = Vec::new();
    let tau0 = challenge_vec(&mut tr, d.nv_w);
    let lambda = tr.challenge_fq4();
    let mt2 = maskers[2].total_plain();
    tr.absorb_fq4(mt2);
    let xn1_p = alpha.pow(N as u128) + FqExt::ONE;
    let z_u = z_eval(r_u);
    let n_coef: Vec<FqExt> = (0..R_COEFS)
        .map(|k| lambda * (ozk[R_B + k] + gamma * z_u * ozk[R_U + k] + xn1_p * ozk[R_Q + k]))
        .collect();
    let (sc_batched, r_w_full, open_w, open_r_m) = sumcheck::prove_batched_w(
        &zw,
        lg,
        alpha_pows_w,
        &tau0,
        lambda,
        sumcheck::LinMask([ozk[SIG_W], ozk[SIG_W + 1]]),
        sumcheck::ClaimMask { coef: &n_coef },
        &mut scratch,
        Some(&mut maskers[2]),
        &mut tr,
    );
    let (r_w, cb) = r_w_full.split_at(d.nv_w);
    let cb = cb[0];
    let me2 = maskers[2].eval();
    tr.absorb_fq4(open_w);
    tr.absorb_fq4(me2);

    tm.mark("SC_batched (W)", &mut clk);

    let tau3 = challenge_vec(&mut tr, d.nv_i);
    let eq_i = eq_table(&tau3);
    let p_tbl: Vec<FqExt> = (0..d.g_pad)
        .map(|i0| {
            let cnt = (0..d.tsz).filter(|&v| h_bit(&zw, &d, i0 * d.hv + v)).count();
            FqExt::from_u64(cnt as u64)
        })
        .collect();
    let mt3 = maskers[3].total_plain();
    tr.absorb_fq4(mt3);
    let (sc5, r5, _) =
        sumcheck::prove(vec![eq_i, p_tbl], 2, &|v| v[0] * v[1], Some(&mut maskers[3]), &mut tr);
    let open_h_sum = h_open(&zw, &d, &half_point(&d, &r5));

    tm.mark("SC5 (1-hot)", &mut clk);

    let mask_totals = [mt0, mt1, mt2, mt3];
    let mask_evals = [me0, me1, me2, maskers[3].eval()];

    debug_assert!({
        let pts: [&[FqExt]; 4] = [&r_u_full, &r_q_full, &r_w_full, &r5];
        (0..4).all(|i| {
            let (nv, deg) = shapes[i];
            let wt = sumcheck::Masker::weights_total_plain(nv, deg);
            let we = sumcheck::Masker::weights_eval(nv, deg, pts[i]);
            pcs::open_linear_bits(&zw, &mask_weights(&d, bases[i], &wt)) == mask_totals[i]
                && pcs::open_linear_bits(&zw, &mask_weights(&d, bases[i], &we)) == mask_evals[i]
        })
    }, "mask_weights disagree with the mask coefficients written into W (verify_linear is a stub and cannot catch this)");

    debug_assert!({
        let hpt = hpack_point(&d, &r_u[d.nv_u - d.nv_h..]);
        let lin = |i: usize, w: &[FqExt]| {
            pcs::open_linear_bits(&zw, &mask_weights(&d, openzk_base(&d, i), w))
        };
        let wt = |pt: &[FqExt]| sumcheck::LinMask::weights(*pt.last().unwrap());
        pcs::open(&zw, r_w) + z_eval(r_w) * lin(SIG_W, &wt(r_w)) == open_w
            && pcs::open(&zw, &hpt) + z_eval(r_u) * lin(SIG_H, &wt(r_u)) == open_h_sc1
            && pcs::open_fq(&t_full, r_q) + z_eval(r_q) * lin(SIG_T, &wt(r_q)) == open_t
            && lin(R_B, &sumcheck::ClaimMask::weights_eval(R_COEFS - 1, c1)) == open_r_b
            && lin(R_Q, &sumcheck::ClaimMask::weights_eval(R_COEFS - 1, cq)) == open_r_q
            && lin(R_B, &m_weights(gamma, z_u, xn1_p, lambda, cb)) == open_r_m
    }, "the open-ZK mask value disagrees with the homomorphic combination of the commitments");

    let proof = Proof {
        c_w,
        c_t,
        s1,
        q_claim,
        u_final,
        open_w,
        open_t,
        open_h_sc1,
        open_h_sum,
        mask_r_evals: [open_r_b, open_r_q, open_r_m],
        sc1_bilinear: sc1,
        sc_quotient,
        sc_batched,
        sc5_onehot: sc5,
        mask_totals,
        mask_evals,
    };
    let st = blind.map(|(_, _, (st, _))| st);
    (ch, st, proof, tm)
}

fn hpack_point(d: &Dims, pt_h: &[FqExt]) -> Vec<FqExt> {
    debug_assert_eq!(pt_h.len(), d.nv_h);
    let nv_k = d.kw_pad.trailing_zeros() as usize;
    let nv_hp = d.hpack_rows.trailing_zeros() as usize;
    let nv_per = d.hpack_per.trailing_zeros() as usize;
    debug_assert_eq!(nv_hp + nv_per, d.nv_h, "hpack_rows·hpack_per must equal h_cells");
    debug_assert_eq!(d.h_row % d.hpack_rows, 0, "the h_pack segment is misaligned => the row prefix is not constant");
    let prefix = d.h_row >> nv_hp;
    let mut pt = Vec::with_capacity(d.nv_w);
    for b in (0..nv_k - nv_hp).rev() {
        pt.push(FqExt::from_u64(((prefix >> b) & 1) as u64));
    }
    pt.extend_from_slice(&pt_h[..nv_hp]);
    for _ in 0..W_COEF_VARS - nv_per {
        pt.push(FqExt::ZERO);
    }
    pt.extend_from_slice(&pt_h[nv_hp..]);
    debug_assert_eq!(pt.len(), d.nv_w);
    pt
}

pub fn verify(params: &HashParams, ch: &[RingElem], proof: &Proof) -> bool {
    verify_impl(params, ch, None, proof)
}

pub fn verify_nizk1(
    params: &HashParams,
    nz: &Nizk1Params,
    st: &BlindStatement,
    proof: &Proof,
) -> bool {
    if st.c_r.len() != nz.com_r.out_len() || st.d_x.len() != nz.com_x.out_len() {
        return false;
    }
    let ctx = Nizk1Ctx::new(params, nz, st);
    verify_impl(params, &st.c_x.clone(), Some(&ctx), proof)
}

fn verify_impl(
    params: &HashParams,
    ch: &[RingElem],
    nzctx: Option<&Nizk1Ctx>,
    proof: &Proof,
) -> bool {
    if ch.len() != params.ell {
        return false;
    }
    let ng = params.num_groups();
    let d = dims(params, nzctx);
    let nv_t = (d.c_cells * T_COEF_SLOTS).trailing_zeros() as usize;

    if proof.c_w.num_vars != d.nv_w || proof.c_t.num_vars != nv_t {
        return false;
    }

    let mut tr = transcript_init(params, ch);
    if let Some(ctx) = nzctx {
        tr.absorb_digest(&ctx.nz.crs_digest);
        tr.absorb_u64(ctx.st.j);
        for e in ctx.st.c_r.iter().chain(&ctx.st.d_x) {
            tr.absorb_fqs(&e.c);
        }
    }
    tr.absorb_u64(proof.c_w.digest);
    tr.absorb_u64(proof.c_t.digest);
    let rho = tr.challenge_fq4();
    let bases = mask_bases(&d);
    let shapes = mask_shapes(&d);
    let alpha = tr.challenge_fq4();
    let rows = build_rows(params, ch, alpha, nzctx);
    let tau = challenge_vec(&mut tr, d.nv_c);

    tr.absorb_fq4(proof.s1);
    tr.absorb_fq4(proof.mask_totals[0]);
    let claim1 = proof.s1 + rho * proof.mask_totals[0];
    let degs1 = sumcheck::round_degs(d.nv_u, 3, 6, W_ROUND_DEG_3);
    let Some((e1, r_u_full)) =
        sumcheck::verify_degs(claim1, &degs1, &proof.sc1_bilinear, &mut tr)
    else {
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

    tr.absorb_fq4(proof.q_claim);
    tr.absorb_fq4(proof.mask_totals[1]);
    let claim_q = proof.q_claim + rho * proof.mask_totals[1];
    let degs_q = sumcheck::round_degs(nv_t, 2, 4, W_ROUND_DEG_2);
    let Some((e_q, r_q_full)) =
        sumcheck::verify_product2(claim_q, &degs_q, &proof.sc_quotient, &mut tr)
    else {
        return false;
    };
    let (r_q, cq) = r_q_full.split_at(nv_t);
    let cq = cq[0];
    tr.absorb_fq4(proof.open_t);
    tr.absorb_fq4(proof.mask_evals[1]);
    let wq_at =
        eq_eval(&tau, &r_q[..d.nv_c]) * alpha_tensor_eval(&r_q[d.nv_c..], alpha, T_ALPHA_LIMIT);
    if e_q - rho * proof.mask_evals[1]
        != (FqExt::ONE - cq) * wq_at * proof.open_t + ind_eval(r_q) * proof.mask_r_evals[1]
    {
        return false;
    }

    let gamma = tr.challenge_fq4();
    let (r_c_part, r_v) = r_u.split_at(d.nv_c);
    let c_hat = contract_a_hat(&rows, r_v);
    let ppub = ppub_sum(&rows, &tau);
    let pu_rc = pu_sum(&rows, r_c_part, r_v);
    let xn1 = alpha.pow(N as u128) + FqExt::ONE;
    let tau0 = challenge_vec(&mut tr, d.nv_w);
    let lambda = tr.challenge_fq4();
    tr.absorb_fq4(proof.mask_totals[2]);
    let claim2_m =
        (proof.s1 - ppub) + gamma * (proof.u_final - pu_rc) + xn1 * proof.q_claim;
    let claim_b = lambda * claim2_m + rho * proof.mask_totals[2];
    let degs_b = sumcheck::round_degs(d.nv_w, 3, 7, W_ROUND_DEG_3);
    let Some((e_b, r_w_full)) =
        sumcheck::verify_batched_w(claim_b, &degs_b, &proof.sc_batched, &mut tr)
    else {
        return false;
    };
    let (r_w, cb) = r_w_full.split_at(d.nv_w);
    let cb = cb[0];
    tr.absorb_fq4(proof.open_w);
    tr.absorb_fq4(proof.mask_evals[2]);
    let nv_k = d.kw_pad.trailing_zeros() as usize;
    let lg_at = lg_mle_eval(params, &rows, &tau, r_c_part, &c_hat, gamma, &r_w[..nv_k]);
    let f2_final = lambda * lg_at * alpha_tensor_eval(&r_w[nv_k..], alpha, N) * proof.open_w;
    let f3_final = eq_eval(&tau0, r_w) * bit_poly(proof.open_w);
    if e_b - rho * proof.mask_evals[2]
        != (FqExt::ONE - cb) * (f2_final + f3_final) + ind_eval(r_w) * proof.mask_r_evals[2]
    {
        return false;
    }

    let tau3 = challenge_vec(&mut tr, d.nv_i);
    let claim5 = (0..ng).fold(FqExt::ZERO, |a, i0| a + eq_at_index(&tau3, i0));
    tr.absorb_fq4(proof.mask_totals[3]);
    let Some((e5, r_5)) = sumcheck::verify(
        claim5 + rho * proof.mask_totals[3],
        d.nv_i,
        2,
        &proof.sc5_onehot,
        &mut tr,
    ) else {
        return false;
    };
    let two_g = FqExt::from_u64(2).pow(d.tsz.trailing_zeros() as u128);
    if e5 - rho * proof.mask_evals[3] != eq_eval(&tau3, &r_5) * two_g * proof.open_h_sum {
        return false;
    }

    let mask_points: [&[FqExt]; 4] = [&r_u_full, &r_q_full, &r_w_full, &r_5];
    for i in 0..4 {
        let (nv, deg) = shapes[i];
        let wt = sumcheck::Masker::weights_total_plain(nv, deg);
        if !pcs::verify_linear(&proof.c_w, &mask_weights(&d, bases[i], &wt), proof.mask_totals[i]) {
            return false;
        }
        let we = sumcheck::Masker::weights_eval(nv, deg, mask_points[i]);
        if !pcs::verify_linear(&proof.c_w, &mask_weights(&d, bases[i], &we), proof.mask_evals[i]) {
            return false;
        }
    }

    let wt = |pt: &[FqExt]| sumcheck::LinMask::weights(*pt.last().unwrap());
    let sig_w = mask_weights(&d, openzk_base(&d, SIG_W), &wt(r_w));
    let sig_h = mask_weights(&d, openzk_base(&d, SIG_H), &wt(r_u));
    let sig_t = mask_weights(&d, openzk_base(&d, SIG_T), &wt(r_q));
    let hpt = hpack_point(&d, &r_u[d.nv_u - d.nv_h..]);
    let unit = FqExt::ONE;
    pcs::verify_combined(
        &[
            (unit, pcs::Term::BitPoint { c: &proof.c_w, point: r_w }),
            (z_eval(r_w), pcs::Term::Linear { c: &proof.c_w, weights: &sig_w }),
        ],
        proof.open_w,
    )
        && pcs::verify_combined(
            &[
                (unit, pcs::Term::FqPoint { c: &proof.c_t, point: r_q }),
                (z_eval(r_q), pcs::Term::Linear { c: &proof.c_w, weights: &sig_t }),
            ],
            proof.open_t,
        )
        && pcs::verify_combined(
            &[
                (unit, pcs::Term::BitPoint { c: &proof.c_w, point: &hpt }),
                (z_eval(r_u), pcs::Term::Linear { c: &proof.c_w, weights: &sig_h }),
            ],
            proof.open_h_sc1,
        )
        && pcs::verify(
            &proof.c_w,
            &hpack_point(&d, &half_point(&d, &r_5)),
            proof.open_h_sum,
        )
        && pcs::verify_linear(
            &proof.c_w,
            &mask_weights(&d, openzk_base(&d, R_B), &sumcheck::ClaimMask::weights_eval(R_COEFS - 1, c1)),
            proof.mask_r_evals[0],
        )
        && pcs::verify_linear(
            &proof.c_w,
            &mask_weights(&d, openzk_base(&d, R_Q), &sumcheck::ClaimMask::weights_eval(R_COEFS - 1, cq)),
            proof.mask_r_evals[1],
        )
        && pcs::verify_linear(
            &proof.c_w,
            &mask_weights(&d, openzk_base(&d, R_B), &m_weights(gamma, z_eval(r_u), xn1, lambda, cb)),
            proof.mask_r_evals[2],
        )
}

#[allow(dead_code)]
fn _row_debug(r: &LinRow) -> usize {
    r.step_i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::bits_to_groups;
    use crate::relation::num_m_rows;
    use crate::rng::insecure_test_secret;
    use crate::transcript::SimpleRng;

    fn setup(n: usize, g: usize, ell: usize, seed: u64) -> (HashParams, Vec<usize>) {
        let params = HashParams::sample(seed, n, g, ell);
        let mut rng = SimpleRng::new(seed ^ 0x5EED);
        let bits: Vec<bool> = (0..n).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        (params, groups)
    }

    fn setup_nizk1(
        seed: u64,
        n: usize,
        g: usize,
        ell: usize,
    ) -> (HashParams, Nizk1Params, Vec<usize>, BlindWitness) {
        let params = HashParams::sample(seed, n, g, ell);
        let nz = Nizk1Params::sample(seed + 1, &params, 3, 2, 2);
        let (_, groups) = setup(n, g, ell, seed + 2);
        let bw = crate::nizk1::sample_blind(
            crate::nizk1::QueryTicket::insecure_for_tests(insecure_test_secret(seed + 3), seed),
            &params,
            &nz,
            &groups,
        );
        (params, nz, groups, bw)
    }

    #[test]
    fn cheat_extra_onehot() {
        let (params, groups) = setup(8, 2, 1, 77);
        let bad_v = (groups[0] + 1) % params.table_size();
        let (ch, _, proof, _) =
            prove_impl(&params, &groups, Sabotage::ExtraOneHot { step: 0, v: bad_v }, None);
        assert!(!verify(&params, &ch, &proof), "a non-1-hot witness passed verification");
    }

    #[test]
    fn noncanonical_gadget_decomposition_gives_a_second_accepted_statement() {
        let mut params = HashParams::sample(4242, 16, 8, 1);
        let groups = vec![7usize, 200usize];
        params.table[groups[1]][0].c[3] = Fq::new(5);

        let (ch, proof) = prove(&params, &groups);
        assert!(verify(&params, &ch, &proof), "an honest proof must verify");

        let (ch2, _, proof2, _) =
            prove_impl(&params, &groups, Sabotage::NonCanonicalDigit { chain: 0, coef: 3 }, None);
        assert_ne!(ch, ch2, "the non-canonical decomposition did not change c_H -- this test has no discriminating power");
        assert!(
            verify(&params, &ch2, &proof2),
            "(if this starts failing, the canonical constraint has been added -- change this test to assert !verify)"
        );
    }

    #[test]
    fn cheat_zero_onehot_row() {
        for step in [0usize, 2] {
            let (params, groups) = setup(8, 2, 1, 79 + step as u64);
            let (ch, _, proof, _) =
                prove_impl(&params, &groups, Sabotage::ZeroOneHotRow { step }, None);
            assert!(!verify(&params, &ch, &proof), "step {step}: zeroing the whole row was accepted");
        }
    }

    #[test]
    fn cheat_flip_m_bit() {
        for (row, coef) in [(0usize, 0usize), (5, 700), (M_BIT_ROWS, N - 1)] {
            let (params, groups) = setup(8, 2, 1, 83);
            assert!(row < num_m_rows(&params));
            let (ch, _, proof, _) =
                prove_impl(&params, &groups, Sabotage::FlipMBit { row, coef }, None);
            assert!(!verify(&params, &ch, &proof), "flipping W({row},{coef}) was accepted");
        }
    }

    #[test]
    fn cheat_wrong_quotient() {
        let (params, groups) = setup(8, 2, 1, 85);
        let nq = crate::relation::num_quotients(&params, None);
        for idx in [0usize, nq - 1] {
            let (ch, _, proof, _) =
                prove_impl(&params, &groups, Sabotage::WrongQuotient { idx }, None);
            assert!(!verify(&params, &ch, &proof), "a forged quotient {idx} was accepted");
        }
    }

    #[test]
    fn cheat_wrong_rho() {
        let (params, nz, groups, bw) = setup_nizk1(301, 8, 2, 1);
        let (_, st, proof, _) =
            prove_impl(&params, &groups, Sabotage::WrongRho { idx: 0 }, Some((&nz, &bw)));
        assert!(!verify_nizk1(&params, &nz, &st.unwrap(), &proof), "a forged ρ was accepted");
    }

    #[test]
    fn cheat_wrong_hpack() {
        let (params, nz, groups, bw) = setup_nizk1(302, 8, 2, 1);
        let (_, st, proof, _) =
            prove_impl(&params, &groups, Sabotage::WrongHpack { idx: 0 }, Some((&nz, &bw)));
        assert!(!verify_nizk1(&params, &nz, &st.unwrap(), &proof), "a forged h_pack was accepted");
    }

    #[test]
    fn w_slots_exactly_fill_the_ring() {
        assert_eq!(W_COEF_SLOTS, N, "a W row must have exactly N slots (otherwise the G10 attack surface returns)");
        assert_eq!(T_COEF_SLOTS, N, "a T cell must have exactly N slots (masking moved to W)");
    }

    #[test]
    #[should_panic(expected = "W may only hold bits")]
    fn put_bits_rejects_non_bit() {
        let mut zw = PackedBits::zeros(2 * W_COEF_SLOTS);
        let mut poly = vec![Fq::ZERO; W_COEF_SLOTS];
        poly[3] = Fq::new(2);
        put_bits(&mut zw, 0, &poly);
    }

    #[test]
    fn put_bits_accepts_and_writes_bits() {
        let mut zw = PackedBits::zeros(2 * W_COEF_SLOTS);
        let mut poly = vec![Fq::ZERO; W_COEF_SLOTS];
        for i in (0..W_COEF_SLOTS).step_by(3) {
            poly[i] = Fq::ONE;
        }
        put_bits(&mut zw, 1, &poly);
        for i in 0..W_COEF_SLOTS {
            assert_eq!(zw.get(W_COEF_SLOTS + i), i % 3 == 0, "slot {i}");
            assert!(!zw.get(i), "row 0 must not be written (slot {i})");
        }
    }

    #[test]
    fn put_digit_bit_reproduces_the_binary_expansion() {
        use crate::ring::gadget_decompose;
        let mut rng = SimpleRng::new(4242);
        let w = RingElem { c: (0..N).map(|_| rng.next_fq()).collect() };
        let digits = gadget_decompose(&w);
        let mut zw = PackedBits::zeros(M_BIT_ROWS * W_COEF_SLOTS);
        for (dd, digit) in digits.iter().enumerate() {
            for b in 0..DIGIT_BITS {
                put_digit_bit(&mut zw, dd * DIGIT_BITS + b, &digit.c, b);
            }
        }
        for c in 0..N {
            let v = w.c[c].0 as u64;
            for k in 0..M_BIT_ROWS {
                assert_eq!(
                    zw.get(k * W_COEF_SLOTS + c),
                    (v >> k) & 1 == 1,
                    "bit {k} of coefficient {c} is wrong (the two-layer grouping disagrees with the base-2 expansion)"
                );
            }
        }
    }

    #[test]
    fn cheat_wrong_cx_blinding() {
        let (params, nz, groups, bw) = setup_nizk1(303, 8, 2, 1);
        let (_, st, proof, _) =
            prove_impl(&params, &groups, Sabotage::WrongCxBlinding, Some((&nz, &bw)));
        assert!(!verify_nizk1(&params, &nz, &st.unwrap(), &proof), "a forged (N1) was accepted");
    }

    #[test]
    fn tampered_onehot_opening_is_rejected() {
        let (params, groups) = setup(8, 2, 1, 78);
        let (ch, _, mut proof, _) = prove_impl(&params, &groups, Sabotage::None, None);
        proof.open_h_sum = proof.open_h_sum + FqExt::ONE;
        assert!(!verify(&params, &ch, &proof));
    }

    #[test]
    fn cheats_are_rejected_with_two_chains() {
        let (params, groups) = setup(8, 2, 2, 641);
        let bad_v = (groups[0] + 1) % params.table_size();
        for sab in [
            Sabotage::ExtraOneHot { step: 0, v: bad_v },
            Sabotage::ZeroOneHotRow { step: 1 },
            Sabotage::FlipMBit { row: 0, coef: 0 },
            Sabotage::FlipMBit { row: num_m_rows(&params) - 1, coef: N - 1 },
            Sabotage::WrongQuotient { idx: 0 },
            Sabotage::WrongQuotient { idx: crate::relation::num_quotients(&params, None) - 1 },
        ] {
            let (ch, _, proof, _) = prove_impl(&params, &groups, sab, None);
            assert!(!verify(&params, &ch, &proof), "{sab:?} was accepted at ℓ=2");
        }
    }

    #[test]
    fn tampered_quotient_opening_is_rejected() {
        let (params, groups) = setup(8, 2, 1, 631);
        let (ch, _, mut proof, _) = prove_impl(&params, &groups, Sabotage::None, None);
        proof.open_t = proof.open_t + FqExt::ONE;
        assert!(!verify(&params, &ch, &proof));
    }

    #[test]
    fn half_point_identity_matches_row_sums() {
        for &(n, g, ell, seed) in
            &[(8usize, 2usize, 1usize, 601u64), (16, 4, 2, 602), (16, 8, 1, 603), (8, 1, 1, 604)]
        {
            let (params, groups) = setup(n, g, ell, seed);
            let d = dims(&params, None);
            let mut zh = PackedBits::zeros(d.h_cells);
            for (i0, &v) in groups.iter().enumerate() {
                zh.set(i0 * d.hv + v);
            }
            let p_tbl: Vec<FqExt> = (0..d.g_pad)
                .map(|i0| {
                    FqExt::from_u64((0..d.tsz).filter(|&v| zh.get(i0 * d.hv + v)).count() as u64)
                })
                .collect();
            for (i0, &p) in p_tbl.iter().enumerate() {
                assert_eq!(p, FqExt::from_u64((i0 < params.num_groups()) as u64));
            }
            let two_g = FqExt::from_u64(2).pow(g as u128);
            let mut rng = SimpleRng::new(seed ^ 0xFACE);
            for _ in 0..4 {
                let r_i: Vec<FqExt> = (0..d.nv_i).map(|_| rng.next_fq4()).collect();
                assert_eq!(
                    two_g * pcs::open(&zh, &half_point(&d, &r_i)),
                    crate::mle::mle_eval(&p_tbl, &r_i),
                    "n={n} g={g} ell={ell}"
                );
            }
        }
    }

    #[test]
    fn commitment_arity_matches_opening_points() {
        for &(n, g, ell, seed) in &[(8usize, 2usize, 1usize, 611u64), (16, 4, 2, 612)] {
            let (params, groups) = setup(n, g, ell, seed);
            let d = dims(&params, None);
            let (_ch, _, proof, _) = prove_impl(&params, &groups, Sabotage::None, None);
            assert_eq!(proof.c_w.num_vars + 1, proof.sc_batched.rounds.len(), "c_w ↔ r_w");
            assert_eq!(proof.c_t.num_vars + 1, proof.sc_quotient.rounds.len(), "c_t ↔ r_q");
            assert_eq!(proof.c_t.num_vars, (d.c_cells * T_COEF_SLOTS).trailing_zeros() as usize);
            assert_eq!(d.nv_h, proof.sc5_onehot.rounds.len() + g, "nv_h ↔ r_5‖½^g");
            assert_eq!(proof.sc1_bilinear.rounds.len(), d.nv_u + 1, "SC1 ↔ nv_u + w");
            let r5 = vec![FqExt::ZERO; proof.sc5_onehot.rounds.len()];
            for pt in [half_point(&d, &r5), vec![FqExt::ONE; d.nv_h]] {
                assert_eq!(pt.len(), d.nv_h);
                assert_eq!(hpack_point(&d, &pt).len(), proof.c_w.num_vars, "length of hpack_point");
            }
        }
    }

    #[test]
    fn mismatched_commitment_arity_is_rejected() {
        let (params, groups) = setup(8, 2, 1, 621);
        for which in 0..2 {
            let (ch, _, mut proof, _) = prove_impl(&params, &groups, Sabotage::None, None);
            match which {
                0 => proof.c_w.num_vars += 1,
                _ => proof.c_t.num_vars += 1,
            }
            assert!(!verify(&params, &ch, &proof), "commitment {which} was accepted despite a mismatched arity");
        }
    }

    #[test]
    fn lg_mle_eval_matches_naive_u_cube_expansion() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (8, 1, 1), (16, 8, 1)] {
            let (params, _) = setup(n, g, ell, 900 + (n * 10 + g) as u64);
            let mut rng = SimpleRng::new(0xA11CE ^ (n as u64) << 8 ^ g as u64);
            let alpha = rng.next_fq4();
            let ch: Vec<RingElem> = (0..ell)
                .map(|_| RingElem { c: (0..N).map(|_| rng.next_fq()).collect() })
                .collect();
            let rows = build_rows(&params, &ch, alpha, None);
            let d = dims(&params, None);
            let nv_k = d.kw_pad.trailing_zeros() as usize;
            let rv = |rng: &mut SimpleRng, k: usize| -> Vec<FqExt> {
                (0..k).map(|_| rng.next_fq4()).collect()
            };
            let tau = rv(&mut rng, d.nv_c);
            let r_c = rv(&mut rng, d.nv_c);
            let r_v = rv(&mut rng, params.group_bits);
            let gamma = rng.next_fq4();
            let r_k = rv(&mut rng, nv_k);

            let c_hat = contract_a_hat(&rows, &r_v);
            let fast = lg_mle_eval(&params, &rows, &tau, &r_c, &c_hat, gamma, &r_k);

            let r_u: Vec<FqExt> = r_c.iter().chain(&r_v).copied().collect();
            let mut naive = FqExt::ZERO;
            for row in &rows.lin {
                let w_m = eq_at_index(&tau, row.cell);
                for &(k, coef) in &row.m_entries {
                    naive = naive + w_m * coef * eq_at_index(&r_k, k);
                }
                if let Some(src) = u_src(row.kind) {
                    for v in 0..d.tsz {
                        let e = gamma * eq_at_index(&r_u, row.cell * d.tsz + v);
                        let pw = bit_scale();
                        for dd in 0..ml_bits(&params) {
                            #[allow(clippy::modulo_one)]
                            let b = dd % DIGIT_BITS;
                            let ah = pw[b] * rows.a_base[v][row.chain][dd / DIGIT_BITS];
                            naive = naive
                                + e * ah * eq_at_index(&r_k, m_row_flat(&params, src, dd));
                        }
                    }
                }
            }
            assert_eq!(fast, naive, "n={n} g={g} ell={ell}");
        }
    }

    #[test]
    fn every_quotient_leaves_the_last_slot_free() {
        assert!(T_ALPHA_LIMIT >= N - 1 && T_ALPHA_LIMIT <= N, "the α truncation bound is implausible");
        for &(n, g, ell, seed) in &[(8usize, 2usize, 1usize, 610u64), (8, 4, 2, 611), (12, 2, 3, 612)]
        {
            let (params, nz, groups, bw) = setup_nizk1(seed, n, g, ell);
            let (ch, wit) = eval_h(&params, &groups);
            let (st, bq) = blind_statement(&params, &nz, &ch, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            for (tag, qs) in [
                ("Phase A", compute_quotients(&params, &wit, None, None)),
                ("Phase B", compute_quotients(&params, &wit, Some(&ctx), Some(&bq))),
            ] {
                assert!(!qs.is_empty(), "{tag} has no quotients (n={n} g={g} ell={ell})");
                for (i, q) in qs.iter().enumerate() {
                    assert_eq!(
                        q.len(),
                        N - 1,
                        "quotient {i} of {tag} has {} coefficients (must be N−1, otherwise the mask would overwrite data)",
                        q.len()
                    );
                }
            }
        }
    }

    #[test]
    fn alpha_tensor_eval_matches_truncated_table() {
        let mut rng = SimpleRng::new(0xA1F4);
        for nv in [1usize, 3, 5, 11] {
            let alpha = rng.next_fq4();
            let size = 1usize << nv;
            let mut pows = Vec::with_capacity(size);
            let mut p = FqExt::ONE;
            for _ in 0..size {
                pows.push(p);
                p = p * alpha;
            }
            for limit in [0, 1, 3.min(size), size / 2, size * 3 / 4, size - 1, size] {
                let tbl: Vec<FqExt> =
                    (0..size).map(|s| if s < limit { pows[s] } else { FqExt::ZERO }).collect();
                for _ in 0..3 {
                    let r: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
                    assert_eq!(
                        alpha_tensor_eval(&r, alpha, limit),
                        crate::mle::mle_eval(&tbl, &r),
                        "nv={nv} limit={limit}"
                    );
                }
            }
        }
    }

    fn mask_test_setup(seed: u64) -> (HashParams, Nizk1Params, Vec<usize>, BlindWitness) {
        setup_nizk1(seed, 8, 2, 1)
    }

    fn w_table_for(
        params: &HashParams,
        d: &Dims,
        wit: &crate::hash::HashWitness,
        ctx: &Nizk1Ctx,
        bw: &BlindWitness,
        groups: &[usize],
    ) -> PackedBits {
        let hb = build_h_bits(params, groups, Sabotage::None);
        build_w_table(params, d, wit, Some(ctx), Some(bw), &hb)
    }

    #[test]
    fn h_cube_has_no_masking_half() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 8, 1)] {
            let (params, nz, groups, bw) = setup_nizk1(912, n, g, ell);
            let (ch, wit) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &ch, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));
            let mut rng = SimpleRng::new(7);
            let rows = build_rows(&params, &st.c_x, rng.next_fq4(), Some(&ctx));

            assert_eq!(d.hv, d.tsz, "the v dimension of the H cube must be exactly 2^g (the mask half was removed)");
            assert_eq!(crate::relation::hv_len(&params), params.table_size());
            assert_eq!(rows.a_base.len(), d.tsz, "a_base must cover the entire v dimension");
            assert_eq!(d.nv_h, d.nv_i + g, "nv_h is out of sync");
            assert_eq!(d.nv_u, d.nv_c + g, "nv_u is out of sync");
            assert_eq!(d.h_cells, d.g_pad * d.tsz);

            let zw = w_table_for(&params, &d, &wit, &ctx, &bw, &groups);
            let p: Vec<usize> = (0..d.g_pad)
                .map(|i0| (0..d.tsz).filter(|&v| h_bit(&zw, &d, i0 * d.hv + v)).count())
                .collect();
            assert_eq!(
                p,
                (0..d.g_pad).map(|i| (i < params.num_groups()) as usize).collect::<Vec<_>>(),
                "the row sums of an honest witness must be 1_{{i<G}}"
            );
            let r_i: Vec<FqExt> = (0..d.nv_i).map(|_| rng.next_fq4()).collect();
            let pt = half_point(&d, &r_i);
            assert_eq!(pt.len(), d.nv_h, "half_point must have length exactly nv_h");
            let g_bits = d.tsz.trailing_zeros() as usize;
            let half = FqExt::from_u64(2).inv();
            assert!(pt[d.nv_i..].iter().all(|&c| c == half), "the tail of half_point must be all ½");
            assert_eq!(pt[d.nv_i..].len(), g_bits);
        }
    }

    #[test]
    fn witness_level_mask_rows_are_gone() {
        let (params, nz, groups, bw) = mask_test_setup(902);
        let (ch, _) = eval_h(&params, &groups);
        let (st, _) = blind_statement(&params, &nz, &ch, &bw);
        let ctx = Nizk1Ctx::new(&params, &nz, &st);
        assert_eq!(
            ctx.w.total,
            ctx.w.rho_x_neg + nz.com_x.rho_len(),
            "the W layout still has a trailing mask segment"
        );
        let a = crate::nizk1::w_layout(&params, None);
        assert_eq!(a.total, a.h_pack + crate::relation::hpack_rows(&params));
    }

    #[test]
    fn mask_coef_region_is_outside_every_constraint() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 8, 1)] {
            let (params, nz, groups, bw) = setup_nizk1(902, n, g, ell);
            let (ch, _) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &ch, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let mut rng = SimpleRng::new(7);
            let rows = build_rows(&params, &st.c_x, rng.next_fq4(), Some(&ctx));
            let d = dims(&params, Some(&ctx));
            let (lo, hi) = (d.w_mask_coef, d.w_mask_coef + MASK_COEF_ROWS);
            assert!(d.w_mask_coef >= ctx.w.total, "the coefficient region overlaps the witness rows");
            assert!(hi <= d.kw_pad, "the coefficient region overflows kw_pad");
            for row in &rows.lin {
                for &(k, _) in &row.m_entries {
                    assert!(
                        !(lo..hi).contains(&k),
                        "constraint {:?} points into the mask coefficient region {k} (this would break soundness)",
                        row.kind
                    );
                }
            }
        }
    }

    #[test]
    fn mask_embedding_in_w_is_recoverable() {
        use crate::rng::{insecure_test_secret, CsRng};
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 8, 1)] {
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
                shapes.iter().map(|&(nv, deg)| sumcheck::Masker::new(&mut rng, nv, deg)).collect();

            let mut t = PackedBits::zeros(d.kw_pad * W_COEF_SLOTS);
            write_mask_bits(&d, &mut t, &ms, &bases);
            assert!(d.w_mask_coef + MASK_COEF_ROWS <= d.kw_pad, "the mask region overflows kw_pad");
            for k in 0..d.w_mask_coef {
                for c in 0..W_COEF_SLOTS {
                    assert!(!t.get(k * W_COEF_SLOTS + c), "the mask collides with witness row {k}");
                }
            }

            let mut tau_rng = CsRng::from_parts("tau", &[&insecure_test_secret(803)]);
            for i in 0..4 {
                let (nv, deg) = shapes[i];
                let (wt, expect) =
                    (sumcheck::Masker::weights_total_plain(nv, deg), ms[i].total_plain());
                assert_eq!(
                    pcs::open_linear_bits(&t, &mask_weights(&d, bases[i], &wt)),
                    expect,
                    "could not recover the sum of mask {i} (n={n} g={g} ell={ell})"
                );
                let r: Vec<FqExt> = (0..nv).map(|_| tau_rng.next_fq4()).collect();
                for (j, &rj) in r.iter().enumerate() {
                    ms[i].fold(j, rj);
                }
                assert_eq!(
                    pcs::open_linear_bits(
                        &t,
                        &mask_weights(&d, bases[i], &sumcheck::Masker::weights_eval(nv, deg, &r))
                    ),
                    ms[i].eval(),
                    "could not recover g(r) of mask {i}"
                );
            }
        }
    }

    #[test]
    fn h_open_matches_full_pcs_open_at_the_mapped_point() {
        let mut saw_per_full = false;
        let mut saw_per_short = false;
        for &(n, g, ell, seed) in
            &[(8usize, 2usize, 1usize, 401u64), (8, 4, 2, 402), (16, 8, 1, 403), (128, 8, 3, 404)]
        {
            let (params, nz, groups, bw) = setup_nizk1(seed, n, g, ell);
            let (ch, wit) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &ch, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let mut rng = SimpleRng::new(seed ^ 0xBEEF);

            for phase_b in [true, false] {
                let d = dims(&params, if phase_b { Some(&ctx) } else { None });
                let hb = build_h_bits(&params, &groups, Sabotage::None);
                let zw = build_w_table(
                    &params,
                    &d,
                    &wit,
                    if phase_b { Some(&ctx) } else { None },
                    if phase_b { Some(&bw) } else { None },
                    &hb,
                );
                if d.hpack_per == W_COEF_SLOTS {
                    saw_per_full = true;
                } else {
                    saw_per_short = true;
                }
                for (idx, &b) in hb.iter().enumerate() {
                    assert_eq!(h_bit(&zw, &d, idx), b, "bit {idx} of the H segment is at the wrong position");
                }
                for _ in 0..4 {
                    let pt: Vec<FqExt> = (0..d.nv_h).map(|_| rng.next_fq4()).collect();
                    assert_eq!(
                        h_open(&zw, &d, &pt),
                        pcs::open(&zw, &hpack_point(&d, &pt)),
                        "n={n} g={g} ell={ell} phase_b={phase_b}"
                    );
                }
                let r_i: Vec<FqExt> = (0..d.nv_i).map(|_| rng.next_fq4()).collect();
                let pt = half_point(&d, &r_i);
                assert_eq!(
                    h_open(&zw, &d, &pt),
                    pcs::open(&zw, &hpack_point(&d, &pt)),
                    "half_point does not hold under the mapping (n={n} g={g})"
                );
            }
        }
        assert!(saw_per_full && saw_per_short, "not both hpack_per branches were exercised");
    }

    fn sweep_one(n: usize, g: usize, ell: usize, seed: u64) {
        let (params, nz, groups, bw) = setup_nizk1(seed, n, g, ell);
        let ng = params.num_groups();
        let tsz = params.table_size();
        let nq_a = crate::relation::num_quotients(&params, None);
        let tag = format!("n={n} g={g} ell={ell}");

        let (ch, proof) = prove(&params, &groups);
        assert!(verify(&params, &ch, &proof), "Phase A roundtrip failed ({tag})");
        let (st, pb) = prove_nizk1(&params, &nz, &groups, &bw);
        assert!(verify_nizk1(&params, &nz, &st, &pb), "Phase B roundtrip failed ({tag})");

        assert_eq!(
            proof.sc1_bilinear.rounds.len(),
            dims(&params, None).nv_u + 1,
            "SC1 round count ({tag})"
        );
        assert_eq!(mask_shapes(&dims(&params, None)).len(), 4, "masker count ({tag})");
        assert_eq!(pb.mask_totals.len(), 4, "mask_totals count ({tag})");
        assert_eq!(pb.mask_evals.len(), 4, "mask_evals count ({tag})");

        let bad_v = (groups[0] + 1) % tsz;
        let mut sabs_a = vec![
            Sabotage::ExtraOneHot { step: 0, v: bad_v },
            Sabotage::ExtraOneHot { step: ng - 1, v: (groups[ng - 1] + 1) % tsz },
            Sabotage::ZeroOneHotRow { step: 0 },
            Sabotage::ZeroOneHotRow { step: ng - 1 },
            Sabotage::FlipMBit { row: 0, coef: 0 },
            Sabotage::FlipMBit { row: num_m_rows(&params) - 1, coef: N - 1 },
            Sabotage::WrongQuotient { idx: 0 },
            Sabotage::WrongQuotient { idx: nq_a - 1 },
        ];
        if ng > 2 {
            sabs_a.push(Sabotage::ZeroOneHotRow { step: 1 });
        }
        for sab in sabs_a {
            let (ch, _, p, _) = prove_impl(&params, &groups, sab, None);
            assert!(!verify(&params, &ch, &p), "Phase A: {sab:?} was accepted ({tag})");
        }

        for sab in [
            Sabotage::WrongRho { idx: 0 },
            Sabotage::WrongHpack { idx: 0 },
            Sabotage::WrongCxBlinding,
            Sabotage::ExtraOneHot { step: 0, v: bad_v },
            Sabotage::ZeroOneHotRow { step: ng - 1 },
            Sabotage::FlipMBit { row: 0, coef: 0 },
        ] {
            let (_, st, p, _) = prove_impl(&params, &groups, sab, Some((&nz, &bw)));
            assert!(
                !verify_nizk1(&params, &nz, &st.expect("Phase B"), &p),
                "Phase B: {sab:?} was accepted ({tag})"
            );
        }

        const NFIELDS: usize = 15;
        let tweak = |p: &mut Proof, which: usize| {
            let one = FqExt::ONE;
            let f: &mut FqExt = match which {
                0 => &mut p.s1,
                1 => &mut p.q_claim,
                2 => &mut p.u_final,
                3 => &mut p.open_w,
                4 => &mut p.open_t,
                5 => &mut p.open_h_sc1,
                6 => &mut p.open_h_sum,
                7..=10 => &mut p.mask_totals[which - 7],
                _ => &mut p.mask_evals[which - 11],
            };
            *f = *f + one;
        };
        for which in 0..NFIELDS {
            let mut p = prove(&params, &groups).1;
            tweak(&mut p, which);
            assert!(!verify(&params, &ch, &p), "Phase A: field {which} is not bound ({tag})");
            let mut q = prove_nizk1(&params, &nz, &groups, &bw).1;
            tweak(&mut q, which);
            assert!(!verify_nizk1(&params, &nz, &st, &q), "Phase B: field {which} is not bound ({tag})");
        }
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
            (128, 8, crate::params::ELL, 954),
        ] {
            sweep_one(n, g, ell, seed);
        }
    }

    #[test]
    fn h_segment_is_inside_the_cube_covered_by_sc3() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 8, 1), (128, 8, 3)] {
            let (params, nz, groups, _bw) = setup_nizk1(930, n, g, ell);
            let (ch, _) = eval_h(&params, &groups);
            let (st, _) = blind_statement(
                &params,
                &nz,
                &ch,
                &crate::nizk1::sample_blind(
                    crate::nizk1::QueryTicket::insecure_for_tests(insecure_test_secret(931), 0),
                    &params,
                    &nz,
                    &groups,
                ),
            );
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            for d in [dims(&params, Some(&ctx)), dims(&params, None)] {
                assert_eq!(d.hpack_rows * d.hpack_per, d.h_cells, "the shape of the H segment is incomplete");
                assert!(d.h_row + d.hpack_rows <= d.kw_pad, "the H segment overflows kw_pad => SC3 cannot cover it");
                for idx in [0, d.h_cells / 2, d.h_cells - 1] {
                    assert!(h_slot(&d, idx) < d.kw_pad * W_COEF_SLOTS, "cell {idx} of H is outside the cube");
                }
                assert_eq!(d.nv_w, (d.kw_pad * W_COEF_SLOTS).trailing_zeros() as usize);
            }
        }
    }

    #[test]
    fn openzk_mask_values_fit_after_the_libra_maskers() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 8, 1), (128, 8, 3)] {
            let (params, nz, groups, _bw) = setup_nizk1(940, n, g, ell);
            let (ch, _) = eval_h(&params, &groups);
            let (st, _) = blind_statement(
                &params,
                &nz,
                &ch,
                &crate::nizk1::sample_blind(
                    crate::nizk1::QueryTicket::insecure_for_tests(insecure_test_secret(941), 0),
                    &params,
                    &nz,
                    &groups,
                ),
            );
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            for d in [dims(&params, Some(&ctx)), dims(&params, None)] {
                let bases = mask_bases(&d);
                let shapes = mask_shapes(&d);
                let mut acc = 0usize;
                for i in 0..4 {
                    assert_eq!(bases[i], acc, "the start offset of masker {i} is out of sync");
                    acc += EXT_DEG * shapes[i].0 * (shapes[i].1 + 1);
                }
                assert_eq!(bases[4], acc, "the open-ZK region does not follow the maskers");
                let last = openzk_base(&d, OPENZK_VALS - 1) + EXT_DEG - 1;
                assert!(
                    (last + 1) * MASK_FQ_BITS <= MASK_COEF_ROWS * W_COEF_SLOTS,
                    "the mask region cannot hold the 12 open-ZK values (increase MASK_COEF_ROWS)"
                );
                assert_eq!(R_Q, R_B + R_COEFS);
                assert_eq!(R_U, R_Q + R_COEFS);
                let w = mask_weights(&d, openzk_base(&d, R_B), &[FqExt::ONE; 3 * R_COEFS]);
                assert_eq!(w.len(), 3 * R_COEFS * EXT_DEG * MASK_FQ_BITS);
                assert!(w.iter().all(|&(i, _)| i < d.kw_pad * W_COEF_SLOTS));
            }
        }
    }
}
