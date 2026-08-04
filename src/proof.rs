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
    num_m_rows, phase_a_cells, u_src, CKind, LinRow, Nizk1Ctx, Rows,
};
use crate::ring::{
    RingElem, DIGIT_BITS, GADGET_BASE, GADGET_LEN, M_BIT_ROWS, N, W_RANGE_BASE,
};
use crate::sumcheck::{self, SumcheckProof};
use crate::transcript::Transcript;
use rayon::prelude::*;

pub const W_COEF_VARS: usize = N.trailing_zeros() as usize;
pub const W_COEF_SLOTS: usize = 1 << W_COEF_VARS;

pub const T_COEF_VARS: usize = W_COEF_VARS + 1;
pub const T_COEF_SLOTS: usize = 1 << T_COEF_VARS;

const _: () = assert!(W_COEF_SLOTS == N, "the W slot count must be exactly N");
const _: () = assert!(T_COEF_SLOTS > N, "T needs a free region of >= N slots for the ZK mask");

pub struct Proof {
    pub c_w: pcs::Commitment,
    pub c_h: pcs::Commitment,
    pub c_t: pcs::Commitment,
    pub s1: FqExt,
    pub u_final: FqExt,
    pub q_claim: FqExt,
    pub sc1_bilinear: SumcheckProof,
    pub sc_quotient: SumcheckProof,
    pub sc_batched: SumcheckProof,
    pub sc4_bit: SumcheckProof,
    pub sc5_onehot: SumcheckProof,
    pub open_w: FqExt,
    pub open_t: FqExt,
    pub open_h_sc1: FqExt,
    pub open_h_bit: FqExt,
    pub open_h_sum: FqExt,
    pub mask_totals: [FqExt; 5],
    pub mask_evals: [FqExt; 5],
    pub open_h_eq: Option<FqExt>,
    pub open_hpack: Option<FqExt>,
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
    for c in [&p.c_w, &p.c_h, &p.c_t] {
        mix(d, c.digest);
        mix(d, c.num_vars as u64);
    }
    for v in
        [p.s1, p.u_final, p.q_claim, p.open_w, p.open_t, p.open_h_sc1, p.open_h_bit, p.open_h_sum]
    {
        mix_fq4(d, v);
    }
    for v in p.mask_totals.iter().chain(&p.mask_evals) {
        mix_fq4(d, *v);
    }
    for v in [p.open_h_eq, p.open_hpack].into_iter().flatten() {
        mix_fq4(d, v);
    }
    for sc in [&p.sc1_bilinear, &p.sc_quotient, &p.sc_batched, &p.sc4_bit, &p.sc5_onehot] {
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

fn mask_slot(d: &Dims, flat: usize) -> usize {
    let free = T_COEF_SLOTS - N;
    let cell = flat / free;
    assert!(
        cell < d.c_cells,
        "the ZK mask coefficients do not fit the free region of t_full: need > {} cells, only {} available",
        flat,
        d.c_cells * free
    );
    cell * T_COEF_SLOTS + N + (flat % free)
}

fn mask_weights(d: &Dims, base: usize, w: &[FqExt]) -> Vec<(usize, FqExt)> {
    let mut out = Vec::with_capacity(w.len() * EXT_DEG);
    for (j, &wj) in w.iter().enumerate() {
        for a in 0..EXT_DEG {
            let mut basis = FqExt::ZERO;
            basis.0[a] = Fq::ONE;
            out.push((mask_slot(d, base + EXT_DEG * j + a), basis * wj));
        }
    }
    out
}

fn mask_shapes(d: &Dims) -> [(usize, usize); 5] {
    [(d.nv_u, 3), (d.nv_t(), 2), (d.nv_w, 3), (d.nv_h, 2), (d.nv_i, 2)]
}

fn mask_bases(d: &Dims) -> [usize; 5] {
    let sh = mask_shapes(d);
    let mut out = [0usize; 5];
    let mut acc = 0;
    for i in 0..5 {
        out[i] = acc;
        acc += EXT_DEG * sh[i].0 * (sh[i].1 + 1);
    }
    out
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
    for (v, &e) in eqv.iter().enumerate() {
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
        let inner =
            eqv.iter().enumerate().fold(FqExt::ZERO, |a, (v, &e)| a + e * rows.u_pub(row, v));
        acc = acc + eq_at_index(r_c, row.cell) * inner;
    }
    acc
}

struct Dims {
    g_pad: usize,
    tsz: usize,
    #[allow(dead_code)]
    nv_j: usize,
    nv_c: usize,
    nv_u: usize,
    nv_i: usize,
    nv_h: usize,
    kw_pad: usize,
    nv_w: usize,
    c_cells: usize,
    u_cells: usize,
    h_cells: usize,
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
    let w_rows = nz.map_or(num_m_rows(params), |c| c.w.total);
    let kw_pad = w_rows.next_power_of_two();
    let nv_j = ell_pad.trailing_zeros() as usize;
    let nv_i = g_pad.trailing_zeros() as usize;
    let gb = params.group_bits;
    let c_cells = (phase_a_cells(params) + nz.map_or(0, |c| c.num_rows())).next_power_of_two();
    let nv_c = c_cells.trailing_zeros() as usize;
    Dims {
        g_pad,
        tsz,
        nv_j,
        nv_c,
        nv_u: nv_c + gb,
        nv_i,
        nv_h: nv_i + gb,
        kw_pad,
        nv_w: (kw_pad * W_COEF_SLOTS).trailing_zeros() as usize,
        c_cells,
        u_cells: c_cells * tsz,
        h_cells: g_pad * tsz,
    }
}

fn put_bits(zw: &mut PackedBits, k: usize, poly: &[Fq]) {
    let acc = poly.iter().fold(0u64, |a, &v| a | v.0 as u64);
    assert!(acc < W_RANGE_BASE, "W may only hold bits, got a row containing {acc} (did the digit skip its base-2 decomposition? see q64.md 4b)");
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
    (cell % d.g_pad) * d.tsz + v
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
    zk_mask: &[RingElem],
) -> PackedBits {
    let ng = params.num_groups();
    let mut zw = PackedBits::zeros(d.kw_pad * W_COEF_SLOTS);
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
            (w.h_pack, &bw.h_pack),
            (w.rho_x_pos, &bw.rho_x_pos),
            (w.rho_x_neg, &bw.rho_x_neg),
        ] {
            for (i, e) in src.iter().enumerate() {
                put_bits(&mut zw, start + i, &e.c);
            }
        }
        debug_assert_eq!(zk_mask.len(), crate::nizk1::ZK_MASK_ROWS);
        for (i, e) in zk_mask.iter().enumerate() {
            put_bits(&mut zw, w.mask + i, &e.c);
        }
    }
    zw
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
        assert!(alt < 1u128 << M_BIT_ROWS, "coefficient value {v} >= 2^{M_BIT_ROWS}-q, so there is no second bit representation");
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
    let zk_mask = crate::nizk1::zk_mask_rows(&mut mrng);

    let mut zw =
        build_w_table(params, &d, &wit, nzctx, blind.as_ref().map(|(_, bw, _)| *bw), &zk_mask);
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
            for (c, &v) in quotients[idx].iter().enumerate() {
                t_full[base + c] = v;
            }
        }
    }

    for (i, m) in maskers.iter().enumerate() {
        for (j, c) in m.coeffs_flat().enumerate() {
            for a in 0..EXT_DEG {
                t_full[mask_slot(&d, bases[i] + EXT_DEG * j + a)] = c.0[a];
            }
        }
    }

    let mut zh = PackedBits::zeros(d.h_cells);
    for (i0, &v) in groups.iter().enumerate() {
        if sab == (Sabotage::ZeroOneHotRow { step: i0 }) {
            continue;
        }
        zh.set(i0 * d.tsz + v);
    }
    if let Sabotage::ExtraOneHot { step, v } = sab {
        zh.set(step * d.tsz + v);
    }

    tm.mark("witness+tables", &mut clk);
    let c_w = pcs::commit(&zw);
    let c_h = pcs::commit(&zh);
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
    tr.absorb_u64(c_h.digest);
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

    let w_rows = nzctx.map_or(num_m_rows(params), |c| c.w.total);
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
        for v in 0..d.tsz {
            eq_ext[cell * d.tsz + v] = eq_tau[cell];
            h_ext[cell * d.tsz + v] =
                if zh.get(h_cell(&d, cell, v)) { FqExt::ONE } else { FqExt::ZERO };
        }
    }
    let pow2 = bit_scale();
    let mut u_table = vec![FqExt::ZERO; d.u_cells];
    for row in &rows.lin {
        let base = row.cell * d.tsz;
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

    let s1 = eq_ext
        .iter()
        .zip(&h_ext)
        .zip(&u_table)
        .fold(FqExt::ZERO, |acc, ((&e, &h), &u)| acc + e * h * u);
    tr.absorb_fq4(s1);

    tm.mark("SC1 tables (U)", &mut clk);

    let mt0 = maskers[0].total_plain();
    tr.absorb_fq4(mt0);
    let (sc1, r_u, finals1) = sumcheck::prove(
        vec![eq_ext, h_ext, u_table],
        3,
        &|v| v[0] * v[1] * v[2],
        Some(&mut maskers[0]),
        &mut tr,
    );
    let open_h_sc1 = finals1[1];
    let u_final = finals1[2];
    tr.absorb_fq4(u_final);

    tm.mark("SC1", &mut clk);

    let eq_tau_q = eq_table(&tau);
    let mut wq = vec![FqExt::ZERO; t_full.len()];
    for cell in 0..d.c_cells {
        let e = eq_tau_q[cell];
        for (s, &ap) in alpha_pows.iter().enumerate() {
            wq[cell * T_COEF_SLOTS + s] = e * ap;
        }
    }
    let q_claim = t_full
        .iter()
        .zip(&wq)
        .fold(FqExt::ZERO, |a, (&t, &w)| a + w * FqExt::from_fq(t));
    tr.absorb_fq4(q_claim);
    let mt1 = maskers[1].total_plain();
    tr.absorb_fq4(mt1);
    let (sc_quotient, r_q) =
        sumcheck::prove_product2(&t_full, wq, Some(&mut maskers[1]), &mut tr);
    let open_t = pcs::open_fq(&t_full, &r_q);

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
    let (sc_batched, _r_w, open_w) = sumcheck::prove_batched_w(
        &zw,
        lg,
        alpha_pows_w,
        &tau0,
        lambda,
        &mut scratch,
        Some(&mut maskers[2]),
        &mut tr,
    );

    tm.mark("SC_batched (W)", &mut clk);

    let tau2 = challenge_vec(&mut tr, d.nv_h);
    let mt3 = maskers[3].total_eq(&tau2);
    tr.absorb_fq4(mt3);
    let (sc4, _r_2, open_h_bit) =
        sumcheck::prove_eq_bitcheck(&tau2, &zh, &mut scratch, Some(&mut maskers[3]), &mut tr);

    tm.mark("SC4 (H bit)", &mut clk);

    let tau3 = challenge_vec(&mut tr, d.nv_i);
    let eq_i = eq_table(&tau3);
    let p_tbl: Vec<FqExt> = (0..d.g_pad)
        .map(|i0| {
            let cnt = (0..d.tsz).filter(|&v| zh.get(i0 * d.tsz + v)).count();
            FqExt::from_u64(cnt as u64)
        })
        .collect();
    let mt4 = maskers[4].total_plain();
    tr.absorb_fq4(mt4);
    let (sc5, r5, _) =
        sumcheck::prove(vec![eq_i, p_tbl], 2, &|v| v[0] * v[1], Some(&mut maskers[4]), &mut tr);
    let open_h_sum = pcs::open(&zh, &half_point(&d, &r5));

    tm.mark("SC5 (1-hot)", &mut clk);

    let (open_h_eq, open_hpack) = match nzctx {
        Some(ctx) => {
            let r6 = challenge_vec(&mut tr, d.nv_h);
            (Some(pcs::open(&zh, &r6)), Some(hpack_open(&zw, &d, ctx, &r6)))
        }
        None => (None, None),
    };

    tm.mark("PhaseB h-link", &mut clk);

    let mask_totals = [mt0, mt1, mt2, mt3, mt4];
    let mask_evals: [FqExt; 5] = std::array::from_fn(|i| maskers[i].eval());

    let proof = Proof {
        c_w,
        c_h,
        c_t,
        s1,
        u_final,
        q_claim,
        sc1_bilinear: sc1,
        sc_quotient,
        sc_batched,
        sc4_bit: sc4,
        sc5_onehot: sc5,
        open_w,
        open_t,
        open_h_sc1,
        open_h_bit,
        open_h_sum,
        mask_totals,
        mask_evals,
        open_h_eq,
        open_hpack,
    };
    let st = blind.map(|(_, _, (st, _))| st);
    (ch, st, proof, tm)
}

fn hpack_open(zw: &PackedBits, d: &Dims, ctx: &Nizk1Ctx, r6: &[FqExt]) -> FqExt {
    let hp = ctx.nz.hpack_len.next_power_of_two();
    let per = d.h_cells / hp;
    let mut tbl = vec![FqExt::ZERO; d.h_cells];
    for r in 0..hp {
        let base = (ctx.w.h_pack + r) * W_COEF_SLOTS;
        for c in 0..per {
            if zw.get(base + c) {
                tbl[r * per + c] = FqExt::ONE;
            }
        }
    }
    crate::mle::mle_eval(&tbl, r6)
}

fn hpack_point(d: &Dims, ctx: &Nizk1Ctx, r6: &[FqExt]) -> Vec<FqExt> {
    let nv_k = d.kw_pad.trailing_zeros() as usize;
    let hp = ctx.nz.hpack_len.next_power_of_two();
    let nv_hp = hp.trailing_zeros() as usize;
    let per = d.h_cells / hp;
    let nv_per = per.trailing_zeros() as usize;
    let prefix = ctx.w.h_pack >> nv_hp;
    let mut pt = Vec::with_capacity(d.nv_w);
    for b in (0..nv_k - nv_hp).rev() {
        pt.push(FqExt::from_u64(((prefix >> b) & 1) as u64));
    }
    pt.extend_from_slice(&r6[..nv_hp]);
    for _ in 0..W_COEF_VARS - nv_per {
        pt.push(FqExt::ZERO);
    }
    pt.extend_from_slice(&r6[nv_hp..]);
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
    if nzctx.is_some() != proof.open_h_eq.is_some() {
        return false;
    }
    let ng = params.num_groups();
    let d = dims(params, nzctx);
    let nv_t = (d.c_cells * T_COEF_SLOTS).trailing_zeros() as usize;

    if proof.c_w.num_vars != d.nv_w
        || proof.c_h.num_vars != d.nv_h
        || proof.c_t.num_vars != nv_t
    {
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
    tr.absorb_u64(proof.c_h.digest);
    tr.absorb_u64(proof.c_t.digest);
    let rho = tr.challenge_fq4();
    let bases = mask_bases(&d);
    let shapes = mask_shapes(&d);
    let alpha = tr.challenge_fq4();
    let rows = build_rows(params, ch, alpha, nzctx);
    let tau = challenge_vec(&mut tr, d.nv_c);
    tr.absorb_fq4(proof.s1);

    tr.absorb_fq4(proof.mask_totals[0]);
    let Some((e1, r_u)) = sumcheck::verify(
        proof.s1 + rho * proof.mask_totals[0],
        d.nv_u,
        3,
        &proof.sc1_bilinear,
        &mut tr,
    ) else {
        return false;
    };
    if e1 - rho * proof.mask_evals[0]
        != eq_eval(&tau, &r_u[..d.nv_c]) * proof.open_h_sc1 * proof.u_final
    {
        return false;
    }
    tr.absorb_fq4(proof.u_final);

    tr.absorb_fq4(proof.q_claim);
    tr.absorb_fq4(proof.mask_totals[1]);
    let Some((e_q, r_q)) = sumcheck::verify_product2(
        proof.q_claim + rho * proof.mask_totals[1],
        nv_t,
        &proof.sc_quotient,
        &mut tr,
    ) else {
        return false;
    };
    let wq_at = eq_eval(&tau, &r_q[..d.nv_c]) * alpha_tensor_eval(&r_q[d.nv_c..], alpha, N);
    if e_q - rho * proof.mask_evals[1] != wq_at * proof.open_t {
        return false;
    }

    let gamma = tr.challenge_fq4();
    let (r_c_part, r_v) = r_u.split_at(d.nv_c);
    let c_hat = contract_a_hat(&rows, r_v);
    let ppub = ppub_sum(&rows, &tau);
    let pu_rc = pu_sum(&rows, r_c_part, r_v);
    let claim2 = (proof.s1 - ppub) + gamma * (proof.u_final - pu_rc);
    let xn1 = alpha.pow(N as u128) + FqExt::ONE;
    let claim2_m = claim2 + xn1 * proof.q_claim;
    let tau0 = challenge_vec(&mut tr, d.nv_w);
    let lambda = tr.challenge_fq4();
    tr.absorb_fq4(proof.mask_totals[2]);
    let Some((e_b, r_w)) = sumcheck::verify_batched_w(
        lambda * claim2_m + rho * proof.mask_totals[2],
        d.nv_w,
        &proof.sc_batched,
        &mut tr,
    ) else {
        return false;
    };
    let nv_k = d.kw_pad.trailing_zeros() as usize;
    let lg_at = lg_mle_eval(params, &rows, &tau, r_c_part, &c_hat, gamma, &r_w[..nv_k]);
    let f2_final = proof.open_w * lg_at * alpha_tensor_eval(&r_w[nv_k..], alpha, N);
    let f3_final = eq_eval(&tau0, &r_w) * bit_poly(proof.open_w);
    if e_b - rho * proof.mask_evals[2] != lambda * f2_final + f3_final {
        return false;
    }

    let tau2 = challenge_vec(&mut tr, d.nv_h);
    tr.absorb_fq4(proof.mask_totals[3]);
    let Some((e4, r_2)) = sumcheck::verify_eq_bitcheck(
        rho * proof.mask_totals[3],
        &tau2,
        &proof.sc4_bit,
        &mut tr,
    ) else {
        return false;
    };
    if e4 != eq_eval(&tau2, &r_2) * (bit_poly(proof.open_h_bit) + rho * proof.mask_evals[3]) {
        return false;
    }

    let tau3 = challenge_vec(&mut tr, d.nv_i);
    let claim5 = (0..ng).fold(FqExt::ZERO, |a, i0| a + eq_at_index(&tau3, i0));
    tr.absorb_fq4(proof.mask_totals[4]);
    let Some((e5, r_5)) = sumcheck::verify(
        claim5 + rho * proof.mask_totals[4],
        d.nv_i,
        2,
        &proof.sc5_onehot,
        &mut tr,
    ) else {
        return false;
    };
    let two_g = FqExt::from_u64(2).pow(d.tsz.trailing_zeros() as u128);
    if e5 - rho * proof.mask_evals[4] != eq_eval(&tau3, &r_5) * two_g * proof.open_h_sum {
        return false;
    }

    if let Some(ctx) = nzctx {
        let r6 = challenge_vec(&mut tr, d.nv_h);
        let (Some(a), Some(b)) = (proof.open_h_eq, proof.open_hpack) else {
            return false;
        };
        if a != b {
            return false;
        }
        let pt = hpack_point(&d, ctx, &r6);
        if !pcs::verify(&proof.c_h, &r6, a) || !pcs::verify(&proof.c_w, &pt, b) {
            return false;
        }
    }

    let mask_points: [&[FqExt]; 5] = [&r_u, &r_q, &r_w, &r_2, &r_5];
    for i in 0..5 {
        let (nv, deg) = shapes[i];
        let wt = if i == 3 {
            sumcheck::Masker::weights_total_eq(nv, deg, &tau2)
        } else {
            sumcheck::Masker::weights_total_plain(nv, deg)
        };
        if !pcs::verify_linear(
            &proof.c_t,
            &mask_weights(&d, bases[i], &wt),
            proof.mask_totals[i],
        ) {
            return false;
        }
        let we = sumcheck::Masker::weights_eval(nv, deg, mask_points[i]);
        if !pcs::verify_linear(
            &proof.c_t,
            &mask_weights(&d, bases[i], &we),
            proof.mask_evals[i],
        ) {
            return false;
        }
    }

    pcs::verify(&proof.c_w, &r_w, proof.open_w)
        && pcs::verify_fq(&proof.c_t, &r_q, proof.open_t)
        && pcs::verify(&proof.c_h, &r_u[d.nv_u - d.nv_h..], proof.open_h_sc1)
        && pcs::verify(&proof.c_h, &r_2, proof.open_h_bit)
        && pcs::verify(&proof.c_h, &half_point(&d, &r_5), proof.open_h_sum)
}

#[allow(dead_code)]
fn _row_debug(r: &LinRow) -> usize {
    r.step_i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::bits_to_groups;
    use crate::rng::CsRng;
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
        let h_cells = g_pad(&params) * params.table_size();
        let mut hb = vec![false; h_cells];
        for (i, &v) in groups.iter().enumerate() {
            hb[i * params.table_size() + v] = true;
        }
        let bw = crate::nizk1::sample_blind(&insecure_test_secret(seed + 3), seed, &nz, &hb);
        (params, nz, groups, bw)
    }

    #[test]
    fn cheat_extra_onehot() {
        let (params, groups) = setup(8, 2, 1, 77);
        let bad_v = (groups[0] + 1) % params.table_size();
        let (ch, _, proof, _) =
            prove_impl(&params, &groups, Sabotage::ExtraOneHot { step: 0, v: bad_v }, None);
        assert!(!verify(&params, &ch, &proof), "a non-1-hot witness unexpectedly passed verification");
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
            assert!(!verify(&params, &ch, &proof), "step {step} with a fully zeroed row unexpectedly passed");
        }
    }

    #[test]
    fn cheat_flip_m_bit() {
        for (row, coef) in [(0usize, 0usize), (5, 700), (M_BIT_ROWS, N - 1)] {
            let (params, groups) = setup(8, 2, 1, 83);
            assert!(row < num_m_rows(&params));
            let (ch, _, proof, _) =
                prove_impl(&params, &groups, Sabotage::FlipMBit { row, coef }, None);
            assert!(!verify(&params, &ch, &proof), "flipping W({row},{coef}) unexpectedly passed");
        }
    }

    #[test]
    fn cheat_wrong_quotient() {
        let (params, groups) = setup(8, 2, 1, 85);
        let nq = crate::relation::num_quotients(&params, None);
        for idx in [0usize, nq - 1] {
            let (ch, _, proof, _) =
                prove_impl(&params, &groups, Sabotage::WrongQuotient { idx }, None);
            assert!(!verify(&params, &ch, &proof), "a forged quotient {idx} unexpectedly passed");
        }
    }

    #[test]
    fn cheat_wrong_rho() {
        let (params, nz, groups, bw) = setup_nizk1(301, 8, 2, 1);
        let (_, st, proof, _) =
            prove_impl(&params, &groups, Sabotage::WrongRho { idx: 0 }, Some((&nz, &bw)));
        assert!(!verify_nizk1(&params, &nz, &st.unwrap(), &proof), "a forged rho unexpectedly passed");
    }

    #[test]
    fn cheat_wrong_hpack() {
        let (params, nz, groups, bw) = setup_nizk1(302, 8, 2, 1);
        let (_, st, proof, _) =
            prove_impl(&params, &groups, Sabotage::WrongHpack { idx: 0 }, Some((&nz, &bw)));
        assert!(!verify_nizk1(&params, &nz, &st.unwrap(), &proof), "a forged h_pack unexpectedly passed");
    }

    #[test]
    fn w_slots_exactly_fill_the_ring() {
        assert_eq!(W_COEF_SLOTS, N, "the slot count of one W row must be exactly N (otherwise the G10 attack surface returns)");
        assert!(T_COEF_SLOTS > N);
        let alpha = FqExt::from_u64(12345);
        let mut ap = FqExt::ONE;
        for s in 0..T_COEF_SLOTS {
            let w = if s < N { ap } else { FqExt::ZERO };
            if s >= N {
                assert_eq!(w, FqExt::ZERO, "the high slots of T unexpectedly carry a non-zero alpha weight");
            }
            ap = ap * alpha;
        }
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
        assert!(!verify_nizk1(&params, &nz, &st.unwrap(), &proof), "a forged (N1) unexpectedly passed");
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
            assert!(!verify(&params, &ch, &proof), "{sab:?} unexpectedly passed with ell=2");
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
                zh.set(i0 * d.tsz + v);
            }
            let p_tbl: Vec<FqExt> = (0..d.g_pad)
                .map(|i0| {
                    FqExt::from_u64((0..d.tsz).filter(|&v| zh.get(i0 * d.tsz + v)).count() as u64)
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
            assert_eq!(proof.c_w.num_vars, proof.sc_batched.rounds.len(), "c_w ↔ r_w");
            assert_eq!(proof.c_h.num_vars, proof.sc4_bit.rounds.len(), "c_h ↔ r_2");
            assert_eq!(proof.c_h.num_vars, proof.sc5_onehot.rounds.len() + g, "c_h ↔ r_5‖½");
            assert_eq!(proof.c_t.num_vars, proof.sc_quotient.rounds.len(), "c_t ↔ r_q");
            assert_eq!(proof.c_t.num_vars, (d.c_cells * T_COEF_SLOTS).trailing_zeros() as usize);
        }
    }

    #[test]
    fn mismatched_commitment_arity_is_rejected() {
        let (params, groups) = setup(8, 2, 1, 621);
        for which in 0..3 {
            let (ch, _, mut proof, _) = prove_impl(&params, &groups, Sabotage::None, None);
            match which {
                0 => proof.c_w.num_vars += 1,
                1 => proof.c_h.num_vars -= 1,
                _ => proof.c_t.num_vars += 1,
            }
            assert!(!verify(&params, &ch, &proof), "commitment {which} with a mismatched arity unexpectedly passed");
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

    #[test]
    fn zk_mask_rows_are_written_and_random() {
        let (params, nz, groups, bw) = mask_test_setup(900);
        let (ch, wit) = eval_h(&params, &groups);
        let (st, _) = blind_statement(&params, &nz, &ch, &bw);
        let ctx = Nizk1Ctx::new(&params, &nz, &st);
        let d = dims(&params, Some(&ctx));
        let mut rng = CsRng::from_parts("voprf-zk-mask-v1", &[&bw.zk_seed]);
        let zk = crate::nizk1::zk_mask_rows(&mut rng);
        let zw = build_w_table(&params, &d, &wit, Some(&ctx), Some(&bw), &zk);

        for row in 0..crate::nizk1::ZK_MASK_ROWS {
            let base = (ctx.w.mask + row) * W_COEF_SLOTS;
            let ones = (0..N).filter(|&c| zw.get(base + c)).count();
            assert!(
                ones > N * 40 / 100 && ones < N * 60 / 100,
                "mask row {row} does not look like random bits: {ones}/{N} (0 means it was never written)"
            );
            assert_eq!(W_COEF_SLOTS, N);
        }
    }

    #[test]
    fn zk_mask_rows_change_open_w_at_a_fixed_point() {
        let (params, nz, groups, bw) = mask_test_setup(901);
        let (ch, wit) = eval_h(&params, &groups);
        let (st, _) = blind_statement(&params, &nz, &ch, &bw);
        let ctx = Nizk1Ctx::new(&params, &nz, &st);
        let d = dims(&params, Some(&ctx));

        let mk = |tag: &[u8; 32]| {
            let mut r = CsRng::from_parts("voprf-zk-mask-v1", &[tag]);
            let zk = crate::nizk1::zk_mask_rows(&mut r);
            build_w_table(&params, &d, &wit, Some(&ctx), Some(&bw), &zk)
        };
        let zw1 = mk(&crate::rng::insecure_test_secret(1));
        let zw2 = mk(&crate::rng::insecure_test_secret(2));

        let mut rng = SimpleRng::new(0x5A5A);
        for _ in 0..4 {
            let r: Vec<FqExt> = (0..d.nv_w).map(|_| rng.next_fq4()).collect();
            assert_ne!(
                pcs::open(&zw1, &r),
                pcs::open(&zw2, &r),
                "the mask rows did not affect W~(r) -- layer (ii) had no effect"
            );
        }
    }

    #[test]
    fn zk_mask_rows_are_outside_every_constraint() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (8, 4, 2), (16, 8, 1)] {
            let (params, nz, groups, bw) = setup_nizk1(902, n, g, ell);
            let (ch, _) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &ch, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let mut rng = SimpleRng::new(7);
            let rows = build_rows(&params, &st.c_x, rng.next_fq4(), Some(&ctx));
            let lo = ctx.w.mask;
            let hi = ctx.w.total;
            assert!(hi > lo, "the mask segment has length 0");
            for row in &rows.lin {
                for &(k, _) in &row.m_entries {
                    assert!(
                        !(lo..hi).contains(&k),
                        "constraint {:?} points at mask row {k} (this would break soundness)",
                        row.kind
                    );
                }
            }
        }
    }

    #[test]
    fn mask_embedding_in_t_full_is_recoverable() {
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

            let mut t = vec![Fq::ZERO; d.c_cells * T_COEF_SLOTS];
            for (i, m) in ms.iter().enumerate() {
                for (j, c) in m.coeffs_flat().enumerate() {
                    for a in 0..EXT_DEG {
                        t[mask_slot(&d, bases[i] + EXT_DEG * j + a)] = c.0[a];
                    }
                }
            }
            for cell in 0..d.c_cells {
                for slot in 0..N {
                    assert_eq!(t[cell * T_COEF_SLOTS + slot], Fq::ZERO, "the mask collided with the quotient region");
                }
            }

            let mut tau_rng = CsRng::from_parts("tau", &[&insecure_test_secret(803)]);
            for i in 0..5 {
                let (nv, deg) = shapes[i];
                let (wt, expect) = if i == 3 {
                    let tau: Vec<FqExt> = (0..nv).map(|_| tau_rng.next_fq4()).collect();
                    (
                        sumcheck::Masker::weights_total_eq(nv, deg, &tau),
                        ms[i].total_eq(&tau),
                    )
                } else {
                    (sumcheck::Masker::weights_total_plain(nv, deg), ms[i].total_plain())
                };
                assert_eq!(
                    pcs::open_linear(&t, &mask_weights(&d, bases[i], &wt)),
                    expect,
                    "the sum for mask {i} cannot be recovered (n={n} g={g} ell={ell})"
                );
                let r: Vec<FqExt> = (0..nv).map(|_| tau_rng.next_fq4()).collect();
                for (j, &rj) in r.iter().enumerate() {
                    ms[i].fold(j, rj);
                }
                assert_eq!(
                    pcs::open_linear(
                        &t,
                        &mask_weights(&d, bases[i], &sumcheck::Masker::weights_eval(nv, deg, &r))
                    ),
                    ms[i].eval(),
                    "g(r) for mask {i} cannot be recovered"
                );
            }
        }
    }

    #[test]
    fn hpack_open_matches_full_pcs_open() {
        for &(n, g, ell, seed) in
            &[(8usize, 2usize, 1usize, 401u64), (8, 4, 2, 402), (16, 8, 1, 403), (128, 8, 3, 404)]
        {
            let (params, nz, groups, bw) = setup_nizk1(seed, n, g, ell);
            let (ch, wit) = eval_h(&params, &groups);
            let (st, _) = blind_statement(&params, &nz, &ch, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));
            let zk = crate::nizk1::zk_mask_rows(&mut CsRng::from_parts("t", &[&[0u8; 32]]));
            let zw = build_w_table(&params, &d, &wit, Some(&ctx), Some(&bw), &zk);
            let mut rng = SimpleRng::new(seed ^ 0xBEEF);
            for _ in 0..4 {
                let r6: Vec<FqExt> = (0..d.nv_h).map(|_| rng.next_fq4()).collect();
                assert_eq!(
                    hpack_open(&zw, &d, &ctx, &r6),
                    pcs::open(&zw, &hpack_point(&d, &ctx, &r6)),
                    "n={n} g={g} ell={ell}"
                );
            }
        }
    }
}
