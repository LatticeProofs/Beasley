
use crate::ext_field::FqExt;
use crate::field::Fq;
use crate::hash::HashWitness;
use crate::params::HashParams;
use crate::nizk1::{w_layout, BlindStatement, Nizk1Params, WLayout};
use crate::ring::{
    bit_weight, gadget_recompose, poly_eval_pows, RingElem, DIGIT_BITS, GADGET_BASE, GADGET_LEN,
    M_BIT_ROWS, N,
};
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CKind {
    Recursion { i: usize },
    Base,
    Output,
    ComRand { which: usize, idx: usize },
    ComMsg { which: usize, idx: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct ConstraintMeta {
    pub kind: CKind,
    pub chain: usize,
    pub step_i: usize,
    pub cell: usize,
    pub t_index: Option<usize>,
}

pub struct Nizk1Ctx<'a> {
    pub nz: &'a Nizk1Params,
    pub st: &'a BlindStatement,
    pub a_r: Vec<Vec<RingElem>>,
    pub w: WLayout,
}

impl<'a> Nizk1Ctx<'a> {
    pub fn new(params: &HashParams, nz: &'a Nizk1Params, st: &'a BlindStatement) -> Self {
        let a_r = crate::nizk1::derive_ar(nz.r_dim, params.ell, &st.c_r);
        Nizk1Ctx { nz, st, a_r, w: w_layout(nz, num_m_rows(params)) }
    }
    pub fn num_rows(&self) -> usize {
        self.nz.com_r.out_len() + self.nz.com_x.out_len()
    }
}

pub fn ell_pad(params: &HashParams) -> usize {
    params.ell.next_power_of_two()
}

pub fn phase_a_cells(params: &HashParams) -> usize {
    ell_pad(params) * g_pad(params)
}

pub fn g_pad(params: &HashParams) -> usize {
    params.num_groups().next_power_of_two()
}

pub fn cell_of(params: &HashParams, chain: usize, step_i: usize) -> usize {
    chain * g_pad(params) + (step_i - 1)
}

fn rho_rows(ctx: &Nizk1Ctx, which: usize) -> (usize, usize) {
    if which == 0 {
        (ctx.w.rho_r_pos, ctx.w.rho_r_neg)
    } else {
        (ctx.w.rho_x_pos, ctx.w.rho_x_neg)
    }
}

pub fn u_src(kind: CKind) -> Option<usize> {
    match kind {
        CKind::Recursion { i } => Some(i + 1),
        CKind::Output => Some(2),
        _ => None,
    }
}

pub fn constraints(params: &HashParams, nz: Option<&Nizk1Ctx>) -> Vec<ConstraintMeta> {
    let ng = params.num_groups();
    let mut out = Vec::new();
    let mut t_count = 0;
    let mut push = |kind, chain, step_i, cell, t: &mut usize, has_t: bool| {
        let t_index = if has_t {
            let i = *t;
            *t += 1;
            Some(i)
        } else {
            None
        };
        out.push(ConstraintMeta { kind, chain, step_i, cell, t_index });
    };
    for j in 0..params.ell {
        for i in 2..ng {
            let cell = cell_of(params, j, i);
            push(CKind::Recursion { i }, j, i, cell, &mut t_count, true);
        }
        let cell = cell_of(params, j, ng);
        push(CKind::Base, j, ng, cell, &mut t_count, false);
        let cell = cell_of(params, j, 1);
        push(CKind::Output, j, 1, cell, &mut t_count, true);
    }
    if let Some(ctx) = nz {
        let mut cell = phase_a_cells(params);
        for (which, ck) in [(0usize, &ctx.nz.com_r), (1usize, &ctx.nz.com_x)] {
            for idx in 0..ck.com_n() {
                push(CKind::ComRand { which, idx }, 0, 0, cell, &mut t_count, true);
                cell += 1;
            }
            for idx in 0..ck.msg_len() {
                push(CKind::ComMsg { which, idx }, 0, 0, cell, &mut t_count, true);
                cell += 1;
            }
        }
    }
    out
}

pub fn num_quotients(params: &HashParams, nz: Option<&Nizk1Ctx>) -> usize {
    (params.num_groups() - 1) * params.ell + nz.map_or(0, |c| c.num_rows())
}

pub fn m_row(params: &HashParams, t: usize, i: usize, k: usize) -> usize {
    let steps = params.num_groups() - 1;
    (t * steps + (i - 2)) * M_BIT_ROWS + k
}

pub fn m_row_flat(params: &HashParams, i: usize, e: usize) -> usize {
    m_row(params, e / M_BIT_ROWS, i, e % M_BIT_ROWS)
}

pub fn ml_bits(params: &HashParams) -> usize {
    params.ell * M_BIT_ROWS
}

pub fn num_m_rows(params: &HashParams) -> usize {
    params.ell * (params.num_groups() - 1) * M_BIT_ROWS
}

pub fn check_witness(
    params: &HashParams,
    ch: &[RingElem],
    groups: &[usize],
    wit: &HashWitness,
) -> bool {
    let ng = params.num_groups();
    if ch.len() != params.ell || groups.len() != ng {
        return false;
    }

    let ml = params.ml();

    let inner = |v: usize, r: usize, md: &[RingElem]| -> RingElem {
        let mut acc = RingElem::zero();
        for d in 0..ml {
            acc = &acc + &(params.a(v, r, d) * &md[d]);
        }
        acc
    };
    let block = |md: &[RingElem], r: usize| -> RingElem {
        gadget_recompose(&md[r * GADGET_LEN..(r + 1) * GADGET_LEN])
    };

    for i in 2..=ng {
        let md = wit.column(i);
        if md.len() != ml {
            return false;
        }
        for m in md {
            for &c in &m.c {
                if c.0 as u64 >= GADGET_BASE {
                    return false;
                }
            }
        }
    }

    for r in 0..params.ell {
        for i in 2..ng {
            if block(wit.column(i), r) != inner(groups[i - 1], r, wit.column(i + 1)) {
                return false;
            }
        }
        if block(wit.column(ng), r) != *params.a(groups[ng - 1], r, 0) {
            return false;
        }
        if inner(groups[0], r, wit.column(2)) != ch[r] {
            return false;
        }
    }
    true
}

pub fn compute_quotients(
    params: &HashParams,
    wit: &HashWitness,
    nz: Option<&Nizk1Ctx>,
    bq: Option<&crate::nizk1::BlindQuotients>,
) -> Vec<Vec<Fq>> {
    let mut ts = Vec::with_capacity(num_quotients(params, nz));
    for meta in constraints(params, nz) {
        match meta.kind {
            CKind::Base => {}
            CKind::Recursion { i } => ts.push(wit.quotient(meta.chain, i + 1).to_vec()),
            CKind::Output => {
                let mut t = wit.quotient(meta.chain, 2).to_vec();
                if let Some(q) = bq {
                    for (a, b) in t.iter_mut().zip(&q.neg_hi_r[meta.chain]) {
                        *a = *a + *b;
                    }
                }
                ts.push(t);
            }
            CKind::ComRand { which, idx } | CKind::ComMsg { which, idx } => {
                let q = bq.expect("Phase B quotients not provided");
                let ck = if which == 0 { &nz.unwrap().nz.com_r } else { &nz.unwrap().nz.com_x };
                let off = if matches!(meta.kind, CKind::ComRand { .. }) { 0 } else { ck.com_n() };
                let src = if which == 0 { &q.cr } else { &q.dx };
                ts.push(src[off + idx].clone());
            }
        }
    }
    ts
}

#[derive(Clone, Debug)]
pub struct LinRow {
    pub cell: usize,
    pub step_i: usize,
    pub kind: CKind,
    pub chain: usize,
    pub m_entries: Vec<(usize, FqExt)>,
    pub p_pub: FqExt,
}

pub struct Rows {
    pub lin: Vec<LinRow>,
    pub a_base: Vec<Vec<Vec<FqExt>>>,
}

pub fn bit_scale() -> Vec<Fq> {
    (0..DIGIT_BITS).map(|b| Fq::new(1u64 << b)).collect()
}

impl Rows {
    pub fn u_pub(&self, row: &LinRow, v: usize) -> FqExt {
        match row.kind {
            CKind::Base => self.a_base[v][row.chain][0],
            _ => FqExt::ZERO,
        }
    }
}

pub fn build_rows(
    params: &HashParams,
    ch: &[RingElem],
    alpha: FqExt,
    nz: Option<&Nizk1Ctx>,
) -> Rows {
    let ng = params.num_groups();

    let mut apw = Vec::with_capacity(N);
    let mut p = FqExt::ONE;
    for _ in 0..N {
        apw.push(p);
        p = p * alpha;
    }
    let ch_hat: Vec<FqExt> = ch.iter().map(|a| poly_eval_pows(&a.c, &apw)).collect();

    let ml = params.ml();
    let a_base: Vec<Vec<Vec<FqExt>>> = params
        .table
        .par_iter()
        .map(|mat| {
            mat.chunks(ml)
                .map(|row| row.iter().map(|e| poly_eval_pows(&e.c, &apw)).collect())
                .collect()
        })
        .collect();

    let ev = |e: &RingElem| poly_eval_pows(&e.c, &apw);
    let evv = |v: &[RingElem]| -> Vec<FqExt> { v.iter().map(ev).collect() };
    let (ar_hat, cr_hat, dx_hat, akey_hat) = match nz {
        Some(ctx) => {
            let ar: Vec<Vec<FqExt>> = ctx.a_r.iter().map(|r| evv(r)).collect();
            let keys: Vec<(Vec<Vec<FqExt>>, Vec<Vec<FqExt>>)> = [&ctx.nz.com_r, &ctx.nz.com_x]
                .iter()
                .map(|ck| {
                    (
                        ck.a.iter().map(|r| evv(r)).collect::<Vec<_>>(),
                        ck.b.iter().map(|r| evv(r)).collect::<Vec<_>>(),
                    )
                })
                .collect();
            (ar, evv(&ctx.st.c_r), evv(&ctx.st.d_x), keys)
        }
        None => (vec![], vec![], vec![], vec![]),
    };

    let mut lin = Vec::new();
    for meta in constraints(params, nz) {
        let j = meta.chain;
        let mut m_entries = Vec::new();
        let mut p_pub = FqExt::ZERO;
        match meta.kind {
            CKind::Recursion { i } => {
                for k in 0..M_BIT_ROWS {
                    m_entries.push((m_row(params, j, i, k), FqExt::from_fq(bit_weight(k))));
                }
            }
            CKind::Base => {
                for k in 0..M_BIT_ROWS {
                    m_entries.push((m_row(params, j, ng, k), FqExt::from_fq(bit_weight(k))));
                }
            }
            CKind::Output => {
                p_pub = ch_hat[j];
                if let Some(ctx) = nz {
                    for u in 0..ctx.nz.r_dim {
                        let a = ar_hat[u][j];
                        m_entries.push((ctx.w.r_pos + u, -a));
                        m_entries.push((ctx.w.r_neg + u, a));
                    }
                }
            }
            CKind::ComRand { which, idx } => {
                let ctx = nz.expect("ComRand requires Phase B parameters");
                let (a_hat_k, _) = &akey_hat[which];
                let (pos, neg) = rho_rows(ctx, which);
                for (t, &coef) in a_hat_k[idx].iter().enumerate() {
                    m_entries.push((pos + t, coef));
                    m_entries.push((neg + t, -coef));
                }
                p_pub = -if which == 0 { cr_hat[idx] } else { dx_hat[idx] };
            }
            CKind::ComMsg { which, idx } => {
                let ctx = nz.expect("ComMsg requires Phase B parameters");
                let (_, b_hat_k) = &akey_hat[which];
                let (pos, neg) = rho_rows(ctx, which);
                let two = FqExt::from_u64(2);
                for (t, &coef) in b_hat_k[idx].iter().enumerate() {
                    m_entries.push((pos + t, two * coef));
                    m_entries.push((neg + t, -(two * coef)));
                }
                let msg_row = if which == 0 {
                    if idx < ctx.nz.r_dim { ctx.w.r_pos + idx } else { ctx.w.r_neg + idx - ctx.nz.r_dim }
                } else {
                    ctx.w.h_pack + idx
                };
                m_entries.push((msg_row, FqExt::ONE));
                let ck_n = if which == 0 { ctx.nz.com_r.com_n() } else { ctx.nz.com_x.com_n() };
                p_pub = -if which == 0 { cr_hat[ck_n + idx] } else { dx_hat[ck_n + idx] };
            }
        }
        lin.push(LinRow {
            cell: meta.cell,
            step_i: meta.step_i,
            kind: meta.kind,
            chain: j,
            m_entries,
            p_pub,
        });
    }
    Rows { lin, a_base }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{bits_to_groups, eval_h};
    use crate::ring::poly_eval;
    use crate::transcript::SimpleRng;

    fn setup(n: usize, g: usize, ell: usize) -> (HashParams, Vec<usize>) {
        let params = HashParams::sample(2024, n, g, ell);
        let mut rng = SimpleRng::new(99);
        let bits: Vec<bool> = (0..n).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        (params, groups)
    }

    #[test]
    fn u_pub_is_the_gadget_column_not_the_bit_row() {
        let (params, groups) = setup(16, 4, 3);
        let (ch, _) = eval_h(&params, &groups);
        let mut rng = SimpleRng::new(20260802);
        let alpha = rng.next_fq4();
        let rows = build_rows(&params, &ch, alpha, None);
        let metas = constraints(&params, None);
        let mut checked = 0usize;
        for (row, meta) in rows.lin.iter().zip(&metas) {
            if meta.kind != CKind::Base {
                continue;
            }
            for v in 0..params.table_size() {
                assert_eq!(
                    rows.u_pub(row, v),
                    params.a(v, row.chain, 0).eval(alpha),
                    "u_pub picked the wrong cell: v={v} chain={}",
                    row.chain
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no Base row was exercised");
        for v in 0..params.table_size() {
            for r in 0..params.ell {
                for d in 0..params.ml() {
                    assert_eq!(
                        rows.a_base[v][r][d],
                        params.a(v, r, d).eval(alpha),
                        "a_base != A^(v)[r][d](alpha): v={v} r={r} d={d}"
                    );
                }
            }
        }
    }

    #[test]
    fn honest_witness_passes_direct_check() {
        let (params, groups) = setup(8, 2, 2);
        let (ch, wit) = eval_h(&params, &groups);
        assert!(check_witness(&params, &ch, &groups, &wit));
    }

    #[test]
    fn tampered_digit_fails_direct_check() {
        let (params, groups) = setup(8, 2, 1);
        let (ch, mut wit) = eval_h(&params, &groups);
        wit.m[0][3].c[100] = wit.m[0][3].c[100] + Fq::ONE;
        assert!(!check_witness(&params, &ch, &groups, &wit));
    }

    #[test]
    fn check_witness_rejects_out_of_range_digit() {
        let (params, groups) = setup(8, 2, 1);
        let (ch, wit) = eval_h(&params, &groups);
        assert!(check_witness(&params, &ch, &groups, &wit), "honest witness should pass");
        let mut bad = eval_h(&params, &groups).1;
        bad.m[0][2].c[7] = Fq::new(GADGET_BASE);
        assert!(
            !check_witness(&params, &ch, &groups, &bad),
            "digit = GADGET_BASE ({GADGET_BASE}) is out of range and must be rejected"
        );
        if GADGET_BASE > 2 {
            let big = (2..=params.num_groups())
                .flat_map(|i| wit.column(i))
                .flat_map(|m| m.c.iter())
                .any(|c| c.0 as u64 >= 2);
            assert!(big, "honest base-{GADGET_BASE} witness has no digit >= 2");
        }
    }

    #[test]
    fn gadget_represents_extreme_field_elements() {
        use crate::ring::{gadget_decompose, gadget_recompose};
        for v in [0u64, 1, 2, crate::field::Q - 1, crate::field::Q / 2, crate::field::Q - 2] {
            let mut e = RingElem::zero();
            for i in 0..N {
                e.c[i] = Fq::new(v);
            }
            let digits = gadget_decompose(&e);
            assert_eq!(digits.len(), GADGET_LEN);
            for d in &digits {
                for c in &d.c {
                    assert!((c.0 as u64) < GADGET_BASE, "digit out of range: v = {v}");
                }
            }
            assert_eq!(gadget_recompose(&digits), e, "v = {v} does not recompose to the original value");
        }
    }

    #[test]
    fn rows_hold_for_honest_witness() {
        let (params, groups) = setup(8, 4, 2);
        let (ch, wit) = eval_h(&params, &groups);
        let quotients = compute_quotients(&params, &wit, None, None);
        let ng = params.num_groups();

        let mut rng = SimpleRng::new(2718);
        let alpha = rng.next_fq4();

        let mut w_hat = vec![FqExt::ZERO; num_m_rows(&params)];
        for i in 2..=ng {
            for (d, digit) in wit.column(i).iter().enumerate() {
                let (t, c) = (d / GADGET_LEN, d % GADGET_LEN);
                for b in 0..DIGIT_BITS {
                    let bits = crate::ring::RingElem {
                        c: digit.c.iter().map(|v| Fq(((v.0 as u64 >> b) & 1) as _)).collect(),
                    };
                    w_hat[m_row(&params, t, i, c * DIGIT_BITS + b)] = bits.eval(alpha);
                }
            }
        }

        let xn1 = alpha.pow(N as u128) + FqExt::ONE;
        let rows = build_rows(&params, &ch, alpha, None);
        let metas = constraints(&params, None);
        for (row, meta) in rows.lin.iter().zip(&metas) {
            let a1 = row.m_entries.iter().fold(row.p_pub, |acc, &(k, c)| acc + c * w_hat[k]);
            let v = groups[row.step_i - 1];
            let u = match u_src(row.kind) {
                Some(src) => (0..ml_bits(&params)).fold(FqExt::ZERO, |acc, e| {
                    #[allow(clippy::modulo_one)]
                    let b = e % DIGIT_BITS;
                    let ah = Fq::new(1u64 << b) * rows.a_base[v][row.chain][e / DIGIT_BITS];
                    acc + ah * w_hat[m_row_flat(&params, src, e)]
                }),
                None => rows.u_pub(row, v),
            };
            let tc = meta.t_index.map(|idx| poly_eval(&quotients[idx], alpha)).unwrap_or(FqExt::ZERO);
            assert_eq!(a1, u + xn1 * tc, "row {:?}", row.kind);
        }
    }
}
