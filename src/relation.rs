use crate::ext_field::Fq4;
use crate::field::Fq;
use crate::hash::HashWitness;
use crate::params::HashParams;
use crate::nizk1::{w_layout, BlindStatement, Nizk1Params, WLayout};
use crate::ring::{gadget_recompose, gadget_scalar, poly_eval_pows, RingElem, BASE, DELTA, N};
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

pub fn m_row(params: &HashParams, j: usize, i: usize, d: usize) -> usize {
    let steps = params.num_groups() - 1;
    (j * steps + (i - 2)) * DELTA + d
}

pub fn num_m_rows(params: &HashParams) -> usize {
    params.ell * (params.num_groups() - 1) * DELTA
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

    let inner = |v: usize, col: &[RingElem]| -> RingElem {
        let mut acc = RingElem::zero();
        for d in 0..DELTA {
            acc = &acc + &(&params.table[v][d] * &col[d]);
        }
        acc
    };

    for j in 0..params.ell {
        for i in 2..=ng {
            for m in wit.column(j, i) {
                for &c in &m.c {
                    if c.0 as u64 >= BASE {
                        return false;
                    }
                }
            }
        }
        for i in 2..ng {
            let lhs = gadget_recompose(wit.column(j, i));
            let rhs = inner(groups[i - 1], wit.column(j, i + 1));
            if lhs != rhs {
                return false;
            }
        }
        let lhs = gadget_recompose(wit.column(j, ng));
        if lhs != params.table[groups[ng - 1]][j] {
            return false;
        }
        if inner(groups[0], wit.column(j, 2)) != ch[j] {
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
                let q = bq.expect("Phase B 的商未提供");
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
    pub m_entries: Vec<(usize, Fq4)>,
    pub p_pub: Fq4,
}

pub struct Rows {
    pub lin: Vec<LinRow>,
    pub a_hat: Vec<Vec<Fq4>>,
}

impl Rows {
    pub fn u_pub(&self, row: &LinRow, v: usize) -> Fq4 {
        match row.kind {
            CKind::Base => self.a_hat[v][row.chain],
            _ => Fq4::ZERO,
        }
    }
}

pub fn build_rows(
    params: &HashParams,
    ch: &[RingElem],
    alpha: Fq4,
    nz: Option<&Nizk1Ctx>,
) -> Rows {
    let ng = params.num_groups();

    let mut apw = Vec::with_capacity(N);
    let mut p = Fq4::ONE;
    for _ in 0..N {
        apw.push(p);
        p = p * alpha;
    }
    let ch_hat: Vec<Fq4> = ch.iter().map(|a| poly_eval_pows(&a.c, &apw)).collect();

    let a_hat: Vec<Vec<Fq4>> = params
        .table
        .par_iter()
        .map(|row| row.iter().map(|e| poly_eval_pows(&e.c, &apw)).collect())
        .collect();

    let ev = |e: &RingElem| poly_eval_pows(&e.c, &apw);
    let evv = |v: &[RingElem]| -> Vec<Fq4> { v.iter().map(ev).collect() };
    let (ar_hat, cr_hat, dx_hat, akey_hat) = match nz {
        Some(ctx) => {
            let ar: Vec<Vec<Fq4>> = ctx.a_r.iter().map(|r| evv(r)).collect();
            let keys: Vec<(Vec<Vec<Fq4>>, Vec<Vec<Fq4>>)> = [&ctx.nz.com_r, &ctx.nz.com_x]
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
        let mut p_pub = Fq4::ZERO;
        match meta.kind {
            CKind::Recursion { i } => {
                for d in 0..DELTA {
                    m_entries.push((m_row(params, j, i, d), Fq4::from_fq(gadget_scalar(d))));
                }
            }
            CKind::Base => {
                for d in 0..DELTA {
                    m_entries.push((m_row(params, j, ng, d), Fq4::from_fq(gadget_scalar(d))));
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
                let ctx = nz.expect("ComRand 需要 Phase B 參數");
                let (a_hat_k, _) = &akey_hat[which];
                let (pos, neg) = rho_rows(ctx, which);
                for (t, &coef) in a_hat_k[idx].iter().enumerate() {
                    m_entries.push((pos + t, coef));
                    m_entries.push((neg + t, -coef));
                }
                p_pub = -if which == 0 { cr_hat[idx] } else { dx_hat[idx] };
            }
            CKind::ComMsg { which, idx } => {
                let ctx = nz.expect("ComMsg 需要 Phase B 參數");
                let (_, b_hat_k) = &akey_hat[which];
                let (pos, neg) = rho_rows(ctx, which);
                let two = Fq4::from_u64(2);
                for (t, &coef) in b_hat_k[idx].iter().enumerate() {
                    m_entries.push((pos + t, two * coef));
                    m_entries.push((neg + t, -(two * coef)));
                }
                let msg_row = if which == 0 {
                    if idx < ctx.nz.r_dim { ctx.w.r_pos + idx } else { ctx.w.r_neg + idx - ctx.nz.r_dim }
                } else {
                    ctx.w.h_pack + idx
                };
                m_entries.push((msg_row, Fq4::ONE));
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
    Rows { lin, a_hat }
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
    fn honest_witness_passes_direct_check() {
        let (params, groups) = setup(8, 2, 2);
        let (ch, wit) = eval_h(&params, &groups);
        assert!(check_witness(&params, &ch, &groups, &wit));
    }

    #[test]
    fn tampered_digit_fails_direct_check() {
        let (params, groups) = setup(8, 2, 1);
        let (ch, mut wit) = eval_h(&params, &groups);
        wit.m[0][0][3].c[100] = wit.m[0][0][3].c[100] + Fq::ONE;
        assert!(!check_witness(&params, &ch, &groups, &wit));
    }

    #[test]
    fn rows_hold_for_honest_witness() {
        let (params, groups) = setup(8, 4, 2);
        let (ch, wit) = eval_h(&params, &groups);
        let quotients = compute_quotients(&params, &wit, None, None);
        let ng = params.num_groups();

        let mut rng = SimpleRng::new(2718);
        let alpha = rng.next_fq4();

        let mut w_hat = vec![Fq4::ZERO; num_m_rows(&params)];
        for j in 0..params.ell {
            for i in 2..=ng {
                for d in 0..DELTA {
                    w_hat[m_row(&params, j, i, d)] = wit.column(j, i)[d].eval(alpha);
                }
            }
        }

        let xn1 = alpha.pow(N as u128) + Fq4::ONE;
        let rows = build_rows(&params, &ch, alpha, None);
        let metas = constraints(&params, None);
        for (row, meta) in rows.lin.iter().zip(&metas) {
            let a1 = row.m_entries.iter().fold(row.p_pub, |acc, &(k, c)| acc + c * w_hat[k]);
            let v = groups[row.step_i - 1];
            let u = match u_src(row.kind) {
                Some(src) => (0..DELTA).fold(Fq4::ZERO, |acc, d| {
                    acc + rows.a_hat[v][d] * w_hat[m_row(&params, row.chain, src, d)]
                }),
                None => rows.u_pub(row, v),
            };
            let tc = meta.t_index.map(|idx| poly_eval(&quotients[idx], alpha)).unwrap_or(Fq4::ZERO);
            assert_eq!(a1, u + xn1 * tc, "row {:?}", row.kind);
        }
    }
}
