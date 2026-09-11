
use crate::ext_field::FqExt;
use crate::field::Fq;
use crate::hash::HashWitness;
use crate::nizk1::{bin_layout, BinLayout, BlindStatement, Nizk1Params};
use crate::params::HashParams;
use crate::ring::{poly_eval_pows, RingElem, N};
use rayon::prelude::*;

pub fn g_pad(params: &HashParams) -> usize {
    params.num_groups().next_power_of_two()
}

pub fn hv_len(params: &HashParams) -> usize {
    params.table_size()
}

pub fn h_cells(params: &HashParams) -> usize {
    g_pad(params) * hv_len(params)
}

pub fn hpack_rows(params: &HashParams) -> usize {
    let rows = h_cells(params).div_ceil(crate::nizk1::HPACK_BITS).max(1);
    debug_assert!(rows.is_power_of_two(), "hpack_rows must be a power of two (precondition of hpack_point)");
    rows
}

pub fn hpack_per(params: &HashParams) -> usize {
    let per = h_cells(params) / hpack_rows(params);
    debug_assert!(per.is_power_of_two() && per <= crate::nizk1::HPACK_BITS);
    per
}

pub fn h_index(params: &HashParams, block: usize, v: usize) -> usize {
    debug_assert!(v < hv_len(params));
    block * hv_len(params) + v
}

pub fn h_bits(params: &HashParams, groups: &[usize]) -> Vec<bool> {
    let mut out = vec![false; h_cells(params)];
    for (i, &v) in groups.iter().enumerate() {
        out[h_index(params, i, v)] = true;
    }
    out
}

pub fn cell_of(params: &HashParams, chain: usize, step_i: usize) -> usize {
    debug_assert!(1 <= step_i && step_i <= params.num_groups());
    chain * g_pad(params) + (step_i - 1)
}

pub fn phase_a_cells(params: &HashParams) -> usize {
    params.ell * g_pad(params)
}

pub fn c_cells(params: &HashParams, nz: Option<&Nizk1Ctx>) -> usize {
    (phase_a_cells(params) + nz.map_or(0, |c| c.num_rows())).next_power_of_two()
}

pub fn b_row(params: &HashParams, i: usize, t: usize) -> usize {
    debug_assert!(i + 1 < params.num_groups() && t < params.ell);
    i * params.ell + t
}

pub fn num_b_rows(params: &HashParams) -> usize {
    (params.num_groups() - 1) * params.ell
}

pub fn b_base(params: &HashParams, nz: Option<&Nizk1Ctx>) -> usize {
    bin_layout(params, nz.map(|c| c.nz)).total.next_power_of_two().max(2)
}

pub fn u_width(params: &HashParams) -> usize {
    params.ell
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CKind {
    Init,
    Step { i: usize },
    Output,
    ComRand { which: usize, idx: usize },
    ComMsg { which: usize, idx: usize },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CGroup {
    Prf,
    Blinding,
}

impl CKind {
    pub fn group(self) -> CGroup {
        match self {
            CKind::Init | CKind::Step { .. } => CGroup::Prf,
            CKind::Output | CKind::ComRand { .. } | CKind::ComMsg { .. } => CGroup::Blinding,
        }
    }
}

pub fn u_src(params: &HashParams, kind: CKind) -> Option<usize> {
    match kind {
        CKind::Step { i } => Some(i - 1),
        CKind::Output => Some(params.num_groups() - 2),
        _ => None,
    }
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
    pub w: BinLayout,
}

impl<'a> Nizk1Ctx<'a> {
    pub fn new(params: &HashParams, nz: &'a Nizk1Params, st: &'a BlindStatement) -> Self {
        let a_r = crate::nizk1::derive_ar(nz.r_dim, params.ell, &st.c_r);
        Nizk1Ctx { nz, st, a_r, w: bin_layout(params, Some(nz)) }
    }
    pub fn num_rows(&self) -> usize {
        self.nz.com_r.out_len() + self.nz.com_x.out_len()
    }
}

pub fn constraints(params: &HashParams, nz: Option<&Nizk1Ctx>) -> Vec<ConstraintMeta> {
    let mut out = prf_constraints(params);
    blinding_constraints(params, nz, &mut out);
    out
}

fn prf_constraints(params: &HashParams) -> Vec<ConstraintMeta> {
    let ng = params.num_groups();
    let mut out = Vec::new();
    let mut t = 0usize;
    for j in 0..params.ell {
        out.push(ConstraintMeta {
            kind: CKind::Init,
            chain: j,
            step_i: 1,
            cell: cell_of(params, j, 1),
            t_index: None,
        });
        for i in 1..ng - 1 {
            out.push(ConstraintMeta {
                kind: CKind::Step { i },
                chain: j,
                step_i: i + 1,
                cell: cell_of(params, j, i + 1),
                t_index: Some(t),
            });
            t += 1;
        }
        out.push(ConstraintMeta {
            kind: CKind::Output,
            chain: j,
            step_i: ng,
            cell: cell_of(params, j, ng),
            t_index: Some(t),
        });
        t += 1;
    }
    out
}

fn blinding_constraints(
    params: &HashParams,
    nz: Option<&Nizk1Ctx>,
    out: &mut Vec<ConstraintMeta>,
) {
    let Some(ctx) = nz else { return };
    let mut t = out.iter().filter_map(|m| m.t_index).max().map_or(0, |x| x + 1);
    let mut cell = phase_a_cells(params);
    for (which, ck) in [(0usize, &ctx.nz.com_r), (1usize, &ctx.nz.com_x)] {
        for idx in 0..ck.com_n() {
            out.push(ConstraintMeta {
                kind: CKind::ComRand { which, idx },
                chain: 0,
                step_i: 0,
                cell,
                t_index: Some(t),
            });
            cell += 1;
            t += 1;
        }
        for idx in 0..ck.msg_len() {
            out.push(ConstraintMeta {
                kind: CKind::ComMsg { which, idx },
                chain: 0,
                step_i: 0,
                cell,
                t_index: Some(t),
            });
            cell += 1;
            t += 1;
        }
    }
}

pub fn num_quotients(params: &HashParams, nz: Option<&Nizk1Ctx>) -> usize {
    (params.num_groups() - 1) * params.ell + nz.map_or(0, |c| c.num_rows())
}

pub fn compute_quotients(
    params: &HashParams,
    wit: &HashWitness,
    nz: Option<&Nizk1Ctx>,
    bq: Option<&crate::nizk1::BlindQuotients>,
) -> Vec<Vec<Fq>> {
    let ng = params.num_groups();
    let mut ts = Vec::with_capacity(num_quotients(params, nz));
    for meta in constraints(params, nz) {
        match meta.kind {
            CKind::Init => {}
            CKind::Step { i } => ts.push(wit.quotient(i, meta.chain).to_vec()),
            CKind::Output => {
                let mut t = wit.quotient(ng - 1, meta.chain).to_vec();
                if let Some(q) = bq {
                    for (a, b) in t.iter_mut().zip(&q.neg_hi_r[meta.chain]) {
                        *a = *a + *b;
                    }
                }
                ts.push(t);
            }
            CKind::ComRand { which, idx } | CKind::ComMsg { which, idx } => {
                let q = bq.expect("the quotients of group B were not provided");
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
    pub bin_entries: Vec<(usize, FqExt)>,
    pub full_entries: Vec<(usize, FqExt)>,
    pub p_pub: FqExt,
}

pub struct Rows {
    pub lin: Vec<LinRow>,
    pub b_base: usize,
    pub a_base: Vec<Vec<Vec<Vec<FqExt>>>>,
}

impl Rows {
    pub fn a_row(&self, row: &LinRow, v: usize) -> &[FqExt] {
        debug_assert!(row.step_i >= 1, "the commitment row has no u side");
        &self.a_base[row.step_i - 1][v][row.chain]
    }

    pub fn u_pub(&self, row: &LinRow, v: usize) -> FqExt {
        match row.kind {
            CKind::Init => self.a_row(row, v)[0],
            _ => FqExt::ZERO,
        }
    }
}

pub fn build_rows(
    params: &HashParams,
    stmt: &[RingElem],
    alpha: FqExt,
    nz: Option<&Nizk1Ctx>,
) -> Rows {
    let m = params.ell;

    let mut apw = Vec::with_capacity(N);
    let mut p = FqExt::ONE;
    for _ in 0..N {
        apw.push(p);
        p = p * alpha;
    }
    let ev = |e: &RingElem| poly_eval_pows(&e.c, &apw);
    let evv = |v: &[RingElem]| -> Vec<FqExt> { v.iter().map(ev).collect() };
    let stmt_hat: Vec<FqExt> = evv(stmt);
    let bb = b_base(params, nz);

    let a_base: Vec<Vec<Vec<Vec<FqExt>>>> = params
        .table
        .iter()
        .map(|blk| {
            blk.par_iter()
                .map(|mat| mat.chunks(m).map(|row| row.iter().map(ev).collect()).collect())
                .collect()
        })
        .collect();

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
        let mut bin_entries = Vec::new();
        let mut full_entries = Vec::new();
        let mut p_pub = FqExt::ZERO;
        match meta.kind {
            CKind::Init => {
                full_entries.push((bb + b_row(params, 0, j), FqExt::ONE));
            }
            CKind::Step { i } => {
                full_entries.push((bb + b_row(params, i, j), FqExt::ONE));
            }
            CKind::Output => {
                p_pub = stmt_hat[j];
                if let Some(ctx) = nz {
                    blinding_linear(ctx, j, &ar_hat, &mut bin_entries);
                }
            }
            CKind::ComRand { which, idx } => {
                let ctx = nz.expect("ComRand requires the parameters of group B");
                let (a_hat_k, _) = &akey_hat[which];
                let (pos, neg) = rho_rows(ctx, which);
                for (t, &coef) in a_hat_k[idx].iter().enumerate() {
                    bin_entries.push((pos + t, coef));
                    bin_entries.push((neg + t, -coef));
                }
                p_pub = -if which == 0 { cr_hat[idx] } else { dx_hat[idx] };
            }
            CKind::ComMsg { which, idx } => {
                let ctx = nz.expect("ComMsg requires the parameters of group B");
                let (_, b_hat_k) = &akey_hat[which];
                let (pos, neg) = rho_rows(ctx, which);
                let two = FqExt::from_u64(2);
                for (t, &coef) in b_hat_k[idx].iter().enumerate() {
                    bin_entries.push((pos + t, two * coef));
                    bin_entries.push((neg + t, -(two * coef)));
                }
                let msg_row = if which == 0 {
                    if idx < ctx.nz.r_dim {
                        ctx.w.r_pos + idx
                    } else {
                        ctx.w.r_neg + idx - ctx.nz.r_dim
                    }
                } else {
                    ctx.w.h_pack + idx
                };
                bin_entries.push((msg_row, FqExt::ONE));
                let ck_n = if which == 0 { ctx.nz.com_r.com_n() } else { ctx.nz.com_x.com_n() };
                p_pub = -if which == 0 { cr_hat[ck_n + idx] } else { dx_hat[ck_n + idx] };
            }
        }
        lin.push(LinRow {
            cell: meta.cell,
            step_i: meta.step_i,
            kind: meta.kind,
            chain: j,
            bin_entries,
            full_entries,
            p_pub,
        });
    }
    Rows { lin, a_base, b_base: bb }
}

fn blinding_linear(
    ctx: &Nizk1Ctx,
    j: usize,
    ar_hat: &[Vec<FqExt>],
    bin_entries: &mut Vec<(usize, FqExt)>,
) {
    for u in 0..ctx.nz.r_dim {
        let a = ar_hat[u][j];
        bin_entries.push((ctx.w.r_pos + u, -a));
        bin_entries.push((ctx.w.r_neg + u, a));
    }
}

fn rho_rows(ctx: &Nizk1Ctx, which: usize) -> (usize, usize) {
    if which == 0 {
        (ctx.w.rho_r_pos, ctx.w.rho_r_neg)
    } else {
        (ctx.w.rho_x_pos, ctx.w.rho_x_neg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{bits_to_groups, eval_h};
    use crate::nizk1::{blind_statement, sample_blind, QueryTicket};
    use crate::ring::poly_eval;
    use crate::rng::insecure_test_secret;
    use crate::transcript::SimpleRng;

    fn setup(n: usize, g: usize, ell: usize, seed: u64) -> (HashParams, Vec<usize>) {
        let params = HashParams::sample(seed, n, g, ell);
        let mut rng = SimpleRng::new(seed ^ 0x5EED);
        let bits: Vec<bool> = (0..n).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        (params, groups)
    }

    fn w_hats(
        params: &HashParams,
        wit: &HashWitness,
        alpha: FqExt,
        groups: &[usize],
        ctx: Option<&Nizk1Ctx>,
        bw: Option<&crate::nizk1::BlindWitness>,
    ) -> (Vec<FqExt>, Vec<FqExt>) {
        let bb = b_base(params, ctx);
        let mut full = vec![FqExt::ZERO; bb + num_b_rows(params)];
        for k in 0..num_b_rows(params) {
            full[bb + k] = wit.b(k / params.ell)[k % params.ell].eval(alpha);
        }

        let lay = bin_layout(params, ctx.map(|c| c.nz));
        let mut bin = vec![FqExt::ZERO; lay.total];
        let hb = h_bits(params, groups);
        let per = hpack_per(params);
        for r in 0..hpack_rows(params) {
            let mut e = RingElem::zero();
            for c in 0..per {
                if hb[r * per + c] {
                    e.c[c] = Fq::ONE;
                }
            }
            bin[lay.h_pack + r] = e.eval(alpha);
        }
        if let (Some(_ctx), Some(bw)) = (ctx, bw) {
            for (start, src) in [
                (lay.r_pos, &bw.r_pos),
                (lay.r_neg, &bw.r_neg),
                (lay.rho_r_pos, &bw.rho_r_pos),
                (lay.rho_r_neg, &bw.rho_r_neg),
                (lay.rho_x_pos, &bw.rho_x_pos),
                (lay.rho_x_neg, &bw.rho_x_neg),
            ] {
                for (i, e) in src.iter().enumerate() {
                    bin[start + i] = e.eval(alpha);
                }
            }
        }
        (bin, full)
    }

    fn rows_hold(n: usize, g: usize, ell: usize, seed: u64, phase_b: bool) {
        let (params, groups) = setup(n, g, ell, seed);
        let ng = params.num_groups();
        let (bx, wit) = eval_h(&params, &groups);

        let mut rng = SimpleRng::new(seed ^ 0xA11CE);
        let alpha = rng.next_fq4();
        let xn1 = alpha.pow(N as u128) + FqExt::ONE;

        let nz = phase_b.then(|| Nizk1Params::sample(seed + 1, &params, 3, 2, 2));
        let bw = nz.as_ref().map(|nz| {
            sample_blind(
                QueryTicket::insecure_for_tests(insecure_test_secret(seed + 2), 0),
                &params,
                nz,
                &groups,
            )
        });
        let sb = nz
            .as_ref()
            .zip(bw.as_ref())
            .map(|(nz, bw)| blind_statement(&params, nz, &bx, bw));
        let ctx = nz
            .as_ref()
            .zip(sb.as_ref())
            .map(|(nz, (st, _))| Nizk1Ctx::new(&params, nz, st));
        let bq = sb.as_ref().map(|(_, q)| q);

        let stmt: Vec<RingElem> =
            if let Some((st, _)) = &sb { st.c_x.clone() } else { bx.clone() };

        let quotients = compute_quotients(&params, &wit, ctx.as_ref(), bq);
        let rows = build_rows(&params, &stmt, alpha, ctx.as_ref());
        let metas = constraints(&params, ctx.as_ref());
        let (w_bin, w_full) =
            w_hats(&params, &wit, alpha, &groups, ctx.as_ref(), bw.as_ref());

        assert_eq!(rows.lin.len(), metas.len());
        assert_eq!(quotients.len(), num_quotients(&params, ctx.as_ref()));

        let tag = format!("n={n} g={g} ell={ell} phase_b={phase_b}");
        for (row, meta) in rows.lin.iter().zip(&metas) {
            let mut a1 = row.p_pub;
            for &(k, c) in &row.bin_entries {
                a1 = a1 + c * w_bin[k];
            }
            for &(k, c) in &row.full_entries {
                a1 = a1 + c * w_full[k];
            }

            let u = if row.step_i == 0 {
                FqExt::ZERO
            } else {
                let v = groups[row.step_i - 1];
                match u_src(&params, row.kind) {
                    Some(src) => (0..u_width(&params)).fold(FqExt::ZERO, |acc, t| {
                        acc + rows.a_row(row, v)[t] * w_full[rows.b_base + b_row(&params, src, t)]
                    }),
                    None => rows.u_pub(row, v),
                }
            };

            let tc = meta
                .t_index
                .map(|i| poly_eval(&quotients[i], alpha))
                .unwrap_or(FqExt::ZERO);

            assert_eq!(a1, u + xn1 * tc, "{:?} chain={} ({tag})", row.kind, row.chain);
        }

        assert_eq!(num_b_rows(&params), (ng - 1) * ell);
        assert_eq!(phase_a_cells(&params), ell * g_pad(&params));
    }

    #[test]
    fn rows_hold_for_honest_witness_phase_a() {
        for &(n, g, ell, seed) in &[
            (8usize, 2usize, 1usize, 601u64),
            (8, 2, 3, 602),
            (8, 4, 2, 603),
            (12, 2, 5, 604),
            (8, 1, 2, 605),
            (16, 4, 5, 606),
        ] {
            rows_hold(n, g, ell, seed, false);
        }
    }

    #[test]
    fn rows_hold_for_honest_witness_phase_b() {
        for &(n, g, ell, seed) in &[
            (8usize, 2usize, 1usize, 701u64),
            (8, 4, 2, 702),
            (12, 2, 3, 703),
            (16, 4, 5, 704),
        ] {
            rows_hold(n, g, ell, seed, true);
        }
    }

    #[test]
    fn constraint_enumeration_is_consistent() {
        for &(n, g, ell) in &[(8usize, 2usize, 3usize), (16, 4, 5), (12, 2, 2)] {
            let (params, _) = setup(n, g, ell, 800);
            let metas = constraints(&params, None);
            let ng = params.num_groups();

            let ts: Vec<usize> = metas.iter().filter_map(|m| m.t_index).collect();
            assert_eq!(ts, (0..ts.len()).collect::<Vec<_>>(), "t_index is not contiguous");
            assert_eq!(ts.len(), num_quotients(&params, None));

            let mut cells: Vec<usize> = metas.iter().map(|m| m.cell).collect();
            cells.sort_unstable();
            let n_before = cells.len();
            cells.dedup();
            assert_eq!(cells.len(), n_before, "duplicate cell");

            for j in 0..ell {
                let mut steps: Vec<usize> =
                    metas.iter().filter(|m| m.chain == j).map(|m| m.step_i).collect();
                steps.sort_unstable();
                assert_eq!(steps, (1..=ng).collect::<Vec<_>>(), "the steps of chain {j} are incomplete");
            }

            assert!(metas.iter().all(|m| match m.kind {
                CKind::Init | CKind::Step { .. } => m.kind.group() == CGroup::Prf,
                _ => m.kind.group() == CGroup::Blinding,
            }));
        }
    }

    #[test]
    fn u_src_points_backwards() {
        let (params, _) = setup(16, 4, 5, 900);
        let ng = params.num_groups();
        assert_eq!(u_src(&params, CKind::Init), None);
        for i in 1..ng - 1 {
            assert_eq!(u_src(&params, CKind::Step { i }), Some(i - 1), "Step {i}");
        }
        assert_eq!(u_src(&params, CKind::Output), Some(ng - 2));
        assert_eq!(u_src(&params, CKind::ComRand { which: 0, idx: 0 }), None);
    }

    #[test]
    fn a_base_is_the_transposed_public_matrix() {
        let (params, groups) = setup(12, 4, 3, 1000);
        let (bx, _) = eval_h(&params, &groups);
        let mut rng = SimpleRng::new(1001);
        let alpha = rng.next_fq4();
        let rows = build_rows(&params, &bx, alpha, None);
        assert_eq!(rows.a_base.len(), params.num_groups());
        for i in 0..params.num_groups() {
            assert_eq!(rows.a_base[i].len(), params.table_size());
            for v in 0..params.table_size() {
                assert_eq!(rows.a_base[i][v].len(), params.ell);
                for j in 0..params.ell {
                    assert_eq!(rows.a_base[i][v][j].len(), params.ell);
                    for t in 0..params.ell {
                        assert_eq!(
                            rows.a_base[i][v][j][t],
                            params.a(i, v, j, t).eval(alpha),
                            "i={i} v={v} j={j} t={t}"
                        );
                    }
                }
            }
        }
        for row in rows.lin.iter().filter(|r| r.step_i >= 1) {
            let i = row.step_i - 1;
            for v in 0..params.table_size() {
                for t in 0..params.ell {
                    assert_eq!(rows.a_row(row, v)[t], params.a(i, v, row.chain, t).eval(alpha));
                }
            }
            if row.kind == CKind::Init {
                assert_eq!(i, 0);
                for v in 0..params.table_size() {
                    assert_eq!(rows.u_pub(row, v), params.a(0, v, row.chain, 0).eval(alpha));
                }
            }
        }
    }

    #[test]
    fn dimension_table_matches_the_plan() {
        let expect: &[(usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize)] = &[
            (4, 32, 32, 16, 512, 1, 160, 31, 256, 155, 186, 103),
            (8, 16, 16, 256, 4096, 8, 80, 38, 128, 75, 113, 124),
        ];
        for &(w, ng, gp, hv, hc, hr, pac, pb, cc, nb, nq, bt) in expect {
            let params = HashParams::dims_only(128, w, 5);
            assert_eq!(params.num_groups(), ng, "w={w} G");
            assert_eq!(g_pad(&params), gp, "w={w} g_pad");
            assert_eq!(hv_len(&params), hv, "w={w} hv");
            assert_eq!(h_cells(&params), hc, "w={w} h_cells");
            assert_eq!(hpack_rows(&params), hr, "w={w} hpack_rows");
            assert_eq!(hpack_per(&params), hc / hr, "w={w} hpack_per");
            assert_eq!(phase_a_cells(&params), pac, "w={w} phase_a_cells");
            assert_eq!(num_b_rows(&params), nb, "w={w} num_b_rows");
            assert_eq!(u_width(&params), 5, "w={w} u_width");

            let nz = Nizk1Params::sample(
                1,
                &params,
                crate::nizk1::R_DIM,
                crate::nizk1::COM_N,
                crate::nizk1::W_SLACK,
            );
            assert_eq!(nz.com_r.out_len() + nz.com_x.out_len(), pb, "w={w} Phase B rows");
            assert_eq!((pac + pb).next_power_of_two(), cc, "w={w} c_cells");
            assert_eq!((ng - 1) * 5 + pb, nq, "w={w} num_quotients");
            let lay = bin_layout(&params, Some(&nz));
            assert_eq!(lay.total, bt, "w={w} bin_layout.total");
            let nv_c = cc.trailing_zeros() as usize;
            assert_eq!(nv_c, if w == 4 { 8 } else { 7 }, "w={w} nv_c");
            assert_eq!(nv_c + w, if w == 4 { 12 } else { 15 }, "w={w} nv_u");
        }
    }

    #[test]
    fn tampered_witness_breaks_the_rows() {
        let (params, groups) = setup(8, 2, 2, 1100);
        let (bx, mut wit) = eval_h(&params, &groups);
        let mut rng = SimpleRng::new(1101);
        let alpha = rng.next_fq4();
        let xn1 = alpha.pow(N as u128) + FqExt::ONE;

        wit.b[0][0].c[7] = wit.b[0][0].c[7] + Fq::ONE;

        let quotients = compute_quotients(&params, &wit, None, None);
        let rows = build_rows(&params, &bx, alpha, None);
        let metas = constraints(&params, None);
        let (w_bin, w_full) = w_hats(&params, &wit, alpha, &groups, None, None);

        let mut broke = 0usize;
        for (row, meta) in rows.lin.iter().zip(&metas) {
            let mut a1 = row.p_pub;
            for &(k, c) in &row.bin_entries {
                a1 = a1 + c * w_bin[k];
            }
            for &(k, c) in &row.full_entries {
                a1 = a1 + c * w_full[k];
            }
            let v = groups[row.step_i - 1];
            let u = match u_src(&params, row.kind) {
                Some(src) => (0..u_width(&params)).fold(FqExt::ZERO, |acc, t| {
                    acc + rows.a_row(row, v)[t] * w_full[rows.b_base + b_row(&params, src, t)]
                }),
                None => rows.u_pub(row, v),
            };
            let tc = meta.t_index.map(|i| poly_eval(&quotients[i], alpha)).unwrap_or(FqExt::ZERO);
            if a1 != u + xn1 * tc {
                broke += 1;
            }
        }
        assert!(broke > 0, "no constraint turned red after tampering with the witness");
    }
}
