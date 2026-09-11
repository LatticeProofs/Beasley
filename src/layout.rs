
use crate::bits::PackedBits;
use crate::ext_field::FqExt;
use crate::field::Fq;
use crate::hash::HashWitness;
use crate::nizk1::{bin_layout, BlindWitness};
use crate::params::HashParams;
use crate::relation::{
    b_base, b_row, c_cells, g_pad, h_cells, hpack_per, hpack_rows, hv_len, num_b_rows,
    num_quotients, Nizk1Ctx,
};
use crate::ring::N;

pub const SLOT_VARS: usize = N.trailing_zeros() as usize;
pub const SLOTS: usize = 1 << SLOT_VARS;
const _: () = assert!(SLOTS == N, "the number of slots per row must be exactly N");

pub const ALPHA_LIMIT: usize = N;

pub const MASK_ROWS: usize = 1;

#[derive(Clone, Copy, Debug)]
pub struct MergedLayout {
    pub bin_rows: usize,
    pub b_base: usize,
    pub quot_base: usize,
    pub mask_base: usize,
    pub rows: usize,
}

impl MergedLayout {
    pub fn new(params: &HashParams, nz: Option<&Nizk1Ctx>) -> Self {
        let bin_rows = b_base(params, nz);
        debug_assert!(bin_rows.is_power_of_two() && bin_rows >= 2);
        let b_base = bin_rows;
        let quot_base = b_base + num_b_rows(params);
        let mask_base = quot_base + num_quotients(params, nz);
        let rows = (mask_base + MASK_ROWS).next_power_of_two();
        MergedLayout { bin_rows, b_base, quot_base, mask_base, rows }
    }
    pub fn b_row_of(&self, params: &HashParams, i: usize, t: usize) -> usize {
        self.b_base + b_row(params, i, t)
    }
    pub fn quot_row_of(&self, t_index: usize) -> usize {
        self.quot_base + t_index
    }
    pub fn is_bin_row(&self, row: usize) -> bool {
        row < self.bin_rows
    }
}

#[derive(Clone, Debug)]
pub struct Dims {
    pub g_pad: usize,
    pub tsz: usize,
    pub hv: usize,
    pub nv_i: usize,
    pub nv_h: usize,
    pub nv_c: usize,
    pub nv_u: usize,
    pub c_cells: usize,
    pub u_cells: usize,
    pub h_cells: usize,
    pub bin_rows: usize,
    pub bin_pad: usize,
    pub nv_bin: usize,
    pub h_row: usize,
    pub hpack_rows: usize,
    pub hpack_per: usize,
    pub hpack_per_bits: u32,
    pub merged: MergedLayout,
    pub nv: usize,
}

pub fn dims(params: &HashParams, nz: Option<&Nizk1Ctx>) -> Dims {
    let gp = g_pad(params);
    let tsz = params.table_size();
    let hv = hv_len(params);
    let w = params.group_bits;
    debug_assert_eq!(hv.trailing_zeros() as usize, w);

    let cc = c_cells(params, nz);
    let nv_c = cc.trailing_zeros() as usize;
    debug_assert!(cc >= gp, "c_cells ({cc}) < g_pad ({gp})");

    let lay = bin_layout(params, nz.map(|c| c.nz));
    let bin_rows = lay.total;
    let merged = MergedLayout::new(params, nz);
    let bin_pad = merged.bin_rows;
    debug_assert_eq!(bin_pad, bin_rows.next_power_of_two().max(2));
    let hp = hpack_rows(params);
    let per = hpack_per(params);

    Dims {
        g_pad: gp,
        tsz,
        hv,
        nv_i: gp.trailing_zeros() as usize,
        nv_h: gp.trailing_zeros() as usize + w,
        nv_c,
        nv_u: nv_c + w,
        c_cells: cc,
        u_cells: cc * hv,
        h_cells: h_cells(params),
        bin_rows,
        bin_pad,
        nv_bin: (bin_pad * SLOTS).trailing_zeros() as usize,
        h_row: lay.h_pack,
        hpack_rows: hp,
        hpack_per: per,
        hpack_per_bits: per.trailing_zeros(),
        merged,
        nv: (merged.rows * SLOTS).trailing_zeros() as usize,
    }
}

pub fn bin_prefix(d: &Dims, pt: &[FqExt]) -> Vec<FqExt> {
    debug_assert_eq!(pt.len(), d.nv_bin);
    let mut out = vec![FqExt::ZERO; d.nv - d.nv_bin];
    out.extend_from_slice(pt);
    out
}

#[inline]
pub fn h_cell(d: &Dims, cell: usize, v: usize) -> usize {
    (cell % d.g_pad) * d.hv + v
}

#[inline]
pub fn h_slot(d: &Dims, idx: usize) -> usize {
    debug_assert!(idx < d.h_cells);
    (d.h_row + (idx >> d.hpack_per_bits)) * SLOTS + (idx & (d.hpack_per - 1))
}

#[inline]
pub fn h_bit(zb: &PackedBits, d: &Dims, idx: usize) -> bool {
    zb.get(h_slot(d, idx))
}

pub fn hpack_point(d: &Dims, pt_h: &[FqExt]) -> Vec<FqExt> {
    debug_assert_eq!(pt_h.len(), d.nv_h);
    let nv_k = d.bin_pad.trailing_zeros() as usize;
    let nv_hp = d.hpack_rows.trailing_zeros() as usize;
    let nv_per = d.hpack_per.trailing_zeros() as usize;
    debug_assert_eq!(nv_hp + nv_per, d.nv_h, "hpack_rows·hpack_per must equal h_cells");
    debug_assert_eq!(d.h_row % d.hpack_rows, 0, "h_pack segment is misaligned => the row prefix is not constant");
    let prefix = d.h_row >> nv_hp;
    let mut pt = Vec::with_capacity(d.nv_bin);
    for b in (0..nv_k - nv_hp).rev() {
        pt.push(FqExt::from_u64(((prefix >> b) & 1) as u64));
    }
    pt.extend_from_slice(&pt_h[..nv_hp]);
    for _ in 0..SLOT_VARS - nv_per {
        pt.push(FqExt::ZERO);
    }
    pt.extend_from_slice(&pt_h[nv_hp..]);
    debug_assert_eq!(pt.len(), d.nv_bin);
    pt
}

pub fn half_point(d: &Dims, r_i: &[FqExt]) -> Vec<FqExt> {
    let half = FqExt::from_u64(2).inv();
    let w = d.tsz.trailing_zeros() as usize;
    r_i.iter().copied().chain(std::iter::repeat_n(half, w)).collect()
}

pub fn alpha_tensor_eval(r_l: &[FqExt], alpha: FqExt, limit: usize) -> FqExt {
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

pub fn build_bin_table(
    d: &Dims,
    h_bits: &[bool],
    bw: Option<&BlindWitness>,
    nz: Option<&Nizk1Ctx>,
) -> PackedBits {
    let mut zb = PackedBits::zeros(d.bin_pad * SLOTS);
    assert_eq!(h_bits.len(), d.h_cells, "the length of h_bits must be h_cells");
    for (idx, &b) in h_bits.iter().enumerate() {
        if b {
            zb.set(h_slot(d, idx));
        }
    }
    if let (Some(ctx), Some(bw)) = (nz, bw) {
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
                put_bits(&mut zb, start + i, &e.c);
            }
        }
    }
    zb
}

pub fn put_bits(zb: &mut PackedBits, k: usize, poly: &[Fq]) {
    let acc = poly.iter().fold(0u64, |a, &v| a | v.0);
    assert!(acc < crate::ring::W_RANGE_BASE, "a binary table may only hold bits, got a row containing {acc}");
    let base = k * SLOTS;
    for (c, &v) in poly.iter().enumerate() {
        zb.or_bit(base + c, v.0 as u32);
    }
}

pub fn build_merged_table(
    params: &HashParams,
    d: &Dims,
    wit: &HashWitness,
    quotients: &[Vec<Fq>],
    metas: &[crate::relation::ConstraintMeta],
    zb: &PackedBits,
) -> Vec<Fq> {
    let m = params.ell;
    let lay = d.merged;
    let mut t = vec![Fq::ZERO; lay.rows * SLOTS];

    assert_eq!(zb.len(), lay.bin_rows * SLOTS, "the size of the shadow zb must equal the binary block");
    for i in 0..zb.len() {
        if zb.get(i) {
            t[i] = Fq::ONE;
        }
    }

    for i in 0..params.num_groups() - 1 {
        for tt in 0..m {
            let base = lay.b_row_of(params, i, tt) * SLOTS;
            for (c, &v) in wit.b(i)[tt].c.iter().enumerate() {
                t[base + c] = v;
            }
        }
    }

    for meta in metas {
        if let Some(idx) = meta.t_index {
            assert!(
                quotients[idx].len() <= SLOTS,
                "the quotient has {} coefficients, more than the {SLOTS} slots of one row",
                quotients[idx].len()
            );
            let base = lay.quot_row_of(idx) * SLOTS;
            for (c, &v) in quotients[idx].iter().enumerate() {
                t[base + c] = v;
            }
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{bits_to_groups, eval_h};
    use crate::nizk1::{blind_statement, sample_blind, Nizk1Params, QueryTicket};
    use crate::relation::{compute_quotients, constraints, h_bits};
    use crate::rng::insecure_test_secret;
    use crate::transcript::SimpleRng;

    fn setup(n: usize, g: usize, ell: usize, seed: u64) -> (HashParams, Vec<usize>) {
        let params = HashParams::sample(seed, n, g, ell);
        let mut rng = SimpleRng::new(seed ^ 0x5EED);
        let bits: Vec<bool> = (0..n).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        (params, groups)
    }

    fn nz_of(params: &HashParams, seed: u64) -> Nizk1Params {
        Nizk1Params::sample(seed, params, 3, 2, 2)
    }

    #[test]
    fn merged_layout_blocks_do_not_overlap() {
        for &(n, g, ell) in &[(8usize, 2usize, 1usize), (16, 4, 5), (128, 4, 5), (128, 8, 5)] {
            let params = HashParams::dims_only(n, g, ell);
            for phase_b in [false, true] {
                let nz = phase_b.then(|| nz_of(&params, 1));
                let st = nz.as_ref().map(|nz| dummy_statement(&params, nz));
                let ctx = nz
                    .as_ref()
                    .zip(st.as_ref())
                    .map(|(nz, st)| Nizk1Ctx::new(&params, nz, st));
                let d = dims(&params, ctx.as_ref());
                let l = d.merged;
                let tag = format!("n={n} g={g} ell={ell} phase_b={phase_b}");
                assert!(l.bin_rows.is_power_of_two() && l.bin_rows >= 2, "binary block ({tag})");
                assert_eq!(l.bin_rows, d.bin_pad, "bin_pad == bin_rows ({tag})");
                assert!(d.bin_rows <= l.bin_rows, "the witness does not fit in the binary block ({tag})");
                assert_eq!(l.b_base, l.bin_rows, "the b segment directly follows the binary block ({tag})");
                assert_eq!(l.quot_base, l.b_base + num_b_rows(&params), "the quotient segment directly follows the b segment ({tag})");
                assert_eq!(
                    l.mask_base,
                    l.quot_base + num_quotients(&params, ctx.as_ref()),
                    "the mask row directly follows the quotient segment ({tag})"
                );
                assert!(l.mask_base + MASK_ROWS <= l.rows, "mask row overflow ({tag})");
                assert!(l.rows.is_power_of_two(), "rows is not a power of two ({tag})");
                assert!(!l.is_bin_row(l.b_base) && l.is_bin_row(l.bin_rows - 1));
                assert_eq!(d.nv, (l.rows * SLOTS).trailing_zeros() as usize);
                assert_eq!(d.nv_bin, (l.bin_rows * SLOTS).trailing_zeros() as usize);
                assert!(d.nv > d.nv_bin, "the binary block must not be the whole table ({tag})");
                let pt = vec![FqExt::ONE; d.nv_bin];
                let px = bin_prefix(&d, &pt);
                assert_eq!(px.len(), d.nv);
                assert!(px[..d.nv - d.nv_bin].iter().all(|z| *z == FqExt::ZERO));
            }
        }
    }

    #[test]
    fn dims_match_the_plan() {
        for &(w, nv_c, nv_u, nv_i, nv_h, rows, nv, bin_pad, nv_bin) in
            &[(4usize, 8usize, 12usize, 5usize, 9usize, 512usize, 18usize, 128usize, 16usize),
              (8, 7, 15, 4, 12, 512, 18, 128, 16)]
        {
            let params = HashParams::dims_only(128, w, 5);
            let nz = Nizk1Params::sample(
                1,
                &params,
                crate::nizk1::R_DIM,
                crate::nizk1::COM_N,
                crate::nizk1::W_SLACK,
            );
            let st = dummy_statement(&params, &nz);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));
            assert_eq!(d.nv_c, nv_c, "w={w} nv_c");
            assert_eq!(d.nv_u, nv_u, "w={w} nv_u");
            assert_eq!(d.nv_i, nv_i, "w={w} nv_i");
            assert_eq!(d.nv_h, nv_h, "w={w} nv_h");
            assert_eq!(d.merged.rows, rows, "w={w} merged.rows");
            assert_eq!(d.nv, nv, "w={w} nv");
            assert_eq!(d.bin_pad, bin_pad, "w={w} bin_pad");
            assert_eq!(d.nv_bin, nv_bin, "w={w} nv_bin");
        }
    }

    #[test]
    fn merged_table_roundtrip() {
        for &(n, g, ell, seed) in &[(8usize, 2usize, 2usize, 401u64), (16, 4, 5, 402)] {
            let (params, groups) = setup(n, g, ell, seed);
            let (bx, wit) = eval_h(&params, &groups);
            let nz = nz_of(&params, seed + 1);
            let bw = sample_blind(
                QueryTicket::insecure_for_tests(insecure_test_secret(seed + 2), 0),
                &params,
                &nz,
                &groups,
            );
            let (st, bq) = blind_statement(&params, &nz, &bx, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let d = dims(&params, Some(&ctx));
            let metas = constraints(&params, Some(&ctx));
            let qs = compute_quotients(&params, &wit, Some(&ctx), Some(&bq));
            let hb = h_bits(&params, &groups);
            let zb = build_bin_table(&d, &hb, Some(&bw), Some(&ctx));
            let tbl = build_merged_table(&params, &d, &wit, &qs, &metas, &zb);
            let l = d.merged;

            assert_eq!(tbl.len(), l.rows * SLOTS);
            for i in 0..l.bin_rows * SLOTS {
                assert_eq!(tbl[i] == Fq::ONE, zb.get(i), "binary slot {i}");
                assert!(tbl[i] == Fq::ZERO || tbl[i] == Fq::ONE);
            }
            for i in 0..params.num_groups() - 1 {
                for t in 0..ell {
                    let base = l.b_row_of(&params, i, t) * SLOTS;
                    assert_eq!(&tbl[base..base + N], &wit.b(i)[t].c[..], "b_{i}[{t}]");
                }
            }
            for meta in &metas {
                if let Some(idx) = meta.t_index {
                    let base = l.quot_row_of(idx) * SLOTS;
                    assert_eq!(&tbl[base..base + N - 1], &qs[idx][..], "quotient {idx}");
                    assert_eq!(tbl[base + N - 1], Fq::ZERO, "slot N−1 of quotient {idx}");
                }
            }
            for i in l.mask_base * SLOTS..l.rows * SLOTS {
                assert_eq!(tbl[i], Fq::ZERO, "slot {i} should not be written by build_merged_table");
            }
        }
    }

    #[test]
    fn every_quotient_leaves_the_last_slot_free() {
        for &(n, g, ell, seed) in &[(8usize, 2usize, 1usize, 501u64), (12, 2, 3, 502), (16, 4, 5, 503)] {
            let (params, groups) = setup(n, g, ell, seed);
            let (bx, wit) = eval_h(&params, &groups);
            let nz = nz_of(&params, seed + 1);
            let bw = sample_blind(
                QueryTicket::insecure_for_tests(insecure_test_secret(seed + 2), 0),
                &params,
                &nz,
                &groups,
            );
            let (st, bq) = blind_statement(&params, &nz, &bx, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            for (tag, qs) in [
                ("Phase A", compute_quotients(&params, &wit, None, None)),
                ("Phase B", compute_quotients(&params, &wit, Some(&ctx), Some(&bq))),
            ] {
                assert!(!qs.is_empty(), "{tag} has no quotients at all");
                for (i, q) in qs.iter().enumerate() {
                    assert_eq!(q.len(), N - 1, "quotient {i} of {tag} has {} coefficients", q.len());
                }
            }
        }
    }

    #[test]
    fn h_open_matches_full_pcs_open_at_the_mapped_point() {
        let mut saw_per_full = false;
        let mut saw_per_short = false;
        for &(n, g, ell, seed) in
            &[(8usize, 2usize, 1usize, 601u64), (8, 4, 2, 602), (16, 8, 1, 603), (128, 4, 3, 604)]
        {
            let (params, groups) = setup(n, g, ell, seed);
            let hb = crate::relation::h_bits(&params, &groups);
            let (bx, _) = eval_h(&params, &groups);
            let nz = nz_of(&params, seed + 1);
            let bw = sample_blind(
                QueryTicket::insecure_for_tests(insecure_test_secret(seed + 2), 0),
                &params,
                &nz,
                &groups,
            );
            let (st, _) = blind_statement(&params, &nz, &bx, &bw);
            let ctx = Nizk1Ctx::new(&params, &nz, &st);
            let mut rng = SimpleRng::new(seed ^ 0xBEEF);

            for phase_b in [true, false] {
                let c = phase_b.then_some(&ctx);
                let d = dims(&params, c);
                let zb = build_bin_table(&d, &hb, phase_b.then_some(&bw), c);
                if d.hpack_per == SLOTS {
                    saw_per_full = true;
                } else {
                    saw_per_short = true;
                }
                for (idx, &b) in hb.iter().enumerate() {
                    assert_eq!(h_bit(&zb, &d, idx), b, "bit {idx} of H is at the wrong position");
                }
                let h_open = |pt: &[FqExt]| {
                    let tbl: Vec<FqExt> = (0..d.h_cells)
                        .map(|i| if h_bit(&zb, &d, i) { FqExt::ONE } else { FqExt::ZERO })
                        .collect();
                    crate::mle::mle_eval(&tbl, pt)
                };
                for _ in 0..3 {
                    let pt: Vec<FqExt> = (0..d.nv_h).map(|_| rng.next_fq4()).collect();
                    assert_eq!(
                        h_open(&pt),
                        crate::pcs::open(&zb, &hpack_point(&d, &pt)),
                        "n={n} g={g} phase_b={phase_b}"
                    );
                }
                let r_i: Vec<FqExt> = (0..d.nv_i).map(|_| rng.next_fq4()).collect();
                let hp = half_point(&d, &r_i);
                assert_eq!(hp.len(), d.nv_h);
                assert_eq!(h_open(&hp), crate::pcs::open(&zb, &hpack_point(&d, &hp)));
            }
        }
        assert!(saw_per_full && saw_per_short, "the two hpack_per branches were not both covered");
    }

    #[test]
    fn half_point_identity_matches_row_sums() {
        for &(n, g, ell, seed) in
            &[(8usize, 2usize, 1usize, 701u64), (16, 4, 2, 702), (12, 2, 3, 703)]
        {
            let (params, groups) = setup(n, g, ell, seed);
            let hb = crate::relation::h_bits(&params, &groups);
            let d = dims(&params, None);
            let zb = build_bin_table(&d, &hb, None, None);

            let p_tbl: Vec<FqExt> = (0..d.g_pad)
                .map(|i| {
                    let cnt = (0..d.tsz).filter(|&v| h_bit(&zb, &d, i * d.hv + v)).count();
                    FqExt::from_u64(cnt as u64)
                })
                .collect();
            for (i, &p) in p_tbl.iter().enumerate() {
                assert_eq!(p, FqExt::from_u64((i < params.num_groups()) as u64), "row sum {i}");
            }
            let two_w = FqExt::from_u64(2).pow(g as u128);
            let mut rng = SimpleRng::new(seed ^ 0xFACE);
            for _ in 0..3 {
                let r_i: Vec<FqExt> = (0..d.nv_i).map(|_| rng.next_fq4()).collect();
                let lhs = two_w * crate::pcs::open(&zb, &hpack_point(&d, &half_point(&d, &r_i)));
                assert_eq!(lhs, crate::mle::mle_eval(&p_tbl, &r_i), "n={n} g={g}");
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

    #[test]
    #[should_panic(expected = "a binary table may only hold bits")]
    fn put_bits_rejects_non_bit() {
        let mut zb = PackedBits::zeros(2 * SLOTS);
        let mut poly = vec![Fq::ZERO; SLOTS];
        poly[3] = Fq::new(2);
        put_bits(&mut zb, 0, &poly);
    }

    fn dummy_statement(params: &HashParams, nz: &Nizk1Params) -> crate::nizk1::BlindStatement {
        use crate::ring::RingElem;
        crate::nizk1::BlindStatement {
            j: 0,
            c_x: vec![RingElem::zero(); params.ell],
            c_r: vec![RingElem::zero(); nz.com_r.out_len()],
            d_x: vec![RingElem::zero(); nz.com_x.out_len()],
        }
    }
}
