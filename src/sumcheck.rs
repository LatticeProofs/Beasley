
use crate::bits::PackedBits;
use crate::ext_field::FqExt;
use crate::field::{Fq, Q};
use crate::mle::eq_table;
use crate::transcript::Transcript;

pub struct SumcheckProof {
    pub rounds: Vec<Vec<FqExt>>,
}

pub fn prove(
    mut tables: Vec<Vec<FqExt>>,
    deg: usize,
    combine: &dyn Fn(&[FqExt]) -> FqExt,
    mut mask: Option<&mut Masker>,
    tr: &mut Transcript,
) -> (SumcheckProof, Vec<FqExt>, Vec<FqExt>) {
    let len = tables[0].len();
    assert!(len.is_power_of_two());
    for t in &tables {
        assert_eq!(t.len(), len);
    }
    let nv = len.trailing_zeros() as usize;

    let mut rounds = Vec::with_capacity(nv);
    let mut r_point = Vec::with_capacity(nv);
    let mut vals_buf = vec![FqExt::ZERO; tables.len()];

    let mask_pts: Vec<u64> = (0..=deg as u64).collect();
    for round in 0..nv {
        let half = tables[0].len() / 2;
        let mut evals = Vec::with_capacity(deg + 1);
        for s in 0..=deg {
            let s_f = FqExt::from_u64(s as u64);
            let mut acc = FqExt::ZERO;
            for cell in 0..half {
                for (k, t) in tables.iter().enumerate() {
                    let lo = t[cell];
                    let hi = t[cell + half];
                    vals_buf[k] = lo + s_f * (hi - lo);
                }
                acc = acc + combine(&vals_buf);
            }
            evals.push(acc);
        }
        if let Some(m) = mask.as_deref_mut() {
            for (e, v) in evals.iter_mut().zip(m.round_plain(round, &mask_pts)) {
                *e = *e + v;
            }
        }
        for &e in &evals {
            tr.absorb_fq4(e);
        }
        let r = tr.challenge_fq4();
        if let Some(m) = mask.as_deref_mut() {
            m.fold(round, r);
        }
        r_point.push(r);
        for t in tables.iter_mut() {
            for i in 0..half {
                t[i] = t[i] + r * (t[i + half] - t[i]);
            }
            t.truncate(half);
        }
        rounds.push(evals);
    }

    let finals: Vec<FqExt> = tables.iter().map(|t| t[0]).collect();
    (SumcheckProof { rounds }, r_point, finals)
}

pub fn verify(
    claim: FqExt,
    nv: usize,
    deg: usize,
    proof: &SumcheckProof,
    tr: &mut Transcript,
) -> Option<(FqExt, Vec<FqExt>)> {
    if proof.rounds.len() != nv {
        return None;
    }
    let mut expect = claim;
    let mut r_point = Vec::with_capacity(nv);
    for evals in &proof.rounds {
        if evals.len() != deg + 1 {
            return None;
        }
        if evals[0] + evals[1] != expect {
            return None;
        }
        for &e in evals {
            tr.absorb_fq4(e);
        }
        let r = tr.challenge_fq4();
        r_point.push(r);
        expect = lagrange_eval(evals, r);
    }
    Some((expect, r_point))
}

pub fn prove_eq_bitcheck(
    tau: &[FqExt],
    table: &PackedBits,
    scratch: &mut Vec<FqExt>,
    mut mask: Option<&mut Masker>,
    tr: &mut Transcript,
) -> (SumcheckProof, Vec<FqExt>, FqExt) {
    let nv = tau.len();
    assert_eq!(table.len(), 1 << nv);
    let mut rounds = Vec::with_capacity(nv);
    let mut r_point = Vec::with_capacity(nv);
    let ext = scratch;
    ext.clear();

    let half0 = 1usize << (nv - 1);
    let eqsuf0 = eq_table(&tau[1..]);
    let mut acc = FqExt::ZERO;
    if half0 >= 64 {
        let words = half0 / 64;
        let hi_word_off = half0 / 64;
        for w in 0..words {
            let mut diff = table.word(w) ^ table.word(hi_word_off + w);
            while diff != 0 {
                let b = diff.trailing_zeros() as usize;
                acc = acc + eqsuf0[w * 64 + b];
                diff &= diff - 1;
            }
        }
    } else {
        for cell in 0..half0 {
            if table.get(cell) != table.get(cell + half0) {
                acc = acc + eqsuf0[cell];
            }
        }
    }
    let (mut h0, mut h2) = (FqExt::ZERO, Fq::new(2) * acc);
    if let Some(m) = mask.as_deref_mut() {
        let v = m.round_eq(0, tau, &[0, 2]);
        h0 = h0 + v[0];
        h2 = h2 + v[1];
    }
    tr.absorb_fq4(h0);
    tr.absorb_fq4(h2);
    let r = tr.challenge_fq4();
    if let Some(m) = mask.as_deref_mut() {
        m.fold(0, r);
    }
    r_point.push(r);
    ext.extend((0..half0).map(|i| {
        let lo = FqExt::from_fq(if table.get(i) { Fq::ONE } else { Fq::ZERO });
        let hi = FqExt::from_fq(if table.get(i + half0) { Fq::ONE } else { Fq::ZERO });
        lo + r * (hi - lo)
    }));
    rounds.push(vec![h0, h2]);

    let mid = nv / 2;
    let eb = crate::simd::eq_table_mont(&tau[mid..]);
    let shift = eb.len().trailing_zeros() as usize;
    let mut sa: Vec<crate::simd::SoaFqExt> = Vec::with_capacity(mid + 1);
    for k in 0..=mid {
        sa.push(crate::simd::eq_table_mont(&tau[mid - k..mid]));
    }
    let mut soa = crate::simd::SoaFqExt::from_aos(ext);

    for j in 1..nv {
        let half = 1usize << (nv - 1 - j);
        let (mut h0, mut h2) = if j < mid {
            crate::simd::bitcheck_evals_split(&soa, &sa[mid - 1 - j], &eb, half)
        } else {
            let eqsuf = crate::simd::eq_table_mont(&tau[j + 1..]);
            crate::simd::bitcheck_evals(&soa, &eqsuf, half)
        };
        if let Some(m) = mask.as_deref_mut() {
            let v = m.round_eq(j, tau, &[0, 2]);
            h0 = h0 + v[0];
            h2 = h2 + v[1];
        }
        tr.absorb_fq4(h0);
        tr.absorb_fq4(h2);
        let r = tr.challenge_fq4();
        if let Some(m) = mask.as_deref_mut() {
            m.fold(j, r);
        }
        r_point.push(r);
        crate::simd::fold(&mut soa, half, r);
        rounds.push(vec![h0, h2]);
    }
    let _ = shift;
    let final_val = soa.get(0);
    (SumcheckProof { rounds }, r_point, final_val)
}

pub fn verify_eq_bitcheck(
    claim: FqExt,
    tau: &[FqExt],
    proof: &SumcheckProof,
    tr: &mut Transcript,
) -> Option<(FqExt, Vec<FqExt>)> {
    let nv = tau.len();
    if proof.rounds.len() != nv {
        return None;
    }
    let mut a = FqExt::ONE;
    let mut c = claim;
    let mut r_point = Vec::with_capacity(nv);
    for (j, evals) in proof.rounds.iter().enumerate() {
        if evals.len() != 2 {
            return None;
        }
        let (h0, h2) = (evals[0], evals[1]);
        let t = tau[j];
        let h1 = (c * a.inv() - (FqExt::ONE - t) * h0) * t.inv();
        tr.absorb_fq4(h0);
        tr.absorb_fq4(h2);
        let r = tr.challenge_fq4();
        r_point.push(r);
        let hr = lagrange_eval(&[h0, h1, h2], r);
        a = a * (t * r + (FqExt::ONE - t) * (FqExt::ONE - r));
        c = a * hr;
    }
    Some((c, r_point))
}

pub fn prove_product2(
    a_base: &[Fq],
    mut b: Vec<FqExt>,
    mut mask: Option<&mut Masker>,
    tr: &mut Transcript,
) -> (SumcheckProof, Vec<FqExt>) {
    let nv = b.len().trailing_zeros() as usize;
    assert_eq!(a_base.len(), b.len());
    let mut rounds = Vec::with_capacity(nv);
    let mut r_point = Vec::with_capacity(nv);
    let mut a_ext: Vec<FqExt> = Vec::new();

    for j in 0..nv {
        let half = 1usize << (nv - 1 - j);
        let (mut g0, mut g2) = (FqExt::ZERO, FqExt::ZERO);
        if j == 0 {
            for cell in 0..half {
                let alo = a_base[cell];
                let ahi = a_base[cell + half];
                let blo = b[cell];
                let bhi = b[cell + half];
                g0 = g0 + alo * blo;
                let a2 = ahi + ahi - alo;
                let b2 = bhi + (bhi - blo);
                g2 = g2 + a2 * b2;
            }
        } else {
            for cell in 0..half {
                let alo = a_ext[cell];
                let ahi = a_ext[cell + half];
                let blo = b[cell];
                let bhi = b[cell + half];
                g0 = g0 + alo * blo;
                let a2 = ahi + (ahi - alo);
                let b2 = bhi + (bhi - blo);
                g2 = g2 + a2 * b2;
            }
        }
        if let Some(m) = mask.as_deref_mut() {
            let v = m.round_plain(j, &[0, 2]);
            g0 = g0 + v[0];
            g2 = g2 + v[1];
        }
        tr.absorb_fq4(g0);
        tr.absorb_fq4(g2);
        let r = tr.challenge_fq4();
        if let Some(m) = mask.as_deref_mut() {
            m.fold(j, r);
        }
        r_point.push(r);
        if j == 0 {
            a_ext = (0..half)
                .map(|i| {
                    let lo = FqExt::from_fq(a_base[i]);
                    lo + r * (FqExt::from_fq(a_base[i + half]) - lo)
                })
                .collect();
        } else {
            for i in 0..half {
                a_ext[i] = a_ext[i] + r * (a_ext[i + half] - a_ext[i]);
            }
            a_ext.truncate(half);
        }
        for i in 0..half {
            b[i] = b[i] + r * (b[i + half] - b[i]);
        }
        b.truncate(half);
        rounds.push(vec![g0, g2]);
    }
    (SumcheckProof { rounds }, r_point)
}

pub fn prove_product2_factored(
    a_base: &PackedBits,
    mut lg: Vec<FqExt>,
    apow: &[FqExt],
    scratch: &mut Vec<FqExt>,
    tr: &mut Transcript,
) -> (SumcheckProof, Vec<FqExt>, FqExt) {
    let s_len = apow.len();
    assert!(s_len.is_power_of_two() && lg.len().is_power_of_two());
    assert_eq!(a_base.len(), lg.len() * s_len);
    let nv_k = lg.len().trailing_zeros() as usize;
    let nv = nv_k + s_len.trailing_zeros() as usize;
    let mut rounds = Vec::with_capacity(nv);
    let mut r_point = Vec::with_capacity(nv);
    let a_ext = scratch;
    a_ext.clear();

    for j in 0..nv_k {
        let half_k = lg.len() / 2;
        let (mut g0, mut g2) = (FqExt::ZERO, FqExt::ZERO);
        for k in 0..half_k {
            let (mut row0, mut row2) = (FqExt::ZERO, FqExt::ZERO);
            if j == 0 {
                let lo_base = k * s_len;
                let hi_base = (k + half_k) * s_len;
                for l in 0..s_len {
                    let lo = a_base.get(lo_base + l);
                    let hi = a_base.get(hi_base + l);
                    if lo {
                        row0 = row0 + apow[l];
                    }
                    let a2 = match (hi, lo) {
                        (false, false) => Fq::ZERO,
                        (false, true) => Fq::new(Q - 1),
                        (true, false) => Fq::new(2),
                        (true, true) => Fq::ONE,
                    };
                    if a2 != Fq::ZERO {
                        row2 = row2 + a2 * apow[l];
                    }
                }
            } else {
                let base_lo = k * s_len;
                let base_hi = (k + half_k) * s_len;
                for l in 0..s_len {
                    let lo = a_ext[base_lo + l];
                    let hi = a_ext[base_hi + l];
                    row0 = row0 + lo * apow[l];
                    row2 = row2 + (hi + (hi - lo)) * apow[l];
                }
            }
            let lg2 = lg[k + half_k] + (lg[k + half_k] - lg[k]);
            g0 = g0 + lg[k] * row0;
            g2 = g2 + lg2 * row2;
        }
        tr.absorb_fq4(g0);
        tr.absorb_fq4(g2);
        let r = tr.challenge_fq4();
        r_point.push(r);
        let half_cells = half_k * s_len;
        if j == 0 {
            a_ext.extend((0..half_cells).map(|i| {
                let lo = FqExt::from_fq(if a_base.get(i) { Fq::ONE } else { Fq::ZERO });
                let hi = FqExt::from_fq(if a_base.get(i + half_cells) { Fq::ONE } else { Fq::ZERO });
                lo + r * (hi - lo)
            }));
        } else {
            for i in 0..half_cells {
                a_ext[i] = a_ext[i] + r * (a_ext[i + half_cells] - a_ext[i]);
            }
            a_ext.truncate(half_cells);
        }
        for k in 0..half_k {
            lg[k] = lg[k] + r * (lg[k + half_k] - lg[k]);
        }
        lg.truncate(half_k);
        rounds.push(vec![g0, g2]);
    }

    if nv_k == 0 {
        a_ext.extend((0..a_base.len()).map(|i| {
            FqExt::from_fq(if a_base.get(i) { Fq::ONE } else { Fq::ZERO })
        }));
    }
    let mut b: Vec<FqExt> = apow.iter().map(|&p| lg[0] * p).collect();
    for _ in nv_k..nv {
        let half = a_ext.len() / 2;
        let (mut g0, mut g2) = (FqExt::ZERO, FqExt::ZERO);
        for cell in 0..half {
            let alo = a_ext[cell];
            let ahi = a_ext[cell + half];
            let blo = b[cell];
            let bhi = b[cell + half];
            g0 = g0 + alo * blo;
            g2 = g2 + (ahi + (ahi - alo)) * (bhi + (bhi - blo));
        }
        tr.absorb_fq4(g0);
        tr.absorb_fq4(g2);
        let r = tr.challenge_fq4();
        r_point.push(r);
        for i in 0..half {
            a_ext[i] = a_ext[i] + r * (a_ext[i + half] - a_ext[i]);
            b[i] = b[i] + r * (b[i + half] - b[i]);
        }
        a_ext.truncate(half);
        b.truncate(half);
        rounds.push(vec![g0, g2]);
    }
    let final_val = a_ext[0];
    (SumcheckProof { rounds }, r_point, final_val)
}

pub fn verify_product2(
    claim: FqExt,
    nv: usize,
    proof: &SumcheckProof,
    tr: &mut Transcript,
) -> Option<(FqExt, Vec<FqExt>)> {
    if proof.rounds.len() != nv {
        return None;
    }
    let mut c = claim;
    let mut r_point = Vec::with_capacity(nv);
    for evals in &proof.rounds {
        if evals.len() != 2 {
            return None;
        }
        let (g0, g2) = (evals[0], evals[1]);
        let g1 = c - g0;
        tr.absorb_fq4(g0);
        tr.absorb_fq4(g2);
        let r = tr.challenge_fq4();
        r_point.push(r);
        c = lagrange_eval(&[g0, g1, g2], r);
    }
    Some((c, r_point))
}

#[allow(clippy::too_many_arguments)]
pub fn prove_batched_w(
    zw: &PackedBits,
    mut lg: Vec<FqExt>,
    apow: &[FqExt],
    tau0: &[FqExt],
    lambda: FqExt,
    scratch: &mut Vec<FqExt>,
    mut mask: Option<&mut Masker>,
    tr: &mut Transcript,
) -> (SumcheckProof, Vec<FqExt>, FqExt) {
    let nv_w = tau0.len();
    let nv_k = lg.len().trailing_zeros() as usize;
    let s_len = apow.len();
    assert_eq!(zw.len(), 1 << nv_w);
    assert_eq!(lg.len() * s_len, 1 << nv_w);

    let mid = nv_k;
    let eb = eq_table(&tau0[mid..]);
    let eb_mask = eb.len() - 1;
    let eb_shift = eb.len().trailing_zeros() as usize;
    let mut sa: Vec<Vec<FqExt>> = Vec::with_capacity(mid + 1);
    for k in 0..=mid {
        sa.push(eq_table(&tau0[mid - k..mid]));
    }

    let apow_soa = crate::simd::SoaFqExt::from_aos(apow);
    let eb_soa = crate::simd::SoaFqExt::from_aos(&eb);
    let mut wsoa: Option<crate::simd::SoaFqExt> = None;

    let w = scratch;
    w.clear();
    let mut r_point = Vec::with_capacity(nv_w);
    let mut rounds = Vec::with_capacity(nv_w);
    let mut b: Vec<FqExt> = Vec::new();
    let mut eq_prefix = FqExt::ONE;

    for round in 0..nv_w {
        let half = 1usize << (nv_w - 1 - round);
        let eqsuf_mat: Vec<FqExt>;
        let ea: &[FqExt] = if round < mid {
            &sa[mid - 1 - round]
        } else {
            eqsuf_mat = eq_table(&tau0[round + 1..]);
            &eqsuf_mat
        };
        let split = round < mid;
        let eq_rem = |cell: usize| -> FqExt {
            if split {
                ea[cell >> eb_shift] * eb[cell & eb_mask]
            } else {
                ea[cell]
            }
        };
        let read_w = |cell: usize, w: &[FqExt]| -> (FqExt, FqExt, FqExt) {
            let (lo, hi) = if round == 0 {
                (
                    FqExt::from_fq(if zw.get(cell) { Fq::ONE } else { Fq::ZERO }),
                    FqExt::from_fq(if zw.get(cell + half) { Fq::ONE } else { Fq::ZERO }),
                )
            } else {
                (w[cell], w[cell + half])
            };
            let d = hi - lo;
            let w2 = hi + d;
            (lo, w2, w2 + d)
        };

        let (mut gf2_0, mut gf2_2, mut gf2_3) = (FqExt::ZERO, FqExt::ZERO, FqExt::ZERO);
        let (mut q3_0, mut q3_2, mut q3_3) = (FqExt::ZERO, FqExt::ZERO, FqExt::ZERO);

        if round < nv_k && round >= 1 {
            let ea_soa = crate::simd::SoaFqExt::from_aos(ea);
            let (gf2, q3) = crate::simd::batched_phase_a_round(
                wsoa.as_ref().unwrap(),
                &apow_soa,
                &ea_soa,
                &eb_soa,
                &lg,
                half,
                s_len,
            );
            gf2_0 = gf2[0];
            gf2_2 = gf2[1];
            gf2_3 = gf2[2];
            q3_0 = q3[0];
            q3_2 = q3[1];
            q3_3 = q3[2];
        } else if round < nv_k {
            let ea_soa = crate::simd::SoaFqExt::from_aos(ea);
            let (gf2, q3) = crate::simd::batched_phase_a_round0(
                zw,
                &apow_soa,
                &ea_soa,
                &eb_soa,
                &lg,
                half,
                s_len,
            );
            gf2_0 = gf2[0];
            gf2_2 = gf2[1];
            gf2_3 = gf2[2];
            q3_0 = q3[0];
            q3_2 = q3[1];
            q3_3 = q3[2];
        } else {
            for cell in 0..half {
                let (w0, w2, w3) = read_w(cell, w);
                let blo = b[cell];
                let bhi = b[cell + half];
                let bd = bhi - blo;
                gf2_0 = gf2_0 + blo * w0;
                gf2_2 = gf2_2 + (bhi + bd) * w2;
                gf2_3 = gf2_3 + (bhi + bd + bd) * w3;
                let e = eq_rem(cell);
                q3_0 = q3_0 + e * (w0 * (w0 - FqExt::ONE));
                q3_2 = q3_2 + e * (w2 * (w2 - FqExt::ONE));
                q3_3 = q3_3 + e * (w3 * (w3 - FqExt::ONE));
            }
        }

        let t = tau0[round];
        let eq1_0 = FqExt::ONE - t;
        let slope = t + t - FqExt::ONE;
        let eq1_2 = eq1_0 + slope + slope;
        let eq1_3 = eq1_2 + slope;
        let mut g0 = lambda * gf2_0 + eq_prefix * eq1_0 * q3_0;
        let mut g2 = lambda * gf2_2 + eq_prefix * eq1_2 * q3_2;
        let mut g3 = lambda * gf2_3 + eq_prefix * eq1_3 * q3_3;
        if let Some(m) = mask.as_deref_mut() {
            let v = m.round_plain(round, &[0, 2, 3]);
            g0 = g0 + v[0];
            g2 = g2 + v[1];
            g3 = g3 + v[2];
        }
        tr.absorb_fq4(g0);
        tr.absorb_fq4(g2);
        tr.absorb_fq4(g3);
        let r = tr.challenge_fq4();
        if let Some(m) = mask.as_deref_mut() {
            m.fold(round, r);
        }
        r_point.push(r);
        eq_prefix = eq_prefix * (t * r + (FqExt::ONE - t) * (FqExt::ONE - r));

        if round == 0 {
            if nv_k >= 2 {
                wsoa = Some(crate::simd::fold_bits_to_soa(zw, half, r));
            } else {
                w.extend((0..half).map(|cell| {
                    let lo = FqExt::from_fq(if zw.get(cell) { Fq::ONE } else { Fq::ZERO });
                    let hi = FqExt::from_fq(if zw.get(cell + half) { Fq::ONE } else { Fq::ZERO });
                    lo + r * (hi - lo)
                }));
            }
        } else if round < nv_k {
            let ws = wsoa.as_mut().unwrap();
            crate::simd::fold(ws, half, r);
            if round + 1 == nv_k {
                let ws = wsoa.take().unwrap();
                w.clear();
                w.extend((0..half).map(|i| ws.get(i)));
            }
        } else {
            for i in 0..half {
                w[i] = w[i] + r * (w[i + half] - w[i]);
            }
            w.truncate(half);
        }
        if round < nv_k {
            let half_k = lg.len() / 2;
            for k in 0..half_k {
                lg[k] = lg[k] + r * (lg[k + half_k] - lg[k]);
            }
            lg.truncate(half_k);
            if round + 1 == nv_k {
                b = apow.iter().map(|&p| lg[0] * p).collect();
            }
        } else {
            let hb = b.len() / 2;
            for i in 0..hb {
                b[i] = b[i] + r * (b[i + hb] - b[i]);
            }
            b.truncate(hb);
        }
        rounds.push(vec![g0, g2, g3]);
    }
    let open_w = w[0];
    (SumcheckProof { rounds }, r_point, open_w)
}

pub fn verify_batched_w(
    claim: FqExt,
    nv_w: usize,
    proof: &SumcheckProof,
    tr: &mut Transcript,
) -> Option<(FqExt, Vec<FqExt>)> {
    if proof.rounds.len() != nv_w {
        return None;
    }
    let mut c = claim;
    let mut r_point = Vec::with_capacity(nv_w);
    for evals in &proof.rounds {
        if evals.len() != 3 {
            return None;
        }
        let (g0, g2, g3) = (evals[0], evals[1], evals[2]);
        let g1 = c - g0;
        tr.absorb_fq4(g0);
        tr.absorb_fq4(g2);
        tr.absorb_fq4(g3);
        let r = tr.challenge_fq4();
        r_point.push(r);
        c = lagrange_eval(&[g0, g1, g2, g3], r);
    }
    Some((c, r_point))
}

pub fn lagrange_eval(evals: &[FqExt], x: FqExt) -> FqExt {
    let deg = evals.len() - 1;
    let mut acc = FqExt::ZERO;
    for i in 0..=deg {
        let mut num = FqExt::ONE;
        let mut den = FqExt::ONE;
        let xi = FqExt::from_u64(i as u64);
        for k in 0..=deg {
            if k == i {
                continue;
            }
            let xk = FqExt::from_u64(k as u64);
            num = num * (x - xk);
            den = den * (xi - xk);
        }
        acc = acc + evals[i] * num * den.inv();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mle::mle_eval;
    use crate::transcript::SimpleRng;

    #[test]
    fn product_sumcheck_roundtrip() {
        let mut rng = SimpleRng::new(5);
        let nv = 6;
        let a: Vec<FqExt> = (0..1 << nv).map(|_| rng.next_fq4()).collect();
        let b: Vec<FqExt> = (0..1 << nv).map(|_| rng.next_fq4()).collect();
        let claim = a.iter().zip(&b).fold(FqExt::ZERO, |s, (&x, &y)| s + x * y);

        let combine = |v: &[FqExt]| v[0] * v[1];
        let mut tr_p = Transcript::new("sc-test");
        let (proof, r_p, finals) = prove(vec![a.clone(), b.clone()], 2, &combine, None, &mut tr_p);

        let mut tr_v = Transcript::new("sc-test");
        let (expect, r_v) = verify(claim, nv, 2, &proof, &mut tr_v).expect("should verify");
        assert_eq!(r_p, r_v);
        assert_eq!(finals[0], mle_eval(&a, &r_v));
        assert_eq!(finals[1], mle_eval(&b, &r_v));
        assert_eq!(expect, finals[0] * finals[1]);
    }

    #[test]
    fn wrong_claim_rejected() {
        let mut rng = SimpleRng::new(6);
        let nv = 4;
        let a: Vec<FqExt> = (0..1 << nv).map(|_| rng.next_fq4()).collect();
        let b: Vec<FqExt> = (0..1 << nv).map(|_| rng.next_fq4()).collect();
        let claim = a.iter().zip(&b).fold(FqExt::ZERO, |s, (&x, &y)| s + x * y);

        let combine = |v: &[FqExt]| v[0] * v[1];
        let mut tr_p = Transcript::new("sc-test2");
        let (proof, _, _) = prove(vec![a, b], 2, &combine, None, &mut tr_p);

        let mut tr_v = Transcript::new("sc-test2");
        assert!(verify(claim + FqExt::ONE, nv, 2, &proof, &mut tr_v).is_none());
    }

    #[test]
    fn lagrange_recovers_polynomial() {
        let p = |x: u64| FqExt::from_u64(3 * x * x + 2 * x + 7);
        let evals = vec![p(0), p(1), p(2)];
        assert_eq!(lagrange_eval(&evals, FqExt::from_u64(5)), p(5));
    }

    #[test]
    fn eq_bitcheck_roundtrip() {
        use crate::mle::{eq_eval, mle_eval};
        let mut rng = SimpleRng::new(7);
        let nv = 6;
        let table: Vec<Fq> =
            (0..1 << nv).map(|_| if rng.next_bool() { Fq::ONE } else { Fq::ZERO }).collect();
        let tau: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();

        let tb: Vec<u8> = table.iter().map(|x| x.0 as u8).collect();
        let table_p = PackedBits::from_bits(&tb, table.len());
        let mut tr_p = Transcript::new("bc-test");
        let (proof, r_p, _) = prove_eq_bitcheck(&tau, &table_p, &mut Vec::new(), None, &mut tr_p);

        let mut tr_v = Transcript::new("bc-test");
        let (expect, r_v) =
            verify_eq_bitcheck(FqExt::ZERO, &tau, &proof, &mut tr_v).expect("verify");
        assert_eq!(r_p, r_v);
        let lifted: Vec<FqExt> = table.iter().map(|&x| FqExt::from_fq(x)).collect();
        let t_r = mle_eval(&lifted, &r_v);
        assert_eq!(expect, eq_eval(&tau, &r_v) * t_r * (t_r - FqExt::ONE));
    }

    #[test]
    fn eq_bitcheck_rejects_tampered_round() {
        let mut rng = SimpleRng::new(8);
        let nv = 5;
        let table: Vec<Fq> =
            (0..1 << nv).map(|_| if rng.next_bool() { Fq::ONE } else { Fq::ZERO }).collect();
        let tau: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
        let tb: Vec<u8> = table.iter().map(|x| x.0 as u8).collect();
        let table_p = PackedBits::from_bits(&tb, table.len());
        let mut tr_p = Transcript::new("bc-test2");
        let (mut proof, _, _) = prove_eq_bitcheck(&tau, &table_p, &mut Vec::new(), None, &mut tr_p);
        proof.rounds[2][1] = proof.rounds[2][1] + FqExt::ONE;

        use crate::mle::{eq_eval, mle_eval};
        let mut tr_v = Transcript::new("bc-test2");
        let (expect, r_v) =
            verify_eq_bitcheck(FqExt::ZERO, &tau, &proof, &mut tr_v).expect("rounds well-formed");
        let lifted: Vec<FqExt> = table.iter().map(|&x| FqExt::from_fq(x)).collect();
        let t_r = mle_eval(&lifted, &r_v);
        assert_ne!(expect, eq_eval(&tau, &r_v) * t_r * (t_r - FqExt::ONE));
    }

    #[test]
    fn product2_factored_matches_materialized() {
        let mut rng = SimpleRng::new(11);
        let nv_k = 3;
        let s_len = 8usize;
        let total = (1 << nv_k) * s_len;
        let a_bits: Vec<u8> = (0..total).map(|_| rng.next_bool() as u8).collect();
        let a: Vec<Fq> = a_bits.iter().map(|&b| Fq::new(b as u64)).collect();
        let a_p = PackedBits::from_bits(&a_bits, total);
        let lg: Vec<FqExt> = (0..1 << nv_k).map(|_| rng.next_fq4()).collect();
        let apow: Vec<FqExt> = (0..s_len).map(|_| rng.next_fq4()).collect();
        let b: Vec<FqExt> = (0..total).map(|i| lg[i / s_len] * apow[i % s_len]).collect();

        let mut tr1 = Transcript::new("p2f");
        let (p1, r1) = prove_product2(&a, b, None, &mut tr1);
        let mut tr2 = Transcript::new("p2f");
        let (p2, r2, _) = prove_product2_factored(&a_p, lg, &apow, &mut Vec::new(), &mut tr2);
        assert_eq!(r1, r2);
        assert_eq!(p1.rounds, p2.rounds);
    }

    #[test]
    fn batched_w_matches_bruteforce() {
        use crate::mle::{eq_eval, mle_eval};
        let mut rng = SimpleRng::new(77);
        let nv_k = 2usize;
        let nv_l = 2usize;
        let nv_w = nv_k + nv_l;
        let s_len = 1 << nv_l;
        let total = 1 << nv_w;
        let w_bits: Vec<u8> = (0..total).map(|_| rng.next_bool() as u8).collect();
        let w_fq4: Vec<FqExt> =
            w_bits.iter().map(|&b| FqExt::from_u64(b as u64)).collect();
        let zw = PackedBits::from_bits(&w_bits, total);
        let lg: Vec<FqExt> = (0..1 << nv_k).map(|_| rng.next_fq4()).collect();
        let apow: Vec<FqExt> = (0..s_len).map(|_| rng.next_fq4()).collect();
        let tau0: Vec<FqExt> = (0..nv_w).map(|_| rng.next_fq4()).collect();
        let lambda = rng.next_fq4();

        let mut claim2 = FqExt::ZERO;
        for cell in 0..total {
            claim2 = claim2 + lg[cell / s_len] * apow[cell % s_len] * w_fq4[cell];
        }

        let mut tr_p = Transcript::new("bw");
        let (proof, r, open_w) =
            prove_batched_w(&zw, lg.clone(), &apow, &tau0, lambda, &mut Vec::new(), None, &mut tr_p);

        let mut wb = w_fq4.clone();
        let mut lgb = lg.clone();
        let mut bb: Vec<FqExt> = Vec::new();
        let mut tr_sim = Transcript::new("bw");
        let mut sim_r: Vec<FqExt> = Vec::new();
        let mut claim_sim = lambda * claim2;
        let mut eqpre = FqExt::ONE;
        for round in 0..nv_w {
            let half = 1 << (nv_w - 1 - round);
            let t = tau0[round];
            let eqrem = |cell: usize| -> FqExt {
                let rem = nv_w - 1 - round;
                let mut e = FqExt::ONE;
                for i in 0..rem {
                    let bit = (cell >> (rem - 1 - i)) & 1;
                    let tt = tau0[round + 1 + i];
                    e = e * if bit == 1 { tt } else { FqExt::ONE - tt };
                }
                e
            };
            let gx = |x: FqExt, wb: &[FqExt], lgb: &[FqExt], bb: &[FqExt]| -> FqExt {
                let mut g = FqExt::ZERO;
                let eq1x = t * x + (FqExt::ONE - t) * (FqExt::ONE - x);
                for cell in 0..half {
                    let wlo = wb[cell];
                    let whi = wb[cell + half];
                    let wx = wlo + x * (whi - wlo);
                    let f2 = if round < nv_k {
                        let hk = lgb.len() / 2;
                        let k = cell / s_len;
                        let l = cell % s_len;
                        (lgb[k] + x * (lgb[k + hk] - lgb[k])) * apow[l] * wx
                    } else {
                        let blo = bb[cell];
                        let bhi = bb[cell + half];
                        (blo + x * (bhi - blo)) * wx
                    };
                    g = g + lambda * f2 + eqpre * eq1x * eqrem(cell) * wx * (wx - FqExt::ONE);
                }
                g
            };
            let bg0 = gx(FqExt::ZERO, &wb, &lgb, &bb);
            let bg1 = gx(FqExt::ONE, &wb, &lgb, &bb);
            let bg2 = gx(FqExt::from_u64(2), &wb, &lgb, &bb);
            let bg3 = gx(FqExt::from_u64(3), &wb, &lgb, &bb);
            assert_eq!(bg0 + bg1, claim_sim, "round {round} g(0)+g(1) != claim");
            assert_eq!(proof.rounds[round][0], bg0, "round {round} g0");
            assert_eq!(proof.rounds[round][1], bg2, "round {round} g2");
            assert_eq!(proof.rounds[round][2], bg3, "round {round} g3");
            tr_sim.absorb_fq4(bg0);
            tr_sim.absorb_fq4(bg2);
            tr_sim.absorb_fq4(bg3);
            let r = tr_sim.challenge_fq4();
            sim_r.push(r);
            eqpre = eqpre * (t * r + (FqExt::ONE - t) * (FqExt::ONE - r));
            claim_sim = lagrange_eval(&[bg0, bg1, bg2, bg3], r);
            for i in 0..half {
                wb[i] = wb[i] + r * (wb[i + half] - wb[i]);
            }
            wb.truncate(half);
            if round < nv_k {
                let hk = lgb.len() / 2;
                for k in 0..hk {
                    lgb[k] = lgb[k] + r * (lgb[k + hk] - lgb[k]);
                }
                lgb.truncate(hk);
                if round + 1 == nv_k {
                    bb = apow.iter().map(|&p| lgb[0] * p).collect();
                }
            } else {
                let hb = bb.len() / 2;
                for i in 0..hb {
                    bb[i] = bb[i] + r * (bb[i + hb] - bb[i]);
                }
                bb.truncate(hb);
            }
        }

        let sim_final =
            lambda * bb[0] * wb[0] + eq_eval(&tau0, &sim_r) * wb[0] * (wb[0] - FqExt::ONE);

        let mut tr_v = Transcript::new("bw");
        let (e_b, r_v) = verify_batched_w(lambda * claim2, nv_w, &proof, &mut tr_v).expect("v");
        assert_eq!(r, r_v);
        assert_eq!(sim_r, r_v);
        assert_eq!(e_b, sim_final, "e_b vs sim-final (verify folding)");
        assert_eq!(open_w, wb[0], "open_w");
        let w_at = mle_eval(&w_fq4, &r_v);
        let lg_at = mle_eval(&lg, &r_v[..nv_k]);
        let apow_at = mle_eval(&apow, &r_v[nv_k..]);
        assert_eq!(w_at, wb[0], "W̃ via mle vs fold");
        assert_eq!(bb[0], lg_at * apow_at, "b̃ vs lg_at·apow_at");
    }

    fn batched_w_reference(
        w0: &[FqExt],
        mut lg: Vec<FqExt>,
        apow: &[FqExt],
        tau0: &[FqExt],
        lambda: FqExt,
        tr: &mut Transcript,
    ) -> (SumcheckProof, Vec<FqExt>, FqExt) {
        let nv_w = tau0.len();
        let nv_k = lg.len().trailing_zeros() as usize;
        let s_len = apow.len();
        let mut w = w0.to_vec();
        let mut b: Vec<FqExt> = Vec::new();
        let mut rounds = Vec::with_capacity(nv_w);
        let mut r_point = Vec::with_capacity(nv_w);
        let mut eqpre = FqExt::ONE;
        for round in 0..nv_w {
            let half = 1usize << (nv_w - 1 - round);
            let t = tau0[round];
            let eqrem = |cell: usize| -> FqExt {
                let rem = nv_w - 1 - round;
                (0..rem).fold(FqExt::ONE, |e, i| {
                    let tt = tau0[round + 1 + i];
                    e * if (cell >> (rem - 1 - i)) & 1 == 1 { tt } else { FqExt::ONE - tt }
                })
            };
            let gx = |x: FqExt, w: &[FqExt], lg: &[FqExt], b: &[FqExt]| -> FqExt {
                let eq1x = t * x + (FqExt::ONE - t) * (FqExt::ONE - x);
                (0..half).fold(FqExt::ZERO, |g, cell| {
                    let (wlo, whi) = (w[cell], w[cell + half]);
                    let wx = wlo + x * (whi - wlo);
                    let bx = if round < nv_k {
                        let hk = lg.len() / 2;
                        let (k, l) = (cell / s_len, cell % s_len);
                        (lg[k] + x * (lg[k + hk] - lg[k])) * apow[l]
                    } else {
                        b[cell] + x * (b[cell + half] - b[cell])
                    };
                    g + lambda * bx * wx + eqpre * eq1x * eqrem(cell) * wx * (wx - FqExt::ONE)
                })
            };
            let (g0, g2, g3) = (
                gx(FqExt::ZERO, &w, &lg, &b),
                gx(FqExt::from_u64(2), &w, &lg, &b),
                gx(FqExt::from_u64(3), &w, &lg, &b),
            );
            tr.absorb_fq4(g0);
            tr.absorb_fq4(g2);
            tr.absorb_fq4(g3);
            let r = tr.challenge_fq4();
            r_point.push(r);
            eqpre = eqpre * (t * r + (FqExt::ONE - t) * (FqExt::ONE - r));
            for i in 0..half {
                w[i] = w[i] + r * (w[i + half] - w[i]);
            }
            w.truncate(half);
            if round < nv_k {
                let hk = lg.len() / 2;
                for k in 0..hk {
                    lg[k] = lg[k] + r * (lg[k + hk] - lg[k]);
                }
                lg.truncate(hk);
                if round + 1 == nv_k {
                    b = apow.iter().map(|&p| lg[0] * p).collect();
                }
            } else {
                let hb = b.len() / 2;
                for i in 0..hb {
                    b[i] = b[i] + r * (b[i + hb] - b[i]);
                }
                b.truncate(hb);
            }
            rounds.push(vec![g0, g2, g3]);
        }
        (SumcheckProof { rounds }, r_point, w[0])
    }

    #[test]
    fn batched_w_reference_matches_prover_on_honest_bits() {
        let mut rng = SimpleRng::new(4242);
        let (nv_k, nv_l) = (3usize, 3usize);
        let (s_len, total) = (1usize << nv_l, 1usize << (nv_k + nv_l));
        let bits: Vec<u8> = (0..total).map(|_| rng.next_bool() as u8).collect();
        let w: Vec<FqExt> = bits.iter().map(|&b| FqExt::from_u64(b as u64)).collect();
        let zw = PackedBits::from_bits(&bits, total);
        let lg: Vec<FqExt> = (0..1 << nv_k).map(|_| rng.next_fq4()).collect();
        let apow: Vec<FqExt> = (0..s_len).map(|_| rng.next_fq4()).collect();
        let tau0: Vec<FqExt> = (0..nv_k + nv_l).map(|_| rng.next_fq4()).collect();
        let lambda = rng.next_fq4();

        let mut t1 = Transcript::new("ref");
        let (p1, r1, o1) =
            prove_batched_w(&zw, lg.clone(), &apow, &tau0, lambda, &mut Vec::new(), None, &mut t1);
        let mut t2 = Transcript::new("ref");
        let (p2, r2, o2) = batched_w_reference(&w, lg, &apow, &tau0, lambda, &mut t2);
        assert_eq!(p1.rounds, p2.rounds);
        assert_eq!(r1, r2);
        assert_eq!(o1, o2);
    }

    fn non_binary_w_is_rejected(bad_cell: usize, bad_val: u64, seed: u64) {
        use crate::mle::{eq_eval, mle_eval};
        let mut rng = SimpleRng::new(seed);
        let (nv_k, nv_l) = (3usize, 3usize);
        let (s_len, nv_w) = (1usize << nv_l, nv_k + nv_l);
        let total = 1usize << nv_w;
        let mut w: Vec<FqExt> =
            (0..total).map(|_| FqExt::from_u64(rng.next_bool() as u64)).collect();
        w[bad_cell] = FqExt::from_u64(bad_val);
        let lg: Vec<FqExt> = (0..1 << nv_k).map(|_| rng.next_fq4()).collect();
        let apow: Vec<FqExt> = (0..s_len).map(|_| rng.next_fq4()).collect();
        let tau0: Vec<FqExt> = (0..nv_w).map(|_| rng.next_fq4()).collect();
        let lambda = rng.next_fq4();

        let claim2 = (0..total)
            .fold(FqExt::ZERO, |a, c| a + lg[c / s_len] * apow[c % s_len] * w[c]);

        let mut tr_p = Transcript::new("nb");
        let (proof, r_p, _) = batched_w_reference(&w, lg.clone(), &apow, &tau0, lambda, &mut tr_p);
        let mut tr_v = Transcript::new("nb");
        let (e_b, r_v) =
            verify_batched_w(lambda * claim2, nv_w, &proof, &mut tr_v).expect("well-formed");
        assert_eq!(r_p, r_v);

        let w_at = mle_eval(&w, &r_v);
        let lg_at = mle_eval(&lg, &r_v[..nv_k]);
        let ap_at = mle_eval(&apow, &r_v[nv_k..]);
        let expect =
            lambda * lg_at * ap_at * w_at + eq_eval(&tau0, &r_v) * w_at * (w_at - FqExt::ONE);
        assert_ne!(e_b, expect, "non-binary W (cell {bad_cell} = {bad_val}) unexpectedly passed SC3");
    }

    #[test]
    fn cheat_non_binary_digit() {
        for (cell, val, seed) in [(0usize, 2u64, 5001u64), (37, 2, 5002), (63, Q - 1, 5003)] {
            non_binary_w_is_rejected(cell, val, seed);
        }
    }

    #[test]
    fn cheat_non_ternary_r() {
        for (cell, seed) in [(5usize, 6001u64), (48, 6002)] {
            non_binary_w_is_rejected(cell, 2, seed);
        }
    }

    fn rand_sc_proof(rng: &mut SimpleRng, max_rounds: usize, max_deg: usize) -> SumcheckProof {
        let n = (rng.next_u64() as usize) % (max_rounds + 2);
        SumcheckProof {
            rounds: (0..n)
                .map(|_| {
                    let l = (rng.next_u64() as usize) % (max_deg + 3);
                    (0..l).map(|_| rng.next_fq4()).collect()
                })
                .collect(),
        }
    }

    #[test]
    fn sumcheck_verifiers_never_panic_on_random_proofs() {
        let mut rng = SimpleRng::new(0x5F022_9001);
        for _ in 0..250_000 {
            let nv = (rng.next_u64() as usize) % 8;
            let deg = 1 + (rng.next_u64() as usize) % 3;
            let p = rand_sc_proof(&mut rng, nv, deg);
            let tau: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
            let claim = rng.next_fq4();
            let _ = verify(claim, nv, deg, &p, &mut Transcript::new("f"));
            let _ = verify_eq_bitcheck(claim, &tau, &p, &mut Transcript::new("f"));
            let _ = verify_product2(claim, nv, &p, &mut Transcript::new("f"));
            let _ = verify_batched_w(claim, nv, &p, &mut Transcript::new("f"));
        }
    }

    #[test]
    fn product2_roundtrip_and_wrong_claim() {
        use crate::mle::mle_eval;
        let mut rng = SimpleRng::new(9);
        let nv = 6;
        let a: Vec<Fq> = (0..1 << nv).map(|_| rng.next_fq()).collect();
        let b: Vec<FqExt> = (0..1 << nv).map(|_| rng.next_fq4()).collect();
        let claim = a.iter().zip(&b).fold(FqExt::ZERO, |s, (&x, &y)| s + x * y);

        let mut tr_p = Transcript::new("p2-test");
        let (proof, r_p) = prove_product2(&a, b.clone(), None, &mut tr_p);

        let mut tr_v = Transcript::new("p2-test");
        let (expect, r_v) = verify_product2(claim, nv, &proof, &mut tr_v).expect("verify");
        assert_eq!(r_p, r_v);
        let lifted: Vec<FqExt> = a.iter().map(|&x| FqExt::from_fq(x)).collect();
        assert_eq!(expect, mle_eval(&lifted, &r_v) * mle_eval(&b, &r_v));

        let mut tr_w = Transcript::new("p2-test");
        let (expect_w, r_w) =
            verify_product2(claim + FqExt::ONE, nv, &proof, &mut tr_w).expect("well-formed");
        assert_ne!(expect_w, mle_eval(&lifted, &r_w) * mle_eval(&b, &r_w));
    }
}

#[cfg(test)]
mod lagrange3 {
    use super::*;
    #[test]
    fn lagrange_deg3() {
        let p = |x: u64| FqExt::from_u64(2*x*x*x + 3*x*x + 5*x + 7);
        let e = vec![p(0), p(1), p(2), p(3)];
        assert_eq!(lagrange_eval(&e, FqExt::from_u64(9)), p(9));
        assert_eq!(lagrange_eval(&e, FqExt::from_u64(2)), p(2));
    }
}

pub struct Masker {
    coef: Vec<Vec<FqExt>>,
    suffix: Vec<FqExt>,
    nv: usize,
    deg: usize,
    rho: FqExt,
    prefix: FqExt,
    pow2: Vec<FqExt>,
}

impl Masker {
    pub fn new(rng: &mut crate::rng::CsRng, nv: usize, deg: usize) -> Self {
        let coef: Vec<Vec<FqExt>> =
            (0..nv).map(|_| (0..=deg).map(|_| rng.next_fq4()).collect()).collect();
        let ends: Vec<FqExt> = coef
            .iter()
            .map(|c| {
                let g0 = c[0];
                let g1 = c.iter().fold(FqExt::ZERO, |a, &v| a + v);
                g0 + g1
            })
            .collect();
        let mut suffix = vec![FqExt::ZERO; nv + 1];
        for i in (0..nv).rev() {
            suffix[i] = suffix[i + 1] + ends[i];
        }
        let mut pow2 = Vec::with_capacity(nv + 1);
        let mut p = FqExt::ONE;
        for _ in 0..=nv {
            pow2.push(p);
            p = p + p;
        }
        Masker { coef, suffix, nv, deg, rho: FqExt::ONE, prefix: FqExt::ZERO, pow2 }
    }

    pub fn num_coeffs(&self) -> usize {
        self.nv * (self.deg + 1)
    }

    pub fn coeffs_flat(&self) -> impl Iterator<Item = FqExt> + '_ {
        self.coef.iter().flat_map(|c| c.iter().copied())
    }

    pub fn set_rho(&mut self, rho: FqExt) {
        self.rho = rho;
    }

    fn gi(&self, i: usize, p: FqExt) -> FqExt {
        self.coef[i].iter().rev().fold(FqExt::ZERO, |acc, &c| acc * p + c)
    }

    pub fn total_plain(&self) -> FqExt {
        if self.nv == 0 {
            return FqExt::ZERO;
        }
        self.pow2[self.nv - 1] * self.suffix[0]
    }

    pub fn total_eq(&self, tau: &[FqExt]) -> FqExt {
        debug_assert_eq!(tau.len(), self.nv);
        (0..self.nv).fold(FqExt::ZERO, |acc, i| {
            let g0 = self.coef[i][0];
            let g1 = self.coef[i].iter().fold(FqExt::ZERO, |a, &v| a + v);
            acc + (FqExt::ONE - tau[i]) * g0 + tau[i] * g1
        })
    }

    pub fn round_plain(&self, round: usize, pts: &[u64]) -> Vec<FqExt> {
        let lead = self.pow2[self.nv - 1 - round];
        let tail = if round + 2 <= self.nv {
            self.pow2[self.nv - 2 - round] * self.suffix[round + 1]
        } else {
            FqExt::ZERO
        };
        pts.iter()
            .map(|&p| {
                let g = self.gi(round, FqExt::from_u64(p));
                self.rho * (lead * (g + self.prefix) + tail)
            })
            .collect()
    }

    pub fn round_eq(&self, round: usize, tau: &[FqExt], pts: &[u64]) -> Vec<FqExt> {
        let tail = ((round + 1)..self.nv).fold(FqExt::ZERO, |acc, i| {
            let g0 = self.coef[i][0];
            let g1 = self.coef[i].iter().fold(FqExt::ZERO, |a, &v| a + v);
            acc + (FqExt::ONE - tau[i]) * g0 + tau[i] * g1
        });
        pts.iter()
            .map(|&p| self.rho * (self.gi(round, FqExt::from_u64(p)) + self.prefix + tail))
            .collect()
    }

    pub fn fold(&mut self, round: usize, r: FqExt) {
        self.prefix = self.prefix + self.gi(round, r);
    }

    pub fn eval(&self) -> FqExt {
        self.prefix
    }

    pub fn weights_eval(nv: usize, deg: usize, r: &[FqExt]) -> Vec<FqExt> {
        let mut out = Vec::with_capacity(nv * (deg + 1));
        for &ri in r.iter().take(nv) {
            let mut p = FqExt::ONE;
            for _ in 0..=deg {
                out.push(p);
                p = p * ri;
            }
        }
        out
    }

    pub fn weights_total_plain(nv: usize, deg: usize) -> Vec<FqExt> {
        let mut lead = FqExt::ONE;
        for _ in 0..nv.saturating_sub(1) {
            lead = lead + lead;
        }
        let mut out = Vec::with_capacity(nv * (deg + 1));
        for _ in 0..nv {
            for k in 0..=deg {
                out.push(if k == 0 { lead + lead } else { lead });
            }
        }
        out
    }

    pub fn weights_total_eq(nv: usize, deg: usize, tau: &[FqExt]) -> Vec<FqExt> {
        let mut out = Vec::with_capacity(nv * (deg + 1));
        for &t in tau.iter().take(nv) {
            for k in 0..=deg {
                out.push(if k == 0 { FqExt::ONE } else { t });
            }
        }
        out
    }
}

#[cfg(test)]
mod mask_tests {
    use super::*;
    use crate::rng::{insecure_test_secret, CsRng};

    fn mk(nv: usize, deg: usize, seed: u64) -> Masker {
        let mut rng = CsRng::from_parts("mask-test", &[&insecure_test_secret(seed)]);
        Masker::new(&mut rng, nv, deg)
    }

    fn brute(m: &Masker, x: &[FqExt]) -> FqExt {
        (0..m.nv).fold(FqExt::ZERO, |a, i| a + m.gi(i, x[i]))
    }

    #[test]
    fn total_plain_matches_brute_force() {
        for (nv, deg) in [(1usize, 2usize), (3, 3), (4, 2), (5, 3)] {
            let m = mk(nv, deg, 1);
            let mut s = FqExt::ZERO;
            for mask in 0..(1u32 << nv) {
                let x: Vec<FqExt> = (0..nv)
                    .map(|i| FqExt::from_u64(((mask >> i) & 1) as u64))
                    .collect();
                s = s + brute(&m, &x);
            }
            assert_eq!(m.total_plain(), s, "nv={nv} deg={deg}");
        }
    }

    #[test]
    fn total_eq_matches_brute_force() {
        for (nv, deg) in [(1usize, 2usize), (3, 3), (4, 2)] {
            let m = mk(nv, deg, 2);
            let mut rng = CsRng::from_parts("tau", &[&insecure_test_secret(9)]);
            let tau: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
            let eqt = eq_table(&tau);
            let mut s = FqExt::ZERO;
            for cell in 0..(1usize << nv) {
                let x: Vec<FqExt> = (0..nv)
                    .map(|i| FqExt::from_u64(((cell >> (nv - 1 - i)) & 1) as u64))
                    .collect();
                s = s + eqt[cell] * brute(&m, &x);
            }
            assert_eq!(m.total_eq(&tau), s, "nv={nv} deg={deg}");
        }
    }

    #[test]
    fn round_evals_telescope_to_the_total() {
        let mut rng = CsRng::from_parts("chal", &[&insecure_test_secret(3)]);
        for (nv, deg) in [(1usize, 2usize), (3, 3), (5, 2), (6, 3)] {
            let mut m = mk(nv, deg, 4);
            m.set_rho(FqExt::ONE);
            let mut claim = m.total_plain();
            for j in 0..nv {
                let v = m.round_plain(j, &[0, 1]);
                assert_eq!(v[0] + v[1], claim, "plain nv={nv} deg={deg} round={j}");
                let r = rng.next_fq4();
                let pts: Vec<u64> = (0..=deg as u64 + 1).collect();
                let vals = m.round_plain(j, &pts);
                claim = lagrange_eval(&vals, r);
                m.fold(j, r);
            }
            assert_eq!(claim, m.eval(), "plain finish: the final claim must be g(r)");

            let mut m = mk(nv, deg, 5);
            m.set_rho(FqExt::ONE);
            let tau: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
            let mut claim = m.total_eq(&tau);
            let mut a = FqExt::ONE;
            for j in 0..nv {
                let v = m.round_eq(j, &tau, &[0, 1]);
                let lhs = a * ((FqExt::ONE - tau[j]) * v[0] + tau[j] * v[1]);
                assert_eq!(lhs, claim, "eq nv={nv} deg={deg} round={j}");
                let r = rng.next_fq4();
                let pts: Vec<u64> = (0..=deg as u64).collect();
                let hr = lagrange_eval(&m.round_eq(j, &tau, &pts), r);
                a = a * (tau[j] * r + (FqExt::ONE - tau[j]) * (FqExt::ONE - r));
                claim = a * hr;
                m.fold(j, r);
            }
        }
    }

    #[test]
    fn linear_weights_match_the_closed_forms() {
        let mut rng = CsRng::from_parts("w", &[&insecure_test_secret(6)]);
        for (nv, deg) in [(1usize, 2usize), (3, 3), (5, 2)] {
            let mut m = mk(nv, deg, 7);
            let dot = |w: &[FqExt], m: &Masker| -> FqExt {
                m.coeffs_flat().zip(w).fold(FqExt::ZERO, |a, (c, &x)| a + c * x)
            };
            assert_eq!(dot(&Masker::weights_total_plain(nv, deg), &m), m.total_plain());
            let tau: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
            assert_eq!(dot(&Masker::weights_total_eq(nv, deg, &tau), &m), m.total_eq(&tau));
            let r: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
            for (j, &rj) in r.iter().enumerate() {
                m.fold(j, rj);
            }
            assert_eq!(dot(&Masker::weights_eval(nv, deg, &r), &m), m.eval());
        }
    }
}
