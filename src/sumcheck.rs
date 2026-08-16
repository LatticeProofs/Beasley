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
    verify_degs(claim, &vec![deg; nv], proof, tr)
}

pub fn verify_degs(
    claim: FqExt,
    degs: &[usize],
    proof: &SumcheckProof,
    tr: &mut Transcript,
) -> Option<(FqExt, Vec<FqExt>)> {
    if proof.rounds.len() != degs.len() {
        return None;
    }
    let mut expect = claim;
    let mut r_point = Vec::with_capacity(degs.len());
    for (evals, &d) in proof.rounds.iter().zip(degs) {
        if evals.len() != d + 1 {
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

#[derive(Clone, Copy)]
pub struct ClaimMask<'a> {
    pub coef: &'a [FqExt],
}

impl ClaimMask<'_> {
    pub fn eval(&self, x: FqExt) -> FqExt {
        self.coef.iter().rev().fold(FqExt::ZERO, |a, &c| a * x + c)
    }
    pub fn total(&self) -> FqExt {
        self.coef[0] + self.coef.iter().fold(FqExt::ZERO, |a, &c| a + c)
    }
    pub fn deg(&self) -> usize {
        self.coef.len() - 1
    }
    pub fn weights_eval(deg: usize, c: FqExt) -> Vec<FqExt> {
        let mut out = Vec::with_capacity(deg + 1);
        let mut p = FqExt::ONE;
        for _ in 0..=deg {
            out.push(p);
            p = p * c;
        }
        out
    }
}

#[derive(Clone, Copy)]
pub struct LinMask(pub [FqExt; 2]);

impl LinMask {
    #[inline]
    pub fn eval(&self, z1: FqExt) -> FqExt {
        self.0[0] + z1 * self.0[1]
    }
    #[inline]
    pub fn weights(z1: FqExt) -> [FqExt; 2] {
        [FqExt::ONE, z1]
    }
}

pub const LIN_MASK_COEFS: usize = 2;

pub fn round_degs(nv: usize, deg: usize, z_deg: usize, w_deg: usize) -> Vec<usize> {
    let mut d = vec![deg; nv];
    if let Some(last) = d.last_mut() {
        *last = z_deg;
    }
    d.push(w_deg);
    d
}

pub fn prove_bilinear_zk(
    mut eq: Vec<FqExt>,
    mut h: Vec<FqExt>,
    mut u: Vec<FqExt>,
    sigma_h: LinMask,
    sigma_u: FqExt,
    rmask: ClaimMask,
    mut mask: Option<&mut Masker>,
    tr: &mut Transcript,
) -> (SumcheckProof, Vec<FqExt>, FqExt, FqExt, FqExt) {
    let len = eq.len();
    assert!(len.is_power_of_two());
    assert_eq!(h.len(), len);
    assert_eq!(u.len(), len);
    let nv = len.trailing_zeros() as usize;
    let mdeg = rmask.deg();
    let msum = rmask.total();

    let mut rounds = Vec::with_capacity(nv + 1);
    let mut r_point = Vec::with_capacity(nv + 1);
    let mut zpre = FqExt::ONE;
    let mut ipre = FqExt::ONE;

    for round in 0..nv {
        let half = eq.len() / 2;
        let last_cube = round + 1 == nv;
        let deg = if last_cube { 6 } else { 3 };
        let pts: Vec<u64> = (0..=deg as u64).collect();
        let mut evals = Vec::with_capacity(deg + 1);
        for &s in &pts {
            let sf = FqExt::from_u64(s);
            let zc = if last_cube { zpre * sf * (FqExt::ONE - sf) } else { FqExt::ZERO };
            let (dh, du) = (zc * sigma_h.eval(sf), zc * sigma_u);
            let mut acc = FqExt::ZERO;
            for cell in 0..half {
                let e = eq[cell] + sf * (eq[cell + half] - eq[cell]);
                let hv = h[cell] + sf * (h[cell + half] - h[cell]) + dh;
                let uv = u[cell] + sf * (u[cell + half] - u[cell]) + du;
                acc = acc + e * hv * uv;
            }
            evals.push(acc + ipre * (FqExt::ONE - sf) * msum);
        }
        if let Some(m) = mask.as_deref_mut() {
            for (e, v) in evals.iter_mut().zip(m.round_plain(round, &pts)) {
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
        for t in [&mut eq, &mut h, &mut u] {
            for i in 0..half {
                t[i] = t[i] + r * (t[i + half] - t[i]);
            }
            t.truncate(half);
        }
        zpre = zpre * r * (FqExt::ONE - r);
        ipre = ipre * (FqExt::ONE - r);
        rounds.push(evals);
    }

    let z1 = r_point[nv - 1];
    let h_dot = h[0] + zpre * sigma_h.eval(z1);
    let u_dot = u[0] + zpre * sigma_u;
    let main = eq[0] * h_dot * u_dot;
    let wdeg = mdeg.max(1).max(mask.as_deref().map_or(0, |m| m.deg()));
    let wpts: Vec<u64> = (0..=wdeg as u64).collect();
    let mut evals: Vec<FqExt> = wpts
        .iter()
        .map(|&s| {
            let sf = FqExt::from_u64(s);
            (FqExt::ONE - sf) * main + ipre * rmask.eval(sf)
        })
        .collect();
    if let Some(m) = mask.as_deref_mut() {
        for (e, v) in evals.iter_mut().zip(m.round_plain(nv, &wpts)) {
            *e = *e + v;
        }
    }
    for &e in &evals {
        tr.absorb_fq4(e);
    }
    let c = tr.challenge_fq4();
    if let Some(m) = mask.as_deref_mut() {
        m.fold(nv, c);
    }
    r_point.push(c);
    rounds.push(evals);

    (SumcheckProof { rounds }, r_point, h_dot, u_dot, rmask.eval(c))
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
    sigma: LinMask,
    rmask: ClaimMask,
    mut mask: Option<&mut Masker>,
    tr: &mut Transcript,
) -> (SumcheckProof, Vec<FqExt>, FqExt, FqExt) {
    let nv = b.len().trailing_zeros() as usize;
    assert_eq!(a_base.len(), b.len());
    let msum = rmask.total();
    let mut rounds = Vec::with_capacity(nv + 1);
    let mut r_point = Vec::with_capacity(nv + 1);
    let mut a_ext: Vec<FqExt> = Vec::new();
    let mut zpre = FqExt::ONE;
    let mut ipre = FqExt::ONE;
    let mut b_final = FqExt::ZERO;

    for j in 0..nv {
        let half = 1usize << (nv - 1 - j);
        let last_cube = j + 1 == nv;
        let pts: &[u64] = if last_cube { &[0, 2, 3, 4] } else { &[0, 2] };
        let mut evals = vec![FqExt::ZERO; pts.len()];
        if last_cube {
            let (alo, ahi) = if j == 0 {
                (FqExt::from_fq(a_base[0]), FqExt::from_fq(a_base[1]))
            } else {
                (a_ext[0], a_ext[1])
            };
            for (k, &s) in pts.iter().enumerate() {
                let sf = FqExt::from_u64(s);
                let zc = zpre * sf * (FqExt::ONE - sf) * sigma.eval(sf);
                let av = alo + sf * (ahi - alo) + zc;
                let bv = b[0] + sf * (b[1] - b[0]);
                evals[k] = av * bv;
            }
        } else if j == 0 {
            let (mut g0, mut g2) = (FqExt::ZERO, FqExt::ZERO);
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
            evals[0] = g0;
            evals[1] = g2;
        } else {
            let (mut g0, mut g2) = (FqExt::ZERO, FqExt::ZERO);
            for cell in 0..half {
                let alo = a_ext[cell];
                let ahi = a_ext[cell + half];
                let blo = b[cell];
                let bhi = b[cell + half];
                g0 = g0 + alo * blo;
                g2 = g2 + (ahi + (ahi - alo)) * (bhi + (bhi - blo));
            }
            evals[0] = g0;
            evals[1] = g2;
        }
        for (k, &s) in pts.iter().enumerate() {
            let sf = FqExt::from_u64(s);
            evals[k] = evals[k] + ipre * (FqExt::ONE - sf) * msum;
        }
        if let Some(m) = mask.as_deref_mut() {
            for (e, v) in evals.iter_mut().zip(m.round_plain(j, pts)) {
                *e = *e + v;
            }
        }
        for &e in &evals {
            tr.absorb_fq4(e);
        }
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
        zpre = zpre * r * (FqExt::ONE - r);
        ipre = ipre * (FqExt::ONE - r);
        b_final = b[0];
        rounds.push(evals);
    }

    let t_dot = a_ext[0] + zpre * sigma.eval(r_point[nv - 1]);
    let main = b_final * t_dot;
    let wdeg = rmask.deg().max(1).max(mask.as_deref().map_or(0, |m| m.deg()));
    let wpts: Vec<u64> = std::iter::once(0).chain(2..=wdeg as u64).collect();
    let mut evals: Vec<FqExt> = wpts
        .iter()
        .map(|&s| {
            let sf = FqExt::from_u64(s);
            (FqExt::ONE - sf) * main + ipre * rmask.eval(sf)
        })
        .collect();
    if let Some(m) = mask.as_deref_mut() {
        for (e, v) in evals.iter_mut().zip(m.round_plain(nv, &wpts)) {
            *e = *e + v;
        }
    }
    for &e in &evals {
        tr.absorb_fq4(e);
    }
    let c = tr.challenge_fq4();
    if let Some(m) = mask.as_deref_mut() {
        m.fold(nv, c);
    }
    r_point.push(c);
    rounds.push(evals);

    (SumcheckProof { rounds }, r_point, t_dot, rmask.eval(c))
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
    degs: &[usize],
    proof: &SumcheckProof,
    tr: &mut Transcript,
) -> Option<(FqExt, Vec<FqExt>)> {
    if proof.rounds.len() != degs.len() {
        return None;
    }
    let mut c = claim;
    let mut r_point = Vec::with_capacity(degs.len());
    for (evals, &d) in proof.rounds.iter().zip(degs) {
        if evals.len() != d {
            return None;
        }
        for &e in evals {
            tr.absorb_fq4(e);
        }
        let r = tr.challenge_fq4();
        r_point.push(r);
        let mut pts = Vec::with_capacity(d + 1);
        pts.push(evals[0]);
        pts.push(c - evals[0]);
        pts.extend_from_slice(&evals[1..]);
        c = lagrange_eval(&pts, r);
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
    sigma_w: LinMask,
    rmask: ClaimMask,
    scratch: &mut Vec<FqExt>,
    mut mask: Option<&mut Masker>,
    tr: &mut Transcript,
) -> (SumcheckProof, Vec<FqExt>, FqExt, FqExt) {
    let nv_w = tau0.len();
    let nv_k = lg.len().trailing_zeros() as usize;
    let s_len = apow.len();
    assert_eq!(zw.len(), 1 << nv_w);
    assert_eq!(lg.len() * s_len, 1 << nv_w);
    assert!(nv_k >= 1 && s_len >= 2, "prove_batched_w requires nv_k ≥ 1 and s_len ≥ 2");

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
    let mut zpre = FqExt::ONE;
    let mut ipre = FqExt::ONE;
    let msum = rmask.total();
    let mut main_final = FqExt::ZERO;
    let mut w_dot = FqExt::ZERO;

    for round in 0..nv_w {
        let half = 1usize << (nv_w - 1 - round);
        if round + 1 == nv_w {
            debug_assert_eq!(half, 1);
            debug_assert!(round >= nv_k, "the last round must fall in phase B (s_len ≥ 2)");
            let (w0, dw) = (w[0], w[1] - w[0]);
            let (b0, db) = (b[0], b[1] - b[0]);
            let t = tau0[round];
            let slope = t + t - FqExt::ONE;
            let eq1_0 = FqExt::ONE - t;
            let pts: [u64; 7] = [0, 2, 3, 4, 5, 6, 7];
            let mut evals: Vec<FqExt> = pts
                .iter()
                .map(|&s| {
                    let sf = FqExt::from_u64(s);
                    let wv = w0 + sf * dw + zpre * sf * (FqExt::ONE - sf) * sigma_w.eval(sf);
                    let bv = b0 + sf * db;
                    let q3 = wv * (wv - FqExt::ONE);
                    let eq1 = eq1_0 + sf * slope;
                    lambda * (bv * wv)
                        + eq_prefix * eq1 * q3
                        + ipre * (FqExt::ONE - sf) * msum
                })
                .collect();
            if let Some(m) = mask.as_deref_mut() {
                for (e, v) in evals.iter_mut().zip(m.round_plain(round, &pts)) {
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
            zpre = zpre * r * (FqExt::ONE - r);
            ipre = ipre * (FqExt::ONE - r);
            w_dot = w0 + r * dw + zpre * sigma_w.eval(r);
            let eq_full = eq_prefix * (t * r + (FqExt::ONE - t) * (FqExt::ONE - r));
            main_final = lambda * (b0 + r * db) * w_dot + eq_full * w_dot * (w_dot - FqExt::ONE);
            rounds.push(evals);
            break;
        }
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
        let ind = ipre * msum;
        let mut g0 = lambda * gf2_0 + eq_prefix * eq1_0 * q3_0 + ind;
        let mut g2 = lambda * gf2_2 + eq_prefix * eq1_2 * q3_2 - ind;
        let mut g3 = lambda * gf2_3 + eq_prefix * eq1_3 * q3_3 - (ind + ind);
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
        zpre = zpre * r * (FqExt::ONE - r);
        ipre = ipre * (FqExt::ONE - r);

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
    let wdeg = rmask.deg().max(1).max(mask.as_deref().map_or(0, |m| m.deg()));
    let wpts: Vec<u64> = std::iter::once(0).chain(2..=wdeg as u64).collect();
    let mut evals: Vec<FqExt> = wpts
        .iter()
        .map(|&s| {
            let sf = FqExt::from_u64(s);
            (FqExt::ONE - sf) * main_final + ipre * rmask.eval(sf)
        })
        .collect();
    if let Some(m) = mask.as_deref_mut() {
        for (e, v) in evals.iter_mut().zip(m.round_plain(nv_w, &wpts)) {
            *e = *e + v;
        }
    }
    for &e in &evals {
        tr.absorb_fq4(e);
    }
    let c = tr.challenge_fq4();
    if let Some(m) = mask.as_deref_mut() {
        m.fold(nv_w, c);
    }
    r_point.push(c);
    rounds.push(evals);

    (SumcheckProof { rounds }, r_point, w_dot, rmask.eval(c))
}

pub fn verify_batched_w(
    claim: FqExt,
    degs: &[usize],
    proof: &SumcheckProof,
    tr: &mut Transcript,
) -> Option<(FqExt, Vec<FqExt>)> {
    if proof.rounds.len() != degs.len() {
        return None;
    }
    let mut c = claim;
    let mut r_point = Vec::with_capacity(degs.len());
    for (evals, &d) in proof.rounds.iter().zip(degs) {
        if evals.len() != d {
            return None;
        }
        let g1 = c - evals[0];
        for &e in evals {
            tr.absorb_fq4(e);
        }
        let r = tr.challenge_fq4();
        r_point.push(r);
        let mut pts = Vec::with_capacity(d + 1);
        pts.push(evals[0]);
        pts.push(g1);
        pts.extend_from_slice(&evals[1..]);
        c = lagrange_eval(&pts, r);
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

        let zr = [FqExt::ZERO; 3];
        let mut tr1 = Transcript::new("p2f");
        let (p1, r1, _, _) =
            prove_product2(&a, b, LinMask([FqExt::ZERO; 2]), ClaimMask { coef: &zr }, None, &mut tr1);
        let mut tr2 = Transcript::new("p2f");
        let (p2, r2, _) = prove_product2_factored(&a_p, lg, &apow, &mut Vec::new(), &mut tr2);
        let nvt = nv_k + s_len.trailing_zeros() as usize;
        assert_eq!(r1[..nvt - 1], r2[..nvt - 1]);
        assert_eq!(p1.rounds[..nvt - 1], p2.rounds[..nvt - 1]);
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
        let w_fq4: Vec<FqExt> = w_bits.iter().map(|&b| FqExt::from_u64(b as u64)).collect();
        let zw = PackedBits::from_bits(&w_bits, total);
        let lg: Vec<FqExt> = (0..1 << nv_k).map(|_| rng.next_fq4()).collect();
        let apow: Vec<FqExt> = (0..s_len).map(|_| rng.next_fq4()).collect();
        let tau0: Vec<FqExt> = (0..nv_w).map(|_| rng.next_fq4()).collect();
        let lambda = rng.next_fq4();
        let sigma_w = LinMask([rng.next_fq4(), rng.next_fq4()]);
        let ncoef: [FqExt; 3] = [rng.next_fq4(), rng.next_fq4(), rng.next_fq4()];
        let nmask = ClaimMask { coef: &ncoef };
        let nsum = nmask.total();

        let mut claim2 = FqExt::ZERO;
        for cell in 0..total {
            claim2 = claim2 + lg[cell / s_len] * apow[cell % s_len] * w_fq4[cell];
        }
        let claim0 = lambda * claim2 + nsum;

        let mut tr_p = Transcript::new("bw");
        let (proof, r, open_w, n_at_c) = prove_batched_w(
            &zw,
            lg.clone(),
            &apow,
            &tau0,
            lambda,
            sigma_w,
            nmask,
            &mut Vec::new(),
            None,
            &mut tr_p,
        );

        let mut wb = w_fq4.clone();
        let mut lgb = lg.clone();
        let mut bb: Vec<FqExt> = Vec::new();
        let mut tr_sim = Transcript::new("bw");
        let mut sim_r: Vec<FqExt> = Vec::new();
        let mut claim_sim = claim0;
        let mut eqpre = FqExt::ONE;
        let mut zpre = FqExt::ONE;
        let mut ipre = FqExt::ONE;
        for round in 0..nv_w {
            let half = 1 << (nv_w - 1 - round);
            let last = round + 1 == nv_w;
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
                let zc =
                    if last { zpre * x * (FqExt::ONE - x) * sigma_w.eval(x) } else { FqExt::ZERO };
                for cell in 0..half {
                    let wlo = wb[cell];
                    let whi = wb[cell + half];
                    let wx = wlo + x * (whi - wlo) + zc;
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
                g + ipre * (FqExt::ONE - x) * nsum
            };
            let nodes: Vec<u64> =
                if last { (0..=7).collect() } else { vec![0, 1, 2, 3] };
            let all: Vec<FqExt> =
                nodes.iter().map(|&s| gx(FqExt::from_u64(s), &wb, &lgb, &bb)).collect();
            assert_eq!(all[0] + all[1], claim_sim, "round {round} g(0)+g(1) != claim");
            let sent: Vec<FqExt> =
                all.iter().enumerate().filter(|(i, _)| *i != 1).map(|(_, &v)| v).collect();
            assert_eq!(proof.rounds[round], sent, "round polynomial at round {round}");
            for &e in &sent {
                tr_sim.absorb_fq4(e);
            }
            let r = tr_sim.challenge_fq4();
            sim_r.push(r);
            eqpre = eqpre * (t * r + (FqExt::ONE - t) * (FqExt::ONE - r));
            claim_sim = lagrange_eval(&all, r);
            for i in 0..half {
                wb[i] = wb[i] + r * (wb[i + half] - wb[i]);
            }
            wb.truncate(half);
            zpre = zpre * r * (FqExt::ONE - r);
            ipre = ipre * (FqExt::ONE - r);
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

        let w_dot = wb[0] + zpre * sigma_w.eval(sim_r[nv_w - 1]);
        let main_final =
            lambda * bb[0] * w_dot + eq_eval(&tau0, &sim_r) * w_dot * (w_dot - FqExt::ONE);
        let wall: Vec<FqExt> = [0u64, 1, 2]
            .iter()
            .map(|&s| {
                let x = FqExt::from_u64(s);
                (FqExt::ONE - x) * main_final + ipre * nmask.eval(x)
            })
            .collect();
        assert_eq!(wall[0] + wall[1], claim_sim, "w round: g(0)+g(1) != claim");
        let wsent: Vec<FqExt> = vec![wall[0], wall[2]];
        assert_eq!(proof.rounds[nv_w], wsent, "round polynomial of the w round");
        for &e in &wsent {
            tr_sim.absorb_fq4(e);
        }
        let c = tr_sim.challenge_fq4();
        sim_r.push(c);
        let sim_final = lagrange_eval(&wall, c);

        let degs = round_degs(nv_w, 3, 7, 2);
        let mut tr_v = Transcript::new("bw");
        let (e_b, r_v) = verify_batched_w(claim0, &degs, &proof, &mut tr_v).expect("v");
        assert_eq!(r, r_v);
        assert_eq!(sim_r, r_v);
        assert_eq!(e_b, sim_final, "e_b vs sim-final (verify folding)");
        assert_eq!(open_w, w_dot, "open_w must be the masked Ẇ(r_w)");
        assert_ne!(open_w, wb[0], "Ẇ(r_w) must not equal W̃(r_w) when σ_W ≠ 0");
        assert_eq!(n_at_c, nmask.eval(c), "N(c) is wrong");
        let ind = sim_r[..nv_w].iter().fold(FqExt::ONE, |a, &x| a * (FqExt::ONE - x));
        assert_eq!(
            e_b,
            (FqExt::ONE - c) * main_final + ind * n_at_c,
            "the final equation of Libra step (f) does not match"
        );
        let w_at = mle_eval(&w_fq4, &r_v[..nv_w]);
        let lg_at = mle_eval(&lg, &r_v[..nv_k]);
        let apow_at = mle_eval(&apow, &r_v[nv_k..nv_w]);
        assert_eq!(w_at, wb[0], "W̃ via mle vs fold");
        assert_eq!(bb[0], lg_at * apow_at, "b̃ vs lg_at·apow_at");
    }

    #[test]
    fn bilinear_zk_keeps_the_claim_and_masks_the_finals() {
        use crate::mle::mle_eval;
        let mut rng = SimpleRng::new(0x0BEE_51);
        let nv = 5usize;
        let n = 1usize << nv;
        let eq: Vec<FqExt> = (0..n).map(|_| rng.next_fq4()).collect();
        let h: Vec<FqExt> = (0..n).map(|_| rng.next_fq4()).collect();
        let u: Vec<FqExt> = (0..n).map(|_| rng.next_fq4()).collect();
        let claim = (0..n).fold(FqExt::ZERO, |a, i| a + eq[i] * h[i] * u[i]);

        let zr = [FqExt::ZERO; 3];
        let rb: [FqExt; 3] = [rng.next_fq4(), rng.next_fq4(), rng.next_fq4()];
        let run = |sh: LinMask, su: FqExt, rc: &[FqExt; 3]| {
            let mut tr = Transcript::new("bz");
            let (p, r, hf, uf, rv) = prove_bilinear_zk(
                eq.clone(),
                h.clone(),
                u.clone(),
                sh,
                su,
                ClaimMask { coef: rc },
                None,
                &mut tr,
            );
            (p, r, hf, uf, rv)
        };
        let (p0, r0, h0, u0, _) = run(LinMask([FqExt::ZERO; 2]), FqExt::ZERO, &zr);
        let sh = LinMask([rng.next_fq4(), rng.next_fq4()]);
        let su = rng.next_fq4();
        let (p1, r1, h1, u1, rb_at_c) = run(sh, su, &rb);
        let msum = rb[0] + (rb[0] + rb[1] + rb[2]);

        assert_eq!(p0.rounds[0][0] + p0.rounds[0][1], claim, "σ changed the claim");
        assert_eq!(
            p1.rounds[0][0] + p1.rounds[0][1],
            claim + msum,
            "claim is not shifted by exactly ΣR_B"
        );
        assert_eq!(p1.rounds.len(), nv + 1);
        assert!(p1.rounds[..nv - 1].iter().all(|x| x.len() == 4));
        assert_eq!(p1.rounds[nv - 1].len(), 7);
        assert_eq!(p1.rounds[nv].len(), 3, "w round: deg 2 ⇒ send 3 values (node 1 is not skipped)");

        let rc = &r1[..nv];
        let z = rc.iter().fold(FqExt::ONE, |a, &x| a * x * (FqExt::ONE - x));
        assert_eq!(h1, mle_eval(&h, rc) + z * sh.eval(rc[nv - 1]), "Ḣ(r) is not H̃(r) + Z(r)R_H(z₁)");
        assert_eq!(u1, mle_eval(&u, rc) + z * su, "U̇(r) is not Ũ(r) + Z(r)σ_U");
        assert_eq!(h0, mle_eval(&h, &r0[..nv]), "must fall back to unmasked when σ=0");
        assert_eq!(u0, mle_eval(&u, &r0[..nv]));
        assert_ne!(h0, h1, "R_H had no effect");
        assert_ne!(u0, u1, "σ_U had no effect");

        let degs = round_degs(nv, 3, 6, 2);
        let mut tr_v = Transcript::new("bz");
        let (e, rv) = verify_degs(claim + msum, &degs, &p1, &mut tr_v).expect("verify");
        assert_eq!(rv, r1);
        let c = r1[nv];
        let ind = rc.iter().fold(FqExt::ONE, |a, &x| a * (FqExt::ONE - x));
        assert_eq!(
            e,
            (FqExt::ONE - c) * mle_eval(&eq, rc) * h1 * u1 + ind * rb_at_c,
            "the final equation of Libra step (f) does not match"
        );
        let mut tr_w = Transcript::new("bz");
        assert!(verify_degs(claim, &degs, &p1, &mut tr_w).is_none(), "an unmasked claim unexpectedly passed");
    }

    #[test]
    fn product2_sigma_shifts_claim_and_final_only() {
        use crate::mle::mle_eval;
        let mut rng = SimpleRng::new(0x0BEE_52);
        let nv = 6usize;
        let n = 1usize << nv;
        let a: Vec<Fq> = (0..n).map(|_| Fq::new(rng.next_u64() % Q)).collect();
        let b: Vec<FqExt> = (0..n).map(|_| rng.next_fq4()).collect();
        let base = (0..n).fold(FqExt::ZERO, |s, i| s + FqExt::from_fq(a[i]) * b[i]);
        let sigma = LinMask([rng.next_fq4(), rng.next_fq4()]);

        let rq: [FqExt; 3] = [rng.next_fq4(), rng.next_fq4(), rng.next_fq4()];
        let msum = rq[0] + (rq[0] + rq[1] + rq[2]);

        let mut tr = Transcript::new("p2s");
        let (p, r, t_dot, rq_at_c) =
            prove_product2(&a, b.clone(), sigma, ClaimMask { coef: &rq }, None, &mut tr);
        let claim = base + msum;

        let degs = round_degs(nv, 2, 4, 2);
        let mut tr_v = Transcript::new("p2s");
        let (e, rv) = verify_product2(claim, &degs, &p, &mut tr_v).expect("verify");
        assert_eq!(rv, r);
        let rc = &r[..nv];
        let lifted: Vec<FqExt> = a.iter().map(|&x| FqExt::from_fq(x)).collect();
        let z = rc.iter().fold(FqExt::ONE, |x, &y| x * y * (FqExt::ONE - y));
        let ind = rc.iter().fold(FqExt::ONE, |x, &y| x * (FqExt::ONE - y));
        assert_eq!(
            t_dot,
            mle_eval(&lifted, rc) + z * sigma.eval(rc[nv - 1]),
            "Ṫ(r) is not T̃(r) + Z(r)R_T(z₁)"
        );
        assert_ne!(t_dot, mle_eval(&lifted, rc), "σ_T had no effect");
        let c = r[nv];
        assert_eq!(
            e,
            (FqExt::ONE - c) * t_dot * mle_eval(&b, rc) + ind * rq_at_c,
            "the final equation does not match"
        );
        let mut tr_w = Transcript::new("p2s");
        let (ew, rw) = verify_product2(base, &degs, &p, &mut tr_w).expect("well-formed");
        let rwc = &rw[..nv];
        let zw2 = rwc.iter().fold(FqExt::ONE, |x, &y| x * y * (FqExt::ONE - y));
        let indw = rwc.iter().fold(FqExt::ONE, |x, &y| x * (FqExt::ONE - y));
        let tw = mle_eval(&lifted, rwc) + zw2 * sigma.eval(rwc[nv - 1]);
        assert_ne!(
            ew,
            (FqExt::ONE - rw[nv]) * tw * mle_eval(&b, rwc)
                + indw * ClaimMask { coef: &rq }.eval(rw[nv]),
            "an unmasked claim unexpectedly passed the final equation"
        );
    }

    fn batched_w_reference(
        w0: &[FqExt],
        mut lg: Vec<FqExt>,
        apow: &[FqExt],
        tau0: &[FqExt],
        lambda: FqExt,
        sigma_w: LinMask,
        rmask: ClaimMask,
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
        let mut zpre = FqExt::ONE;
        let mut ipre = FqExt::ONE;
        let msum = rmask.total();
        let mut main_final = FqExt::ZERO;
        for round in 0..nv_w {
            let half = 1usize << (nv_w - 1 - round);
            let last = round + 1 == nv_w;
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
                let zc =
                    if last { zpre * x * (FqExt::ONE - x) * sigma_w.eval(x) } else { FqExt::ZERO };
                (0..half).fold(FqExt::ZERO, |g, cell| {
                    let (wlo, whi) = (w[cell], w[cell + half]);
                    let wx = wlo + x * (whi - wlo) + zc;
                    let bx = if round < nv_k {
                        let hk = lg.len() / 2;
                        let (k, l) = (cell / s_len, cell % s_len);
                        (lg[k] + x * (lg[k + hk] - lg[k])) * apow[l]
                    } else {
                        b[cell] + x * (b[cell + half] - b[cell])
                    };
                    g + lambda * bx * wx + eqpre * eq1x * eqrem(cell) * wx * (wx - FqExt::ONE)
                }) + ipre * (FqExt::ONE - x) * msum
            };
            let pts: &[u64] = if last { &[0, 2, 3, 4, 5, 6, 7] } else { &[0, 2, 3] };
            let evals: Vec<FqExt> =
                pts.iter().map(|&s| gx(FqExt::from_u64(s), &w, &lg, &b)).collect();
            for &e in &evals {
                tr.absorb_fq4(e);
            }
            let r = tr.challenge_fq4();
            r_point.push(r);
            eqpre = eqpre * (t * r + (FqExt::ONE - t) * (FqExt::ONE - r));
            for i in 0..half {
                w[i] = w[i] + r * (w[i + half] - w[i]);
            }
            w.truncate(half);
            zpre = zpre * r * (FqExt::ONE - r);
            ipre = ipre * (FqExt::ONE - r);
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
            if last {
                let wd = w[0] + zpre * sigma_w.eval(r_point[nv_w - 1]);
                main_final = lambda * b[0] * wd + eqpre * wd * (wd - FqExt::ONE);
            }
            rounds.push(evals);
        }
        let wall: Vec<FqExt> = [0u64, 2]
            .iter()
            .map(|&s| {
                let x = FqExt::from_u64(s);
                (FqExt::ONE - x) * main_final + ipre * rmask.eval(x)
            })
            .collect();
        for &e in &wall {
            tr.absorb_fq4(e);
        }
        let c = tr.challenge_fq4();
        r_point.push(c);
        rounds.push(wall);
        let _ = c;
        let z1 = r_point[nv_w - 1];
        (SumcheckProof { rounds }, r_point, w[0] + zpre * sigma_w.eval(z1))
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
        let sigma_w = LinMask([rng.next_fq4(), rng.next_fq4()]);

        let ncoef: [FqExt; 3] = [rng.next_fq4(), rng.next_fq4(), rng.next_fq4()];
        let nmask = ClaimMask { coef: &ncoef };
        let mut t1 = Transcript::new("ref");
        let (p1, r1, o1, _) = prove_batched_w(
            &zw,
            lg.clone(),
            &apow,
            &tau0,
            lambda,
            sigma_w,
            nmask,
            &mut Vec::new(),
            None,
            &mut t1,
        );
        let mut t2 = Transcript::new("ref");
        let (p2, r2, o2) = batched_w_reference(&w, lg, &apow, &tau0, lambda, sigma_w, nmask, &mut t2);
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

        let zr = [FqExt::ZERO; 3];
        let mut tr_p = Transcript::new("nb");
        let (proof, r_p, _) = batched_w_reference(
            &w,
            lg.clone(),
            &apow,
            &tau0,
            lambda,
            LinMask([FqExt::ZERO; 2]),
            ClaimMask { coef: &zr },
            &mut tr_p,
        );
        let degs = round_degs(nv_w, 3, 7, 2);
        let mut tr_v = Transcript::new("nb");
        let (e_b, r_v) =
            verify_batched_w(lambda * claim2, &degs, &proof, &mut tr_v).expect("well-formed");
        assert_eq!(r_p, r_v);

        let rc = &r_v[..nv_w];
        let cb = r_v[nv_w];
        let w_at = mle_eval(&w, rc);
        let lg_at = mle_eval(&lg, &r_v[..nv_k]);
        let ap_at = mle_eval(&apow, &r_v[nv_k..nv_w]);
        let expect = (FqExt::ONE - cb)
            * (lambda * lg_at * ap_at * w_at + eq_eval(&tau0, rc) * w_at * (w_at - FqExt::ONE));
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
            let degs: Vec<usize> = (0..nv).map(|_| deg).collect();
            let _ = verify_product2(claim, &degs, &p, &mut Transcript::new("f"));
            let _ = verify_batched_w(claim, &degs, &p, &mut Transcript::new("f"));
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

        let zr = [FqExt::ZERO; 3];
        let mut tr_p = Transcript::new("p2-test");
        let (proof, r_p, _, _) =
            prove_product2(&a, b.clone(), LinMask([FqExt::ZERO; 2]), ClaimMask { coef: &zr }, None, &mut tr_p);

        let degs = round_degs(nv, 2, 4, 2);
        let mut tr_v = Transcript::new("p2-test");
        let (expect, r_v) = verify_product2(claim, &degs, &proof, &mut tr_v).expect("verify");
        assert_eq!(r_p, r_v);
        let lifted: Vec<FqExt> = a.iter().map(|&x| FqExt::from_fq(x)).collect();
        let rc = &r_v[..nv];
        let cq = r_v[nv];
        assert_eq!(expect, (FqExt::ONE - cq) * mle_eval(&lifted, rc) * mle_eval(&b, rc));

        let mut tr_w = Transcript::new("p2-test");
        let (expect_w, r_w) =
            verify_product2(claim + FqExt::ONE, &degs, &proof, &mut tr_w).expect("well-formed");
        assert_ne!(
            expect_w,
            (FqExt::ONE - r_w[nv]) * mle_eval(&lifted, &r_w[..nv]) * mle_eval(&b, &r_w[..nv])
        );
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

    pub fn deg(&self) -> usize {
        self.deg
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
