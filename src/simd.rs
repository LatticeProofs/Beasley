use crate::bits::PackedBits;
use crate::ext_field::{Fq2, FqExt, EXT_DEG};
use crate::field::{reduce128, Fq, C, Q};
use rayon::prelude::*;

const QN: u64 = Q;

const RAW_ONE: u64 = 1;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn use_avx2() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        is_x86_feature_detected!("avx2") && std::env::var_os("VOPRF_NO_AVX2").is_none()
    })
}

#[inline(always)]
fn to_raw(x: Fq) -> u64 {
    x.0
}
#[inline(always)]
fn from_raw(x: u64) -> Fq {
    debug_assert!(x < QN, "from_raw received a non-canonical value");
    Fq(x)
}
#[inline(always)]
fn fq_mul(a: u64, b: u64) -> u64 {
    reduce128((a as u128) * (b as u128))
}
#[inline(always)]
fn addq(a: u64, b: u64) -> u64 {
    let (s, carry) = a.overflowing_add(b);
    let s = if carry { s.wrapping_add(C) } else { s };
    if s >= QN { s - QN } else { s }
}
#[inline(always)]
fn subq(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a.wrapping_sub(b).wrapping_add(QN) }
}

#[derive(Clone)]
pub struct SoaFqExt {
    pub c: [Vec<u64>; EXT_DEG],
}

impl SoaFqExt {
    pub fn len(&self) -> usize {
        self.c[0].len()
    }
    pub fn is_empty(&self) -> bool {
        self.c[0].is_empty()
    }
    pub fn from_aos(v: &[FqExt]) -> Self {
        let mut c: [Vec<u64>; EXT_DEG] = Default::default();
        for (k, arr) in c.iter_mut().enumerate() {
            *arr = v.iter().map(|e| to_raw(e.0[k])).collect();
        }
        SoaFqExt { c }
    }
    pub fn get(&self, i: usize) -> FqExt {
        Fq2(core::array::from_fn(|k| from_raw(self.c[k][i])))
    }
    pub fn truncate(&mut self, n: usize) {
        for arr in self.c.iter_mut() {
            arr.truncate(n);
        }
    }
}

#[inline(always)]
fn fqext_mul_scalar(a: [u64; EXT_DEG], b: [u64; EXT_DEG]) -> [u64; EXT_DEG] {
    let p00 = fq_mul(a[0], b[0]);
    let p11 = fq_mul(a[1], b[1]);
    let p01 = fq_mul(a[0], b[1]);
    let p10 = fq_mul(a[1], b[0]);
    [addq(p00, addq(p11, p11)), addq(p01, p10)]
}

#[inline(always)]
fn to_ext(a: [u64; EXT_DEG]) -> FqExt {
    Fq2(core::array::from_fn(|k| from_raw(a[k])))
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #![allow(unsafe_op_in_unsafe_fn)]

    use super::{addq, subq, EXT_DEG, RAW_ONE};
    use crate::bits::PackedBits;
    use crate::ext_field::FqExt;
    use crate::field::{C, Q};
    use core::arch::x86_64::*;

    const MASK32: i64 = 0xFFFF_FFFF;
    const SIGN: i64 = i64::MIN;

    #[target_feature(enable = "avx2")]
    fn ge_u(a: __m256i, b: __m256i) -> __m256i {
        let s = _mm256_set1_epi64x(SIGN);
        let gt = _mm256_cmpgt_epi64(_mm256_xor_si256(b, s), _mm256_xor_si256(a, s));
        _mm256_xor_si256(gt, _mm256_set1_epi64x(-1))
    }

    #[target_feature(enable = "avx2")]
    fn lt_u(a: __m256i, b: __m256i) -> __m256i {
        let s = _mm256_set1_epi64x(SIGN);
        _mm256_cmpgt_epi64(_mm256_xor_si256(b, s), _mm256_xor_si256(a, s))
    }

    #[target_feature(enable = "avx2")]
    fn mul_c(x: __m256i) -> __m256i {
        let t = _mm256_sub_epi64(_mm256_slli_epi64(x, 6), _mm256_slli_epi64(x, 2));
        _mm256_sub_epi64(t, x)
    }

    #[target_feature(enable = "avx2")]
    fn csub_q(s: __m256i) -> __m256i {
        let qv = _mm256_set1_epi64x(Q as i64);
        _mm256_blendv_epi8(s, _mm256_sub_epi64(s, qv), ge_u(s, qv))
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn add_v(a: __m256i, b: __m256i) -> __m256i {
        let s = _mm256_add_epi64(a, b);
        let carry = lt_u(s, a);
        let s = _mm256_add_epi64(s, _mm256_and_si256(carry, _mm256_set1_epi64x(C as i64)));
        csub_q(s)
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn sub_v(a: __m256i, b: __m256i) -> __m256i {
        let t = _mm256_sub_epi64(a, b);
        let borrow = lt_u(a, b);
        _mm256_sub_epi64(t, _mm256_and_si256(borrow, _mm256_set1_epi64x(C as i64)))
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn mul_v(a: __m256i, b: __m256i) -> __m256i {
        let mask = _mm256_set1_epi64x(MASK32);
        let ah = _mm256_srli_epi64(a, 32);
        let bh = _mm256_srli_epi64(b, 32);
        let p0 = _mm256_mul_epu32(a, b);
        let p1 = _mm256_mul_epu32(a, bh);
        let p2 = _mm256_mul_epu32(ah, b);
        let p3 = _mm256_mul_epu32(ah, bh);

        let s1 = _mm256_add_epi64(
            _mm256_add_epi64(_mm256_srli_epi64(p1, 32), _mm256_srli_epi64(p2, 32)),
            _mm256_and_si256(p3, mask),
        );
        let a_acc = _mm256_add_epi64(_mm256_and_si256(p0, mask), mul_c(s1));
        let b_acc = _mm256_add_epi64(
            _mm256_add_epi64(_mm256_srli_epi64(p0, 32), _mm256_and_si256(p1, mask)),
            _mm256_add_epi64(_mm256_and_si256(p2, mask), mul_c(_mm256_srli_epi64(p3, 32))),
        );
        let a2 = _mm256_add_epi64(a_acc, mul_c(_mm256_srli_epi64(b_acc, 32)));
        let shifted = _mm256_slli_epi64(_mm256_and_si256(b_acc, mask), 32);
        let s = _mm256_add_epi64(shifted, a2);
        let carry = lt_u(s, shifted);
        let s = _mm256_add_epi64(s, _mm256_and_si256(carry, _mm256_set1_epi64x(C as i64)));
        csub_q(s)
    }

    #[target_feature(enable = "avx2")]
    fn ext_mul_v(a: [__m256i; 2], b: [__m256i; 2]) -> [__m256i; 2] {
        let p11 = mul_v(a[1], b[1]);
        [
            add_v(mul_v(a[0], b[0]), add_v(p11, p11)),
            add_v(mul_v(a[0], b[1]), mul_v(a[1], b[0])),
        ]
    }

    #[target_feature(enable = "avx2")]
    unsafe fn load(p: &[u64], i: usize) -> __m256i {
        debug_assert!(i + 4 <= p.len(), "AVX2 load out of bounds: i = {i}, len = {}", p.len());
        unsafe { _mm256_loadu_si256(p.as_ptr().add(i) as *const __m256i) }
    }

    #[target_feature(enable = "avx2")]
    fn hsum(v: __m256i) -> u64 {
        let mut t = [0u64; 4];
        unsafe { _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v) };
        addq(addq(t[0], t[1]), addq(t[2], t[3]))
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fold(ext: &mut super::SoaFqExt, half: usize, rm: [u64; EXT_DEG]) {
        let rv = [_mm256_set1_epi64x(rm[0] as i64), _mm256_set1_epi64x(rm[1] as i64)];
        let n4 = half & !3;
        for i in (0..n4).step_by(4) {
            let lo = [load(&ext.c[0], i), load(&ext.c[1], i)];
            let hi = [load(&ext.c[0], i + half), load(&ext.c[1], i + half)];
            let d = [sub_v(hi[0], lo[0]), sub_v(hi[1], lo[1])];
            let rd = ext_mul_v(rv, d);
            for k in 0..EXT_DEG {
                _mm256_storeu_si256(
                    ext.c[k].as_mut_ptr().add(i) as *mut __m256i,
                    add_v(lo[k], rd[k]),
                );
            }
        }
        for i in n4..half {
            let lo: [u64; EXT_DEG] = core::array::from_fn(|k| ext.c[k][i]);
            let hi: [u64; EXT_DEG] = core::array::from_fn(|k| ext.c[k][i + half]);
            let d: [u64; EXT_DEG] = core::array::from_fn(|k| subq(hi[k], lo[k]));
            let rd = super::fqext_mul_scalar(rm, d);
            for k in 0..EXT_DEG {
                ext.c[k][i] = addq(lo[k], rd[k]);
            }
        }
        ext.truncate(half);
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn batched_phase_a_round(
        w: &super::SoaFqExt,
        apow: &super::SoaFqExt,
        ea: &super::SoaFqExt,
        eb: &super::SoaFqExt,
        lg: &[FqExt],
        half: usize,
        s_len: usize,
        k_lo: usize,
        k_hi: usize,
    ) -> ([FqExt; 3], [FqExt; 3]) {
        let half_k = lg.len() / 2;
        let mut gf2 = [FqExt::ZERO; 3];
        let mut q3 = [FqExt::ZERO; 3];
        let one = _mm256_set1_epi64x(RAW_ONE as i64);
        let n4 = s_len & !3;

        for k in k_lo..k_hi {
            let ea_k: [u64; EXT_DEG] = core::array::from_fn(|j| ea.c[j][k]);
            let mut rd = [[_mm256_setzero_si256(); EXT_DEG]; 3];
            let mut qk = [[_mm256_setzero_si256(); EXT_DEG]; 3];
            let base = k * s_len;

            for l in (0..n4).step_by(4) {
                let cell = base + l;
                let lo = [load(&w.c[0], cell), load(&w.c[1], cell)];
                let hi = [load(&w.c[0], cell + half), load(&w.c[1], cell + half)];
                let ap = [load(&apow.c[0], l), load(&apow.c[1], l)];
                let ebl = [load(&eb.c[0], l), load(&eb.c[1], l)];

                let d = [sub_v(hi[0], lo[0]), sub_v(hi[1], lo[1])];
                let w2 = [add_v(hi[0], d[0]), add_v(hi[1], d[1])];
                let w3 = [add_v(w2[0], d[0]), add_v(w2[1], d[1])];

                let f2_0 = ext_mul_v(ap, lo);
                let f2_i = ext_mul_v(ap, d);
                for j in 0..EXT_DEG {
                    let f2_2 = add_v(f2_0[j], add_v(f2_i[j], f2_i[j]));
                    let f2_3 = add_v(f2_2, f2_i[j]);
                    rd[0][j] = add_v(rd[0][j], f2_0[j]);
                    rd[1][j] = add_v(rd[1][j], f2_2);
                    rd[2][j] = add_v(rd[2][j], f2_3);
                }
                for (xi, wv) in [lo, w2, w3].into_iter().enumerate() {
                    let wm1 = [sub_v(wv[0], one), wv[1]];
                    let p = ext_mul_v(wv, wm1);
                    let f3 = ext_mul_v(ebl, p);
                    for j in 0..EXT_DEG {
                        qk[xi][j] = add_v(qk[xi][j], f3[j]);
                    }
                }
            }

            let mut rds = [[0u64; EXT_DEG]; 3];
            let mut qks = [[0u64; EXT_DEG]; 3];
            for x in 0..3 {
                for j in 0..EXT_DEG {
                    rds[x][j] = hsum(rd[x][j]);
                    qks[x][j] = hsum(qk[x][j]);
                }
            }
            for l in n4..s_len {
                let cell = base + l;
                let lo: [u64; EXT_DEG] = core::array::from_fn(|j| w.c[j][cell]);
                let hi: [u64; EXT_DEG] = core::array::from_fn(|j| w.c[j][cell + half]);
                let ap: [u64; EXT_DEG] = core::array::from_fn(|j| apow.c[j][l]);
                let ebl: [u64; EXT_DEG] = core::array::from_fn(|j| eb.c[j][l]);
                let d: [u64; EXT_DEG] = core::array::from_fn(|j| subq(hi[j], lo[j]));
                let w2: [u64; EXT_DEG] = core::array::from_fn(|j| addq(hi[j], d[j]));
                let w3: [u64; EXT_DEG] = core::array::from_fn(|j| addq(w2[j], d[j]));
                let f2_0 = super::fqext_mul_scalar(ap, lo);
                let f2_i = super::fqext_mul_scalar(ap, d);
                for j in 0..EXT_DEG {
                    let f2_2 = addq(f2_0[j], addq(f2_i[j], f2_i[j]));
                    rds[0][j] = addq(rds[0][j], f2_0[j]);
                    rds[1][j] = addq(rds[1][j], f2_2);
                    rds[2][j] = addq(rds[2][j], addq(f2_2, f2_i[j]));
                }
                for (xi, wv) in [lo, w2, w3].into_iter().enumerate() {
                    let mut wm1 = wv;
                    wm1[0] = subq(wv[0], RAW_ONE);
                    let f3 = super::fqext_mul_scalar(ebl, super::fqext_mul_scalar(wv, wm1));
                    for j in 0..EXT_DEG {
                        qks[xi][j] = addq(qks[xi][j], f3[j]);
                    }
                }
            }

            let lg_lo = lg[k];
            let lg_hi = lg[k + half_k];
            let dl = lg_hi - lg_lo;
            let lg_at = [lg_lo, lg_hi + dl, lg_hi + dl + dl];
            for xi in 0..3 {
                gf2[xi] = gf2[xi] + lg_at[xi] * super::to_ext(rds[xi]);
                q3[xi] = q3[xi] + super::to_ext(super::fqext_mul_scalar(ea_k, qks[xi]));
            }
        }
        (gf2, q3)
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn batched_phase_a_round0(
        zw: &PackedBits,
        apow: &super::SoaFqExt,
        ea: &super::SoaFqExt,
        eb: &super::SoaFqExt,
        lg: &[FqExt],
        half: usize,
        s_len: usize,
        k_lo: usize,
        k_hi: usize,
    ) -> ([FqExt; 3], [FqExt; 3]) {
        let half_k = lg.len() / 2;
        let mut gf2 = [FqExt::ZERO; 3];
        let mut q3 = [FqExt::ZERO; 3];
        let n4 = s_len & !3;
        let zero = _mm256_setzero_si256();

        for k in k_lo..k_hi {
            let ea_k: [u64; EXT_DEG] = core::array::from_fn(|j| ea.c[j][k]);
            let mut rd = [[zero; EXT_DEG]; 3];
            let mut q2 = [zero; EXT_DEG];
            let mut q6 = [zero; EXT_DEG];
            let base = k * s_len;

            for l in (0..n4).step_by(4) {
                let cell = base + l;
                let mut lo_m = [0i64; 4];
                let mut hi_m = [0i64; 4];
                let mut df_m = [0i64; 4];
                for t in 0..4 {
                    let a = zw.get(cell + t);
                    let b = zw.get(cell + t + half);
                    lo_m[t] = if a { -1 } else { 0 };
                    hi_m[t] = if b { -1 } else { 0 };
                    df_m[t] = if a != b { -1 } else { 0 };
                }
                let lom = _mm256_loadu_si256(lo_m.as_ptr() as *const __m256i);
                let him = _mm256_loadu_si256(hi_m.as_ptr() as *const __m256i);
                let dfm = _mm256_loadu_si256(df_m.as_ptr() as *const __m256i);

                for j in 0..EXT_DEG {
                    let ap = load(&apow.c[j], l);
                    let apl = _mm256_and_si256(ap, lom);
                    let aph = _mm256_and_si256(ap, him);
                    let two_h = add_v(aph, aph);
                    let f2_2 = sub_v(two_h, apl);
                    let f2_3 = sub_v(add_v(two_h, aph), add_v(apl, apl));
                    rd[0][j] = add_v(rd[0][j], apl);
                    rd[1][j] = add_v(rd[1][j], f2_2);
                    rd[2][j] = add_v(rd[2][j], f2_3);

                    let e = _mm256_and_si256(load(&eb.c[j], l), dfm);
                    let e2 = add_v(e, e);
                    q2[j] = add_v(q2[j], e2);
                    q6[j] = add_v(q6[j], add_v(e2, add_v(e2, e2)));
                }
            }

            let mut rds = [[0u64; EXT_DEG]; 3];
            let mut qks = [[0u64; EXT_DEG]; 3];
            for x in 0..3 {
                for j in 0..EXT_DEG {
                    rds[x][j] = hsum(rd[x][j]);
                }
            }
            for j in 0..EXT_DEG {
                qks[1][j] = hsum(q2[j]);
                qks[2][j] = hsum(q6[j]);
            }
            for l in n4..s_len {
                let cell = base + l;
                let a = zw.get(cell);
                let b = zw.get(cell + half);
                for j in 0..EXT_DEG {
                    let ap = apow.c[j][l];
                    let apl = if a { ap } else { 0 };
                    let aph = if b { ap } else { 0 };
                    let two_h = addq(aph, aph);
                    rds[0][j] = addq(rds[0][j], apl);
                    rds[1][j] = addq(rds[1][j], subq(two_h, apl));
                    rds[2][j] =
                        addq(rds[2][j], subq(addq(two_h, aph), addq(apl, apl)));
                    if a != b {
                        let e = eb.c[j][l];
                        let e2 = addq(e, e);
                        qks[1][j] = addq(qks[1][j], e2);
                        qks[2][j] = addq(qks[2][j], addq(e2, addq(e2, e2)));
                    }
                }
            }

            let lg_lo = lg[k];
            let lg_hi = lg[k + half_k];
            let dl = lg_hi - lg_lo;
            let lg_at = [lg_lo, lg_hi + dl, lg_hi + dl + dl];
            for xi in 0..3 {
                gf2[xi] = gf2[xi] + lg_at[xi] * super::to_ext(rds[xi]);
                q3[xi] = q3[xi] + super::to_ext(super::fqext_mul_scalar(ea_k, qks[xi]));
            }
        }
        (gf2, q3)
    }

    #[cfg(test)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn prim_probe(op: u8, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let av = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let bv = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
        let r = match op {
            0 => add_v(av, bv),
            1 => sub_v(av, bv),
            _ => mul_v(av, bv),
        };
        let mut out = [0u64; 4];
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, r);
        out
    }
}

#[inline]
fn split_k<F>(half_k: usize, cells: usize, f: F) -> ([FqExt; 3], [FqExt; 3])
where
    F: Fn(usize, usize) -> ([FqExt; 3], [FqExt; 3]) + Sync + Send,
{
    const PAR_MIN_CELLS: usize = 1 << 14;
    if cells < PAR_MIN_CELLS || half_k < 2 {
        return f(0, half_k);
    }
    let chunk = (half_k / (4 * rayon::current_num_threads())).max(1);
    let nch = half_k.div_ceil(chunk);
    let parts: Vec<([FqExt; 3], [FqExt; 3])> = (0..nch)
        .into_par_iter()
        .map(|c| f(c * chunk, ((c + 1) * chunk).min(half_k)))
        .collect();
    let mut gf2 = [FqExt::ZERO; 3];
    let mut q3 = [FqExt::ZERO; 3];
    for (a, b) in parts {
        for x in 0..3 {
            gf2[x] = gf2[x] + a[x];
            q3[x] = q3[x] + b[x];
        }
    }
    (gf2, q3)
}

pub fn batched_phase_a_round(
    w: &SoaFqExt,
    apow: &SoaFqExt,
    ea: &SoaFqExt,
    eb: &SoaFqExt,
    lg: &[FqExt],
    half: usize,
    s_len: usize,
) -> ([FqExt; 3], [FqExt; 3]) {
    split_k(lg.len() / 2, half, |lo, hi| {
        #[cfg(target_arch = "x86_64")]
        {
            if use_avx2() {
                return unsafe {
                    avx2::batched_phase_a_round(w, apow, ea, eb, lg, half, s_len, lo, hi)
                };
            }
        }
        batched_phase_a_round_scalar(w, apow, ea, eb, lg, half, s_len, lo, hi)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn batched_phase_a_round_scalar(
    w: &SoaFqExt,
    apow: &SoaFqExt,
    ea: &SoaFqExt,
    eb: &SoaFqExt,
    lg: &[FqExt],
    half: usize,
    s_len: usize,
    k_lo: usize,
    k_hi: usize,
) -> ([FqExt; 3], [FqExt; 3]) {
    let half_k = lg.len() / 2;
    let mut gf2 = [FqExt::ZERO; 3];
    let mut q3 = [FqExt::ZERO; 3];
    let getm = |t: &SoaFqExt, i: usize| -> [u64; EXT_DEG] { core::array::from_fn(|k| t.c[k][i]) };
    for k in k_lo..k_hi {
        let ea_k = getm(ea, k);
        let mut rd = [[0u64; EXT_DEG]; 3];
        let mut qk = [[0u64; EXT_DEG]; 3];
        for l in 0..s_len {
            let cell = k * s_len + l;
            let lo = getm(w, cell);
            let hi = getm(w, cell + half);
            let ap = getm(apow, l);
            let ebl = getm(eb, l);
            let d: [u64; EXT_DEG] = core::array::from_fn(|j| subq(hi[j], lo[j]));
            let w2: [u64; EXT_DEG] = core::array::from_fn(|j| addq(hi[j], d[j]));
            let w3: [u64; EXT_DEG] = core::array::from_fn(|j| addq(w2[j], d[j]));
            let f2_0 = fqext_mul_scalar(ap, lo);
            let f2_i = fqext_mul_scalar(ap, d);
            for j in 0..EXT_DEG {
                let f2_2 = addq(f2_0[j], addq(f2_i[j], f2_i[j]));
                let f2_3 = addq(f2_2, f2_i[j]);
                rd[0][j] = addq(rd[0][j], f2_0[j]);
                rd[1][j] = addq(rd[1][j], f2_2);
                rd[2][j] = addq(rd[2][j], f2_3);
            }
            for (xi, wv) in [lo, w2, w3].into_iter().enumerate() {
                let mut wm1 = wv;
                wm1[0] = subq(wv[0], RAW_ONE);
                let p = fqext_mul_scalar(wv, wm1);
                let f3 = fqext_mul_scalar(ebl, p);
                for j in 0..EXT_DEG {
                    qk[xi][j] = addq(qk[xi][j], f3[j]);
                }
            }
        }
        let lg_lo = lg[k];
        let lg_hi = lg[k + half_k];
        let dl = lg_hi - lg_lo;
        let lg_at = [lg_lo, lg_hi + dl, lg_hi + dl + dl];
        for xi in 0..3 {
            gf2[xi] = gf2[xi] + lg_at[xi] * to_ext(rd[xi]);
            q3[xi] = q3[xi] + to_ext(fqext_mul_scalar(ea_k, qk[xi]));
        }
    }
    (gf2, q3)
}

pub fn batched_phase_a_round0(
    zw: &PackedBits,
    apow: &SoaFqExt,
    ea: &SoaFqExt,
    eb: &SoaFqExt,
    lg: &[FqExt],
    half: usize,
    s_len: usize,
) -> ([FqExt; 3], [FqExt; 3]) {
    split_k(lg.len() / 2, half, |lo, hi| {
        #[cfg(target_arch = "x86_64")]
        {
            if use_avx2() {
                return unsafe {
                    avx2::batched_phase_a_round0(zw, apow, ea, eb, lg, half, s_len, lo, hi)
                };
            }
        }
        batched_phase_a_round0_scalar(zw, apow, ea, eb, lg, half, s_len, lo, hi)
    })
}

#[inline]
pub fn fold_bits_lut(r: FqExt) -> [FqExt; 4] {
    [FqExt::ZERO, r, FqExt::ONE - r, FqExt::ONE]
}

pub fn fold_bits_to_soa(zw: &PackedBits, half: usize, r: FqExt) -> SoaFqExt {
    let lut = fold_bits_lut(r);
    let mut c: [Vec<u64>; EXT_DEG] = Default::default();
    c.par_iter_mut().enumerate().for_each(|(k, arr)| {
        *arr = (0..half)
            .into_par_iter()
            .map(|cell| {
                let idx = ((zw.get(cell) as usize) << 1) | zw.get(cell + half) as usize;
                to_raw(lut[idx].0[k])
            })
            .collect();
    });
    SoaFqExt { c }
}

#[allow(clippy::too_many_arguments)]
pub fn batched_phase_a_round0_scalar(
    zw: &PackedBits,
    apow: &SoaFqExt,
    ea: &SoaFqExt,
    eb: &SoaFqExt,
    lg: &[FqExt],
    half: usize,
    s_len: usize,
    k_lo: usize,
    k_hi: usize,
) -> ([FqExt; 3], [FqExt; 3]) {
    let half_k = lg.len() / 2;
    let mut gf2 = [FqExt::ZERO; 3];
    let mut q3 = [FqExt::ZERO; 3];
    let getm = |t: &SoaFqExt, i: usize| -> [u64; EXT_DEG] { core::array::from_fn(|k| t.c[k][i]) };
    for k in k_lo..k_hi {
        let ea_k = getm(ea, k);
        let mut rd = [[0u64; EXT_DEG]; 3];
        let mut qk = [[0u64; EXT_DEG]; 3];
        for l in 0..s_len {
            let cell = k * s_len + l;
            let lo = if zw.get(cell) { RAW_ONE } else { 0 };
            let hi = if zw.get(cell + half) { RAW_ONE } else { 0 };
            let d = subq(hi, lo);
            let w2 = addq(hi, d);
            let w3 = addq(w2, d);
            let ap = getm(apow, l);
            let ebl = getm(eb, l);
            for j in 0..EXT_DEG {
                let f2_0 = fq_mul(ap[j], lo);
                let f2_i = fq_mul(ap[j], d);
                let f2_2 = addq(f2_0, addq(f2_i, f2_i));
                let f2_3 = addq(f2_2, f2_i);
                rd[0][j] = addq(rd[0][j], f2_0);
                rd[1][j] = addq(rd[1][j], f2_2);
                rd[2][j] = addq(rd[2][j], f2_3);
            }
            for (xi, wx) in [lo, w2, w3].into_iter().enumerate() {
                let p = fq_mul(wx, subq(wx, RAW_ONE));
                for j in 0..EXT_DEG {
                    qk[xi][j] = addq(qk[xi][j], fq_mul(ebl[j], p));
                }
            }
        }
        let lg_lo = lg[k];
        let lg_hi = lg[k + half_k];
        let dl = lg_hi - lg_lo;
        let lg_at = [lg_lo, lg_hi + dl, lg_hi + dl + dl];
        for xi in 0..3 {
            gf2[xi] = gf2[xi] + lg_at[xi] * to_ext(rd[xi]);
            q3[xi] = q3[xi] + to_ext(fqext_mul_scalar(ea_k, qk[xi]));
        }
    }
    (gf2, q3)
}

pub fn fold(ext: &mut SoaFqExt, half: usize, r: FqExt) {
    let rm: [u64; EXT_DEG] = core::array::from_fn(|k| to_raw(r.0[k]));
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx2() {
            unsafe { avx2::fold(ext, half, rm) };
            return;
        }
    }
    fold_scalar(ext, half, rm);
}

fn fold_scalar(ext: &mut SoaFqExt, half: usize, rm: [u64; EXT_DEG]) {
    for i in 0..half {
        let lo: [u64; EXT_DEG] = core::array::from_fn(|k| ext.c[k][i]);
        let hi: [u64; EXT_DEG] = core::array::from_fn(|k| ext.c[k][i + half]);
        let d: [u64; EXT_DEG] = core::array::from_fn(|k| subq(hi[k], lo[k]));
        let rd = fqext_mul_scalar(rm, d);
        for k in 0..EXT_DEG {
            ext.c[k][i] = addq(lo[k], rd[k]);
        }
    }
    ext.truncate(half);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mle::mle_eval;
    use crate::transcript::SimpleRng;

    fn rand_ext(rng: &mut SimpleRng) -> FqExt {
        rng.next_fq4()
    }

    #[test]
    fn plain_form_roundtrip_and_mul() {
        assert_eq!(RAW_ONE, to_raw(Fq::ONE));
        let mut rng = SimpleRng::new(1);
        for _ in 0..5000 {
            let a = rng.next_fq();
            let b = rng.next_fq();
            assert_eq!(from_raw(to_raw(a)), a);
            assert_eq!(from_raw(fq_mul(to_raw(a), to_raw(b))), a * b);
        }
    }

    #[test]
    fn addq_subq_handle_overflow() {
        let m = Q - 1;
        assert_eq!(addq(m, m), (Fq(m) + Fq(m)).0);
        assert_eq!(addq(m, 1), 0);
        assert_eq!(subq(0, 1), m);
        assert_eq!(subq(1, m), (Fq(1) - Fq(m)).0);
        let mut rng = SimpleRng::new(77);
        for _ in 0..20000 {
            let a = rng.next_fq();
            let b = rng.next_fq();
            assert_eq!(addq(a.0, b.0), (a + b).0);
            assert_eq!(subq(a.0, b.0), (a - b).0);
        }
    }

    #[test]
    fn kernel_mul_matches_ext_field() {
        let mut rng = SimpleRng::new(2);
        for _ in 0..20000 {
            let a = rand_ext(&mut rng);
            let b = rand_ext(&mut rng);
            let got = fqext_mul_scalar(
                core::array::from_fn(|k| a.0[k].0),
                core::array::from_fn(|k| b.0[k].0),
            );
            assert_eq!(to_ext(got), a * b);
        }
    }

    #[test]
    fn fold_matches_naive() {
        let mut rng = SimpleRng::new(4);
        let nv = 8;
        let tbl: Vec<FqExt> = (0..1 << nv).map(|_| rand_ext(&mut rng)).collect();
        let r = rand_ext(&mut rng);
        let mut soa = SoaFqExt::from_aos(&tbl);
        let half = tbl.len() / 2;
        fold(&mut soa, half, r);
        for i in 0..half {
            assert_eq!(soa.get(i), tbl[i] + r * (tbl[i + half] - tbl[i]), "cell {i}");
        }
    }

    #[test]
    fn fold_bits_to_soa_matches_general_fold() {
        let mut rng = SimpleRng::new(5);
        let nv = 10;
        let len = 1usize << nv;
        let mut zw = PackedBits::zeros(len);
        let mut aos = Vec::with_capacity(len);
        for i in 0..len {
            let b = rng.next_bool();
            if b {
                zw.set(i);
            }
            aos.push(if b { FqExt::ONE } else { FqExt::ZERO });
        }
        let r = rand_ext(&mut rng);
        let half = len / 2;
        let got = fold_bits_to_soa(&zw, half, r);
        let mut want = SoaFqExt::from_aos(&aos);
        fold(&mut want, half, r);
        for i in 0..half {
            assert_eq!(got.get(i), want.get(i), "cell {i}");
        }
    }

    #[test]
    fn round0_matches_general_round_on_bits() {
        let mut rng = SimpleRng::new(9);
        let (k_len, s_len) = (8usize, 16usize);
        let cells = k_len * s_len;
        let len = cells * 2;
        let mut zw = PackedBits::zeros(len);
        let mut aos = Vec::with_capacity(len);
        for i in 0..len {
            let b = rng.next_bool();
            if b {
                zw.set(i);
            }
            aos.push(if b { FqExt::ONE } else { FqExt::ZERO });
        }
        let w = SoaFqExt::from_aos(&aos);
        let apow = SoaFqExt::from_aos(&(0..s_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
        let ea = SoaFqExt::from_aos(&(0..k_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
        let eb = SoaFqExt::from_aos(&(0..s_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
        let lg: Vec<FqExt> = (0..2 * k_len).map(|_| rand_ext(&mut rng)).collect();
        let a = batched_phase_a_round0_scalar(&zw, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len);
        let b = batched_phase_a_round_scalar(&w, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len);
        assert_eq!(a, b);
    }

    #[test]
    fn split_k_is_partition_invariant() {
        let mut rng = SimpleRng::new(10);
        let (k_len, s_len) = (64usize, 64usize);
        let cells = k_len * s_len;
        let aos: Vec<FqExt> = (0..cells * 2).map(|_| rand_ext(&mut rng)).collect();
        let w = SoaFqExt::from_aos(&aos);
        let apow = SoaFqExt::from_aos(&(0..s_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
        let ea = SoaFqExt::from_aos(&(0..k_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
        let eb = SoaFqExt::from_aos(&(0..s_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
        let lg: Vec<FqExt> = (0..2 * k_len).map(|_| rand_ext(&mut rng)).collect();
        let whole = batched_phase_a_round_scalar(&w, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len);
        let par = batched_phase_a_round(&w, &apow, &ea, &eb, &lg, cells, s_len);
        assert_eq!(whole, par);
    }

    #[cfg(target_arch = "x86_64")]
    fn have_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_primitives_match_scalar_on_adversarial_values() {
        if !have_avx2() {
            eprintln!("skipped: AVX2 not available");
            return;
        }
        let edge: Vec<u64> = vec![
            0,
            1,
            2,
            (1u64 << 63) - 1,
            1u64 << 63,
            (1u64 << 63) + 1,
            Q - 3,
            Q - 2,
            Q - 1,
            Q / 2,
            Q / 2 + 1,
            (1u64 << 32) - 1,
            1u64 << 32,
            0xFFFF_FFFF_FFFF_FFFF % Q,
        ];
        let mut cases: Vec<(u64, u64)> = Vec::new();
        for &a in &edge {
            for &b in &edge {
                cases.push((a % Q, b % Q));
            }
        }
        let mut rng = SimpleRng::new(31337);
        for _ in 0..20000 {
            cases.push((rng.next_fq().0, rng.next_fq().0));
        }
        for chunk in cases.chunks(4) {
            let mut a = [0u64; 4];
            let mut b = [0u64; 4];
            for (i, &(x, y)) in chunk.iter().enumerate() {
                a[i] = x;
                b[i] = y;
            }
            let got_add = unsafe { avx2::prim_probe(0, a, b) };
            let got_sub = unsafe { avx2::prim_probe(1, a, b) };
            let got_mul = unsafe { avx2::prim_probe(2, a, b) };
            for i in 0..4 {
                assert_eq!(got_add[i], addq(a[i], b[i]), "add_v: a={} b={}", a[i], b[i]);
                assert_eq!(got_sub[i], subq(a[i], b[i]), "sub_v: a={} b={}", a[i], b[i]);
                assert_eq!(got_mul[i], fq_mul(a[i], b[i]), "mul_v: a={} b={}", a[i], b[i]);
                assert!(got_add[i] < Q && got_sub[i] < Q && got_mul[i] < Q);
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_kernels_match_scalar() {
        if !have_avx2() {
            eprintln!("skipped: AVX2 not available");
            return;
        }
        let mut rng = SimpleRng::new(20260802);

        for nv in [1usize, 2, 3, 5, 9] {
            let len = 1usize << nv;
            let tbl: Vec<FqExt> = (0..len).map(|_| rand_ext(&mut rng)).collect();
            let r = rand_ext(&mut rng);
            let half = len / 2;
            let mut a = SoaFqExt::from_aos(&tbl);
            let mut b = SoaFqExt::from_aos(&tbl);
            let rm: [u64; EXT_DEG] = core::array::from_fn(|k| r.0[k].0);
            unsafe { avx2::fold(&mut a, half, rm) };
            fold_scalar(&mut b, half, rm);
            for i in 0..half {
                assert_eq!(a.get(i), b.get(i), "fold nv={nv} cell={i}");
            }
        }

        for (k_len, s_len) in [(8usize, 16usize), (4, 8), (3, 6), (5, 4)] {
            let cells = k_len * s_len;
            let len = cells * 2;
            let mut zw = PackedBits::zeros(len);
            let mut aos = Vec::with_capacity(len);
            for i in 0..len {
                let b = rng.next_bool();
                if b {
                    zw.set(i);
                }
                aos.push(if b { FqExt::ONE } else { FqExt::ZERO });
            }
            let w_bits = SoaFqExt::from_aos(&aos);
            let w_rand =
                SoaFqExt::from_aos(&(0..len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
            let apow =
                SoaFqExt::from_aos(&(0..s_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
            let ea =
                SoaFqExt::from_aos(&(0..k_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
            let eb =
                SoaFqExt::from_aos(&(0..s_len).map(|_| rand_ext(&mut rng)).collect::<Vec<_>>());
            let lg: Vec<FqExt> = (0..2 * k_len).map(|_| rand_ext(&mut rng)).collect();

            assert_eq!(
                unsafe {
                    avx2::batched_phase_a_round(
                        &w_rand, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len,
                    )
                },
                batched_phase_a_round_scalar(
                    &w_rand, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len
                ),
                "round k={k_len} s={s_len}"
            );
            assert_eq!(
                unsafe {
                    avx2::batched_phase_a_round0(
                        &zw, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len,
                    )
                },
                batched_phase_a_round0_scalar(&zw, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len),
                "round0 k={k_len} s={s_len}"
            );
            assert_eq!(
                unsafe {
                    avx2::batched_phase_a_round0(
                        &zw, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len,
                    )
                },
                batched_phase_a_round_scalar(
                    &w_bits, &apow, &ea, &eb, &lg, cells, s_len, 0, k_len
                ),
                "round0 specialization disagrees with the general round k={k_len} s={s_len}"
            );
        }
    }

    #[test]
    fn soa_roundtrip() {
        let mut rng = SimpleRng::new(11);
        let tbl: Vec<FqExt> = (0..64).map(|_| rand_ext(&mut rng)).collect();
        let soa = SoaFqExt::from_aos(&tbl);
        assert_eq!(soa.len(), tbl.len());
        for i in 0..tbl.len() {
            assert_eq!(soa.get(i), tbl[i]);
        }
        let pt: Vec<FqExt> = (0..6).map(|_| rand_ext(&mut rng)).collect();
        let _ = mle_eval(&tbl, &pt);
    }
}
