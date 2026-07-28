use crate::bits::PackedBits;
use crate::ext_field::Fq4;
use crate::field::{reduce64, Fq, C, Q};
use rayon::prelude::*;

const QN: u64 = Q;
const NPRIME: u64 = 0;

#[inline(always)]
pub fn to_mont(x: Fq) -> u64 {
    x.0 as u64
}
#[inline(always)]
pub fn from_mont(x: u64) -> Fq {
    Fq(x as u32)
}
#[inline(always)]
fn mont_mul(a: u64, b: u64) -> u64 {
    reduce64(a * b) as u64
}
#[inline(always)]
fn addq(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= QN { s - QN } else { s }
}
#[inline(always)]
fn subq(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a + QN - b }
}

#[derive(Clone)]
pub struct SoaFq4 {
    pub c: [Vec<u64>; 4],
}

impl SoaFq4 {
    pub fn len(&self) -> usize {
        self.c[0].len()
    }
    pub fn from_aos(v: &[Fq4]) -> Self {
        let mut c: [Vec<u64>; 4] = Default::default();
        for k in 0..4 {
            c[k] = v.iter().map(|e| to_mont(e.0[k])).collect();
        }
        SoaFq4 { c }
    }
    pub fn get(&self, i: usize) -> Fq4 {
        Fq4([
            from_mont(self.c[0][i]),
            from_mont(self.c[1][i]),
            from_mont(self.c[2][i]),
            from_mont(self.c[3][i]),
        ])
    }
    pub fn truncate(&mut self, n: usize) {
        for k in 0..4 {
            self.c[k].truncate(n);
        }
    }
}

#[inline(always)]
fn fq4_mul_scalar(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let pr = |p: u64| (p & 0xFFFF_FFFF) + C * (p >> 32);
    let m = |i: usize, j: usize| pr(a[i] * b[j]);
    let o0 = reduce64(m(0, 0) + 2 * (m(1, 3) + m(2, 2) + m(3, 1))) as u64;
    let o1 = reduce64(m(0, 1) + m(1, 0) + 2 * (m(2, 3) + m(3, 2))) as u64;
    let o2 = reduce64(m(0, 2) + m(1, 1) + m(2, 0) + 2 * m(3, 3)) as u64;
    let o3 = reduce64(m(0, 3) + m(1, 2) + m(2, 1) + m(3, 0)) as u64;
    [o0, o1, o2, o3]
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #![allow(unsafe_op_in_unsafe_fn)]
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn splat(x: u64) -> __m256i {
        _mm256_set1_epi64x(x as i64)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mmul(a: __m256i, b: __m256i, q: __m256i, _np: __m256i) -> __m256i {
        let mask = _mm256_set1_epi64x(0xFFFF_FFFF);
        let c99 = _mm256_set1_epi64x(99);
        let t = _mm256_mul_epu32(a, b);
        let v = _mm256_add_epi64(_mm256_and_si256(t, mask), _mm256_mul_epu32(_mm256_srli_epi64(t, 32), c99));
        let v = _mm256_add_epi64(_mm256_and_si256(v, mask), _mm256_mul_epu32(_mm256_srli_epi64(v, 32), c99));
        let keep = _mm256_cmpgt_epi64(q, v);
        _mm256_blendv_epi8(_mm256_sub_epi64(v, q), v, keep)
    }
    #[target_feature(enable = "avx2")]
    unsafe fn madd(a: __m256i, b: __m256i, q: __m256i) -> __m256i {
        let s = _mm256_add_epi64(a, b);
        let sub = _mm256_sub_epi64(s, q);
        let keep = _mm256_cmpgt_epi64(q, s);
        _mm256_blendv_epi8(sub, s, keep)
    }
    #[target_feature(enable = "avx2")]
    unsafe fn msub(a: __m256i, b: __m256i, q: __m256i) -> __m256i {
        let d = _mm256_sub_epi64(a, b);
        let dq = _mm256_add_epi64(d, q);
        let blt = _mm256_cmpgt_epi64(b, a);
        _mm256_blendv_epi8(d, dq, blt)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn fq4_mul(a: &[__m256i; 4], b: &[__m256i; 4], q: __m256i, _np: __m256i) -> [__m256i; 4] {
        let mask = _mm256_set1_epi64x(0xFFFF_FFFF);
        let c99 = _mm256_set1_epi64x(99);
        let pr = |p: __m256i| {
            _mm256_add_epi64(_mm256_and_si256(p, mask), _mm256_mul_epu32(_mm256_srli_epi64(p, 32), c99))
        };
        let prod = |i: usize, j: usize| pr(_mm256_mul_epu32(a[i], b[j]));
        let full = |acc: __m256i| {
            let v = pr(acc);
            let keep = _mm256_cmpgt_epi64(q, v);
            _mm256_blendv_epi8(_mm256_sub_epi64(v, q), v, keep)
        };
        let dbl = |x: __m256i| _mm256_add_epi64(x, x);
        let a3 = |x: __m256i, y: __m256i, z: __m256i| _mm256_add_epi64(_mm256_add_epi64(x, y), z);
        let lo0 = prod(0, 0);
        let hi0 = a3(prod(1, 3), prod(2, 2), prod(3, 1));
        let lo1 = _mm256_add_epi64(prod(0, 1), prod(1, 0));
        let hi1 = _mm256_add_epi64(prod(2, 3), prod(3, 2));
        let lo2 = a3(prod(0, 2), prod(1, 1), prod(2, 0));
        let hi2 = prod(3, 3);
        let lo3 = _mm256_add_epi64(_mm256_add_epi64(prod(0, 3), prod(1, 2)), _mm256_add_epi64(prod(2, 1), prod(3, 0)));
        [
            full(_mm256_add_epi64(lo0, dbl(hi0))),
            full(_mm256_add_epi64(lo1, dbl(hi1))),
            full(_mm256_add_epi64(lo2, dbl(hi2))),
            full(lo3),
        ]
    }

    #[inline(always)]
    unsafe fn load4(s: &[u64], i: usize) -> __m256i {
        _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i)
    }
    #[inline(always)]
    unsafe fn store4(s: &mut [u64], i: usize, v: __m256i) {
        _mm256_storeu_si256(s.as_mut_ptr().add(i) as *mut __m256i, v);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn bitcheck_evals(ext: &SoaFq4, eqsuf: &SoaFq4, half: usize) -> (Fq4, Fq4) {
        let q = splat(QN);
        let np = splat(NPRIME as u64);
        let one = splat(super::MONT_ONE);
        let mut acc0 = [_mm256_setzero_si256(); 4];
        let mut acc2 = [_mm256_setzero_si256(); 4];

        let chunks = half / 4;
        for ci in 0..chunks {
            let i = ci * 4;
            let lo = [load4(&ext.c[0], i), load4(&ext.c[1], i), load4(&ext.c[2], i), load4(&ext.c[3], i)];
            let hi = [
                load4(&ext.c[0], i + half),
                load4(&ext.c[1], i + half),
                load4(&ext.c[2], i + half),
                load4(&ext.c[3], i + half),
            ];
            let eq = [load4(&eqsuf.c[0], i), load4(&eqsuf.c[1], i), load4(&eqsuf.c[2], i), load4(&eqsuf.c[3], i)];
            let mut z2 = [_mm256_setzero_si256(); 4];
            for k in 0..4 {
                z2[k] = madd(hi[k], msub(hi[k], lo[k], q), q);
            }
            let lo_m1 = [msub(lo[0], one, q), lo[1], lo[2], lo[3]];
            let z2_m1 = [msub(z2[0], one, q), z2[1], z2[2], z2[3]];
            let t0 = fq4_mul(&eq, &fq4_mul(&lo, &lo_m1, q, np), q, np);
            let t2 = fq4_mul(&eq, &fq4_mul(&z2, &z2_m1, q, np), q, np);
            for k in 0..4 {
                acc0[k] = _mm256_add_epi64(acc0[k], t0[k]);
                acc2[k] = _mm256_add_epi64(acc2[k], t2[k]);
            }
        }
        let mut h0 = horiz_reduce(&acc0);
        let mut h2 = horiz_reduce(&acc2);
        for i in (chunks * 4)..half {
            let lo = [ext.c[0][i], ext.c[1][i], ext.c[2][i], ext.c[3][i]];
            let hi = [ext.c[0][i + half], ext.c[1][i + half], ext.c[2][i + half], ext.c[3][i + half]];
            let (t0, t2) = super::tail_bit_terms(lo, hi, [eqsuf.c[0][i], eqsuf.c[1][i], eqsuf.c[2][i], eqsuf.c[3][i]]);
            for k in 0..4 {
                h0[k] = addq(h0[k], t0[k]);
                h2[k] = addq(h2[k], t2[k]);
            }
        }
        (
            Fq4([from_mont(h0[0]), from_mont(h0[1]), from_mont(h0[2]), from_mont(h0[3])]),
            Fq4([from_mont(h2[0]), from_mont(h2[1]), from_mont(h2[2]), from_mont(h2[3])]),
        )
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn bitcheck_evals_split(
        ext: &SoaFq4,
        ea: &SoaFq4,
        eb: &SoaFq4,
        half: usize,
    ) -> (Fq4, Fq4) {
        let q = splat(QN);
        let np = splat(NPRIME as u64);
        let one = splat(super::MONT_ONE);
        let eb_n = eb.len();
        let ea_n = ea.len();
        let mut acc0 = [_mm256_setzero_si256(); 4];
        let mut acc2 = [_mm256_setzero_si256(); 4];

        for hi in 0..ea_n {
            let eav = [splat(ea.c[0][hi]), splat(ea.c[1][hi]), splat(ea.c[2][hi]), splat(ea.c[3][hi])];
            let base = hi * eb_n;
            let chunks = eb_n / 4;
            for ci in 0..chunks {
                let lo = ci * 4;
                let cell = base + lo;
                let ebv = [load4(&eb.c[0], lo), load4(&eb.c[1], lo), load4(&eb.c[2], lo), load4(&eb.c[3], lo)];
                let eq = fq4_mul(&eav, &ebv, q, np);
                let el = [load4(&ext.c[0], cell), load4(&ext.c[1], cell), load4(&ext.c[2], cell), load4(&ext.c[3], cell)];
                let eh = [
                    load4(&ext.c[0], cell + half),
                    load4(&ext.c[1], cell + half),
                    load4(&ext.c[2], cell + half),
                    load4(&ext.c[3], cell + half),
                ];
                let mut z2 = [_mm256_setzero_si256(); 4];
                for k in 0..4 {
                    z2[k] = madd(eh[k], msub(eh[k], el[k], q), q);
                }
                let el_m1 = [msub(el[0], one, q), el[1], el[2], el[3]];
                let z2_m1 = [msub(z2[0], one, q), z2[1], z2[2], z2[3]];
                let t0 = fq4_mul(&eq, &fq4_mul(&el, &el_m1, q, np), q, np);
                let t2 = fq4_mul(&eq, &fq4_mul(&z2, &z2_m1, q, np), q, np);
                for k in 0..4 {
                    acc0[k] = _mm256_add_epi64(acc0[k], t0[k]);
                    acc2[k] = _mm256_add_epi64(acc2[k], t2[k]);
                }
            }
            for lo in (chunks * 4)..eb_n {
                let cell = base + lo;
                let eqv = fq4_mul_scalar(
                    [ea.c[0][hi], ea.c[1][hi], ea.c[2][hi], ea.c[3][hi]],
                    [eb.c[0][lo], eb.c[1][lo], eb.c[2][lo], eb.c[3][lo]],
                );
                let el = [ext.c[0][cell], ext.c[1][cell], ext.c[2][cell], ext.c[3][cell]];
                let eh = [ext.c[0][cell + half], ext.c[1][cell + half], ext.c[2][cell + half], ext.c[3][cell + half]];
                let (t0, t2) = super::tail_bit_terms(el, eh, eqv);
                let mut tmp0 = [0u64; 4];
                let mut tmp2 = [0u64; 4];
                for k in 0..4 {
                    tmp0[k] = t0[k];
                    tmp2[k] = t2[k];
                }
                acc0 = add_scalar_into(acc0, tmp0);
                acc2 = add_scalar_into(acc2, tmp2);
            }
        }
        let h0 = horiz_reduce(&acc0);
        let h2 = horiz_reduce(&acc2);
        (
            Fq4([from_mont(h0[0]), from_mont(h0[1]), from_mont(h0[2]), from_mont(h0[3])]),
            Fq4([from_mont(h2[0]), from_mont(h2[1]), from_mont(h2[2]), from_mont(h2[3])]),
        )
    }

    #[target_feature(enable = "avx2")]
    unsafe fn add_scalar_into(mut acc: [__m256i; 4], v: [u64; 4]) -> [__m256i; 4] {
        let add = [
            _mm256_set_epi64x(0, 0, 0, v[0] as i64),
            _mm256_set_epi64x(0, 0, 0, v[1] as i64),
            _mm256_set_epi64x(0, 0, 0, v[2] as i64),
            _mm256_set_epi64x(0, 0, 0, v[3] as i64),
        ];
        for k in 0..4 {
            acc[k] = _mm256_add_epi64(acc[k], add[k]);
        }
        acc
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn fold(ext: &mut SoaFq4, half: usize, rm: [u64; 4]) {
        let q = splat(QN);
        let np = splat(NPRIME as u64);
        let rv = [splat(rm[0]), splat(rm[1]), splat(rm[2]), splat(rm[3])];
        let chunks = half / 4;
        for ci in 0..chunks {
            let i = ci * 4;
            let lo = [load4(&ext.c[0], i), load4(&ext.c[1], i), load4(&ext.c[2], i), load4(&ext.c[3], i)];
            let hi = [
                load4(&ext.c[0], i + half),
                load4(&ext.c[1], i + half),
                load4(&ext.c[2], i + half),
                load4(&ext.c[3], i + half),
            ];
            let mut d = [_mm256_setzero_si256(); 4];
            for k in 0..4 {
                d[k] = msub(hi[k], lo[k], q);
            }
            let rd = fq4_mul(&rv, &d, q, np);
            for k in 0..4 {
                store4(&mut ext.c[k], i, madd(lo[k], rd[k], q));
            }
        }
        for i in (chunks * 4)..half {
            let lo = [ext.c[0][i], ext.c[1][i], ext.c[2][i], ext.c[3][i]];
            let hi = [ext.c[0][i + half], ext.c[1][i + half], ext.c[2][i + half], ext.c[3][i + half]];
            let mut d = [0u64; 4];
            for k in 0..4 {
                d[k] = subq(hi[k], lo[k]);
            }
            let rd = fq4_mul_scalar(rm, d);
            for k in 0..4 {
                ext.c[k][i] = addq(lo[k], rd[k]);
            }
        }
        ext.truncate(half);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn batched_phase_a_round(
        w: &SoaFq4,
        apow: &SoaFq4,
        ea: &SoaFq4,
        eb: &SoaFq4,
        lg: &[Fq4],
        half: usize,
        s_len: usize,
        k_lo: usize,
        k_hi: usize,
    ) -> ([Fq4; 3], [Fq4; 3]) {
        let q = splat(QN);
        let np = splat(NPRIME as u64);
        let one = splat(super::MONT_ONE);
        let half_k = lg.len() / 2;
        let chunks = s_len / 4;
        let mut gf2 = [Fq4::ZERO; 3];
        let mut q3 = [Fq4::ZERO; 3];

        for k in k_lo..k_hi {
            let mut rd = [[_mm256_setzero_si256(); 4]; 3];
            let mut qk = [[_mm256_setzero_si256(); 4]; 3];
            let base = k * s_len;
            for c in 0..chunks {
                let l = c * 4;
                let cell = base + l;
                let lo = [load4(&w.c[0], cell), load4(&w.c[1], cell), load4(&w.c[2], cell), load4(&w.c[3], cell)];
                let hi = [
                    load4(&w.c[0], cell + half),
                    load4(&w.c[1], cell + half),
                    load4(&w.c[2], cell + half),
                    load4(&w.c[3], cell + half),
                ];
                let ap = [load4(&apow.c[0], l), load4(&apow.c[1], l), load4(&apow.c[2], l), load4(&apow.c[3], l)];
                let ebl = [load4(&eb.c[0], l), load4(&eb.c[1], l), load4(&eb.c[2], l), load4(&eb.c[3], l)];
                let mut d = [_mm256_setzero_si256(); 4];
                for j in 0..4 {
                    d[j] = msub(hi[j], lo[j], q);
                }
                let mut wx = [lo, [_mm256_setzero_si256(); 4], [_mm256_setzero_si256(); 4]];
                for j in 0..4 {
                    wx[1][j] = madd(hi[j], d[j], q);
                    wx[2][j] = madd(wx[1][j], d[j], q);
                }
                let f2_0 = fq4_mul(&ap, &lo, q, np);
                let f2_i = fq4_mul(&ap, &d, q, np);
                for j in 0..4 {
                    let two_i = madd(f2_i[j], f2_i[j], q);
                    let f2_2 = madd(f2_0[j], two_i, q);
                    let f2_3 = madd(f2_2, f2_i[j], q);
                    rd[0][j] = _mm256_add_epi64(rd[0][j], f2_0[j]);
                    rd[1][j] = _mm256_add_epi64(rd[1][j], f2_2);
                    rd[2][j] = _mm256_add_epi64(rd[2][j], f2_3);
                }
                for xi in 0..3 {
                    let wv = wx[xi];
                    let wm1 = [msub(wv[0], one, q), wv[1], wv[2], wv[3]];
                    let p = fq4_mul(&wv, &wm1, q, np);
                    let f3 = fq4_mul(&ebl, &p, q, np);
                    for j in 0..4 {
                        qk[xi][j] = _mm256_add_epi64(qk[xi][j], f3[j]);
                    }
                }
            }
            let ea_k = [ea.c[0][k], ea.c[1][k], ea.c[2][k], ea.c[3][k]];
            let lg_lo = lg[k];
            let lg_hi = lg[k + half_k];
            let dl = lg_hi - lg_lo;
            let lg_at = [lg_lo, lg_hi + dl, lg_hi + dl + dl];
            for xi in 0..3 {
                let rdm = horiz_reduce(&rd[xi]);
                let rd_fq = mont_to_fq4(rdm);
                gf2[xi] = gf2[xi] + lg_at[xi] * rd_fq;
                let qkm = horiz_reduce(&qk[xi]);
                let qcontrib = fq4_mul_scalar(ea_k, qkm);
                q3[xi] = q3[xi]
                    + Fq4([
                        from_mont(qcontrib[0]),
                        from_mont(qcontrib[1]),
                        from_mont(qcontrib[2]),
                        from_mont(qcontrib[3]),
                    ]);
            }
        }
        (gf2, q3)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn batched_phase_a_round0(
        zw: &PackedBits,
        apow: &SoaFq4,
        ea: &SoaFq4,
        eb: &SoaFq4,
        lg: &[Fq4],
        half: usize,
        s_len: usize,
        k_lo: usize,
        k_hi: usize,
    ) -> ([Fq4; 3], [Fq4; 3]) {
        let q = splat(QN);
        let np = splat(NPRIME as u64);
        let one = splat(super::MONT_ONE);
        let mo = super::MONT_ONE as i64;
        let half_k = lg.len() / 2;
        let chunks = s_len / 4;
        let mut gf2 = [Fq4::ZERO; 3];
        let mut q3 = [Fq4::ZERO; 3];

        for k in k_lo..k_hi {
            let mut rd = [[_mm256_setzero_si256(); 4]; 3];
            let mut qk = [[_mm256_setzero_si256(); 4]; 3];
            let base = k * s_len;
            for c in 0..chunks {
                let l = c * 4;
                let cell = base + l;
                let bit = |i: usize| if zw.get(i) { mo } else { 0 };
                let lo = _mm256_set_epi64x(bit(cell + 3), bit(cell + 2), bit(cell + 1), bit(cell));
                let hi = _mm256_set_epi64x(
                    bit(cell + half + 3),
                    bit(cell + half + 2),
                    bit(cell + half + 1),
                    bit(cell + half),
                );
                let d = msub(hi, lo, q);
                let w2 = madd(hi, d, q);
                let w3 = madd(w2, d, q);
                let ap = [load4(&apow.c[0], l), load4(&apow.c[1], l), load4(&apow.c[2], l), load4(&apow.c[3], l)];
                let ebl = [load4(&eb.c[0], l), load4(&eb.c[1], l), load4(&eb.c[2], l), load4(&eb.c[3], l)];
                for j in 0..4 {
                    let f2_0 = mmul(ap[j], lo, q, np);
                    let f2_i = mmul(ap[j], d, q, np);
                    let f2_2 = madd(f2_0, madd(f2_i, f2_i, q), q);
                    let f2_3 = madd(f2_2, f2_i, q);
                    rd[0][j] = _mm256_add_epi64(rd[0][j], f2_0);
                    rd[1][j] = _mm256_add_epi64(rd[1][j], f2_2);
                    rd[2][j] = _mm256_add_epi64(rd[2][j], f2_3);
                }
                for (xi, wx) in [lo, w2, w3].into_iter().enumerate() {
                    let p = mmul(wx, msub(wx, one, q), q, np);
                    for j in 0..4 {
                        qk[xi][j] = _mm256_add_epi64(qk[xi][j], mmul(ebl[j], p, q, np));
                    }
                }
            }
            let ea_k = [ea.c[0][k], ea.c[1][k], ea.c[2][k], ea.c[3][k]];
            let lg_lo = lg[k];
            let lg_hi = lg[k + half_k];
            let dl = lg_hi - lg_lo;
            let lg_at = [lg_lo, lg_hi + dl, lg_hi + dl + dl];
            for xi in 0..3 {
                let rd_fq = mont_to_fq4(horiz_reduce(&rd[xi]));
                gf2[xi] = gf2[xi] + lg_at[xi] * rd_fq;
                let qcontrib = fq4_mul_scalar(ea_k, horiz_reduce(&qk[xi]));
                q3[xi] = q3[xi] + mont_to_fq4(qcontrib);
            }
        }
        (gf2, q3)
    }

    #[inline(always)]
    fn mont_to_fq4(a: [u64; 4]) -> Fq4 {
        Fq4([from_mont(a[0]), from_mont(a[1]), from_mont(a[2]), from_mont(a[3])])
    }

    #[target_feature(enable = "avx2")]
    unsafe fn horiz_reduce(acc: &[__m256i; 4]) -> [u64; 4] {
        let mut out = [0u64; 4];
        let mut tmp = [0u64; 4];
        for k in 0..4 {
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc[k]);
            out[k] = (tmp[0] + tmp[1] + tmp[2] + tmp[3]) % QN;
        }
        out
    }
}

pub fn eq_table_mont(tau: &[Fq4]) -> SoaFq4 {
    let mut aos: Vec<[u64; 4]> = vec![[MONT_ONE, 0, 0, 0]];
    for t in tau.iter().rev() {
        let tm = [to_mont(t.0[0]), to_mont(t.0[1]), to_mont(t.0[2]), to_mont(t.0[3])];
        let mut one_minus = [subq(MONT_ONE, tm[0]), subq(0, tm[1]), subq(0, tm[2]), subq(0, tm[3])];
        for k in 1..4 {
            one_minus[k] = if tm[k] == 0 { 0 } else { QN - tm[k] };
        }
        let mut next = Vec::with_capacity(aos.len() * 2);
        for &v in &aos {
            next.push(fq4_mul_scalar(v, one_minus));
        }
        for &v in &aos {
            next.push(fq4_mul_scalar(v, tm));
        }
        aos = next;
    }
    let mut c: [Vec<u64>; 4] = Default::default();
    for k in 0..4 {
        c[k] = aos.iter().map(|v| v[k]).collect();
    }
    SoaFq4 { c }
}

#[inline(always)]
fn tail_bit_terms(lo: [u64; 4], hi: [u64; 4], eq: [u64; 4]) -> ([u64; 4], [u64; 4]) {
    let mut z2 = [0u64; 4];
    for k in 0..4 {
        z2[k] = addq(hi[k], subq(hi[k], lo[k]));
    }
    let mut lo_m1 = lo;
    lo_m1[0] = subq(lo[0], MONT_ONE);
    let mut z2_m1 = z2;
    z2_m1[0] = subq(z2[0], MONT_ONE);
    (
        fq4_mul_scalar(eq, fq4_mul_scalar(lo, lo_m1)),
        fq4_mul_scalar(eq, fq4_mul_scalar(z2, z2_m1)),
    )
}

const MONT_ONE: u64 = 1;

pub fn bitcheck_evals(ext: &SoaFq4, eqsuf: &SoaFq4, half: usize) -> (Fq4, Fq4) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::bitcheck_evals(ext, eqsuf, half) };
        }
    }
    bitcheck_evals_scalar(ext, eqsuf, half)
}

pub fn bitcheck_evals_split(ext: &SoaFq4, ea: &SoaFq4, eb: &SoaFq4, half: usize) -> (Fq4, Fq4) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::bitcheck_evals_split(ext, ea, eb, half) };
        }
    }
    bitcheck_evals_split_scalar(ext, ea, eb, half)
}

pub fn bitcheck_evals_split_scalar(ext: &SoaFq4, ea: &SoaFq4, eb: &SoaFq4, half: usize) -> (Fq4, Fq4) {
    let eb_n = eb.len();
    let mut h0 = [0u64; 4];
    let mut h2 = [0u64; 4];
    for hi in 0..ea.len() {
        let ea_v = [ea.c[0][hi], ea.c[1][hi], ea.c[2][hi], ea.c[3][hi]];
        for lo in 0..eb_n {
            let cell = hi * eb_n + lo;
            let eq = fq4_mul_scalar(ea_v, [eb.c[0][lo], eb.c[1][lo], eb.c[2][lo], eb.c[3][lo]]);
            let el = [ext.c[0][cell], ext.c[1][cell], ext.c[2][cell], ext.c[3][cell]];
            let eh = [ext.c[0][cell + half], ext.c[1][cell + half], ext.c[2][cell + half], ext.c[3][cell + half]];
            let (t0, t2) = tail_bit_terms(el, eh, eq);
            for k in 0..4 {
                h0[k] = addq(h0[k], t0[k]);
                h2[k] = addq(h2[k], t2[k]);
            }
        }
    }
    (
        Fq4([from_mont(h0[0]), from_mont(h0[1]), from_mont(h0[2]), from_mont(h0[3])]),
        Fq4([from_mont(h2[0]), from_mont(h2[1]), from_mont(h2[2]), from_mont(h2[3])]),
    )
}

#[inline]
fn split_k<F>(half_k: usize, cells: usize, f: F) -> ([Fq4; 3], [Fq4; 3])
where
    F: Fn(usize, usize) -> ([Fq4; 3], [Fq4; 3]) + Sync + Send,
{
    const PAR_MIN_CELLS: usize = 1 << 14;
    if cells < PAR_MIN_CELLS || half_k < 2 {
        return f(0, half_k);
    }
    let chunk = (half_k / (4 * rayon::current_num_threads())).max(1);
    let nch = half_k.div_ceil(chunk);
    let parts: Vec<([Fq4; 3], [Fq4; 3])> = (0..nch)
        .into_par_iter()
        .map(|c| f(c * chunk, ((c + 1) * chunk).min(half_k)))
        .collect();
    let mut gf2 = [Fq4::ZERO; 3];
    let mut q3 = [Fq4::ZERO; 3];
    for (a, b) in parts {
        for x in 0..3 {
            gf2[x] = gf2[x] + a[x];
            q3[x] = q3[x] + b[x];
        }
    }
    (gf2, q3)
}

pub fn batched_phase_a_round(
    w: &SoaFq4,
    apow: &SoaFq4,
    ea: &SoaFq4,
    eb: &SoaFq4,
    lg: &[Fq4],
    half: usize,
    s_len: usize,
) -> ([Fq4; 3], [Fq4; 3]) {
    split_k(lg.len() / 2, half, |lo, hi| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe {
                    avx2::batched_phase_a_round(w, apow, ea, eb, lg, half, s_len, lo, hi)
                };
            }
        }
        batched_phase_a_round_scalar(w, apow, ea, eb, lg, half, s_len, lo, hi)
    })
}

pub fn batched_phase_a_round_scalar(
    w: &SoaFq4,
    apow: &SoaFq4,
    ea: &SoaFq4,
    eb: &SoaFq4,
    lg: &[Fq4],
    half: usize,
    s_len: usize,
    k_lo: usize,
    k_hi: usize,
) -> ([Fq4; 3], [Fq4; 3]) {
    let half_k = lg.len() / 2;
    let mut gf2 = [Fq4::ZERO; 3];
    let mut q3 = [Fq4::ZERO; 3];
    let getm = |t: &SoaFq4, i: usize| [t.c[0][i], t.c[1][i], t.c[2][i], t.c[3][i]];
    let tofq = |a: [u64; 4]| Fq4([from_mont(a[0]), from_mont(a[1]), from_mont(a[2]), from_mont(a[3])]);
    for k in k_lo..k_hi {
        let ea_k = getm(ea, k);
        let mut rd = [[0u64; 4]; 3];
        let mut qk = [[0u64; 4]; 3];
        for l in 0..s_len {
            let cell = k * s_len + l;
            let lo = getm(w, cell);
            let hi = getm(w, cell + half);
            let ap = getm(apow, l);
            let ebl = getm(eb, l);
            let mut d = [0u64; 4];
            for j in 0..4 {
                d[j] = subq(hi[j], lo[j]);
            }
            let w2: [u64; 4] = core::array::from_fn(|j| addq(hi[j], d[j]));
            let w3: [u64; 4] = core::array::from_fn(|j| addq(w2[j], d[j]));
            let f2_0 = fq4_mul_scalar(ap, lo);
            let f2_i = fq4_mul_scalar(ap, d);
            for j in 0..4 {
                let f2_2 = addq(f2_0[j], addq(f2_i[j], f2_i[j]));
                let f2_3 = addq(f2_2, f2_i[j]);
                rd[0][j] = addq(rd[0][j], f2_0[j]);
                rd[1][j] = addq(rd[1][j], f2_2);
                rd[2][j] = addq(rd[2][j], f2_3);
            }
            for (xi, wv) in [lo, w2, w3].into_iter().enumerate() {
                let mut wm1 = wv;
                wm1[0] = subq(wv[0], MONT_ONE);
                let p = fq4_mul_scalar(wv, wm1);
                let f3 = fq4_mul_scalar(ebl, p);
                for j in 0..4 {
                    qk[xi][j] = addq(qk[xi][j], f3[j]);
                }
            }
        }
        let lg_lo = lg[k];
        let lg_hi = lg[k + half_k];
        let dl = lg_hi - lg_lo;
        let lg_at = [lg_lo, lg_hi + dl, lg_hi + dl + dl];
        for xi in 0..3 {
            gf2[xi] = gf2[xi] + lg_at[xi] * tofq(rd[xi]);
            q3[xi] = q3[xi] + tofq(fq4_mul_scalar(ea_k, qk[xi]));
        }
    }
    (gf2, q3)
}

pub fn batched_phase_a_round0(
    zw: &PackedBits,
    apow: &SoaFq4,
    ea: &SoaFq4,
    eb: &SoaFq4,
    lg: &[Fq4],
    half: usize,
    s_len: usize,
) -> ([Fq4; 3], [Fq4; 3]) {
    split_k(lg.len() / 2, half, |lo, hi| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe {
                    avx2::batched_phase_a_round0(zw, apow, ea, eb, lg, half, s_len, lo, hi)
                };
            }
        }
        batched_phase_a_round0_scalar(zw, apow, ea, eb, lg, half, s_len, lo, hi)
    })
}

#[inline]
pub fn fold_bits_lut(r: Fq4) -> [Fq4; 4] {
    [Fq4::ZERO, r, Fq4::ONE - r, Fq4::ONE]
}

pub fn fold_bits_to_soa(zw: &PackedBits, half: usize, r: Fq4) -> SoaFq4 {
    let lut = fold_bits_lut(r);
    let mut c: [Vec<u64>; 4] = Default::default();
    c.par_iter_mut().enumerate().for_each(|(k, arr)| {
        *arr = (0..half)
            .into_par_iter()
            .map(|cell| {
                let idx = ((zw.get(cell) as usize) << 1) | zw.get(cell + half) as usize;
                to_mont(lut[idx].0[k])
            })
            .collect();
    });
    SoaFq4 { c }
}

pub fn batched_phase_a_round0_scalar(
    zw: &PackedBits,
    apow: &SoaFq4,
    ea: &SoaFq4,
    eb: &SoaFq4,
    lg: &[Fq4],
    half: usize,
    s_len: usize,
    k_lo: usize,
    k_hi: usize,
) -> ([Fq4; 3], [Fq4; 3]) {
    let half_k = lg.len() / 2;
    let mut gf2 = [Fq4::ZERO; 3];
    let mut q3 = [Fq4::ZERO; 3];
    let getm = |t: &SoaFq4, i: usize| [t.c[0][i], t.c[1][i], t.c[2][i], t.c[3][i]];
    let tofq = |a: [u64; 4]| Fq4([from_mont(a[0]), from_mont(a[1]), from_mont(a[2]), from_mont(a[3])]);
    for k in k_lo..k_hi {
        let ea_k = getm(ea, k);
        let mut rd = [[0u64; 4]; 3];
        let mut qk = [[0u64; 4]; 3];
        for l in 0..s_len {
            let cell = k * s_len + l;
            let lo = if zw.get(cell) { MONT_ONE } else { 0 };
            let hi = if zw.get(cell + half) { MONT_ONE } else { 0 };
            let d = subq(hi, lo);
            let w2 = addq(hi, d);
            let w3 = addq(w2, d);
            let ap = getm(apow, l);
            let ebl = getm(eb, l);
            for j in 0..4 {
                let f2_0 = mont_mul(ap[j], lo);
                let f2_i = mont_mul(ap[j], d);
                let f2_2 = addq(f2_0, addq(f2_i, f2_i));
                let f2_3 = addq(f2_2, f2_i);
                rd[0][j] = addq(rd[0][j], f2_0);
                rd[1][j] = addq(rd[1][j], f2_2);
                rd[2][j] = addq(rd[2][j], f2_3);
            }
            for (xi, wx) in [lo, w2, w3].into_iter().enumerate() {
                let p = mont_mul(wx, subq(wx, MONT_ONE));
                for j in 0..4 {
                    qk[xi][j] = addq(qk[xi][j], mont_mul(ebl[j], p));
                }
            }
        }
        let lg_lo = lg[k];
        let lg_hi = lg[k + half_k];
        let dl = lg_hi - lg_lo;
        let lg_at = [lg_lo, lg_hi + dl, lg_hi + dl + dl];
        for xi in 0..3 {
            gf2[xi] = gf2[xi] + lg_at[xi] * tofq(rd[xi]);
            q3[xi] = q3[xi] + tofq(fq4_mul_scalar(ea_k, qk[xi]));
        }
    }
    (gf2, q3)
}

pub fn fold(ext: &mut SoaFq4, half: usize, r: Fq4) {
    let rm = [to_mont(r.0[0]), to_mont(r.0[1]), to_mont(r.0[2]), to_mont(r.0[3])];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::fold(ext, half, rm) };
            return;
        }
    }
    fold_scalar(ext, half, rm);
}

pub fn bitcheck_evals_scalar(ext: &SoaFq4, eqsuf: &SoaFq4, half: usize) -> (Fq4, Fq4) {
    let mut h0 = [0u64; 4];
    let mut h2 = [0u64; 4];
    for i in 0..half {
        let lo = [ext.c[0][i], ext.c[1][i], ext.c[2][i], ext.c[3][i]];
        let hi = [ext.c[0][i + half], ext.c[1][i + half], ext.c[2][i + half], ext.c[3][i + half]];
        let eq = [eqsuf.c[0][i], eqsuf.c[1][i], eqsuf.c[2][i], eqsuf.c[3][i]];
        let (t0, t2) = tail_bit_terms(lo, hi, eq);
        for k in 0..4 {
            h0[k] = addq(h0[k], t0[k]);
            h2[k] = addq(h2[k], t2[k]);
        }
    }
    (
        Fq4([from_mont(h0[0]), from_mont(h0[1]), from_mont(h0[2]), from_mont(h0[3])]),
        Fq4([from_mont(h2[0]), from_mont(h2[1]), from_mont(h2[2]), from_mont(h2[3])]),
    )
}

pub fn fold_scalar(ext: &mut SoaFq4, half: usize, rm: [u64; 4]) {
    for i in 0..half {
        let lo = [ext.c[0][i], ext.c[1][i], ext.c[2][i], ext.c[3][i]];
        let hi = [ext.c[0][i + half], ext.c[1][i + half], ext.c[2][i + half], ext.c[3][i + half]];
        let mut d = [0u64; 4];
        for k in 0..4 {
            d[k] = subq(hi[k], lo[k]);
        }
        let rd = fq4_mul_scalar(rm, d);
        for k in 0..4 {
            ext.c[k][i] = addq(lo[k], rd[k]);
        }
    }
    ext.truncate(half);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::SimpleRng;

    #[test]
    fn mont_roundtrip_and_mul() {
        assert_eq!(MONT_ONE, to_mont(Fq::ONE));
        let mut rng = SimpleRng::new(1);
        for _ in 0..5000 {
            let a = rng.next_fq();
            let b = rng.next_fq();
            assert_eq!(from_mont(to_mont(a)), a);
            assert_eq!(from_mont(mont_mul(to_mont(a), to_mont(b))), a * b);
        }
    }

    #[test]
    fn fq4_mul_scalar_matches() {
        let mut rng = SimpleRng::new(2);
        for _ in 0..3000 {
            let a = rng.next_fq4();
            let b = rng.next_fq4();
            let am = [to_mont(a.0[0]), to_mont(a.0[1]), to_mont(a.0[2]), to_mont(a.0[3])];
            let bm = [to_mont(b.0[0]), to_mont(b.0[1]), to_mont(b.0[2]), to_mont(b.0[3])];
            let cm = fq4_mul_scalar(am, bm);
            let c = Fq4([from_mont(cm[0]), from_mont(cm[1]), from_mont(cm[2]), from_mont(cm[3])]);
            assert_eq!(c, a * b);
        }
    }

    #[test]
    fn eq_table_mont_matches() {
        use crate::mle::eq_table;
        let mut rng = SimpleRng::new(5);
        for nv in [1usize, 3, 5] {
            let tau: Vec<Fq4> = (0..nv).map(|_| rng.next_fq4()).collect();
            let want = eq_table(&tau);
            let got = eq_table_mont(&tau);
            assert_eq!(got.len(), want.len());
            for i in 0..want.len() {
                assert_eq!(got.get(i), want[i], "nv={nv} i={i}");
            }
        }
    }

    #[test]
    fn evals_split_matches_materialized() {
        let mut rng = SimpleRng::new(9);
        for (ea_n, eb_n) in [(1usize, 4usize), (4, 8), (8, 8), (2, 16)] {
            let half = ea_n * eb_n;
            let n = 2 * half;
            let ext_aos: Vec<Fq4> = (0..n).map(|_| rng.next_fq4()).collect();
            let ea_aos: Vec<Fq4> = (0..ea_n).map(|_| rng.next_fq4()).collect();
            let eb_aos: Vec<Fq4> = (0..eb_n).map(|_| rng.next_fq4()).collect();
            let eqsuf: Vec<Fq4> =
                (0..half).map(|c| ea_aos[c / eb_n] * eb_aos[c % eb_n]).collect();

            let ext = SoaFq4::from_aos(&ext_aos);
            let ea = SoaFq4::from_aos(&ea_aos);
            let eb = SoaFq4::from_aos(&eb_aos);
            let eqm = SoaFq4::from_aos(&eqsuf);

            let want = bitcheck_evals(&ext, &eqm, half);
            let got_s = bitcheck_evals_split_scalar(&ext, &ea, &eb, half);
            let got_d = bitcheck_evals_split(&ext, &ea, &eb, half);
            assert_eq!(want, got_s, "scalar split at ({ea_n},{eb_n})");
            assert_eq!(want, got_d, "simd split at ({ea_n},{eb_n})");
        }
    }

    #[test]
    fn simd_equals_scalar() {
        let mut rng = SimpleRng::new(3);
        for &half in &[1usize, 3, 4, 7, 16, 33, 64] {
            let n = 2 * half;
            let ext_aos: Vec<Fq4> = (0..n).map(|_| rng.next_fq4()).collect();
            let eq_aos: Vec<Fq4> = (0..half).map(|_| rng.next_fq4()).collect();
            let r = rng.next_fq4();
            let rm = [to_mont(r.0[0]), to_mont(r.0[1]), to_mont(r.0[2]), to_mont(r.0[3])];
            let eqsuf = SoaFq4::from_aos(&eq_aos);

            let e = SoaFq4::from_aos(&ext_aos);
            let hs = bitcheck_evals_scalar(&e, &eqsuf, half);
            let hd = bitcheck_evals(&e, &eqsuf, half);
            assert_eq!(hs, hd, "evals mismatch at half={half}");
            let mut ref0 = Fq4::ZERO;
            let mut ref2 = Fq4::ZERO;
            for i in 0..half {
                let lo = e.get(i);
                let hi = e.get(i + half);
                let z2 = hi + (hi - lo);
                ref0 = ref0 + eq_aos[i] * (lo * (lo - Fq4::ONE));
                ref2 = ref2 + eq_aos[i] * (z2 * (z2 - Fq4::ONE));
            }
            assert_eq!(hd, (ref0, ref2), "evals vs definition at half={half}");

            let mut e1 = SoaFq4::from_aos(&ext_aos);
            fold_scalar(&mut e1, half, rm);
            let mut e2 = SoaFq4::from_aos(&ext_aos);
            fold(&mut e2, half, r);
            for i in 0..half {
                let lo = ext_aos[i];
                let hi = ext_aos[i + half];
                let want = lo + r * (hi - lo);
                assert_eq!(e1.get(i), want);
                assert_eq!(e2.get(i), want);
            }
        }
    }
}
