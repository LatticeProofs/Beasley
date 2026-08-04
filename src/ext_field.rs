use crate::field::{reduce64, Fq, C, Q};
use std::ops::{Add, Mul, Neg, Sub};

pub const W: u64 = 2;

pub const EXT_DEG: usize = 4;

pub type FqExt = Fq4;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Fq4(pub [Fq; 4]);

impl Fq4 {
    pub const ZERO: Fq4 = Fq4([Fq(0); 4]);
    pub const ONE: Fq4 = Fq4([Fq(1), Fq(0), Fq(0), Fq(0)]);

    #[inline(always)]
    pub const fn from_coeffs(c: [Fq; EXT_DEG]) -> Self {
        Fq4(c)
    }

    #[inline(always)]
    pub fn from_fn(f: impl FnMut(usize) -> Fq) -> Self {
        Fq4(core::array::from_fn(f))
    }

    #[inline(always)]
    pub fn coeffs(&self) -> &[Fq; EXT_DEG] {
        &self.0
    }

    pub fn from_fq(x: Fq) -> Self {
        Fq4([x, Fq::ZERO, Fq::ZERO, Fq::ZERO])
    }

    pub fn from_u64(v: u64) -> Self {
        Fq4::from_fq(Fq::new(v))
    }

    pub fn pow(self, mut e: u128) -> Self {
        let mut base = self;
        let mut acc = Fq4::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            e >>= 1;
        }
        acc
    }

    pub fn inv(self) -> Self {
        let q = Q as u128;
        self.pow(q * q * q * q - 2)
    }
}

impl Add for Fq4 {
    type Output = Fq4;
    #[inline(always)]
    fn add(self, r: Fq4) -> Fq4 {
        let mut c = [Fq::ZERO; 4];
        for i in 0..4 {
            c[i] = self.0[i] + r.0[i];
        }
        Fq4(c)
    }
}

impl Sub for Fq4 {
    type Output = Fq4;
    #[inline(always)]
    fn sub(self, r: Fq4) -> Fq4 {
        let mut c = [Fq::ZERO; 4];
        for i in 0..4 {
            c[i] = self.0[i] - r.0[i];
        }
        Fq4(c)
    }
}

impl Neg for Fq4 {
    type Output = Fq4;
    fn neg(self) -> Fq4 {
        Fq4::ZERO - self
    }
}

impl Mul for Fq4 {
    type Output = Fq4;
    #[inline(always)]
    fn mul(self, r: Fq4) -> Fq4 {
        let a = [
            self.0[0].0 as u64,
            self.0[1].0 as u64,
            self.0[2].0 as u64,
            self.0[3].0 as u64,
        ];
        let b = [r.0[0].0 as u64, r.0[1].0 as u64, r.0[2].0 as u64, r.0[3].0 as u64];
        let pr = |p: u64| (p & 0xFFFF_FFFF) + C * (p >> 32);
        let m = |i: usize, j: usize| pr(a[i] * b[j]);
        let c0 = reduce64(m(0, 0) + 2 * (m(1, 3) + m(2, 2) + m(3, 1)));
        let c1 = reduce64(m(0, 1) + m(1, 0) + 2 * (m(2, 3) + m(3, 2)));
        let c2 = reduce64(m(0, 2) + m(1, 1) + m(2, 0) + 2 * m(3, 3));
        let c3 = reduce64(m(0, 3) + m(1, 2) + m(2, 1) + m(3, 0));
        Fq4([Fq(c0), Fq(c1), Fq(c2), Fq(c3)])
    }
}

pub fn poly_eval_pows(coeffs: &[Fq], alpha_pows: &[Fq4]) -> Fq4 {
    let mut acc = [0u64; EXT_DEG];
    for (&c, ap) in coeffs.iter().zip(alpha_pows) {
        let cv = c.0 as u64;
        for k in 0..EXT_DEG {
            let p = cv * ap.0[k].0 as u64;
            acc[k] += (p & 0xFFFF_FFFF) + C * (p >> 32);
        }
    }
    Fq4(core::array::from_fn(|k| Fq(reduce64(acc[k]))))
}

#[derive(Clone, Copy, Debug)]
pub struct LazyExtSum([u64; EXT_DEG]);

impl Default for LazyExtSum {
    fn default() -> Self {
        LazyExtSum([0; EXT_DEG])
    }
}

impl LazyExtSum {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn add(&mut self, x: &Fq4) {
        for k in 0..EXT_DEG {
            self.0[k] += x.0[k].0 as u64;
        }
    }
    #[inline(always)]
    pub fn finish(self) -> Fq4 {
        Fq4(core::array::from_fn(|k| Fq(reduce64(self.0[k]))))
    }
}

impl Mul<Fq4> for Fq {
    type Output = Fq4;
    #[inline(always)]
    fn mul(self, r: Fq4) -> Fq4 {
        let mut c = [Fq::ZERO; 4];
        for i in 0..4 {
            c[i] = self * r.0[i];
        }
        Fq4(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_is_nonsquare_so_x4_minus_2_is_irreducible() {
        assert_eq!(Fq::new(W).pow((Q - 1) / 2), Fq::new(Q - 1));
    }

    #[test]
    fn mul_and_inv() {
        let a = Fq4([Fq::new(3), Fq::new(1), Fq::new(4), Fq::new(1)]);
        let b = Fq4([Fq::new(5), Fq::new(9), Fq::new(2), Fq::new(6)]);
        assert_eq!(a * b, b * a);
        assert_eq!(a * a.inv(), Fq4::ONE);
        assert_eq!((a * b) * b.inv(), a);
    }

    #[test]
    fn x_times_x3_wraps_to_w() {
        let x = Fq4([Fq::ZERO, Fq::ONE, Fq::ZERO, Fq::ZERO]);
        let x3 = Fq4([Fq::ZERO, Fq::ZERO, Fq::ZERO, Fq::ONE]);
        assert_eq!(x * x3, Fq4::from_u64(W));
    }

    #[test]
    fn mul_matches_naive_u128() {
        let mut s = 777u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(97);
            Fq::new(s % Q)
        };
        let naive = |a: Fq4, b: Fq4| {
            let av: [u128; 4] = core::array::from_fn(|i| a.0[i].0 as u128);
            let bv: [u128; 4] = core::array::from_fn(|i| b.0[i].0 as u128);
            let mut r = [0u128; 7];
            for i in 0..4 {
                for j in 0..4 {
                    r[i + j] += av[i] * bv[j];
                }
            }
            for k in (4..7).rev() {
                r[k - 4] += (W as u128) * r[k];
            }
            Fq4(core::array::from_fn(|i| Fq((r[i] % Q as u128) as u32)))
        };
        for _ in 0..3000 {
            let a = Fq4([next(), next(), next(), next()]);
            let b = Fq4([next(), next(), next(), next()]);
            assert_eq!(a * b, naive(a, b));
        }
        let e = Fq4([Fq::new(Q - 1); 4]);
        assert_eq!(e * e, naive(e, e));
    }

    #[test]
    fn subfield_embedding_is_homomorphic() {
        let a = Fq::new(123456);
        let b = Fq::new(987654);
        assert_eq!(Fq4::from_fq(a) * Fq4::from_fq(b), Fq4::from_fq(a * b));
    }
}
