use crate::field::{reduce128, Fq, C, Q};
use std::ops::{Add, Mul, Neg, Sub};

pub const EXT_DEG: usize = 2;

pub type FqExt = Fq2;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Fq2(pub [Fq; EXT_DEG]);

impl Fq2 {
    pub const ZERO: Fq2 = Fq2([Fq(0); EXT_DEG]);
    pub const ONE: Fq2 = Fq2([Fq(1), Fq(0)]);

    #[inline(always)]
    pub fn from_fn(f: impl FnMut(usize) -> Fq) -> Self {
        Fq2(core::array::from_fn(f))
    }

    #[inline(always)]
    pub fn coeffs(&self) -> &[Fq; EXT_DEG] {
        &self.0
    }

    pub fn from_fq(x: Fq) -> Self {
        Fq2([x, Fq::ZERO])
    }

    pub fn from_u64(v: u64) -> Self {
        Fq2::from_fq(Fq::new(v))
    }

    pub fn pow(self, mut e: u128) -> Self {
        let mut base = self;
        let mut acc = Fq2::ONE;
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
        self.pow(q * q - 2)
    }
}

impl Add for Fq2 {
    type Output = Fq2;
    #[inline(always)]
    fn add(self, r: Fq2) -> Fq2 {
        Fq2([self.0[0] + r.0[0], self.0[1] + r.0[1]])
    }
}

impl Sub for Fq2 {
    type Output = Fq2;
    #[inline(always)]
    fn sub(self, r: Fq2) -> Fq2 {
        Fq2([self.0[0] - r.0[0], self.0[1] - r.0[1]])
    }
}

impl Neg for Fq2 {
    type Output = Fq2;
    #[inline(always)]
    fn neg(self) -> Fq2 {
        Fq2::ZERO - self
    }
}

impl Mul for Fq2 {
    type Output = Fq2;
    #[inline(always)]
    fn mul(self, r: Fq2) -> Fq2 {
        let (a0, a1) = (self.0[0].0 as u128, self.0[1].0 as u128);
        let (b0, b1) = (r.0[0].0 as u128, r.0[1].0 as u128);
        let p00 = Fq(reduce128(a0 * b0));
        let p11 = Fq(reduce128(a1 * b1));
        let p01 = Fq(reduce128(a0 * b1));
        let p10 = Fq(reduce128(a1 * b0));
        Fq2([p00 + (p11 + p11), p01 + p10])
    }
}

pub fn poly_eval_pows(coeffs: &[Fq], alpha_pows: &[Fq2]) -> Fq2 {
    let mut acc = [0u128; EXT_DEG];
    for (&c, ap) in coeffs.iter().zip(alpha_pows) {
        let cv = c.0 as u128;
        for k in 0..EXT_DEG {
            let p = cv * (ap.0[k].0 as u128);
            acc[k] += (p as u64) as u128 + (C as u128) * ((p >> 64) as u128);
        }
    }
    Fq2(core::array::from_fn(|k| Fq(reduce128(acc[k]))))
}

#[derive(Clone, Copy, Debug)]
pub struct LazyExtSum([u128; EXT_DEG]);

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
    pub fn add(&mut self, x: &Fq2) {
        for k in 0..EXT_DEG {
            self.0[k] += x.0[k].0 as u128;
        }
    }
    #[inline(always)]
    pub fn finish(self) -> Fq2 {
        Fq2(core::array::from_fn(|k| Fq(reduce128(self.0[k]))))
    }
}

impl Mul<Fq2> for Fq {
    type Output = Fq2;
    #[inline(always)]
    fn mul(self, r: Fq2) -> Fq2 {
        Fq2([self * r.0[0], self * r.0[1]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_is_nonsquare_so_x2_minus_2_is_irreducible() {
        assert_eq!(Q % 8, 5);
        assert_eq!(Fq::new(2).pow((Q - 1) / 2), Fq::new(Q - 1));
        assert_eq!(Fq::new(Q - 1).pow((Q - 1) / 2), Fq::ONE);
    }

    #[test]
    fn t_squared_is_two() {
        let t = Fq2([Fq::ZERO, Fq::ONE]);
        assert_eq!(t * t, Fq2([Fq::new(2), Fq::ZERO]));
        for k in 1..64u64 {
            let a = Fq2([Fq::new(k), Fq::new(k * 7 + 1)]);
            assert_eq!(a * a.inv(), Fq2::ONE, "k = {k}");
        }
    }

    #[test]
    fn mul_and_inv() {
        let a = Fq2([Fq::new(3), Fq::new(1)]);
        let b = Fq2([Fq::new(5), Fq::new(9)]);
        assert_eq!(a * b, b * a);
        assert_eq!(a * a.inv(), Fq2::ONE);
        assert_eq!((a * b) * b.inv(), a);
    }

    #[test]
    fn fq2_mul_matches_naive() {
        let q = Q as u128;
        let naive = |a: Fq2, b: Fq2| {
            let m = |x: Fq, y: Fq| ((x.0 as u128 * y.0 as u128) % q) as u64;
            let r0 = (m(a.0[0], b.0[0]) as u128 + 2 * m(a.0[1], b.0[1]) as u128) % q;
            let r1 = (m(a.0[0], b.0[1]) as u128 + m(a.0[1], b.0[0]) as u128) % q;
            Fq2([Fq(r0 as u64), Fq(r1 as u64)])
        };
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Fq::new(s ^ (s >> 31))
        };
        for _ in 0..50_000 {
            let a = Fq2([next(), next()]);
            let b = Fq2([next(), next()]);
            assert_eq!(a * b, naive(a, b));
        }
        let e = Fq2([Fq::new(Q - 1); EXT_DEG]);
        assert_eq!(e * e, naive(e, e));
        let z = Fq2([Fq::new(Q - 1), Fq::ZERO]);
        assert_eq!(z * z, naive(z, z));
    }

    #[test]
    fn subfield_embedding_is_homomorphic() {
        let a = Fq::new(123456789012345);
        let b = Fq::new(987654321098765);
        assert_eq!(Fq2::from_fq(a) * Fq2::from_fq(b), Fq2::from_fq(a * b));
    }

    #[test]
    fn poly_eval_pows_matches_horner() {
        let mut s = 12345u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(97);
            Fq::new(s ^ (s >> 29))
        };
        let alpha = Fq2([next(), next()]);
        let coeffs: Vec<Fq> = (0..512).map(|_| next()).collect();
        let mut pows = Vec::with_capacity(512);
        let mut p = Fq2::ONE;
        for _ in 0..512 {
            pows.push(p);
            p = p * alpha;
        }
        let got = poly_eval_pows(&coeffs, &pows);
        let want = coeffs.iter().rev().fold(Fq2::ZERO, |acc, &c| acc * alpha + Fq2::from_fq(c));
        assert_eq!(got, want);
        let worst: Vec<Fq> = vec![Fq::new(Q - 1); 512];
        let wp = vec![Fq2([Fq::new(Q - 1); EXT_DEG]); 512];
        let got_w = poly_eval_pows(&worst, &wp);
        let want_w =
            worst.iter().zip(&wp).fold(Fq2::ZERO, |acc, (&c, &ap)| acc + c * ap);
        assert_eq!(got_w, want_w, "unreduced accumulation bound overflowed");
    }

    #[test]
    fn lazy_ext_sum_survives_worst_case() {
        let x = Fq2([Fq::new(Q - 1); EXT_DEG]);
        let mut acc = LazyExtSum::new();
        let mut want = Fq2::ZERO;
        for _ in 0..512 {
            acc.add(&x);
            want = want + x;
        }
        assert_eq!(acc.finish(), want);
    }
}
