use crate::ext_field::Fq4;
use crate::field::{reduce64, Fq, C};
use crate::ntt::negacyclic_mul_1536;
use std::ops::{Add, Mul, Neg, Sub};

pub const N: usize = 1536;

pub const BASE: u64 = 2;
pub const DELTA: usize = 32;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingElem {
    pub c: Vec<Fq>,
}

impl RingElem {
    pub fn zero() -> Self {
        RingElem { c: vec![Fq::ZERO; N] }
    }

    pub fn constant(v: u64) -> Self {
        let mut r = RingElem::zero();
        r.c[0] = Fq::new(v);
        r
    }

    pub fn eval(&self, alpha: Fq4) -> Fq4 {
        poly_eval(&self.c, alpha)
    }
}

pub fn poly_eval(coeffs: &[Fq], alpha: Fq4) -> Fq4 {
    let mut acc = Fq4::ZERO;
    for &c in coeffs.iter().rev() {
        acc = acc * alpha + Fq4::from_fq(c);
    }
    acc
}

pub fn poly_eval_pows(coeffs: &[Fq], alpha_pows: &[Fq4]) -> Fq4 {
    let mut acc = [0u64; 4];
    for (&c, ap) in coeffs.iter().zip(alpha_pows) {
        let cv = c.0 as u64;
        for k in 0..4 {
            let p = cv * ap.0[k].0 as u64;
            acc[k] += (p & 0xFFFF_FFFF) + C * (p >> 32);
        }
    }
    Fq4([
        Fq(reduce64(acc[0])),
        Fq(reduce64(acc[1])),
        Fq(reduce64(acc[2])),
        Fq(reduce64(acc[3])),
    ])
}

impl Add for &RingElem {
    type Output = RingElem;
    fn add(self, r: &RingElem) -> RingElem {
        RingElem { c: self.c.iter().zip(&r.c).map(|(&a, &b)| a + b).collect() }
    }
}

impl Sub for &RingElem {
    type Output = RingElem;
    fn sub(self, r: &RingElem) -> RingElem {
        RingElem { c: self.c.iter().zip(&r.c).map(|(&a, &b)| a - b).collect() }
    }
}

impl Neg for &RingElem {
    type Output = RingElem;
    fn neg(self) -> RingElem {
        RingElem { c: self.c.iter().map(|&a| -a).collect() }
    }
}

impl Mul for &RingElem {
    type Output = RingElem;
    fn mul(self, r: &RingElem) -> RingElem {
        RingElem { c: negacyclic_mul_1536(&self.c, &r.c) }
    }
}

pub fn reduce_only(full: &[Fq]) -> RingElem {
    assert!(full.len() <= 2 * N - 1);
    let mut red = vec![Fq::ZERO; N];
    let lo = full.len().min(N);
    red[..lo].copy_from_slice(&full[..lo]);
    for (k, &v) in full.iter().skip(N).enumerate() {
        red[k] = red[k] - v;
    }
    RingElem { c: red }
}

pub fn reduce_full(full: &[Fq]) -> (RingElem, Vec<Fq>) {
    let mut t = vec![Fq::ZERO; N - 1];
    if full.len() > N {
        t[..full.len() - N].copy_from_slice(&full[N..]);
    }
    (reduce_only(full), t)
}

pub fn gadget_scalar(d: usize) -> Fq {
    Fq::new(BASE.pow(d as u32))
}

pub fn gadget_decompose(a: &RingElem) -> Vec<RingElem> {
    const _: () = assert!(BASE == 2, "此特化假設 base-2 gadget");
    (0..DELTA)
        .map(|d| RingElem { c: a.c.iter().map(|&x| Fq((x.0 >> d) & 1)).collect() })
        .collect()
}

pub fn gadget_decompose_slice(coeffs: &[Fq]) -> Vec<Vec<Fq>> {
    let mut out = vec![vec![Fq::ZERO; coeffs.len()]; DELTA];
    for (i, &coef) in coeffs.iter().enumerate() {
        let mut v = coef.0 as u64;
        for d in 0..DELTA {
            out[d][i] = Fq::new(v % BASE);
            v /= BASE;
        }
        debug_assert_eq!(v, 0);
    }
    out
}

pub fn gadget_recompose(digits: &[RingElem]) -> RingElem {
    let mut acc = RingElem::zero();
    for (d, m) in digits.iter().enumerate() {
        let g = gadget_scalar(d);
        for i in 0..N {
            acc.c[i] = acc.c[i] + g * m.c[i];
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::mul_polys;
    use crate::transcript::SimpleRng;

    fn random_elem(rng: &mut SimpleRng) -> RingElem {
        RingElem { c: (0..N).map(|_| rng.next_fq()).collect() }
    }

    #[test]
    fn quotient_identity_at_random_alpha() {
        let mut rng = SimpleRng::new(42);
        let a = random_elem(&mut rng);
        let b = random_elem(&mut rng);
        let full = mul_polys(&a.c, &b.c);
        let (red, t) = reduce_full(&full);

        let alpha = rng.next_fq4();
        let lhs = poly_eval(&full, alpha);
        let rhs = red.eval(alpha) + (alpha.pow(N as u128) + Fq4::ONE) * poly_eval(&t, alpha);
        assert_eq!(lhs, rhs);
        assert_eq!(lhs, a.eval(alpha) * b.eval(alpha));
    }

    #[test]
    fn negacyclic_wraparound() {
        let mut a = RingElem::zero();
        a.c[N - 1] = Fq::ONE;
        let mut b = RingElem::zero();
        b.c[1] = Fq::ONE;
        let mut expect = RingElem::zero();
        expect.c[0] = -Fq::ONE;
        assert_eq!(&a * &b, expect);
    }

    #[test]
    fn gadget_roundtrip_and_digit_range() {
        let mut rng = SimpleRng::new(7);
        let a = random_elem(&mut rng);
        let digits = gadget_decompose(&a);
        assert_eq!(digits.len(), DELTA);
        for m in &digits {
            for &c in &m.c {
                assert!(c.0 < BASE as u32);
            }
        }
        assert_eq!(gadget_recompose(&digits), a);
    }
}
