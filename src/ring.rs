use crate::ext_field::FqExt;
use crate::field::Fq;
use crate::ntt::negacyclic_mul_n;
use std::ops::{Add, Mul, Neg, Sub};

pub const N: usize = 512;

const _: () = assert!(N.is_power_of_two(), "N must be a power of two, otherwise X^N+1 is reducible (P1)");

pub const DIGIT_BITS: usize = 8;

pub const GADGET_BASE: u64 = 1 << DIGIT_BITS;

pub const W_RANGE_BASE: u64 = 2;

pub const M_BIT_ROWS: usize = 64;

pub const GADGET_LEN: usize = M_BIT_ROWS / DIGIT_BITS;

const _: () = assert!(
    GADGET_LEN * DIGIT_BITS == M_BIT_ROWS,
    "DIGIT_BITS must divide M_BIT_ROWS, otherwise the top digit is not fully packed"
);

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

    pub fn eval(&self, alpha: FqExt) -> FqExt {
        poly_eval(&self.c, alpha)
    }
}

pub fn poly_eval(coeffs: &[Fq], alpha: FqExt) -> FqExt {
    let mut acc = FqExt::ZERO;
    for &c in coeffs.iter().rev() {
        acc = acc * alpha + FqExt::from_fq(c);
    }
    acc
}

pub use crate::ext_field::poly_eval_pows;

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
        RingElem { c: negacyclic_mul_n(&self.c, &r.c) }
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
    debug_assert!(d < GADGET_LEN);
    Fq::new(GADGET_BASE.pow(d as u32))
}

pub fn bit_weight(k: usize) -> Fq {
    debug_assert!(k < M_BIT_ROWS);
    Fq::new(1u64 << k)
}

pub fn gadget_decompose(a: &RingElem) -> Vec<RingElem> {
    const MASK: u64 = GADGET_BASE - 1;
    (0..GADGET_LEN)
        .map(|d| {
            let sh = d * DIGIT_BITS;
            RingElem { c: a.c.iter().map(|&x| Fq(((x.0 as u64 >> sh) & MASK) as _)).collect() }
        })
        .collect()
}

pub fn gadget_decompose_slice(coeffs: &[Fq]) -> Vec<Vec<Fq>> {
    let mut out = vec![vec![Fq::ZERO; coeffs.len()]; GADGET_LEN];
    for (i, &coef) in coeffs.iter().enumerate() {
        let mut v = coef.0 as u64;
        for d in 0..GADGET_LEN {
            out[d][i] = Fq::new(v % GADGET_BASE);
            v /= GADGET_BASE;
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
    fn modulus_is_irreducible_over_z() {
        assert!(N.is_power_of_two(), "N = {N} is not a power of two => X^N+1 is reducible (P1)");
        assert_eq!(M_BIT_ROWS, (64 - crate::field::Q.leading_zeros()) as usize);
    }

    #[test]
    fn gadget_constants_are_consistent() {
        assert_eq!(GADGET_BASE, 1u64 << DIGIT_BITS);
        assert_eq!(GADGET_LEN * DIGIT_BITS, M_BIT_ROWS, "the digit grouping must exactly cover M_BIT_ROWS");
        assert_eq!(
            M_BIT_ROWS,
            (64 - crate::field::Q.leading_zeros()) as usize,
            "M_BIT_ROWS must be ⌈log₂ q⌉"
        );
        assert_eq!(W_RANGE_BASE, 2, "the W table only ever holds bits");
        assert!(
            (GADGET_LEN as u32) * (DIGIT_BITS as u32) >= 64 - crate::field::Q.leading_zeros(),
            "gadget too short: the range of the top digit does not cover q"
        );
    }

    #[test]
    fn bit_weight_is_the_composite_gadget() {
        for d in 0..GADGET_LEN {
            for b in 0..DIGIT_BITS {
                let composed = gadget_scalar(d) * Fq::new(1u64 << b);
                assert_eq!(composed, bit_weight(d * DIGIT_BITS + b), "d={d} b={b}");
            }
        }
        if DIGIT_BITS == 1 {
            for k in 0..M_BIT_ROWS {
                assert_eq!(gadget_scalar(k), bit_weight(k));
            }
        }
    }

    #[test]
    fn two_level_decomposition_roundtrip() {
        let mut rng = SimpleRng::new(20260801);
        let a = random_elem(&mut rng);
        let digits = gadget_decompose(&a);
        assert_eq!(digits.len(), GADGET_LEN);
        let mut acc = RingElem::zero();
        for (d, dig) in digits.iter().enumerate() {
            for i in 0..N {
                let v = dig.c[i].0 as u64;
                assert!(v < GADGET_BASE, "digit out of range [0,B)");
                for b in 0..DIGIT_BITS {
                    let bit = (v >> b) & 1;
                    assert!(bit < W_RANGE_BASE, "the second layer must be bits");
                    acc.c[i] = acc.c[i] + bit_weight(d * DIGIT_BITS + b) * Fq::new(bit);
                }
            }
        }
        assert_eq!(acc, a, "the two-layer decomposition does not recompose to the original value");
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
        let rhs = red.eval(alpha) + (alpha.pow(N as u128) + FqExt::ONE) * poly_eval(&t, alpha);
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
        assert_eq!(digits.len(), GADGET_LEN);
        for m in &digits {
            for &c in &m.c {
                assert!((c.0 as u64) < GADGET_BASE);
            }
        }
        assert_eq!(gadget_recompose(&digits), a);
    }
}
