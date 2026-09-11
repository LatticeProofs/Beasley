
use crate::ext_field::FqExt;
use crate::field::Fq;
use crate::ntt::negacyclic_mul_n;
use std::ops::{Add, Mul, Neg, Sub};

pub const N: usize = 512;

const _: () = assert!(N.is_power_of_two(), "N must be a power of two, otherwise X^N+1 is reducible");

pub const FQ_BITS: usize = 64;

const _: () = assert!(
    FQ_BITS == (64 - crate::field::Q.leading_zeros()) as usize,
    "FQ_BITS must be ⌈log₂ q⌉"
);

pub const W_RANGE_BASE: u64 = 2;

pub fn bit_weight(k: usize) -> Fq {
    debug_assert!(k < FQ_BITS);
    Fq::new(1u64 << k)
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingElem {
    pub c: Vec<Fq>,
}

impl RingElem {
    pub fn zero() -> Self {
        RingElem { c: vec![Fq::ZERO; N] }
    }

    pub fn is_binary(&self) -> bool {
        self.c.iter().all(|c| c.0 <= 1)
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
        assert!(N.is_power_of_two(), "N = {N} is not a power of two ⇒ X^N+1 is reducible");
        assert_eq!(FQ_BITS, (64 - crate::field::Q.leading_zeros()) as usize);
        assert_eq!(W_RANGE_BASE, 2);
    }

    #[test]
    fn bit_weight_is_a_power_of_two() {
        for k in 0..FQ_BITS {
            assert_eq!(bit_weight(k), Fq::new(1u64 << k));
            assert_eq!(bit_weight(k).0, 1u64 << k, "k = {k} was unexpectedly reduced");
        }
    }

    #[test]
    fn quotient_identity_at_random_alpha() {
        let mut rng = SimpleRng::new(42);
        let a = random_elem(&mut rng);
        let b = random_elem(&mut rng);
        let full = mul_polys(&a.c, &b.c);
        let (red, hi) = reduce_full(&full);

        let alpha = rng.next_fq4();
        let lhs = poly_eval(&full, alpha);
        let rhs = red.eval(alpha) + (alpha.pow(N as u128) + FqExt::ONE) * poly_eval(&hi, alpha);
        assert_eq!(lhs, rhs);
        assert_eq!(lhs, a.eval(alpha) * b.eval(alpha));
        assert_eq!(hi.len(), N - 1);
    }

    #[test]
    fn protocol_convention_uses_negated_quotient() {
        let mut rng = SimpleRng::new(4242);
        let a = random_elem(&mut rng);
        let b = RingElem { c: (0..N).map(|_| if rng.next_bool() { Fq::ONE } else { Fq::ZERO }).collect() };
        let full = mul_polys(&a.c, &b.c);
        let (red, hi) = reduce_full(&full);
        let t: Vec<Fq> = hi.iter().map(|&v| -v).collect();

        let alpha = rng.next_fq4();
        let xn1 = alpha.pow(N as u128) + FqExt::ONE;
        assert_eq!(red.eval(alpha), poly_eval(&full, alpha) + xn1 * poly_eval(&t, alpha));
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
}
