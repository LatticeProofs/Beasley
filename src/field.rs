use std::ops::{Add, Mul, Neg, Sub};

pub const Q: u64 = 4294967197;

pub const C: u64 = 99;

pub const GENERATOR: u64 = 6;

pub const FQ_BYTES: usize = ((64 - Q.leading_zeros()) as usize).div_ceil(8);

const _: () = assert!(FQ_BYTES == 4, "q32 Fq must be 4 bytes");

#[inline(always)]
pub fn fq_le_bytes(x: Fq) -> [u8; FQ_BYTES] {
    let mut out = [0u8; FQ_BYTES];
    out.copy_from_slice(&(x.0 as u64).to_le_bytes()[..FQ_BYTES]);
    out
}

#[inline(always)]
pub fn fq_from_words(mut next_u32: impl FnMut() -> u32) -> u64 {
    let lo = next_u32() as u64;
    if FQ_BYTES > 4 { lo | ((next_u32() as u64) << 32) } else { lo }
}

#[inline(always)]
pub fn reduce64(v: u64) -> u32 {
    let v = (v & 0xFFFF_FFFF) + C * (v >> 32);
    let v = (v & 0xFFFF_FFFF) + C * (v >> 32);
    (if v >= Q { v - Q } else { v }) as u32
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Fq(pub u32);

impl Fq {
    pub const ZERO: Fq = Fq(0);
    pub const ONE: Fq = Fq(1);

    pub fn new(v: u64) -> Self {
        Fq((v % Q) as u32)
    }

    pub fn pow(self, mut e: u64) -> Self {
        let mut base = self;
        let mut acc = Fq::ONE;
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
        self.pow(Q - 2)
    }

    pub fn root_of_unity(order: u64) -> Self {
        assert_eq!((Q - 1) % order, 0, "order must divide q-1");
        Fq::new(GENERATOR).pow((Q - 1) / order)
    }
}

impl Add for Fq {
    type Output = Fq;
    #[inline(always)]
    fn add(self, rhs: Fq) -> Fq {
        let s = self.0 as u64 + rhs.0 as u64;
        Fq(if s >= Q { (s - Q) as u32 } else { s as u32 })
    }
}

impl Sub for Fq {
    type Output = Fq;
    #[inline(always)]
    fn sub(self, rhs: Fq) -> Fq {
        let s = self.0 as u64 + Q - rhs.0 as u64;
        Fq(if s >= Q { (s - Q) as u32 } else { s as u32 })
    }
}

impl Mul for Fq {
    type Output = Fq;
    #[inline(always)]
    fn mul(self, rhs: Fq) -> Fq {
        Fq(reduce64(self.0 as u64 * rhs.0 as u64))
    }
}

impl Neg for Fq {
    type Output = Fq;
    #[inline(always)]
    fn neg(self) -> Fq {
        Fq::ZERO - self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let a = Fq::new(Q - 1);
        let b = Fq::new(2);
        assert_eq!(a + b, Fq::ONE);
        assert_eq!(Fq::ZERO - b, Fq::new(Q - 2));
        assert_eq!(a * a, Fq::ONE);
    }

    #[test]
    fn inverse() {
        for v in [1u64, 2, 6, 12345, Q - 1] {
            let a = Fq::new(v);
            assert_eq!(a * a.inv(), Fq::ONE);
        }
    }

    #[test]
    fn reduce_matches_mod() {
        let mut s = 0x1234_5678_9abc_def0u64;
        for _ in 0..100000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            assert_eq!(reduce64(s) as u64, s % Q);
        }
        let m = Q - 1;
        assert_eq!(reduce64(m * m) as u64, (m * m) % Q);
    }

    #[test]
    fn generator_has_full_order() {
        let g = Fq::new(GENERATOR);
        for p in [2u64, 3, 13, 67, 163, 2521] {
            assert_ne!(g.pow((Q - 1) / p), Fq::ONE);
        }
    }

    #[test]
    fn roots_of_unity() {
        let w = Fq::root_of_unity(4);
        assert_eq!(w.pow(4), Fq::ONE);
        assert_eq!(w.pow(2), Fq::new(Q - 1));
    }
}
