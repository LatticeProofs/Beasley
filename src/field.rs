use std::ops::{Add, Mul, Neg, Sub};

pub const Q: u64 = 18446744073709551557;

pub const C: u64 = 59;

pub const GENERATOR: u64 = 7;

#[inline(always)]
pub fn mul_c_ref(h: u64) -> u64 {
    debug_assert!(h < 1u64 << 58, "mul_c_ref overflow: h = {h}");
    (h << 6) - (h << 2) - h
}

pub const FQ_BYTES: usize = ((64 - Q.leading_zeros()) as usize).div_ceil(8);

const _: () = assert!(FQ_BYTES == 8, "Fq for q64 must be 8 bytes");

#[inline(always)]
pub fn fq_le_bytes(x: Fq) -> [u8; FQ_BYTES] {
    let mut out = [0u8; FQ_BYTES];
    out.copy_from_slice(&x.0.to_le_bytes()[..FQ_BYTES]);
    out
}

#[inline(always)]
pub fn fq_from_words(mut next_u32: impl FnMut() -> u32) -> u64 {
    let lo = next_u32() as u64;
    if FQ_BYTES > 4 { lo | ((next_u32() as u64) << 32) } else { lo }
}

#[inline(always)]
pub fn reduce128(t: u128) -> u64 {
    let (lo, hi) = (t as u64, (t >> 64) as u64);
    let r1 = lo as u128 + (C as u128) * (hi as u128);
    let (l2, h2) = (r1 as u64, (r1 >> 64) as u64);
    let (s, carry) = l2.overflowing_add(mul_c_ref(h2));
    let s = if carry { s.wrapping_add(C) } else { s };
    if s >= Q { s - Q } else { s }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Fq(pub u64);

impl Fq {
    pub const ZERO: Fq = Fq(0);
    pub const ONE: Fq = Fq(1);

    pub fn new(v: u64) -> Self {
        Fq(v % Q)
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
        let (s, carry) = self.0.overflowing_add(rhs.0);
        let s = if carry { s.wrapping_add(C) } else { s };
        Fq(if s >= Q { s - Q } else { s })
    }
}

impl Sub for Fq {
    type Output = Fq;
    #[inline(always)]
    fn sub(self, rhs: Fq) -> Fq {
        Fq(if self.0 >= rhs.0 { self.0 - rhs.0 } else { self.0.wrapping_sub(rhs.0).wrapping_add(Q) })
    }
}

impl Mul for Fq {
    type Output = Fq;
    #[inline(always)]
    fn mul(self, rhs: Fq) -> Fq {
        Fq(reduce128((self.0 as u128) * (rhs.0 as u128)))
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

    fn naive(t: u128) -> u64 {
        (t % Q as u128) as u64
    }

    #[test]
    fn basic_ops() {
        let a = Fq::new(Q - 1);
        let b = Fq::new(2);
        assert_eq!(a + b, Fq::ONE);
        assert_eq!(Fq::ZERO - b, Fq::new(Q - 2));
        assert_eq!(a * a, Fq::ONE);
        assert_eq!(a + a, Fq::new(Q - 2));
    }

    #[test]
    fn inverse() {
        for v in [1u64, 2, 7, 12345, Q - 1, Q / 2] {
            let a = Fq::new(v);
            assert_eq!(a * a.inv(), Fq::ONE);
        }
    }

    #[test]
    fn reduce128_matches_u128_mod() {
        let mut s: u128 = 0x1234_5678_9abc_def0_0fed_cba9_8765_4321;
        for _ in 0..1_000_000 {
            s = s
                .wrapping_mul(0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645)
                .wrapping_add(0x1442_7952_1CBD_A5B0_5D0A_A83F_7E6A_1B7F);
            assert_eq!(reduce128(s), naive(s), "t = {s}");
        }
        let m = (Q - 1) as u128;
        for t in [
            m * m,
            (Q as u128) * (Q as u128) - 1,
            u128::MAX,
            u128::MAX - 1,
            1u128 << 64,
            (1u128 << 64) - 1,
            Q as u128,
            (Q as u128) - 1,
            (Q as u128) * 3,
            0,
        ] {
            assert_eq!(reduce128(t), naive(t), "t = {t}");
        }
        for hi in [C, C - 1, C - 2, 1u64, 2, 0xFFFF_FFFF_FFFF_FFFF] {
            for delta in 0..2048u64 {
                let lo = u64::MAX - delta;
                let t = ((hi as u128) << 64) | lo as u128;
                assert_eq!(reduce128(t), naive(t), "hi = {hi}, lo = {lo}");
            }
        }
    }

    #[test]
    fn generator_has_full_order() {
        let factors: [u64; 5] = [2, 11, 137, 547, 5594472617641];
        let mut prod: u128 = 2;
        for f in factors {
            prod *= f as u128;
        }
        assert_eq!(prod, (Q - 1) as u128, "prime factorization is wrong");
        let g = Fq::new(GENERATOR);
        for p in factors {
            assert_ne!(g.pow((Q - 1) / p), Fq::ONE, "GENERATOR = {GENERATOR} is not a generator for p = {p}");
        }
        for bad in [4u64, 6, 9, 10, 11] {
            let b = Fq::new(bad);
            assert!(
                factors.iter().any(|&p| b.pow((Q - 1) / p) == Fq::ONE),
                "{bad} turns out to be a generator?"
            );
        }
    }

    #[test]
    fn roots_of_unity() {
        assert_eq!((Q - 1) % 4, 0);
        assert_ne!((Q - 1) % 8, 0, "v_2(q-1) should be 2");
        let w = Fq::root_of_unity(2);
        assert_eq!(w, Fq::new(Q - 1));
        assert_eq!(w.pow(2), Fq::ONE);
        assert_eq!(Fq::root_of_unity(4).pow(4), Fq::ONE);
        assert_ne!((Q - 1) % (2 * crate::ring::N as u64), 0, "if it divides, a direct NTT should be used instead");
    }

    #[test]
    fn q_is_five_mod_eight() {
        assert_eq!(Q % 8, 5);
        assert_eq!((1u128 << 64) - Q as u128, C as u128, "2^64 - q must equal C");
        assert_eq!(Fq::new(2).pow((Q - 1) / 2), Fq::new(Q - 1), "2 must be a non-residue");
        assert_eq!(Fq::new(Q - 1).pow((Q - 1) / 2), Fq::ONE, "-1 turns out to be a non-residue?");
    }

    #[test]
    fn mul_c_ref_matches_multiplication() {
        for h in [0u64, 1, 2, C - 1, C, C + 1, 1000, (1u64 << 58) - 1] {
            assert_eq!(mul_c_ref(h), C.wrapping_mul(h), "h = {h}");
        }
        let mut s: u64 = 0x9182_3746_5510_ABCD;
        for _ in 0..20000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let h = s >> 6;
            assert_eq!(mul_c_ref(h), C.wrapping_mul(h), "h = {h}");
        }
    }

    #[test]
    fn serialisation_width() {
        assert_eq!(FQ_BYTES, 8);
        let x = Fq::new(0x0123_4567_89ab_cdef);
        assert_eq!(fq_le_bytes(x), 0x0123_4567_89ab_cdefu64.to_le_bytes());
    }
}
