use crate::ext_field::Fq4;
use crate::field::{Fq, Q};
use crate::keccak::Shake128;

const T_RNG: u32 = u32::from_le_bytes(*b"RNG_");

pub struct CsRng {
    h: Shake128,
    bitbuf: u64,
    bitcnt: u32,
}

impl CsRng {
    pub fn from_parts(domain: &str, parts: &[&[u8]]) -> Self {
        let mut h = Shake128::new();
        h.absorb_u32(T_RNG);
        h.absorb_u32(domain.len() as u32);
        h.absorb_bytes(domain.as_bytes());
        h.absorb_u32(parts.len() as u32);
        for p in parts {
            h.absorb_u32(p.len() as u32);
            h.absorb_bytes(p);
        }
        CsRng { h, bitbuf: 0, bitcnt: 0 }
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.h.squeeze_u32()
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        lo | (hi << 32)
    }

    #[inline]
    pub fn next_fq(&mut self) -> Fq {
        loop {
            let v = self.next_u32();
            if (v as u64) < Q {
                return Fq(v);
            }
        }
    }

    pub fn next_fq4(&mut self) -> Fq4 {
        Fq4([self.next_fq(), self.next_fq(), self.next_fq(), self.next_fq()])
    }

    #[inline]
    pub fn next_bool(&mut self) -> bool {
        if self.bitcnt == 0 {
            self.bitbuf = self.next_u64();
            self.bitcnt = 64;
        }
        let b = self.bitbuf & 1 == 1;
        self.bitbuf >>= 1;
        self.bitcnt -= 1;
        b
    }

    pub fn fill_bytes(&mut self, out: &mut [u8]) {
        self.h.squeeze(out);
    }
}

pub fn insecure_test_secret(x: u64) -> [u8; 32] {
    let mut h = Shake128::new();
    h.absorb_bytes(b"INSECURE-TEST-SEED");
    h.absorb_u64(x);
    let mut out = [0u8; 32];
    h.squeeze(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_domain_separated() {
        let s = insecure_test_secret(1);
        let seq = |d: &str, s: &[u8]| {
            let mut r = CsRng::from_parts(d, &[s]);
            (0..8).map(|_| r.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(seq("a", &s), seq("a", &s), "同種子必須決定性");
        assert_ne!(seq("a", &s), seq("b", &s), "domain 不同 ⇒ 序列不同");
        assert_ne!(seq("a", &s), seq("a", &insecure_test_secret(2)), "種子不同 ⇒ 序列不同");
    }

    #[test]
    fn seed_parts_are_framed() {
        let a = CsRng::from_parts("d", &[b"ab", b"c"]).next_u64();
        let b = CsRng::from_parts("d", &[b"a", b"bc"]).next_u64();
        let c = CsRng::from_parts("d", &[b"abc"]).next_u64();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn next_fq_is_uniform_over_range() {
        let mut r = CsRng::from_parts("u", &[&insecure_test_secret(7)]);
        const N: usize = 20000;
        let mut hi = 0usize;
        for _ in 0..N {
            let v = r.next_fq();
            assert!((v.0 as u64) < Q);
            if v.0 as u64 >= Q / 2 {
                hi += 1;
            }
        }
        assert!(hi > N * 45 / 100 && hi < N * 55 / 100, "分布傾斜：上半 {hi}/{N}");
    }

    #[test]
    fn next_bool_is_balanced_across_buffer_boundaries() {
        let mut r = CsRng::from_parts("b", &[&insecure_test_secret(9)]);
        const N: usize = 64 * 1000;
        let ones = (0..N).filter(|_| r.next_bool()).count();
        assert!(ones > N * 48 / 100 && ones < N * 52 / 100, "bit 不平衡：{ones}/{N}");

        let mut r = CsRng::from_parts("b2", &[&insecure_test_secret(10)]);
        let mut per_pos = [0usize; 64];
        for _ in 0..2000 {
            for p in 0..64 {
                if r.next_bool() {
                    per_pos[p] += 1;
                }
            }
        }
        for (p, &c) in per_pos.iter().enumerate() {
            assert!(c > 800 && c < 1200, "buffer 位置 {p} 傾斜：{c}/2000");
        }
    }

    #[test]
    fn mixed_calls_are_deterministic() {
        let run = || {
            let mut r = CsRng::from_parts("m", &[&insecure_test_secret(11)]);
            let mut v = Vec::new();
            for i in 0..50 {
                if i % 3 == 0 {
                    v.push(r.next_u32() as u64);
                } else {
                    v.push(r.next_bool() as u64);
                }
            }
            v
        };
        assert_eq!(run(), run());
    }
}
