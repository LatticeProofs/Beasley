use crate::ext_field::Fq4;
use crate::field::{Fq, Q};
use crate::keccak::Shake128;

const T_DOMAIN: u32 = u32::from_le_bytes(*b"DOMN");
const T_U64: u32 = u32::from_le_bytes(*b"U64_");
const T_FQ: u32 = u32::from_le_bytes(*b"FQ__");
const T_FQS: u32 = u32::from_le_bytes(*b"FQS_");
const T_FQ4: u32 = u32::from_le_bytes(*b"FQ4_");
const T_DIGEST: u32 = u32::from_le_bytes(*b"DGST");
const T_CHAL: u32 = u32::from_le_bytes(*b"CHAL");

pub struct Transcript {
    h: Shake128,
}

impl Transcript {
    pub fn new(domain: &str) -> Self {
        let mut h = Shake128::new();
        h.absorb_u32(T_DOMAIN);
        h.absorb_u32(domain.len() as u32);
        h.absorb_bytes(domain.as_bytes());
        Transcript { h }
    }

    pub fn absorb_u64(&mut self, x: u64) {
        self.h.absorb_u32(T_U64);
        self.h.absorb_u64(x);
    }

    pub fn absorb_fq(&mut self, x: Fq) {
        self.h.absorb_u32(T_FQ);
        self.h.absorb_u32(x.0);
    }

    pub fn absorb_fqs(&mut self, xs: &[Fq]) {
        self.h.absorb_u32(T_FQS);
        self.h.absorb_u32(xs.len() as u32);
        for &x in xs {
            self.h.absorb_u32(x.0);
        }
    }

    pub fn absorb_fq4(&mut self, x: Fq4) {
        self.h.absorb_u32(T_FQ4);
        for c in x.0 {
            self.h.absorb_u32(c.0);
        }
    }

    pub fn absorb_digest(&mut self, d: &[u8; 32]) {
        self.h.absorb_u32(T_DIGEST);
        self.h.absorb_bytes(d);
    }

    pub fn challenge_fq(&mut self) -> Fq {
        self.h.absorb_u32(T_CHAL);
        loop {
            let v = self.h.squeeze_u32();
            if (v as u64) < Q {
                return Fq(v);
            }
        }
    }

    pub fn challenge_fq4(&mut self) -> Fq4 {
        Fq4([
            self.challenge_fq(),
            self.challenge_fq(),
            self.challenge_fq(),
            self.challenge_fq(),
        ])
    }

    pub fn challenge_u64(&mut self) -> u64 {
        self.h.absorb_u32(T_CHAL);
        let lo = self.h.squeeze_u32() as u64;
        let hi = self.h.squeeze_u32() as u64;
        lo | (hi << 32)
    }

    pub fn finalize_digest(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        self.h.squeeze(&mut out);
        out
    }
}

pub struct SimpleRng {
    state: u64,
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: splitmix64(seed ^ 0xA5A5_5A5A_DEAD_BEEF) }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    pub fn next_fq(&mut self) -> Fq {
        Fq::new(self.next_u64() % Q)
    }

    pub fn next_fq4(&mut self) -> Fq4 {
        Fq4([self.next_fq(), self.next_fq(), self.next_fq(), self.next_fq()])
    }

    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(f: impl Fn(&mut Transcript)) -> Fq4 {
        let mut t = Transcript::new("test");
        f(&mut t);
        t.challenge_fq4()
    }

    #[test]
    fn transcript_is_deterministic_and_order_sensitive() {
        assert_eq!(
            ch(|t| {
                t.absorb_u64(1);
                t.absorb_u64(2)
            }),
            ch(|t| {
                t.absorb_u64(1);
                t.absorb_u64(2)
            })
        );
        assert_ne!(
            ch(|t| {
                t.absorb_u64(1);
                t.absorb_u64(2)
            }),
            ch(|t| {
                t.absorb_u64(2);
                t.absorb_u64(1)
            })
        );
        let mut a = Transcript::new("dom-a");
        let mut b = Transcript::new("dom-b");
        assert_ne!(a.challenge_fq4(), b.challenge_fq4());
    }

    #[test]
    fn absorb_types_are_domain_separated() {
        let one = Fq::ONE;
        let by_u64 = ch(|t| t.absorb_u64(1));
        let by_fq = ch(|t| t.absorb_fq(one));
        let by_fqs = ch(|t| t.absorb_fqs(&[one]));
        let by_fq4 = ch(|t| t.absorb_fq4(Fq4::ONE));
        let all = [by_u64, by_fq, by_fqs, by_fq4];
        for i in 0..4 {
            for j in i + 1..4 {
                assert_ne!(all[i], all[j], "型別 {i} 與 {j} 撞了");
            }
        }
        let two = Fq::new(2);
        assert_ne!(
            ch(|t| t.absorb_fqs(&[one, two])),
            ch(|t| {
                t.absorb_fqs(&[one]);
                t.absorb_fqs(&[two])
            })
        );
    }

    #[test]
    fn canonical_encoding_is_pinned() {
        let mut t = Transcript::new("pin");
        t.absorb_fqs(&[Fq(0), Fq(1), Fq(Q as u32 - 1)]);
        t.absorb_fq4(Fq4([Fq(7), Fq(8), Fq(9), Fq(10)]));
        t.absorb_u64(0x0123_4567_89ab_cdef);
        let d = t.challenge_fq4();
        assert_eq!(
            [d.0[0].0, d.0[1].0, d.0[2].0, d.0[3].0],
            [379608887, 773585188, 179761591, 1944056472],
            "transcript 的 canonical 編碼改變了 —— 這會讓舊 proof 全部失效，確認是有意的"
        );
    }

    #[test]
    fn challenges_are_uniform_over_fq() {
        let mut t = Transcript::new("unif");
        let mut hi = 0usize;
        const N: usize = 20000;
        for _ in 0..N {
            let v = t.challenge_fq();
            assert!((v.0 as u64) < Q);
            if v.0 as u64 >= Q / 2 {
                hi += 1;
            }
        }
        assert!(hi > N * 45 / 100 && hi < N * 55 / 100, "分布傾斜：上半 {hi}/{N}");
    }

    #[test]
    fn challenge_binds_subsequent_absorbs() {
        let with = {
            let mut t = Transcript::new("dup");
            t.absorb_u64(1);
            let _ = t.challenge_fq4();
            t.absorb_u64(2);
            t.challenge_fq4()
        };
        let without = {
            let mut t = Transcript::new("dup");
            t.absorb_u64(1);
            t.absorb_u64(2);
            t.challenge_fq4()
        };
        assert_ne!(with, without);

        let a = {
            let mut t = Transcript::new("dup");
            t.absorb_u64(1);
            let _ = t.challenge_fq4();
            t.challenge_fq4()
        };
        let b = {
            let mut t = Transcript::new("dup");
            t.absorb_u64(1);
            let _ = t.challenge_fq4();
            let _ = t.challenge_fq4();
            t.challenge_fq4()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn large_absorb_is_avalanche_sensitive() {
        let big: Vec<Fq> = (0..5000u64).map(Fq::new).collect();
        let base = ch(|t| t.absorb_fqs(&big));
        let mut tweaked = big.clone();
        tweaked[4999] = tweaked[4999] + Fq::ONE;
        assert_ne!(base, ch(|t| t.absorb_fqs(&tweaked)));
        assert_eq!(base, ch(|t| t.absorb_fqs(&big)));
    }
}
