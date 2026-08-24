use crate::field::Fq;
use aes::Aes128;
use ctr::cipher::{KeyIvInit, StreamCipher};

type Aes128Ctr = ctr::Ctr64LE<Aes128>;

const BLOCKS: usize = 32;
const BUFLEN: usize = BLOCKS * 16;

pub struct AesPrg {
    ciph: Aes128Ctr,
    buf: [u8; BUFLEN],
    pos: usize,
}

impl AesPrg {
    pub fn from_parts(domain: &str, parts: &[&[u8]]) -> Self {
        let mut h = crate::keccak::Shake128::new();
        h.absorb_u32(u32::from_le_bytes(*b"AESP"));
        h.absorb_u32(domain.len() as u32);
        h.absorb_bytes(domain.as_bytes());
        for p in parts {
            h.absorb_u32(p.len() as u32);
            h.absorb_bytes(p);
        }
        let mut kn = [0u8; 24];
        h.squeeze(&mut kn);
        let mut key = [0u8; 16];
        key.copy_from_slice(&kn[..16]);
        Self::from_key_nonce(key, u64::from_le_bytes(kn[16..].try_into().unwrap()))
    }

    pub fn from_key_nonce(key: [u8; 16], nonce: u64) -> Self {
        let mut iv = [0u8; 16];
        iv[8..].copy_from_slice(&nonce.to_le_bytes());
        AesPrg {
            ciph: Aes128Ctr::new(&key.into(), &iv.into()),
            buf: [0u8; BUFLEN],
            pos: BUFLEN,
        }
    }

    #[inline]
    fn refill(&mut self) {
        self.buf = [0u8; BUFLEN];
        self.ciph.apply_keystream(&mut self.buf);
        self.pos = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == BUFLEN {
            self.refill();
        }
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        (self.next_u32() as u64) | ((self.next_u32() as u64) << 32)
    }

    #[inline]
    pub fn next_fq(&mut self) -> Fq {
        loop {
            let v = crate::field::fq_from_words(|| self.next_u32());
            if v < crate::field::Q {
                return Fq(v as _);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_block_layout_is_ctr_le_nonce_le() {
        use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
        let key: [u8; 16] = std::array::from_fn(|i| i as u8);
        let nonce = 0x0123_4567_89ab_cdefu64;
        let mut prg = AesPrg::from_key_nonce(key, nonce);
        let ciph = Aes128::new(&GenericArray::from(key));
        for j in 0..BLOCKS as u64 + 3 {
            let mut want = [0u8; 16];
            want[..8].copy_from_slice(&j.to_le_bytes());
            want[8..].copy_from_slice(&nonce.to_le_bytes());
            let mut b = GenericArray::from(want);
            ciph.encrypt_block(&mut b);
            let got: Vec<u8> =
                (0..4).flat_map(|_| prg.next_u32().to_le_bytes()).collect();
            assert_eq!(got, b.as_slice(), "block {j}");
        }
    }

    #[test]
    fn keystream_regression() {
        let mut r = AesPrg::from_key_nonce([0x5au8; 16], 0x0123_4567_89ab_cdef);
        let got: Vec<u32> = (0..12).map(|_| r.next_u32()).collect();
        assert_eq!(
            got,
            vec![
                0x6c6fdac7, 0x21109bdd, 0x2d965d01, 0xde4366e6, 0xe52a084e, 0xe547ad78,
                0x7d3bf5bf, 0x886daaae, 0xa63564e2, 0x865a67b8, 0xbf141071, 0x1877a123,
            ]
        );
        let mut r2 = AesPrg::from_parts("kat", &[b"seed"]);
        let got2: Vec<u32> = (0..4).map(|_| r2.next_u32()).collect();
        assert_eq!(got2, vec![0x0d8afdd3, 0x55948392, 0x9ba4e3b2, 0x56aa949f]);
    }

    #[test]
    fn counter_advances_across_refill() {
        let mut r = AesPrg::from_key_nonce([7u8; 16], 42);
        let n = BUFLEN / 4;
        let first: Vec<u32> = (0..n).map(|_| r.next_u32()).collect();
        let second: Vec<u32> = (0..n).map(|_| r.next_u32()).collect();
        assert_ne!(first, second, "second buffer equals the first => the counter did not advance");
    }

    #[test]
    fn domain_separated_and_deterministic() {
        let take = |d: &str, p: &[u8]| {
            let mut r = AesPrg::from_parts(d, &[p]);
            (0..8).map(|_| r.next_u32()).collect::<Vec<_>>()
        };
        assert_eq!(take("a", b"x"), take("a", b"x"));
        assert_ne!(take("a", b"x"), take("b", b"x"));
        assert_ne!(take("a", b"x"), take("a", b"y"));
        let mut r1 = AesPrg::from_parts("dom", &[b"ab", b""]);
        let mut r2 = AesPrg::from_parts("dom", &[b"a", b"b"]);
        assert_ne!(r1.next_u32(), r2.next_u32());
    }

    #[test]
    fn next_fq_is_uniform_over_range() {
        let mut r = AesPrg::from_parts("uniform", &[b"seed"]);
        let (mut hi, n) = (0usize, 200_000usize);
        for _ in 0..n {
            let v = r.next_fq().0 as u64;
            assert!(v < crate::field::Q);
            if v >= crate::field::Q / 2 {
                hi += 1;
            }
        }
        assert!(hi > n * 48 / 100 && hi < n * 52 / 100, "high-half ratio {hi}/{n}");
    }
}
