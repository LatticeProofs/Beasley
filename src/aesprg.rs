use crate::field::Fq;

const BLOCKS: usize = 8;
const BUFLEN: usize = BLOCKS * 16;

pub struct AesPrg {
    rk: [u8; 176],
    nonce: u64,
    ctr: u64,
    buf: [u8; BUFLEN],
    pos: usize,
    bitbuf: u64,
    bitcnt: u32,
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
        AesPrg {
            rk: key_expansion(&key),
            nonce,
            ctr: 0,
            buf: [0u8; BUFLEN],
            pos: BUFLEN,
            bitbuf: 0,
            bitcnt: 0,
        }
    }

    #[inline]
    fn refill(&mut self) {
        let mut blocks = [[0u8; 16]; BLOCKS];
        for (j, b) in blocks.iter_mut().enumerate() {
            b[..8].copy_from_slice(&(self.ctr + j as u64).to_le_bytes());
            b[8..].copy_from_slice(&self.nonce.to_le_bytes());
        }
        self.ctr += BLOCKS as u64;
        encrypt_blocks(&self.rk, &mut blocks);
        for (j, b) in blocks.iter().enumerate() {
            self.buf[j * 16..(j + 1) * 16].copy_from_slice(b);
        }
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
        for chunk in out.chunks_mut(4) {
            let w = self.next_u32().to_le_bytes();
            chunk.copy_from_slice(&w[..chunk.len()]);
        }
    }
}

const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

fn key_expansion(key: &[u8; 16]) -> [u8; 176] {
    let mut rk = [0u8; 176];
    rk[..16].copy_from_slice(key);
    for i in 4..44 {
        let mut t = [rk[(i - 1) * 4], rk[(i - 1) * 4 + 1], rk[(i - 1) * 4 + 2], rk[(i - 1) * 4 + 3]];
        if i % 4 == 0 {
            t.rotate_left(1);
            for b in t.iter_mut() {
                *b = SBOX[*b as usize];
            }
            t[0] ^= RCON[i / 4 - 1];
        }
        for j in 0..4 {
            rk[i * 4 + j] = rk[(i - 4) * 4 + j] ^ t[j];
        }
    }
    rk
}

#[inline]
fn xtime(x: u8) -> u8 {
    (x << 1) ^ (((x >> 7) & 1) * 0x1b)
}

fn encrypt_block_soft(rk: &[u8; 176], b: &mut [u8; 16]) {
    for j in 0..16 {
        b[j] ^= rk[j];
    }
    for r in 1..=10 {
        for x in b.iter_mut() {
            *x = SBOX[*x as usize];
        }
        let s = *b;
        for c in 0..4 {
            for row in 0..4 {
                b[c * 4 + row] = s[((c + row) % 4) * 4 + row];
            }
        }
        if r != 10 {
            for c in 0..4 {
                let a = [b[c * 4], b[c * 4 + 1], b[c * 4 + 2], b[c * 4 + 3]];
                let t = a[0] ^ a[1] ^ a[2] ^ a[3];
                for row in 0..4 {
                    b[c * 4 + row] = a[row] ^ t ^ xtime(a[row] ^ a[(row + 1) % 4]);
                }
            }
        }
        for j in 0..16 {
            b[j] ^= rk[r * 16 + j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "aes,sse2")]
unsafe fn encrypt_blocks_ni(rk: &[u8; 176], blocks: &mut [[u8; 16]; BLOCKS]) {
    use std::arch::x86_64::*;
    unsafe {
        let k: [__m128i; 11] =
            std::array::from_fn(|i| _mm_loadu_si128(rk.as_ptr().add(i * 16) as *const __m128i));
        let mut b: [__m128i; BLOCKS] =
            std::array::from_fn(|j| _mm_loadu_si128(blocks[j].as_ptr() as *const __m128i));
        for x in b.iter_mut() {
            *x = _mm_xor_si128(*x, k[0]);
        }
        for rk_r in k.iter().take(10).skip(1) {
            for x in b.iter_mut() {
                *x = _mm_aesenc_si128(*x, *rk_r);
            }
        }
        for x in b.iter_mut() {
            *x = _mm_aesenclast_si128(*x, k[10]);
        }
        for (j, x) in b.iter().enumerate() {
            _mm_storeu_si128(blocks[j].as_mut_ptr() as *mut __m128i, *x);
        }
    }
}

#[inline]
fn encrypt_blocks(rk: &[u8; 176], blocks: &mut [[u8; 16]; BLOCKS]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("aes") {
            unsafe { encrypt_blocks_ni(rk, blocks) };
            return;
        }
    }
    for b in blocks.iter_mut() {
        encrypt_block_soft(rk, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fips197_c1_vector() {
        let key: [u8; 16] = std::array::from_fn(|i| i as u8);
        let pt: [u8; 16] =
            [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        let want: [u8; 16] =
            [0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a];
        let rk = key_expansion(&key);

        let mut soft = pt;
        encrypt_block_soft(&rk, &mut soft);
        assert_eq!(soft, want, "software AES-128 does not match FIPS-197 C.1");

        let mut blocks = [pt; BLOCKS];
        encrypt_blocks(&rk, &mut blocks);
        for (j, b) in blocks.iter().enumerate() {
            if j == 0 {
                assert_eq!(*b, want, "encrypt_blocks (possibly via AES-NI) does not match FIPS-197 C.1");
            }
        }
    }

    #[test]
    fn ni_matches_soft() {
        let rk = key_expansion(&[0x5au8; 16]);
        let mut blocks: [[u8; 16]; BLOCKS] =
            std::array::from_fn(|j| std::array::from_fn(|i| (i * 7 + j * 31) as u8));
        let mut want = blocks;
        for b in want.iter_mut() {
            encrypt_block_soft(&rk, b);
        }
        encrypt_blocks(&rk, &mut blocks);
        assert_eq!(blocks, want);
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
        assert!(hi > n * 48 / 100 && hi < n * 52 / 100, "upper-half ratio {hi}/{n}");
    }
}
