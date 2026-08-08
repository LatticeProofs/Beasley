pub const RATE: usize = 168;

const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];
const RHO: [u32; 24] =
    [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];
const PI: [usize; 24] =
    [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];

pub fn keccak_f(a: &mut [u64; 25]) {
    for &rc in RC.iter() {
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut last = a[1];
        for i in 0..24 {
            let j = PI[i];
            let tmp = a[j];
            a[j] = last.rotate_left(RHO[i]);
            last = tmp;
        }
        for y in 0..5 {
            let row: [u64; 5] = core::array::from_fn(|x| a[x + 5 * y]);
            for x in 0..5 {
                a[x + 5 * y] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
            }
        }
        a[0] ^= rc;
    }
}

#[derive(Clone)]
pub struct Shake128 {
    st: [u64; 25],
    pos: usize,
    squeezing: bool,
}

impl Default for Shake128 {
    fn default() -> Self {
        Self::new()
    }
}

impl Shake128 {
    pub fn new() -> Self {
        Shake128 { st: [0u64; 25], pos: 0, squeezing: false }
    }

    #[inline(always)]
    pub fn absorb_u32(&mut self, v: u32) {
        if self.squeezing {
            self.reabsorb();
        }
        let i = self.pos;
        self.st[i >> 3] ^= (v as u64) << (8 * (i & 7));
        self.pos = i + 4;
        if self.pos == RATE {
            keccak_f(&mut self.st);
            self.pos = 0;
        }
    }

    #[inline(always)]
    pub fn absorb_u64(&mut self, v: u64) {
        self.absorb_u32(v as u32);
        self.absorb_u32((v >> 32) as u32);
    }

    pub fn absorb_bytes(&mut self, data: &[u8]) {
        let mut it = data.chunks(4);
        for ch in &mut it {
            let mut w = [0u8; 4];
            w[..ch.len()].copy_from_slice(ch);
            self.absorb_u32(u32::from_le_bytes(w));
        }
    }

    fn finish_absorb(&mut self) {
        let i = self.pos;
        self.st[i >> 3] ^= 0x1Fu64 << (8 * (i & 7));
        self.st[(RATE - 1) >> 3] ^= 0x80u64 << (8 * ((RATE - 1) & 7));
        keccak_f(&mut self.st);
        self.pos = 0;
        self.squeezing = true;
    }

    fn reabsorb(&mut self) {
        keccak_f(&mut self.st);
        self.pos = 0;
        self.squeezing = false;
    }

    #[inline]
    pub fn squeeze_u32(&mut self) -> u32 {
        if !self.squeezing {
            self.finish_absorb();
        }
        if self.pos == RATE {
            keccak_f(&mut self.st);
            self.pos = 0;
        }
        let i = self.pos;
        let v = (self.st[i >> 3] >> (8 * (i & 7))) as u32;
        self.pos = i + 4;
        v
    }

    pub fn squeeze(&mut self, out: &mut [u8]) {
        for ch in out.chunks_mut(4) {
            let w = self.squeeze_u32().to_le_bytes();
            ch.copy_from_slice(&w[..ch.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shake128(msg: &[u8], n: usize) -> Vec<u8> {
        let mut s = Shake128::new();
        s.absorb_bytes(msg);
        let mut out = vec![0u8; n];
        s.squeeze(&mut out);
        out
    }

    #[test]
    fn keccak_matches_official_vectors() {
        let mut st = [0u64; 25];
        keccak_f(&mut st);
        assert_eq!(st[0], 0xF1258F7940E1DDE7, "keccak_f lane 0 on the all-zero state");

        assert_eq!(
            hex(&shake128(b"", 32)),
            "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
        );

        let mut st = [0u64; 25];
        st[0] ^= 0x06;
        st[(136 - 1) / 8] ^= 0x80u64 << (8 * ((136 - 1) % 8));
        keccak_f(&mut st);
        let out: Vec<u8> = st[..4].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(
            hex(&out),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
    }

    #[test]
    fn shake128_long_output_is_prefix_stable() {
        let short = shake128(b"", 32);
        let long = shake128(b"", 400);
        assert_eq!(&long[..32], &short[..]);
        assert_ne!(&long[..168], &long[168..336]);
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn keccak_f_is_not_identity_and_diffuses() {
        let mut a = [0u64; 25];
        keccak_f(&mut a);
        assert_ne!(a, [0u64; 25]);
        let mut b = [0u64; 25];
        b[0] = 1;
        keccak_f(&mut b);
        assert_ne!(a, b);
        let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert!(diff >= 20, "insufficient avalanche: only {diff}/25 lanes changed");
    }

    #[test]
    fn absorb_is_streaming_consistent() {
        let msg: Vec<u8> = (0..1024u32).flat_map(|i| i.to_le_bytes()).collect();
        let one = shake128(&msg, 64);
        let mut s = Shake128::new();
        for ch in msg.chunks(4) {
            s.absorb_u32(u32::from_le_bytes(ch.try_into().unwrap()));
        }
        let mut two = vec![0u8; 64];
        s.squeeze(&mut two);
        assert_eq!(one, two);
    }
}
