
pub struct PackedBits {
    words: Vec<u64>,
    len: usize,
}

impl PackedBits {
    pub fn from_bits(bits: &[u8], len: usize) -> Self {
        assert_eq!(bits.len(), len);
        let mut words = vec![0u64; (len + 63) / 64];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                words[i >> 6] |= 1u64 << (i & 63);
            }
        }
        PackedBits { words, len }
    }

    pub fn zeros(len: usize) -> Self {
        PackedBits { words: vec![0u64; (len + 63) / 64], len }
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize) {
        self.words[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline(always)]
    pub fn or_bit(&mut self, i: usize, v: u32) {
        self.words[i >> 6] |= (v as u64) << (i & 63);
    }

    #[inline(always)]
    pub fn flip(&mut self, i: usize) {
        self.words[i >> 6] ^= 1u64 << (i & 63);
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> bool {
        (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }

    #[inline(always)]
    pub fn word(&self, w: usize) -> u64 {
        self.words[w]
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn words(&self) -> &[u64] {
        &self.words
    }
}
