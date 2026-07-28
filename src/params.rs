use crate::ntt::{neg_inner_product, to_spectra, Spectra};
use crate::ring::{gadget_decompose, RingElem, DELTA, N};
use crate::transcript::SimpleRng;
use rayon::prelude::*;

pub struct HashParams {
    pub n_bits: usize,
    pub group_bits: usize,
    pub ell: usize,
    pub a0: Vec<RingElem>,
    pub a1: Vec<RingElem>,
    pub table: Vec<Vec<RingElem>>,
    pub crs_digest: [u8; 32],
}

impl HashParams {
    pub fn sample(seed: u64, n_bits: usize, group_bits: usize, ell: usize) -> Self {
        assert!(group_bits >= 1 && group_bits.is_power_of_two(), "g 必須是 2 的冪");
        assert!(n_bits % group_bits == 0, "g 必須整除 n_bits");
        assert!(n_bits / group_bits >= 2, "至少要兩群");
        assert!(1 <= ell && ell <= DELTA);
        let mut rng = SimpleRng::new(seed);
        let mut sample_vec = || {
            (0..DELTA)
                .map(|_| RingElem { c: (0..N).map(|_| rng.next_fq()).collect() })
                .collect::<Vec<_>>()
        };
        let a0 = sample_vec();
        let a1 = sample_vec();
        let table = precompute_table(&a0, &a1, group_bits);
        let crs_digest = crs_digest(n_bits, group_bits, ell, &a0, &a1);
        HashParams { n_bits, group_bits, ell, a0, a1, table, crs_digest }
    }

    pub fn num_groups(&self) -> usize {
        self.n_bits / self.group_bits
    }

    pub fn table_size(&self) -> usize {
        1 << self.group_bits
    }

    pub fn entry(&self, v: usize) -> &[RingElem] {
        &self.table[v]
    }

    pub fn spectra_for(&self, v: usize) -> Vec<Spectra> {
        self.table[v].iter().map(|e| to_spectra(&e.c)).collect()
    }
}

pub fn crs_digest(
    n_bits: usize,
    group_bits: usize,
    ell: usize,
    a0: &[RingElem],
    a1: &[RingElem],
) -> [u8; 32] {
    let mut tr = crate::transcript::Transcript::new("voprf-crs-v1");
    tr.absorb_u64(n_bits as u64);
    tr.absorb_u64(group_bits as u64);
    tr.absorb_u64(ell as u64);
    tr.absorb_u64(DELTA as u64);
    tr.absorb_u64(N as u64);
    for a in a0.iter().chain(a1) {
        tr.absorb_fqs(&a.c);
    }
    tr.finalize_digest()
}

pub fn precompute_table(a0: &[RingElem], a1: &[RingElem], group_bits: usize) -> Vec<Vec<RingElem>> {
    let mut cur: Vec<Vec<RingElem>> = vec![a0.to_vec(), a1.to_vec()];
    let mut leaves = 1usize;
    while leaves < group_bits {
        let half = cur.len();
        let specs: Vec<Vec<Spectra>> =
            cur.iter().map(|p| p.iter().map(|e| to_spectra(&e.c)).collect()).collect();
        let digits: Vec<Vec<Vec<RingElem>>> =
            cur.iter().map(|q| q.iter().map(gadget_decompose).collect()).collect();

        let next: Vec<Vec<RingElem>> = (0..half * half)
            .into_par_iter()
            .map(|v| {
                let (w, wp) = (v / half, v % half);
                (0..DELTA).map(|col| neg_inner_product(&specs[w], &digits[wp][col])).collect()
            })
            .collect();
        cur = next;
        leaves *= 2;
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::gadget_recompose;

    fn direct(a: &[Vec<RingElem>], v: usize, bits: usize) -> Vec<RingElem> {
        if bits == 1 {
            return a[v].clone();
        }
        let half = bits / 2;
        let (w, wp) = (v >> half, v & ((1 << half) - 1));
        let left = direct(a, w, half);
        let right = direct(a, wp, half);
        (0..DELTA)
            .map(|col| {
                let m = gadget_decompose(&right[col]);
                let mut acc = RingElem::zero();
                for d in 0..DELTA {
                    acc = &acc + &(&left[d] * &m[d]);
                }
                acc
            })
            .collect()
    }

    #[test]
    fn table_g1_is_a0_a1() {
        let p = HashParams::sample(1, 8, 1, 1);
        assert_eq!(p.table_size(), 2);
        assert_eq!(p.table[0], p.a0);
        assert_eq!(p.table[1], p.a1);
    }

    #[test]
    fn table_matches_direct_recursion() {
        let p = HashParams::sample(2, 8, 4, 1);
        let leaves = vec![p.a0.clone(), p.a1.clone()];
        assert_eq!(p.table.len(), 16);
        for v in [0usize, 5, 11, 15] {
            assert_eq!(p.table[v], direct(&leaves, v, 4), "v = {v}");
        }
    }

    #[test]
    fn gadget_roundtrip_on_table() {
        let p = HashParams::sample(3, 8, 2, 1);
        for v in 0..p.table_size() {
            for d in 0..DELTA {
                assert_eq!(gadget_recompose(&gadget_decompose(&p.table[v][d])), p.table[v][d]);
            }
        }
    }
}
