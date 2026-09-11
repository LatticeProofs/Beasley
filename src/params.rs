
use crate::aesprg::AesPrg;
use crate::field::Fq;
use crate::ntt::{to_spectra_bin, Spectra};
use crate::ring::{RingElem, N};
use rayon::prelude::*;

pub const ELL: usize = 5;

pub struct HashParams {
    pub n_bits: usize,
    pub group_bits: usize,
    pub ell: usize,
    pub table: Vec<Vec<Vec<RingElem>>>,
    spectra_cache: Option<Vec<Vec<Vec<Spectra>>>>,
    pub crs_digest: [u8; 32],
}

impl HashParams {
    pub fn sample(seed: u64, n_bits: usize, group_bits: usize, ell: usize) -> Self {
        assert!(group_bits >= 1, "g must be at least 1");
        assert!(n_bits % group_bits == 0, "g must divide n_bits");
        assert!(n_bits / group_bits >= 2, "at least two blocks are required");
        assert!(ell >= 1, "module rank must be at least 1");
        let tsz = 1usize << group_bits;
        let ng = n_bits / group_bits;
        let dims = crs_dims(n_bits, group_bits, ell, tsz);
        let flat: Vec<Vec<RingElem>> = (0..ng * tsz)
            .into_par_iter()
            .map(|idx| {
                let (i, v) = (idx / tsz, idx % tsz);
                let mut rng = AesPrg::from_parts(
                    "blmr-crs-table-v2-per-block-binary-transposed",
                    &[&seed.to_le_bytes(), &dims, &i.to_le_bytes(), &v.to_le_bytes()],
                );
                sample_symbol(&mut || rng.next_bool(), ell)
                    .unwrap_or_else(|| panic!(
                        "block {i} symbol {v} drew a non-invertible matrix {MAX_RESAMPLE} times in a row -- practically impossible, check the PRG"
                    ))
            })
            .collect();
        let mut flat = flat.into_iter();
        let table: Vec<Vec<Vec<RingElem>>> =
            (0..ng).map(|_| flat.by_ref().take(tsz).collect()).collect();

        let crs_digest = crs_digest(seed, n_bits, group_bits, ell);
        HashParams { n_bits, group_bits, ell, table, spectra_cache: None, crs_digest }
    }

    pub fn num_groups(&self) -> usize {
        self.n_bits / self.group_bits
    }

    pub fn table_size(&self) -> usize {
        1usize << self.group_bits
    }

    pub fn num_matrices(&self) -> usize {
        self.num_groups() * self.table_size()
    }

    pub fn a(&self, i: usize, v: usize, r: usize, t: usize) -> &RingElem {
        debug_assert!(r < self.ell && t < self.ell);
        &self.table[i][v][r * self.ell + t]
    }

    #[cfg(test)]
    pub fn dims_only(n_bits: usize, group_bits: usize, ell: usize) -> Self {
        assert!(n_bits % group_bits == 0 && n_bits / group_bits >= 2 && ell >= 1);
        HashParams {
            n_bits,
            group_bits,
            ell,
            table: vec![vec![Vec::new(); 1usize << group_bits]; n_bits / group_bits],
            spectra_cache: None,
            crs_digest: crs_digest(0, n_bits, group_bits, ell),
        }
    }

    pub fn precompute_spectra(&mut self) {
        if self.spectra_cache.is_none() {
            self.spectra_cache = Some(
                self.table
                    .par_iter()
                    .map(|blk| {
                        blk.iter()
                            .map(|m| m.iter().map(|e| to_spectra_bin(&e.c)).collect())
                            .collect()
                    })
                    .collect(),
            );
        }
    }

    pub fn spectra_get(&self, i: usize, v: usize) -> std::borrow::Cow<'_, [Spectra]> {
        match &self.spectra_cache {
            Some(c) => std::borrow::Cow::Borrowed(&c[i][v]),
            None => std::borrow::Cow::Owned(self.spectra_for(i, v)),
        }
    }

    pub fn spectra_for(&self, i: usize, v: usize) -> Vec<Spectra> {
        self.table[i][v].iter().map(|e| to_spectra_bin(&e.c)).collect()
    }
}

const MAX_RESAMPLE: usize = 64;

fn sample_symbol(next_bool: &mut impl FnMut() -> bool, ell: usize) -> Option<Vec<RingElem>> {
    let cells = ell * ell;
    for _ in 0..MAX_RESAMPLE {
        let a: Vec<RingElem> = (0..cells)
            .map(|_| RingElem {
                c: (0..N).map(|_| if next_bool() { Fq::ONE } else { Fq::ZERO }).collect(),
            })
            .collect();
        if !crate::polygcd::is_invertible(&a, ell) {
            continue;
        }
        return Some(
            (0..cells)
                .map(|idx| {
                    let (r, t) = (idx / ell, idx % ell);
                    a[t * ell + r].clone()
                })
                .collect(),
        );
    }
    None
}

fn crs_dims(n_bits: usize, group_bits: usize, ell: usize, tsz: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(48);
    let ng = n_bits / group_bits;
    for v in [n_bits as u64, group_bits as u64, ell as u64, tsz as u64, ng as u64, N as u64] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn crs_digest(seed: u64, n_bits: usize, group_bits: usize, ell: usize) -> [u8; 32] {
    let mut tr = crate::transcript::Transcript::new("blmr-crs-v2-per-block-seed");
    tr.absorb_u64(seed);
    tr.absorb_u64(n_bits as u64);
    tr.absorb_u64(group_bits as u64);
    tr.absorb_u64(ell as u64);
    tr.absorb_u64(N as u64);
    tr.absorb_u64((1usize << group_bits) as u64);
    tr.absorb_u64((n_bits / group_bits) as u64);
    tr.finalize_digest()
}

pub fn crs_digest_from_table(
    n_bits: usize,
    group_bits: usize,
    ell: usize,
    table: &[Vec<Vec<RingElem>>],
) -> [u8; 32] {
    let mut tr = crate::transcript::Transcript::new("blmr-crs-v2-per-block-table");
    tr.absorb_u64(n_bits as u64);
    tr.absorb_u64(group_bits as u64);
    tr.absorb_u64(ell as u64);
    tr.absorb_u64(N as u64);
    tr.absorb_u64(table.len() as u64);
    for blk in table {
        tr.absorb_u64(blk.len() as u64);
        for mat in blk {
            for e in mat {
                tr.absorb_fqs(&e.c);
            }
        }
    }
    tr.finalize_digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_binary_and_invertible() {
        let p = HashParams::sample(1, 8, 4, 3);
        assert_eq!(p.table_size(), 16);
        assert_eq!(p.table.len(), p.num_groups(), "one set per block");
        assert_eq!(p.num_matrices(), 2 * 16);
        for (i, blk) in p.table.iter().enumerate() {
            assert_eq!(blk.len(), p.table_size(), "symbol count of block {i}");
            assert!(blk.iter().all(|r| r.len() == p.ell * p.ell), "the shape must be m×m");
            for (v, mat) in blk.iter().enumerate() {
                assert!(mat.iter().all(|e| e.is_binary()), "the coefficients of block {i} symbol {v} are not bits");
                assert!(crate::polygcd::is_invertible(mat, p.ell), "block {i} symbol {v} is not invertible");
            }
        }
    }

    #[test]
    fn table_stores_the_transpose() {
        let p = HashParams::sample(2, 8, 2, 4);
        let m = p.ell;
        for i in 0..p.num_groups() {
            for v in 0..p.table_size() {
                let a_orig: Vec<RingElem> =
                    (0..m * m).map(|idx| p.a(i, v, idx % m, idx / m).clone()).collect();
                for j in 0..m {
                    assert_eq!(&a_orig[0 * m + j], p.a(i, v, j, 0), "b_0[{j}] is derived incorrectly");
                }
            }
        }
    }

    #[test]
    fn sampling_is_deterministic_and_domain_separated() {
        let a = HashParams::sample(7, 8, 4, 2);
        let b = HashParams::sample(7, 8, 4, 2);
        assert_eq!(a.table, b.table);
        assert_eq!(a.crs_digest, b.crs_digest);

        let c = HashParams::sample(8, 8, 4, 2);
        assert_ne!(a.table, c.table);
        assert_ne!(a.crs_digest, c.crs_digest);

        let d = HashParams::sample(7, 8, 4, 3);
        assert_ne!(a.crs_digest, d.crs_digest);
    }

    #[test]
    fn seed_determines_the_table() {
        let a = HashParams::sample(2024, 8, 4, 2);
        let b = HashParams::sample(2024, 8, 4, 2);
        let dt = |p: &HashParams| crs_digest_from_table(p.n_bits, p.group_bits, p.ell, &p.table);

        assert_eq!(a.crs_digest, b.crs_digest);
        assert_eq!(a.table, b.table, "the same seed unexpectedly produced a different table -- the premise of the seed-based digest is broken");
        assert_eq!(dt(&a), dt(&b));

        let c = HashParams::sample(2025, 8, 4, 2);
        assert_ne!(a.crs_digest, c.crs_digest);
        assert_ne!(dt(&a), dt(&c), "the table-based digest failed to separate CRSs from different seeds");
    }

    #[test]
    fn symbols_are_independent_within_and_across_blocks() {
        let p = HashParams::sample(3, 12, 4, 2);
        assert_eq!(p.num_groups(), 3);
        let all: Vec<(usize, usize)> =
            (0..p.num_groups()).flat_map(|i| (0..p.table_size()).map(move |v| (i, v))).collect();
        for (x, &(i, v)) in all.iter().enumerate() {
            for &(i2, v2) in &all[x + 1..] {
                assert_ne!(p.table[i][v], p.table[i2][v2], "({i},{v}) is identical to ({i2},{v2})");
            }
        }
    }

    #[test]
    fn bits_are_balanced() {
        let p = HashParams::sample(11, 8, 2, 2);
        let (mut ones, mut total) = (0usize, 0usize);
        for mat in p.table.iter().flatten() {
            for e in mat {
                for c in &e.c {
                    ones += c.0 as usize;
                    total += 1;
                }
            }
        }
        assert!(ones > total * 45 / 100 && ones < total * 55 / 100, "ratio of ones {ones}/{total}");
    }

    #[test]
    fn spectra_matches_general_path() {
        let p = HashParams::sample(5, 8, 2, 2);
        let s = p.spectra_for(1, 0);
        for (i, e) in p.table[1][0].iter().enumerate() {
            let want = crate::ntt::to_spectra(&e.c);
            for pi in 0..crate::ntt::NHOT {
                assert_eq!(s[i].fwd[pi], want.fwd[pi], "element {i} prime {pi}");
            }
        }
    }

    #[test]
    fn resampling_actually_retries() {
        for ell in [1usize, 2, 3] {
            let cells = ell * ell;
            let mut rng = crate::transcript::SimpleRng::new(0xBADC0DE ^ ell as u64);
            let next = |used: &mut usize, rng: &mut crate::transcript::SimpleRng| {
                *used += 1;
                if *used <= cells * N {
                    false
                } else {
                    rng.next_bool()
                }
            };
            let mut consumed = 0usize;
            let a = sample_symbol(&mut || next(&mut consumed, &mut rng), ell)
                .expect("the second round should draw an invertible matrix");
            assert!(consumed >= 2 * cells * N, "ell={ell}: no resampling happened (consumed={consumed})");
            assert_eq!(consumed % (cells * N), 0, "each round consumes exactly m²·N bits");
            assert!(crate::polygcd::is_invertible(&a, ell), "ell={ell}: returned a non-invertible matrix");
            assert!(a.iter().all(|e| e.is_binary()), "ell={ell}: returned a non-binary matrix");
        }
    }

    #[test]
    fn resampling_gives_up_after_the_cap() {
        assert!(sample_symbol(&mut || false, 2).is_none(), "an all-zero PRG unexpectedly drew an invertible matrix");
    }

}
