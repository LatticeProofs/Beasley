use crate::ntt::{to_spectra, Spectra};
use crate::ring::{RingElem, GADGET_LEN, N};
use crate::aesprg::AesPrg;
use rayon::prelude::*;

#[cfg(feature = "q32")]
pub const ELL: usize = 3;
#[cfg(feature = "q64")]
pub const ELL: usize = 5;

pub const fn ml_of(ell: usize) -> usize {
    ell * GADGET_LEN
}

pub struct HashParams {
    pub n_bits: usize,
    pub group_bits: usize,
    pub ell: usize,
    pub table: Vec<Vec<RingElem>>,
    spectra_cache: Option<Vec<Vec<Spectra>>>,
    pub crs_digest: [u8; 32],
}

impl HashParams {
    pub fn sample(seed: u64, n_bits: usize, group_bits: usize, ell: usize) -> Self {
        assert!(group_bits >= 1, "g must be at least 1");
        assert!(n_bits % group_bits == 0, "g must divide n_bits");
        assert!(n_bits / group_bits >= 2, "at least two groups are required");
        assert!(ell >= 1, "module rank must be at least 1");
        let tsz = 1usize << group_bits;
        let dims = crs_dims(n_bits, group_bits, ell, tsz);
        let cells = ell * ml_of(ell);

        let table: Vec<Vec<RingElem>> = (0..tsz)
            .into_par_iter()
            .map(|v| {
                let mut rng = AesPrg::from_parts(
                    "voprf-crs-table-v3-aesctr",
                    &[&seed.to_le_bytes(), &dims, &v.to_le_bytes()],
                );
                (0..cells)
                    .map(|_| RingElem { c: (0..N).map(|_| rng.next_fq()).collect() })
                    .collect()
            })
            .collect();

        let crs_digest = crs_digest(seed, n_bits, group_bits, ell);
        let mut p = HashParams { n_bits, group_bits, ell, table, spectra_cache: None, crs_digest };
        if std::env::var_os("VOPRF_CACHE_SPECTRA").is_some() {
            p.precompute_spectra();
        }
        p
    }

    pub fn num_groups(&self) -> usize {
        self.n_bits / self.group_bits
    }

    pub fn table_size(&self) -> usize {
        self.table.len()
    }

    pub fn ml(&self) -> usize {
        ml_of(self.ell)
    }

    pub fn a(&self, v: usize, r: usize, d: usize) -> &RingElem {
        debug_assert!(r < self.ell && d < self.ml());
        &self.table[v][r * self.ml() + d]
    }

    pub fn precompute_spectra(&mut self) {
        if self.spectra_cache.is_none() {
            self.spectra_cache =
                Some(self.table.par_iter().map(|m| m.iter().map(|e| to_spectra(&e.c)).collect()).collect());
        }
    }

    pub fn spectra_cached(&self) -> bool {
        self.spectra_cache.is_some()
    }

    pub fn spectra_get(&self, v: usize) -> std::borrow::Cow<'_, [Spectra]> {
        match &self.spectra_cache {
            Some(c) => std::borrow::Cow::Borrowed(&c[v]),
            None => std::borrow::Cow::Owned(self.spectra_for(v)),
        }
    }

    pub fn entry(&self, v: usize) -> &[RingElem] {
        &self.table[v]
    }

    pub fn spectra_for(&self, v: usize) -> Vec<Spectra> {
        self.table[v].iter().map(|e| to_spectra(&e.c)).collect()
    }
}

fn crs_dims(n_bits: usize, group_bits: usize, ell: usize, tsz: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(48);
    for v in [n_bits as u64, group_bits as u64, ell as u64, tsz as u64, GADGET_LEN as u64, N as u64] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn crs_digest(seed: u64, n_bits: usize, group_bits: usize, ell: usize) -> [u8; 32] {
    let mut tr = crate::transcript::Transcript::new("voprf-crs-v4-seed");
    tr.absorb_u64(seed);
    tr.absorb_u64(n_bits as u64);
    tr.absorb_u64(group_bits as u64);
    tr.absorb_u64(ell as u64);
    tr.absorb_u64(ml_of(ell) as u64);
    tr.absorb_u64(GADGET_LEN as u64);
    tr.absorb_u64(N as u64);
    tr.absorb_u64((1usize << group_bits) as u64);
    tr.finalize_digest()
}

pub fn crs_digest_from_table(
    n_bits: usize,
    group_bits: usize,
    ell: usize,
    table: &[Vec<RingElem>],
) -> [u8; 32] {
    let mut tr = crate::transcript::Transcript::new("voprf-crs-v3-module-rank");
    tr.absorb_u64(n_bits as u64);
    tr.absorb_u64(group_bits as u64);
    tr.absorb_u64(ell as u64);
    tr.absorb_u64(ml_of(ell) as u64);
    tr.absorb_u64(GADGET_LEN as u64);
    tr.absorb_u64(N as u64);
    tr.absorb_u64(table.len() as u64);
    for row in table {
        for e in row {
            tr.absorb_fqs(&e.c);
        }
    }
    tr.finalize_digest()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{gadget_decompose, gadget_recompose};

    #[test]
    fn table_rows_are_independent() {
        let p = HashParams::sample(1, 8, 4, 2);
        assert_eq!(p.table_size(), 16);
        assert_eq!(p.ml(), 2 * GADGET_LEN);
        assert!(p.table.iter().all(|r| r.len() == p.ell * p.ml()));
        for a in 0..p.table_size() {
            for b in (a + 1)..p.table_size() {
                assert_ne!(p.table[a], p.table[b], "matrix {a} and matrix {b} are identical");
            }
        }
        let half = 4usize;
        let (w, wp) = (2usize, 3usize);
        let ml = p.ml();
        let mut combined: Vec<RingElem> = Vec::with_capacity(p.ell * ml);
        for r in 0..p.ell {
            for col in 0..ml {
                let mut acc = RingElem::zero();
                for t in 0..p.ell {
                    let dg = gadget_decompose(p.a(wp, t, col));
                    for c in 0..GADGET_LEN {
                        acc = &acc + &(p.a(w, r, t * GADGET_LEN + c) * &dg[c]);
                    }
                }
                combined.push(acc);
            }
        }
        assert_ne!(p.table[w * half + wp], combined, "the table still looks combined (e(T) would go back to 2)");
    }

    #[test]
    fn sampling_is_deterministic_and_domain_separated() {
        let a = HashParams::sample(7, 8, 4, 1);
        let b = HashParams::sample(7, 8, 4, 1);
        assert_eq!(a.table, b.table);
        assert_eq!(a.crs_digest, b.crs_digest);
        let c = HashParams::sample(8, 8, 4, 1);
        assert_ne!(a.table, c.table);
        assert_ne!(a.crs_digest, c.crs_digest);
        let d = HashParams::sample(7, 8, 4, 2);
        assert_ne!(a.crs_digest, d.crs_digest);
        assert_ne!(a.table, d.table, "ell is part of the sampling domain too");
    }

    #[test]
    fn seed_determines_the_table() {
        let a = HashParams::sample(2024, 8, 4, 2);
        let b = HashParams::sample(2024, 8, 4, 2);
        let dt = |p: &HashParams| crs_digest_from_table(p.n_bits, p.group_bits, p.ell, &p.table);

        assert_eq!(a.crs_digest, b.crs_digest, "digests for the same seed unexpectedly differ");
        assert_eq!(a.table, b.table, "the same seed unexpectedly produced different tables -- the premise of the seed-based digest is broken");
        assert_eq!(dt(&a), dt(&b), "the table-based digest must agree too");

        let c = HashParams::sample(2025, 8, 4, 2);
        assert_ne!(a.crs_digest, c.crs_digest, "digests for different seeds are unexpectedly identical");
        assert_ne!(dt(&a), dt(&c), "the table-based digest failed to distinguish CRSs from different seeds");
    }

    #[test]
    fn table_coefficients_are_uniform_over_range() {
        let p = HashParams::sample(11, 8, 2, 1);
        let (mut hi, mut total) = (0usize, 0usize);
        for row in &p.table {
            for e in row {
                for c in &e.c {
                    assert!((c.0 as u64) < crate::field::Q, "out of range [0,q)");
                    if (c.0 as u64) >= crate::field::Q / 2 {
                        hi += 1;
                    }
                    total += 1;
                }
            }
        }
        assert!(hi > total * 45 / 100 && hi < total * 55 / 100, "upper-half ratio {hi}/{total}");
    }

    #[test]
    fn gadget_roundtrip_on_table() {
        let p = HashParams::sample(3, 8, 2, 2);
        for v in 0..p.table_size() {
            for e in &p.table[v] {
                assert_eq!(gadget_recompose(&gadget_decompose(e)), *e);
            }
        }
    }
}
