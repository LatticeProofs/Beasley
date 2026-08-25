use crate::ntt::{to_spectra, Spectra};
use crate::ring::{RingElem, GADGET_LEN, N};
use crate::aesprg::AesPrg;
use rayon::prelude::*;

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
        assert!(group_bits >= 1, "g must be >= 1");
        assert!(n_bits % group_bits == 0, "g must divide n_bits");
        assert!(n_bits / group_bits >= 2, "need at least two groups");
        assert!(ell >= 1, "module rank must be >= 1");
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

    #[cfg(test)]
    pub fn dims_only(n_bits: usize, group_bits: usize, ell: usize) -> Self {
        assert!(n_bits % group_bits == 0 && n_bits / group_bits >= 2 && ell >= 1);
        HashParams {
            n_bits,
            group_bits,
            ell,
            table: vec![Vec::new(); 1usize << group_bits],
            spectra_cache: None,
            crs_digest: crs_digest(0, n_bits, group_bits, ell),
        }
    }

    pub fn precompute_spectra(&mut self) {
        if self.spectra_cache.is_none() {
            self.spectra_cache =
                Some(self.table.par_iter().map(|m| m.iter().map(|e| to_spectra(&e.c)).collect()).collect());
        }
    }

    pub fn spectra_get(&self, v: usize) -> std::borrow::Cow<'_, [Spectra]> {
        match &self.spectra_cache {
            Some(c) => std::borrow::Cow::Borrowed(&c[v]),
            None => std::borrow::Cow::Owned(self.spectra_for(v)),
        }
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
        assert_ne!(a.table, d.table, "ell is part of the sampling domain");
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
