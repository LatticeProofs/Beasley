use crate::field::Fq;
use crate::ntt::neg_and_quotient_rows;
use crate::params::HashParams;
use crate::ring::{gadget_decompose, RingElem};
use rayon::prelude::*;

pub struct HashWitness {
    pub m: Vec<Vec<RingElem>>,
    pub t: Vec<Vec<Vec<Fq>>>,
}

impl HashWitness {
    pub fn column(&self, i: usize) -> &[RingElem] {
        &self.m[i - 2]
    }
    pub fn quotient(&self, r: usize, i: usize) -> &[Fq] {
        &self.t[i - 2][r]
    }
}

pub fn gadget_decompose_vec(y: &[RingElem]) -> Vec<RingElem> {
    y.iter().flat_map(|e| gadget_decompose(e)).collect()
}

pub fn bits_to_groups(params: &HashParams, bits: &[bool]) -> Vec<usize> {
    assert_eq!(bits.len(), params.n_bits);
    bits.chunks(params.group_bits)
        .map(|ch| ch.iter().fold(0usize, |acc, &b| (acc << 1) | b as usize))
        .collect()
}

pub fn groups_to_bits(params: &HashParams, groups: &[usize]) -> Vec<bool> {
    let g = params.group_bits;
    let mut out = Vec::with_capacity(params.n_bits);
    for &v in groups {
        for b in (0..g).rev() {
            out.push((v >> b) & 1 == 1);
        }
    }
    out
}

pub fn eval_h(params: &HashParams, groups: &[usize]) -> (Vec<RingElem>, HashWitness) {
    let ng = params.num_groups();
    assert_eq!(groups.len(), ng);
    assert!(groups.iter().all(|&v| v < params.table_size()));

    let specs: Vec<_> = groups.par_iter().map(|&v| params.spectra_get(v)).collect();

    let mut y: Vec<RingElem> =
        (0..params.ell).map(|r| params.a(groups[ng - 1], r, 0).clone()).collect();

    let mut m_desc: Vec<Vec<RingElem>> = Vec::with_capacity(ng - 1);
    let mut t_desc: Vec<Vec<Vec<Fq>>> = Vec::with_capacity(ng - 1);
    for i in (1..=ng - 1).rev() {
        let md = gadget_decompose_vec(&y);
        let (next, ts): (Vec<RingElem>, Vec<Vec<Fq>>) =
            neg_and_quotient_rows(&specs[i - 1], params.ell, &md).into_iter().unzip();
        m_desc.push(md);
        t_desc.push(ts);
        y = next;
    }
    m_desc.reverse();
    t_desc.reverse();

    (y, HashWitness { m: m_desc, t: t_desc })
}

pub fn eval_h_naive(params: &HashParams, groups: &[usize]) -> Vec<RingElem> {
    let ng = params.num_groups();
    assert_eq!(groups.len(), ng);
    let (m, ml) = (params.ell, params.ml());

    let mut z: Vec<RingElem> = params.table[groups[ng - 1]].clone();
    for i in (1..=ng - 1).rev() {
        let v = groups[i - 1];
        let gi: Vec<Vec<RingElem>> = (0..ml)
            .map(|col| gadget_decompose_vec(&(0..m).map(|t| z[t * ml + col].clone()).collect::<Vec<_>>()))
            .collect();
        let mut next = vec![RingElem::zero(); m * ml];
        for r in 0..m {
            for col in 0..ml {
                let mut acc = RingElem::zero();
                for d in 0..ml {
                    acc = &acc + &(params.a(v, r, d) * &gi[col][d]);
                }
                next[r * ml + col] = acc;
            }
        }
        z = next;
    }

    (0..m).map(|r| z[r * ml].clone()).collect()
}
