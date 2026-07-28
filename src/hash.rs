use crate::field::Fq;
use crate::ntt::neg_and_quotient;
use crate::params::HashParams;
use crate::ring::{gadget_decompose, RingElem, DELTA};
use rayon::prelude::*;

pub struct HashWitness {
    pub m: Vec<Vec<Vec<RingElem>>>,
    pub t: Vec<Vec<Vec<Fq>>>,
}

impl HashWitness {
    pub fn column(&self, j: usize, i: usize) -> &[RingElem] {
        &self.m[j][i - 2]
    }
    pub fn quotient(&self, j: usize, i: usize) -> &[Fq] {
        &self.t[j][i - 2]
    }
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

    let specs: Vec<_> = groups.par_iter().map(|&v| params.spectra_for(v)).collect();

    let mut outputs = Vec::with_capacity(params.ell);
    let mut m_all = Vec::with_capacity(params.ell);
    let mut t_all = Vec::with_capacity(params.ell);

    for j in 0..params.ell {
        let mut w = params.table[groups[ng - 1]][j].clone();
        let mut m_desc: Vec<Vec<RingElem>> = Vec::with_capacity(ng - 1);
        let mut t_desc: Vec<Vec<Fq>> = Vec::with_capacity(ng - 1);
        for i in (1..=ng - 1).rev() {
            let col = gadget_decompose(&w);
            let (next, t) = neg_and_quotient(&specs[i - 1], &col);
            m_desc.push(col);
            t_desc.push(t);
            w = next;
        }
        m_desc.reverse();
        t_desc.reverse();
        outputs.push(w);
        m_all.push(m_desc);
        t_all.push(t_desc);
    }

    (outputs, HashWitness { m: m_all, t: t_all })
}

pub fn eval_h_naive(params: &HashParams, groups: &[usize]) -> Vec<RingElem> {
    let ng = params.num_groups();
    assert_eq!(groups.len(), ng);

    let mut w: Vec<RingElem> = params.table[groups[ng - 1]].clone();
    for i in (1..=ng - 1).rev() {
        let a = &params.table[groups[i - 1]];
        let mut next = vec![RingElem::zero(); DELTA];
        for col in 0..DELTA {
            let m_col = gadget_decompose(&w[col]);
            for d in 0..DELTA {
                next[col] = &next[col] + &(&a[d] * &m_col[d]);
            }
        }
        w = next;
    }

    w.truncate(params.ell);
    w
}
