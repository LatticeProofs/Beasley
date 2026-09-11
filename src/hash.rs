
use crate::field::Fq;
use crate::ntt::neg_and_quotient_rows_wide;
use crate::params::HashParams;
use crate::ring::RingElem;
use rayon::prelude::*;

pub struct HashWitness {
    pub b: Vec<Vec<RingElem>>,
    pub t: Vec<Vec<Vec<Fq>>>,
}

impl HashWitness {
    pub fn b(&self, i: usize) -> &[RingElem] {
        &self.b[i]
    }
    pub fn quotient(&self, i: usize, j: usize) -> &[Fq] {
        &self.t[i - 1][j]
    }
    pub fn steps(&self) -> usize {
        self.t.len()
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
    let m = params.ell;
    assert_eq!(groups.len(), ng);
    assert!(groups.iter().all(|&v| v < params.table_size()));

    let specs: Vec<_> =
        groups.par_iter().enumerate().map(|(i, &v)| params.spectra_get(i, v)).collect();

    let mut cur: Vec<RingElem> = (0..m).map(|j| params.a(0, groups[0], j, 0).clone()).collect();

    let mut b_out: Vec<Vec<RingElem>> = Vec::with_capacity(ng - 1);
    let mut t_out: Vec<Vec<Vec<Fq>>> = Vec::with_capacity(ng - 1);

    for i in 1..ng {
        let (next, ts): (Vec<RingElem>, Vec<Vec<Fq>>) =
            neg_and_quotient_rows_wide(&specs[i], m, &cur).into_iter().unzip();
        b_out.push(std::mem::replace(&mut cur, next));
        t_out.push(ts);
    }

    debug_assert_eq!(b_out.len(), ng - 1);
    debug_assert_eq!(t_out.len(), ng - 1);
    (cur, HashWitness { b: b_out, t: t_out })
}

pub fn eval_h_naive(params: &HashParams, groups: &[usize]) -> Vec<RingElem> {
    let ng = params.num_groups();
    let m = params.ell;
    assert_eq!(groups.len(), ng);

    let orig = |i: usize, v: usize, r: usize, c: usize| params.a(i, v, c, r);
    let mut z: Vec<RingElem> =
        (0..m * m).map(|idx| orig(0, groups[0], idx / m, idx % m).clone()).collect();

    for (i, &v) in groups.iter().enumerate().skip(1) {
        let mut next = vec![RingElem::zero(); m * m];
        for r in 0..m {
            for c in 0..m {
                let mut acc = RingElem::zero();
                for k in 0..m {
                    acc = &acc + &(&z[r * m + k] * orig(i, v, k, c));
                }
                next[r * m + c] = acc;
            }
        }
        z = next;
    }
    (0..m).map(|c| z[c].clone()).collect()
}

pub fn check_witness(
    params: &HashParams,
    bx: &[RingElem],
    groups: &[usize],
    wit: &HashWitness,
) -> bool {
    let ng = params.num_groups();
    let m = params.ell;
    if bx.len() != m || groups.len() != ng || wit.b.len() != ng - 1 || wit.t.len() != ng - 1 {
        return false;
    }

    for j in 0..m {
        if wit.b(0)[j] != *params.a(0, groups[0], j, 0) {
            return false;
        }
    }
    for i in 1..ng {
        let prev = wit.b(i - 1);
        let expect: Vec<RingElem> = (0..m)
            .map(|j| {
                let mut acc = RingElem::zero();
                for t in 0..m {
                    acc = &acc + &(params.a(i, groups[i], j, t) * &prev[t]);
                }
                acc
            })
            .collect();
        let got: &[RingElem] = if i == ng - 1 { bx } else { wit.b(i) };
        if got != expect.as_slice() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext_field::FqExt;
    use crate::ring::{poly_eval, N};
    use crate::transcript::SimpleRng;

    fn setup(n: usize, g: usize, ell: usize, seed: u64) -> (HashParams, Vec<usize>) {
        let params = HashParams::sample(seed, n, g, ell);
        let mut rng = SimpleRng::new(seed ^ 0x5EED);
        let bits: Vec<bool> = (0..n).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        (params, groups)
    }

    #[test]
    fn column_trick_matches_full_matrix_product() {
        for &(n, g, ell, seed) in
            &[(8usize, 2usize, 2usize, 101u64), (8, 4, 3, 102), (12, 2, 5, 103), (8, 1, 4, 104)]
        {
            let (params, groups) = setup(n, g, ell, seed);
            let (bx, _) = eval_h(&params, &groups);
            assert_eq!(bx, eval_h_naive(&params, &groups), "n={n} g={g} ell={ell}");
        }
    }

    #[test]
    fn honest_witness_passes_direct_check() {
        let (params, groups) = setup(8, 2, 3, 201);
        let (bx, wit) = eval_h(&params, &groups);
        assert!(check_witness(&params, &bx, &groups, &wit));
        assert_eq!(wit.b.len(), params.num_groups() - 1);
        assert_eq!(wit.steps(), params.num_groups() - 1);
    }

    #[test]
    fn tampered_witness_fails_direct_check() {
        let (params, groups) = setup(8, 2, 2, 202);
        let (bx, mut wit) = eval_h(&params, &groups);
        wit.b[0][1].c[100] = wit.b[0][1].c[100] + Fq::ONE;
        assert!(!check_witness(&params, &bx, &groups, &wit));
    }

    #[test]
    fn quotients_satisfy_the_ring_switching_identity() {
        let (params, groups) = setup(12, 2, 3, 301);
        let ng = params.num_groups();
        let m = params.ell;
        let (bx, wit) = eval_h(&params, &groups);

        let mut rng = SimpleRng::new(0xA1);
        let alpha = rng.next_fq4();
        let xn1 = alpha.pow(N as u128) + FqExt::ONE;

        for i in 1..ng {
            let prev = wit.b(i - 1);
            let cur: &[RingElem] = if i == ng - 1 { &bx } else { wit.b(i) };
            for j in 0..m {
                let full_at = (0..m).fold(FqExt::ZERO, |acc, t| {
                    acc + params.a(i, groups[i], j, t).eval(alpha) * prev[t].eval(alpha)
                });
                let t_at = poly_eval(wit.quotient(i, j), alpha);
                assert_eq!(
                    cur[j].eval(alpha),
                    full_at + xn1 * t_at,
                    "quotient sign or index misaligned at step {i} row {j}"
                );
            }
            for j in 0..m {
                assert_eq!(wit.quotient(i, j).len(), N - 1);
            }
        }
    }

    #[test]
    fn different_inputs_give_different_outputs() {
        let params = HashParams::sample(401, 16, 4, 3);
        let ng = params.num_groups();
        let a = eval_h(&params, &vec![0usize; ng]).0;
        let mut g2 = vec![0usize; ng];
        g2[ng - 1] = 1;
        let b = eval_h(&params, &g2).0;
        assert_ne!(a, b);
        let mut g3 = vec![0usize; ng];
        g3[0] = 1;
        assert_ne!(a, eval_h(&params, &g3).0, "the first block did not affect the output?");
    }

    #[test]
    fn same_weight_inputs_give_different_outputs() {
        for &ell in &[1usize, 2] {
            let params = HashParams::sample(402 + ell as u64, 16, 4, ell);
            let ng = params.num_groups();
            let x = vec![3usize, 5, 3, 5];
            let y = vec![5usize, 3, 5, 3];
            let z = vec![3usize, 3, 5, 5];
            assert_eq!(x.len(), ng);
            let hx = eval_h(&params, &x).0;
            assert_ne!(hx, eval_h(&params, &y).0, "ell={ell}: same output after permutation (weight collision)");
            assert_ne!(hx, eval_h(&params, &z).0, "ell={ell}: same output after permutation (weight collision)");
        }
    }

    #[test]
    fn same_symbol_differs_across_blocks() {
        let params = HashParams::sample(403, 16, 4, 2);
        for v in 0..params.table_size() {
            let a0: Vec<_> = (0..4).map(|k| params.a(0, v, k / 2, k % 2).clone()).collect();
            let a1: Vec<_> = (0..4).map(|k| params.a(1, v, k / 2, k % 2).clone()).collect();
            assert_ne!(a0, a1, "symbol {v} has the same matrix in block 0 and block 1");
        }
    }

    #[test]
    #[ignore]
    fn eval_h_breakdown() {
        use std::time::Instant;
        const REPS: u32 = 20;
        for &w in &[4usize, 8] {
            let params = HashParams::sample(20260901, 128, w, crate::params::ELL);
            let mut rng = SimpleRng::new(42);
            let bits: Vec<bool> = (0..128).map(|_| rng.next_bool()).collect();
            let groups = bits_to_groups(&params, &bits);
            let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3 / REPS as f64;

            let t = Instant::now();
            for _ in 0..REPS {
                for (i, &v) in groups.iter().enumerate() {
                    std::hint::black_box(params.spectra_get(i, v));
                }
            }
            let t_spec = ms(t);

            let t = Instant::now();
            for _ in 0..REPS {
                std::hint::black_box(eval_h(&params, &groups));
            }
            let t_all = ms(t);

            let ng = params.num_groups();
            let m = params.ell;
            println!(
                "w={w}: eval_h {t_all:.3} ms = spectra {t_spec:.3} ms ({:.0}%, {} forward NTTs) + chain {:.3} ms ({:.0}%, {} ring mults)",
                100.0 * t_spec / t_all,
                ng * m * m,
                t_all - t_spec,
                100.0 * (t_all - t_spec) / t_all,
                (ng - 1) * m * m
            );
        }
    }

    #[test]
    fn bit_group_roundtrip() {
        let params = HashParams::dims_only(24, 4, 2);
        let mut rng = SimpleRng::new(9);
        let bits: Vec<bool> = (0..24).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        assert_eq!(groups_to_bits(&params, &groups), bits);
    }
}
