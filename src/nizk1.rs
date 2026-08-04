use crate::field::Fq;
use crate::hash::HashWitness;
use crate::ntt::{full_inner_product, neg_and_quotient_rows, to_spectra, Spectra};
use crate::params::HashParams;
use crate::ring::{reduce_only, RingElem, N};
use crate::rng::CsRng;

#[cfg(feature = "q32")]
pub const R_DIM: usize = 6;
#[cfg(feature = "q64")]
pub const R_DIM: usize = 10;
#[cfg(feature = "q32")]
pub const COM_N: usize = 6;
#[cfg(feature = "q64")]
pub const COM_N: usize = 5;
#[cfg(feature = "q32")]
pub const W_SLACK: usize = 3;
#[cfg(feature = "q64")]
pub const W_SLACK: usize = 5;

pub const HPACK_BITS: usize = 512;

const _: () = assert!(HPACK_BITS.is_power_of_two() && HPACK_BITS <= N, "one h_pack row does not fit into a single ring element");

pub struct ComKey {
    pub a: Vec<Vec<RingElem>>,
    pub b: Vec<Vec<RingElem>>,
    spec: Vec<Spectra>,
}

impl ComKey {
    pub fn sample(rng: &mut CsRng, com_n: usize, msg_len: usize, w: usize) -> Self {
        let rho_len = com_n + msg_len + w;
        assert!(rho_len > com_n + msg_len, "the hiding of Appendix F requires w >= 1");
        let re = |rng: &mut CsRng| RingElem { c: (0..N).map(|_| rng.next_fq()).collect() };
        let a: Vec<Vec<RingElem>> =
            (0..com_n).map(|_| (0..rho_len).map(|_| re(rng)).collect()).collect();
        let b: Vec<Vec<RingElem>> =
            (0..msg_len).map(|_| (0..rho_len).map(|_| re(rng)).collect()).collect();
        let spec: Vec<Spectra> =
            a.iter().chain(&b).flat_map(|row| row.iter()).map(|e| to_spectra(&e.c)).collect();
        ComKey { a, b, spec }
    }
    pub fn com_n(&self) -> usize {
        self.a.len()
    }
    pub fn msg_len(&self) -> usize {
        self.b.len()
    }
    pub fn rho_len(&self) -> usize {
        self.a.first().map(|r| r.len()).unwrap_or_else(|| self.b[0].len())
    }
    pub fn out_len(&self) -> usize {
        self.com_n() + self.msg_len()
    }
}

pub fn derive_ar(r_dim: usize, ell: usize, c_r: &[RingElem]) -> Vec<Vec<RingElem>> {
    let mut enc = Vec::with_capacity(c_r.len() * N * crate::field::FQ_BYTES);
    for e in c_r {
        debug_assert_eq!(e.c.len(), N);
        for c in &e.c {
            debug_assert!((c.0 as u64) < crate::field::Q, "c_r coefficients must be canonical representatives");
            enc.extend_from_slice(&crate::field::fq_le_bytes(*c));
        }
    }
    let mut dims = Vec::with_capacity(24);
    for v in [r_dim as u64, ell as u64, N as u64] {
        dims.extend_from_slice(&v.to_le_bytes());
    }
    let mut rng = CsRng::from_parts("voprf-Ar-v1", &[&dims, &enc]);
    (0..r_dim)
        .map(|_| {
            (0..ell)
                .map(|_| RingElem { c: (0..N).map(|_| rng.next_fq()).collect() })
                .collect()
        })
        .collect()
}

pub struct Nizk1Params {
    pub r_dim: usize,
    pub com_r: ComKey,
    pub com_x: ComKey,
    pub hpack_len: usize,
    pub crs_digest: [u8; 32],
}

fn nizk1_crs_digest(r_dim: usize, hpack_len: usize, com_r: &ComKey, com_x: &ComKey) -> [u8; 32] {
    let mut tr = crate::transcript::Transcript::new("voprf-nizk1-crs-v1");
    tr.absorb_u64(N as u64);
    tr.absorb_u64(r_dim as u64);
    tr.absorb_u64(hpack_len as u64);
    for ck in [com_r, com_x] {
        tr.absorb_u64(ck.com_n() as u64);
        tr.absorb_u64(ck.msg_len() as u64);
        tr.absorb_u64(ck.rho_len() as u64);
        for row in ck.a.iter().chain(&ck.b) {
            for e in row {
                tr.absorb_fqs(&e.c);
            }
        }
    }
    tr.finalize_digest()
}

impl Nizk1Params {
    pub fn sample(seed: u64, params: &HashParams, r_dim: usize, com_n: usize, w: usize) -> Self {
        let mut rng = CsRng::from_parts("voprf-nizk1-crs-v1", &[&seed.to_le_bytes()]);
        let h_cells = params.num_groups().next_power_of_two() * params.table_size();
        assert!(h_cells.is_power_of_two(), "h_cells = g_pad*2^g must be a power of two (precondition for hpack indexing)");
        let hpack_len = h_cells.div_ceil(HPACK_BITS).max(1);
        debug_assert!(hpack_len.is_power_of_two(), "hpack_len must be a power of two (see hpack_bits)");
        let com_r = ComKey::sample(&mut rng, com_n, 2 * r_dim, w);
        let com_x = ComKey::sample(&mut rng, com_n, hpack_len, w);
        let crs_digest = nizk1_crs_digest(r_dim, hpack_len, &com_r, &com_x);
        Nizk1Params { r_dim, com_r, com_x, hpack_len, crs_digest }
    }

    pub fn hpack_bits(&self, h_cells: usize) -> usize {
        (h_cells / self.hpack_len).min(HPACK_BITS).max(1)
    }
}

pub struct BlindWitness {
    pub counter: u64,
    pub r_pos: Vec<RingElem>,
    pub r_neg: Vec<RingElem>,
    pub rho_r_pos: Vec<RingElem>,
    pub rho_r_neg: Vec<RingElem>,
    pub rho_x_pos: Vec<RingElem>,
    pub rho_x_neg: Vec<RingElem>,
    pub h_pack: Vec<RingElem>,
    pub zk_seed: [u8; 32],
}

pub struct BlindStatement {
    pub j: u64,
    pub c_x: Vec<RingElem>,
    pub c_r: Vec<RingElem>,
    pub d_x: Vec<RingElem>,
}

pub struct BlindQuotients {
    pub neg_hi_r: Vec<Vec<Fq>>,
    pub cr: Vec<Vec<Fq>>,
    pub dx: Vec<Vec<Fq>>,
}

fn rand_bits(rng: &mut CsRng, n: usize) -> Vec<RingElem> {
    (0..n)
        .map(|_| RingElem {
            c: (0..N).map(|_| if rng.next_bool() { Fq::ONE } else { Fq::ZERO }).collect(),
        })
        .collect()
}

pub fn sample_blind(
    secret: &[u8; 32],
    counter: u64,
    nz: &Nizk1Params,
    h_bits: &[bool],
) -> BlindWitness {
    let mut rng = CsRng::from_parts("voprf-nizk1-blind-v1", &[secret, &counter.to_le_bytes()]);
    let rho_len = nz.com_r.rho_len();
    BlindWitness {
        counter,
        r_pos: rand_bits(&mut rng, nz.r_dim),
        r_neg: rand_bits(&mut rng, nz.r_dim),
        rho_r_pos: rand_bits(&mut rng, rho_len),
        rho_r_neg: rand_bits(&mut rng, rho_len),
        rho_x_pos: rand_bits(&mut rng, nz.com_x.rho_len()),
        rho_x_neg: rand_bits(&mut rng, nz.com_x.rho_len()),
        h_pack: pack_h(nz, h_bits),
        zk_seed: {
            let mut sd = [0u8; 32];
            rng.fill_bytes(&mut sd);
            sd
        },
    }
}

pub fn pack_h(nz: &Nizk1Params, h_bits: &[bool]) -> Vec<RingElem> {
    let per = nz.hpack_bits(h_bits.len());
    (0..nz.hpack_len)
        .map(|r| {
            let mut e = RingElem::zero();
            for c in 0..per {
                let idx = r * per + c;
                if idx < h_bits.len() && h_bits[idx] {
                    e.c[c] = Fq::ONE;
                }
            }
            e
        })
        .collect()
}

#[cfg(test)]
fn diff(pos: &[RingElem], neg: &[RingElem]) -> Vec<RingElem> {
    pos.iter().zip(neg).map(|(a, b)| a - b).collect()
}

fn lin_with_quotient(
    pub_coeffs: &[RingElem],
    pos: &[RingElem],
    neg: &[RingElem],
    scale2: bool,
) -> (RingElem, Vec<Fq>) {
    let specs: Vec<_> = pub_coeffs.iter().map(|e| to_spectra(&e.c)).collect();
    let fp = full_inner_product(&specs, pos);
    let fneg = full_inner_product(&specs, neg);
    let mut full: Vec<Fq> = fp.iter().zip(&fneg).map(|(&a, &b)| a - b).collect();
    if scale2 {
        for v in full.iter_mut() {
            *v = *v + *v;
        }
    }
    let hi: Vec<Fq> = full[N..].to_vec();
    (reduce_only(&full), hi)
}

pub fn blind_statement(
    params: &HashParams,
    nz: &Nizk1Params,
    ch: &[RingElem],
    w: &BlindWitness,
) -> (BlindStatement, BlindQuotients) {

    let msg_r: Vec<RingElem> = w.r_pos.iter().chain(&w.r_neg).cloned().collect();
    let (c_r, q_cr) = commit(&nz.com_r, &w.rho_r_pos, &w.rho_r_neg, &msg_r);

    let a_r = derive_ar(nz.r_dim, params.ell, &c_r);

    let mut c_x = Vec::with_capacity(params.ell);
    let mut neg_hi_r = Vec::with_capacity(params.ell);
    for j in 0..params.ell {
        let col: Vec<RingElem> = (0..nz.r_dim).map(|u| a_r[u][j].clone()).collect();
        let (red, hi) = lin_with_quotient(&col, &w.r_pos, &w.r_neg, false);
        c_x.push(&red + &ch[j]);
        neg_hi_r.push(hi.iter().map(|&v| -v).collect());
    }

    let (d_x, q_dx) = commit(&nz.com_x, &w.rho_x_pos, &w.rho_x_neg, &w.h_pack);

    (
        BlindStatement { j: w.counter, c_x, c_r, d_x },
        BlindQuotients { neg_hi_r, cr: q_cr, dx: q_dx },
    )
}

fn commit(
    ck: &ComKey,
    rho_pos: &[RingElem],
    rho_neg: &[RingElem],
    msg: &[RingElem],
) -> (Vec<RingElem>, Vec<Vec<Fq>>) {
    let rows = ck.out_len();
    let rp = neg_and_quotient_rows(&ck.spec, rows, rho_pos);
    let rn = neg_and_quotient_rows(&ck.spec, rows, rho_neg);

    let mut out = Vec::with_capacity(rows);
    let mut qs = Vec::with_capacity(rows);
    for (i, ((red_p, t_p), (red_n, t_n))) in rp.into_iter().zip(rn).enumerate() {
        let mut red = &red_p - &red_n;
        let mut hi: Vec<Fq> = t_n.iter().zip(&t_p).map(|(&a, &b)| a - b).collect();
        if i >= ck.com_n() {
            red = &red + &red;
            for v in hi.iter_mut() {
                *v = *v + *v;
            }
            red = &red + &msg[i - ck.com_n()];
        }
        out.push(red);
        qs.push(hi);
    }
    (out, qs)
}

pub fn check_blind(
    params: &HashParams,
    nz: &Nizk1Params,
    ch: &[RingElem],
    w: &BlindWitness,
    st: &BlindStatement,
) -> bool {
    let (st2, _) = blind_statement(params, nz, ch, w);
    let all_bits = |v: &[RingElem]| v.iter().all(|e| e.c.iter().all(|c| c.0 <= 1));
    all_bits(&w.r_pos)
        && all_bits(&w.r_neg)
        && all_bits(&w.rho_r_pos)
        && all_bits(&w.rho_r_neg)
        && all_bits(&w.rho_x_pos)
        && all_bits(&w.rho_x_neg)
        && all_bits(&w.h_pack)
        && st2.j == st.j
        && st2.c_x == st.c_x
        && st2.c_r == st.c_r
        && st2.d_x == st.d_x
}

pub const ZK_MASK_ROWS: usize = 1;

const _: () = assert!(ZK_MASK_ROWS >= 1, "ZK witness-level masking needs at least one row");

pub fn zk_mask_rows(rng: &mut CsRng) -> Vec<RingElem> {
    rand_bits(rng, ZK_MASK_ROWS)
}

pub fn extra_w_rows(nz: &Nizk1Params) -> usize {
    2 * nz.r_dim
        + 2 * nz.com_r.rho_len()
        + nz.hpack_len.next_power_of_two()
        + 2 * nz.com_x.rho_len()
        + ZK_MASK_ROWS
}

pub struct WLayout {
    pub r_pos: usize,
    pub r_neg: usize,
    pub rho_r_pos: usize,
    pub rho_r_neg: usize,
    pub h_pack: usize,
    pub rho_x_pos: usize,
    pub rho_x_neg: usize,
    pub mask: usize,
    pub total: usize,
}

pub fn w_layout(nz: &Nizk1Params, base: usize) -> WLayout {
    let rho_r = nz.com_r.rho_len();
    let rho_x = nz.com_x.rho_len();
    let hp = nz.hpack_len.next_power_of_two();
    let r_pos = base;
    let r_neg = r_pos + nz.r_dim;
    let rho_r_pos = r_neg + nz.r_dim;
    let rho_r_neg = rho_r_pos + rho_r;
    let after = rho_r_neg + rho_r;
    let h_pack = after.next_multiple_of(hp);
    let rho_x_pos = h_pack + hp;
    let rho_x_neg = rho_x_pos + rho_x;
    let mask = rho_x_neg + rho_x;
    WLayout {
        r_pos,
        r_neg,
        rho_r_pos,
        rho_r_neg,
        h_pack,
        rho_x_pos,
        rho_x_neg,
        mask,
        total: mask + ZK_MASK_ROWS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::SimpleRng;
    use crate::hash::{bits_to_groups, eval_h};
    use crate::rng::insecure_test_secret;

    #[test]
    fn statement_recomputation_matches() {
        let params = HashParams::sample(5, 8, 2, 1);
        let nz = Nizk1Params::sample(6, &params, 3, 2, 2);
        let mut rng = SimpleRng::new(7);
        let bits: Vec<bool> = (0..8).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        let (ch, _) = eval_h(&params, &groups);
        let h_cells = params.num_groups().next_power_of_two() * params.table_size();
        let mut hb = vec![false; h_cells];
        for (i, &v) in groups.iter().enumerate() {
            hb[i * params.table_size() + v] = true;
        }
        let w = sample_blind(&insecure_test_secret(8), 0, &nz, &hb);
        let (st, _) = blind_statement(&params, &nz, &ch, &w);
        assert!(check_blind(&params, &nz, &ch, &w, &st));
    }

    #[test]
    fn blinding_depends_on_the_secret() {
        let params = HashParams::sample(21, 8, 2, 1);
        let nz = Nizk1Params::sample(22, &params, 3, 2, 2);
        let h_cells = params.num_groups().next_power_of_two() * params.table_size();
        let hb = vec![false; h_cells];
        let a = sample_blind(&insecure_test_secret(1), 0, &nz, &hb);
        let b = sample_blind(&insecure_test_secret(2), 0, &nz, &hb);
        let a2 = sample_blind(&insecure_test_secret(1), 0, &nz, &hb);
        assert_eq!(a.r_pos, a2.r_pos, "the same secret must be deterministic");
        assert_eq!(a.rho_x_pos, a2.rho_x_pos);
        assert_ne!(a.r_pos, b.r_pos, "different secret => different R+");
        assert_ne!(a.rho_r_pos, b.rho_r_pos, "different secret => different rho_r+");
        assert_ne!(a.rho_x_pos, b.rho_x_pos, "different secret => different rho_x+");
        assert_ne!(a.r_pos, a.r_neg);
        assert_ne!(a.rho_r_pos, a.rho_x_pos);
    }

    #[test]
    fn blinding_differs_across_counters() {
        let params = HashParams::sample(41, 8, 2, 1);
        let nz = Nizk1Params::sample(42, &params, 3, 2, 2);
        let h_cells = params.num_groups().next_power_of_two() * params.table_size();
        let hb = vec![false; h_cells];
        let secret = insecure_test_secret(43);
        let ws: Vec<_> = (0..3u64).map(|j| sample_blind(&secret, j, &nz, &hb)).collect();
        for i in 0..ws.len() {
            assert_eq!(ws[i].counter, i as u64);
            for k in (i + 1)..ws.len() {
                assert_ne!(ws[i].r_pos, ws[k].r_pos, "counter {i} vs {k}: identical R+");
                assert_ne!(ws[i].r_neg, ws[k].r_neg, "counter {i} vs {k}: identical R-");
                assert_ne!(ws[i].rho_r_pos, ws[k].rho_r_pos, "counter {i} vs {k}: identical rho_r+");
                assert_ne!(ws[i].rho_x_pos, ws[k].rho_x_pos, "counter {i} vs {k}: identical rho_x+");
            }
        }
        assert_eq!(ws[1].r_pos, sample_blind(&secret, 1, &nz, &hb).r_pos);
    }

    #[test]
    fn blinding_is_binary_and_balanced() {
        let params = HashParams::sample(31, 8, 2, 1);
        let nz = Nizk1Params::sample(32, &params, 3, 2, 2);
        let h_cells = params.num_groups().next_power_of_two() * params.table_size();
        let w = sample_blind(&insecure_test_secret(33), 0, &nz, &vec![false; h_cells]);
        for (name, seg) in [
            ("r_pos", &w.r_pos),
            ("r_neg", &w.r_neg),
            ("rho_r_pos", &w.rho_r_pos),
            ("rho_r_neg", &w.rho_r_neg),
            ("rho_x_pos", &w.rho_x_pos),
            ("rho_x_neg", &w.rho_x_neg),
        ] {
            let total: usize = seg.iter().map(|e| e.c.len()).sum();
            let ones: usize = seg
                .iter()
                .flat_map(|e| e.c.iter())
                .map(|c| {
                    assert!(c.0 <= 1, "{name} is not a bit");
                    c.0 as usize
                })
                .sum();
            assert!(
                ones > total * 40 / 100 && ones < total * 60 / 100,
                "{name} imbalanced: {ones}/{total}"
            );
        }
    }

    #[test]
    fn c_x_matches_direct_ring_arithmetic() {
        let params = HashParams::sample(9, 8, 2, 2);
        let nz = Nizk1Params::sample(10, &params, 2, 1, 1);
        let mut rng = SimpleRng::new(11);
        let bits: Vec<bool> = (0..8).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        let (ch, _) = eval_h(&params, &groups);
        let h_cells = params.num_groups().next_power_of_two() * params.table_size();
        let mut hb = vec![false; h_cells];
        for (i, &v) in groups.iter().enumerate() {
            hb[i * params.table_size() + v] = true;
        }
        let w = sample_blind(&insecure_test_secret(12), 0, &nz, &hb);
        let (st, _) = blind_statement(&params, &nz, &ch, &w);
        let a_r = derive_ar(nz.r_dim, params.ell, &st.c_r);
        let r = diff(&w.r_pos, &w.r_neg);
        for j in 0..params.ell {
            let mut acc = ch[j].clone();
            for u in 0..nz.r_dim {
                acc = &acc + &(&r[u] * &a_r[u][j]);
            }
            assert_eq!(acc, st.c_x[j], "chain {j}");
        }
    }

    #[test]
    fn ar_is_derived_from_cr() {
        let mut cr: Vec<RingElem> = (0..3)
            .map(|i| RingElem { c: (0..N).map(|k| Fq(((i * 7 + k) as u64 % 1000) as _)).collect() })
            .collect();
        let a = derive_ar(2, 3, &cr);
        assert_eq!(a, derive_ar(2, 3, &cr), "the same input must be deterministic");
        cr[1].c[500] = cr[1].c[500] + Fq::ONE;
        let b = derive_ar(2, 3, &cr);
        assert_ne!(a, b, "A_r did not change after modifying c_r");
        assert_ne!(derive_ar(2, 3, &cr)[0][0], derive_ar(3, 2, &cr)[0][0]);
    }

    #[test]
    fn ar_coefficients_are_uniform_over_range() {
        let cr: Vec<RingElem> =
            (0..2).map(|_| RingElem { c: (0..N).map(|k| Fq(k as _)).collect() }).collect();
        let a = derive_ar(2, 2, &cr);
        let (mut hi, mut total) = (0usize, 0usize);
        for row in &a {
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
}

pub type _PhaseAWitness = HashWitness;
