use crate::field::Fq;
use crate::ntt::{full_inner_product, neg_and_quotient_rows, to_spectra, Spectra};
use crate::params::HashParams;
use crate::ring::{reduce_only, RingElem, N};
use crate::rng::CsRng;

pub const R_DIM: usize = 10;
pub const COM_N: usize = 5;
pub const W_SLACK: usize = 5;

pub const HPACK_BITS: usize = 512;

const _: () = assert!(HPACK_BITS.is_power_of_two() && HPACK_BITS <= N, "an h_pack row must fit in one ring element");

pub struct ComKey {
    pub a: Vec<Vec<RingElem>>,
    pub b: Vec<Vec<RingElem>>,
    spec: Vec<Spectra>,
}

impl ComKey {
    pub fn sample(rng: &mut CsRng, com_n: usize, msg_len: usize, w: usize) -> Self {
        let rho_len = com_n + msg_len + w;
        assert!(rho_len > com_n + msg_len, "hiding (Appendix F) requires w >= 1");
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
        let h_cells = crate::relation::h_cells(params);
        assert!(h_cells.is_power_of_two(), "h_cells = g_pad*2^g must be a power of two (required by hpack indexing)");
        let hpack_len = crate::relation::hpack_rows(params);
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

pub struct QueryTicket {
    secret: [u8; 32],
    counter: u64,
}

impl QueryTicket {
    pub fn counter(&self) -> u64 {
        self.counter
    }

    pub fn insecure_for_tests(secret: [u8; 32], counter: u64) -> Self {
        QueryTicket { secret, counter }
    }
}

pub struct QueryCounter {
    secret: [u8; 32],
    next: u64,
}

impl QueryCounter {
    pub fn new(secret: [u8; 32]) -> Self {
        QueryCounter { secret, next: 0 }
    }

    pub fn resume(secret: [u8; 32], next: u64) -> Self {
        QueryCounter { secret, next }
    }

    pub fn peek(&self) -> u64 {
        self.next
    }

    pub fn issue(&mut self) -> QueryTicket {
        let counter = self.next;
        self.next = self.next.checked_add(1).expect("query counter overflow: rotate the client secret");
        QueryTicket { secret: self.secret, counter }
    }
}

pub fn sample_blind(
    ticket: QueryTicket,
    params: &HashParams,
    nz: &Nizk1Params,
    groups: &[usize],
) -> BlindWitness {
    let QueryTicket { secret, counter } = ticket;
    let mut rng =
        CsRng::from_parts("voprf-nizk1-blind-v1", &[&secret, &counter.to_le_bytes()]);
    let rho_len = nz.com_r.rho_len();
    let r_pos = rand_bits(&mut rng, nz.r_dim);
    let r_neg = rand_bits(&mut rng, nz.r_dim);
    let rho_r_pos = rand_bits(&mut rng, rho_len);
    let rho_r_neg = rand_bits(&mut rng, rho_len);
    let rho_x_pos = rand_bits(&mut rng, nz.com_x.rho_len());
    let rho_x_neg = rand_bits(&mut rng, nz.com_x.rho_len());
    let h_pack = pack_h(nz, &crate::relation::h_bits(params, groups));
    let mut zk_seed = [0u8; 32];
    rng.fill_bytes(&mut zk_seed);
    BlindWitness {
        counter,
        r_pos,
        r_neg,
        rho_r_pos,
        rho_r_neg,
        rho_x_pos,
        rho_x_neg,
        h_pack,
        zk_seed,
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

pub fn extra_w_rows(params: &HashParams, nz: Option<&Nizk1Params>) -> usize {
    w_layout(params, nz).total - crate::relation::num_m_rows(params)
}

pub struct WLayout {
    pub h_pack: usize,
    pub r_pos: usize,
    pub r_neg: usize,
    pub rho_r_pos: usize,
    pub rho_r_neg: usize,
    pub rho_x_pos: usize,
    pub rho_x_neg: usize,
    pub total: usize,
}

pub fn w_layout(params: &HashParams, nz: Option<&Nizk1Params>) -> WLayout {
    let base = crate::relation::num_m_rows(params);
    let hp = crate::relation::hpack_rows(params);
    let h_pack = base.next_multiple_of(hp);
    let after_h = h_pack + hp;
    let Some(nz) = nz else {
        return WLayout {
            h_pack,
            r_pos: after_h,
            r_neg: after_h,
            rho_r_pos: after_h,
            rho_r_neg: after_h,
            rho_x_pos: after_h,
            rho_x_neg: after_h,
            total: after_h,
        };
    };
    let rho_r = nz.com_r.rho_len();
    let rho_x = nz.com_x.rho_len();
    debug_assert_eq!(nz.hpack_len, hp, "Nizk1Params::hpack_len is out of sync with relation::hpack_rows");
    let r_pos = after_h;
    let r_neg = r_pos + nz.r_dim;
    let rho_r_pos = r_neg + nz.r_dim;
    let rho_r_neg = rho_r_pos + rho_r;
    let rho_x_pos = rho_r_neg + rho_r;
    let rho_x_neg = rho_x_pos + rho_x;
    WLayout {
        h_pack,
        r_pos,
        r_neg,
        rho_r_pos,
        rho_r_neg,
        rho_x_pos,
        rho_x_neg,
        total: rho_x_neg + rho_x,
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
        let w = sample_blind(QueryTicket::insecure_for_tests(insecure_test_secret(8), 0), &params, &nz, &groups);
        let (st, _) = blind_statement(&params, &nz, &ch, &w);
        assert!(check_blind(&params, &nz, &ch, &w, &st));
    }

    #[test]
    fn c_x_matches_direct_ring_arithmetic() {
        let params = HashParams::sample(9, 8, 2, 2);
        let nz = Nizk1Params::sample(10, &params, 2, 1, 1);
        let mut rng = SimpleRng::new(11);
        let bits: Vec<bool> = (0..8).map(|_| rng.next_bool()).collect();
        let groups = bits_to_groups(&params, &bits);
        let (ch, _) = eval_h(&params, &groups);
        let w = sample_blind(QueryTicket::insecure_for_tests(insecure_test_secret(12), 0), &params, &nz, &groups);
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
    fn hpack_segment_layout_invariants() {
        use crate::relation::{h_cells, hpack_per, hpack_rows, num_m_rows};
        let mut saw_rounding = false;
        let mut saw_single_row = false;
        let mut saw_padding_blocks = false;
        for &(n, g) in &[
            (8usize, 1usize),
            (8, 2),
            (8, 4),
            (8, 8),
            (12, 2),
            (12, 1),
            (16, 1),
            (16, 2),
            (16, 4),
            (16, 8),
            (24, 8),
            (32, 8),
            (48, 8),
            (64, 8),
            (128, 8),
            (256, 8),
            (512, 8),
            (1024, 8),
            (1040, 8),
        ] {
            for ell in [1usize, 2, 3, 5] {
                if n % g != 0 || n / g < 2 {
                    continue;
                }
                let params = HashParams::dims_only(n, g, ell);
                let hp = hpack_rows(&params);
                let per = hpack_per(&params);
                assert!(hp.is_power_of_two(), "hpack_rows is not a power of two (n={n} g={g})");
                assert!(per.is_power_of_two(), "hpack_per is not a power of two (n={n} g={g})");
                assert_eq!(hp * per, h_cells(&params), "incomplete shape (n={n} g={g})");
                assert!(per <= N, "a W row cannot hold {per} cells (n={n} g={g})");
                if params.num_groups() < crate::relation::g_pad(&params) {
                    saw_padding_blocks = true;
                }
                if hp == 1 {
                    saw_single_row = true;
                }
                if num_m_rows(&params) % hp != 0 {
                    saw_rounding = true;
                }
                let nz = Nizk1Params::sample(1, &params, 3, 2, 2);
                assert_eq!(nz.hpack_len, hp, "Nizk1Params::hpack_len out of sync (n={n} g={g})");
                for lay in [w_layout(&params, None), w_layout(&params, Some(&nz))] {
                    assert_eq!(lay.h_pack % hp, 0, "h_pack not aligned to hpack_rows (n={n} g={g})");
                    assert!(lay.h_pack >= num_m_rows(&params), "h_pack overlaps the m bit rows");
                    assert!(lay.h_pack + hp <= lay.total, "h_pack segment exceeds total");
                }
                let b = w_layout(&params, Some(&nz));
                let mut prev = b.h_pack + hp;
                for (name, start) in [
                    ("r_pos", b.r_pos),
                    ("r_neg", b.r_neg),
                    ("rho_r_pos", b.rho_r_pos),
                    ("rho_r_neg", b.rho_r_neg),
                    ("rho_x_pos", b.rho_x_pos),
                    ("rho_x_neg", b.rho_x_neg),
                ] {
                    assert!(start >= prev, "{name} overlaps the previous segment (n={n} g={g})");
                    prev = start;
                }
                assert_eq!(b.total, b.rho_x_neg + nz.com_x.rho_len());
            }
        }
        assert!(saw_single_row, "no shape with hpack_rows == 1 was exercised");
        assert!(saw_padding_blocks, "no shape with g_pad > G (padding blocks) was exercised");
        assert!(saw_rounding, "no shape requiring num_m_rows round-up was exercised");
    }

    #[test]
    fn ar_is_derived_from_cr() {
        let mut cr: Vec<RingElem> = (0..3)
            .map(|i| RingElem { c: (0..N).map(|k| Fq(((i * 7 + k) as u64 % 1000) as _)).collect() })
            .collect();
        let a = derive_ar(2, 3, &cr);
        assert_eq!(a, derive_ar(2, 3, &cr), "same input must be deterministic");
        cr[1].c[500] = cr[1].c[500] + Fq::ONE;
        let b = derive_ar(2, 3, &cr);
        assert_ne!(a, b, "A_r did not change after modifying c_r");
        assert_ne!(derive_ar(2, 3, &cr)[0][0], derive_ar(3, 2, &cr)[0][0]);
    }

}
