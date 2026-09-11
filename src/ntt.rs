
use crate::field::{Fq, Q};
use crate::ring::{RingElem, N};
use rayon::prelude::*;
use std::sync::OnceLock;
use tfhe_ntt::prime32::Plan;

const L: usize = 2 * N;

const PRIMES: [u64; 5] =
    [2147473409, 2147389441, 2147387393, 2147377153, 2147358721];

const _: () = assert!(L.is_power_of_two());
const _: () = {
    let mut i = 0;
    while i < PRIMES.len() {
        assert!((PRIMES[i] - 1) % (2 * L as u64) == 0, "the primes cannot support a negacyclic-L NTT");
        i += 1;
    }
};
pub const NHOT: usize = 3;
pub const NGEN: usize = 5;

const _: () = assert!(NHOT <= NGEN && NGEN <= PRIMES.len(), "not enough primes");

struct Engine {
    plans: [Plan; NPRIMES],
}
impl Engine {
    fn get() -> &'static Engine {
        static E: OnceLock<Engine> = OnceLock::new();
        E.get_or_init(|| Engine {
            plans: std::array::from_fn(|i| {
                Plan::try_new(L, PRIMES[i] as u32).expect("2N-NTT plan")
            }),
        })
    }
}

fn fwd_into(a: &[Fq], p: u64, plan: &Plan, buf: &mut [u32]) {
    buf.fill(0);
    for (i, &v) in a.iter().enumerate() {
        buf[i] = (v.0 as u64 % p) as u32;
    }
    plan.fwd(buf);
}

fn fwd_small_into(a: &[Fq], plan: &Plan, buf: &mut [u32]) {
    const _: () = assert!(SMALL_BOUND <= PRIMES[0], "small inputs must be < every RNS prime so that the % p can be skipped");
    buf.fill(0);
    for (i, &v) in a.iter().enumerate() {
        debug_assert!((v.0 as u64) < SMALL_BOUND, "the coefficients of small inputs must be < {SMALL_BOUND}");
        buf[i] = v.0 as u32;
    }
    plan.fwd(buf);
}

pub const SMALL_BOUND: u64 = crate::ring::W_RANGE_BASE;

macro_rules! with_prime {
    ($pi:expr, $p:ident, $body:block) => {
        match $pi {
            0 => {
                const $p: u64 = PRIMES[0];
                $body
            }
            1 => {
                const $p: u64 = PRIMES[1];
                $body
            }
            2 => {
                const $p: u64 = PRIMES[2];
                $body
            }
            3 => {
                const $p: u64 = PRIMES[3];
                $body
            }
            4 => {
                const $p: u64 = PRIMES[4];
                $body
            }
            other => unreachable!("with_prime!: PRIMES only has {} entries, got index {}", NPRIMES, other),
        }
    };
}

#[inline(always)]
fn shoup_pre(w: u32, p: u64) -> u32 {
    (((w as u64) << 32) / p) as u32
}

#[inline(always)]
fn shoup_mul_lazy(w: u32, w_pre: u32, x: u32, p: u64) -> u64 {
    let q_hat = (w_pre as u64 * x as u64) >> 32;
    (w as u64) * (x as u64) - q_hat * p
}

const NPRIMES: usize = PRIMES.len();

const fn inv_mod(a: u64, m: u64) -> u64 {
    let (mut t, mut newt): (i128, i128) = (0, 1);
    let (mut r, mut newr): (i128, i128) = (m as i128, (a % m) as i128);
    while newr != 0 {
        let q = r / newr;
        let tmp = t - q * newt;
        t = newt;
        newt = tmp;
        let tmp = r - q * newr;
        r = newr;
        newr = tmp;
    }
    if t < 0 {
        t += m as i128;
    }
    t as u64
}

const INV_P: [[u64; NPRIMES]; NPRIMES] = {
    let mut out = [[0u64; NPRIMES]; NPRIMES];
    let mut i = 0;
    while i < NPRIMES {
        let mut j = 0;
        while j < i {
            out[j][i] = inv_mod(PRIMES[j], PRIMES[i]);
            j += 1;
        }
        i += 1;
    }
    out
};

const M_MOD_Q: [u64; NPRIMES] = {
    let mut out = [0u64; NPRIMES];
    let mut acc: u64 = 1;
    let mut i = 0;
    while i < NPRIMES {
        out[i] = acc;
        acc = ((acc as u128 * PRIMES[i] as u128) % Q as u128) as u64;
        i += 1;
    }
    out
};

#[inline(always)]
fn crt_modq<const K: usize>(r: [u64; K]) -> Fq {
    debug_assert!(K <= NPRIMES);
    let mut t = [0u64; K];
    t[0] = r[0];
    let mut i = 1;
    while i < K {
        let pi = PRIMES[i];
        debug_assert!(r[i] < pi);
        let mut x = r[i];
        let mut j = 0;
        while j < i {
            let tj = if t[j] >= pi { t[j] - pi } else { t[j] };
            x = (x + pi - tj) % pi;
            x = x * INV_P[j][i] % pi;
            j += 1;
        }
        t[i] = x;
        i += 1;
    }
    let mut acc = Fq(t[0] as _);
    let mut i = 1;
    while i < K {
        acc = acc + Fq(t[i] as _) * Fq(M_MOD_Q[i] as _);
        i += 1;
    }
    acc
}

const _: () = {
    assert!(M_MOD_Q[0] == 1, "the 0th weight of the mixed radix must be 1");
    let mut i = 0;
    while i < NPRIMES {
        assert!(PRIMES[i] < Q, "RNS primes must be < q (precondition for the reduction-free 0th term)");
        let mut j = 0;
        while j < NPRIMES {
            assert!(PRIMES[j] < 2 * PRIMES[i], "the primes differ by less than a factor of 2 (precondition for a single conditional subtraction)");
            j += 1;
        }
        i += 1;
    }
};

#[inline(always)]
fn crt_hot_modq(r: [u64; NHOT]) -> Fq {
    crt_modq::<NHOT>(r)
}

#[inline]
fn crt_gen_modq(r: [u64; NGEN]) -> Fq {
    crt_modq::<NGEN>(r)
}

#[derive(Clone)]
pub struct Spectra {
    pub(crate) fwd: [Vec<u32>; NHOT],
    pub(crate) shoup: [Vec<u32>; NHOT],
}

pub fn to_spectra(a: &[Fq]) -> Spectra {
    to_spectra_impl::<false>(a)
}

pub fn to_spectra_bin(a: &[Fq]) -> Spectra {
    to_spectra_impl::<true>(a)
}

fn to_spectra_impl<const SMALL: bool>(a: &[Fq]) -> Spectra {
    let e = Engine::get();
    let mut fwd: [Vec<u32>; NHOT] = Default::default();
    let mut shoup: [Vec<u32>; NHOT] = Default::default();
    let mut buf = vec![0u32; L];
    for pi in 0..NHOT {
        if SMALL {
            fwd_small_into(a, &e.plans[pi], &mut buf);
        } else {
            fwd_into(a, PRIMES[pi], &e.plans[pi], &mut buf);
        }
        shoup[pi] = with_prime!(pi, P, { buf.iter().map(|&w| shoup_pre(w, P)).collect() });
        fwd[pi] = buf.clone();
    }
    Spectra { fwd, shoup }
}

fn inner_full_mixed<const SMALL_COLS: bool>(specs: &[Spectra], cols: &[RingElem]) -> Vec<Fq> {
    let e = Engine::get();
    assert!(cols.len() <= specs.len());
    debug_assert!(
        !SMALL_COLS || cols.iter().all(|c| c.c.iter().all(|v| (v.0 as u64) < SMALL_BOUND)),
        "hot-path precondition: under SMALL_COLS the coefficients of cols must be bits -- the RNS bound for NHOT primes depends on it"
    );

    let res: Vec<Vec<u32>> = (0..NHOT)
        .into_par_iter()
        .map(|pi| {
            let plan = &e.plans[pi];
            with_prime!(pi, P, {
                let acc = cols
                    .par_iter()
                    .enumerate()
                    .fold(
                        || (vec![0u64; L], vec![0u32; L]),
                        |(mut acc, mut cf), (d, col)| {
                            if SMALL_COLS {
                                fwd_small_into(&col.c, plan, &mut cf);
                            } else {
                                fwd_into(&col.c, P, plan, &mut cf);
                            }
                            let sf = &specs[d].fwd[pi];
                            let sh = &specs[d].shoup[pi];
                            for i in 0..L {
                                acc[i] += shoup_mul_lazy(sf[i], sh[i], cf[i], P);
                            }
                            (acc, cf)
                        },
                    )
                    .map(|(a, _)| a)
                    .reduce(
                        || vec![0u64; L],
                        |mut a, b| {
                            for (x, y) in a.iter_mut().zip(&b) {
                                *x += *y;
                            }
                            a
                        },
                    );
                let mut buf: Vec<u32> = acc.iter().map(|&x| (x % P) as u32).collect();
                plan.normalize(&mut buf);
                plan.inv(&mut buf);
                buf
            })
        })
        .collect();

    (0..2 * N - 1).map(|k| crt_hot_modq(std::array::from_fn(|i| res[i][k] as u64))).collect()
}

fn inner_full_general(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    let e = Engine::get();
    let res: Vec<Vec<u32>> = (0..NGEN)
        .into_par_iter()
        .map(|pi| {
            let p = PRIMES[pi];
            let plan = &e.plans[pi];
            let mut af = vec![0u32; L];
            let mut bf = vec![0u32; L];
            fwd_into(a, p, plan, &mut af);
            fwd_into(b, p, plan, &mut bf);
            for i in 0..L {
                af[i] = ((af[i] as u64 * bf[i] as u64) % p) as u32;
            }
            plan.normalize(&mut af);
            plan.inv(&mut af);
            af
        })
        .collect();
    (0..a.len() + b.len() - 1)
        .map(|k| crt_gen_modq(std::array::from_fn(|i| res[i][k] as u64)))
        .collect()
}

fn fold_neg(full: &[Fq]) -> Vec<Fq> {
    let mut c = vec![Fq::ZERO; N];
    for i in 0..N {
        let hi = if N + i < full.len() { full[N + i] } else { Fq::ZERO };
        c[i] = full[i] - hi;
    }
    c
}
pub fn neg_and_quotient(specs: &[Spectra], cols: &[RingElem]) -> (RingElem, Vec<Fq>) {
    let full = inner_full_mixed::<true>(specs, cols);
    let neg = fold_neg(&full);
    let t: Vec<Fq> = full[N..].iter().map(|&v| -v).collect();
    (RingElem { c: neg }, t)
}

pub fn full_inner_product(specs: &[Spectra], cols: &[RingElem]) -> Vec<Fq> {
    inner_full_mixed::<true>(specs, cols)
}

pub fn neg_and_quotient_rows(
    specs: &[Spectra],
    rows: usize,
    cols: &[RingElem],
) -> Vec<(RingElem, Vec<Fq>)> {
    neg_and_quotient_rows_impl::<true>(specs, rows, cols)
}

pub fn neg_and_quotient_rows_wide(
    specs: &[Spectra],
    rows: usize,
    cols: &[RingElem],
) -> Vec<(RingElem, Vec<Fq>)> {
    neg_and_quotient_rows_impl::<false>(specs, rows, cols)
}

fn neg_and_quotient_rows_impl<const SMALL_COLS: bool>(
    specs: &[Spectra],
    rows: usize,
    cols: &[RingElem],
) -> Vec<(RingElem, Vec<Fq>)> {
    let e = Engine::get();
    let ml = cols.len();
    assert_eq!(specs.len(), rows * ml, "specs must be rows × K (row-major)");
    debug_assert!(
        !SMALL_COLS || cols.iter().all(|c| c.c.iter().all(|v| (v.0 as u64) < SMALL_BOUND)),
        "hot-path precondition: under SMALL_COLS the coefficients of cols must be bits"
    );

    let res: Vec<Vec<Vec<u32>>> = (0..NHOT)
        .into_par_iter()
        .map(|pi| {
            let plan = &e.plans[pi];
            with_prime!(pi, P, {
                let cf: Vec<Vec<u32>> = cols
                    .iter()
                    .map(|col| {
                        let mut buf = vec![0u32; L];
                        if SMALL_COLS {
                            fwd_small_into(&col.c, plan, &mut buf);
                        } else {
                            fwd_into(&col.c, P, plan, &mut buf);
                        }
                        buf
                    })
                    .collect();
                (0..rows)
                    .into_par_iter()
                    .map(|r| {
                        let mut acc = vec![0u64; L];
                        for d in 0..ml {
                            let s = &specs[r * ml + d];
                            let (sf, sh) = (&s.fwd[pi], &s.shoup[pi]);
                            let cfd = &cf[d];
                            for i in 0..L {
                                acc[i] += shoup_mul_lazy(sf[i], sh[i], cfd[i], P);
                            }
                        }
                        let mut buf: Vec<u32> = acc.iter().map(|&x| (x % P) as u32).collect();
                        plan.normalize(&mut buf);
                        plan.inv(&mut buf);
                        buf
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();

    (0..rows)
        .map(|r| {
            let full: Vec<Fq> = (0..2 * N - 1)
                .map(|k| crt_hot_modq(std::array::from_fn(|i| res[i][r][k] as u64)))
                .collect();
            let neg = fold_neg(&full);
            let t: Vec<Fq> = full[N..].iter().map(|&v| -v).collect();
            (RingElem { c: neg }, t)
        })
        .collect()
}

pub fn negacyclic_mul_n(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    fold_neg(&inner_full_general(a, b))
}
pub fn full_mul_n(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    inner_full_general(a, b)
}

pub fn mul_polys(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    let mut out = vec![Fq::ZERO; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] = out[i + j] + x * y;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::SimpleRng;

    fn random_poly(rng: &mut SimpleRng, len: usize) -> Vec<Fq> {
        (0..len).map(|_| rng.next_fq()).collect()
    }
    fn random_bits(rng: &mut SimpleRng, len: usize) -> Vec<Fq> {
        (0..len).map(|_| if rng.next_bool() { Fq::ONE } else { Fq::ZERO }).collect()
    }
    fn schoolbook_full(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
        let mut out = vec![Fq::ZERO; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                out[i + j] = out[i + j] + x * y;
            }
        }
        out
    }

    #[test]
    fn shoup_lazy_mul_matches_mod() {
        let mut rng = SimpleRng::new(31337);
        for pi in 0..3 {
            let p = PRIMES[pi];
            let check = |w: u64, x: u64| {
                let r = shoup_mul_lazy(w as u32, shoup_pre(w as u32, p), x as u32, p);
                assert!(r < 2 * p, "the lazy result must be < 2p (p{pi}, w={w}, x={x})");
                assert_eq!(r % p, w * x % p, "p{pi}, w={w}, x={x}");
            };
            for (w, x) in [(0u64, 0u64), (0, p - 1), (p - 1, 0), (p - 1, p - 1), (1, p - 1)] {
                check(w, x);
            }
            for _ in 0..20000 {
                check(rng.next_u64() % p, rng.next_u64() % p);
            }
        }
    }

    #[test]
    fn ntt_length_is_exactly_2n() {
        assert_eq!(L, 2 * N);
        assert!(L.is_power_of_two(), "L is not a power of two => the tfhe-ntt Plan cannot be built");
        for (i, &p) in PRIMES.iter().enumerate() {
            assert_eq!((p - 1) % (2 * L as u64), 0, "p{i} = {p} does not support negacyclic-{L}");
            assert!(Plan::try_new(L, p as u32).is_some(), "Plan({L}) for p{i} cannot be built");
        }
    }

    #[test]
    fn pad2n_paths_match_schoolbook() {
        let mut rng = SimpleRng::new(2);
        let a = random_poly(&mut rng, N);
        let b = random_poly(&mut rng, N);
        let full = schoolbook_full(&a, &b);
        let mut neg_ref = vec![Fq::ZERO; N];
        for k in 0..N {
            let lo = full[k];
            let hi = if N + k < full.len() { full[N + k] } else { Fq::ZERO };
            neg_ref[k] = lo - hi;
        }
        assert_eq!(negacyclic_mul_n(&a, &b), neg_ref);
        assert_eq!(full_mul_n(&a, &b), full);
    }

    #[test]
    fn spectral_inner_products_match_pairwise() {
        use crate::ring::reduce_only;
        let mut rng = SimpleRng::new(4);
        let d_cnt = 5;
        let a_polys: Vec<Vec<Fq>> = (0..d_cnt).map(|_| random_poly(&mut rng, N)).collect();
        let cols: Vec<RingElem> =
            (0..d_cnt).map(|_| RingElem { c: random_bits(&mut rng, N) }).collect();
        let specs: Vec<Spectra> = a_polys.iter().map(|p| to_spectra(p)).collect();
        let mut full_ref = vec![Fq::ZERO; 2 * N - 1];
        for d in 0..d_cnt {
            for (k, &v) in schoolbook_full(&a_polys[d], &cols[d].c).iter().enumerate() {
                full_ref[k] = full_ref[k] + v;
            }
        }
        assert_eq!(full_inner_product(&specs, &cols), full_ref);
        assert_eq!(neg_and_quotient(&specs, &cols).0, reduce_only(&full_ref));
    }

    #[test]
    fn worst_case_mixed_rns_bound_is_exact() {
        const K_EVAL_H: usize = crate::params::ELL;
        const K_LIN: usize = 10;
        const K_COMMIT: usize = 30;
        let k_max = K_EVAL_H.max(K_LIN).max(K_COMMIT);
        assert_eq!(k_max, 30);

        let ones = RingElem { c: vec![Fq::ONE; N] };
        let maxv = vec![Fq((Q - 1) as _); N];

        for small_cols in [true, false] {
            let (specs, cols): (Vec<Spectra>, Vec<RingElem>) = if small_cols {
                (
                    (0..k_max).map(|_| to_spectra(&maxv)).collect(),
                    (0..k_max).map(|_| ones.clone()).collect(),
                )
            } else {
                (
                    (0..k_max).map(|_| to_spectra_bin(&ones.c)).collect(),
                    (0..k_max).map(|_| RingElem { c: maxv.clone() }).collect(),
                )
            };
            let got = if small_cols {
                full_inner_product(&specs, &cols)
            } else {
                let r = neg_and_quotient_rows_wide(&specs, 1, &cols);
                let _ = &r;
                inner_full_mixed::<false>(&specs, &cols)
            };

            let qm1 = (Q - 1) as u128;
            let mut peak = 0u128;
            for k in 0..2 * N - 1 {
                let pairs = (k.min(N - 1) - k.saturating_sub(N - 1) + 1) as u128;
                let exact = k_max as u128 * qm1 * pairs;
                peak = peak.max(exact);
                assert_eq!(got[k].0 as u128, exact % Q as u128, "small_cols={small_cols} k={k}");
            }
            assert_eq!(peak, k_max as u128 * qm1 * N as u128);

            let modulus: u128 = (0..NHOT).map(|i| PRIMES[i] as u128).product();
            assert!(peak < modulus, "the bound for {NHOT} primes is too small: peak = {peak}, Pi p = {modulus}");

            let bits = 128 - peak.leading_zeros();
            assert_eq!(bits, 78, "the upper bound is not 2^77.9 (the table in the `ntt.rs` module documentation must be kept in sync): 2^{bits}");
            let margin_bits = (128 - modulus.leading_zeros()) - bits;
            assert!(margin_bits >= 15, "only 2^{margin_bits} of RNS headroom left");
        }
    }

    #[test]
    fn to_spectra_bin_matches_general_on_bits() {
        let mut rng = SimpleRng::new(0xB17);
        for _ in 0..8 {
            let bits = random_bits(&mut rng, N);
            let a = to_spectra(&bits);
            let b = to_spectra_bin(&bits);
            for pi in 0..NHOT {
                assert_eq!(a.fwd[pi], b.fwd[pi], "fwd differs for prime {pi}");
                assert_eq!(a.shoup[pi], b.shoup[pi], "shoup differs for prime {pi}");
            }
        }
    }

    #[test]
    fn both_small_sides_agree() {
        let mut rng = SimpleRng::new(0x5DE);
        let k = 5;
        let wide: Vec<Vec<Fq>> = (0..k).map(|_| random_poly(&mut rng, N)).collect();
        let bits: Vec<RingElem> =
            (0..k).map(|_| RingElem { c: random_bits(&mut rng, N) }).collect();

        let sa: Vec<Spectra> = wide.iter().map(|p| to_spectra(p)).collect();
        let a = inner_full_mixed::<true>(&sa, &bits);

        let sb: Vec<Spectra> = bits.iter().map(|b| to_spectra_bin(&b.c)).collect();
        let cb: Vec<RingElem> = wide.iter().map(|p| RingElem { c: p.clone() }).collect();
        let b = inner_full_mixed::<false>(&sb, &cb);

        assert_eq!(a, b, "the inner products of the two directions disagree (did fwd_small_into swap the operands?)");

        let mut refv = vec![Fq::ZERO; 2 * N - 1];
        for d in 0..k {
            for (i, &v) in schoolbook_full(&wide[d], &bits[d].c).iter().enumerate() {
                refv[i] = refv[i] + v;
            }
        }
        assert_eq!(a, refv);
    }

    #[test]
    fn wide_rows_match_schoolbook() {
        use crate::ring::reduce_only;
        let mut rng = SimpleRng::new(0x71D);
        let (rows, k) = (5usize, 5usize);
        let abin: Vec<RingElem> =
            (0..rows * k).map(|_| RingElem { c: random_bits(&mut rng, N) }).collect();
        let specs: Vec<Spectra> = abin.iter().map(|e| to_spectra_bin(&e.c)).collect();
        let cols: Vec<RingElem> =
            (0..k).map(|_| RingElem { c: random_poly(&mut rng, N) }).collect();

        let out = neg_and_quotient_rows_wide(&specs, rows, &cols);
        assert_eq!(out.len(), rows);
        for r in 0..rows {
            let mut full = vec![Fq::ZERO; 2 * N - 1];
            for d in 0..k {
                for (i, &v) in schoolbook_full(&abin[r * k + d].c, &cols[d].c).iter().enumerate() {
                    full[i] = full[i] + v;
                }
            }
            assert_eq!(out[r].0, reduce_only(&full), "reduced value of row {r}");
            assert_eq!(out[r].1.len(), N - 1);
            for (i, &v) in out[r].1.iter().enumerate() {
                assert_eq!(v, -full[N + i], "quotient of row {r} (the protocol convention is -hi)");
            }
        }
    }

    #[test]
    fn binary_path_agrees_with_general_path() {
        let mut rng = SimpleRng::new(20260726);
        for _ in 0..3 {
            let a = random_poly(&mut rng, N);
            let b = random_bits(&mut rng, N);
            let spec = to_spectra(&a);
            let via_binary =
                full_inner_product(std::slice::from_ref(&spec), &[RingElem { c: b.clone() }]);
            assert_eq!(via_binary, full_mul_n(&a, &b));
        }
    }

    #[test]
    #[ignore]
    fn ntt_plan_microbench() {
        use std::time::Instant;
        use tfhe_ntt::prime32::Plan;
        let p = PRIMES[0] as u32;
        let mut rng = SimpleRng::new(1);
        let src: Vec<u32> = (0..4096).map(|_| (rng.next_fq().0 as u64 % PRIMES[0]) as u32).collect();
        let iters = 3000;
        let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / iters as f64;

        println!("\n--- NTT forward microbenchmark (average over {iters} runs, single prime; currently L = {L}) ---");
        for len in [512usize, 1024, 2048, 4096] {
            let Some(plan) = Plan::try_new(len, p) else {
                println!("  L = {len:<5}   (p = {p} does not support negacyclic-{len}, skipped)");
                continue;
            };
            let mut buf = src[..len].to_vec();
            let t = Instant::now();
            for _ in 0..iters {
                plan.fwd(&mut buf);
            }
            let d = t.elapsed();
            println!("  L = {len:<5}{:8.3} us{}", per(d), if len == L { "   ← current" } else { "" });
        }
    }

    #[test]
    #[ignore]
    fn ntt_innerfull_breakdown() {
        const BENCH_K: usize = crate::params::ELL;
        use std::time::Instant;
        let mut rng = SimpleRng::new(9);
        let a: Vec<Vec<Fq>> = (0..BENCH_K).map(|_| random_poly(&mut rng, N)).collect();
        let cols: Vec<RingElem> = (0..BENCH_K)
            .map(|_| RingElem {
                c: (0..N).map(|_| Fq((rng.next_u64() % SMALL_BOUND) as _)).collect(),
            })
            .collect();

        let t = Instant::now();
        let specs: Vec<Spectra> = a.iter().map(|p| to_spectra(p)).collect();
        let t_spec = t.elapsed();

        let iters = 50;
        let _ = neg_and_quotient(&specs, &cols);
        let t = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(neg_and_quotient(&specs, &cols));
        }
        let t_call = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let e = Engine::get();
        let mut buf = vec![0u32; L];
        let nfwd = NHOT * (BENCH_K + 1);
        let t = Instant::now();
        for _ in 0..nfwd {
            e.plans[0].fwd(&mut buf);
        }
        let t_ntt = t.elapsed().as_secs_f64() * 1e6;

        println!("\n--- inner_full breakdown (delta={BENCH_K} columns, right operand is bits, {NHOT} primes) ---");
        println!(
            "  to_spectra x {BENCH_K} (incl. {NHOT} primes + Shoup)  {:9.1} us  ({:.1} us each)",
            t_spec.as_secs_f64() * 1e6,
            t_spec.as_secs_f64() * 1e6 / BENCH_K as f64
        );
        println!(
            "  neg_and_quotient, one call ({} threads)  {:9.1} us",
            rayon::current_num_threads(),
            t_call
        );
        println!("  reference: {nfwd} {L}-NTTs (single-threaded, sequential)  {:9.1} us", t_ntt);
        println!("  pointwise MAC volume: {NHOT} primes x {BENCH_K} cols x {L} points = {} ops\n", NHOT * BENCH_K * L);
    }
}
