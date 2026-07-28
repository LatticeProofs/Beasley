use crate::field::{reduce64, Fq, Q};
use crate::ring::RingElem;
use rayon::prelude::*;
use std::sync::OnceLock;
use tfhe_ntt::prime32::Plan;

const N: usize = 1536;
const L: usize = 4096;
const PRIMES: [u64; 3] = [2013265921, 2147377153, 2147352577];
pub const NHOT: usize = 2;
const INV_P0_MOD_P1: u64 = 1228140770;
const INV_P0P1_MOD_P2: u64 = 407920109;
#[allow(dead_code)]
const M: u128 = 9283523221290398450993356801;

struct Engine {
    plans: [Plan; 3],
}
impl Engine {
    fn get() -> &'static Engine {
        static E: OnceLock<Engine> = OnceLock::new();
        E.get_or_init(|| Engine {
            plans: [
                Plan::try_new(L, PRIMES[0] as u32).expect("4096-NTT plan p0"),
                Plan::try_new(L, PRIMES[1] as u32).expect("4096-NTT plan p1"),
                Plan::try_new(L, PRIMES[2] as u32).expect("4096-NTT plan p2"),
            ],
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

fn fwd_binary_into(a: &[Fq], plan: &Plan, buf: &mut [u32]) {
    buf.fill(0);
    for (i, &v) in a.iter().enumerate() {
        debug_assert!(v.0 <= 1);
        buf[i] = v.0;
    }
    plan.fwd(buf);
}

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
            _ => {
                const $p: u64 = PRIMES[2];
                $body
            }
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

#[inline(always)]
fn crt2_modq(r0: u64, r1: u64) -> Fq {
    let (p0, p1) = (PRIMES[0], PRIMES[1]);
    debug_assert!(r0 < p0 && r1 < p1);
    let t1 = ((r1 + p1 - r0) % p1) * INV_P0_MOD_P1 % p1;
    Fq(reduce64(r0 + p0 * t1))
}

#[inline]
fn crt3(r0: u64, r1: u64, r2: u64) -> u128 {
    let (p0, p1, p2) = (PRIMES[0], PRIMES[1], PRIMES[2]);
    let t1 = ((r1 + p1 - r0 % p1) % p1) * INV_P0_MOD_P1 % p1;
    let x = r0 as u128 + p0 as u128 * t1 as u128;
    let xp2 = (x % p2 as u128) as u64;
    let t2 = ((r2 + p2 - xp2) % p2) * INV_P0P1_MOD_P2 % p2;
    x + (p0 as u128 * p1 as u128) * t2 as u128
}
#[inline]
fn crt3_modq(r0: u64, r1: u64, r2: u64) -> Fq {
    Fq((crt3(r0, r1, r2) % Q as u128) as u32)
}

pub struct Spectra {
    fwd: [Vec<u32>; NHOT],
    shoup: [Vec<u32>; NHOT],
}

pub fn to_spectra(a: &[Fq]) -> Spectra {
    let e = Engine::get();
    let mut fwd: [Vec<u32>; NHOT] = Default::default();
    let mut shoup: [Vec<u32>; NHOT] = Default::default();
    let mut buf = vec![0u32; L];
    for pi in 0..NHOT {
        fwd_into(a, PRIMES[pi], &e.plans[pi], &mut buf);
        shoup[pi] = with_prime!(pi, P, { buf.iter().map(|&w| shoup_pre(w, P)).collect() });
        fwd[pi] = buf.clone();
    }
    Spectra { fwd, shoup }
}

fn inner_full_binary(specs: &[Spectra], cols: &[RingElem]) -> Vec<Fq> {
    let e = Engine::get();
    assert!(cols.len() <= specs.len());
    debug_assert!(
        cols.iter().all(|c| c.c.iter().all(|v| v.0 <= 1)),
        "熱路徑前提：右運算元的係數必須是 bit —— 2 質數的 RNS 界據此而定（見模組說明）"
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
                            fwd_binary_into(&col.c, plan, &mut cf);
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

    (0..2 * N - 1).map(|k| crt2_modq(res[0][k] as u64, res[1][k] as u64)).collect()
}

fn inner_full_general(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    let e = Engine::get();
    let res: Vec<Vec<u32>> = (0..3)
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
        .map(|k| crt3_modq(res[0][k] as u64, res[1][k] as u64, res[2][k] as u64))
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
fn fold_cyc(full: &[Fq]) -> Vec<Fq> {
    let mut c = vec![Fq::ZERO; N];
    for i in 0..N {
        let hi = if N + i < full.len() { full[N + i] } else { Fq::ZERO };
        c[i] = full[i] + hi;
    }
    c
}

pub fn neg_inner_product(specs: &[Spectra], cols: &[RingElem]) -> RingElem {
    RingElem { c: fold_neg(&inner_full_binary(specs, cols)) }
}

pub fn neg_and_quotient(specs: &[Spectra], cols: &[RingElem]) -> (RingElem, Vec<Fq>) {
    let full = inner_full_binary(specs, cols);
    let neg = fold_neg(&full);
    let t: Vec<Fq> = full[N..].iter().map(|&v| -v).collect();
    (RingElem { c: neg }, t)
}

pub fn full_inner_product(specs: &[Spectra], cols: &[RingElem]) -> Vec<Fq> {
    inner_full_binary(specs, cols)
}

pub fn negacyclic_mul_1536(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    fold_neg(&inner_full_general(a, b))
}
pub fn cyclic_mul_1536(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    fold_cyc(&inner_full_general(a, b))
}
pub fn full_mul_1536(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
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
                assert!(r < 2 * p, "lazy 結果必須 < 2p（p{pi}, w={w}, x={x}）");
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
    fn pad4096_paths_match_schoolbook() {
        let mut rng = SimpleRng::new(2);
        let a = random_poly(&mut rng, N);
        let b = random_poly(&mut rng, N);
        let full = schoolbook_full(&a, &b);
        let mut neg_ref = vec![Fq::ZERO; N];
        let mut cyc_ref = vec![Fq::ZERO; N];
        for k in 0..N {
            let lo = full[k];
            let hi = if N + k < full.len() { full[N + k] } else { Fq::ZERO };
            neg_ref[k] = lo - hi;
            cyc_ref[k] = lo + hi;
        }
        assert_eq!(negacyclic_mul_1536(&a, &b), neg_ref);
        assert_eq!(cyclic_mul_1536(&a, &b), cyc_ref);
        assert_eq!(full_mul_1536(&a, &b), full);
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
        assert_eq!(neg_inner_product(&specs, &cols), reduce_only(&full_ref));
    }

    #[test]
    fn worst_case_binary_rns_bound_is_exact() {
        use crate::ring::DELTA;
        let ones = RingElem { c: vec![Fq::ONE; N] };
        let maxv = vec![Fq(Q as u32 - 1); N];
        let specs: Vec<Spectra> = (0..DELTA).map(|_| to_spectra(&maxv)).collect();
        let cols: Vec<RingElem> = (0..DELTA).map(|_| ones.clone()).collect();

        let got = full_inner_product(&specs, &cols);
        let qm1 = (Q - 1) as u128;
        let mut peak = 0u128;
        for k in 0..2 * N - 1 {
            let pairs = (k.min(N - 1) - k.saturating_sub(N - 1) + 1) as u128;
            let exact = DELTA as u128 * qm1 * pairs;
            peak = peak.max(exact);
            assert_eq!(got[k].0 as u128, exact % Q as u128, "k = {k}");
        }
        assert_eq!(peak, DELTA as u128 * qm1 * N as u128);
        assert!(peak < PRIMES[0] as u128 * PRIMES[1] as u128, "2 質數的界不夠：{peak}");
        assert!(peak >> 47 > 0 && peak >> 48 == 0, "上界應在 2^47.58 附近：{peak}");

        let (neg, t) = neg_and_quotient(&specs, &cols);
        for i in 0..N {
            let hi = if N + i < 2 * N - 1 { got[N + i] } else { Fq::ZERO };
            assert_eq!(neg.c[i], got[i] - hi);
        }
        for (i, &v) in t.iter().enumerate() {
            assert_eq!(v, -got[N + i]);
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
            assert_eq!(via_binary, full_mul_1536(&a, &b));
        }
    }

    #[test]
    #[ignore]
    fn ntt_plan_microbench() {
        use std::time::Instant;
        use tfhe_ntt::prime32::Plan;
        let p = PRIMES[0] as u32;
        let plan4096 = Plan::try_new(4096, p).unwrap();
        let plan1024 = Plan::try_new(1024, p).unwrap();
        let plan2048 = Plan::try_new(2048, p).unwrap();
        let mut rng = SimpleRng::new(1);
        let src: Vec<u32> = (0..4096).map(|_| (rng.next_fq().0 as u64 % PRIMES[0]) as u32).collect();
        let iters = 3000;

        let mut buf = src.clone();
        let t = Instant::now();
        for _ in 0..iters {
            plan4096.fwd(&mut buf);
            buf[0] = buf[0].wrapping_add(0);
        }
        let t4096 = t.elapsed();

        let mut b1 = src[..1024].to_vec();
        let mut b2 = src[1024..2048].to_vec();
        let mut b3 = src[2048..3072].to_vec();
        let t = Instant::now();
        for _ in 0..iters {
            plan1024.fwd(&mut b1);
            plan1024.fwd(&mut b2);
            plan1024.fwd(&mut b3);
        }
        let t3x1024 = t.elapsed();

        let mut b = src[..2048].to_vec();
        let t = Instant::now();
        for _ in 0..iters {
            plan2048.fwd(&mut b);
        }
        let t2048 = t.elapsed();

        let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / iters as f64;
        println!("\n--- NTT forward 微基準（{iters} 次平均，單質數）---");
        println!("  1 x 4096（現況）      {:8.3} us", per(t4096));
        println!(
            "  3 x 1024（radix-3）   {:8.3} us   ({:.0}% of 4096，尚未含重組層)",
            per(t3x1024),
            per(t3x1024) / per(t4096) * 100.0
        );
        println!("  1 x 2048（參考）      {:8.3} us", per(t2048));
    }

    #[test]
    #[ignore]
    fn ntt_innerfull_breakdown() {
        use crate::ring::DELTA;
        use std::time::Instant;
        let mut rng = SimpleRng::new(9);
        let a: Vec<Vec<Fq>> = (0..DELTA).map(|_| random_poly(&mut rng, N)).collect();
        let cols: Vec<RingElem> =
            (0..DELTA).map(|_| RingElem { c: random_bits(&mut rng, N) }).collect();

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
        let nfwd = NHOT * (DELTA + 1);
        let t = Instant::now();
        for _ in 0..nfwd {
            e.plans[0].fwd(&mut buf);
        }
        let t_ntt = t.elapsed().as_secs_f64() * 1e6;

        println!("\n--- inner_full 拆解（delta={DELTA} column，右運算元是 bit，{NHOT} 質數）---");
        println!(
            "  to_spectra x {DELTA}（含 {NHOT} 質數 + Shoup）  {:9.1} us  ({:.1} us / 個)",
            t_spec.as_secs_f64() * 1e6,
            t_spec.as_secs_f64() * 1e6 / DELTA as f64
        );
        println!(
            "  neg_and_quotient 一次呼叫（{} 執行緒）  {:9.1} us",
            rayon::current_num_threads(),
            t_call
        );
        println!("  參考：{nfwd} 個 4096-NTT（單執行緒序列）  {:9.1} us", t_ntt);
        println!("  逐點 MAC 規模: {NHOT} 質數 x {DELTA} col x {L} 點 = {} 次\n", NHOT * DELTA * L);
    }
}
