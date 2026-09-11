
use crate::field::Fq;
use crate::ntt::{neg_and_quotient_rows_wide, to_spectra_bin};
use crate::ring::{RingElem, N};

fn trim(a: &[Fq]) -> &[Fq] {
    let mut n = a.len();
    while n > 0 && a[n - 1] == Fq::ZERO {
        n -= 1;
    }
    &a[..n]
}

fn poly_rem(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    let b = trim(b);
    assert!(!b.is_empty(), "division by the zero polynomial");
    let mut r: Vec<Fq> = trim(a).to_vec();
    let db = b.len() - 1;
    let binv = b[db].inv();
    while r.len() > db {
        let dr = r.len() - 1;
        let coef = r[dr] * binv;
        let shift = dr - db;
        for i in 0..=db {
            r[shift + i] = r[shift + i] - coef * b[i];
        }
        let mut n = r.len();
        while n > 0 && r[n - 1] == Fq::ZERO {
            n -= 1;
        }
        r.truncate(n);
    }
    r
}

fn modulus_poly() -> Vec<Fq> {
    let mut m = vec![Fq::ZERO; N + 1];
    m[0] = Fq::ONE;
    m[N] = Fq::ONE;
    m
}

pub fn is_unit(a: &RingElem) -> bool {
    debug_assert_eq!(a.c.len(), N);
    if trim(&a.c).is_empty() {
        return false;
    }
    let mut r0 = modulus_poly();
    let mut r1 = trim(&a.c).to_vec();
    while !r1.is_empty() {
        let r2 = poly_rem(&r0, &r1);
        r0 = r1;
        r1 = r2;
    }
    trim(&r0).len() == 1
}

pub fn det_rq(mat: &[RingElem], m: usize) -> RingElem {
    assert_eq!(mat.len(), m * m, "the matrix must be m×m in row-major order");
    assert!(m <= 16, "the DP in det_rq only supports m ≤ 16");
    let mut dp: Vec<Option<RingElem>> = vec![None; 1usize << m];
    let mut one = RingElem::zero();
    one.c[0] = Fq::ONE;
    dp[0] = Some(one);

    for s in 1usize..(1 << m) {
        let k = s.count_ones() as usize;
        let row = k - 1;
        let row_neg = row % 2 == 1;
        let mut acc = RingElem::zero();
        let mut sign_pos = 0usize;
        for j in 0..m {
            if s & (1 << j) == 0 {
                continue;
            }
            let sub = s & !(1 << j);
            let minor = dp[sub].as_ref().expect("the DP order guarantees the subset has already been computed");
            let term = &mat[row * m + j] * minor;
            if row_neg ^ (sign_pos % 2 == 1) {
                acc = &acc - &term;
            } else {
                acc = &acc + &term;
            }
            sign_pos += 1;
        }
        dp[s] = Some(acc);
    }
    dp[(1 << m) - 1].take().unwrap()
}

pub fn det_rq_bin(mat: &[RingElem], m: usize) -> RingElem {
    assert_eq!(mat.len(), m * m, "the matrix must be m×m in row-major order");
    assert!(m <= 16, "the DP only supports m ≤ 16");
    debug_assert!(mat.iter().all(|e| e.is_binary()), "the input of det_rq_bin must be binary");

    let specs: Vec<_> = mat.iter().map(|e| to_spectra_bin(&e.c)).collect();

    let mut dp: Vec<Option<RingElem>> = vec![None; 1usize << m];
    let mut one = RingElem::zero();
    one.c[0] = Fq::ONE;
    dp[0] = Some(one);

    let zero = RingElem::zero();
    for s in 1usize..(1 << m) {
        let k = s.count_ones() as usize;
        let row = k - 1;
        let row_neg = row % 2 == 1;

        let mut cols: Vec<RingElem> = vec![zero.clone(); m];
        let mut sign_pos = 0usize;
        for j in 0..m {
            if s & (1 << j) == 0 {
                continue;
            }
            let minor = dp[s & !(1 << j)].as_ref().expect("the DP order guarantees the subset has already been computed");
            cols[j] = if row_neg ^ (sign_pos % 2 == 1) { -minor } else { minor.clone() };
            sign_pos += 1;
        }
        let row_specs = &specs[row * m..(row + 1) * m];
        let (red, _t) = neg_and_quotient_rows_wide(row_specs, 1, &cols).pop().unwrap();
        dp[s] = Some(red);
    }
    dp[(1 << m) - 1].take().unwrap()
}

pub fn is_invertible(mat: &[RingElem], m: usize) -> bool {
    let det =
        if mat.iter().all(|e| e.is_binary()) { det_rq_bin(mat, m) } else { det_rq(mat, m) };
    is_unit(&det)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::SimpleRng;

    fn re(rng: &mut SimpleRng) -> RingElem {
        RingElem { c: (0..N).map(|_| rng.next_fq()).collect() }
    }
    fn re_bin(rng: &mut SimpleRng) -> RingElem {
        RingElem { c: (0..N).map(|_| if rng.next_bool() { Fq::ONE } else { Fq::ZERO }).collect() }
    }

    #[test]
    fn trivial_units() {
        let mut one = RingElem::zero();
        one.c[0] = Fq::ONE;
        assert!(is_unit(&one));
        assert!(!is_unit(&RingElem::zero()));
        let mut c = RingElem::zero();
        c.c[0] = Fq::new(12345);
        assert!(is_unit(&c));
    }

    #[test]
    fn factors_of_the_modulus_are_not_units() {
        let s = Fq::new(2).pow((crate::field::Q - 1) / 4);
        assert_eq!(s * s, Fq::new(crate::field::Q - 1), "s² must be −1");

        let mut f1 = RingElem::zero();
        f1.c[N / 2] = Fq::ONE;
        f1.c[0] = -s;
        let mut f2 = RingElem::zero();
        f2.c[N / 2] = Fq::ONE;
        f2.c[0] = s;

        assert!(!is_unit(&f1), "X^{{N/2}} − s unexpectedly is a unit");
        assert!(!is_unit(&f2), "X^{{N/2}} + s unexpectedly is a unit");
        assert_eq!(&f1 * &f2, RingElem::zero(), "f1·f2 should be X^N+1 ≡ 0");
    }

    #[test]
    fn random_elements_are_units() {
        let mut rng = SimpleRng::new(7);
        for _ in 0..12 {
            assert!(is_unit(&re(&mut rng)));
            assert!(is_unit(&re_bin(&mut rng)));
        }
    }

    #[test]
    fn det_matches_closed_form() {
        let mut rng = SimpleRng::new(11);
        let m2: Vec<RingElem> = (0..4).map(|_| re(&mut rng)).collect();
        let want2 = &(&m2[0] * &m2[3]) - &(&m2[1] * &m2[2]);
        assert_eq!(det_rq(&m2, 2), want2);
        let a: Vec<RingElem> = (0..9).map(|_| re(&mut rng)).collect();
        let p = |i: usize, j: usize, k: usize| &(&a[i] * &a[j]) * &a[k];
        let want3 = &(&(&p(0, 4, 8) + &p(1, 5, 6)) + &p(2, 3, 7))
            - &(&(&p(2, 4, 6) + &p(0, 5, 7)) + &p(1, 3, 8));
        assert_eq!(det_rq(&a, 3), want3);
    }

    #[test]
    fn structural_determinants() {
        let m = 5;
        let mut one = RingElem::zero();
        one.c[0] = Fq::ONE;
        let mut id: Vec<RingElem> = (0..m * m).map(|_| RingElem::zero()).collect();
        for i in 0..m {
            id[i * m + i] = one.clone();
        }
        assert_eq!(det_rq(&id, m), one);
        assert!(is_invertible(&id, m));

        let mut rng = SimpleRng::new(13);
        let mut bad: Vec<RingElem> = (0..m * m).map(|_| re(&mut rng)).collect();
        for j in 0..m {
            bad[1 * m + j] = bad[0 * m + j].clone();
        }
        assert_eq!(det_rq(&bad, m), RingElem::zero());
        assert!(!is_invertible(&bad, m));
    }

    #[test]
    #[ignore]
    fn invertibility_breakdown() {
        use std::time::Instant;
        let m = crate::params::ELL;
        let tsz = 256usize;
        let mut rng = SimpleRng::new(1);
        let mats: Vec<Vec<RingElem>> =
            (0..tsz).map(|_| (0..m * m).map(|_| re_bin(&mut rng)).collect()).collect();

        let t = Instant::now();
        let dets_gen: Vec<RingElem> = mats.iter().map(|a| det_rq(a, m)).collect();
        let t_gen = t.elapsed();

        let t = Instant::now();
        let dets: Vec<RingElem> = mats.iter().map(|a| det_rq_bin(a, m)).collect();
        let t_det = t.elapsed();
        assert_eq!(dets, dets_gen, "the two paths disagree");

        let t = Instant::now();
        let n_ok = dets.iter().filter(|d| is_unit(d)).count();
        let t_gcd = t.elapsed();

        println!("
--- invertibility check ({tsz} {m}x{m} binary matrices, single-threaded) ---");
        println!("  det_rq     (5-prime general path): {:?}  ({:.2} ms/matrix)",
            t_gen, t_gen.as_secs_f64() * 1e3 / tsz as f64);
        println!("  det_rq_bin (3-prime hot path)    : {:?}  ({:.2} ms/matrix)  {:.2}x",
            t_det, t_det.as_secs_f64() * 1e3 / tsz as f64,
            t_gen.as_secs_f64() / t_det.as_secs_f64());
        println!("  is_unit    (gcd, O(N^2))         : {:?}  ({:.2} ms/matrix)",
            t_gcd, t_gcd.as_secs_f64() * 1e3 / tsz as f64);
        println!("  total                            : {:?}", t_det + t_gcd);
        println!("  acceptance rate                  : {n_ok}/{tsz}
");
        assert_eq!(n_ok, tsz, "the acceptance rate is not 100%");
    }

    #[test]
    fn det_bin_matches_general_path() {
        let mut rng = SimpleRng::new(0xDE7);
        for m in [2usize, 3, 5] {
            for _ in 0..3 {
                let mat: Vec<RingElem> = (0..m * m).map(|_| re_bin(&mut rng)).collect();
                assert_eq!(det_rq_bin(&mat, m), det_rq(&mat, m), "m = {m}");
            }
        }
    }

    #[test]
    fn random_binary_matrices_are_invertible() {
        let mut rng = SimpleRng::new(2024);
        let m = crate::params::ELL;
        let mut ok = 0;
        for _ in 0..8 {
            let mat: Vec<RingElem> = (0..m * m).map(|_| re_bin(&mut rng)).collect();
            if is_invertible(&mat, m) {
                ok += 1;
            }
        }
        assert_eq!(ok, 8, "the acceptance rate for random binary matrices is too low (rejection sampling would be slow)");
    }
}
