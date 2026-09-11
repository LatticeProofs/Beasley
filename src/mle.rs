
use crate::ext_field::FqExt;

pub fn mle_eval(table: &[FqExt], point: &[FqExt]) -> FqExt {
    assert_eq!(table.len(), 1 << point.len());
    let mut buf = table.to_vec();
    for &r in point {
        let half = buf.len() / 2;
        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }
    buf[0]
}

pub fn eq_table(tau: &[FqExt]) -> Vec<FqExt> {
    let mut table = vec![FqExt::ONE];
    for &t in tau.iter().rev() {
        let mut next = Vec::with_capacity(table.len() * 2);
        for &v in &table {
            next.push(v * (FqExt::ONE - t));
        }
        for &v in &table {
            next.push(v * t);
        }
        table = next;
    }
    table
}

pub fn eq_at_index(point: &[FqExt], index: usize) -> FqExt {
    let nv = point.len();
    let mut acc = FqExt::ONE;
    for (j, &p) in point.iter().enumerate() {
        let bit = (index >> (nv - 1 - j)) & 1;
        acc = acc * if bit == 1 { p } else { FqExt::ONE - p };
    }
    acc
}

pub fn eq_eval(tau: &[FqExt], r: &[FqExt]) -> FqExt {
    assert_eq!(tau.len(), r.len());
    let mut acc = FqExt::ONE;
    for (&t, &x) in tau.iter().zip(r) {
        acc = acc * (t * x + (FqExt::ONE - t) * (FqExt::ONE - x));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::SimpleRng;

    #[test]
    fn mle_agrees_on_boolean_points() {
        let mut rng = SimpleRng::new(1);
        let nv = 4;
        let table: Vec<FqExt> = (0..1 << nv).map(|_| rng.next_fq4()).collect();
        let point: Vec<FqExt> = [0u64, 1, 1, 1].iter().map(|&b| FqExt::from_u64(b)).collect();
        assert_eq!(mle_eval(&table, &point), table[7]);
    }

    #[test]
    fn eq_at_index_matches_mle_of_indicator() {
        let mut rng = SimpleRng::new(3);
        let nv = 4;
        let r: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
        for idx in [0usize, 5, 15] {
            let mut table = vec![FqExt::ZERO; 1 << nv];
            table[idx] = FqExt::ONE;
            assert_eq!(mle_eval(&table, &r), eq_at_index(&r, idx));
        }
    }

    #[test]
    fn eq_table_matches_eq_eval() {
        let mut rng = SimpleRng::new(2);
        let nv = 5;
        let tau: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
        let table = eq_table(&tau);
        let r: Vec<FqExt> = (0..nv).map(|_| rng.next_fq4()).collect();
        assert_eq!(mle_eval(&table, &r), eq_eval(&tau, &r));
    }
}
