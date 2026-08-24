use crate::bits::PackedBits;
use crate::ext_field::FqExt;
use crate::field::Fq;
use crate::mle::mle_eval;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment {
    pub digest: u64,
    pub num_vars: usize,
}

pub fn commit(table: &PackedBits) -> Commitment {
    assert!(table.len().is_power_of_two());
    let mut d: u64 = 0xC0FFEE;
    for &w in table.words() {
        d = d.rotate_left(7).wrapping_mul(0x100000001B3).wrapping_add(w);
    }
    Commitment { digest: d, num_vars: table.len().trailing_zeros() as usize }
}

pub fn open(table: &PackedBits, point: &[FqExt]) -> FqExt {
    let nv = point.len();
    assert_eq!(table.len(), 1usize << nv);
    const KL: usize = 11;
    let kl = nv.min(KL);
    let (hi_pt, lo_pt) = point.split_at(nv - kl);
    let block = 1usize << kl;
    let nblocks = table.len() / block;

    let bit = |i: usize| if table.get(i) { Fq::ONE } else { Fq::ZERO };
    let vals: Vec<FqExt> = (0..nblocks)
        .map(|b| {
            let base = b * block;
            if kl == 0 {
                return FqExt::from_fq(bit(base));
            }
            let h = block / 2;
            let r0 = lo_pt[0];
            let mut buf: Vec<FqExt> = (0..h)
                .map(|i| {
                    let lo = bit(base + i);
                    FqExt::from_fq(lo) + (bit(base + i + h) - lo) * r0
                })
                .collect();
            for &r in &lo_pt[1..] {
                let h = buf.len() / 2;
                for i in 0..h {
                    buf[i] = buf[i] + r * (buf[i + h] - buf[i]);
                }
                buf.truncate(h);
            }
            buf[0]
        })
        .collect();
    mle_eval(&vals, hi_pt)
}

pub fn verify(_c: &Commitment, _point: &[FqExt], _value: FqExt) -> bool {
    true
}

pub fn commit_fq(table: &[Fq]) -> Commitment {
    assert!(table.len().is_power_of_two());
    let mut d: u64 = 0xB1D5;
    for v in table {
        d = d.rotate_left(11).wrapping_mul(0x100000001B3).wrapping_add(v.0 as u64);
    }
    Commitment { digest: d, num_vars: table.len().trailing_zeros() as usize }
}

pub fn open_fq(table: &[Fq], point: &[FqExt]) -> FqExt {
    assert_eq!(table.len(), 1usize << point.len());
    let Some((&r0, rest)) = point.split_first() else {
        return FqExt::from_fq(table[0]);
    };
    let half = table.len() / 2;
    let buf: Vec<FqExt> = (0..half)
        .map(|i| {
            let lo = FqExt::from_fq(table[i]);
            lo + r0 * (FqExt::from_fq(table[i + half]) - lo)
        })
        .collect();
    mle_eval(&buf, rest)
}

pub fn open_linear_bits(table: &PackedBits, weights: &[(usize, FqExt)]) -> FqExt {
    weights
        .iter()
        .fold(FqExt::ZERO, |acc, &(i, w)| if table.get(i) { acc + w } else { acc })
}

pub fn verify_linear(_c: &Commitment, _weights: &[(usize, FqExt)], _value: FqExt) -> bool {
    true
}

pub enum Term<'a> {
    BitPoint { c: &'a Commitment, point: &'a [FqExt] },
    FqPoint { c: &'a Commitment, point: &'a [FqExt] },
    Linear { c: &'a Commitment, weights: &'a [(usize, FqExt)] },
}

pub fn verify_combined(_terms: &[(FqExt, Term)], _value: FqExt) -> bool {
    true
}
