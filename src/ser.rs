use crate::ext_field::{FqExt, EXT_DEG};
use crate::field::{fq_le_bytes, Fq, FQ_BYTES};
use crate::pcs::Commitment;
use crate::proof::Proof;
use crate::sumcheck::SumcheckProof;

pub const FQ4_BYTES: usize = EXT_DEG * FQ_BYTES;

const LEN_BYTES: usize = 4;

const DIGEST_BYTES: usize = 8;

const NUM_VARS_BYTES: usize = 4;

const N_PUB_SCALARS: usize = 7 + 3 + 4 + 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProofBytes {
    pub sumcheck: usize,
    pub public_scalars: usize,
    pub framing: usize,
    pub commitments_stub: usize,
}

impl ProofBytes {
    pub fn transcript(&self) -> usize {
        self.sumcheck + self.public_scalars
    }

    pub fn total(&self) -> usize {
        self.sumcheck + self.public_scalars + self.framing + self.commitments_stub
    }
}

struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { out: Vec::new() }
    }
    fn u32(&mut self, v: usize) {
        self.out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn fq4(&mut self, v: FqExt) {
        for c in v.coeffs() {
            self.out.extend_from_slice(&fq_le_bytes(*c));
        }
    }
    fn sumcheck(&mut self, sc: &SumcheckProof) {
        self.u32(sc.rounds.len());
        for r in &sc.rounds {
            self.u32(r.len());
            for &v in r {
                self.fq4(v);
            }
        }
    }
    fn commitment(&mut self, c: &Commitment) {
        self.u64(c.digest);
        self.u32(c.num_vars);
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, at: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let s = self.buf.get(self.at..end)?;
        self.at = end;
        Some(s)
    }
    fn u32(&mut self) -> Option<usize> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    }
    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Some(u64::from_le_bytes(a))
    }
    fn fq(&mut self) -> Option<Fq> {
        let b = self.take(FQ_BYTES)?;
        let mut a = [0u8; 8];
        a[..FQ_BYTES].copy_from_slice(b);
        let v = u64::from_le_bytes(a);
        if v >= crate::field::Q {
            return None;
        }
        Some(Fq(v))
    }
    fn fq4(&mut self) -> Option<FqExt> {
        let mut out = [Fq::ZERO; EXT_DEG];
        for o in out.iter_mut() {
            *o = self.fq()?;
        }
        Some(FqExt::from_fn(|i| out[i]))
    }
    fn sumcheck(&mut self, cap: usize) -> Option<SumcheckProof> {
        let nr = self.u32()?;
        if nr > cap {
            return None;
        }
        let mut rounds = Vec::with_capacity(nr);
        for _ in 0..nr {
            let n = self.u32()?;
            if n > cap {
                return None;
            }
            let mut r = Vec::with_capacity(n);
            for _ in 0..n {
                r.push(self.fq4()?);
            }
            rounds.push(r);
        }
        Some(SumcheckProof { rounds })
    }
    fn commitment(&mut self) -> Option<Commitment> {
        let digest = self.u64()?;
        let num_vars = self.u32()?;
        Some(Commitment { digest, num_vars })
    }
}

impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();
        for sc in [&self.sc1_bilinear, &self.sc_quotient, &self.sc_batched, &self.sc5_onehot] {
            w.sumcheck(sc);
        }
        for v in pub_scalars(self) {
            w.fq4(v);
        }
        w.commitment(&self.c_w);
        w.commitment(&self.c_t);
        w.out
    }

    pub fn from_bytes(buf: &[u8]) -> Option<Proof> {
        let cap = buf.len() / FQ4_BYTES + 1;
        let mut r = Reader::new(buf);
        let sc1_bilinear = r.sumcheck(cap)?;
        let sc_quotient = r.sumcheck(cap)?;
        let sc_batched = r.sumcheck(cap)?;
        let sc5_onehot = r.sumcheck(cap)?;
        let s1 = r.fq4()?;
        let q_claim = r.fq4()?;
        let u_final = r.fq4()?;
        let open_w = r.fq4()?;
        let open_t = r.fq4()?;
        let open_h_sc1 = r.fq4()?;
        let open_h_sum = r.fq4()?;
        let mut mask_r_evals = [FqExt::ZERO; 3];
        for v in mask_r_evals.iter_mut() {
            *v = r.fq4()?;
        }
        let mut mask_totals = [FqExt::ZERO; 4];
        for v in mask_totals.iter_mut() {
            *v = r.fq4()?;
        }
        let mut mask_evals = [FqExt::ZERO; 4];
        for v in mask_evals.iter_mut() {
            *v = r.fq4()?;
        }
        let c_w = r.commitment()?;
        let c_t = r.commitment()?;
        if r.at != buf.len() {
            return None;
        }
        Some(Proof {
            c_w,
            c_t,
            s1,
            q_claim,
            u_final,
            open_w,
            open_t,
            open_h_sc1,
            open_h_sum,
            mask_r_evals,
            sc1_bilinear,
            sc_quotient,
            sc_batched,
            sc5_onehot,
            mask_totals,
            mask_evals,
        })
    }

    pub fn size_breakdown(&self) -> ProofBytes {
        let scs = [&self.sc1_bilinear, &self.sc_quotient, &self.sc_batched, &self.sc5_onehot];
        let sc_vals: usize = scs.iter().map(|p| p.rounds.iter().map(|r| r.len()).sum::<usize>()).sum();
        let sc_rounds: usize = scs.iter().map(|p| p.rounds.len()).sum();
        ProofBytes {
            sumcheck: sc_vals * FQ4_BYTES,
            public_scalars: pub_scalars(self).len() * FQ4_BYTES,
            framing: (scs.len() + sc_rounds) * LEN_BYTES,
            commitments_stub: 2 * (DIGEST_BYTES + NUM_VARS_BYTES),
        }
    }

    pub fn num_rounds(&self) -> usize {
        [&self.sc1_bilinear, &self.sc_quotient, &self.sc_batched, &self.sc5_onehot]
            .iter()
            .map(|p| p.rounds.len())
            .sum()
    }
}

fn pub_scalars(p: &Proof) -> [FqExt; N_PUB_SCALARS] {
    [
        p.s1,
        p.q_claim,
        p.u_final,
        p.open_w,
        p.open_t,
        p.open_h_sc1,
        p.open_h_sum,
        p.mask_r_evals[0],
        p.mask_r_evals[1],
        p.mask_r_evals[2],
        p.mask_totals[0],
        p.mask_totals[1],
        p.mask_totals[2],
        p.mask_totals[3],
        p.mask_evals[0],
        p.mask_evals[1],
        p.mask_evals[2],
        p.mask_evals[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::bits_to_groups;
    use crate::nizk1::{sample_blind, Nizk1Params, QueryCounter};
    use crate::params::HashParams;
    use crate::proof::{proof_fingerprint, prove_nizk1, verify_nizk1};
    use crate::rng::insecure_test_secret;

    fn sample_proof(seed: u64) -> (HashParams, Nizk1Params, crate::nizk1::BlindStatement, Proof) {
        let params = HashParams::sample(seed, 8, 2, 1);
        let nz = Nizk1Params::sample(seed + 1, &params, 3, 2, 2);
        let bits: Vec<bool> = (0..8).map(|i| (seed >> i) & 1 == 1).collect();
        let groups = bits_to_groups(&params, &bits);
        let mut ctr = QueryCounter::new(insecure_test_secret(seed + 2));
        let bw = sample_blind(ctr.issue(), &params, &nz, &groups);
        let (st, proof) = prove_nizk1(&params, &nz, &groups, &bw);
        (params, nz, st, proof)
    }

    #[test]
    fn roundtrip_is_bit_exact_and_still_verifies() {
        let (params, nz, st, proof) = sample_proof(1300);
        let bytes = proof.to_bytes();
        let back = Proof::from_bytes(&bytes).expect("valid encoding failed to decode");
        assert_eq!(
            proof_fingerprint(&proof),
            proof_fingerprint(&back),
            "fingerprint changed after roundtrip: a field is missing from the encoding"
        );
        assert_eq!(back.to_bytes(), bytes, "re-encoding must be byte-identical");
        assert!(verify_nizk1(&params, &nz, &st, &back), "decoded proof failed to verify");
    }

    #[test]
    fn serialised_size_matches_the_breakdown() {
        for seed in [1310u64, 1311, 1312] {
            let (_, _, _, proof) = sample_proof(seed);
            let b = proof.size_breakdown();
            assert_eq!(proof.to_bytes().len(), b.total(), "seed {seed}");
            assert_eq!(b.transcript(), b.sumcheck + b.public_scalars);
            assert_eq!(FQ4_BYTES, 16);
            assert_eq!(b.public_scalars, 18 * FQ4_BYTES);
        }
    }

}
