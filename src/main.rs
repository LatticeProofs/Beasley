use std::time::Instant;
use voprf::hash::bits_to_groups;
use voprf::params::HashParams;
use voprf::proof::{prove, verify};
use voprf::transcript::SimpleRng;

fn main() {
    let n_bits = 128;
    let group_bits = 8;
    let ell = 1;
    println!("Proof of Hash demo: n_bits = {n_bits}, group_bits = {group_bits}, ell = {ell}");

    let t0 = Instant::now();
    let params = HashParams::sample(20260713, n_bits, group_bits, ell);
    println!("CRS precompute ({} rows): {:?}", params.table_size(), t0.elapsed());

    let mut rng = SimpleRng::new(42);
    let bits: Vec<bool> = (0..n_bits).map(|_| rng.next_bool()).collect();
    let groups = bits_to_groups(&params, &bits);

    let t0 = Instant::now();
    let (ch, proof) = prove(&params, &groups);
    println!("prove:  {:?}", t0.elapsed());

    let t1 = Instant::now();
    let ok = verify(&params, &ch, &proof);
    println!("verify: {:?} -> {}", t1.elapsed(), if ok { "ACCEPT" } else { "REJECT" });
    assert!(ok);
}
