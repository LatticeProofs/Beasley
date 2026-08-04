#[cfg(all(feature = "q32", feature = "q64"))]
compile_error!("q32 and q64 are mutually exclusive: use --no-default-features --features q64");
#[cfg(not(any(feature = "q32", feature = "q64")))]
compile_error!("either q32 or q64 must be selected (--features q32 or --features q64)");

pub mod bits;
pub mod keccak;
pub mod rng;
pub mod aesprg;

#[cfg(feature = "q32")]
#[path = "field.rs"]
pub mod field;
#[cfg(feature = "q64")]
#[path = "field64.rs"]
pub mod field;

#[cfg(feature = "q32")]
#[path = "ext_field.rs"]
pub mod ext_field;
#[cfg(feature = "q64")]
#[path = "ext_field2.rs"]
pub mod ext_field;

pub mod ntt;
pub mod ring;
pub mod params;
pub mod hash;
pub mod nizk1;
pub mod relation;
pub mod mle;

#[cfg(feature = "q32")]
#[path = "simd.rs"]
pub mod simd;
#[cfg(feature = "q64")]
#[path = "simd64.rs"]
pub mod simd;

pub mod sumcheck;
pub mod pcs;
pub mod transcript;
pub mod proof;
pub mod report;
