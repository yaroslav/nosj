//! Direct dtoa shootout against C fpconv: identical value stream (raw
//! doubles extracted from a corpus file), per-float nanoseconds.
//! Companion C driver lives in the benchmarking notes; both sides loop
//! the same reps over the same array into a small reused buffer.

use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/floats.f64".to_string());
    let bytes = std::fs::read(&path).expect("float stream file");
    let vals: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut out = Vec::with_capacity(4 << 20);
    let mut total = 0usize;
    let reps = 50;
    let t0 = Instant::now();
    for _ in 0..reps {
        out.clear();
        for &v in &vals {
            nosj::emit::write_f64(&mut out, v);
        }
        total += out.len();
    }
    let ns = t0.elapsed().as_nanos() as f64;
    println!(
        "rust write_f64: {:.1} ns/float (checksum {total})",
        ns / (reps as f64 * vals.len() as f64)
    );
}
