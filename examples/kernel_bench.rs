//! Kernel micro-benchmarks for cross-ISA A/B work.
//!
//! Run natively on each target:
//!   cargo run --release --example kernel_bench                       (host)
//!   cargo run --release --example kernel_bench --target x86_64-apple-darwin
//!
//! Timings are wall-clock over fixed iteration counts with black_box
//! fencing; compare runs on the SAME machine only.

use std::hint::black_box;
use std::time::Instant;

use nosj::emit::{self, EscapeMode};
use nosj::{Buffers, Sink, Writer, parse};

struct CountSink {
    n: u64,
}

impl Sink for CountSink {
    type Error = ();
    fn null(&mut self) -> Result<(), ()> {
        self.n += 1;
        Ok(())
    }
    fn boolean(&mut self, _: bool) -> Result<(), ()> {
        self.n += 1;
        Ok(())
    }
    fn int(&mut self, _: i64) -> Result<(), ()> {
        self.n += 1;
        Ok(())
    }
    fn float(&mut self, _: f64) -> Result<(), ()> {
        self.n += 1;
        Ok(())
    }
    fn big_int(&mut self, _: &str) -> Result<(), ()> {
        self.n += 1;
        Ok(())
    }
    fn str(&mut self, s: &str) -> Result<(), ()> {
        self.n += black_box(s.len() as u64);
        Ok(())
    }
    fn key(&mut self, k: &str) -> Result<(), ()> {
        self.n += black_box(k.len() as u64);
        Ok(())
    }
    fn mark(&self) -> usize {
        self.n as usize
    }
    fn end_array(&mut self, _: usize, _: usize) -> Result<(), ()> {
        Ok(())
    }
    fn end_object(&mut self, _: usize, _: usize) -> Result<(), ()> {
        Ok(())
    }
}

/// Warmup runs one iteration per this many measured iterations.
const WARMUP_DIVISOR: u32 = 10;

fn bench(name: &str, iters: u32, mut f: impl FnMut()) {
    for _ in 0..=(iters / WARMUP_DIVISOR) {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let dt = t0.elapsed();
    println!(
        "{name:<28} {:>10.1} ns/iter",
        dt.as_nanos() as f64 / f64::from(iters)
    );
}

fn main() {
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x86_64"
    };
    #[allow(unused_mut)]
    let mut feats = String::new();
    #[cfg(target_arch = "x86_64")]
    {
        for f in ["ssse3", "avx2", "avx512bw"] {
            if std::arch::is_x86_feature_detected!("ssse3") && f == "ssse3"
                || f == "avx2" && std::arch::is_x86_feature_detected!("avx2")
                || f == "avx512bw" && std::arch::is_x86_feature_detected!("avx512bw")
            {
                feats.push_str(f);
                feats.push(' ');
            }
        }
    }
    println!("== kernel_bench on {arch} {feats}==");

    // --- escape/emit shapes ---
    let key = "created_at";
    let short = "str1234word";
    let medium = "The quick brown fox jumps over the lazy dog and keeps going.";
    let long_clean = "lorem ipsum dolor sit amet consectetur adipiscing elit ".repeat(64);
    // tolstoy shape: escape roughly every 90 bytes
    let dense = ("Съезжались к дому богатой невесты; \"сказал\" он.\n".repeat(2) + "x").repeat(40);
    let unicode = "καλημέρα κόσμε — 混沌のドキュメント 🎉".repeat(8);

    let mut out: Vec<u8> = Vec::with_capacity(1 << 20);
    for (name, s, iters) in [
        ("escape key(10B)", key, 2_000_000),
        ("escape short(11B)", short, 2_000_000),
        ("escape medium(61B)", medium, 1_000_000),
        ("escape long_clean(3.5K)", long_clean.as_str(), 100_000),
        ("escape dense_esc(~4K)", dense.as_str(), 100_000),
        ("escape unicode(~500B)", unicode.as_str(), 300_000),
    ] {
        bench(name, iters, || {
            out.clear();
            emit::escape_into(
                black_box(&mut out),
                black_box(s.as_bytes()),
                EscapeMode::Standard,
            );
            black_box(&out);
        });
    }
    bench("escape ascii_only(uni)", 200_000, || {
        out.clear();
        emit::escape_into(
            black_box(&mut out),
            black_box(unicode.as_bytes()),
            EscapeMode::AsciiOnly,
        );
        black_box(&out);
    });

    // --- numbers ---
    bench("write_f64 mixed", 1_000_000, || {
        out.clear();
        for f in [
            1.5f64,
            65.61361694335938,
            1e15,
            0.3333333333333333,
            -61.14917,
        ] {
            emit::write_f64(black_box(&mut out), black_box(f));
        }
        black_box(&out);
    });
    bench("write_i64 mixed", 1_000_000, || {
        out.clear();
        for i in [7i64, 4242, -1234567, 1466583266021785600] {
            emit::write_i64(black_box(&mut out), black_box(i));
        }
        black_box(&out);
    });

    // --- writer document shape (object-heavy) ---
    bench("writer 32-pair object", 200_000, || {
        out.clear();
        let mut w = Writer::compact(&mut out);
        w.begin_object();
        for i in 0..32 {
            w.key(black_box(["id", "name", "count", "flag"][i & 3]));
            w.int(i as i64);
        }
        w.end_object();
        black_box(&out);
    });

    // --- parse shapes (cursor + counting sink) ---
    let doc_strings = format!(
        "[{}]",
        (0..256)
            .map(|i| format!("\"value string number {i}\""))
            .collect::<Vec<_>>()
            .join(",")
    );
    let doc_numbers = format!(
        "[{}]",
        (0..256)
            .map(|i| format!("{}.{}", i * 37, i * 7919 % 100000))
            .collect::<Vec<_>>()
            .join(",")
    );
    let doc_object = format!(
        "{{{}}}",
        (0..128)
            .map(|i| format!("\"key_{i}\":{i}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut bufs = Buffers::new();
    for (name, doc, iters) in [
        ("cursor 256 strings", &doc_strings, 200_000),
        ("cursor 256 numbers", &doc_numbers, 200_000),
        ("cursor 128-pair object", &doc_object, 200_000),
    ] {
        bench(name, iters, || {
            let mut sink = CountSink { n: 0 };
            parse(black_box(doc.as_str()), &mut bufs, &mut sink).unwrap();
            black_box(sink.n);
        });
    }
}
