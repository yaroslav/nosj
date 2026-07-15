//! Comparison benchmark against the popular Rust JSON libraries, over the
//! corpus in `benchmark/` (the classic simdjson set plus real-world API
//! payloads).
//!
//! ```text
//! cargo run --release --example compare [file-name ...]
//! ```
//!
//! Methodology: contenders alternate in shuffled round-robin blocks across
//! several rounds (sequential per-library runs drift more than the margins
//! being measured). Parse throughput = input bytes / wall time; generate
//! throughput = output bytes / wall time. Quit Docker Desktop and other
//! heavy background work first.
//!
//! nosj is measured two ways, because it deliberately has no DOM:
//! - `nosj (events)`: `parse` into a black-boxed counting sink,
//!   the crate's design point, where the host builds its own values.
//! - `nosj (tree)`: the same drive building an owned `Value` tree
//!   (defined below), making it apples-to-apples with the DOM libraries.
//!
//! Generation: every library serializes its own parsed representation;
//! nosj serializes the `Value` tree through its `Writer`.

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use nosj::{Buffers, Sink, Writer, parse};

/// Minimal owned JSON tree for the apples-to-apples runs.
enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

/// Builds [`Value`] from the event stream via the flat value-stack protocol.
struct TreeSink {
    stack: Vec<Value>,
    keys: Vec<String>,
}

impl Sink for TreeSink {
    type Error = ();
    fn null(&mut self) -> Result<(), ()> {
        self.stack.push(Value::Null);
        Ok(())
    }
    fn boolean(&mut self, v: bool) -> Result<(), ()> {
        self.stack.push(Value::Bool(v));
        Ok(())
    }
    fn int(&mut self, v: i64) -> Result<(), ()> {
        self.stack.push(Value::Int(v));
        Ok(())
    }
    fn float(&mut self, v: f64) -> Result<(), ()> {
        self.stack.push(Value::Float(v));
        Ok(())
    }
    fn big_int(&mut self, digits: &str) -> Result<(), ()> {
        // Out of i64 range: keep the digits as text (bench corpus has none).
        self.stack.push(Value::Str(digits.to_owned()));
        Ok(())
    }
    fn str(&mut self, v: &str) -> Result<(), ()> {
        self.stack.push(Value::Str(v.to_owned()));
        Ok(())
    }
    fn key(&mut self, k: &str) -> Result<(), ()> {
        self.keys.push(k.to_owned());
        Ok(())
    }
    fn mark(&self) -> usize {
        self.stack.len()
    }
    fn end_array(&mut self, mark: usize, _len: usize) -> Result<(), ()> {
        let items = self.stack.split_off(mark);
        self.stack.push(Value::Array(items));
        Ok(())
    }
    fn end_object(&mut self, mark: usize, pairs: usize) -> Result<(), ()> {
        let values = self.stack.split_off(mark);
        let keys = self.keys.split_off(self.keys.len() - pairs);
        self.stack
            .push(Value::Object(keys.into_iter().zip(values).collect()));
        Ok(())
    }
}

/// Event-counting sink: black-boxes every payload so the traversal work
/// cannot be optimized away, allocates nothing.
struct CountSink {
    n: u64,
}

impl Sink for CountSink {
    type Error = ();
    fn null(&mut self) -> Result<(), ()> {
        self.n += 1;
        Ok(())
    }
    fn boolean(&mut self, v: bool) -> Result<(), ()> {
        self.n += u64::from(black_box(v));
        Ok(())
    }
    fn int(&mut self, v: i64) -> Result<(), ()> {
        self.n = self.n.wrapping_add(black_box(v) as u64);
        Ok(())
    }
    fn float(&mut self, v: f64) -> Result<(), ()> {
        self.n = self.n.wrapping_add(black_box(v).to_bits());
        Ok(())
    }
    fn big_int(&mut self, d: &str) -> Result<(), ()> {
        self.n += black_box(d).len() as u64;
        Ok(())
    }
    fn str(&mut self, s: &str) -> Result<(), ()> {
        self.n += black_box(s).len() as u64;
        Ok(())
    }
    fn key(&mut self, k: &str) -> Result<(), ()> {
        self.n += black_box(k).len() as u64;
        Ok(())
    }
    fn mark(&self) -> usize {
        0
    }
    fn end_array(&mut self, _: usize, _: usize) -> Result<(), ()> {
        Ok(())
    }
    fn end_object(&mut self, _: usize, _: usize) -> Result<(), ()> {
        Ok(())
    }
}

fn write_value(w: &mut Writer, v: &Value) {
    match v {
        Value::Null => w.null(),
        Value::Bool(b) => w.boolean(*b),
        Value::Int(i) => w.int(*i),
        Value::Float(f) => w.float(*f),
        Value::Str(s) => w.str(s),
        Value::Array(items) => {
            w.begin_array();
            for item in items {
                write_value(w, item);
            }
            w.end_array();
        }
        Value::Object(pairs) => {
            w.begin_object();
            for (k, v) in pairs {
                w.key(k);
                write_value(w, v);
            }
            w.end_object();
        }
    }
}

/// One measured block: run `f` repeatedly for ~`block_secs`, return
/// (elapsed seconds, iterations).
fn block(block_secs: f64, mut f: impl FnMut()) -> (f64, u32) {
    let t0 = Instant::now();
    let mut n = 0u32;
    while t0.elapsed().as_secs_f64() < block_secs {
        f();
        n += 1;
    }
    (t0.elapsed().as_secs_f64(), n)
}

struct Contender {
    name: &'static str,
    /// Returns bytes processed per iteration (input for parse, output for
    /// generate); the harness turns time + bytes into MB/s.
    parse: Option<Box<dyn FnMut() -> usize>>,
    generate: Option<Box<dyn FnMut() -> usize>>,
}

#[allow(clippy::too_many_lines)]
fn contenders(data: &str) -> Vec<Contender> {
    let input = data.to_owned();
    let len = input.len();

    let mut list = Vec::new();

    // nosj, events only (the design point: host builds its own values).
    {
        let input = input.clone();
        let mut bufs = Buffers::new();
        list.push(Contender {
            name: "nosj (events)",
            parse: Some(Box::new(move || {
                let mut sink = CountSink { n: 0 };
                parse(black_box(input.as_str()), &mut bufs, &mut sink).unwrap();
                black_box(sink.n);
                len
            })),
            generate: None,
        });
    }

    // nosj, building an owned tree + generating from it via Writer.
    {
        let parse_input = input.clone();
        let mut parse_bufs = Buffers::new();
        let mut gen_bufs = Buffers::new();
        let mut tree_sink = TreeSink {
            stack: Vec::new(),
            keys: Vec::new(),
        };
        parse(input.as_str(), &mut gen_bufs, &mut tree_sink).unwrap();
        let tree = tree_sink.stack.pop().expect("one root value");
        let mut out = Vec::with_capacity(len + GENERATE_SLACK);
        list.push(Contender {
            name: "nosj (tree)",
            parse: Some(Box::new(move || {
                let mut sink = TreeSink {
                    stack: Vec::new(),
                    keys: Vec::new(),
                };
                parse(black_box(parse_input.as_str()), &mut parse_bufs, &mut sink).unwrap();
                black_box(&sink.stack);
                len
            })),
            generate: Some(Box::new(move || {
                out.clear();
                let mut w = Writer::compact(&mut out);
                write_value(&mut w, black_box(&tree));
                black_box(&out);
                out.len()
            })),
        });
    }

    // nosj tree-generate with the opt-in shortest-float mode: same
    // machinery, zmij floats instead of the pinned fpconv format.
    #[cfg(feature = "shortest-floats")]
    {
        let mut gen_bufs = Buffers::new();
        let mut tree_sink = TreeSink {
            stack: Vec::new(),
            keys: Vec::new(),
        };
        parse(input.as_str(), &mut gen_bufs, &mut tree_sink).unwrap();
        let tree = tree_sink.stack.pop().expect("one root value");
        let mut out = Vec::with_capacity(len + GENERATE_SLACK);
        // Mutate-style construction: WriteOptions is #[non_exhaustive].
        let mut cfg = nosj::WriteOptions::default();
        cfg.float = nosj::FloatFormat::Shortest;
        list.push(Contender {
            name: "nosj (tree, shortest floats)",
            parse: None,
            generate: Some(Box::new(move || {
                out.clear();
                let mut w = Writer::new(&mut out, &cfg);
                write_value(&mut w, black_box(&tree));
                black_box(&out);
                out.len()
            })),
        });
    }

    // serde_json: Value parse + to_string.
    {
        let parse_input = input.clone();
        let value: serde_json::Value = serde_json::from_str(&input).unwrap();
        list.push(Contender {
            name: "serde_json",
            parse: Some(Box::new(move || {
                let v: serde_json::Value =
                    serde_json::from_str(black_box(parse_input.as_str())).unwrap();
                black_box(&v);
                len
            })),
            generate: Some(Box::new(move || {
                let s = serde_json::to_string(black_box(&value)).unwrap();
                let n = s.len();
                black_box(s);
                n
            })),
        });
    }

    // sonic-rs: Value parse + to_string.
    {
        let parse_input = input.clone();
        let value: sonic_rs::Value = sonic_rs::from_str(&input).unwrap();
        list.push(Contender {
            name: "sonic-rs",
            parse: Some(Box::new(move || {
                let v: sonic_rs::Value =
                    sonic_rs::from_str(black_box(parse_input.as_str())).unwrap();
                black_box(&v);
                len
            })),
            generate: Some(Box::new(move || {
                let s = sonic_rs::to_string(black_box(&value)).unwrap();
                let n = s.len();
                black_box(s);
                n
            })),
        });
    }

    // simd-json: owned Value parse (requires a mutable copy) + to_string.
    {
        let parse_bytes = input.clone().into_bytes();
        let mut scratch = parse_bytes.clone();
        let value: simd_json::OwnedValue = {
            let mut copy = parse_bytes.clone();
            simd_json::to_owned_value(&mut copy).unwrap()
        };
        list.push(Contender {
            name: "simd-json",
            parse: Some(Box::new(move || {
                scratch.copy_from_slice(black_box(&parse_bytes));
                let v = simd_json::to_owned_value(&mut scratch).unwrap();
                black_box(&v);
                len
            })),
            generate: Some(Box::new(move || {
                let s = simd_json::to_string(black_box(&value)).unwrap();
                let n = s.len();
                black_box(s);
                n
            })),
        });
    }

    // simd-json, borrowed mode: its headline fast path (zero-copy strings
    // into the mutable input buffer).
    {
        let parse_bytes = input.clone().into_bytes();
        let mut scratch = parse_bytes.clone();
        list.push(Contender {
            name: "simd-json (borrowed)",
            parse: Some(Box::new(move || {
                scratch.copy_from_slice(black_box(&parse_bytes));
                let v = simd_json::to_borrowed_value(&mut scratch).unwrap();
                black_box(&v);
                len
            })),
            generate: None,
        });
    }

    // jiter: owned JsonValue parse (no generator).
    {
        let parse_input = input.clone();
        list.push(Contender {
            name: "jiter",
            parse: Some(Box::new(move || {
                let v = jiter::JsonValue::parse(black_box(parse_input.as_bytes()), false).unwrap();
                black_box(&v);
                len
            })),
            generate: None,
        });
    }

    // json-rust: the classic pure-Rust DOM library.
    {
        let parse_input = input.clone();
        let value = json::parse(&input).unwrap();
        list.push(Contender {
            name: "json-rust",
            parse: Some(Box::new(move || {
                let v = json::parse(black_box(parse_input.as_str())).unwrap();
                black_box(&v);
                len
            })),
            generate: Some(Box::new(move || {
                let s = black_box(&value).dump();
                let n = s.len();
                black_box(s);
                n
            })),
        });
    }

    // json-steroids: zero-copy-leaning newcomer (2026).
    {
        let parse_input = input.clone();
        let value = json_steroids::parse(&input).unwrap();
        list.push(Contender {
            name: "json-steroids",
            parse: Some(Box::new(move || {
                let v = json_steroids::parse(black_box(parse_input.as_str())).unwrap();
                black_box(&v);
                len
            })),
            generate: Some(Box::new(move || {
                let s = json_steroids::to_string(black_box(&value));
                let n = s.len();
                black_box(s);
                n
            })),
        });
    }

    // asmjson: AVX-512-specialized lazy DOM; on non-x86_64 this exercises
    // its fallback path (no generator). Experimental with incomplete
    // coverage, so probe first and sit the file out on failure. Absent
    // entirely on x86_64 macOS and Windows, where its ELF assembly
    // cannot build (the dependency is target-gated to match).
    #[cfg(not(any(windows, all(target_arch = "x86_64", target_os = "macos"))))]
    if asmjson::parse_to_dom(&input, None).is_some() {
        let parse_input = input.clone();
        list.push(Contender {
            name: "asmjson",
            parse: Some(Box::new(move || {
                let dom = asmjson::parse_to_dom(black_box(parse_input.as_str()), None)
                    .expect("probed as parseable");
                black_box(&dom);
                len
            })),
            generate: None,
        });
    } else {
        eprintln!("note: asmjson failed to parse this file; skipped for it");
        list.push(Contender {
            name: "asmjson",
            parse: None,
            generate: None,
        });
    }

    list
}

/// Rounds of the rotation × per-block wall time: enough to stabilize the
/// big files without the full corpus taking more than a few minutes.
const ROUNDS: usize = 4;
const BLOCK_SECS: f64 = 0.2;

/// Throughput is reported in decimal megabytes per second.
const BYTES_PER_MB: f64 = 1_000_000.0;

/// Accumulator indices: each contender collects one (seconds, bytes) pair
/// per direction.
const DIR_PARSE: usize = 0;
const DIR_GENERATE: usize = 1;
const DIRECTIONS: usize = 2;

/// Room for the structural characters generation adds beyond the source
/// length when pre-sizing nosj's output buffer.
const GENERATE_SLACK: usize = 64;

/// (seconds, bytes) accumulators for one file: per contender, per direction.
type FileAcc = Vec<[(f64, u64); DIRECTIONS]>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut files: Vec<_> = std::fs::read_dir("benchmark")
        .expect("run from the crate root (benchmark/ not found)")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "json").then_some(p)
        })
        .collect();
    files.sort();
    if !args.is_empty() {
        files.retain(|p| {
            args.iter()
                .any(|a| p.file_stem().is_some_and(|s| s == a.as_str()))
        });
    }

    let mut results: Vec<(String, FileAcc)> = Vec::new();
    let mut names: Vec<&'static str> = Vec::new();

    for path in &files {
        let data = std::fs::read_to_string(path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let mut cs = contenders(&data);
        if names.is_empty() {
            names = cs.iter().map(|c| c.name).collect();
        }

        let mut acc = vec![[(0.0f64, 0u64); 2]; cs.len()];
        // Deterministic rotation stands in for shuffling: every contender
        // occupies every position across rounds.
        for round in 0..ROUNDS {
            for slot in 0..cs.len() {
                let idx = (slot + round) % cs.len();
                if let Some(parse) = cs[idx].parse.as_mut() {
                    let mut bytes = 0u64;
                    let (secs, _) = block(BLOCK_SECS, || bytes += parse() as u64);
                    acc[idx][DIR_PARSE].0 += secs;
                    acc[idx][DIR_PARSE].1 += bytes;
                }
                if let Some(generate) = cs[idx].generate.as_mut() {
                    let mut bytes = 0u64;
                    let (secs, _) = block(BLOCK_SECS, || bytes += generate() as u64);
                    acc[idx][DIR_GENERATE].0 += secs;
                    acc[idx][DIR_GENERATE].1 += bytes;
                }
            }
        }
        results.push((name, acc));
    }

    // One table per direction: wide contender lists read better split.
    for (direction, title) in [(DIR_PARSE, "Parse"), (DIR_GENERATE, "Generate")] {
        // Columns with no data in this direction (parse-only contenders in
        // the generate table) are dropped entirely.
        let active: Vec<usize> = (0..names.len())
            .filter(|&i| results.iter().any(|(_, acc)| acc[i][direction].0 > 0.0))
            .collect();

        println!("\n### {title} (MB/s)\n");
        let mut header = String::from("| file |");
        let mut rule = String::from("|---|");
        for &i in &active {
            let _ = write!(header, " {} |", names[i]);
            rule.push_str("---:|");
        }
        println!("{header}");
        println!("{rule}");

        for (name, acc) in &results {
            let mut row = String::new();
            let _ = write!(row, "| {name} |");
            for &i in &active {
                let (secs, bytes) = acc[i][direction];
                if secs > 0.0 {
                    let mbs = bytes as f64 / secs / BYTES_PER_MB;
                    let _ = write!(row, " {mbs:.0} |");
                } else {
                    let _ = write!(row, " — |");
                }
            }
            println!("{row}");
        }
    }
}
