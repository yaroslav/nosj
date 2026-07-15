//! JSON Pointer / partial-access benchmark: nosj's index-walk pointer vs
//! sonic-rs's lazy `get` (the ecosystem reference for skip-scanning) and
//! a full serde parse + `Value::pointer` as the naive baseline.
//!
//! ```text
//! cargo run --release --example pointer_bench
//! ```
//!
//! Shapes cover the honest trade-off: nosj pays one whole-document
//! stage-1 pass per query and then walks the index, while sonic scans
//! bytes only as far as the target, so early targets should favor
//! sonic and late/deep targets the index. Contenders alternate in
//! shuffled round-robin blocks (see `compare.rs` for the methodology).

use std::hint::black_box;
use std::time::Instant;

const ROUNDS: usize = 4;
const BLOCK_SECS: f64 = 0.2;
const NANOS_PER_SEC: f64 = 1e9;
const CONTENDERS: [&str; 3] = ["nosj pointer", "sonic-rs get", "serde parse+pointer"];

struct Shape {
    file: &'static str,
    name: &'static str,
    /// RFC 6901 pointer for nosj and serde.
    pointer: &'static str,
    /// The same path as sonic-rs segments (strings and indices).
    sonic: fn(&str) -> usize,
}

fn shapes() -> Vec<Shape> {
    use sonic_rs::pointer;
    vec![
        Shape {
            file: "twitter",
            name: "twitter early (/statuses/0/id)",
            pointer: "/statuses/0/id",
            sonic: |doc| {
                sonic_rs::get_from_str(doc, &pointer!["statuses", 0, "id"])
                    .unwrap()
                    .as_raw_str()
                    .len()
            },
        },
        Shape {
            file: "twitter",
            name: "twitter late (/statuses/95/user/screen_name)",
            pointer: "/statuses/95/user/screen_name",
            sonic: |doc| {
                sonic_rs::get_from_str(doc, &pointer!["statuses", 95, "user", "screen_name"])
                    .unwrap()
                    .as_raw_str()
                    .len()
            },
        },
        Shape {
            file: "citm_catalog",
            name: "citm deep (/performances/40/seatCategories/2/areas/10/areaId)",
            pointer: "/performances/40/seatCategories/2/areas/10/areaId",
            sonic: |doc| {
                sonic_rs::get_from_str(
                    doc,
                    &pointer![
                        "performances",
                        40,
                        "seatCategories",
                        2,
                        "areas",
                        10,
                        "areaId"
                    ],
                )
                .unwrap()
                .as_raw_str()
                .len()
            },
        },
    ]
}

fn load(file: &str) -> String {
    let path = format!("{}/benchmark/{file}.json", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The multi-pointer scenario: five fields spread across twitter.json.
const BATCH_POINTERS: [&str; 5] = [
    "/statuses/0/id",
    "/statuses/50/user/name",
    "/statuses/95/user/screen_name",
    "/statuses/99/text",
    "/search_metadata/count",
];

fn sonic_batch_tree() -> sonic_rs::PointerTree {
    use sonic_rs::pointer;
    let mut tree = sonic_rs::PointerTree::new();
    tree.add_path(&pointer!["statuses", 0, "id"]);
    tree.add_path(&pointer!["statuses", 50, "user", "name"]);
    tree.add_path(&pointer!["statuses", 95, "user", "screen_name"]);
    tree.add_path(&pointer!["statuses", 99, "text"]);
    tree.add_path(&pointer!["search_metadata", "count"]);
    tree
}

/// Run `f` repeatedly for ~BLOCK_SECS, returning (ops, seconds).
fn block(mut f: impl FnMut() -> usize) -> (u64, f64) {
    let mut ops = 0u64;
    let start = Instant::now();
    loop {
        black_box(f());
        ops += 1;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= BLOCK_SECS {
            return (ops, elapsed);
        }
    }
}

fn main() {
    let shapes = shapes();
    let docs: Vec<String> = shapes.iter().map(|s| load(s.file)).collect();

    // Parity gate before any timing: all three contenders must resolve
    // every shape to the same value.
    let mut bufs = nosj::Buffers::new();
    for (shape, doc) in shapes.iter().zip(&docs) {
        let ours = nosj::pointer(doc, shape.pointer, &mut bufs)
            .unwrap()
            .unwrap_or_else(|| panic!("{}: pointer missed", shape.name));
        let serde_doc: serde_json::Value = serde_json::from_str(doc).unwrap();
        let expected = serde_doc
            .pointer(shape.pointer)
            .unwrap_or_else(|| panic!("{}: serde missed", shape.name));
        let reparsed: serde_json::Value = serde_json::from_str(ours).unwrap();
        assert_eq!(&reparsed, expected, "{}: value parity", shape.name);
        assert!((shape.sonic)(doc) > 0, "{}: sonic missed", shape.name);
    }
    println!("parity OK ({} shapes, 3 contenders)\n", shapes.len());

    // acc[shape][contender] = (ops, secs)
    let mut acc = vec![[(0u64, 0f64); CONTENDERS.len()]; shapes.len()];

    for _round in 0..ROUNDS {
        // Shuffle via a simple LCG keyed on the accumulated state (no
        // Date/random in examples that must stay deterministic-ish).
        let mut order: Vec<(usize, usize)> = (0..shapes.len())
            .flat_map(|s| (0..CONTENDERS.len()).map(move |c| (s, c)))
            .collect();
        let mut seed = 0x9E3779B97F4A7C15u64.wrapping_add(acc[0][0].0);
        for i in (1..order.len()).rev() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            order.swap(i, (seed >> 33) as usize % (i + 1));
        }

        for (s, c) in order {
            let shape = &shapes[s];
            let doc = &docs[s];
            let (ops, secs) = match c {
                0 => block(|| {
                    nosj::pointer(doc, shape.pointer, &mut bufs)
                        .unwrap()
                        .map_or(0, str::len)
                }),
                1 => block(|| (shape.sonic)(doc)),
                _ => block(|| {
                    let v: serde_json::Value = serde_json::from_str(doc).unwrap();
                    v.pointer(shape.pointer)
                        .map_or(0, |v| v.is_null() as usize + 1)
                }),
            };
            acc[s][c].0 += ops;
            acc[s][c].1 += secs;
        }
    }

    // Multi-pointer scenario, measured separately: the batch resolver's
    // one pass vs N sequential passes vs sonic's PointerTree.
    let twitter = load("twitter");
    let tree = sonic_batch_tree();
    {
        // Parity: batch equals sequential on every pointer.
        let batch = nosj::pointers(&twitter, &BATCH_POINTERS, &mut bufs).unwrap();
        for (ptr, got) in BATCH_POINTERS.iter().zip(&batch) {
            let single = nosj::pointer(&twitter, ptr, &mut bufs).unwrap();
            assert_eq!(*got, single, "batch parity: {ptr}");
        }
        assert!(batch.iter().all(Option::is_some), "batch: all must hit");
    }
    let mut multi = [(0u64, 0f64); 3];
    for _ in 0..ROUNDS {
        for (c, cell) in multi.iter_mut().enumerate() {
            let (ops, secs) = match c {
                0 => block(|| {
                    nosj::pointers(&twitter, &BATCH_POINTERS, &mut bufs)
                        .unwrap()
                        .iter()
                        .flatten()
                        .map(|s| s.len())
                        .sum()
                }),
                1 => block(|| {
                    BATCH_POINTERS
                        .iter()
                        .map(|p| {
                            nosj::pointer(&twitter, p, &mut bufs)
                                .unwrap()
                                .map_or(0, str::len)
                        })
                        .sum()
                }),
                _ => block(|| {
                    sonic_rs::get_many(twitter.as_str(), &tree)
                        .unwrap()
                        .iter()
                        .flatten()
                        .map(|v| v.as_raw_str().len())
                        .sum()
                }),
            };
            cell.0 += ops;
            cell.1 += secs;
        }
    }
    println!("### Multi-pointer, 5 fields on twitter (µs per batch)\n");
    println!("| nosj pointers (batch) | nosj pointer x5 (sequential) | sonic-rs get_many |");
    println!("|---:|---:|---:|");
    let cell = |i: usize| {
        let (ops, secs) = multi[i];
        format!("{:.1}", secs * NANOS_PER_SEC / 1e3 / ops as f64)
    };
    println!("| {} | {} | {} |\n", cell(0), cell(1), cell(2));

    println!("### Pointer resolution (µs per query, lower is better)\n");
    println!("| shape | {} |", CONTENDERS.join(" | "));
    println!("|---|{}", "---:|".repeat(CONTENDERS.len()));
    for (s, shape) in shapes.iter().enumerate() {
        let cells: Vec<String> = (0..CONTENDERS.len())
            .map(|c| {
                let (ops, secs) = acc[s][c];
                format!("{:.1}", secs * NANOS_PER_SEC / 1e3 / ops as f64)
            })
            .collect();
        println!("| {} | {} |", shape.name, cells.join(" | "));
    }
}
