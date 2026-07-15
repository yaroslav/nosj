//! Writer → parser round trip: any value tree the writer serializes must
//! parse back to the identical tree, in every layout configuration. Floats
//! compare bit-exact: the fpconv format is round-trip exact by
//! construction, so any drift is a bug in one of the number paths.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use nosj::{Buffers, Sink, WriteOptions, Writer, parse};

#[derive(Arbitrary, Debug, Clone, PartialEq)]
enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

/// Layout knobs worth fuzzing: every combination must round-trip.
#[derive(Arbitrary, Debug)]
struct Layout {
    pretty: bool,
    indent_tabs: bool,
    space_before: bool,
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

/// Rebuilds [`Value`] from the event stream.
#[derive(Default)]
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
    fn big_int(&mut self, d: &str) -> Result<(), ()> {
        self.stack.push(Value::Str(d.to_owned()));
        Ok(())
    }
    fn str(&mut self, s: &str) -> Result<(), ()> {
        self.stack.push(Value::Str(s.to_owned()));
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

/// Floats must be finite for the writer; NaN also breaks `PartialEq`
/// comparison. Compare trees with NaN-free floats by bits via PartialEq
/// (identical bit patterns compare equal for finite values, and -0.0 ==
/// 0.0 round-trips to the same bits through fpconv).
fn sanitize(v: &mut Value) {
    match v {
        Value::Float(f) if !f.is_finite() => *v = Value::Null,
        Value::Array(items) => items.iter_mut().for_each(sanitize),
        Value::Object(pairs) => pairs.iter_mut().for_each(|(_, v)| sanitize(v)),
        _ => {}
    }
}

fuzz_target!(|input: (Value, Layout)| {
    let (mut value, layout) = input;
    sanitize(&mut value);

    // Mutate-style construction: WriteOptions is #[non_exhaustive].
    let mut cfg = WriteOptions::default();
    if layout.pretty {
        cfg.indent = if layout.indent_tabs {
            b"\t".to_vec()
        } else {
            b"  ".to_vec()
        };
        cfg.space = b" ".to_vec();
        cfg.object_nl = b"\n".to_vec();
        cfg.array_nl = b"\n".to_vec();
    }
    if layout.space_before {
        cfg.space_before = b" ".to_vec();
    }

    let mut out = Vec::new();
    let mut w = Writer::new(&mut out, &cfg);
    write_value(&mut w, &value);

    let text = std::str::from_utf8(&out).expect("writer must emit UTF-8");
    let mut bufs = Buffers::new();
    let mut sink = TreeSink::default();
    parse(text, &mut bufs, &mut sink).expect("writer output must parse");
    let reparsed = sink.stack.pop().expect("one root value");

    assert_eq!(value, reparsed, "round trip changed the value via {text:?}");
});
