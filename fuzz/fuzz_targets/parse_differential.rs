//! The two push parsers (stage-1 index driver and fused cursor) must
//! produce identical event streams (or the same rejection) for every
//! input. Any divergence is a bug in one of them.

#![no_main]

use libfuzzer_sys::fuzz_target;
use nosj::{Buffers, Sink, parse, parse_indexed};
use std::fmt::Write;

/// Records the event stream as a canonical string.
#[derive(Default)]
struct TraceSink {
    out: String,
    values: usize,
}

impl Sink for TraceSink {
    type Error = ();

    fn null(&mut self) -> Result<(), ()> {
        self.out.push_str("n;");
        self.bump()
    }
    fn boolean(&mut self, v: bool) -> Result<(), ()> {
        self.out.push_str(if v { "T;" } else { "F;" });
        self.bump()
    }
    fn int(&mut self, v: i64) -> Result<(), ()> {
        let _ = write!(self.out, "i{v};");
        self.bump()
    }
    fn float(&mut self, v: f64) -> Result<(), ()> {
        let _ = write!(self.out, "f{:016x};", v.to_bits());
        self.bump()
    }
    fn big_int(&mut self, d: &str) -> Result<(), ()> {
        let _ = write!(self.out, "B{d};");
        self.bump()
    }
    fn str(&mut self, s: &str) -> Result<(), ()> {
        let _ = write!(self.out, "s{s};");
        self.bump()
    }
    fn key(&mut self, k: &str) -> Result<(), ()> {
        let _ = write!(self.out, "k{k};");
        Ok(())
    }
    fn str_bytes(&mut self, s: &[u8]) -> Result<(), ()> {
        let _ = write!(self.out, "S{s:02x?};");
        self.bump()
    }
    fn key_bytes(&mut self, k: &[u8]) -> Result<(), ()> {
        let _ = write!(self.out, "K{k:02x?};");
        Ok(())
    }
    fn mark(&self) -> usize {
        self.values
    }
    fn end_array(&mut self, mark: usize, len: usize) -> Result<(), ()> {
        let _ = write!(self.out, "]{mark},{len};");
        self.values = mark;
        self.bump()
    }
    fn end_object(&mut self, mark: usize, pairs: usize) -> Result<(), ()> {
        let _ = write!(self.out, "}}{mark},{pairs};");
        self.values = mark;
        self.bump()
    }
}

impl TraceSink {
    fn bump(&mut self) -> Result<(), ()> {
        self.values += 1;
        Ok(())
    }
}

fuzz_target!(|input: &str| {
    let mut bufs = Buffers::new();

    let mut via_driver = TraceSink::default();
    let driver = parse_indexed(input, &mut bufs, &mut via_driver);

    let mut via_cursor = TraceSink::default();
    let cursor = parse(input, &mut bufs, &mut via_cursor);

    match (&driver, &cursor) {
        (Ok(()), Ok(())) => assert_eq!(
            via_driver.out, via_cursor.out,
            "driver and cursor accepted {input:?} with different events"
        ),
        (Ok(()), Err(_)) | (Err(_), Ok(())) => {
            panic!("driver={driver:?} cursor={cursor:?} disagree on {input:?}")
        }
        (Err(_), Err(_)) => {}
    }
});
