//! Benchmark: measure lexer throughput in tokens/sec.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use moo_rust::lexer::{Lexer, Pattern, Rule};

fn json_rules() -> Vec<Rule> {
    vec![
        Rule {
            default_type: "STRING".into(),
            patterns: vec![Pattern::regex("\"(?:[^\"\\\\]|\\\\.)*\"")],
            ..Rule::default()
        },
        Rule {
            default_type: "NUMBER".into(),
            patterns: vec![Pattern::regex("-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")],
            ..Rule::default()
        },
        Rule {
            default_type: "LBRACE".into(),
            patterns: vec![Pattern::literal("{")],
            ..Rule::default()
        },
        Rule {
            default_type: "RBRACE".into(),
            patterns: vec![Pattern::literal("}")],
            ..Rule::default()
        },
        Rule {
            default_type: "LBRACKET".into(),
            patterns: vec![Pattern::literal("[")],
            ..Rule::default()
        },
        Rule {
            default_type: "RBRACKET".into(),
            patterns: vec![Pattern::literal("]")],
            ..Rule::default()
        },
        Rule {
            default_type: "COLON".into(),
            patterns: vec![Pattern::literal(":")],
            ..Rule::default()
        },
        Rule {
            default_type: "COMMA".into(),
            patterns: vec![Pattern::literal(",")],
            ..Rule::default()
        },
        Rule {
            default_type: "SPACE".into(),
            patterns: vec![Pattern::regex("[ \\t\\n\\r]+")],
            line_breaks: true,
            ..Rule::default()
        },
    ]
}

// Helper constructors (the compile step can't handle these directly, so we build them manually)
impl Pattern {
    fn regex(s: &str) -> Self {
        Pattern::Regex(s.to_string())
    }
    fn literal(s: &str) -> Self {
        Pattern::Literal(s.to_string())
    }
}

fn generate_json(n: usize) -> String {
    let mut s = String::from("{\"items\": [");
    for i in 0..n {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{{\"id\": {}, \"name\": \"item{}\"}}", i, i));
    }
    s.push_str("]}");
    s
}

fn bench_tokenize(c: &mut Criterion) {
    let input = generate_json(100);
    let rules = json_rules();

    let mut group = c.benchmark_group("tokenize");
    group.throughput(Throughput::Bytes(input.len() as u64));
    group.bench_function("json_100_items", |b| {
        b.iter(|| {
            let mut lexer = Lexer::compile(rules.clone()).unwrap();
            lexer.reset(black_box(&input), None);
            let _ = black_box(lexer.tokenize());
        });
    });
    group.finish();
}

fn bench_compile(c: &mut Criterion) {
    let rules = json_rules();
    c.bench_function("compile_json_lexer", |b| {
        b.iter(|| {
            let _ = black_box(Lexer::compile(rules.clone()).unwrap());
        });
    });
}

fn bench_spectral(c: &mut Criterion) {
    use moo_rust::spectral::SpectralOptimizer;
    let rules = json_rules();
    c.bench_function("spectral_analyze_json", |b| {
        let opt = SpectralOptimizer::new(rules.clone());
        b.iter(|| {
            let _ = black_box(opt.analyze(100));
        });
    });
}

fn bench_conservation(c: &mut Criterion) {
    use moo_rust::conservation::ConservationTracker;
    let types: Vec<String> = (0..1000).map(|i| match i % 5 {
        0 => "STRING",
        1 => "NUMBER",
        2 => "LBRACE",
        3 => "COLON",
        _ => "COMMA",
    }.to_string()).collect();

    c.bench_function("conservation_1000_tokens", |b| {
        let mut tracker = ConservationTracker::new(50);
        let types = types.clone();
        b.iter(|| {
            tracker = ConservationTracker::new(50);
            let _ = black_box(tracker.analyze(&types));
        });
    });
}

criterion_group!(benches, bench_tokenize, bench_compile, bench_spectral, bench_conservation);
criterion_main!(benches);
