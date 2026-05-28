//! Example: tokenize a JavaScript subset using moo-rust.

use moo_rust::lexer::{Lexer, Pattern, Rule};
use moo_rust::conservation::ConservationTracker;
use moo_rust::spectral::SpectralOptimizer;

fn js_rules() -> Vec<Rule> {
    vec![
        Rule {
            default_type: "KEYWORD".into(),
            patterns: vec![
                Pattern::Literal("function".into()),
                Pattern::Literal("return".into()),
                Pattern::Literal("var".into()),
                Pattern::Literal("let".into()),
                Pattern::Literal("const".into()),
                Pattern::Literal("if".into()),
                Pattern::Literal("else".into()),
                Pattern::Literal("for".into()),
                Pattern::Literal("while".into()),
            ],
            ..Rule::default()
        },
        Rule {
            default_type: "STRING".into(),
            patterns: vec![Pattern::Regex("\"(?:[^\"\\\\]|\\\\.)*\"".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "NUMBER".into(),
            patterns: vec![Pattern::Regex("[0-9]+(?:\\.[0-9]+)?".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "IDENT".into(),
            patterns: vec![Pattern::Regex("[a-zA-Z_$][a-zA-Z0-9_$]*".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "OP".into(),
            patterns: vec![Pattern::Regex("==|!=|<=|>=|<<|>>|[+\\-*/%=<>!&|^~]".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "LPAREN".into(),
            patterns: vec![Pattern::Literal("(".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "RPAREN".into(),
            patterns: vec![Pattern::Literal(")".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "LBRACE".into(),
            patterns: vec![Pattern::Literal("{".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "RBRACE".into(),
            patterns: vec![Pattern::Literal("}".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "SEMI".into(),
            patterns: vec![Pattern::Literal(";".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "COMMA".into(),
            patterns: vec![Pattern::Literal(",".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "DOT".into(),
            patterns: vec![Pattern::Literal(".".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "COMMENT".into(),
            patterns: vec![
                Pattern::Regex("//[^\\n]*".into()),
                Pattern::Regex("/\\*[\\s\\S]*?\\*/".into()),
            ],
            line_breaks: true,
            ..Rule::default()
        },
        Rule {
            default_type: "SPACE".into(),
            patterns: vec![Pattern::Regex("[ \\t]+".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "NL".into(),
            patterns: vec![Pattern::Regex("\\n".into())],
            line_breaks: true,
            ..Rule::default()
        },
    ]
}

fn main() {
    let input = r#"function add(a, b) {
  // sum two numbers
  return a + b;
}"#;

    println!("=== JS Tokenizer Example ===\n");

    let rules = js_rules();
    let mut lexer = Lexer::compile(rules.clone()).unwrap();
    lexer.reset(input, None);

    println!("Input:\n{}\n", input);
    println!("Tokens:");

    match lexer.tokenize() {
        Ok(tokens) => {
            for tok in &tokens {
                println!("  {:12} {:20} L{}:C{}", tok.token_type, tok.value, tok.line, tok.col);
            }
            println!("\nTotal: {} tokens", tokens.len());

            // Run spectral analysis
            let opt = SpectralOptimizer::new(rules.clone());
            let report = opt.analyze(100);
            println!("\nSpectral Analysis:");
            println!("  Spectral gap:     {:.6}", report.spectral_gap);
            println!("  Optimal ordering: {:?}", report.optimal_ordering);

            // Run conservation tracking
            let types: Vec<String> = tokens.iter().map(|t| t.token_type.clone()).collect();
            let mut tracker = ConservationTracker::new(10);
            let cons_report = tracker.analyze(&types);
            println!("\nConservation Analysis:");
            println!("  Ratio:     {:.4}", cons_report.conservation_ratio);
            println!("  Mean:      {:.4}", cons_report.mean_conservation);
            println!("  Variance:  {:.6}", cons_report.gradient_variance);
            println!("  Anomalies: {}", cons_report.anomalies.len());
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
