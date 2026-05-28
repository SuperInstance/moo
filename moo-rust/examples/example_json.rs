//! Example: tokenize JSON using moo-rust.

use moo_rust::lexer::{Lexer, Pattern, Rule};

fn json_rules() -> Vec<Rule> {
    vec![
        Rule {
            default_type: "STRING".into(),
            patterns: vec![Pattern::Regex("\"(?:[^\"\\\\]|\\\\.)*\"".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "NUMBER".into(),
            patterns: vec![Pattern::Regex("-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "TRUE".into(),
            patterns: vec![Pattern::Literal("true".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "FALSE".into(),
            patterns: vec![Pattern::Literal("false".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "NULL".into(),
            patterns: vec![Pattern::Literal("null".into())],
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
            default_type: "LBRACKET".into(),
            patterns: vec![Pattern::Literal("[".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "RBRACKET".into(),
            patterns: vec![Pattern::Literal("]".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "COLON".into(),
            patterns: vec![Pattern::Literal(":".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "COMMA".into(),
            patterns: vec![Pattern::Literal(",".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "SPACE".into(),
            patterns: vec![Pattern::Regex("[ \\t\\r\\n]+".into())],
            line_breaks: true,
            ..Rule::default()
        },
    ]
}

fn main() {
    let input = r#"{"name": "Alice", "age": 30, "scores": [95, 87.5, -3], "active": true, "note": null}"#;

    let rules = json_rules();
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset(input, None);

    println!("Tokenizing JSON: {}", input);
    println!("{:-<60}", "");

    match lexer.tokenize() {
        Ok(tokens) => {
            for tok in &tokens {
                println!("{:12} | {:30} | L{}:C{}", tok.token_type, tok.value, tok.line, tok.col);
            }
            println!("{:-<60}", "");
            println!("Total tokens: {}", tokens.len());
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
