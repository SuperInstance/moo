//! Basic lexer tests: tokens, states, keywords, errors.

use moo_rust::lexer::*;

fn make_number_rule() -> Rule {
    Rule {
        default_type: "NUMBER".into(),
        patterns: vec![Pattern::Regex("[0-9]+".into())],
        line_breaks: false,
        ..Rule::default()
    }
}

fn make_ident_rule() -> Rule {
    Rule {
        default_type: "IDENT".into(),
        patterns: vec![Pattern::Regex("[a-zA-Z_][a-zA-Z0-9_]*".into())],
        line_breaks: false,
        ..Rule::default()
    }
}

fn make_space_rule() -> Rule {
    Rule {
        default_type: "SPACE".into(),
        patterns: vec![Pattern::Regex("[ \\t]+".into())],
        line_breaks: false,
        ..Rule::default()
    }
}

fn make_newline_rule() -> Rule {
    Rule {
        default_type: "NL".into(),
        patterns: vec![Pattern::Literal("\\n".into())],
        line_breaks: true,
        ..Rule::default()
    }
}

#[test]
fn test_basic_number() {
    let rules = vec![make_number_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("42", None);
    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].token_type, "NUMBER");
    assert_eq!(tokens[0].value, "42");
}

#[test]
fn test_basic_multiple_tokens() {
    let rules = vec![
        make_number_rule(),
        Rule {
            default_type: "OP".into(),
            patterns: vec![Pattern::Literal("+".into())],
            ..Rule::default()
        },
        make_space_rule(),
    ];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("1 + 2", None);
    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].value, "1");
    assert_eq!(tokens[1].value, "+");
    assert_eq!(tokens[2].value, "2");
}

#[test]
fn test_single_char_fast_path() {
    let rules = vec![
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
        make_number_rule(),
    ];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("(42)", None);
    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].token_type, "LPAREN");
    assert_eq!(tokens[1].token_type, "NUMBER");
    assert_eq!(tokens[2].token_type, "RPAREN");
}

#[test]
fn test_ident() {
    let rules = vec![make_ident_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("hello_world", None);
    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens[0].token_type, "IDENT");
    assert_eq!(tokens[0].value, "hello_world");
}

#[test]
fn test_empty_input() {
    let rules = vec![make_number_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("", None);
    assert!(lexer.next_token().unwrap().is_none());
}

#[test]
fn test_token_offset() {
    let rules = vec![make_space_rule(), make_number_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("  42", None);
    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens[0].offset, 0);
    assert_eq!(tokens[1].offset, 2);
}

#[test]
fn test_token_line_col() {
    let rules = vec![
        make_number_rule(),
        Rule {
            default_type: "NL".into(),
            patterns: vec![Pattern::Regex("\\n".into())],
            line_breaks: true,
            ..Rule::default()
        },
    ];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("42\n99", None);
    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens[0].line, 1);
    assert_eq!(tokens[1].token_type, "NL");
    assert_eq!(tokens[2].line, 2);
}

#[test]
fn test_reset() {
    let rules = vec![make_number_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("42", None);
    let tok1 = lexer.next_token().unwrap();
    assert!(tok1.is_some());

    lexer.reset("99", None);
    let tok2 = lexer.next_token().unwrap();
    assert_eq!(tok2.unwrap().value, "99");
}

#[test]
fn test_save_restore() {
    let rules = vec![make_number_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("1234", None);
    let _ = lexer.next_token().unwrap(); // consume "1234"

    let saved = lexer.save();
    assert_eq!(saved.line, 1);
    assert!(saved.state.is_some());
}

#[test]
fn test_clone_lexer() {
    let rules = vec![make_number_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    let cloned = lexer.clone_lexer();
    // Cloned lexer is independent
    lexer.reset("42", None);
    assert!(lexer.next_token().unwrap().is_some());
}

#[test]
fn test_has_always_true() {
    let rules = vec![make_number_rule()];
    let lexer = Lexer::compile(rules).unwrap();
    assert!(lexer.has("NUMBER"));
    assert!(lexer.has("NONEXISTENT"));
}

#[test]
fn test_format_error() {
    let rules = vec![make_number_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("42", None);
    let tok = lexer.next_token().unwrap().unwrap();
    let msg = lexer.format_error(&tok, "test error");
    assert!(msg.contains("test error"));
}

#[test]
fn test_iterator() {
    let rules = vec![make_number_rule(), make_space_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("1 2 3", None);
    let tokens: Vec<Result<Token, CompileError>> = lexer.collect();
    assert_eq!(tokens.len(), 5);
    assert!(tokens.iter().all(|t| t.is_ok()));
}

#[test]
fn test_keywords() {
    use std::collections::HashMap;
    let kw_map: HashMap<String, Vec<String>> = {
        let mut m = HashMap::new();
        m.insert("KEYWORD".into(), vec!["if".into(), "else".into()]);
        m.insert("BOOL".into(), vec!["true".into(), "false".into()]);
        m
    };
    let kw_fn = keywords(kw_map);
    assert_eq!(kw_fn("if"), Some("KEYWORD".to_string()));
    assert_eq!(kw_fn("else"), Some("KEYWORD".to_string()));
    assert_eq!(kw_fn("true"), Some("BOOL".to_string()));
    assert_eq!(kw_fn("unknown"), None);
}

#[test]
fn test_error_on_no_match() {
    let rules = vec![make_number_rule()];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("abc", None);
    // Should get an error token (moo doesn't throw unless error: true)
    let result = lexer.next_token();
    // The default error rule has should_throw=true, so this should be an error
    assert!(result.is_err());
}

#[test]
fn test_multi_state_basic() {
    let mut state_map = std::collections::HashMap::new();

    // "main" state: matches numbers, pushes to "string" on quote
    state_map.insert(
        "main".to_string(),
        vec![
            Rule {
                default_type: "NUMBER".into(),
                patterns: vec![Pattern::Regex("[0-9]+".into())],
                ..Rule::default()
            },
            Rule {
                default_type: "QUOTE".into(),
                patterns: vec![Pattern::Literal("\"".into())],
                push: Some("string".into()),
                ..Rule::default()
            },
        ],
    );

    // "string" state: matches chars until end quote
    state_map.insert(
        "string".to_string(),
        vec![
            Rule {
                default_type: "STRING_CONTENT".into(),
                patterns: vec![Pattern::Regex("[^\"]+".into())],
                ..Rule::default()
            },
            Rule {
                default_type: "QUOTE".into(),
                patterns: vec![Pattern::Literal("\"".into())],
                pop: true,
                ..Rule::default()
            },
        ],
    );

    let mut lexer = Lexer::states(state_map, Some("main")).unwrap();
    lexer.reset("42\"hello\"", None);

    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens[0].token_type, "NUMBER");
    assert_eq!(tokens[0].value, "42");
    assert_eq!(tokens[1].token_type, "QUOTE");
    assert_eq!(tokens[2].token_type, "STRING_CONTENT");
    assert_eq!(tokens[2].value, "hello");
    assert_eq!(tokens[3].token_type, "QUOTE");
}

#[test]
fn test_literal_escape() {
    // Test that special regex chars in literals are escaped
    let rules = vec![
        Rule {
            default_type: "DOT".into(),
            patterns: vec![Pattern::Literal(".".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "STAR".into(),
            patterns: vec![Pattern::Literal("*".into())],
            ..Rule::default()
        },
    ];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset(".", None);
    let tokens = lexer.tokenize().unwrap();
    assert_eq!(tokens[0].token_type, "DOT");
}

#[test]
fn test_ambiguous_patterns() {
    // Keywords vs identifiers: keyword rule should come first
    let rules = vec![
        Rule {
            default_type: "KEYWORD".into(),
            patterns: vec![Pattern::Literal("if".into()), Pattern::Literal("else".into())],
            ..Rule::default()
        },
        Rule {
            default_type: "IDENT".into(),
            patterns: vec![Pattern::Regex("[a-zA-Z_][a-zA-Z0-9_]*".into())],
            ..Rule::default()
        },
    ];
    let mut lexer = Lexer::compile(rules).unwrap();
    lexer.reset("if foo else", None);
    // Note: "if" is a literal, but IDENT regex also matches it.
    // Moo's behavior: first matching rule wins.
    let tokens = lexer.tokenize().unwrap();
    assert!(tokens.len() >= 3);
}

#[test]
fn test_error_display() {
    let err = CompileError::InvalidPattern("test".into());
    assert!(err.to_string().contains("test"));
    let err = CompileError::MissingState("foo".into());
    assert!(err.to_string().contains("foo"));
}

#[test]
fn test_token_display() {
    let tok = Token {
        token_type: "NUM".into(),
        value: "42".into(),
        text: "42".into(),
        offset: 0,
        line: 1,
        col: 1,
        line_breaks: 0,
    };
    assert_eq!(format!("{}", tok), "42");
}

#[test]
fn test_token_serialize() {
    let tok = Token {
        token_type: "NUM".into(),
        value: "42".into(),
        text: "42".into(),
        offset: 0,
        line: 1,
        col: 1,
        line_breaks: 0,
    };
    let json = serde_json::to_string(&tok).unwrap();
    assert!(json.contains("NUM"));
    assert!(json.contains("42"));
}
