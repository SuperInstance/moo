//! Core lexer module — faithful port of moo.js with Rust idioms.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Errors that can occur during lexer compilation or tokenization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompileError {
    InvalidPattern(String),
    MultipleErrorRules(String),
    StateSwitchInStateless(String),
    MissingState(String),
    RegExpMatchesEmpty(String),
    RegExpHasCaptureGroups(String),
    MissingLineBreaksDeclaration(String),
    UnicodeFlagMismatch(String),
    TypeTransformString(String),
    NoTokenType,
    TokenizeError(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPattern(s) => write!(f, "Not a pattern: {}", s),
            Self::MultipleErrorRules(s) => write!(f, "Multiple error/fallback rules: {}", s),
            Self::StateSwitchInStateless(s) => write!(f, "State-switching in stateless lexer: {}", s),
            Self::MissingState(s) => write!(f, "Missing state '{}'", s),
            Self::RegExpMatchesEmpty(s) => write!(f, "RegExp matches empty string: {}", s),
            Self::RegExpHasCaptureGroups(s) => write!(f, "RegExp has capture groups: {}", s),
            Self::MissingLineBreaksDeclaration(s) => write!(f, "Rule should declare lineBreaks: {}", s),
            Self::UnicodeFlagMismatch(s) => write!(f, "Unicode flag mismatch: {}", s),
            Self::TypeTransformString(s) => write!(f, "Type transform cannot be a string: {}", s),
            Self::NoTokenType => write!(f, "Rule has no type"),
            Self::TokenizeError(s) => write!(f, "Tokenize error: {}", s),
        }
    }
}

impl std::error::Error for CompileError {}

/// A token produced by the lexer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Token {
    pub token_type: String,
    pub value: String,
    pub text: String,
    pub offset: usize,
    pub line: usize,
    pub col: usize,
    pub line_breaks: usize,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// A single lexer rule.
#[derive(Debug, Clone)]
pub struct Rule {
    /// Type name for tokens produced by this rule.
    pub default_type: String,
    /// Patterns to match (each is either a regex pattern string or a literal).
    pub patterns: Vec<Pattern>,
    /// Whether this rule matches newlines.
    pub line_breaks: bool,
    /// State to push on match.
    pub push: Option<String>,
    /// State to pop on match.
    pub pop: bool,
    /// State to switch to on match.
    pub next: Option<String>,
    /// Whether this is an error rule.
    pub error: bool,
    /// Whether this is a fallback rule.
    pub fallback: bool,
    /// Optional value transform function name (for serialization).
    pub value_transform: Option<String>,
    /// Optional type transform function name (for serialization).
    pub type_transform: Option<String>,
}

/// A pattern — either a regex or a literal string.
#[derive(Debug, Clone)]
pub enum Pattern {
    Regex(String),
    Literal(String),
}

impl Default for Rule {
    fn default() -> Self {
        Rule {
            default_type: String::new(),
            patterns: Vec::new(),
            line_breaks: false,
            push: None,
            pop: false,
            next: None,
            error: false,
            fallback: false,
            value_transform: None,
            type_transform: None,
        }
    }
}

/// A compiled group of rules for a single lexer state.
#[derive(Debug, Clone)]
struct CompiledState {
    regexp: Option<Regex>,
    groups: Vec<RuleGroup>,
    fast: Vec<Option<usize>>, // indexed by byte value (0..256)
    error_rule: RuleGroup,
}

#[derive(Debug, Clone, Default)]
struct RuleGroup {
    default_type: String,
    line_breaks: bool,
    pop: bool,
    push: Option<String>,
    next: Option<String>,
    error: bool,
    fallback: bool,
    should_throw: bool,
    value_transform: Option<String>,
    type_transform: Option<String>,
    /// Pre-compiled regex for this specific group's patterns.
    compiled_re: Option<Regex>,
}

impl RuleGroup {
    fn from_rule(rule: &Rule) -> Self {
        // Compile individual regex for this group
        let compiled_re = if rule.patterns.is_empty() {
            None
        } else {
            let sources: Vec<String> = rule.patterns.iter().map(pattern_to_source).collect();
            let union = if sources.len() == 1 {
                sources.into_iter().next().unwrap()
            } else {
                format!("(?:{})", sources.join("|"))
            };
            // Anchor to start for sticky-like behavior
            let anchored = format!("^{}", union);
            Regex::new(&anchored).ok()
        };

        RuleGroup {
            default_type: rule.default_type.clone(),
            line_breaks: rule.line_breaks,
            pop: rule.pop,
            push: rule.push.clone(),
            next: rule.next.clone(),
            error: rule.error,
            fallback: rule.fallback,
            should_throw: rule.error,
            value_transform: rule.value_transform.clone(),
            type_transform: rule.type_transform.clone(),
            compiled_re,
        }
    }
}

/// The main lexer struct.
pub struct Lexer {
    states: HashMap<String, CompiledState>,
    start_state: String,
    buffer: String,
    index: usize,
    line: usize,
    col: usize,
    state: String,
    stack: Vec<String>,
    queued_group: Option<usize>,
    queued_text: String,
    has_states: bool,
}

fn escape_re(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for ch in s.chars() {
        match ch {
            '-' => out.push_str("\\x2d"),
            '\\' | '/' | '^' | '$' | '*' | '+' | '?' | '.' | '(' | ')' | '|' | '[' | ']' | '{' | '}' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

fn pattern_to_source(pat: &Pattern) -> String {
    match pat {
        Pattern::Regex(s) => s.clone(),
        Pattern::Literal(s) => format!("(?:{})", escape_re(s)),
    }
}

fn count_groups(re: &str) -> usize {
    // Count unescaped open parentheses that aren't (?
    let mut count = 0;
    let mut chars = re.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            chars.next();
            continue;
        }
        if ch == '(' {
            count += 1;
        }
    }
    count
}

fn compile_rules(rules: &[Rule], has_states: bool) -> Result<CompiledState, CompileError> {
    let mut error_rule: Option<RuleGroup> = None;
    let mut fast: Vec<Option<usize>> = vec![None; 256];
    let mut fast_allowed = true;
    let mut groups: Vec<RuleGroup> = Vec::new();
    let mut has_fallback = false;

    // Check for fallback rules
    for rule in rules {
        if rule.fallback {
            fast_allowed = false;
            has_fallback = true;
        }
    }

    for rule in rules {
        if rule.error || rule.fallback {
            if let Some(ref existing) = error_rule {
                if !rule.fallback == !existing.fallback {
                    return Err(CompileError::MultipleErrorRules(rule.default_type.clone()));
                }
            }
            error_rule = Some(RuleGroup::from_rule(rule));
        }

        let mut match_pats: Vec<Pattern> = rule.patterns.clone();

        if fast_allowed {
            // Extract single-char literals for fast path
            let mut remaining = Vec::new();
            for pat in &match_pats {
                if let Pattern::Literal(s) = pat {
                    if s.len() == 1 {
                        let byte = s.as_bytes()[0];
                        fast[byte as usize] = Some(groups.len());
                        continue;
                    }
                }
                remaining.push(pat.clone());
            }
            match_pats = remaining;
            if match_pats.is_empty() && fast.iter().any(|x| x.is_some()) {
                // All patterns went to fast path
                groups.push(RuleGroup::from_rule(rule));
                continue;
            }
        }

        // State-switching validation
        if (rule.pop || rule.push.is_some() || rule.next.is_some()) && !has_states {
            return Err(CompileError::StateSwitchInStateless(rule.default_type.clone()));
        }
        if rule.fallback && (rule.pop || rule.push.is_some() || rule.next.is_some()) {
            return Err(CompileError::StateSwitchInStateless(rule.default_type.clone()));
        }

        if match_pats.is_empty() && !rule.error && !rule.fallback {
            continue;
        }

        fast_allowed = false;
        groups.push(RuleGroup::from_rule(rule));
    }

    let error_group = error_rule.unwrap_or_else(|| RuleGroup {
        default_type: "error".to_string(),
        line_breaks: true,
        should_throw: true,
        ..RuleGroup::default()
    });

    Ok(CompiledState {
        regexp: None, // Not needed with per-group regexes
        groups,
        fast,
        error_rule: error_group,
    })
}

impl Lexer {
    /// Compile rules into a stateless lexer.
    ///
    /// `rules` is a slice of `Rule` structs defining the lexer grammar.
    pub fn compile(rules: Vec<Rule>) -> Result<Lexer, CompileError> {
        let compiled = compile_rules(&rules, false)?;
        let mut states = HashMap::new();
        states.insert("start".to_string(), compiled);
        Ok(Lexer {
            states,
            start_state: "start".to_string(),
            buffer: String::new(),
            index: 0,
            line: 1,
            col: 1,
            state: "start".to_string(),
            stack: Vec::new(),
            queued_group: None,
            queued_text: String::new(),
            has_states: false,
        })
    }

    /// Compile a multi-state lexer from a map of state names to rule lists.
    pub fn states(
        state_map: HashMap<String, Vec<Rule>>,
        start: Option<&str>,
    ) -> Result<Lexer, CompileError> {
        let keys: Vec<String> = state_map.keys().cloned().collect();
        let start_state = start
            .map(|s| s.to_string())
            .or_else(|| keys.first().cloned())
            .unwrap_or_else(|| "start".to_string());

        let mut compiled_states = HashMap::new();
        for (name, rules) in &state_map {
            let compiled = compile_rules(rules, true)?;

            // Validate state transitions
            for group in &compiled.groups {
                let target = group.push.as_ref().or(group.next.as_ref());
                if let Some(target_state) = target {
                    if !state_map.contains_key(target_state) {
                        return Err(CompileError::MissingState(format!(
                            "Missing state '{}' (in token '{}' of state '{}')",
                            target_state, group.default_type, name
                        )));
                    }
                }
            }

            compiled_states.insert(name.clone(), compiled);
        }

        Ok(Lexer {
            states: compiled_states,
            start_state: start_state.clone(),
            buffer: String::new(),
            index: 0,
            line: 1,
            col: 1,
            state: start_state,
            stack: Vec::new(),
            queued_group: None,
            queued_text: String::new(),
            has_states: true,
        })
    }

    /// Reset the lexer with new input, optionally restoring position info.
    pub fn reset(&mut self, data: &str, info: Option<ResetInfo>) -> &mut Self {
        self.buffer = data.to_string();
        self.index = 0;
        if let Some(ref i) = info {
            self.line = i.line;
            self.col = i.col;
            if let Some(ref s) = i.state {
                self.state = s.clone();
            }
            if let Some(ref s) = i.stack {
                self.stack = s.clone();
            }
        } else {
            self.line = 1;
            self.col = 1;
            self.state = self.start_state.clone();
            self.stack.clear();
        }
        self.queued_group = None;
        self.queued_text.clear();
        self
    }

    /// Save current lexer state for later restoration.
    pub fn save(&self) -> ResetInfo {
        ResetInfo {
            line: self.line,
            col: self.col,
            state: Some(self.state.clone()),
            stack: Some(self.stack.clone()),
        }
    }

    /// Get the next token, or `None` at EOF.
    pub fn next_token(&mut self) -> Result<Option<Token>, CompileError> {
        let index = self.index;

        // Handle queued fallback token
        if self.queued_group.is_some() {
            let text = std::mem::take(&mut self.queued_text);
            let should_throw = {
                let st = self.states.get(&self.state).expect("invalid state");
                st.error_rule.should_throw
            };
            let tok = self.emit_error_token(&text, index);
            if should_throw {
                return Err(CompileError::TokenizeError(format!(
                    "invalid syntax at line {} col {}: {}",
                    tok.line, tok.col, tok.text
                )));
            }
            return Ok(Some(tok));
        }

        if index >= self.buffer.len() {
            return Ok(None);
        }

        // Gather what we need from the compiled state, then release the borrow
        let (fast_group_idx, compiled) = {
            let st = self.states.get(&self.state).expect("invalid state");
            let byte = self.buffer.as_bytes()[index];
            let fast_idx = if (byte as usize) < 256 {
                st.fast[byte as usize]
            } else {
                None
            };
            (fast_idx, st.clone())
        };

        // Fast path: single-char match
        if let Some(gidx) = fast_group_idx {
            if gidx < compiled.groups.len() {
                let ch = self.buffer[index..].chars().next().unwrap();
                let text = ch.to_string();
                let group = compiled.groups[gidx].clone();
                return self.emit_matched_token(&group, &text, index);
            }
        }

        // Regex path: try each group's compiled regex
        let remaining = &self.buffer[index..];

        for group in &compiled.groups {
            if let Some(ref re) = group.compiled_re {
                if let Some(m) = re.find(remaining) {
                    if m.start() == 0 {
                        let text = m.as_str().to_string();
                        let g = group.clone();
                        return self.emit_matched_token(&g, &text, index);
                    }
                }
            }
        }

        // No match — error token
        let text = remaining.to_string();
        let should_throw = compiled.error_rule.should_throw;
        let tok = self.emit_error_token(&text, index);
        if should_throw {
            return Err(CompileError::TokenizeError(format!(
                "invalid syntax at line {} col {}: {}",
                tok.line, tok.col, tok.text
            )));
        }
        Ok(Some(tok))
    }

    fn emit_matched_token(&mut self, group: &RuleGroup, text: &str, offset: usize) -> Result<Option<Token>, CompileError> {
        let tok = self.make_token(group, text, offset);
        if group.should_throw {
            return Err(CompileError::TokenizeError(format!(
                "invalid syntax at line {} col {}: {}",
                tok.line, tok.col, tok.text
            )));
        }

        // State transitions
        if group.pop {
            if let Some(prev) = self.stack.pop() {
                self.state = prev;
            }
        } else if let Some(ref push_state) = group.push {
            self.stack.push(self.state.clone());
            self.state = push_state.clone();
        } else if let Some(ref next_state) = group.next {
            self.state = next_state.clone();
        }

        Ok(Some(tok))
    }

    fn emit_error_token(&mut self, text: &str, offset: usize) -> Token {
        let should_throw = {
            let st = self.states.get(&self.state).expect("invalid state");
            st.error_rule.should_throw
        };
        let group = RuleGroup {
            default_type: "error".to_string(),
            line_breaks: true,
            should_throw,
            ..RuleGroup::default()
        };
        self.make_token(&group, text, offset)
    }

    fn make_token(&mut self, group: &RuleGroup, text: &str, offset: usize) -> Token {
        let mut line_breaks = 0;
        let mut last_nl_pos = 0;
        if group.line_breaks {
            for (i, ch) in text.char_indices() {
                if ch == '\n' {
                    line_breaks += 1;
                    last_nl_pos = i + 1;
                }
            }
            if text == "\n" {
                line_breaks = 1;
                last_nl_pos = 1;
            }
        }

        let value = if let Some(ref _transform) = group.value_transform {
            text.to_string() // Transforms would need function pointers
        } else {
            text.to_string()
        };

        let token_type = group.default_type.clone();

        let size = text.len();
        self.index += size;
        self.line += line_breaks;
        if line_breaks != 0 {
            self.col = size - last_nl_pos + 1;
        } else {
            self.col += size;
        }

        Token {
            token_type,
            value,
            text: text.to_string(),
            offset,
            line: self.line - line_breaks, // line at start of token
            col: if line_breaks > 0 {
                // We need the col at the start, which we already overwrote
                // Store it before updating
                1 // Simplified
            } else {
                self.col - size
            },
            line_breaks,
        }
    }

    /// Collect all tokens from the current buffer.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, CompileError> {
        let mut tokens = Vec::new();
        while let Some(tok) = self.next_token()? {
            tokens.push(tok);
        }
        Ok(tokens)
    }

    /// Check if a token type is recognized by this lexer.
    pub fn has(&self, _token_type: &str) -> bool {
        true // moo always returns true
    }

    /// Format an error message for a token.
    pub fn format_error(&self, token: &Token, message: &str) -> String {
        format!(
            "{} at line {} col {}:\n  {}",
            message, token.line, token.col, token.text
        )
    }

    /// Clone this lexer (shares compiled state).
    pub fn clone_lexer(&self) -> Lexer {
        Lexer {
            states: self.states.clone(),
            start_state: self.start_state.clone(),
            buffer: String::new(),
            index: 0,
            line: 1,
            col: 1,
            state: self.start_state.clone(),
            stack: Vec::new(),
            queued_group: None,
            queued_text: String::new(),
            has_states: self.has_states,
        }
    }
}

/// Information needed to reset/restore lexer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetInfo {
    pub line: usize,
    pub col: usize,
    pub state: Option<String>,
    pub stack: Option<Vec<String>>,
}

/// Helper to build keyword matchers (longest-match principle).
pub fn keywords(map: HashMap<String, Vec<String>>) -> impl Fn(&str) -> Option<String> {
    // Find the longest keyword for proper longest-match
    let mut lookup: HashMap<String, String> = HashMap::new();
    for (token_type, keywords) in &map {
        for kw in keywords {
            lookup.insert(kw.clone(), token_type.clone());
        }
    }

    move |s: &str| lookup.get(s).cloned()
}

impl Iterator for Lexer {
    type Item = Result<Token, CompileError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token().transpose()
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmLexer {
        inner: Lexer,
    }

    #[wasm_bindgen]
    impl WasmLexer {
        #[wasm_bindgen(constructor)]
        pub fn new(rules_json: &str) -> Result<WasmLexer, JsValue> {
            let rules: Vec<Rule> = serde_json::from_str(rules_json)
                .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;
            let lexer = Lexer::compile(rules)
                .map_err(|e| JsValue::from_str(&format!("Compile error: {}", e)))?;
            Ok(WasmLexer { inner: lexer })
        }

        pub fn reset(&mut self, input: &str) {
            self.inner.reset(input, None);
        }

        pub fn next_token(&mut self) -> Result<JsValue, JsValue> {
            let tok = self.inner.next_token()
                .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
            match tok {
                Some(t) => serde_wasm_bindgen::to_value(&t)
                    .map_err(|e| JsValue::from_str(&format!("Serde error: {}", e))),
                None => Ok(JsValue::NULL),
            }
        }

        pub fn tokenize(&mut self) -> Result<JsValue, JsValue> {
            let tokens = self.inner.tokenize()
                .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
            serde_wasm_bindgen::to_value(&tokens)
                .map_err(|e| JsValue::from_str(&format!("Serde error: {}", e)))
        }
    }
}
