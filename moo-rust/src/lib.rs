//! # moo-rust: Spectral-optimized tokenizer
//!
//! A Rust port of the moo JavaScript tokenizer, augmented with spectral
//! optimization (Fiedler-vector state ordering) and conservation tracking.

pub mod lexer;
pub mod spectral;
pub mod conservation;
pub mod perf;

pub use lexer::{Lexer, Token, Rule, CompileError};
pub use spectral::{SpectralOptimizer, SpectralReport};
pub use conservation::{ConservationTracker, ConservationReport};
