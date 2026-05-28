//! Spectral optimization tests.

use moo_rust::lexer::{Pattern, Rule};
use moo_rust::spectral::SpectralOptimizer;

fn make_test_rules() -> Vec<Rule> {
    vec![
        Rule { default_type: "NUMBER".into(), patterns: vec![Pattern::Regex("[0-9]+".into())], ..Rule::default() },
        Rule { default_type: "IDENT".into(), patterns: vec![Pattern::Regex("[a-zA-Z_][a-zA-Z0-9_]*".into())], ..Rule::default() },
        Rule { default_type: "OP".into(), patterns: vec![Pattern::Literal("+".into())], ..Rule::default() },
        Rule { default_type: "LPAREN".into(), patterns: vec![Pattern::Literal("(".into())], ..Rule::default() },
        Rule { default_type: "RPAREN".into(), patterns: vec![Pattern::Literal(")".into())], ..Rule::default() },
        Rule { default_type: "STRING".into(), patterns: vec![Pattern::Regex("\"[^\"]*\"".into())], line_breaks: false, ..Rule::default() },
        Rule { default_type: "SPACE".into(), patterns: vec![Pattern::Regex("[ \\t]+".into())], ..Rule::default() },
        Rule { default_type: "NL".into(), patterns: vec![Pattern::Regex("\\n".into())], line_breaks: true, ..Rule::default() },
    ]
}

#[test]
fn test_adjacency_diagonal_zero() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    for i in 0..adj.len() {
        assert_eq!(adj[i][i], 0.0);
    }
}

#[test]
fn test_adjacency_nonnegative() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    for row in &adj {
        for &val in row {
            assert!(val >= 0.0);
        }
    }
}

#[test]
fn test_adjacency_symmetric() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let n = adj.len();
    for i in 0..n {
        for j in 0..n {
            assert!((adj[i][j] - adj[j][i]).abs() < 1e-10, "Not symmetric at ({},{})", i, j);
        }
    }
}

#[test]
fn test_laplacian_row_sum_zero() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let lap = opt.compute_laplacian(&adj);
    for i in 0..lap.len() {
        let sum: f64 = lap[i].iter().sum();
        assert!(sum.abs() < 1e-10, "Row {} sum = {}", i, sum);
    }
}

#[test]
fn test_laplacian_diagonal_positive() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let lap = opt.compute_laplacian(&adj);
    for i in 0..lap.len() {
        assert!(lap[i][i] >= 0.0, "Diagonal at {} is {}", i, lap[i][i]);
    }
}

#[test]
fn test_laplacian_off_diagonal_nonpositive() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let lap = opt.compute_laplacian(&adj);
    for i in 0..lap.len() {
        for j in 0..lap.len() {
            if i != j {
                assert!(lap[i][j] <= 0.0 || lap[i][j].abs() < 1e-10,
                    "Off-diagonal ({},{}) = {}", i, j, lap[i][j]);
            }
        }
    }
}

#[test]
fn test_eigenvalues_first_is_zero() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let lap = opt.compute_laplacian(&adj);
    let eigenvalues = opt.compute_eigenvalues(&lap, 200);
    assert!(!eigenvalues.is_empty());
    // First eigenvalue of Laplacian should be ~0
    assert!(eigenvalues[0].abs() < 0.5, "First eigenvalue = {}", eigenvalues[0]);
}

#[test]
fn test_eigenvalues_sorted() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let lap = opt.compute_laplacian(&adj);
    let eigenvalues = opt.compute_eigenvalues(&lap, 200);
    for i in 1..eigenvalues.len() {
        assert!(eigenvalues[i] >= eigenvalues[i - 1] - 1e-10,
            "Not sorted at {}: {} >= {}", i, eigenvalues[i], eigenvalues[i-1]);
    }
}

#[test]
fn test_spectral_gap_positive() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let report = opt.analyze(200);
    assert!(report.spectral_gap >= 0.0, "Spectral gap = {}", report.spectral_gap);
}

#[test]
fn test_fiedler_vector_length() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let lap = opt.compute_laplacian(&adj);
    let fiedler = opt.compute_fiedler_vector(&lap, 200);
    assert_eq!(fiedler.len(), n);
}

#[test]
fn test_fiedler_vector_normalized() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let lap = opt.compute_laplacian(&adj);
    let fiedler = opt.compute_fiedler_vector(&lap, 200);
    let norm: f64 = fiedler.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 0.1, "Fiedler norm = {}", norm);
}

#[test]
fn test_fiedler_ordering_complete() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let lap = opt.compute_laplacian(&adj);
    let fiedler = opt.compute_fiedler_vector(&lap, 200);
    let ordering = opt.fiedler_ordering(&fiedler);
    assert_eq!(ordering.len(), n);
    // Should be a permutation
    let mut sorted = ordering.clone();
    sorted.sort();
    assert_eq!(sorted, (0..n).collect::<Vec<_>>());
}

#[test]
fn test_optimal_ordering() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let report = opt.analyze(200);
    assert_eq!(report.optimal_ordering.len(), 8);
    // All types should be present
    assert!(report.optimal_ordering.contains(&"NUMBER".to_string()));
    assert!(report.optimal_ordering.contains(&"IDENT".to_string()));
}

#[test]
fn test_rule_tensions() {
    let rules = make_test_rules();
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let tensions = opt.compute_rule_tensions();
    assert_eq!(tensions.len(), n - 1);
    for t in &tensions {
        assert!(*t >= 0.0 && *t <= 1.0, "Tension out of range: {}", t);
    }
}

#[test]
fn test_rule_tension_self_overlap() {
    let rules = vec![
        Rule { default_type: "A".into(), patterns: vec![Pattern::Literal("x".into())], ..Rule::default() },
        Rule { default_type: "B".into(), patterns: vec![Pattern::Literal("x".into())], ..Rule::default() },
    ];
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let tensions = opt.compute_rule_tensions();
    assert_eq!(tensions.len(), 1);
    // Same charset => overlap = 1.0
    assert!((tensions[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_rule_tension_disjoint() {
    let rules = vec![
        Rule { default_type: "A".into(), patterns: vec![Pattern::Literal("a".into())], ..Rule::default() },
        Rule { default_type: "B".into(), patterns: vec![Pattern::Literal("b".into())], ..Rule::default() },
    ];
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let tensions = opt.compute_rule_tensions();
    // Disjoint => overlap = 0.0
    assert!(tensions[0].abs() < 1e-10);
}

#[test]
fn test_normalized_laplacian() {
    let rules = make_test_rules();

    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    let norm_lap = opt.compute_normalized_laplacian(&adj);
    assert_eq!(norm_lap.len(), n);
    assert_eq!(norm_lap[0].len(), n);
}

#[test]
fn test_reorder_rules() {
    let rules = make_test_rules();

    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let reordered = opt.reorder_rules(200);
    assert_eq!(reordered.len(), n);
    // All types should be present
    let types: Vec<&str> = reordered.iter().map(|r| r.default_type.as_str()).collect();
    assert!(types.contains(&"NUMBER"));
    assert!(types.contains(&"IDENT"));
}

#[test]
fn test_charset_overlap_regex() {
    let rules = vec![
        Rule { default_type: "DIGIT".into(), patterns: vec![Pattern::Regex("[0-9]".into())], ..Rule::default() },
        Rule { default_type: "ALPHA".into(), patterns: vec![Pattern::Regex("[a-z]".into())], ..Rule::default() },
    ];
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let adj = opt.build_adjacency();
    // Digits and lowercase letters are disjoint
    assert_eq!(adj[0][1], 0.0);
}

#[test]
fn test_empty_rules() {
    let opt = SpectralOptimizer::new(vec![]);
    let adj = opt.build_adjacency();
    assert!(adj.is_empty());
    let eigenvalues = opt.compute_eigenvalues(&adj, 100);
    assert!(eigenvalues.is_empty());
}

#[test]
fn test_single_rule() {
    let rules = vec![
        Rule { default_type: "A".into(), patterns: vec![Pattern::Literal("x".into())], ..Rule::default() },
    ];
    let n = rules.len(); let opt = SpectralOptimizer::new(rules);
    let report = opt.analyze(100);
    assert_eq!(report.eigenvalues.len(), 1);
    assert_eq!(report.optimal_ordering, vec!["A"]);
}
