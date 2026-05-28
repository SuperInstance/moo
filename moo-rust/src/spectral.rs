//! Spectral optimization module.
//!
//! Builds a state-transition graph from lexer rules, computes the graph Laplacian,
//! derives the Fiedler vector, and reorders rules/states for cache-optimal matching.

use crate::lexer::Rule;
use std::collections::HashMap;

/// Report from spectral analysis.
#[derive(Debug, Clone)]
pub struct SpectralReport {
    /// Eigenvalues of the normalized Laplacian.
    pub eigenvalues: Vec<f64>,
    /// Spectral gap (second-smallest eigenvalue).
    pub spectral_gap: f64,
    /// Fiedler vector values.
    pub fiedler_vector: Vec<f64>,
    /// Optimal ordering of rule types.
    pub optimal_ordering: Vec<String>,
    /// Tension scores between adjacent rules.
    pub rule_tensions: Vec<f64>,
}

/// Spectral optimizer for lexer rules.
pub struct SpectralOptimizer {
    rules: Vec<Rule>,
    n: usize,
}

impl SpectralOptimizer {
    /// Create a new optimizer for the given rules.
    pub fn new(rules: Vec<Rule>) -> Self {
        let n = rules.len();
        SpectralOptimizer { rules, n }
    }

    /// Build an adjacency matrix from rule character-set overlaps.
    pub fn build_adjacency(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let char_sets: Vec<Vec<bool>> = self.rules.iter().map(|r| self.rule_charset(r)).collect();

        let mut adj = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let overlap = self.charset_overlap(&char_sets[i], &char_sets[j]);
                adj[i][j] = overlap;
                adj[j][i] = overlap;
            }
        }
        adj
    }

    /// Compute the degree matrix and Laplacian L = D - A.
    pub fn compute_laplacian(&self, adj: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut degree = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                degree[i] += adj[i][j];
            }
        }

        let mut lap = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    lap[i][j] = degree[i];
                } else {
                    lap[i][j] = -adj[i][j];
                }
            }
        }
        lap
    }

    /// Compute normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}.
    pub fn compute_normalized_laplacian(&self, adj: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut degree = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                degree[i] += adj[i][j];
            }
        }

        let mut d_inv_sqrt = vec![0.0f64; n];
        for i in 0..n {
            d_inv_sqrt[i] = if degree[i] > 0.0 {
                1.0 / degree[i].sqrt()
            } else {
                0.0
            };
        }

        let mut norm_lap = vec![vec![0.0f64; n]; n];
        let lap = self.compute_laplacian(adj);
        for i in 0..n {
            for j in 0..n {
                norm_lap[i][j] = d_inv_sqrt[i] * lap[i][j] * d_inv_sqrt[j];
            }
        }
        norm_lap
    }

    /// Compute eigenvalues using power iteration (for the smallest eigenvalues).
    /// Returns eigenvalues sorted in ascending order.
    pub fn compute_eigenvalues(&self, lap: &[Vec<f64>], max_iter: usize) -> Vec<f64> {
        let n = self.n;
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![0.0];
        }

        // Use QR-like approach via repeated deflation
        let mut eigenvalues = Vec::new();
        let mut current_matrix = lap.to_vec();

        for _ in 0..n {
            // Power iteration for dominant eigenvalue
            let mut v = vec![1.0f64; n];
            let norm = (n as f64).sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }

            let mut eigenval = 0.0f64;
            for _ in 0..max_iter {
                // w = M * v
                let mut w = vec![0.0f64; n];
                for i in 0..n {
                    for j in 0..n {
                        w[i] += current_matrix[i][j] * v[j];
                    }
                }

                // Rayleigh quotient
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for i in 0..n {
                    num += w[i] * v[i];
                    den += v[i] * v[i];
                }
                eigenval = num / den.max(1e-15);

                // Normalize w
                let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
                for x in w.iter_mut() {
                    *x /= w_norm;
                }
                v = w;
            }
            eigenvalues.push(eigenval);

            // Deflate: subtract eigenvalue * v * v^T
            for i in 0..n {
                for j in 0..n {
                    current_matrix[i][j] -= eigenval * v[i] * v[j];
                }
            }
        }

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }

    /// Compute the Fiedler vector (eigenvector of second-smallest eigenvalue).
    pub fn compute_fiedler_vector(&self, lap: &[Vec<f64>], max_iter: usize) -> Vec<f64> {
        let n = self.n;
        if n <= 1 {
            return vec![0.0];
        }

        // Shift to find the second eigenvalue: use inverse iteration shifted away from 0
        // First, estimate the spectral range
        let eigenvalues = self.compute_eigenvalues(lap, max_iter);
        let lambda2 = if eigenvalues.len() >= 2 {
            eigenvalues[1]
        } else {
            0.1
        };

        // For the Fiedler vector, we want the eigenvector corresponding to lambda2.
        // Use a simpler approach: orthogonalize against the constant vector and iterate.
        let mut v = vec![0.0f64; n];
        // Initialize with a non-trivial vector orthogonal to constant
        for i in 0..n {
            v[i] = (i as f64) - (n as f64 - 1.0) / 2.0;
        }

        // Orthogonalize against constant vector
        let mean = v.iter().sum::<f64>() / (n as f64);
        for x in v.iter_mut() {
            *x -= mean;
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
        for x in v.iter_mut() {
            *x /= norm;
        }

        // Power iteration on the shifted matrix
        let shift = lambda2;
        for _ in 0..max_iter {
            // w = (L - shift*I) * v  (approximation)
            let mut w = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += lap[i][j] * v[j];
                }
                w[i] -= shift * v[i];
            }

            // Orthogonalize against constant vector
            let mean = w.iter().sum::<f64>() / (n as f64);
            for x in w.iter_mut() {
                *x -= mean;
            }

            let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
            for x in w.iter_mut() {
                *x /= w_norm;
            }
            v = w;
        }

        v
    }

    /// Get the optimal ordering of rules based on the Fiedler vector.
    pub fn fiedler_ordering(&self, fiedler: &[f64]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = fiedler.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().map(|(i, _)| i).collect()
    }

    /// Compute tension between adjacent rules (character-set overlap measure).
    pub fn compute_rule_tensions(&self) -> Vec<f64> {
        let char_sets: Vec<Vec<bool>> = self.rules.iter().map(|r| self.rule_charset(r)).collect();
        let mut tensions = Vec::new();
        for i in 0..self.n.saturating_sub(1) {
            tensions.push(self.charset_overlap(&char_sets[i], &char_sets[i + 1]));
        }
        tensions
    }

    /// Run full spectral analysis and return a report.
    pub fn analyze(&self, max_iter: usize) -> SpectralReport {
        let adj = self.build_adjacency();
        let lap = self.compute_laplacian(&adj);
        let eigenvalues = self.compute_eigenvalues(&lap, max_iter);
        let spectral_gap = if eigenvalues.len() >= 2 {
            eigenvalues[1]
        } else {
            0.0
        };
        let fiedler = self.compute_fiedler_vector(&lap, max_iter);
        let ordering = self.fiedler_ordering(&fiedler);
        let optimal_ordering: Vec<String> = ordering.iter().map(|&i| self.rules[i].default_type.clone()).collect();
        let rule_tensions = self.compute_rule_tensions();

        SpectralReport {
            eigenvalues,
            spectral_gap,
            fiedler_vector: fiedler,
            optimal_ordering,
            rule_tensions,
        }
    }

    /// Reorder rules according to spectral analysis for optimal matching.
    pub fn reorder_rules(&self, max_iter: usize) -> Vec<Rule> {
        let report = self.analyze(max_iter);
        let ordering = self.fiedler_ordering(&report.fiedler_vector);
        ordering.into_iter().map(|i| self.rules[i].clone()).collect()
    }

    fn rule_charset(&self, rule: &Rule) -> Vec<bool> {
        // 128-entry ASCII charset membership
        let mut charset = [false; 128];
        for pat in &rule.patterns {
            match pat {
                crate::lexer::Pattern::Literal(s) => {
                    for ch in s.chars() {
                        let code = ch as usize;
                        if code < 128 {
                            charset[code] = true;
                        }
                    }
                }
                crate::lexer::Pattern::Regex(s) => {
                    self.parse_regex_charset(s, &mut charset);
                }
            }
        }
        charset.to_vec()
    }

    fn parse_regex_charset(&self, re: &str, charset: &mut [bool; 128]) {
        let chars: Vec<char> = re.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            match chars[i] {
                '\\' => {
                    i += 1;
                    if i < chars.len() {
                        match chars[i] {
                            'd' => {
                                for c in '0'..='9' {
                                    charset[c as usize] = true;
                                }
                            }
                            'w' => {
                                for c in 'a'..='z' {
                                    charset[c as usize] = true;
                                }
                                for c in 'A'..='Z' {
                                    charset[c as usize] = true;
                                }
                                for c in '0'..='9' {
                                    charset[c as usize] = true;
                                }
                                charset['_' as usize] = true;
                            }
                            's' => {
                                charset[' ' as usize] = true;
                                charset['\t' as usize] = true;
                                charset['\n' as usize] = true;
                                charset['\r' as usize] = true;
                            }
                            'n' => charset['\n' as usize] = true,
                            't' => charset['\t' as usize] = true,
                            'r' => charset['\r' as usize] = true,
                            c => {
                                let code = c as usize;
                                if code < 128 {
                                    charset[code] = true;
                                }
                            }
                        }
                    }
                }
                '[' => {
                    // Character class
                    i += 1;
                    let negate = i < chars.len() && chars[i] == '^';
                    if negate {
                        i += 1;
                    }
                    let mut class_chars = vec![false; 128];
                    while i < chars.len() && chars[i] != ']' {
                        if chars[i] == '\\' && i + 1 < chars.len() {
                            i += 1;
                            let code = chars[i] as usize;
                            if code < 128 {
                                class_chars[code] = true;
                            }
                        } else if i + 2 < chars.len() && chars[i + 1] == '-' && chars[i + 2] != ']' {
                            let start = chars[i] as usize;
                            let end = chars[i + 2] as usize;
                            for c in start.min(end)..=start.max(end).min(127) {
                                class_chars[c] = true;
                            }
                            i += 2;
                        } else {
                            let code = chars[i] as usize;
                            if code < 128 {
                                class_chars[code] = true;
                            }
                        }
                        i += 1;
                    }
                    if negate {
                        for v in class_chars.iter_mut() {
                            *v = !*v;
                        }
                    }
                    for j in 0..128 {
                        if class_chars[j] {
                            charset[j] = true;
                        }
                    }
                }
                '.' => {
                    // Match any (except newline) — mark all printable ASCII
                    for j in 32..127 {
                        charset[j] = true;
                    }
                }
                '(' | ')' | '+' | '*' | '?' | '{' | '}' | '|' | '^' | '$' => {
                    // Control chars, skip
                }
                c => {
                    let code = c as usize;
                    if code < 128 {
                        charset[code] = true;
                    }
                }
            }
            i += 1;
        }
    }

    fn charset_overlap(&self, a: &[bool], b: &[bool]) -> f64 {
        let intersection = a.iter().zip(b.iter()).filter(|(&x, &y)| x && y).count();
        let union = a.iter().zip(b.iter()).filter(|(&x, &y)| x || y).count();
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::{Pattern, Rule};

    #[test]
    fn test_adjacency_symmetric() {
        let rules = vec![
            Rule { default_type: "A".into(), patterns: vec![Pattern::Literal("x".into())], ..Default::default() },
            Rule { default_type: "B".into(), patterns: vec![Pattern::Literal("y".into())], ..Default::default() },
        ];
        let opt = SpectralOptimizer::new(rules);
        let adj = opt.build_adjacency();
        for i in 0..2 {
            for j in 0..2 {
                assert!((adj[i][j] - adj[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_laplacian_row_sum_zero() {
        let rules = vec![
            Rule { default_type: "A".into(), patterns: vec![Pattern::Literal("abc".into())], ..Default::default() },
            Rule { default_type: "B".into(), patterns: vec![Pattern::Literal("bcd".into())], ..Default::default() },
            Rule { default_type: "C".into(), patterns: vec![Pattern::Literal("cde".into())], ..Default::default() },
        ];
        let opt = SpectralOptimizer::new(rules);
        let adj = opt.build_adjacency();
        let lap = opt.compute_laplacian(&adj);
        for i in 0..3 {
            let row_sum: f64 = lap[i].iter().sum();
            assert!(row_sum.abs() < 1e-10, "Row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_fiedler_ordering() {
        let rules = vec![
            Rule { default_type: "LBRACKET".into(), patterns: vec![Pattern::Literal("[".into())], ..Default::default() },
            Rule { default_type: "NUMBER".into(), patterns: vec![Pattern::Regex("[0-9]+".into())], ..Default::default() },
            Rule { default_type: "IDENT".into(), patterns: vec![Pattern::Regex("[a-zA-Z_][a-zA-Z0-9_]*".into())], ..Default::default() },
        ];
        let opt = SpectralOptimizer::new(rules);
        let report = opt.analyze(100);
        assert!(!report.optimal_ordering.is_empty());
        assert_eq!(report.optimal_ordering.len(), 3);
    }
}
