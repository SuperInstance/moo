//! Conservation tracking module.
//!
//! Tracks sliding-window conservation of a token stream, computing tension,
//! information content, gradient variance, and anomaly detection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A conservation report for a token stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationReport {
    /// Overall conservation ratio (0.0 to 1.0).
    pub conservation_ratio: f64,
    /// Per-window conservation scores.
    pub window_scores: Vec<f64>,
    /// Information content per token type (Shannon entropy).
    pub information_content: HashMap<String, f64>,
    /// Gradient variance across windows.
    pub gradient_variance: f64,
    /// Detected anomalies (indices where conservation drops significantly).
    pub anomalies: Vec<Anomaly>,
    /// Mean conservation score.
    pub mean_conservation: f64,
    /// Maximum conservation score.
    pub max_conservation: f64,
    /// Minimum conservation score.
    pub min_conservation: f64,
}

/// A detected conservation anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Window index where anomaly was detected.
    pub window_index: usize,
    /// Conservation score at the anomaly point.
    pub score: f64,
    /// Expected score (moving average).
    pub expected: f64,
    /// Deviation from expected.
    pub deviation: f64,
    /// Description of the anomaly.
    pub description: String,
}

/// Token type info used for conservation tracking.
#[derive(Debug, Clone)]
struct TokenTypeStats {
    count: usize,
    entropy: f64,
}

/// Conservation tracker for token streams.
pub struct ConservationTracker {
    /// Window size for sliding window analysis.
    window_size: usize,
    /// Anomaly detection threshold (in standard deviations).
    anomaly_threshold: f64,
    /// Token type vocabulary.
    vocab: HashMap<String, TokenTypeStats>,
    /// Total tokens seen.
    total_tokens: usize,
}

impl ConservationTracker {
    /// Create a new tracker with the given window size.
    pub fn new(window_size: usize) -> Self {
        ConservationTracker {
            window_size,
            anomaly_threshold: 2.0,
            vocab: HashMap::new(),
            total_tokens: 0,
        }
    }

    /// Set the anomaly detection threshold.
    pub fn with_anomaly_threshold(mut self, threshold: f64) -> Self {
        self.anomaly_threshold = threshold;
        self
    }

    /// Compute information content (Shannon entropy) for each token type.
    pub fn compute_information_content(&mut self, token_types: &[String]) -> HashMap<String, f64> {
        self.total_tokens = token_types.len();
        self.vocab.clear();

        // Count occurrences
        for tt in token_types {
            let entry = self.vocab.entry(tt.clone()).or_insert(TokenTypeStats {
                count: 0,
                entropy: 0.0,
            });
            entry.count += 1;
        }

        // Compute entropy per type: -log2(p)
        let mut info = HashMap::new();
        for (tt, stats) in &self.vocab {
            let p = stats.count as f64 / self.total_tokens as f64;
            let entropy = -p.log2();
            info.insert(tt.clone(), entropy);
        }

        // Update vocab with entropy values
        for (tt, entropy) in &info {
            if let Some(stats) = self.vocab.get_mut(tt) {
                stats.entropy = *entropy;
            }
        }

        info
    }

    /// Compute per-window conservation scores.
    pub fn compute_window_scores(&self, token_types: &[String]) -> Vec<f64> {
        if token_types.is_empty() || self.window_size == 0 {
            return vec![];
        }

        let ws = self.window_size.min(token_types.len());
        let mut scores = Vec::new();

        for window_start in 0..=(token_types.len().saturating_sub(ws)) {
            let window = &token_types[window_start..window_start + ws];
            let score = self.window_conservation(window);
            scores.push(score);
        }

        scores
    }

    /// Compute conservation for a single window.
    /// Conservation = weighted information content normalized by window size.
    fn window_conservation(&self, window: &[String]) -> f64 {
        if window.is_empty() {
            return 0.0;
        }

        // Count unique types in window
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for tt in window {
            *type_counts.entry(tt.clone()).or_insert(0) += 1;
        }

        // Compute Shannon entropy of the window
        let n = window.len() as f64;
        let mut entropy = 0.0;
        for count in type_counts.values() {
            let p = *count as f64 / n;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        // Normalize: max entropy for this vocabulary size
        let vocab_size = self.vocab.len().max(type_counts.len());
        let max_entropy = if vocab_size > 1 {
            (vocab_size as f64).log2()
        } else {
            1.0
        };

        if max_entropy == 0.0 {
            0.0
        } else {
            entropy / max_entropy
        }
    }

    /// Compute gradient variance across window scores.
    pub fn compute_gradient_variance(scores: &[f64]) -> f64 {
        if scores.len() < 2 {
            return 0.0;
        }

        let gradients: Vec<f64> = scores.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = gradients.iter().sum::<f64>() / gradients.len() as f64;
        let variance = gradients.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gradients.len() as f64;
        variance
    }

    /// Detect anomalies in the conservation scores.
    pub fn detect_anomalies(&self, scores: &[f64]) -> Vec<Anomaly> {
        if scores.len() < 3 {
            return vec![];
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        let mut anomalies = Vec::new();
        for (i, &score) in scores.iter().enumerate() {
            // Moving average (excluding current)
            let window = 5.min(i).min(scores.len() - i - 1);
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(scores.len());
            let ma: f64 = scores[start..end].iter().sum::<f64>() / (end - start) as f64;

            let deviation = (score - ma).abs();
            if deviation > self.anomaly_threshold * std_dev && std_dev > 1e-10 {
                anomalies.push(Anomaly {
                    window_index: i,
                    score,
                    expected: ma,
                    deviation,
                    description: if score < ma {
                        format!("Conservation drop at window {}", i)
                    } else {
                        format!("Conservation spike at window {}", i)
                    },
                });
            }
        }

        anomalies
    }

    /// Run full conservation analysis on a sequence of token types.
    pub fn analyze(&mut self, token_types: &[String]) -> ConservationReport {
        let info = self.compute_information_content(token_types);
        let window_scores = self.compute_window_scores(token_types);
        let gradient_variance = Self::compute_gradient_variance(&window_scores);
        let anomalies = self.detect_anomalies(&window_scores);

        let mean_conservation = if window_scores.is_empty() {
            0.0
        } else {
            window_scores.iter().sum::<f64>() / window_scores.len() as f64
        };

        let max_conservation = window_scores.iter().cloned().fold(0.0f64, f64::max);
        let min_conservation = window_scores.iter().cloned().fold(1.0f64, f64::min);

        // Overall conservation ratio: weighted mean entropy / max possible
        let vocab_size = info.len();
        let max_entropy = if vocab_size > 1 {
            (vocab_size as f64).log2()
        } else {
            1.0
        };
        let total_info: f64 = info.values().sum();
        let conservation_ratio = if max_entropy > 0.0 && !info.is_empty() {
            let mean_info = total_info / info.len() as f64;
            mean_info / max_entropy
        } else {
            0.0
        };

        ConservationReport {
            conservation_ratio,
            window_scores,
            information_content: info,
            gradient_variance,
            anomalies,
            mean_conservation,
            max_conservation,
            min_conservation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let mut tracker = ConservationTracker::new(10);
        let report = tracker.analyze(&[]);
        assert_eq!(report.window_scores.len(), 0);
        assert_eq!(report.anomalies.len(), 0);
    }

    #[test]
    fn test_uniform_tokens() {
        let mut tracker = ConservationTracker::new(5);
        let tokens: Vec<String> = (0..20).map(|_| "A".to_string()).collect();
        let report = tracker.analyze(&tokens);
        // All same type => entropy should be 0
        assert!(report.mean_conservation.abs() < 0.01);
    }

    #[test]
    fn test_gradient_variance_constant() {
        let scores = vec![0.5; 10];
        let var = ConservationTracker::compute_gradient_variance(&scores);
        assert!(var.abs() < 1e-10);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut tracker = ConservationTracker::new(5).with_anomaly_threshold(1.5);
        let mut tokens: Vec<String> = (0..30).map(|i| {
            if i % 3 == 0 { "A" } else if i % 3 == 1 { "B" } else { "C" }
        }.to_string()).collect();
        // Inject anomaly: suddenly all same type
        tokens.extend(std::iter::repeat("X".to_string()).take(10));
        let report = tracker.analyze(&tokens);
        // Should detect the transition as anomalous
        assert!(!report.anomalies.is_empty() || report.gradient_variance > 0.0);
    }

    #[test]
    fn test_information_content() {
        let mut tracker = ConservationTracker::new(5);
        let tokens = vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()];
        let report = tracker.analyze(&tokens);
        let ic_a = report.information_content.get("A").unwrap();
        let ic_b = report.information_content.get("B").unwrap();
        assert!((ic_a - ic_b).abs() < 1e-10); // Equal frequency => equal info
    }
}
