//! Conservation tracking tests.

use moo_rust::conservation::{ConservationReport, ConservationTracker};

#[test]
fn test_empty_input() {
    let mut tracker = ConservationTracker::new(10);
    let report = tracker.analyze(&[]);
    assert_eq!(report.window_scores.len(), 0);
    assert_eq!(report.anomalies.len(), 0);
}

#[test]
fn test_uniform_stream() {
    let mut tracker = ConservationTracker::new(5);
    let tokens: Vec<String> = (0..20).map(|_| "A".to_string()).collect();
    let report = tracker.analyze(&tokens);
    // All same type => very low conservation (no diversity)
    assert!(report.mean_conservation < 0.5);
}

#[test]
fn test_diverse_stream() {
    let mut tracker = ConservationTracker::new(10);
    let tokens: Vec<String> = (0..40).map(|i| {
        match i % 4 {
            0 => "A",
            1 => "B",
            2 => "C",
            _ => "D",
        }.to_string()
    }).collect();
    let report = tracker.analyze(&tokens);
    // Diverse stream => higher conservation
    assert!(report.mean_conservation > 0.5);
}

#[test]
fn test_gradient_variance_constant() {
    let scores = vec![0.5; 10];
    let var = ConservationTracker::compute_gradient_variance(&scores);
    assert!(var.abs() < 1e-10);
}

#[test]
fn test_gradient_variance_changing() {
    let scores: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
    let var = ConservationTracker::compute_gradient_variance(&scores);
    assert!(var > 0.0);
}

#[test]
fn test_gradient_variance_single() {
    let scores = vec![0.5];
    let var = ConservationTracker::compute_gradient_variance(&scores);
    assert_eq!(var, 0.0);
}

#[test]
fn test_information_content_equal() {
    let mut tracker = ConservationTracker::new(5);
    let tokens = vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()];
    let report = tracker.analyze(&tokens);
    let ic_a = report.information_content.get("A").unwrap();
    let ic_b = report.information_content.get("B").unwrap();
    assert!((ic_a - ic_b).abs() < 1e-10);
}

#[test]
fn test_information_content_rare_vs_common() {
    let mut tracker = ConservationTracker::new(5);
    let mut tokens = vec!["A".to_string(); 9];
    tokens.push("B".to_string());
    let report = tracker.analyze(&tokens);
    let ic_a = report.information_content.get("A").unwrap();
    let ic_b = report.information_content.get("B").unwrap();
    // Rare tokens have higher information content
    assert!(ic_b > ic_a);
}

#[test]
fn test_anomaly_detection_clean() {
    let mut tracker = ConservationTracker::new(10).with_anomaly_threshold(3.0);
    let tokens: Vec<String> = (0..50).map(|i| match i % 3 {
        0 => "A",
        1 => "B",
        _ => "C",
    }.to_string()).collect();
    let report = tracker.analyze(&tokens);
    // Regular pattern should have few anomalies
    assert!(report.anomalies.len() <= 3);
}

#[test]
fn test_anomaly_detection_boundary() {
    let mut tracker = ConservationTracker::new(5).with_anomaly_threshold(1.0);
    let mut tokens: Vec<String> = (0..30).map(|i| match i % 3 {
        0 => "A",
        1 => "B",
        _ => "C",
    }.to_string()).collect();
    // Inject sudden uniformity
    tokens.extend(std::iter::repeat("X".to_string()).take(15));
    let report = tracker.analyze(&tokens);
    // Should detect the transition
    // The gradient variance should be non-zero at least
    assert!(report.gradient_variance > 0.0);
}

#[test]
fn test_window_size_larger_than_input() {
    let mut tracker = ConservationTracker::new(100);
    let tokens = vec!["A".to_string(), "B".to_string()];
    let report = tracker.analyze(&tokens);
    // Should still work, window shrinks to input size
    assert!(report.window_scores.len() <= 2);
}

#[test]
fn test_conservation_ratio_range() {
    let mut tracker = ConservationTracker::new(10);
    let tokens: Vec<String> = (0..20).map(|i| match i % 5 {
        0 => "A",
        1 => "B",
        2 => "C",
        3 => "D",
        _ => "E",
    }.to_string()).collect();
    let report = tracker.analyze(&tokens);
    assert!(report.mean_conservation >= 0.0);
    assert!(report.mean_conservation <= 1.0);
}

#[test]
fn test_report_serialization() {
    let mut tracker = ConservationTracker::new(10);
    let tokens = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let report = tracker.analyze(&tokens);
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("conservation_ratio"));
    assert!(json.contains("window_scores"));
    assert!(json.contains("anomalies"));
}

#[test]
fn test_min_max_conservation() {
    let mut tracker = ConservationTracker::new(5);
    let tokens: Vec<String> = (0..20).map(|i| match i % 3 {
        0 => "A",
        1 => "B",
        _ => "C",
    }.to_string()).collect();
    let report = tracker.analyze(&tokens);
    assert!(report.max_conservation >= report.min_conservation);
}

#[test]
fn test_anomaly_description() {
    let mut tracker = ConservationTracker::new(5).with_anomaly_threshold(0.5);
    let mut tokens: Vec<String> = (0..30).map(|i| match i % 4 {
        0 => "A",
        1 => "B",
        2 => "C",
        _ => "D",
    }.to_string()).collect();
    tokens.extend(std::iter::repeat("Z".to_string()).take(20));
    let report = tracker.analyze(&tokens);
    for anomaly in &report.anomalies {
        assert!(!anomaly.description.is_empty());
        assert!(anomaly.description.contains("window"));
        assert!(anomaly.deviation > 0.0);
    }
}

#[test]
fn test_single_token_type() {
    let mut tracker = ConservationTracker::new(5);
    let tokens = vec!["X".to_string(); 10];
    let report = tracker.analyze(&tokens);
    assert_eq!(report.information_content.len(), 1);
}

#[test]
fn test_large_vocabulary() {
    let mut tracker = ConservationTracker::new(20);
    let tokens: Vec<String> = (0..100).map(|i| format!("T{}", i % 20)).collect();
    let report = tracker.analyze(&tokens);
    assert_eq!(report.information_content.len(), 20);
}
