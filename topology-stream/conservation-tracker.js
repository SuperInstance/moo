'use strict';

/**
 * ConservationTracker — tracks "information tension" across a sliding window
 * of tokens. Tension = information content (-log2 freq). Anomalies are flagged
 * when the conservation score drops > 2σ below the running mean.
 */

class ConservationTracker {
  constructor(windowSize = 200) {
    this.windowSize = windowSize;
    this.freq = new Map();     // token → count
    this.total = 0;
    this.window = [];          // recent tensions
    this.tensions = [];        // all tensions for stats
    this.scoreHistory = [];    // rolling conservation scores
    this.anomalies = [];       // flagged anomalies
  }

  _tension(token) {
    const count = this.freq.get(token) || 1;
    const p = count / (this.total + 1);
    return -Math.log2(Math.max(p, 1e-12));
  }

  push(token) {
    this.total++;
    this.freq.set(token, (this.freq.get(token) || 0) + 1);

    const tension = this._tension(token);
    this.window.push(tension);
    if (this.window.length > this.windowSize) this.window.shift();

    // compute gradient (finite differences of tension in window)
    if (this.window.length >= 2) {
      const grad = this.window[this.window.length - 1] - this.window[this.window.length - 2];
      this.tensions.push(grad);
      if (this.tensions.length > this.windowSize * 2) this.tensions.shift();
    }

    // compute conservation score = 1 - normalized gradient variance
    if (this.tensions.length >= 10) {
      const score = this._computeScore();
      this.scoreHistory.push(score);
      if (this.scoreHistory.length > 1000) this.scoreHistory.shift();

      // anomaly detection
      if (this.scoreHistory.length >= 20) {
        const recent = this.scoreHistory.slice(-20);
        const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
        const std = Math.sqrt(recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length) || 1e-12;
        if (score < mean - 2 * std) {
          this.anomalies.push({
            token,
            score,
            mean,
            std,
            deviation: (score - mean) / std,
            index: this.total,
          });
          // keep bounded
          if (this.anomalies.length > 500) this.anomalies.shift();
        }
      }
    }
  }

  pushAll(tokens) {
    for (const t of tokens) this.push(t);
  }

  _computeScore() {
    const g = this.tensions;
    if (g.length < 2) return 1;
    const mean = g.reduce((a, b) => a + b, 0) / g.length;
    const variance = g.reduce((a, b) => a + (b - mean) ** 2, 0) / g.length;
    // normalize: score close to 1 = low variance (conserved), 0 = high variance
    return 1 / (1 + variance);
  }

  getScore() {
    if (this.tensions.length < 2) return 1;
    return this._computeScore();
  }

  getAnomalies(limit = 20) {
    return this.anomalies.slice(-limit);
  }

  reset() {
    this.freq.clear();
    this.total = 0;
    this.window = [];
    this.tensions = [];
    this.scoreHistory = [];
    this.anomalies = [];
  }
}

module.exports = { ConservationTracker };
