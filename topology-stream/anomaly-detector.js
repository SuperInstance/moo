'use strict';

/**
 * AnomalyDetector — compares current persistence signature against a reference.
 * Uses a simplified Wasserstein (Earth Mover's) distance on 1D sorted values.
 */

class AnomalyDetector {
  constructor(threshold = 0.35) {
    this.threshold = threshold;
    this.reference = null;     // reference fingerprint (sorted number[])
    this.history = [];         // recent signatures for smoothing
    this.historySize = 10;
    this.flags = [];           // flagged anomalies
  }

  /**
   * Set the reference signature (e.g., from a known-good corpus).
   * @param {number[]} signature - sorted eigenvalue / spectral fingerprint
   */
  setReference(signature) {
    this.reference = Float64Array.from(signature).sort();
  }

  /**
   * Compute simplified 1D Wasserstein distance between two sorted arrays.
   * Pads shorter array with its last value.
   */
  static wasserstein1D(a, b) {
    const n = Math.max(a.length, b.length);
    if (n === 0) return 0;
    let dist = 0;
    for (let i = 0; i < n; i++) {
      const va = i < a.length ? a[i] : a[a.length - 1];
      const vb = i < b.length ? b[i] : b[b.length - 1];
      dist += Math.abs(va - vb);
    }
    return dist / n;
  }

  /**
   * Feed a new spectral signature, compare against reference.
   * Returns { distance, isAnomaly, threshold }
   */
  push(signature) {
    const sorted = Float64Array.from(signature).sort();

    // smooth: average recent signatures
    this.history.push(sorted);
    if (this.history.length > this.historySize) this.history.shift();

    const smoothed = this._smooth();

    if (!this.reference) {
      // first run: set as reference
      this.reference = smoothed;
      return { distance: 0, isAnomaly: false, threshold: this.threshold };
    }

    const distance = AnomalyDetector.wasserstein1D(smoothed, this.reference);
    const isAnomaly = distance > this.threshold;

    const result = { distance, isAnomaly, threshold: this.threshold };

    if (isAnomaly) {
      this.flags.push({
        distance,
        timestamp: Date.now(),
        signature: Array.from(signature),
      });
      if (this.flags.length > 200) this.flags.shift();
    }

    return result;
  }

  _smooth() {
    if (this.history.length === 0) return new Float64Array(0);
    const len = this.history[this.history.length - 1].length;
    const out = new Float64Array(len);
    for (const sig of this.history) {
      for (let i = 0; i < len; i++) {
        out[i] += i < sig.length ? sig[i] : sig[sig.length - 1];
      }
    }
    for (let i = 0; i < len; i++) out[i] /= this.history.length;
    return out;
  }

  getFlags(limit = 20) {
    return this.flags.slice(-limit);
  }

  reset() {
    this.reference = null;
    this.history = [];
    this.flags = [];
  }
}

module.exports = { AnomalyDetector };
