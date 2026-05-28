#!/usr/bin/env node
'use strict';

const {
  TokenGraph,
  computeLaplacian,
  powerIteration,
  buildTransitionMatrix,
  spectralFingerprint,
  detectLanguage,
  tokenize,
} = require('./topology-stream');

const { ConservationTracker } = require('./conservation-tracker');
const { AnomalyDetector } = require('./anomaly-detector');

// ─── Sample Code ─────────────────────────────────────────────────────────────

const samples = {
  javascript: `
    const fs = require('fs');
    async function readFile(path) {
      try {
        const data = await fs.promises.readFile(path, 'utf8');
        return JSON.parse(data);
      } catch (err) {
        console.error(err);
        throw err;
      }
    }
    class DataProcessor {
      constructor(options) {
        this.options = options;
        this.cache = new Map();
      }
      async process(key) {
        if (this.cache.has(key)) return this.cache.get(key);
        const result = await this._transform(key);
        this.cache.set(key, result);
        return result;
      }
      _transform(key) {
        return { key, timestamp: Date.now(), hash: key.length };
      }
    }
    export default DataProcessor;
  `,

  python: `
    import os
    from typing import List, Optional

    def fibonacci(n: int) -> List[int]:
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

    class DataStore:
        def __init__(self, path: str):
            self.path = path
            self._cache = {}

        def get(self, key: str) -> Optional[str]:
            if key not in self._cache:
                self._cache[key] = self._load(key)
            return self._cache[key]

        def _load(self, key):
            with open(os.path.join(self.path, key)) as f:
                return f.read()

    if __name__ == "__main__":
        store = DataStore("/tmp/data")
        print(store.get("config"))
  `,

  sql: `
    SELECT
      u.id,
      u.name,
      COUNT(o.id) AS order_count,
      SUM(o.total) AS total_spent,
      AVG(o.total) AS avg_order
    FROM users u
    LEFT JOIN orders o ON o.user_id = u.id
    WHERE u.created_at > '2024-01-01'
      AND u.active IS NOT NULL
    GROUP BY u.id, u.name
    HAVING COUNT(o.id) > 5
    ORDER BY total_spent DESC
    LIMIT 50 OFFSET 0;

    CREATE INDEX idx_orders_user ON orders(user_id);
    CREATE INDEX idx_orders_date ON orders(created_at);

    INSERT INTO audit_log (action, user_id, timestamp)
    VALUES ('report_generated', CURRENT_USER, NOW());
  `,

  json: `
    {
      "database": {
        "host": "localhost",
        "port": 5432,
        "name": "production",
        "credentials": {
          "user": "admin",
          "password": "secret"
        },
        "pool": {
          "min": 5,
          "max": 20,
          "idleTimeout": 30000
        }
      },
      "features": ["auth", "logging", "caching"],
      "version": "2.1.0",
      "metadata": {
        "created": "2024-06-15T10:30:00Z",
        "updated": "2025-01-20T14:45:00Z",
        "author": "devops"
      }
    }
  `,
};

// ─── Run Analysis ─────────────────────────────────────────────────────────────

console.log('═'.repeat(70));
console.log('  MOO Topology Stream — Demo');
console.log('═'.repeat(70));

for (const [lang, code] of Object.entries(samples)) {
  console.log(`\n${'─'.repeat(70)}`);
  console.log(`  Language: ${lang.toUpperCase()}`);
  console.log('─'.repeat(70));

  const tokens = tokenize(code);
  console.log(`  Tokens: ${tokens.length}`);

  // ── 1. Language Detection ──
  const detection = detectLanguage(tokens);
  console.log(`  Detected: ${detection.language} (confidence: ${(detection.confidence * 100).toFixed(1)}%)`);
  if (detection.scores) {
    const ranked = Object.entries(detection.scores)
      .sort((a, b) => b[1] - a[1])
      .map(([l, s]) => `${l}=${s.toFixed(3)}`)
      .join(', ');
    console.log(`  Scores:   ${ranked}`);
  }

  // ── 2. Token Graph + Spectral Analysis ──
  const graph = new TokenGraph(100);
  graph.pushAll(tokens);

  const adj = graph.getAdjMatrix();
  console.log(`  Graph nodes: ${adj.length}`);

  if (adj.length > 1) {
    const laplacian = computeLaplacian(adj);
    const { eigenvalue, eigenvector } = powerIteration(laplacian, 30);
    console.log(`  Laplacian dominant eigenvalue: ${eigenvalue.toFixed(6)}`);

    const T = buildTransitionMatrix(adj);
    const fp = spectralFingerprint(T);
    console.log(`  Spectral fingerprint (top ${fp.length}): [${fp.map(v => v.toFixed(4)).join(', ')}]`);
  }

  // ── 3. Conservation Tracker ──
  const tracker = new ConservationTracker(200);
  tracker.pushAll(tokens);
  console.log(`  Conservation score: ${tracker.getScore().toFixed(4)}`);
  const anomalies = tracker.getAnomalies();
  if (anomalies.length > 0) {
    console.log(`  Anomalies detected: ${anomalies.length}`);
    for (const a of anomalies.slice(-3)) {
      console.log(`    → token "${a.token}" score=${a.score.toFixed(4)} deviation=${a.deviation.toFixed(2)}σ`);
    }
  } else {
    console.log(`  Anomalies: none`);
  }

  // ── 4. Anomaly Detector (Wasserstein) ──
  const detector = new AnomalyDetector(0.35);
  if (adj.length > 1) {
    const T = buildTransitionMatrix(adj);
    const sig = spectralFingerprint(T);
    // first push establishes reference
    detector.push(sig);
    // push again — should be similar
    const result = detector.push(sig);
    console.log(`  Wasserstein self-distance: ${result.distance.toFixed(4)} (anomaly: ${result.isAnomaly})`);
  }
}

// ── Cross-language anomaly check ──
console.log(`\n${'═'.repeat(70)}`);
console.log('  Cross-Language Anomaly Detection');
console.log('═'.repeat(70));

const detector = new AnomalyDetector(0.35);

for (const [lang, code] of Object.entries(samples)) {
  const tokens = tokenize(code);
  const graph = new TokenGraph(100);
  graph.pushAll(tokens);
  const adj = graph.getAdjMatrix();

  if (adj.length > 1) {
    const T = buildTransitionMatrix(adj);
    const sig = spectralFingerprint(T);
    const result = detector.push(sig);
    const label = result.isAnomaly ? '⚠️  ANOMALY' : '✓ OK';
    console.log(`  ${lang.padEnd(12)} distance=${result.distance.toFixed(4)}  ${label}`);
  }
}

console.log(`\n${'─'.repeat(70)}`);
console.log('  Done.');
