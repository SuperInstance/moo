'use strict';

// ─── Token Co-occurrence Graph ───────────────────────────────────────────────

class TokenGraph {
  constructor(windowSize = 100) {
    this.windowSize = windowSize;
    this.window = [];
    this.adj = new Map(); // tokenId → Map(tokenId → count)
    this.tokenIds = new Map();
    this.nextId = 0;
  }

  _ensureId(tok) {
    if (!this.tokenIds.has(tok)) {
      this.tokenIds.set(tok, this.nextId++);
    }
    return this.tokenIds.get(tok);
  }

  push(token) {
    const id = this._ensureId(token);
    // Add edges from every token in window to this token
    for (const prevId of this.window) {
      if (prevId === id) continue;
      if (!this.adj.has(prevId)) this.adj.set(prevId, new Map());
      const row = this.adj.get(prevId);
      row.set(id, (row.get(id) || 0) + 1);
      // symmetric
      if (!this.adj.has(id)) this.adj.set(id, new Map());
      const row2 = this.adj.get(id);
      row2.set(prevId, (row2.get(prevId) || 0) + 1);
    }
    this.window.push(id);
    if (this.window.length > this.windowSize) this.window.shift();
  }

  pushAll(tokens) {
    for (const t of tokens) this.push(t);
  }

  getAdjMatrix() {
    const n = this.nextId;
    const M = Array.from({ length: n }, () => new Float64Array(n));
    for (const [i, row] of this.adj) {
      for (const [j, val] of row) {
        M[i][j] = val;
      }
    }
    return M;
  }

  getTokenList() {
    const list = new Array(this.nextId);
    for (const [tok, id] of this.tokenIds) list[id] = tok;
    return list;
  }

  reset() {
    this.window = [];
    this.adj.clear();
    this.tokenIds.clear();
    this.nextId = 0;
  }
}

// ─── Normalized Laplacian ────────────────────────────────────────────────────

function computeLaplacian(adjMatrix) {
  const n = adjMatrix.length;
  if (n === 0) return [];

  // degree
  const deg = new Float64Array(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) deg[i] += adjMatrix[i][j];

  // D^{-1/2}
  const invSqrtD = new Float64Array(n);
  for (let i = 0; i < n; i++)
    invSqrtD[i] = deg[i] > 0 ? 1 / Math.sqrt(deg[i]) : 0;

  // L_norm = I - D^{-1/2} A D^{-1/2}
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = i === j ? 1 : 0;
      L[i][j] = val - invSqrtD[i] * adjMatrix[i][j] * invSqrtD[j];
    }
  }
  return L;
}

// ─── Power Iteration ─────────────────────────────────────────────────────────

function powerIteration(L, n_iters = 20) {
  const n = L.length;
  if (n === 0) return { eigenvalue: 0, eigenvector: [] };

  // random init
  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = Math.random() - 0.5;

  let eigenvalue = 0;
  for (let iter = 0; iter < n_iters; iter++) {
    // multiply
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) w[i] += L[i][j] * v[j];

    // norm
    let norm = 0;
    for (let i = 0; i < n; i++) norm += w[i] * w[i];
    norm = Math.sqrt(norm) || 1e-12;

    eigenvalue = 0;
    for (let i = 0; i < n; i++) {
      v[i] = w[i] / norm;
      eigenvalue += w[i] * (i < v.length ? v[i] : 0);
    }
  }
  return { eigenvalue, eigenvector: Array.from(v) };
}

// ─── Transition Matrix ───────────────────────────────────────────────────────

function buildTransitionMatrix(adjMatrix) {
  const n = adjMatrix.length;
  const T = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let j = 0; j < n; j++) rowSum += adjMatrix[i][j];
    if (rowSum === 0) {
      // uniform if isolated
      for (let j = 0; j < n; j++) T[i][j] = 1 / n;
    } else {
      for (let j = 0; j < n; j++) T[i][j] = adjMatrix[i][j] / rowSum;
    }
  }
  return T;
}

// ─── Spectral Fingerprint ────────────────────────────────────────────────────

function spectralFingerprint(transitionMatrix) {
  const n = transitionMatrix.length;
  if (n === 0) return [];

  // Get top-k eigenvalues via repeated deflation
  const k = Math.min(n, 10);
  const eigenvalues = [];

  let M = transitionMatrix.map(r => Float64Array.from(r));

  for (let ev = 0; ev < k; ev++) {
    let v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.random() - 0.5;

    let lambda = 0;
    for (let iter = 0; iter < 30; iter++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) w[i] += M[i][j] * v[j];
      let norm = 0;
      for (let i = 0; i < n; i++) norm += w[i] * w[i];
      norm = Math.sqrt(norm) || 1e-12;
      lambda = 0;
      for (let i = 0; i < n; i++) {
        v[i] = w[i] / norm;
        lambda += w[i] * v[i];
      }
    }
    eigenvalues.push(lambda);

    // deflate
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        M[i][j] -= lambda * v[i] * v[j];
  }

  return eigenvalues.sort((a, b) => b - a);
}

// ─── Language Detection ──────────────────────────────────────────────────────

// Reference token-pattern signatures per language
const LANG_SIGNATURES = {
  javascript: {
    keywords: new Set([
      'function', 'const', 'let', 'var', 'return', 'if', 'else', 'for',
      'while', 'class', 'import', 'export', 'from', '=>', '===', '!==',
      'async', 'await', 'new', 'this', 'try', 'catch', 'throw',
    ]),
    weight: { keywords: 0.35, brackets: 0.25, operators: 0.2, structure: 0.2 },
  },
  python: {
    keywords: new Set([
      'def', 'class', 'import', 'from', 'return', 'if', 'elif', 'else',
      'for', 'while', 'with', 'as', 'try', 'except', 'raise', 'yield',
      'lambda', 'None', 'True', 'False', 'self', 'print', 'in', 'not',
    ]),
    weight: { keywords: 0.35, brackets: 0.25, operators: 0.2, structure: 0.2 },
  },
  sql: {
    keywords: new Set([
      'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE',
      'TABLE', 'INDEX', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON',
      'AND', 'OR', 'NOT', 'NULL', 'IS', 'IN', 'GROUP', 'BY', 'ORDER',
      'HAVING', 'LIMIT', 'OFFSET', 'AS', 'DISTINCT', 'COUNT', 'SUM',
      'AVG', 'MIN', 'MAX', 'SET', 'VALUES', 'INTO', 'ALTER', 'DROP',
    ]),
    weight: { keywords: 0.45, brackets: 0.15, operators: 0.2, structure: 0.2 },
  },
  json: {
    keywords: new Set([]),
    weight: { keywords: 0.05, brackets: 0.45, operators: 0.15, structure: 0.35 },
  },
};

function tokenize(code) {
  // Simple tokenizer — splits into words, operators, brackets
  return code
    .replace(/\/\/.*$/gm, '')       // strip line comments
    .replace(/\/\*[\s\S]*?\*\//g, '') // strip block comments
    .replace(/#.*/g, '')            // strip Python comments (rough)
    .match(/[\w]+|[{}[\]()=><!+\-*/%&|^~;:.,?@#\\'"`]+|\S/g) || [];
}

function detectLanguage(tokens) {
  if (!tokens || tokens.length === 0) return { language: 'unknown', confidence: 0 };

  const tokenSet = new Set(tokens.map(t => t.toLowerCase ? t.toLowerCase() : t));
  const scores = {};

  for (const [lang, sig] of Object.entries(LANG_SIGNATURES)) {
    let kwHits = 0;
    for (const kw of sig.keywords) {
      if (tokenSet.has(kw.toLowerCase ? kw.toLowerCase() : kw)) kwHits++;
    }
    const kwScore = sig.keywords.size > 0 ? kwHits / sig.keywords.size : 0;

    // bracket density
    const bracketTokens = tokens.filter(t => /^[{}\[\]()]+$/.test(t));
    const bracketScore = bracketTokens.length / tokens.length;

    // operator density
    const opTokens = tokens.filter(t => /^[=><!+\-*/%&|^~]+$/.test(t));
    const opScore = opTokens.length / tokens.length;

    // structure: colons, semicolons, commas
    const structTokens = tokens.filter(t => /^[;:.,]$/.test(t));
    const structScore = structTokens.length / tokens.length;

    scores[lang] =
      kwScore * sig.weight.keywords +
      bracketScore * sig.weight.brackets +
      opScore * sig.weight.operators +
      structScore * sig.weight.structure;
  }

  // JSON special case: lots of brackets + colons but no keywords
  const colonCount = tokens.filter(t => t === ':').length;
  const braceCount = tokens.filter(t => t === '{' || t === '}').length;
  if (colonCount > 3 && braceCount > 4 && scores.json < 0.4) {
    scores.json += 0.3;
  }

  const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  const total = Object.values(scores).reduce((a, b) => a + b, 0) || 1;
  return {
    language: best[0],
    confidence: Math.min(best[1] / total, 1),
    scores,
  };
}

// ─── Exports ─────────────────────────────────────────────────────────────────

module.exports = {
  TokenGraph,
  computeLaplacian,
  powerIteration,
  buildTransitionMatrix,
  spectralFingerprint,
  detectLanguage,
  tokenize,
  LANG_SIGNATURES,
};
