/**
 * conservation-tracker.js — Conservation tracking engine
 *
 * Builds a token transition graph at runtime, computes tension between
 * token types (information content), and tracks sliding-window conservation
 * using a simplified Laplacian eigenvector analysis.
 *
 * Pure JS, no external dependencies.
 */

'use strict'

// ---------------------------------------------------------------------------
// Matrix utilities (pure JS, no numpy)
// ---------------------------------------------------------------------------

function matMul(A, B) {
  const m = A.length, n = B[0].length, p = B.length
  const C = Array.from({length: m}, () => new Array(n).fill(0))
  for (let i = 0; i < m; i++)
    for (let k = 0; k < p; k++)
      for (let j = 0; j < n; j++)
        C[i][j] += A[i][k] * B[k][j]
  return C
}

function matTranspose(A) {
  const m = A.length, n = A[0].length
  return Array.from({length: n}, (_, j) => Array.from({length: m}, (_, i) => A[i][j]))
}

function matVecMul(A, v) {
  return A.map(row => row.reduce((s, a, j) => s + a * v[j], 0))
}

function vecNorm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0))
}

function vecNormalize(v) {
  const n = vecNorm(v)
  if (n < 1e-12) return v.map(() => 0)
  return v.map(x => x / n)
}

function vecDot(a, b) {
  return a.reduce((s, x, i) => s + x * b[i], 0)
}

// Power iteration to find dominant eigenvector
function powerIteration(M, iterations = 50) {
  const n = M.length
  let v = Array.from({length: n}, () => Math.random() - 0.5)
  v = vecNormalize(v)
  for (let i = 0; i < iterations; i++) {
    v = matVecMul(M, v)
    v = vecNormalize(v)
  }
  // Rayleigh quotient for eigenvalue
  const Mv = matVecMul(M, v)
  const eigenvalue = vecDot(v, Mv)
  return { eigenvalue, eigenvector: v }
}

// ---------------------------------------------------------------------------
// Token transition graph
// ---------------------------------------------------------------------------

class TransitionGraph {
  constructor(tokenTypes) {
    this.types = tokenTypes                    // array of type names
    this.index = {}                            // name -> index
    tokenTypes.forEach((t, i) => this.index[t] = i)
    const n = tokenTypes.length
    this.counts = Array.from({length: n}, () => new Array(n).fill(0))
    this.typeCounts = new Array(n).fill(0)
    this.totalTransitions = 0
    this.tokenSequence = []                    // raw sequence for sliding window
  }

  record(fromType, toType) {
    const i = this.index[fromType]
    const j = this.index[toType]
    if (i === undefined || j === undefined) return
    this.counts[i][j]++
    this.typeCounts[i]++
    this.totalTransitions++
    this.tokenSequence.push({from: fromType, to: toType})
  }

  // Transition probability matrix P[i][j] = P(to=j | from=i)
  probabilityMatrix() {
    const n = this.types.length
    const P = Array.from({length: n}, () => new Array(n).fill(0))
    for (let i = 0; i < n; i++) {
      const total = this.typeCounts[i]
      if (total === 0) {
        // No outgoing transitions: use uniform distribution
        for (let j = 0; j < n; j++) P[i][j] = 1 / n
      } else {
        for (let j = 0; j < n; j++) {
          P[i][j] = this.counts[i][j] / total
        }
      }
    }
    return P
  }

  // Laplacian: L = D - A (unnormalized)
  laplacian() {
    const n = this.types.length
    const A = Array.from({length: n}, (_, i) =>
      this.counts[i].map(c => c > 0 ? 1 : 0)
    )
    const D = Array.from({length: n}, (_, i) => {
      const deg = A[i].reduce((s, x) => s + x, 0)
      return Array.from({length: n}, (_, j) => i === j ? deg : 0)
    })
    const L = Array.from({length: n}, (_, i) =>
      Array.from({length: n}, (_, j) => D[i][j] - A[i][j])
    )
    return L
  }
}

// ---------------------------------------------------------------------------
// Information-theoretic tension
// ---------------------------------------------------------------------------

/**
 * Shannon entropy of a probability distribution
 */
function entropy(probs) {
  let h = 0
  for (const p of probs) {
    if (p > 0) h -= p * Math.log2(p)
  }
  return h
}

/**
 * Compute per-type "information content" (surprisal) based on observed frequency.
 * Types that are rare carry more information.
 */
function computeInformationContent(typeCounts, totalTokens) {
  const total = totalTokens || 1
  return typeCounts.map(c => {
    const p = (c + 1) / (total + typeCounts.length) // Laplace smoothing
    return -Math.log2(p)
  })
}

/**
 * Tension between two types = |I(i) - I(j)| where I is information content.
 * High tension = big information surprise in the transition.
 */
function transitionTension(infoContent, i, j) {
  return Math.abs(infoContent[i] - infoContent[j])
}

// ---------------------------------------------------------------------------
// Conservation tracker
// ---------------------------------------------------------------------------

class ConservationTracker {
  constructor(tokenTypes, options = {}) {
    this.graph = new TransitionGraph(tokenTypes)
    this.types = tokenTypes
    this.windowSize = options.windowSize || 20
    this.threshold = options.threshold || 0.5
    this.violations = []
    this.tokenBuffer = []        // sliding window of {type, value, offset}
    this.nestingStacks = {}      // for balanced delimiter tracking
    this.depthHistory = []       // depth at each step
    this.tensionHistory = []     // window-averaged tension at each step
  }

  /**
   * Feed a token into the tracker.
   * @param {Object} token - {type, value, offset, ...}
   * @param {Object|null} prevToken - previous token (null for first)
   */
  feed(token, prevToken) {
    // Record transition
    if (prevToken) {
      this.graph.record(prevToken.type, token.type)
    }

    // Update sliding window
    this.tokenBuffer.push(token)
    if (this.tokenBuffer.length > this.windowSize * 2) {
      this.tokenBuffer = this.tokenBuffer.slice(-this.windowSize)
    }

    // Compute window tension
    const window = this.tokenBuffer.slice(-this.windowSize)
    if (window.length >= 2) {
      const tension = this._windowTension(window)
      this.tensionHistory.push(tension)
    }
  }

  /**
   * Compute average transition tension in a window of tokens.
   */
  _windowTension(tokens) {
    if (tokens.length < 2) return 0
    const info = computeInformationContent(
      this.graph.typeCounts,
      this.graph.totalTransitions + tokens.length
    )
    let totalTension = 0
    let count = 0
    for (let i = 1; i < tokens.length; i++) {
      const a = this.graph.index[tokens[i - 1].type]
      const b = this.graph.index[tokens[i].type]
      if (a !== undefined && b !== undefined) {
        totalTension += transitionTension(info, a, b)
        count++
      }
    }
    return count > 0 ? totalTension / count : 0
  }

  /**
   * Compute the overall conservation score [0..1].
   * Uses Laplacian eigenvector analysis: a well-conserved system
   * has its energy concentrated in the dominant eigenvector.
   */
  conservationScore() {
    if (this.graph.totalTransitions < 2) return 1.0

    const L = this.graph.laplacian()
    const n = this.types.length

    if (n < 2) return 1.0

    // The second-smallest eigenvalue of L (algebraic connectivity)
    // measures how well-connected the graph is.
    // We approximate via: compute dominant eigenvector of P (transition matrix),
    // then measure how much "energy" is in that direction.
    const P = this.graph.probabilityMatrix()

    // Symmetrize for eigenvector analysis: S = (P + P^T) / 2
    const PT = matTranspose(P)
    const S = S_fromP(P, PT)

    const { eigenvalue, eigenvector } = powerIteration(S, 100)

    // Conservation score: ratio of dominant eigenvalue to sum of all
    // Approximated by how peaked the eigenvector is
    const maxComp = Math.max(...eigenvector.map(Math.abs))
    const totalComp = eigenvector.reduce((s, x) => s + Math.abs(x), 0)
    const concentration = totalComp > 0 ? maxComp / (totalComp / n) : 1

    // Also factor in tension history consistency
    const tensionVar = this._tensionVariance()
    const tensionScore = 1.0 / (1.0 + tensionVar * 10)

    // Blend: conservation is high when structure is ordered AND tension is smooth
    return Math.min(1.0, Math.max(0.0, tensionScore * 0.6 + Math.min(concentration / n, 1) * 0.4))
  }

  _tensionVariance() {
    if (this.tensionHistory.length < 2) return 0
    const mean = this.tensionHistory.reduce((s, x) => s + x, 0) / this.tensionHistory.length
    const variance = this.tensionHistory.reduce((s, x) => s + (x - mean) ** 2, 0) / this.tensionHistory.length
    return variance
  }

  /**
   * Get current sliding-window tension
   */
  currentTension() {
    return this.tensionHistory.length > 0
      ? this.tensionHistory[this.tensionHistory.length - 1]
      : 0
  }

  /**
   * Detect conservation violations — windows where tension exceeds threshold.
   */
  detectViolations() {
    const violations = []
    const window = this.tokenBuffer.slice(-this.windowSize)

    if (window.length < 2) return violations

    const info = computeInformationContent(
      this.graph.typeCounts,
      this.graph.totalTransitions + window.length
    )

    for (let i = 1; i < window.length; i++) {
      const a = this.graph.index[window[i - 1].type]
      const b = this.graph.index[window[i].type]
      if (a !== undefined && b !== undefined) {
        const t = transitionTension(info, a, b)
        if (t > this.threshold) {
          violations.push({
            type: 'high_tension',
            from: window[i - 1],
            to: window[i],
            tension: t,
            threshold: this.threshold,
            message: `High tension (${t.toFixed(3)}) between ${window[i - 1].type} and ${window[i].type} at offset ${window[i].offset}`
          })
        }
      }
    }
    return violations
  }
}

function S_fromP(P, PT) {
  const n = P.length
  return Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) => (P[i][j] + PT[i][j]) / 2)
  )
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  ConservationTracker,
  TransitionGraph,
  computeInformationContent,
  transitionTension,
  entropy,
  // matrix utilities exposed for testing
  matMul,
  matTranspose,
  matVecMul,
  vecNorm,
  vecNormalize,
  vecDot,
  powerIteration,
}
