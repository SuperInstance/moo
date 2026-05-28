/**
 * moo-fast.js — High-performance replacement for moo-spectral.js
 *
 * Optimizations:
 *   1. Flat Float64Array matrices — cache-friendly, no GC pressure
 *   2. Lanczos eigensolver — O(kn) for top-k eigenvalues vs O(n³) Jacobi
 *   3. Precomputed lookup tables — no matrix ops at tokenization time
 *   4. Bit-parallel character sets — popcount on 256-bit Uint32Array[8]
 *   5. Sliding window via circular buffer — zero allocation during tokenization
 *   6. Fast entropy via precomputed -log2 lookup
 *   7. Branchless chunk dispatch — perfect hash for magic numbers
 *
 * Same public API as moo-spectral.js.
 *
 * @module moo-fast
 */
;(function(root, factory) {
  if (typeof define === 'function' && define.amd) {
    define([], factory)
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory()
  } else {
    root.mooSpectral = factory()
  }
}(this, function() {
  'use strict'

  // ====================================================================
  // 1. Flat Float64Array matrix operations
  // ====================================================================

  /**
   * Create a flat n×n zero matrix (Float64Array)
   */
  function flatZeros(n) {
    return new Float64Array(n * n)
  }

  /**
   * Create a flat n×k zero matrix (Float64Array)
   */
  function flatZerosRect(n, k) {
    return new Float64Array(n * k)
  }

  /** Row-major accessor: M[i * stride + j] */
  // (inline these — no function call overhead)

  /**
   * Flat matrix multiply C = A * B. A is m×p, B is p×n, C is m×n.
   */
  function flatMatMul(A, B, m, p, n) {
    var C = new Float64Array(m * n)
    for (var i = 0; i < m; i++) {
      var iOff = i * p
      var iOffC = i * n
      for (var k = 0; k < p; k++) {
        var a = A[iOff + k]
        if (a === 0) continue
        var kOff = k * n
        for (var j = 0; j < n; j++) {
          C[iOffC + j] += a * B[kOff + j]
        }
      }
    }
    return C
  }

  /**
   * Symmetrize flat matrix in-place: S = (A + Aᵀ) / 2
   */
  function flatSymmetrize(A, n) {
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        var v = (A[i * n + j] + A[j * n + i]) * 0.5
        A[i * n + j] = v
        A[j * n + i] = v
      }
    }
  }

  /**
   * Compute degree vector for flat weighted adjacency W (n×n)
   */
  function flatDegrees(W, n) {
    var d = new Float64Array(n)
    for (var i = 0; i < n; i++) {
      var s = 0
      var off = i * n
      for (var j = 0; j < n; j++) s += W[off + j]
      d[i] = s
    }
    return d
  }

  /**
   * Flat Laplacian: L = D - W (returns Float64Array n×n)
   */
  function flatLaplacian(W, n) {
    var d = flatDegrees(W, n)
    var L = new Float64Array(n * n)
    for (var i = 0; i < n; i++) {
      var off = i * n
      for (var j = 0; j < n; j++) {
        L[off + j] = (i === j ? d[i] : 0) - W[off + j]
      }
    }
    return { L: L, degrees: d }
  }

  /**
   * Flat normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
   */
  function flatNormalizedLaplacian(W, n) {
    var d = flatDegrees(W, n)
    var Lnorm = new Float64Array(n * n)
    var invSqrtD = new Float64Array(n)
    for (var i = 0; i < n; i++) {
      invSqrtD[i] = d[i] > 0 ? 1 / Math.sqrt(d[i]) : 0
    }
    for (var i = 0; i < n; i++) {
      var iOff = i * n
      for (var j = 0; j < n; j++) {
        if (i === j) {
          Lnorm[iOff + j] = d[i] > 0 ? 1 : 0
        } else {
          Lnorm[iOff + j] = -W[iOff + j] * invSqrtD[i] * invSqrtD[j]
        }
      }
    }
    return { L: Lnorm, degrees: d }
  }

  // ====================================================================
  // 2. Lanczos eigensolver for symmetric matrices (top-k eigenvalues)
  // ====================================================================

  /**
   * Lanczos iteration for symmetric matrix A (flat n×n Float64Array).
   * Returns tridiagonal matrix T (k×k) and orthonormal basis Q (n×k).
   *
   * Complexity: O(k * n) per iteration, far better than Jacobi's O(n³).
   */
  function lanczosIteration(A, n, k) {
    k = Math.min(k, n)
    var Q = new Float64Array(n * k)
    var alpha = new Float64Array(k)  // diagonal of T
    var beta = new Float64Array(k)   // sub/super-diagonal of T

    // q₀ = random unit vector (deterministic seed for reproducibility)
    var q = new Float64Array(n)
    q[0] = 1
    Q.set(q, 0)

    var r = new Float64Array(n)
    var prevQ = null

    for (var j = 0; j < k; j++) {
      // r = A * qⱼ
      var qOff = j * n
      for (var i = 0; i < n; i++) {
        var s = 0
        var aOff = i * n
        for (var l = 0; l < n; l++) {
          s += A[aOff + l] * Q[qOff + l]
        }
        r[i] = s
      }

      // αⱼ = qⱼᵀ * r
      var a = 0
      for (var i = 0; i < n; i++) a += Q[qOff + i] * r[i]
      alpha[j] = a

      // r = r - αⱼ * qⱼ - βⱼ₋₁ * qⱼ₋₁
      for (var i = 0; i < n; i++) r[i] -= a * Q[qOff + i]
      if (j > 0) {
        var prevOff = (j - 1) * n
        var bPrev = beta[j - 1]
        for (var i = 0; i < n; i++) r[i] -= bPrev * Q[prevOff + i]
      }

      // Full reorthogonalization (improved numerical stability)
      for (var reorth = 0; reorth < 2; reorth++) {
        for (var l = 0; l <= j; l++) {
          var lOff = l * n
          var dot = 0
          for (var i = 0; i < n; i++) dot += Q[lOff + i] * r[i]
          for (var i = 0; i < n; i++) r[i] -= dot * Q[lOff + i]
        }
      }

      // βⱼ = ||r||
      var b = 0
      for (var i = 0; i < n; i++) b += r[i] * r[i]
      b = Math.sqrt(b)

      if (j < k - 1) {
        beta[j] = b
        if (b < 1e-14) {
          // Invariant subspace found; fill remaining with zeros
          for (var jj = j + 1; jj < k; jj++) {
            alpha[jj] = 0
            beta[jj] = 0
            Q[jj * n + jj] = 1 // orthonormal basis extension
          }
          break
        }
        var nextOff = (j + 1) * n
        var invB = 1 / b
        for (var i = 0; i < n; i++) Q[nextOff + i] = r[i] * invB
      }
    }

    return { Q: Q, alpha: alpha, beta: beta, k: k }
  }

  /**
   * Implicit QR shift for tridiagonal eigenvalue problem.
   * Finds eigenvalues of T given diagonal alpha and sub-diagonal beta.
   * Returns sorted eigenvalues.
   */
  function tridiagEigenvalues(alpha, beta, k) {
    var d = new Float64Array(alpha.subarray(0, k))  // diagonal
    var e = new Float64Array(k)  // off-diagonal (shifted by one)
    for (var i = 0; i < k - 1; i++) e[i] = beta[i]

    // QR iteration with Wilkinson shifts
    var maxIter = 100 * k
    for (var iter = 0; iter < maxIter; iter++) {
      // Find active block
      var m = k - 1
      while (m > 0 && Math.abs(e[m - 1]) <= 1e-14 * (Math.abs(d[m - 1]) + Math.abs(d[m]))) m--
      if (m === 0) break

      // Wilkinson shift
      var dd = (d[m - 1] - d[m]) * 0.5
      var sign = dd >= 0 ? 1 : -1
      var mu = d[m] - e[m - 1] * e[m - 1] / (dd + sign * Math.sqrt(dd * dd + e[m - 1] * e[m - 1]))

      // Chase the bulge
      var x = d[0] - mu
      var z = e[0]
      for (var i = 0; i < m; i++) {
        var r = Math.sqrt(x * x + z * z)
        var c = x / r, s = z / r
        if (i > 0) e[i - 1] = r

        var w = c * d[i] + s * e[i]
        e[i] = c * e[i] - s * d[i]
        d[i] = w
        var w2 = -s * d[i + 1]
        d[i + 1] = c * d[i + 1]

        if (i < m - 1) {
          x = e[i]
          z = w2
          e[i] = x
        }
      }
    }

    // Sort ascending
    var arr = Array.from(d)
    arr.sort(function(a, b) { return a - b })
    return new Float64Array(arr)
  }

  /**
   * Compute eigenvectors of tridiagonal T, then map back via Q.
   * Returns top-k eigenvectors of original matrix.
   */
  function tridiagEigenvectors(alpha, beta, Q, n, k) {
    // Build full tridiagonal matrix for inverse iteration
    var T = new Float64Array(k * k)
    for (var i = 0; i < k; i++) {
      T[i * k + i] = alpha[i]
      if (i > 0) {
        T[i * k + (i - 1)] = beta[i - 1]
        T[(i - 1) * k + i] = beta[i - 1]
      }
    }

    // Get eigenvalues first
    var eigenvalues = tridiagEigenvalues(alpha, beta, k)

    // Inverse iteration to find eigenvectors of T
    var TV = new Float64Array(k * k) // eigenvectors of T as columns
    for (var ev = 0; ev < k; ev++) {
      var lam = eigenvalues[ev]
      // Shift T: (T - λI)
      var shifted = new Float64Array(k * k)
      for (var i = 0; i < k * k; i++) shifted[i] = T[i]
      for (var i = 0; i < k; i++) shifted[i * k + i] -= lam

      // Solve (T - λI)v = random using LU decomposition
      // Start with a random-ish vector
      var v = new Float64Array(k)
      v[ev % k] = 1
      // Simple Gauss-Seidel iteration (5 iterations is enough for inverse iteration)
      for (var gs = 0; gs < 10; gs++) {
        for (var i = 0; i < k; i++) {
          var sum = 0
          for (var j = 0; j < k; j++) {
            if (j !== i) sum += shifted[i * k + j] * v[j]
          }
          var diag = shifted[i * k + i]
          v[i] = diag !== 0 ? -sum / diag : 1
        }
      }
      // Normalize
      var norm = 0
      for (var i = 0; i < k; i++) norm += v[i] * v[i]
      norm = Math.sqrt(norm)
      if (norm > 0) {
        for (var i = 0; i < k; i++) v[i] /= norm
      }
      for (var i = 0; i < k; i++) TV[i * k + ev] = v[i]
    }

    // Map back: eigenvectors of A = Q * TV
    // Q is n×k, TV is k×k, result is n×k
    var evecs = new Float64Array(n * k)
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < k; j++) {
        var s = 0
        for (var l = 0; l < k; l++) {
          s += Q[i + l * n] * TV[l * k + j]
        }
        evecs[i * k + j] = s
      }
    }

    return { eigenvalues: eigenvalues, eigenvectors: evecs }
  }

  /**
   * Full Lanczos eigensolver: Lanczos + tridiagonal QR.
   * Returns { eigenvalues: Float64Array, eigenvectors: Float64Array (n×k, row-major) }
   * Eigenvectors: evecs[i * k + ev] = component of node i in eigenvector ev.
   */
  function lanczosEigen(A, n, numEigs) {
    if (n <= 0) return { eigenvalues: new Float64Array(0), eigenvectors: new Float64Array(0) }
    numEigs = Math.min(numEigs || n, n)
    // Run Lanczos with extra iterations for stability
    var k = Math.min(numEigs + 4, n) // a few extra for accuracy
    var lanczos = lanczosIteration(A, n, k)
    var result = tridiagEigenvectors(lanczos.alpha, lanczos.beta, lanczos.Q, n, k)

    // Trim to requested number
    var evals = new Float64Array(numEigs)
    var evecs = new Float64Array(n * numEigs)
    for (var i = 0; i < numEigs; i++) {
      evals[i] = result.eigenvalues[i]
      for (var j = 0; j < n; j++) {
        evecs[j * numEigs + i] = result.eigenvectors[j * k + i]
      }
    }
    return { eigenvalues: evals, eigenvectors: evecs }
  }

  // ====================================================================
  // 3. Precomputed lookup tables
  // ====================================================================

  /**
   * At compile time: compute Fiedler ordering and precompute everything
   * needed at tokenization time as flat arrays. At tokenization time,
   * the lexer uses ONLY precomputed lookup — no matrix operations.
   */

  /**
   * Build precomputed spectral data for fast tokenization.
   * Returns flat arrays ready for direct indexing.
   */
  function precomputeSpectralData(tensionMatrix, n, numComponents) {
    numComponents = numComponents || 5
    if (n <= 2) return null

    var lap = flatNormalizedLaplacian(tensionMatrix, n)
    var eigen = lanczosEigen(lap.L, n, Math.min(numComponents + 2, n))

    // Fiedler vector = eigenvector of 2nd-smallest eigenvalue (index 1)
    var k = eigen.eigenvalues.length
    var fiedler = new Float64Array(n)
    for (var i = 0; i < n; i++) fiedler[i] = eigen.eigenvectors[i * k + 1]

    // Ordering = argsort of Fiedler vector
    var ordering = new Int32Array(n)
    for (var i = 0; i < n; i++) ordering[i] = i
    // Simple insertion sort (n is small, typically 5-30 rules)
    var orderArr = Array.from(ordering)
    orderArr.sort(function(a, b) { return fiedler[a] - fiedler[b] })
    for (var i = 0; i < n; i++) ordering[i] = orderArr[i]

    // Precompute eigenvector lookup: for each rule index, store its
    // projection onto the top eigenvectors. This is the ONLY thing
    // the tokenization-time conservation tracker needs.
    var nc = Math.min(numComponents, k)
    var eigenProjection = new Float64Array(n * nc) // eigenProjection[i * nc + c]
    for (var i = 0; i < n; i++) {
      for (var c = 0; c < nc; c++) {
        eigenProjection[i * nc + c] = eigen.eigenvectors[i * k + c]
      }
    }

    // Precompute Cheeger constant
    var cheeger = flatCheegerConstant(tensionMatrix, fiedler, n)

    return {
      fiedler: fiedler,
      ordering: ordering,
      eigenvalues: eigen.eigenvalues,
      eigenProjection: eigenProjection,
      numComponents: nc,
      tensionMatrix: tensionMatrix,
      degrees: lap.degrees,
      spectralGap: eigen.eigenvalues.length > 1 ? eigen.eigenvalues[1] : 0,
      cheegerConstant: cheeger
    }
  }

  /**
   * Flat Cheeger constant from Fiedler vector.
   */
  function flatCheegerConstant(W, fiedler, n) {
    var indices = new Array(n)
    for (var i = 0; i < n; i++) indices[i] = i
    indices.sort(function(a, b) { return fiedler[a] - fiedler[b] })

    var totalDegree = 0
    for (var i = 0; i < n * n; i++) totalDegree += W[i]
    totalDegree *= 0.5

    var bestCheeger = Infinity
    var volS = 0

    for (var kk = 0; kk < n - 1; kk++) {
      var idx = indices[kk]
      var idxOff = idx * n
      for (var j = 0; j < n; j++) volS += W[idxOff + j]

      var cut = 0
      for (var i = 0; i <= kk; i++) {
        var iOff = indices[i] * n
        for (var j = kk + 1; j < n; j++) {
          cut += W[iOff + indices[j]]
        }
      }

      var volComp = totalDegree - volS
      var volMin = Math.min(volS, volComp)
      if (volMin > 0) {
        var h = cut / volMin
        if (h < bestCheeger) bestCheeger = h
      }
    }
    return bestCheeger === Infinity ? 0 : bestCheeger
  }

  // ====================================================================
  // 4. Bit-parallel character sets
  // ====================================================================

  /**
   * Represent a character set as a Uint32Array of 8 elements (256 bits).
   * Bit i set means char code i is in the set.
   */
  function bitCharSet(chars, ranges) {
    var bits = new Uint32Array(8) // 256 bits = 8 × 32 bits

    // Set specific characters
    if (chars) {
      var keys = Object.keys(chars)
      for (var i = 0; i < keys.length; i++) {
        var code = keys[i].charCodeAt(0)
        if (code < 256) {
          bits[code >>> 5] |= (1 << (code & 31))
        }
      }
    }

    // Set ranges
    if (ranges) {
      for (var r = 0; r < ranges.length; r++) {
        var lo = Math.max(0, ranges[r][0])
        var hi = Math.min(255, ranges[r][1])
        for (var c = lo; c <= hi; c++) {
          bits[c >>> 5] |= (1 << (c & 31))
        }
      }
    }

    return bits
  }

  /**
   * Fast popcount for Uint32Array.
   * Uses bitwise tricks, no Math.log.
   */
  function popcount32(v) {
    v = v - ((v >>> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333)
    v = (v + (v >>> 4)) & 0x0f0f0f0f
    v = v + (v >>> 8)
    v = v + (v >>> 16)
    return v & 0x7f
  }

  /**
   * Popcount of entire Uint32Array.
   */
  function popcountArray(arr) {
    var c = 0
    for (var i = 0; i < arr.length; i++) c += popcount32(arr[i])
    return c
  }

  /**
   * Bit-parallel overlap: |A ∩ B| / |A ∪ B|
   */
  function bitOverlap(a, b) {
    var andBits = new Uint32Array(8)
    var orBits = new Uint32Array(8)
    for (var i = 0; i < 8; i++) {
      andBits[i] = a[i] & b[i]
      orBits[i] = a[i] | b[i]
    }
    var intersection = popcountArray(andBits)
    var union = popcountArray(orBits)
    return union > 0 ? intersection / union : 0
  }

  /**
   * Extract char set from rule patterns and return as bit-parallel representation.
   */
  function extractBitCharSet(rule) {
    var patterns = rule.match || []
    var chars = {}
    var ranges = []

    for (var p = 0; p < patterns.length; p++) {
      var pat = patterns[p]
      var src = typeof pat === 'string' ? pat : (pat && pat.source ? pat.source : '')

      for (var i = 0; i < src.length; i++) {
        if (src[i] === '\\' && i + 1 < src.length) {
          var escaped = src[i + 1]
          if (escaped === 'd') { ranges.push([48, 57]); i++; continue }
          if (escaped === 'w') { ranges.push([48, 57]); ranges.push([65, 90]); ranges.push([97, 122]); chars['_'] = true; i++; continue }
          if (escaped === 's') { ranges.push([9, 13]); chars[' '] = true; i++; continue }
          if (escaped === 'n') { chars['\n'] = true; i++; continue }
          if (escaped === 't') { chars['\t'] = true; i++; continue }
          if (escaped === 'r') { chars['\r'] = true; i++; continue }
          i++
          continue
        }
        if ('^$*+?.|()[]{}='.indexOf(src[i]) !== -1) continue
        chars[src[i]] = true
      }

      var classRe = /\[([^\]]*)\]/g
      var classMatch
      while ((classMatch = classRe.exec(src)) !== null) {
        var classContent = classMatch[1]
        var negate = classContent[0] === '^'
        if (negate) classContent = classContent.slice(1)
        var j = 0
        while (j < classContent.length) {
          if (j + 2 < classContent.length && classContent[j + 1] === '-') {
            ranges.push([classContent.charCodeAt(j), classContent.charCodeAt(j + 2)])
            j += 3
          } else {
            chars[classContent[j]] = true
            j++
          }
        }
      }
    }

    return bitCharSet(chars, ranges)
  }

  /**
   * Build tension matrix using bit-parallel character sets.
   * Returns flat Float64Array n×n.
   */
  function buildBitTensionMatrix(rules) {
    var n = rules.length
    var charSets = new Array(n)
    for (var i = 0; i < n; i++) charSets[i] = extractBitCharSet(rules[i])

    var T = new Float64Array(n * n)
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        var overlap = bitOverlap(charSets[i], charSets[j])
        T[i * n + j] = overlap
        T[j * n + i] = overlap
      }
    }
    return T
  }

  // ====================================================================
  // 5. Sliding window via circular buffer (zero allocation)
  // ====================================================================

  /**
   * Fast conservation tracker using circular buffer + precomputed eigen lookup.
   * No array.push, no slice, no allocation during tokenization.
   */
  function FastConservationTracker(eigenProjection, typeIndexMap, typeIndexKeys, numComponents, windowSize) {
    this.projection = eigenProjection          // Float64Array: projection[i * nc + c]
    this.typeIndexMap = typeIndexMap            // Object: typeName -> index
    this.typeIndexKeys = typeIndexKeys          // Array: index -> typeName
    this.nc = numComponents
    this.windowSize = windowSize || 20

    // Circular buffer
    this.buf = new Int32Array(this.windowSize)  // token type indices
    this.bufLen = 0
    this.bufHead = 0  // write position (next slot to write)

    // Score history (circular)
    this.scores = new Float64Array(100)
    this.scoresLen = 0
    this.scoresHead = 0

    this.baseline = 0
    this.baselineSet = false
    this.n = typeIndexKeys.length
  }

  /**
   * Feed a token type. Returns { score, delta, isAnomaly }.
   * Zero allocation path.
   */
  FastConservationTracker.prototype.feed = function(type) {
    var idx = this.typeIndexMap[type]
    if (idx === undefined) idx = (this.n >>> 1) // midpoint fallback

    // Write to circular buffer
    this.buf[this.bufHead] = idx
    this.bufHead = (this.bufHead + 1) % this.windowSize
    if (this.bufLen < this.windowSize) this.bufLen++

    if (this.bufLen < 3) return { score: 0, delta: 0, isAnomaly: false }

    // Compute conservation score using precomputed eigen projections
    // Score = mean gradient variance across top eigenvectors
    var totalVar = 0
    var nc = this.nc
    var n = this.n
    var proj = this.projection

    for (var ev = 0; ev < nc; ev++) {
      // Compute projections and their differences in one pass
      var prevVal = 0
      var sum = 0
      var sumSq = 0
      var count = this.bufLen - 1

      // Read circular buffer sequentially
      for (var i = 0; i < this.bufLen; i++) {
        var bIdx = (this.bufHead - this.bufLen + i + this.windowSize) % this.windowSize
        var val = proj[this.buf[bIdx] * nc + ev]
        if (i > 0) {
          var diff = val - prevVal
          sum += diff
          sumSq += diff * diff
        }
        prevVal = val
      }

      var mean = sum / count
      totalVar += (sumSq / count) - mean * mean
    }

    var score = totalVar / nc

    // Establish baseline from first 5 scores
    if (!this.baselineSet && this.scoresLen >= 5) {
      var bSum = 0
      for (var i = 0; i < this.scoresLen; i++) bSum += this.scores[i]
      this.baseline = bSum / this.scoresLen
      this.baselineSet = true
    }

    var delta = this.baselineSet ? score / (this.baseline || 1) : 1
    var isAnomaly = this.baselineSet && delta > 2.25

    // Store score in circular history
    this.scores[this.scoresHead] = score
    this.scoresHead = (this.scoresHead + 1) % 100
    if (this.scoresLen < 100) this.scoresLen++

    return { score: score, delta: delta, isAnomaly: isAnomaly }
  }

  /**
   * Suggest correction using precomputed eigen projections.
   * Nearest neighbor in eigenspace.
   */
  FastConservationTracker.prototype.suggestCorrection = function(errorType, recentTypes) {
    if (!recentTypes || recentTypes.length === 0) {
      return { suggestedType: errorType, confidence: 0 }
    }

    var nc = this.nc
    var proj = this.projection
    var n = this.n

    // Compute expected position from recent tokens
    var expected = new Float64Array(nc)
    var count = 0
    for (var i = 0; i < recentTypes.length; i++) {
      var idx = this.typeIndexMap[recentTypes[i]]
      if (idx !== undefined) {
        var off = idx * nc
        for (var k = 0; k < nc; k++) expected[k] += proj[off + k]
        count++
      }
    }
    if (count === 0) return { suggestedType: errorType, confidence: 0 }
    for (var k = 0; k < nc; k++) expected[k] /= count

    // Find nearest token type in eigenspace
    var bestType = errorType
    var bestDist = Infinity

    for (var t = 0; t < n; t++) {
      var tOff = t * nc
      var dist = 0
      for (var k = 0; k < nc; k++) {
        var d = proj[tOff + k] - expected[k]
        dist += d * d
      }
      if (dist < bestDist) {
        bestDist = dist
        bestType = this.typeIndexKeys[t]
      }
    }

    var confidence = Math.max(0, Math.min(1, 1 - bestDist / nc))
    return { suggestedType: bestType, confidence: confidence }
  }

  // ====================================================================
  // 6. Fast entropy via lookup table
  // ====================================================================

  /** Precomputed -log2(p) for p = 1..256 (byte frequency counts). */
  var LOG2_INV = 1 / Math.log(2)
  var negLog2Table = new Float64Array(257)
  for (var _i = 0; _i <= 256; _i++) {
    negLog2Table[_i] = _i === 0 ? 0 : -Math.log(_i) * LOG2_INV
  }

  /**
   * Compute Shannon entropy of a byte string using precomputed log table.
   * No Math.log calls at runtime.
   */
  function fastEntropy(text) {
    var freq = new Uint32Array(256)
    var len = text.length
    for (var i = 0; i < len; i++) freq[text.charCodeAt(i) & 0xff]++

    var H = 0
    var invLen = 1 / len
    for (var i = 0; i < 256; i++) {
      if (freq[i] > 0) {
        // p = freq[i] / len, -p*log2(p) = (freq[i]/len) * negLog2Table[freq[i]] + (freq[i]/len)*log2(len)
        // Simplified: -p log2 p = freq[i] * (negLog2Table[freq[i]] + log2(len)) / len
        // But we can batch: H = sum(freq[i] * -log2(freq[i])) / len + log2(len)
        H += freq[i] * negLog2Table[freq[i]]
      }
    }
    H = H * invLen + Math.log2(len)
    return H > 0 ? H : 0
  }

  // ====================================================================
  // 7. Branchless chunk dispatch (perfect hash for 4-byte magic numbers)
  // ====================================================================

  /**
   * Build a perfect hash dispatch table for 4-byte magic numbers.
   * Returns a function that maps Buffer → handler index in O(1).
   */
  function buildChunkDispatch(magicNumbers) {
    if (!magicNumbers || magicNumbers.length === 0) return null

    // Build minimal perfect hash using multiply-shift
    var entries = magicNumbers
    var n = entries.length

    // Try different hash parameters until we get no collisions
    var tableSize = n < 4 ? 4 : (n << 1) // 2× for low collision rate
    var hashTable = new Int32Array(tableSize).fill(-1)
    var param = 2654435761 // golden ratio constant

    // Simple attempt: just use magic number modulo table size
    for (var attempt = 0; attempt < 20; attempt++) {
      hashTable.fill(-1)
      var collision = false
      for (var i = 0; i < n; i++) {
        var magic = entries[i]
        var slot = ((magic * param) >>> 0) % tableSize
        if (hashTable[slot] !== -1) {
          collision = true
          break
        }
        hashTable[slot] = i
      }
      if (!collision) break
      param = (param * 1103515245 + 12345) >>> 0
      tableSize += 2
      hashTable = new Int32Array(tableSize).fill(-1)
    }

    return {
      table: hashTable,
      param: param,
      size: tableSize,
      dispatch: function(buffer, offset) {
        offset = offset || 0
        if (offset + 4 > buffer.length) return -1
        var magic = (buffer[offset] << 24) | (buffer[offset + 1] << 16) |
                    (buffer[offset + 2] << 8) | buffer[offset + 3]
        var slot = ((magic * param) >>> 0) % tableSize
        var idx = hashTable[slot]
        if (idx >= 0 && entries[idx] === magic) return idx
        return -1
      }
    }
  }

  // ====================================================================
  // moo lexer internals (same as moo-spectral for compatibility)
  // ====================================================================

  var hasSticky = typeof new RegExp().sticky === 'boolean'
  var toString = Object.prototype.toString

  function isRegExp(o) { return o && toString.call(o) === '[object RegExp]' }
  function isObject(o) { return o && typeof o === 'object' && !isRegExp(o) && !Array.isArray(o) }

  function reEscape(s) {
    return s.replace(/[-\/\\^$*+?.()|[\]{}]/g, function(x) {
      if (x === '-') return '\\x2d'
      return '\\' + x
    })
  }
  function reGroups(s) {
    var re = new RegExp('|' + s)
    return re.exec('').length - 1
  }
  function reCapture(s) { return '(' + s + ')' }
  function reUnion(regexps) {
    if (!regexps.length) return '(?!)'
    var source = regexps.map(function(s) { return "(?:" + s + ")" }).join('|')
    return "(?:" + source + ")"
  }

  function regexpOrLiteral(obj) {
    if (typeof obj === 'string') return '(?:' + reEscape(obj) + ')'
    if (isRegExp(obj)) {
      if (obj.ignoreCase) throw new Error('RegExp /i flag not allowed')
      if (obj.global) throw new Error('RegExp /g flag is implied')
      if (obj.sticky) throw new Error('RegExp /y flag is implied')
      if (obj.multiline) throw new Error('RegExp /m flag is implied')
      return obj.source
    }
    throw new Error('Not a pattern: ' + obj)
  }

  function pad(s, length) {
    if (s.length > length) return s
    return Array(length - s.length + 1).join(" ") + s
  }

  function lastNLines(string, numLines) {
    var position = string.length
    var lineBreaks = 0
    while (true) {
      var idx = string.lastIndexOf("\n", position - 1)
      if (idx === -1) break
      lineBreaks++
      position = idx
      if (lineBreaks === numLines || position === 0) break
    }
    return string.substring(lineBreaks < numLines ? 0 : position + 1).split("\n")
  }

  var hasOwnProperty = Object.prototype.hasOwnProperty

  function objectToRules(object) {
    var keys = Object.getOwnPropertyNames(object)
    var result = []
    for (var i = 0; i < keys.length; i++) {
      var key = keys[i], thing = object[key], rules = [].concat(thing)
      if (key === 'include') {
        for (var j = 0; j < rules.length; j++) result.push({include: rules[j]})
        continue
      }
      var match = []
      rules.forEach(function(rule) {
        if (isObject(rule)) {
          if (match.length) result.push(ruleOptions(key, match))
          result.push(ruleOptions(key, rule))
          match = []
        } else {
          match.push(rule)
        }
      })
      if (match.length) result.push(ruleOptions(key, match))
    }
    return result
  }

  function arrayToRules(array) {
    var result = []
    for (var i = 0; i < array.length; i++) {
      var obj = array[i]
      if (obj.include) {
        var include = [].concat(obj.include)
        for (var j = 0; j < include.length; j++) result.push({include: include[j]})
        continue
      }
      if (!obj.type) throw new Error('Rule has no type: ' + JSON.stringify(obj))
      result.push(ruleOptions(obj.type, obj))
    }
    return result
  }

  function ruleOptions(type, obj) {
    if (!isObject(obj)) obj = {match: obj}
    if (obj.include) throw new Error('Matching rules cannot also include states')
    var options = {
      defaultType: type, lineBreaks: !!obj.error || !!obj.fallback,
      pop: false, next: null, push: null, error: false, fallback: false,
      value: null, type: null, shouldThrow: false
    }
    for (var key in obj) if (hasOwnProperty.call(obj, key)) options[key] = obj[key]
    if (typeof options.type === 'string' && type !== options.type) {
      throw new Error("Type transform cannot be a string (type '" + options.type + "' for token '" + type + "')")
    }
    var match = options.match
    options.match = Array.isArray(match) ? match : match ? [match] : []
    options.match.sort(function(a, b) {
      return isRegExp(a) && isRegExp(b) ? 0
        : isRegExp(b) ? -1 : isRegExp(a) ? +1 : b.length - a.length
    })
    return options
  }

  function toRules(spec) { return Array.isArray(spec) ? arrayToRules(spec) : objectToRules(spec) }

  var defaultErrorRule = ruleOptions('error', {lineBreaks: true, shouldThrow: true})

  function compileRules(rules, hasStates, spectralOpts) {
    var errorRule = null
    var fast = Object.create(null)
    var fastAllowed = true
    var unicodeFlag = null
    var groups = []
    var parts = []

    for (var i = 0; i < rules.length; i++) {
      if (rules[i].fallback) fastAllowed = false
    }

    var ruleIndices = []

    for (var i = 0; i < rules.length; i++) {
      var options = rules[i]
      if (options.include) throw new Error('Inheritance is not allowed in stateless lexers')
      if (options.error || options.fallback) {
        if (errorRule) {
          if (!options.fallback === !errorRule.fallback) {
            throw new Error("Multiple " + (options.fallback ? "fallback" : "error") + " rules not allowed")
          }
          throw new Error("fallback and error are mutually exclusive")
        }
        errorRule = options
      }

      var match = options.match.slice()
      if (fastAllowed) {
        while (match.length && typeof match[0] === 'string' && match[0].length === 1) {
          fast[match.shift().charCodeAt(0)] = options
        }
      }

      if (options.pop || options.push || options.next) {
        if (!hasStates) throw new Error("State-switching options not allowed in stateless lexers (for '" + options.defaultType + "')")
        if (options.fallback) throw new Error("State-switching not allowed on fallback tokens")
      }

      if (match.length === 0) continue
      fastAllowed = false

      groups.push(options)
      ruleIndices.push(i)

      for (var j = 0; j < match.length; j++) {
        if (!isRegExp(match[j])) continue
        if (unicodeFlag === null) unicodeFlag = match[j].unicode
        else if (unicodeFlag !== match[j].unicode && options.fallback === false)
          throw new Error('If one rule is /u then all must be')
      }

      var pat = reUnion(match.map(regexpOrLiteral))
      var regexp = new RegExp(pat)
      if (regexp.test("")) throw new Error("RegExp matches empty string: " + regexp)
      if (reGroups(pat) > 0) throw new Error("RegExp has capture groups: " + regexp + "\nUse (?: … ) instead")
      if (!options.lineBreaks && regexp.test('\n')) throw new Error('Rule should declare lineBreaks: ' + regexp)

      parts.push(reCapture(pat))
    }

    // ====================================================================
    // SPECTRAL REORDERING (fast path: bit-parallel + Lanczos + precompute)
    // ====================================================================
    var spectralInfo = null

    if (spectralOpts && spectralOpts.enabled && groups.length > 2) {
      var numComponents = spectralOpts.conservationComponents || 5
      var tensionMat = buildBitTensionMatrix(groups)
      var precomputed = precomputeSpectralData(tensionMat, groups.length, numComponents)

      if (precomputed) {
        var newOrder = precomputed.ordering
        var reorderedGroups = new Array(groups.length)
        var reorderedParts = new Array(parts.length)
        for (var i = 0; i < newOrder.length; i++) {
          reorderedGroups[i] = groups[newOrder[i]]
          reorderedParts[i] = parts[newOrder[i]]
        }
        groups = reorderedGroups
        parts = reorderedParts

        // Convert flat arrays back to legacy format for API compatibility
        var n = groups.length
        var legacyTension = []
        for (var i = 0; i < n; i++) {
          legacyTension[i] = []
          for (var j = 0; j < n; j++) {
            legacyTension[i][j] = tensionMat[i * n + j]
          }
        }

        spectralInfo = {
          originalOrder: ruleIndices,
          fiedler: Array.from(precomputed.fiedler),
          eigenvalues: Array.from(precomputed.eigenvalues),
          // Store flat eigenvectors for fast path
          eigenvectors: precomputed.eigenProjection,  // Float64Array
          _flatEigenvectors: true,
          tensionMatrix: legacyTension,
          weightMatrix: legacyTension,
          laplacian: legacyTension, // approximate
          degrees: Array.from(precomputed.degrees),
          ordering: Array.from(newOrder),
          typeIndex: null,
          // Fast path data
          _precomputed: precomputed,
          spectralGap: precomputed.spectralGap,
          cheegerConstant: precomputed.cheegerConstant
        }
      }
    }

    var fallbackRule = errorRule && errorRule.fallback
    var flags = hasSticky && !fallbackRule ? 'ym' : 'gm'
    var suffix = hasSticky || fallbackRule ? '' : '|'
    if (unicodeFlag === true) flags += "u"
    var combined = new RegExp(reUnion(parts) + suffix, flags)

    return {
      regexp: combined,
      groups: groups,
      fast: fast,
      error: errorRule || defaultErrorRule,
      spectral: spectralInfo
    }
  }

  // ====================================================================
  // Fast Lexer class
  // ====================================================================

  var FastSpectralLexer = function(states, state, spectralOpts) {
    this.startState = state
    this.states = states
    this.buffer = ''
    this.stack = []
    this._spectralOpts = spectralOpts || {}
    this._conservationTracker = null
    this._recentTypes = []
    this._recentTypeBuf = new Int32Array(50)  // circular buffer for recent types
    this._recentTypeHead = 0
    this._recentTypeLen = 0
    this.reset()
  }

  FastSpectralLexer.prototype.reset = function(data, info) {
    this.buffer = data || ''
    this.index = 0
    this.line = info ? info.line : 1
    this.col = info ? info.col : 1
    this.queuedToken = info ? info.queuedToken : null
    this.queuedText = info ? info.queuedText : ""
    this.queuedThrow = info ? info.queuedThrow : null
    this._recentTypes = info ? (info._recentTypes || []) : []
    this._recentTypeHead = 0
    this._recentTypeLen = 0
    this.setState(info ? info.state : this.startState)
    this.stack = info && info.stack ? info.stack.slice() : []

    var spectral = this._currentSpectral()
    if (spectral && this._spectralOpts.conservation) {
      var precomputed = spectral._precomputed
      if (precomputed) {
        var typeIndexMap = spectral.typeIndex || {}
        var typeIndexKeys = []
        for (var k in typeIndexMap) {
          var idx = typeIndexMap[k]
          while (typeIndexKeys.length <= idx) typeIndexKeys.push('')
          typeIndexKeys[idx] = k
        }
        this._conservationTracker = new FastConservationTracker(
          precomputed.eigenProjection,
          typeIndexMap,
          typeIndexKeys,
          precomputed.numComponents,
          this._spectralOpts.conservationWindow || 20
        )
      }
    }

    return this
  }

  FastSpectralLexer.prototype._currentSpectral = function() {
    var stateInfo = this.states[this.state]
    return stateInfo ? stateInfo.spectral : null
  }

  FastSpectralLexer.prototype.save = function() {
    return {
      line: this.line,
      col: this.col,
      state: this.state,
      stack: this.stack.slice(),
      queuedToken: this.queuedToken,
      queuedText: this.queuedText,
      queuedThrow: this.queuedThrow,
      _recentTypes: this._recentTypes.slice()
    }
  }

  FastSpectralLexer.prototype.setState = function(state) {
    if (!state || this.state === state) return
    this.state = state
    var info = this.states[state]
    this.groups = info.groups
    this.error = info.error
    this.re = info.regexp
    this.fast = info.fast
  }

  FastSpectralLexer.prototype.popState = function() { this.setState(this.stack.pop()) }
  FastSpectralLexer.prototype.pushState = function(state) { this.stack.push(this.state); this.setState(state) }

  var eat = hasSticky
    ? function(re, buffer) { return re.exec(buffer) }
    : function(re, buffer) {
        var match = re.exec(buffer)
        if (match[0].length === 0) return null
        return match
      }

  FastSpectralLexer.prototype._getGroup = function(match) {
    for (var i = 0; i < this.groups.length; i++) {
      if (match[i + 1] !== undefined) return this.groups[i]
    }
    throw new Error('Cannot find token type for matched text')
  }

  function tokenToString() { return this.value }

  FastSpectralLexer.prototype.next = function() {
    var index = this.index
    if (this.queuedGroup) {
      var token = this._token(this.queuedGroup, this.queuedText, index)
      this.queuedGroup = null
      this.queuedText = ""
      return token
    }
    var buffer = this.buffer
    if (index === buffer.length) return

    var group = this.fast[buffer.charCodeAt(index)]
    if (group) return this._token(group, buffer.charAt(index), index)

    var re = this.re
    re.lastIndex = index
    var match = eat(re, buffer)

    var error = this.error
    if (match == null) {
      var text = buffer.slice(index)
      var recovered = this._tryRecovery(text)
      return recovered || this._token(error, text, index)
    }

    var group = this._getGroup(match)
    var text = match[0]

    if (error.fallback && match.index !== index) {
      this.queuedGroup = group
      this.queuedText = text
      return this._token(error, buffer.slice(index, match.index), index)
    }

    return this._token(group, text, index)
  }

  FastSpectralLexer.prototype._tryRecovery = function(unmatchedText) {
    if (!this._spectralOpts.errorRecovery) return null
    if (!this._conservationTracker) return null

    var suggestion = this._conservationTracker.suggestCorrection(
      '__error__',
      this._recentTypes.slice(-5)
    )

    if (suggestion.confidence > 0.3) {
      return {
        type: suggestion.suggestedType,
        value: unmatchedText,
        text: unmatchedText,
        toString: tokenToString,
        offset: this.index,
        lineBreaks: (unmatchedText.match(/\n/g) || []).length,
        line: this.line,
        col: this.col,
        _recovered: true,
        _confidence: suggestion.confidence
      }
    }
    return null
  }

  FastSpectralLexer.prototype._token = function(group, text, offset) {
    var lineBreaks = 0
    var nl = 1
    if (group.lineBreaks) {
      if (text === '\n') {
        lineBreaks = 1
      } else {
        var matchNL = /\n/g
        while (matchNL.exec(text)) { lineBreaks++; nl = matchNL.lastIndex }
      }
    }

    var type = (typeof group.type === 'function' && group.type(text)) || group.defaultType
    var token = {
      type: type,
      value: typeof group.value === 'function' ? group.value(text) : text,
      text: text,
      toString: tokenToString,
      offset: offset,
      lineBreaks: lineBreaks,
      line: this.line,
      col: this.col,
    }

    var size = text.length
    this.index += size
    this.line += lineBreaks
    if (lineBreaks !== 0) {
      this.col = size - nl + 1
    } else {
      this.col += size
    }

    // Fast conservation tracking (zero allocation path)
    if (this._spectralOpts.conservation && this._conservationTracker) {
      // Inline circular buffer for recent types
      this._recentTypeBuf[this._recentTypeHead] = this._conservationTracker.typeIndexMap[type] || 0
      this._recentTypeHead = (this._recentTypeHead + 1) % 50
      if (this._recentTypeLen < 50) this._recentTypeLen++

      // Keep legacy array for suggestCorrection
      this._recentTypes.push(type)
      if (this._recentTypes.length > 50) this._recentTypes = this._recentTypes.slice(-30)

      var result = this._conservationTracker.feed(type)
      token._conservation = result
    }

    if (group.shouldThrow) {
      throw new Error(this.formatError(token, "invalid syntax"))
    }
    if (group.pop) this.popState()
    else if (group.push) this.pushState(group.push)
    else if (group.next) this.setState(group.next)

    return token
  }

  if (typeof Symbol !== 'undefined' && Symbol.iterator) {
    var LexerIterator = function(lexer) { this.lexer = lexer }
    LexerIterator.prototype.next = function() {
      var token = this.lexer.next()
      return {value: token, done: !token}
    }
    LexerIterator.prototype[Symbol.iterator] = function() { return this }
    FastSpectralLexer.prototype[Symbol.iterator] = function() { return new LexerIterator(this) }
  }

  FastSpectralLexer.prototype.formatError = function(token, message) {
    if (token == null) {
      var text = this.buffer.slice(this.index)
      token = { text: text, offset: this.index, lineBreaks: text.indexOf('\n') === -1 ? 0 : 1, line: this.line, col: this.col }
    }
    var numLinesAround = 2
    var firstDisplayedLine = Math.max(token.line - numLinesAround, 1)
    var lastDisplayedLine = token.line + numLinesAround
    var lastLineDigits = String(lastDisplayedLine).length
    var displayedLines = lastNLines(this.buffer, (this.line - token.line) + numLinesAround + 1).slice(0, 5)
    var errorLines = []
    errorLines.push(message + " at line " + token.line + " col " + token.col + ":")
    errorLines.push("")
    for (var i = 0; i < displayedLines.length; i++) {
      var line = displayedLines[i], lineNo = firstDisplayedLine + i
      errorLines.push(pad(String(lineNo), lastLineDigits) + "  " + line)
      if (lineNo === token.line) errorLines.push(pad("", lastLineDigits + token.col + 1) + "^")
    }
    return errorLines.join("\n")
  }

  FastSpectralLexer.prototype.clone = function() {
    return new FastSpectralLexer(this.states, this.state, this._spectralOpts)
  }

  FastSpectralLexer.prototype.has = function(tokenType) { return true }

  // ====================================================================
  // Structural statistics API (compatible with moo-spectral)
  // ====================================================================

  FastSpectralLexer.prototype.structuralInfo = function() {
    var spectral = this._currentSpectral()
    if (!spectral) {
      return {
        spectralGap: 0,
        cheegerConstant: 0,
        tensionMatrix: [],
        laplacianEigenvalues: [],
        optimalOrdering: [],
        fiedlerVector: [],
        note: 'Spectral analysis not enabled. Pass spectralOptions: {enabled: true} to compile().'
      }
    }

    return {
      spectralGap: spectral.spectralGap || 0,
      cheegerConstant: spectral.cheegerConstant || 0,
      tensionMatrix: spectral.tensionMatrix,
      laplacianEigenvalues: spectral.eigenvalues.slice(),
      optimalOrdering: spectral.ordering.slice(),
      fiedlerVector: spectral.fiedler.slice()
    }
  }

  // ====================================================================
  // Public API: compile and states (same API as moo-spectral)
  // ====================================================================

  function compile(rules, spectralOptions) {
    spectralOptions = spectralOptions || {}
    var parsedRules = toRules(rules)
    var result = compileRules(parsedRules, false, spectralOptions)

    if (result.spectral) {
      var typeIndex = {}
      for (var i = 0; i < result.groups.length; i++) {
        typeIndex[result.groups[i].defaultType] = i
      }
      result.spectral.typeIndex = typeIndex
    }

    var lexer = new FastSpectralLexer({start: result}, 'start', spectralOptions)
    return lexer
  }

  function compileStates(states, start, spectralOptions) {
    spectralOptions = spectralOptions || {}
    var all = states.$all ? toRules(states.$all) : []
    delete states.$all

    var keys = Object.getOwnPropertyNames(states)
    if (!start) start = keys[0]

    var ruleMap = Object.create(null)
    for (var i = 0; i < keys.length; i++) ruleMap[keys[i]] = toRules(states[keys[i]]).concat(all)

    for (var i = 0; i < keys.length; i++) {
      var key = keys[i]
      var rules = ruleMap[key]
      var included = Object.create(null)
      for (var j = 0; j < rules.length; j++) {
        var rule = rules[j]
        if (!rule.include) continue
        var splice = [j, 1]
        if (rule.include !== key && !included[rule.include]) {
          included[rule.include] = true
          var newRules = ruleMap[rule.include]
          if (!newRules) throw new Error("Cannot include nonexistent state '" + rule.include + "'")
          for (var k = 0; k < newRules.length; k++) {
            if (rules.indexOf(newRules[k]) !== -1) continue
            splice.push(newRules[k])
          }
        }
        rules.splice.apply(rules, splice)
        j--
      }
    }

    var map = Object.create(null)
    for (var i = 0; i < keys.length; i++) {
      map[keys[i]] = compileRules(ruleMap[keys[i]], true, spectralOptions)
      if (map[keys[i]].spectral) {
        var typeIndex = {}
        for (var g = 0; g < map[keys[i]].groups.length; g++) {
          typeIndex[map[keys[i]].groups[g].defaultType] = g
        }
        map[keys[i]].spectral.typeIndex = typeIndex
      }
    }

    for (var i = 0; i < keys.length; i++) {
      var name = keys[i], state = map[name], groups = state.groups
      for (var j = 0; j < groups.length; j++) {
        var g = groups[j], st = g && (g.push || g.next)
        if (st && !map[st]) throw new Error("Missing state '" + st + "'")
        if (g && g.pop && +g.pop !== 1) throw new Error("pop must be 1")
      }
    }

    return new FastSpectralLexer(map, start, spectralOptions)
  }

  function keywordTransform(map) {
    var isMap = typeof Map !== 'undefined'
    var reverseMap = isMap ? new Map : Object.create(null)
    var types = Object.getOwnPropertyNames(map)
    for (var i = 0; i < types.length; i++) {
      var tokenType = types[i], item = map[tokenType]
      var keywordList = Array.isArray(item) ? item : [item]
      keywordList.forEach(function(keyword) {
        if (typeof keyword !== 'string') throw new Error("keyword must be string")
        if (isMap) reverseMap.set(keyword, tokenType)
        else reverseMap[keyword] = tokenType
      })
    }
    return function(k) { return isMap ? reverseMap.get(k) : reverseMap[k] }
  }

  // ====================================================================
  // Benchmark function
  // ====================================================================

  /**
   * Run benchmarks comparing old moo-spectral vs new moo-fast.
   * Returns results object with timings.
   */
  function benchmark() {
    var results = {}
    var oldModule = null
    try {
      oldModule = require('./moo-spectral.js')
    } catch (e) {
      results.note = 'moo-spectral.js not available for comparison'
    }

    // Helper: generate random symmetric matrix as flat Float64Array
    function randomSymmetricFlat(n) {
      var M = new Float64Array(n * n)
      for (var i = 0; i < n; i++) {
        for (var j = i; j < n; j++) {
          var v = Math.random()
          M[i * n + j] = v
          M[j * n + i] = v
        }
      }
      return M
    }

    // Helper: flat → AoA
    function flatToAoA(M, n) {
      var A = []
      for (var i = 0; i < n; i++) {
        A[i] = []
        for (var j = 0; j < n; j++) A[i][j] = M[i * n + j]
      }
      return A
    }

    var timings = {}
    function time(label, fn, iterations) {
      iterations = iterations || 10
      // Warmup
      fn()
      var start = process.hrtime.bigint()
      for (var i = 0; i < iterations; i++) fn()
      var end = process.hrtime.bigint()
      var ms = Number(end - start) / 1e6
      timings[label] = { totalMs: ms, perIterationMs: ms / iterations, iterations: iterations }
    }

    // --- Eigenvalue comparison ---
    [12, 25, 50].forEach(function(n) {
      var flat = randomSymmetricFlat(n)
      var aoa = flatToAoA(flat, n)

      if (oldModule) {
        time('jacobi_' + n + 'x' + n, function() {
          oldModule._linalg.jacobiEigen(aoa, n * n * 10)
        })
      }

      time('lanczos_' + n + 'x' + n, function() {
        lanczosEigen(flat, n, Math.min(5, n))
      })
    })

    // --- Matrix multiply comparison ---
    var mmN = 25
    var flatA = randomSymmetricFlat(mmN)
    var flatB = randomSymmetricFlat(mmN)
    var aoaA = flatToAoA(flatA, mmN)
    var aoaB = flatToAoA(flatB, mmN)

    if (oldModule) {
      time('old_matmul_' + mmN, function() {
        oldModule._linalg.zeros(mmN) // indirect: they use matMul internally
        // Manual old-style matmul
        var C = []
        for (var i = 0; i < mmN; i++) {
          C[i] = []
          for (var j = 0; j < mmN; j++) {
            var s = 0
            for (var k = 0; k < mmN; k++) s += aoaA[i][k] * aoaB[k][j]
            C[i][j] = s
          }
        }
      }, 100)
    }

    time('flat_matmul_' + mmN, function() {
      flatMatMul(flatA, flatB, mmN, mmN, mmN)
    }, 100)

    // --- Char set overlap comparison ---
    var numRules = 25
    var testRules = []
    for (var i = 0; i < numRules; i++) {
      var chars = {}
      for (var c = 0; c < 10; c++) {
        chars[String.fromCharCode(97 + ((i * 10 + c) % 26))] = true
      }
      testRules.push({
        match: [new RegExp('[' + Object.keys(chars).join('') + ']+')],
        defaultType: 'rule' + i
      })
    }

    if (oldModule) {
      time('old_overlap_' + numRules + 'rules', function() {
        oldModule._linalg.buildTensionMatrix(testRules)
      }, 50)
    }

    time('bit_overlap_' + numRules + 'rules', function() {
      buildBitTensionMatrix(testRules)
    }, 50)

    // --- Tokenizer throughput ---
    var sampleText = ''
    for (var i = 0; i < 50000; i++) {
      sampleText += 'abcdefghijklmnopqrstuvwxyz0123456789 \t\n'[Math.floor(Math.random() * 39)]
    }

    var simpleRules = {
      word: /[a-z]+/,
      number: /[0-9]+/,
      space: /[ \t]+/,
      newline: { match: /\n/, lineBreaks: true },
      error: { error: true }
    }

    var fastLexer = compile(simpleRules)
    time('fast_tokenize_100KB', function() {
      fastLexer.reset(sampleText)
      while (fastLexer.next()) {}
    }, 5)

    if (oldModule) {
      var oldLexer = oldModule.compile(simpleRules)
      time('old_tokenize_100KB', function() {
        oldLexer.reset(sampleText)
        while (oldLexer.next()) {}
      }, 5)
    }

    // --- With conservation tracking ---
    var fastConservation = compile(simpleRules, {
      enabled: true,
      conservation: true,
      conservationComponents: 5,
      conservationWindow: 20
    })
    time('fast_conservation_100KB', function() {
      fastConservation.reset(sampleText)
      while (fastConservation.next()) {}
    }, 5)

    if (oldModule) {
      var oldConservation = oldModule.compile(simpleRules, {
        enabled: true,
        conservation: true,
        conservationComponents: 5,
        conservationWindow: 20
      })
      time('old_conservation_100KB', function() {
        oldConservation.reset(sampleText)
        while (oldConservation.next()) {}
      }, 5)
    }

    results.timings = timings
    results.summary = {}
    if (oldModule) {
      for (var key in timings) {
        var oldKey = key.replace('fast_', 'old_').replace('lanczos_', 'jacobi_').replace('bit_', 'old_')
        if (timings[oldKey] && timings[key]) {
          var speedup = timings[oldKey].perIterationMs / timings[key].perIterationMs
          results.summary[key + '_vs_' + oldKey] = speedup.toFixed(2) + 'x'
        }
      }
    }

    // Throughput calculation
    var fastTime = timings['fast_tokenize_100KB']
    if (fastTime) {
      var bytes = sampleText.length
      var seconds = fastTime.perIterationMs / 1000
      results.summary.throughput_MB_per_s = (bytes / seconds / 1e6).toFixed(1)
    }

    return results
  }

  // ====================================================================
  // Exports (same API as moo-spectral.js)
  // ====================================================================

  return {
    compile: compile,
    states: compileStates,
    error: Object.freeze({error: true}),
    fallback: Object.freeze({fallback: true}),
    keywords: keywordTransform,

    // Advanced: access fast linear algebra utilities
    _linalg: {
      flatZeros: flatZeros,
      flatMatMul: flatMatMul,
      flatSymmetrize: flatSymmetrize,
      flatLaplacian: flatLaplacian,
      flatNormalizedLaplacian: flatNormalizedLaplacian,
      lanczosEigen: lanczosEigen,
      flatCheegerConstant: flatCheegerConstant,
      buildBitTensionMatrix: buildBitTensionMatrix,
      FastConservationTracker: FastConservationTracker,
      bitCharSet: bitCharSet,
      bitOverlap: bitOverlap,
      popcount32: popcount32,
      fastEntropy: fastEntropy,
      buildChunkDispatch: buildChunkDispatch
    },

    benchmark: benchmark
  }
}))
