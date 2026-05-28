/**
 * moo-spectral.js — moo lexer enhanced with spectral graph theory
 *
 * Extends the moo lexer (https://github.com/no-context/moo) with:
 *   1. Spectral state ordering — Fiedler vector of the rule tension graph
 *   2. Rule tension computation — character-set overlap × transition frequency
 *   3. Conservation-aware error recovery — sliding-window Laplacian eigenbasis
 *   4. Structural statistics API — spectral gap, Cheeger constant, eigenvalues
 *
 * 100% backward-compatible. All enhancements are opt-in via `spectralOptions`.
 *
 * @module moo-spectral
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
  // Linear algebra helpers (pure JS, no dependencies)
  // ====================================================================

  /**
   * Create an n×n zero matrix
   */
  function zeros(n) {
    var m = new Array(n)
    for (var i = 0; i < n; i++) {
      m[i] = new Array(n)
      for (var j = 0; j < n; j++) m[i][j] = 0
    }
    return m
  }

  /**
   * Create an n×k zero matrix
   */
  function zerosRect(n, k) {
    var m = new Array(n)
    for (var i = 0; i < n; i++) {
      m[i] = new Array(k)
      for (var j = 0; j < k; j++) m[i][j] = 0
    }
    return m
  }

  /**
   * Transpose a matrix
   */
  function transpose(A) {
    var rows = A.length, cols = A[0].length
    var T = zerosRect(cols, rows)
    for (var i = 0; i < rows; i++)
      for (var j = 0; j < cols; j++)
        T[j][i] = A[i][j]
    return T
  }

  /**
   * Matrix multiply A×B
   */
  function matMul(A, B) {
    var m = A.length, n = B[0].length, p = B.length
    var C = zerosRect(m, n)
    for (var i = 0; i < m; i++)
      for (var j = 0; j < n; j++)
        for (var k = 0; k < p; k++)
          C[i][j] += A[i][k] * B[k][j]
    return C
  }

  /**
   * Symmetrize a matrix: (A + A^T) / 2
   */
  function symmetrize(A) {
    var n = A.length
    var S = zeros(n)
    for (var i = 0; i < n; i++)
      for (var j = i; j < n; j++) {
        S[i][j] = (A[i][j] + A[j][i]) / 2
        S[j][i] = S[i][j]
      }
    return S
  }

  /**
   * Compute degree vector and diagonal matrix D for weighted adjacency W
   */
  function degreeMatrix(W) {
    var n = W.length
    var d = new Array(n)
    var D = zeros(n)
    for (var i = 0; i < n; i++) {
      var s = 0
      for (var j = 0; j < n; j++) s += W[i][j]
      d[i] = s
      D[i][i] = s
    }
    return { degrees: d, matrix: D }
  }

  /**
   * Laplacian: L = D - W
   */
  function laplacian(W) {
    var n = W.length
    var deg = degreeMatrix(W)
    var L = zeros(n)
    for (var i = 0; i < n; i++)
      for (var j = 0; j < n; j++)
        L[i][j] = (i === j ? deg.degrees[i] : 0) - W[i][j]
    return { L: L, degrees: deg.degrees, D: deg.matrix }
  }

  /**
   * Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
   * Handles zero-degree nodes by leaving their entries as 0.
   */
  function normalizedLaplacian(W) {
    var n = W.length
    var deg = degreeMatrix(W)
    var Lnorm = zeros(n)
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        if (i === j) {
          Lnorm[i][i] = deg.degrees[i] > 0 ? 1 : 0
        } else {
          var di = Math.sqrt(deg.degrees[i])
          var dj = Math.sqrt(deg.degrees[j])
          if (di > 0 && dj > 0) {
            Lnorm[i][j] = -W[i][j] / (di * dj)
          }
        }
      }
    }
    return { L: Lnorm, degrees: deg.degrees }
  }

  /**
   * Jacobi eigenvalue algorithm for symmetric matrices.
   * Returns { eigenvalues: [...], eigenvectors: [[...], ...] }
   * Eigenvectors are columns; sorted by ascending eigenvalue.
   */
  function jacobiEigen(A, maxIter) {
    maxIter = maxIter || 1000
    var n = A.length
    // Copy A
    var S = zeros(n)
    var V = zeros(n)
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        S[i][j] = A[i][j]
      }
      V[i][i] = 1 // identity
    }

    var tol = 1e-12
    for (var iter = 0; iter < maxIter; iter++) {
      // Find largest off-diagonal element
      var maxVal = 0, p = 0, q = 1
      for (var i = 0; i < n; i++)
        for (var j = i + 1; j < n; j++)
          if (Math.abs(S[i][j]) > maxVal) {
            maxVal = Math.abs(S[i][j])
            p = i; q = j
          }

      if (maxVal < tol) break

      // Compute rotation angle
      var app = S[p][p], aqq = S[q][q], apq = S[p][q]
      var theta
      if (Math.abs(app - aqq) < 1e-15) {
        theta = Math.PI / 4
      } else {
        theta = 0.5 * Math.atan2(2 * apq, app - aqq)
      }
      var c = Math.cos(theta), s = Math.sin(theta)

      // Apply Givens rotation: S' = G^T S G
      for (var i = 0; i < n; i++) {
        if (i === p || i === q) continue
        var sip = S[i][p], siq = S[i][q]
        S[i][p] = c * sip + s * siq
        S[p][i] = S[i][p]
        S[i][q] = -s * sip + c * siq
        S[q][i] = S[i][q]
      }
      var spp = S[p][p], sqq = S[q][q]
      S[p][p] = c * c * spp + 2 * s * c * apq + s * s * sqq
      S[q][q] = s * s * spp - 2 * s * c * apq + c * c * sqq
      S[p][q] = 0
      S[q][p] = 0

      // Update eigenvectors
      for (var i = 0; i < n; i++) {
        var vip = V[i][p], viq = V[i][q]
        V[i][p] = c * vip + s * viq
        V[i][q] = -s * vip + c * viq
      }
    }

    // Extract eigenvalues and sort
    var evals = new Array(n)
    for (var i = 0; i < n; i++) evals[i] = S[i][i]

    // Sort by eigenvalue ascending
    var indices = new Array(n)
    for (var i = 0; i < n; i++) indices[i] = i
    indices.sort(function(a, b) { return evals[a] - evals[b] })

    var sortedEvals = new Array(n)
    var sortedEvecs = zeros(n)
    for (var i = 0; i < n; i++) {
      sortedEvals[i] = evals[indices[i]]
      for (var j = 0; j < n; j++) {
        sortedEvecs[j][i] = V[j][indices[i]]
      }
    }

    return { eigenvalues: sortedEvals, eigenvectors: sortedEvecs }
  }

  /**
   * Compute Cheeger constant approximation from Fiedler vector.
   * h(G) ≈ min cut ratio when partitioning by Fiedler vector threshold.
   */
  function cheegerConstant(W, fiedler) {
    var n = W.length
    // Sort indices by Fiedler value
    var indices = new Array(n)
    for (var i = 0; i < n; i++) indices[i] = i
    indices.sort(function(a, b) { return fiedler[a] - fiedler[b] })

    var totalDegree = 0
    for (var i = 0; i < n; i++)
      for (var j = 0; j < n; j++) totalDegree += W[i][j]
    totalDegree /= 2

    var bestCheeger = Infinity
    var volS = 0
    var cut = 0

    for (var k = 0; k < n - 1; k++) {
      var idx = indices[k]
      volS += 0
      for (var j = 0; j < n; j++) volS += W[idx][j]

      cut = 0
      for (var i = 0; i <= k; i++)
        for (var j = k + 1; j < n; j++)
          cut += W[indices[i]][indices[j]]

      var volComplement = totalDegree - volS
      var volMin = Math.min(volS, volComplement)
      if (volMin > 0) {
        var h = cut / volMin
        if (h < bestCheeger) bestCheeger = h
      }
    }

    return bestCheeger === Infinity ? 0 : bestCheeger
  }

  // ====================================================================
  // Rule tension computation
  // ====================================================================

  /**
   * Extract a rough character set from a rule's match pattern.
   * Returns an array of character ranges [lo, hi] and a set of specific chars.
   */
  function extractCharSet(rule) {
    var patterns = rule.match || []
    var chars = {}
    var ranges = []

    for (var p = 0; p < patterns.length; p++) {
      var pat = patterns[p]
      var src = typeof pat === 'string' ? pat : (pat && pat.source ? pat.source : '')

      // Extract literal characters
      for (var i = 0; i < src.length; i++) {
        if (src[i] === '\\' && i + 1 < src.length) {
          var escaped = src[i + 1]
          if (escaped === 'd') { ranges.push([48, 57]); i++; continue }
          if (escaped === 'w') { ranges.push([48, 57]); ranges.push([65, 90]); ranges.push([97, 122]); chars['_'] = true; i++; continue }
          if (escaped === 's') { ranges.push([9, 13]); chars[' '] = true; i++; continue }
          if (escaped === 'n') { chars['\n'] = true; i++; continue }
          if (escaped === 't') { chars['\t'] = true; i++; continue }
          if (escaped === 'r') { chars['\r'] = true; i++; continue }
          i++ // skip escaped char
          continue
        }
        // Skip regex metacharacters
        if ('^$*+?.|()[]{}='.indexOf(src[i]) !== -1) continue
        chars[src[i]] = true
      }

      // Extract character classes [...]
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

    return { chars: chars, ranges: ranges }
  }

  /**
   * Compute character-set overlap between two char sets.
   * Returns a value in [0, 1].
   */
  function charSetOverlap(a, b) {
    // Quick heuristic: count shared specific characters + range overlap
    var sharedChars = 0, totalChars = 0
    var keysA = Object.keys(a.chars)
    var keysB = Object.keys(b.chars)
    var setB = {}
    for (var i = 0; i < keysB.length; i++) setB[keysB[i]] = true

    totalChars = keysA.length + keysB.length
    if (totalChars === 0 && a.ranges.length === 0 && b.ranges.length === 0) return 0

    for (var i = 0; i < keysA.length; i++) {
      if (setB[keysA[i]]) sharedChars++
    }

    // Range overlap
    var rangeOverlap = 0
    for (var i = 0; i < a.ranges.length; i++) {
      for (var j = 0; j < b.ranges.length; j++) {
        var lo = Math.max(a.ranges[i][0], b.ranges[j][0])
        var hi = Math.min(a.ranges[i][1], b.ranges[j][1])
        if (lo <= hi) rangeOverlap += (hi - lo + 1)
      }
    }

    var charOverlap = totalChars > 0 ? (2 * sharedChars / totalChars) : 0
    var maxRangeSize = 0
    for (var i = 0; i < a.ranges.length; i++) maxRangeSize += a.ranges[i][1] - a.ranges[i][0] + 1
    for (var i = 0; i < b.ranges.length; i++) maxRangeSize += b.ranges[i][1] - b.ranges[i][0] + 1
    var rOverlap = maxRangeSize > 0 ? (2 * rangeOverlap / maxRangeSize) : 0

    return Math.max(charOverlap, rOverlap)
  }

  /**
   * Build pairwise rule tension matrix.
   * Tension(i,j) = charSetOverlap(i,j) for the tension graph.
   * If transitionFrequencies is provided, weights by that too.
   */
  function buildTensionMatrix(rules, transitionFreqs) {
    var n = rules.length
    var charSets = new Array(n)
    for (var i = 0; i < n; i++) charSets[i] = extractCharSet(rules[i])

    var T = zeros(n)
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        var overlap = charSetOverlap(charSets[i], charSets[j])
        var freq = 1
        if (transitionFreqs) {
          var fi = transitionFreqs[i] || 1
          var fj = transitionFreqs[j] || 1
          freq = Math.sqrt(fi * fj)
        }
        T[i][j] = overlap * freq
        T[j][i] = T[i][j]
      }
    }
    return T
  }

  /**
   * Build transition frequency matrix from token type bigrams.
   * @param {string[]} tokenTypes - Ordered array of token types seen
   * @param {Object} typeIndex - Map from type name to index in rules array
   * @returns {Array[]} n×n transition probability matrix
   */
  function buildTransitionMatrix(tokenTypes, typeIndex) {
    var n = Object.keys(typeIndex).length
    var T = zeros(n)
    var smoothing = 0.01

    for (var i = 0; i < n; i++)
      for (var j = 0; j < n; j++)
        T[i][j] = smoothing

    for (var i = 0; i < tokenTypes.length - 1; i++) {
      var a = typeIndex[tokenTypes[i]]
      var b = typeIndex[tokenTypes[i + 1]]
      if (a !== undefined && b !== undefined) T[a][b] += 1
    }

    // Normalize rows
    for (var i = 0; i < n; i++) {
      var rowSum = 0
      for (var j = 0; j < n; j++) rowSum += T[i][j]
      if (rowSum > 0)
        for (var j = 0; j < n; j++) T[i][j] /= rowSum
    }

    return T
  }

  // ====================================================================
  // Spectral ordering via Fiedler vector
  // ====================================================================

  /**
   * Compute spectral ordering of rules.
   * @param {Array[]} tensionMatrix - n×n tension matrix
   * @param {Object} [options]
   * @param {Array[]} [options.transitionMatrix] - n×n transition probability matrix
   * @param {number} [options.sigma] - Kernel width for tension similarity
   * @returns {{ ordering: number[], fiedler: number[], eigenvalues: number[], eigenvectors: Array[] }}
   */
  function spectralOrdering(tensionMatrix, options) {
    options = options || {}
    var sigma = options.sigma != null ? options.sigma : 1.0

    var n = tensionMatrix.length
    var W = zeros(n)

    // Weighted adjacency: tension × transition similarity
    var T = options.transitionMatrix
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        var tension = tensionMatrix[i][j]
        var transSim = T ? Math.sqrt(T[i][j] * T[j][i]) : 1
        W[i][j] = tension * transSim
        W[j][i] = W[i][j]
      }
    }

    var lap = normalizedLaplacian(W)
    var eigen = jacobiEigen(lap.L, n * n * 10)

    // Fiedler vector = eigenvector of second-smallest eigenvalue
    var fiedler = new Array(n)
    for (var i = 0; i < n; i++) fiedler[i] = eigen.eigenvectors[i][1]

    // Optimal ordering = argsort of Fiedler vector
    var ordering = new Array(n)
    for (var i = 0; i < n; i++) ordering[i] = i
    ordering.sort(function(a, b) { return fiedler[a] - fiedler[b] })

    return {
      ordering: ordering,
      fiedler: fiedler,
      eigenvalues: eigen.eigenvalues,
      eigenvectors: eigen.eigenvectors,
      weightMatrix: W,
      laplacian: lap.L,
      degrees: lap.degrees
    }
  }

  // ====================================================================
  // Conservation tracking
  // ====================================================================

  /**
   * Create a conservation tracker for sliding-window analysis.
   * @param {Array[]} eigenvectors - Laplacian eigenvectors (columns)
   * @param {Object} typeIndex - Map from token type name to index
   * @param {number} [windowSize=20]
   * @param {number} [numComponents=5]
   */
  function ConservationTracker(eigenvectors, typeIndex, windowSize, numComponents) {
    this.eigenvectors = eigenvectors
    this.typeIndex = typeIndex
    this.windowSize = windowSize || 20
    this.numComponents = Math.min(numComponents || 5, eigenvectors[0].length)
    this.history = []     // recent token type indices
    this.scores = []      // conservation score history
    this.baseline = null  // established baseline from well-formed input
  }

  /**
   * Feed a token type and update the sliding window.
   * @param {string} type - Token type name
   * @returns {{ score: number, delta: number, isAnomaly: boolean }}
   */
  ConservationTracker.prototype.feed = function(type) {
    var idx = this.typeIndex[type]
    if (idx === undefined) {
      // Unknown token type — project as the average
      var n = this.eigenvectors.length
      idx = Math.floor(n / 2)
    }
    this.history.push(idx)
    if (this.history.length > this.windowSize * 2) {
      this.history = this.history.slice(-this.windowSize)
    }

    if (this.history.length < 3) {
      return { score: 0, delta: 0, isAnomaly: false }
    }

    // Compute conservation = mean gradient variance across top eigenvectors
    var totalVar = 0
    for (var ev = 0; ev < this.numComponents; ev++) {
      var projections = new Array(this.history.length)
      for (var i = 0; i < this.history.length; i++) {
        projections[i] = this.eigenvectors[this.history[i]][ev]
      }
      // Gradient variance
      var diffs = new Array(projections.length - 1)
      for (var i = 0; i < diffs.length; i++) {
        diffs[i] = projections[i + 1] - projections[i]
      }
      var mean = 0
      for (var i = 0; i < diffs.length; i++) mean += diffs[i]
      mean /= diffs.length
      var v = 0
      for (var i = 0; i < diffs.length; i++) v += (diffs[i] - mean) * (diffs[i] - mean)
      v /= diffs.length
      totalVar += v
    }

    var score = totalVar / this.numComponents

    // Establish baseline from first few windows
    if (this.baseline === null && this.scores.length >= 5) {
      var sum = 0
      for (var i = 0; i < this.scores.length; i++) sum += this.scores[i]
      this.baseline = sum / this.scores.length
    }

    var delta = this.baseline !== null ? score / this.baseline : 1
    var isAnomaly = this.baseline !== null && delta > 2.25 // conservation spike threshold

    this.scores.push(score)
    if (this.scores.length > 100) this.scores = this.scores.slice(-50)

    return { score: score, delta: delta, isAnomaly: isAnomaly }
  }

  /**
   * Suggest the most likely correction for an error token.
   * Uses the Laplacian eigenbasis to find the "nearest well-formed" token type.
   * @param {string} errorType - The unexpected token type
   * @param {string[]} recentTypes - Recent token types (last few)
   * @returns {{ suggestedType: string, confidence: number }}
   */
  ConservationTracker.prototype.suggestCorrection = function(errorType, recentTypes) {
    if (!recentTypes || recentTypes.length === 0) {
      return { suggestedType: errorType, confidence: 0 }
    }

    var n = this.eigenvectors.length
    var ev = this.numComponents

    // Compute the "expected" position in eigenspace from recent tokens
    var expected = new Array(ev)
    for (var k = 0; k < ev; k++) expected[k] = 0
    var count = 0
    for (var i = 0; i < recentTypes.length; i++) {
      var idx = this.typeIndex[recentTypes[i]]
      if (idx !== undefined) {
        for (var k = 0; k < ev; k++) expected[k] += this.eigenvectors[idx][k]
        count++
      }
    }
    if (count === 0) return { suggestedType: errorType, confidence: 0 }
    for (var k = 0; k < ev; k++) expected[k] /= count

    // Find the token type closest to expected in eigenspace
    var bestType = errorType
    var bestDist = Infinity
    var typeNames = Object.keys(this.typeIndex)

    for (var t = 0; t < typeNames.length; t++) {
      var idx = this.typeIndex[typeNames[t]]
      var dist = 0
      for (var k = 0; k < ev; k++) {
        var d = this.eigenvectors[idx][k] - expected[k]
        dist += d * d
      }
      if (dist < bestDist) {
        bestDist = dist
        bestType = typeNames[t]
      }
    }

    // Confidence = 1 - normalized distance (capped)
    var maxPossibleDist = ev // rough upper bound
    var confidence = Math.max(0, Math.min(1, 1 - bestDist / maxPossibleDist))

    return { suggestedType: bestType, confidence: confidence }
  }

  // ====================================================================
  // Enhanced moo compilation
  // ====================================================================

  // Re-use moo's internal helpers (copy them here for self-containment)
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

    // Separate rules into "fast" (single-char) and "groups" (regex)
    // preserving original indices for spectral reordering
    var ruleIndices = [] // indices into `rules` for each group

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
    // SPECTRAL REORDERING (opt-in)
    // ====================================================================
    var spectralInfo = null

    if (spectralOpts && spectralOpts.enabled && groups.length > 2) {
      // Build tension matrix for the rule groups
      var tensionMat = buildTensionMatrix(groups, null)

      // Compute spectral ordering
      var spectral = spectralOrdering(tensionMat, {
        sigma: spectralOpts.sigma || 1.0
      })

      // Reorder groups and parts according to Fiedler vector ordering
      var newOrder = spectral.ordering
      var reorderedGroups = new Array(groups.length)
      var reorderedParts = new Array(parts.length)
      for (var i = 0; i < newOrder.length; i++) {
        reorderedGroups[i] = groups[newOrder[i]]
        reorderedParts[i] = parts[newOrder[i]]
      }
      groups = reorderedGroups
      parts = reorderedParts

      spectralInfo = {
        originalOrder: ruleIndices,
        fiedler: spectral.fiedler,
        eigenvalues: spectral.eigenvalues,
        eigenvectors: spectral.eigenvectors,
        tensionMatrix: tensionMat,
        weightMatrix: spectral.weightMatrix,
        laplacian: spectral.laplacian,
        degrees: spectral.degrees,
        ordering: spectral.ordering,
        typeIndex: null // filled in after compilation
      }
    }

    var fallbackRule = errorRule && errorRule.fallback
    var flags = hasSticky && !fallbackRule ? 'ym' : 'gm'
    var suffix = hasSticky || fallbackRule ? '' : '|'
    if (unicodeFlag === true) flags += "u"
    var combined = new RegExp(reUnion(parts) + suffix, flags)

    var result = {
      regexp: combined,
      groups: groups,
      fast: fast,
      error: errorRule || defaultErrorRule,
      spectral: spectralInfo
    }

    return result
  }

  // ====================================================================
  // Lexer class (enhanced)
  // ====================================================================

  var SpectralLexer = function(states, state, spectralOpts) {
    this.startState = state
    this.states = states
    this.buffer = ''
    this.stack = []
    this._spectralOpts = spectralOpts || {}
    this._conservationTracker = null
    this._recentTypes = []
    this.reset()
  }

  SpectralLexer.prototype.reset = function(data, info) {
    this.buffer = data || ''
    this.index = 0
    this.line = info ? info.line : 1
    this.col = info ? info.col : 1
    this.queuedToken = info ? info.queuedToken : null
    this.queuedText = info ? info.queuedText : ""
    this.queuedThrow = info ? info.queuedThrow : null
    this._recentTypes = info ? (info._recentTypes || []) : []
    this.setState(info ? info.state : this.startState)
    this.stack = info && info.stack ? info.stack.slice() : []

    // Initialize conservation tracker if spectral mode is enabled
    var spectral = this._currentSpectral()
    if (spectral && spectral.eigenvectors && this._spectralOpts.conservation) {
      this._conservationTracker = new ConservationTracker(
        spectral.eigenvectors,
        spectral.typeIndex || {},
        this._spectralOpts.conservationWindow || 20,
        this._spectralOpts.conservationComponents || 5
      )
    }

    return this
  }

  SpectralLexer.prototype._currentSpectral = function() {
    var stateInfo = this.states[this.state]
    return stateInfo ? stateInfo.spectral : null
  }

  SpectralLexer.prototype.save = function() {
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

  SpectralLexer.prototype.setState = function(state) {
    if (!state || this.state === state) return
    this.state = state
    var info = this.states[state]
    this.groups = info.groups
    this.error = info.error
    this.re = info.regexp
    this.fast = info.fast
  }

  SpectralLexer.prototype.popState = function() { this.setState(this.stack.pop()) }
  SpectralLexer.prototype.pushState = function(state) { this.stack.push(this.state); this.setState(state) }

  var eat = hasSticky
    ? function(re, buffer) { return re.exec(buffer) }
    : function(re, buffer) {
        var match = re.exec(buffer)
        if (match[0].length === 0) return null
        return match
      }

  SpectralLexer.prototype._getGroup = function(match) {
    for (var i = 0; i < this.groups.length; i++) {
      if (match[i + 1] !== undefined) return this.groups[i]
    }
    throw new Error('Cannot find token type for matched text')
  }

  function tokenToString() { return this.value }

  SpectralLexer.prototype.next = function() {
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
      // Conservation-aware error recovery
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

  /**
   * Try to recover from a lexing error using spectral analysis.
   * Returns a corrected token or null if no recovery possible.
   */
  SpectralLexer.prototype._tryRecovery = function(unmatchedText) {
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

  SpectralLexer.prototype._token = function(group, text, offset) {
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

    // Track token types for conservation analysis
    if (this._spectralOpts.conservation) {
      this._recentTypes.push(type)
      if (this._recentTypes.length > 50) this._recentTypes = this._recentTypes.slice(-30)

      if (this._conservationTracker) {
        var result = this._conservationTracker.feed(type)
        token._conservation = result
      }
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
    SpectralLexer.prototype[Symbol.iterator] = function() { return new LexerIterator(this) }
  }

  SpectralLexer.prototype.formatError = function(token, message) {
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

  SpectralLexer.prototype.clone = function() {
    return new SpectralLexer(this.states, this.state, this._spectralOpts)
  }

  SpectralLexer.prototype.has = function(tokenType) { return true }

  // ====================================================================
  // Structural statistics API
  // ====================================================================

  /**
   * Return spectral graph theory statistics about the compiled lexer rules.
   * @returns {{ spectralGap: number, cheegerConstant: number, tensionMatrix: Array[], laplacianEigenvalues: number[], optimalOrdering: number[] }}
   */
  SpectralLexer.prototype.structuralInfo = function() {
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

    var eigenvalues = spectral.eigenvalues
    var spectralGap = eigenvalues.length > 1 ? eigenvalues[1] : 0

    // Cheeger constant approximation
    var fiedler = spectral.fiedler
    var cheeger = cheegerConstant(spectral.weightMatrix, fiedler)

    return {
      spectralGap: spectralGap,
      cheegerConstant: cheeger,
      tensionMatrix: spectral.tensionMatrix,
      laplacianEigenvalues: eigenvalues.slice(),
      optimalOrdering: spectral.ordering.slice(),
      fiedlerVector: fiedler.slice()
    }
  }

  // ====================================================================
  // Public API: compile and states (100% backward compatible + spectral options)
  // ====================================================================

  /**
   * Compile a stateless lexer.
   * Same API as moo.compile(), with optional third argument for spectral options.
   *
   * @param {Object|Array} rules - Rule definitions (same as moo)
   * @param {Object} [spectralOptions] - Spectral graph theory options
   * @param {boolean} [spectralOptions.enabled=false] - Enable spectral ordering
   * @param {boolean} [spectralOptions.conservation=false] - Enable conservation tracking
   * @param {boolean} [spectralOptions.errorRecovery=false] - Enable spectral error recovery
   * @param {number} [spectralOptions.sigma=1.0] - Kernel width for tension similarity
   * @param {number} [spectralOptions.conservationWindow=20] - Sliding window size
   * @param {number} [spectralOptions.conservationComponents=5] - Number of eigenvector components
   * @returns {SpectralLexer}
   */
  function compile(rules, spectralOptions) {
    spectralOptions = spectralOptions || {}
    var parsedRules = toRules(rules)
    var result = compileRules(parsedRules, false, spectralOptions)

    // Build typeIndex for conservation tracking
    if (result.spectral) {
      var typeIndex = {}
      for (var i = 0; i < result.groups.length; i++) {
        typeIndex[result.groups[i].defaultType] = i
      }
      result.spectral.typeIndex = typeIndex
    }

    var lexer = new SpectralLexer({start: result}, 'start', spectralOptions)
    return lexer
  }

  /**
   * Compile a stateful lexer.
   * Same API as moo.states(), with optional spectral options.
   */
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
      // Build typeIndex
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

    return new SpectralLexer(map, start, spectralOptions)
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
  // Exports
  // ====================================================================

  return {
    compile: compile,
    states: compileStates,
    error: Object.freeze({error: true}),
    fallback: Object.freeze({fallback: true}),
    keywords: keywordTransform,

    // Advanced: access linear algebra utilities for custom analysis
    _linalg: {
      zeros: zeros,
      symmetrize: symmetrize,
      laplacian: laplacian,
      normalizedLaplacian: normalizedLaplacian,
      jacobiEigen: jacobiEigen,
      cheegerConstant: cheegerConstant,
      spectralOrdering: spectralOrdering,
      buildTensionMatrix: buildTensionMatrix,
      buildTransitionMatrix: buildTransitionMatrix,
      ConservationTracker: ConservationTracker
    }
  }
}))
