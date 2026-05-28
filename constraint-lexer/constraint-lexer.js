/**
 * constraint-lexer.js — Constraint-native DSL for lexer specification
 *
 * Takes a constraint specification (JS object), compiles constraints into
 * moo lexer rules, and returns a lexer that tracks conservation at runtime.
 *
 * Uses moo as the backend lexer engine.
 */

'use strict'

const moo = require('../moo.js')
const {
  ConservationTracker,
  computeInformationContent,
  transitionTension,
} = require('./conservation-tracker')

// ===========================================================================
// Constraint implementations
// ===========================================================================

/**
 * Balanced delimiters constraint.
 * Tracks nesting depth of open/close token pairs.
 * Violation: closing without matching open, or unclosed at EOF.
 */
class BalancedDelimitersConstraint {
  constructor(openType, closeType) {
    this.openType = openType
    this.closeType = closeType
    this.name = `balanced(${openType}, ${closeType})`
    this.stack = []
    this.violations = []
    this.maxDepth = 0
  }

  reset() {
    this.stack = []
    this.violations = []
    this.maxDepth = 0
  }

  check(token, prevToken) {
    if (token.type === this.openType) {
      this.stack.push(token)
      this.maxDepth = Math.max(this.maxDepth, this.stack.length)
    } else if (token.type === this.closeType) {
      if (this.stack.length === 0) {
        this.violations.push({
          type: 'unmatched_close',
          token,
          message: `Unmatched ${this.closeType} at offset ${token.offset}: "${token.value}"`,
          severity: 'error',
        })
      } else {
        this.stack.pop()
      }
    }
  }

  finalize() {
    while (this.stack.length > 0) {
      const tok = this.stack.pop()
      this.violations.push({
        type: 'unclosed_open',
        token: tok,
        message: `Unclosed ${this.openType} at offset ${tok.offset}: "${tok.value}"`,
        severity: 'error',
      })
    }
    return this.violations
  }
}

/**
 * No adjacent tokens of given types constraint.
 * Violation: two tokens of the same type(s) appear consecutively.
 */
class NoAdjacentConstraint {
  constructor(tokenTypes) {
    this.types = Array.isArray(tokenTypes) ? new Set(tokenTypes) : new Set([tokenTypes])
    this.name = `noAdjacent(${[...this.types].join(', ')})`
    this.prevType = null
    this.violations = []
  }

  reset() {
    this.prevType = null
    this.violations = []
  }

  check(token, prevToken) {
    if (prevToken && this.types.has(token.type) && this.types.has(prevToken.type)) {
      this.violations.push({
        type: 'adjacent_forbidden',
        from: prevToken,
        to: token,
        message: `Adjacent ${token.type} tokens at offsets ${prevToken.offset}-${token.offset}: "${prevToken.value}" "${token.value}"`,
        severity: 'warning',
      })
    }
  }

  finalize() {
    return this.violations
  }
}

/**
 * Smooth transitions constraint.
 * Uses conservation tracker's tension metric to detect rough transitions.
 */
class SmoothTransitionsConstraint {
  constructor(options = {}) {
    this.threshold = options.threshold || 0.5
    this.windowSize = options.windowSize || 10
    this.name = `smoothTransitions(threshold=${this.threshold})`
    this.violations = []
    this.tensionWindow = []
  }

  reset() {
    this.violations = []
    this.tensionWindow = []
  }

  check(token, prevToken, tracker) {
    if (!prevToken || !tracker) return

    const info = computeInformationContent(
      tracker.graph.typeCounts,
      tracker.graph.totalTransitions + 1
    )
    const a = tracker.graph.index[prevToken.type]
    const b = tracker.graph.index[token.type]
    if (a === undefined || b === undefined) return

    const tension = transitionTension(info, a, b)
    this.tensionWindow.push(tension)
    if (this.tensionWindow.length > this.windowSize) {
      this.tensionWindow.shift()
    }

    // Check if average window tension exceeds threshold
    const avgTension = this.tensionWindow.reduce((s, x) => s + x, 0) / this.tensionWindow.length
    if (avgTension > this.threshold) {
      this.violations.push({
        type: 'rough_transition',
        from: prevToken,
        to: token,
        tension: avgTension,
        threshold: this.threshold,
        message: `Transition tension ${avgTension.toFixed(3)} exceeds threshold ${this.threshold} at offset ${token.offset}`,
        severity: 'warning',
      })
    }
  }

  finalize() {
    return this.violations
  }
}

/**
 * Indentation tracking constraint (for Python-like languages).
 * Tracks indent/dedent balance, ensures proper nesting.
 */
class IndentationConstraint {
  constructor(options = {}) {
    this.indentType = options.indentType || 'INDENT'
    this.dedentType = options.dedentType || 'DEDENT'
    this.name = `indentation(${this.indentType}, ${this.dedentType})`
    this.indentStack = [0]
    this.violations = []
  }

  reset() {
    this.indentStack = [0]
    this.violations = []
  }

  check(token, prevToken) {
    if (token.type === this.indentType) {
      this.indentStack.push(this.indentStack.length)
    } else if (token.type === this.dedentType) {
      if (this.indentStack.length <= 1) {
        this.violations.push({
          type: 'extra_dedent',
          token,
          message: `Extra ${this.dedentType} at offset ${token.offset}`,
          severity: 'error',
        })
      } else {
        this.indentStack.pop()
      }
    }
  }

  finalize() {
    if (this.indentStack.length > 1) {
      this.violations.push({
        type: 'unclosed_indent',
        message: `Unclosed indentation: ${this.indentStack.length - 1} levels remain`,
        severity: 'error',
      })
    }
    return this.violations
  }
}

/**
 * Type information conservation constraint.
 * Tracks that expressions maintain type consistency through the token stream.
 * Uses a simple stack-based type tracking approach.
 */
class TypeConservationConstraint {
  constructor(typeRules = {}) {
    // typeRules maps token types to their "semantic type" (e.g., NUMBER -> 'value', OP -> 'operator')
    this.typeRules = typeRules
    this.name = `typeConservation(${Object.keys(typeRules).length} rules)`
    this.violations = []
    this.typeStack = []
    this.expectedNext = null
  }

  reset() {
    this.violations = []
    this.typeStack = []
    this.expectedNext = null
  }

  check(token, prevToken) {
    const semanticType = this.typeRules[token.type]
    if (!semanticType) return

    if (this.expectedNext && semanticType !== this.expectedNext) {
      this.violations.push({
        type: 'type_mismatch',
        token,
        expected: this.expectedNext,
        actual: semanticType,
        message: `Expected type '${this.expectedNext}' but got '${semanticType}' (${token.type}) at offset ${token.offset}`,
        severity: 'warning',
      })
    }

    // Simple type state machine: after 'operator', expect 'value'; after 'value', expect 'operator' or 'close'
    if (semanticType === 'operator') {
      this.expectedNext = 'value'
    } else if (semanticType === 'value') {
      this.expectedNext = 'operator'
    }
  }

  finalize() {
    return this.violations
  }
}

/**
 * Keyword context preservation constraint.
 * Ensures certain keywords are followed by appropriate token types.
 */
class KeywordContextConstraint {
  constructor(contextRules = {}) {
    // contextRules: { KEYWORD_NAME: { followedBy: ['IDENT', 'LPAREN'], ... }, ... }
    this.contextRules = contextRules
    this.name = `keywordContext(${Object.keys(contextRules).length} rules)`
    this.violations = []
    this.expectingContext = null
  }

  reset() {
    this.violations = []
    this.expectingContext = null
  }

  check(token, prevToken) {
    // Check if previous token set up a context expectation
    if (this.expectingContext && prevToken) {
      const rule = this.expectingContext
      if (rule.followedBy && !rule.followedBy.includes(token.type)) {
        this.violations.push({
          type: 'keyword_context_violation',
          from: prevToken,
          to: token,
          expected: rule.followedBy,
          message: `After ${prevToken.type} "${prevToken.value}", expected ${rule.followedBy.join('|')} but got ${token.type} "${token.value}" at offset ${token.offset}`,
          severity: 'warning',
        })
      }
      this.expectingContext = null
    }

    // Set up new context if this token is a keyword with rules
    if (this.contextRules[token.value]) {
      this.expectingContext = this.contextRules[token.value]
    }
  }

  finalize() {
    return this.violations
  }
}

// ===========================================================================
// Constraint factory functions (public API)
// ===========================================================================

const constraints = {
  /**
   * Balanced delimiter tracking: ensures open/close pairs match.
   */
  balancedDelimiters(openType, closeType) {
    return new BalancedDelimitersConstraint(openType, closeType)
  },

  /**
   * Smooth transition constraint: tension between tokens stays below threshold.
   */
  smoothTransitions(options = {}) {
    return new SmoothTransitionsConstraint(options)
  },

  /**
   * No adjacent tokens of specified types.
   */
  noAdjacent(tokenTypes) {
    return new NoAdjacentConstraint(tokenTypes)
  },

  /**
   * Indentation tracking (Python-style).
   */
  indentation(options = {}) {
    return new IndentationConstraint(options)
  },

  /**
   * Type information conservation.
   */
  typeConservation(typeRules) {
    return new TypeConservationConstraint(typeRules)
  },

  /**
   * Keyword context preservation.
   */
  keywordContext(contextRules) {
    return new KeywordContextConstraint(contextRules)
  },
}

// ===========================================================================
// compile() — Main entry point
// ===========================================================================

/**
 * Compile a constraint specification into a moo-backed lexer.
 *
 * @param {Object} spec
 * @param {Object} spec.rules — Token rules (same format as moo.compile)
 * @param {Array} spec.constraints — Array of constraint instances
 * @param {Object} [spec.options] — Options for conservation tracking
 * @returns {Object} Lexer with { tokenize(input) }
 */
function compile(spec) {
  const { rules, constraints: constraintList = [], options = {} } = spec

  // Build moo lexer
  const mooLexer = moo.compile(rules)

  // Get token type names for conservation tracker
  const tokenTypes = Object.keys(rules)

  return new ConstraintLexer(mooLexer, constraintList, tokenTypes, options)
}

// ===========================================================================
// ConstraintLexer — Wraps moo lexer with constraint checking
// ===========================================================================

class ConstraintLexer {
  constructor(mooLexer, constraintList, tokenTypes, options = {}) {
    this.mooLexer = mooLexer
    this.constraintList = constraintList
    this.tokenTypes = tokenTypes
    this.trackerOptions = options.tracker || {}
  }

  /**
   * Tokenize input with constraint checking.
   *
   * @param {string} input
   * @returns {{ tokens: Array, violations: Array, conservationScore: number }}
   */
  tokenize(input) {
    // Reset moo lexer
    this.mooLexer.reset(input)

    // Reset all constraints
    for (const c of this.constraintList) {
      if (c.reset) c.reset()
    }

    // Create conservation tracker
    const tracker = new ConservationTracker(this.tokenTypes, this.trackerOptions)

    // Collect all tokens
    const tokens = []
    let token
    while ((token = this.mooLexer.next())) {
      tokens.push(token)
    }

    // Run constraints over token stream
    // We maintain two previous-token pointers:
    //  - prevToken: the immediately previous token (including WS)
    //  - prevTokenNoWS: the previous non-WS token
    let prevToken = null
    let prevTokenNoWS = null
    for (const tok of tokens) {
      // Feed to conservation tracker (includes WS)
      tracker.feed(tok, prevToken)

      // Check each constraint, passing the non-WS prev for adjacency/context checks
      for (const c of this.constraintList) {
        if (c.check) {
          // NoAdjacent and KeywordContext skip WS tokens entirely
          if (c instanceof NoAdjacentConstraint || c instanceof KeywordContextConstraint) {
            if (tok.type !== 'WS') c.check(tok, prevTokenNoWS, tracker)
          } else {
            c.check(tok, prevToken, tracker)
          }
        }
      }

      prevToken = tok
      if (tok.type !== 'WS') prevTokenNoWS = tok
    }

    // Finalize constraints (e.g., check for unclosed delimiters)
    const violations = []
    for (const c of this.constraintList) {
      if (c.finalize) {
        const v = c.finalize()
        if (v) violations.push(...v)
      }
    }

    // Add conservation tracker violations
    violations.push(...tracker.detectViolations())

    // Compute overall conservation score
    const conservationScore = tracker.conservationScore()

    return {
      tokens,
      violations,
      conservationScore,
      tracker,
    }
  }

  /**
   * Low-level access: get the underlying moo lexer for incremental use.
   */
  getMooLexer() {
    return this.mooLexer
  }
}

// ===========================================================================
// Exports
// ===========================================================================

module.exports = {
  compile,
  constraints,
  // Expose constraint classes for advanced use
  BalancedDelimitersConstraint,
  NoAdjacentConstraint,
  SmoothTransitionsConstraint,
  IndentationConstraint,
  TypeConservationConstraint,
  KeywordContextConstraint,
  // Re-export tracker for direct use
  ConservationTracker,
}
