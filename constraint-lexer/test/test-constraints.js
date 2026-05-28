/**
 * test/test-constraints.js — Unit tests for each constraint type
 */

'use strict'

const assert = require('assert')
const cl = require('../constraint-lexer')

let passed = 0
let failed = 0

function test(name, fn) {
  try {
    fn()
    console.log(`  ✓ ${name}`)
    passed++
  } catch (e) {
    console.log(`  ✗ ${name}: ${e.message}`)
    failed++
  }
}

console.log('=== Constraint Tests ===\n')

// ---------------------------------------------------------------------------
// BalancedDelimitersConstraint
// ---------------------------------------------------------------------------
console.log('BalancedDelimiters:')

test('balanced parens pass', () => {
  const lexer = cl.compile({
    rules: {
      LPAREN: '(',
      RPAREN: ')',
      IDENT: /[a-z]+/,
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.balancedDelimiters('LPAREN', 'RPAREN'),
    ],
  })
  const result = lexer.tokenize('(foo (bar) baz)')
  const errors = result.violations.filter(v => v.type === 'unmatched_close' || v.type === 'unclosed_open')
  assert.strictEqual(errors.length, 0)
})

test('unmatched close paren detected', () => {
  const lexer = cl.compile({
    rules: {
      LPAREN: '(',
      RPAREN: ')',
      IDENT: /[a-z]+/,
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.balancedDelimiters('LPAREN', 'RPAREN'),
    ],
  })
  const result = lexer.tokenize('foo ) bar')
  const errors = result.violations.filter(v => v.type === 'unmatched_close')
  assert.strictEqual(errors.length, 1)
})

test('unclosed open paren detected', () => {
  const lexer = cl.compile({
    rules: {
      LPAREN: '(',
      RPAREN: ')',
      IDENT: /[a-z]+/,
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.balancedDelimiters('LPAREN', 'RPAREN'),
    ],
  })
  const result = lexer.tokenize('(foo (bar')
  const errors = result.violations.filter(v => v.type === 'unclosed_open')
  assert.strictEqual(errors.length, 2)
})

test('deeply nested balanced parens pass', () => {
  const lexer = cl.compile({
    rules: {
      LPAREN: '(',
      RPAREN: ')',
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.balancedDelimiters('LPAREN', 'RPAREN'),
    ],
  })
  const result = lexer.tokenize('((( )))')
  const errors = result.violations.filter(v => v.type === 'unmatched_close' || v.type === 'unclosed_open')
  assert.strictEqual(errors.length, 0)
})

// ---------------------------------------------------------------------------
// NoAdjacentConstraint
// ---------------------------------------------------------------------------
console.log('\nNoAdjacent:')

test('no adjacent operators pass', () => {
  const lexer = cl.compile({
    rules: {
      NUMBER: /[0-9]+/,
      OP: /[+\-*/]/,
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.noAdjacent('OP'),
    ],
  })
  const result = lexer.tokenize('1 + 2 * 3')
  const adj = result.violations.filter(v => v.type === 'adjacent_forbidden')
  assert.strictEqual(adj.length, 0)
})

test('adjacent operators detected', () => {
  const lexer = cl.compile({
    rules: {
      NUMBER: /[0-9]+/,
      OP: /[+\-*/]/,
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.noAdjacent('OP'),
    ],
  })
  const result = lexer.tokenize('1 + + 2')
  const adj = result.violations.filter(v => v.type === 'adjacent_forbidden')
  assert.ok(adj.length >= 1, 'Should detect adjacent operators')
})

test('no adjacent with multiple types', () => {
  const lexer = cl.compile({
    rules: {
      NUMBER: /[0-9]+/,
      PLUS: '+',
      MINUS: '-',
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.noAdjacent(['PLUS', 'MINUS']),
    ],
  })
  const result = lexer.tokenize('1 + - 2')
  const adj = result.violations.filter(v => v.type === 'adjacent_forbidden')
  assert.ok(adj.length >= 1)
})

// ---------------------------------------------------------------------------
// SmoothTransitionsConstraint
// ---------------------------------------------------------------------------
console.log('\nSmoothTransitions:')

test('smooth token sequence has score', () => {
  const lexer = cl.compile({
    rules: {
      A: 'a',
      B: 'b',
      C: 'c',
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.smoothTransitions({ threshold: 0.1, windowSize: 3 }),
    ],
  })
  const result = lexer.tokenize('a b c a b c a b c')
  assert.ok(result.conservationScore >= 0, 'Score should be non-negative')
})

// ---------------------------------------------------------------------------
// TypeConservationConstraint
// ---------------------------------------------------------------------------
console.log('\nTypeConservation:')

test('alternating value-operator passes', () => {
  const lexer = cl.compile({
    rules: {
      NUMBER: /[0-9]+/,
      OP: /[+\-*/]/,
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.typeConservation({
        NUMBER: 'value',
        OP: 'operator',
      }),
    ],
  })
  const result = lexer.tokenize('1 + 2 * 3')
  const typeErrors = result.violations.filter(v => v.type === 'type_mismatch')
  assert.strictEqual(typeErrors.length, 0)
})

test('consecutive values detected', () => {
  const lexer = cl.compile({
    rules: {
      NUMBER: /[0-9]+/,
      OP: /[+\-*/]/,
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.typeConservation({
        NUMBER: 'value',
        OP: 'operator',
      }),
    ],
  })
  const result = lexer.tokenize('1 + 2 3')
  const typeErrors = result.violations.filter(v => v.type === 'type_mismatch')
  assert.ok(typeErrors.length >= 1, 'Should detect consecutive values')
})

// ---------------------------------------------------------------------------
// KeywordContextConstraint
// ---------------------------------------------------------------------------
console.log('\nKeywordContext:')

test('correct keyword context passes', () => {
  const lexer = cl.compile({
    rules: {
      DEF: 'def',
      IDENT: /[a-zA-Z_]\w*/,
      LPAREN: '(',
      RPAREN: ')',
      COLON: ':',
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.keywordContext({
        'def': { followedBy: ['IDENT'] },
      }),
    ],
  })
  const result = lexer.tokenize('def foo:')
  const ctxErrors = result.violations.filter(v => v.type === 'keyword_context_violation')
  assert.strictEqual(ctxErrors.length, 0)
})

test('wrong keyword context detected', () => {
  const lexer = cl.compile({
    rules: {
      DEF: 'def',
      IDENT: /[a-zA-Z_]\w*/,
      NUMBER: /[0-9]+/,
      LPAREN: '(',
      RPAREN: ')',
      COLON: ':',
      WS: { match: /[ \t]+/, lineBreaks: false },
    },
    constraints: [
      cl.constraints.keywordContext({
        'def': { followedBy: ['IDENT'] },
      }),
    ],
  })
  const result = lexer.tokenize('def 123:')
  const ctxErrors = result.violations.filter(v => v.type === 'keyword_context_violation')
  assert.strictEqual(ctxErrors.length, 1)
})

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
console.log(`\n${'='.repeat(40)}`)
console.log(`Results: ${passed} passed, ${failed} failed`)
if (failed > 0) process.exit(1)
