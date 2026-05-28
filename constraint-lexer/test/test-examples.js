/**
 * test/test-examples.js — Tests on the example grammars
 */

'use strict'

const assert = require('assert')

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

console.log('=== Example Grammar Tests ===\n')

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------
console.log('JSON:')

const jsonLexer = require('../examples/json')

test('valid JSON tokenizes without violations', () => {
  const result = jsonLexer.tokenize('{"name": "Alice", "age": 30}')
  const errors = result.violations.filter(v => v.severity === 'error')
  assert.strictEqual(errors.length, 0, `Unexpected errors: ${errors.map(e => e.message).join('; ')}`)
  assert.ok(result.tokens.length > 0)
})

test('missing closing brace detected', () => {
  const result = jsonLexer.tokenize('{"a": 1')
  const errors = result.violations.filter(v => v.type === 'unclosed_open')
  assert.ok(errors.length >= 1, 'Should detect unclosed brace')
})

test('extra closing brace detected', () => {
  const result = jsonLexer.tokenize('{"a": 1}}')
  const errors = result.violations.filter(v => v.type === 'unmatched_close')
  assert.ok(errors.length >= 1, 'Should detect unmatched close brace')
})

test('nested JSON tokenizes correctly', () => {
  const result = jsonLexer.tokenize('{"a": {"b": [1, 2, 3]}}')
  const errors = result.violations.filter(v => v.severity === 'error')
  assert.strictEqual(errors.length, 0, `Unexpected errors: ${errors.map(e => e.message).join('; ')}`)
})

test('conservation score is between 0 and 1', () => {
  const result = jsonLexer.tokenize('{"x": 1, "y": true}')
  assert.ok(result.conservationScore >= 0 && result.conservationScore <= 1,
    `Score ${result.conservationScore} out of range`)
})

// ---------------------------------------------------------------------------
// Python
// ---------------------------------------------------------------------------
console.log('\nPython:')

const pythonLexer = require('../examples/python')

test('simple function tokenizes', () => {
  const result = pythonLexer.tokenize('def hello():\n    pass')
  const braceErrors = result.violations.filter(v => v.type === 'unclosed_open' || v.type === 'unmatched_close')
  assert.strictEqual(braceErrors.length, 0)
})

test('missing close paren detected', () => {
  const result = pythonLexer.tokenize('def hello(:')
  const errors = result.violations.filter(v => v.type === 'unclosed_open')
  assert.ok(errors.length >= 1, 'Should detect unclosed paren')
})

test('conservation score computed', () => {
  const result = pythonLexer.tokenize('x = 1 + 2')
  assert.ok(result.conservationScore >= 0)
})

// ---------------------------------------------------------------------------
// SQL
// ---------------------------------------------------------------------------
console.log('\nSQL:')

const sqlLexer = require('../examples/sql')

test('simple SELECT tokenizes', () => {
  const result = sqlLexer.tokenize('SELECT name FROM users;')
  const errors = result.violations.filter(v => v.severity === 'error')
  assert.strictEqual(errors.length, 0, `Unexpected errors: ${errors.map(e => e.message).join('; ')}`)
})

test('subquery with balanced parens', () => {
  const result = sqlLexer.tokenize('SELECT * FROM (SELECT id FROM products)')
  const parenErrors = result.violations.filter(v => v.type === 'unclosed_open' || v.type === 'unmatched_close')
  assert.strictEqual(parenErrors.length, 0)
})

test('unmatched paren detected', () => {
  const result = sqlLexer.tokenize('SELECT * FROM (SELECT id')
  const errors = result.violations.filter(v => v.type === 'unclosed_open')
  assert.ok(errors.length >= 1)
})

test('conservation score in range', () => {
  const result = sqlLexer.tokenize('SELECT a, b FROM t WHERE x > 1 ORDER BY a;')
  assert.ok(result.conservationScore >= 0 && result.conservationScore <= 1)
})

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
console.log(`\n${'='.repeat(40)}`)
console.log(`Results: ${passed} passed, ${failed} failed`)
if (failed > 0) process.exit(1)
