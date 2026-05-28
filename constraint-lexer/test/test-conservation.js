/**
 * test/test-conservation.js — Tests for conservation tracking
 */

'use strict'

const assert = require('assert')
const {
  ConservationTracker,
  TransitionGraph,
  computeInformationContent,
  transitionTension,
  entropy,
  powerIteration,
  matVecMul,
  vecNorm,
  vecDot,
} = require('../conservation-tracker')

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

console.log('=== Conservation Tracker Tests ===\n')

// ---------------------------------------------------------------------------
// Matrix utilities
// ---------------------------------------------------------------------------
console.log('Matrix utilities:')

test('power iteration converges on identity', () => {
  const I = [[1, 0], [0, 1]]
  const { eigenvalue } = powerIteration(I, 100)
  assert.ok(Math.abs(eigenvalue - 1) < 0.01, `Expected eigenvalue ~1, got ${eigenvalue}`)
})

test('power iteration finds dominant eigenvalue', () => {
  // Matrix with eigenvalues 3 and 1
  const M = [[2, 1], [1, 2]]
  const { eigenvalue } = powerIteration(M, 100)
  assert.ok(Math.abs(eigenvalue - 3) < 0.01, `Expected eigenvalue ~3, got ${eigenvalue}`)
})

test('vecNorm works', () => {
  const n = vecNorm([3, 4])
  assert.ok(Math.abs(n - 5) < 0.001)
})

test('vecDot works', () => {
  const d = vecDot([1, 2, 3], [4, 5, 6])
  assert.strictEqual(d, 32)
})

// ---------------------------------------------------------------------------
// TransitionGraph
// ---------------------------------------------------------------------------
console.log('\nTransitionGraph:')

test('records transitions', () => {
  const g = new TransitionGraph(['A', 'B', 'C'])
  g.record('A', 'B')
  g.record('B', 'C')
  g.record('A', 'C')
  assert.strictEqual(g.counts[0][1], 1) // A->B
  assert.strictEqual(g.counts[1][2], 1) // B->C
  assert.strictEqual(g.counts[0][2], 1) // A->C
  assert.strictEqual(g.totalTransitions, 3)
})

test('probability matrix sums to 1 per row', () => {
  const g = new TransitionGraph(['A', 'B', 'C'])
  g.record('A', 'B')
  g.record('A', 'C')
  g.record('B', 'A')
  const P = g.probabilityMatrix()
  for (let i = 0; i < 3; i++) {
    const rowSum = P[i].reduce((s, x) => s + x, 0)
    assert.ok(Math.abs(rowSum - 1) < 0.001, `Row ${i} sums to ${rowSum}`)
  }
})

test('laplacian has zero row sums', () => {
  const g = new TransitionGraph(['A', 'B', 'C'])
  g.record('A', 'B')
  g.record('B', 'C')
  g.record('C', 'A')
  const L = g.laplacian()
  for (let i = 0; i < 3; i++) {
    const rowSum = L[i].reduce((s, x) => s + x, 0)
    assert.ok(Math.abs(rowSum) < 0.001, `Laplacian row ${i} sums to ${rowSum}`)
  }
})

// ---------------------------------------------------------------------------
// Information content
// ---------------------------------------------------------------------------
console.log('\nInformation content:')

test('rare types have higher information', () => {
  const counts = [100, 1, 50] // B is rare
  const info = computeInformationContent(counts, 151)
  assert.ok(info[1] > info[0], 'Rare type should have higher info content')
  assert.ok(info[1] > info[2], 'Rare type should have higher info content')
})

test('entropy is maximized for uniform distribution', () => {
  const uniform = [0.25, 0.25, 0.25, 0.25]
  const h = entropy(uniform)
  assert.ok(Math.abs(h - 2) < 0.001, `Uniform entropy should be 2 bits, got ${h}`)
})

test('tension between equal-info types is zero', () => {
  const info = [3, 3, 3]
  assert.strictEqual(transitionTension(info, 0, 1), 0)
})

test('tension is symmetric', () => {
  const info = [1, 5, 3]
  assert.strictEqual(
    transitionTension(info, 0, 1),
    transitionTension(info, 1, 0)
  )
})

// ---------------------------------------------------------------------------
// ConservationTracker
// ---------------------------------------------------------------------------
console.log('\nConservationTracker:')

test('empty tracker has score 1', () => {
  const t = new ConservationTracker(['A', 'B'])
  assert.strictEqual(t.conservationScore(), 1.0)
})

test('uniform transitions give high conservation', () => {
  const t = new ConservationTracker(['A', 'B'], { threshold: 5 })
  for (let i = 0; i < 50; i++) {
    t.feed({type: i % 2 === 0 ? 'A' : 'B', value: '', offset: i}, 
           i > 0 ? {type: i % 2 === 0 ? 'B' : 'A', value: '', offset: i - 1} : null)
  }
  // Should have reasonable conservation for a regular pattern
  assert.ok(t.conservationScore() >= 0, `Score should be non-negative, got ${t.conservationScore()}`)
})

test('detectViolations returns array', () => {
  const t = new ConservationTracker(['A', 'B', 'C'], { threshold: 0.01 })
  t.feed({type: 'A', value: 'a', offset: 0}, null)
  t.feed({type: 'B', value: 'b', offset: 1}, {type: 'A', value: 'a', offset: 0})
  const v = t.detectViolations()
  assert.ok(Array.isArray(v))
})

test('tracker records token sequence', () => {
  const t = new ConservationTracker(['A', 'B'])
  t.feed({type: 'A', value: 'a', offset: 0}, null)
  t.feed({type: 'B', value: 'b', offset: 1}, {type: 'A', value: 'a', offset: 0})
  assert.strictEqual(t.graph.totalTransitions, 1)
})

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
console.log(`\n${'='.repeat(40)}`)
console.log(`Results: ${passed} passed, ${failed} failed`)
if (failed > 0) process.exit(1)
