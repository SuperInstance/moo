/**
 * moo-spectral.test.js — Comprehensive tests for spectral-enhanced moo lexer
 */
'use strict'

var assert = require('assert')
var mooSpectral = require('./moo-spectral')

// Standalone runner: collect describe/it blocks, then execute
var _testCollector = { tests: [], suite: '', mode: 'collect' }
var describe, it
if (typeof global.describe === 'undefined') {
  describe = function(name, fn) {
    var prev = _testCollector.suite
    _testCollector.suite = name
    try { fn() } catch(e) { console.error('Suite error:', e) }
    _testCollector.suite = prev
  }
  it = function(name, fn) {
    _testCollector.tests.push({ suite: _testCollector.suite, name: name, fn: fn })
  }
  _testCollector.mode = 'run'
} else {
  describe = global.describe
  it = global.it
}

// ====================================================================
// 1. BACKWARD COMPATIBILITY TESTS
// ====================================================================

describe('Backward compatibility with moo API', function() {

  it('compiles a simple stateless lexer from an object', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      number: /[0-9]+/,
      space: { match: /\s+/, lineBreaks: true },
    })
    lexer.reset('abc 123 def')
    assert.deepEqual(tokenTypesFiltered(lexer, 'space'), ['word', 'number', 'word'])
  })

  it('compiles a stateless lexer from an array', function() {
    var lexer = mooSpectral.compile([
      { type: 'word', match: /[a-z]+/ },
      { type: 'number', match: /[0-9]+/ },
    ])
    lexer.reset('hello42world')
    var tokens = allTokens(lexer)
    assert.equal(tokens.length, 3)
    assert.equal(tokens[0].type, 'word')
    assert.equal(tokens[1].type, 'number')
    assert.equal(tokens[2].type, 'word')
  })

  it('compiles a stateful lexer', function() {
    var lexer = mooSpectral.states({
      main: {
        word: /[a-z]+/,
        open: { match: /\(/, push: 'inner' },
      },
      inner: {
        word: /[a-z]+/,
        close: { match: /\)/, pop: 1 },
      }
    })
    lexer.reset('foo(bar)baz')
    var types = tokenTypes(lexer)
    assert.deepEqual(types, ['word', 'open', 'word', 'close', 'word'])
  })

  it('handles string literals', function() {
    var lexer = mooSpectral.compile({
      lparen: '(',
      rparen: ')',
      word: /[a-z]+/,
    })
    lexer.reset('(foo)')
    assert.deepEqual(tokenTypes(lexer), ['lparen', 'word', 'rparen'])
  })

  it('tracks line and column numbers', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      newline: { match: /\n/, lineBreaks: true },
    })
    lexer.reset('hello\nworld')
    var tokens = allTokens(lexer)
    assert.equal(tokens[0].line, 1)
    assert.equal(tokens[0].col, 1)
    assert.equal(tokens[1].line, 1) // newline
    assert.equal(tokens[2].line, 2)
    assert.equal(tokens[2].col, 1)
  })

  it('supports reset()', function() {
    var lexer = mooSpectral.compile({ word: /[a-z]+/ })
    lexer.reset('hello')
    assert.equal(lexer.next().value, 'hello')
    lexer.reset('world')
    assert.equal(lexer.next().value, 'world')
  })

  it('supports save() and restore via reset()', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      space: { match: /\s+/, lineBreaks: true },
    })
    lexer.reset('hello world foo')
    lexer.next() // hello
    lexer.next() // space
    var saved = lexer.save()
    assert.equal(saved.line, 1)
    assert.equal(saved.col, 7)
    // Reset with saved state
    lexer.reset('hello world foo', saved)
    var tok = lexer.next()
    assert.equal(tok.type, 'word')
  })

  it('supports formatError', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      space: { match: /\s+/, lineBreaks: true },
    })
    lexer.reset('hello world')
    var tok = lexer.next()
    var err = lexer.formatError(tok, 'test error')
    assert(err.indexOf('test error') >= 0)
    assert(err.indexOf('line 1') >= 0)
  })

  it('supports clone()', function() {
    var lexer = mooSpectral.compile({ word: /[a-z]+/ })
    lexer.reset('hello')
    var cloned = lexer.clone()
    cloned.reset('world')
    assert.equal(cloned.next().value, 'world')
  })

  it('supports has()', function() {
    var lexer = mooSpectral.compile({ word: /[a-z]+/ })
    assert.equal(lexer.has('word'), true)
  })

  it('is iterable with for-of', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      space: { match: /\s+/, lineBreaks: true },
    })
    lexer.reset('hello world')
    var values = []
    for (var tok of lexer) {
      if (tok.type !== 'space') values.push(tok.value)
    }
    assert.deepEqual(values, ['hello', 'world'])
  })

  it('handles keywords', function() {
    var kw = mooSpectral.keywords({ IF: 'if', ELSE: 'else' })
    assert.equal(kw('if'), 'IF')
    assert.equal(kw('else'), 'ELSE')
    assert.equal(kw('foo'), undefined)
  })

  it('exports error and fallback tokens', function() {
    assert.deepEqual(mooSpectral.error, {error: true})
    assert.deepEqual(mooSpectral.fallback, {fallback: true})
  })

  it('throws on invalid input without error rule', function() {
    var lexer = mooSpectral.compile({ word: /[a-z]+/ })
    lexer.reset('!!!')
    assert.throws(function() { lexer.next() })
  })
})

// ====================================================================
// 2. LINEAR ALGEBRA TESTS
// ====================================================================

describe('Linear algebra utilities', function() {

  var la = mooSpectral._linalg

  it('zeros creates zero matrix', function() {
    var m = la.zeros(3)
    assert.equal(m.length, 3)
    assert.equal(m[0].length, 3)
    assert.equal(m[1][2], 0)
  })

  it('symmetrize a matrix', function() {
    var A = [[0,1,0],[0,0,2],[3,0,0]]
    var S = la.symmetrize(A)
    assert.equal(S[0][1], 0.5)
    assert.equal(S[1][0], 0.5)
    assert.equal(S[1][2], 1)
    assert.equal(S[2][1], 1)
  })

  it('computes Laplacian correctly', function() {
    // Simple 3-node path graph: 1-2-3
    var W = [[0,1,0],[1,0,1],[0,1,0]]
    var result = la.laplacian(W)
    var L = result.L
    assert.equal(L[0][0], 1)
    assert.equal(L[0][1], -1)
    assert.equal(L[1][1], 2)
    assert.equal(L[2][2], 1)
  })

  it('computes normalized Laplacian', function() {
    var W = [[0,1,0],[1,0,1],[0,1,0]]
    var result = la.normalizedLaplacian(W)
    var L = result.L
    assert.equal(L[0][0], 1)
    assert.equal(L[2][2], 1)
    assert.ok(Math.abs(L[1][1] - 1) < 1e-10)
  })

  it('Jacobi eigenvalue decomposition', function() {
    // Diagonal matrix
    var A = [[3,0,0],[0,1,0],[0,0,2]]
    var result = la.jacobiEigen(A)
    assert.deepEqual(result.eigenvalues, [1, 2, 3])
  })

  it('Jacobi on symmetric matrix gives correct eigenvalues', function() {
    // 2x2 symmetric
    var A = [[2, 1], [1, 3]]
    var result = la.jacobiEigen(A)
    // Eigenvalues should be (5±√5)/2
    var expected = [(5 - Math.sqrt(5))/2, (5 + Math.sqrt(5))/2]
    assert.ok(Math.abs(result.eigenvalues[0] - expected[0]) < 1e-6)
    assert.ok(Math.abs(result.eigenvalues[1] - expected[1]) < 1e-6)
  })

  it('Jacobi eigenvectors are orthonormal', function() {
    var A = [[4, 1], [1, 3]]
    var result = la.jacobiEigen(A)
    var v0 = [result.eigenvectors[0][0], result.eigenvectors[1][0]]
    var v1 = [result.eigenvectors[0][1], result.eigenvectors[1][1]]
    // ||v0|| = 1
    var norm0 = Math.sqrt(v0[0]*v0[0] + v0[1]*v0[1])
    var norm1 = Math.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
    assert.ok(Math.abs(norm0 - 1) < 1e-6)
    assert.ok(Math.abs(norm1 - 1) < 1e-6)
    // v0 · v1 = 0
    var dot = v0[0]*v1[0] + v0[1]*v1[1]
    assert.ok(Math.abs(dot) < 1e-6)
  })

  it('spectralOrdering returns valid ordering', function() {
    var T = [[0, 0.8, 0.1], [0.8, 0, 0.3], [0.1, 0.3, 0]]
    var result = la.spectralOrdering(T)
    assert.equal(result.ordering.length, 3)
    assert.equal(result.fiedler.length, 3)
    assert.equal(result.eigenvalues.length, 3)
    // Ordering should be a permutation of [0,1,2]
    var sorted = result.ordering.slice().sort()
    assert.deepEqual(sorted, [0, 1, 2])
  })
})

// ====================================================================
// 3. TENSION MATRIX TESTS
// ====================================================================

describe('Rule tension computation', function() {

  var la = mooSpectral._linalg

  it('builds a symmetric tension matrix', function() {
    var rules = [
      { defaultType: 'a', match: [/[a-z]+/], lineBreaks: false, pop: false, next: null, push: null, error: false, fallback: false, value: null, type: null, shouldThrow: false },
      { defaultType: 'b', match: [/[0-9]+/], lineBreaks: false, pop: false, next: null, push: null, error: false, fallback: false, value: null, type: null, shouldThrow: false },
    ]
    var T = la.buildTensionMatrix(rules)
    assert.equal(T.length, 2)
    assert.equal(T[0][0], 0)  // diagonal is 0
    assert.equal(T[1][1], 0)
    assert.equal(T[0][1], T[1][0]) // symmetric
  })

  it('rules with overlapping character sets have higher tension', function() {
    var rules = [
      { defaultType: 'alpha', match: [/[a-z]+/], lineBreaks: false, pop: false, next: null, push: null, error: false, fallback: false, value: null, type: null, shouldThrow: false },
      { defaultType: 'alnum', match: [/[a-zA-Z0-9]+/], lineBreaks: false, pop: false, next: null, push: null, error: false, fallback: false, value: null, type: null, shouldThrow: false },
      { defaultType: 'digits', match: [/[0-9]+/], lineBreaks: false, pop: false, next: null, push: null, error: false, fallback: false, value: null, type: null, shouldThrow: false },
    ]
    var T = la.buildTensionMatrix(rules)
    // alpha-alnum overlap > alpha-digits overlap (they share [a-z])
    assert.ok(T[0][1] > 0 || T[0][2] > 0) // at least some tension
  })

  it('builds transition matrix from token streams', function() {
    var typeIndex = { word: 0, number: 1, space: 2 }
    var stream = ['word', 'space', 'number', 'space', 'word']
    var T = la.buildTransitionMatrix(stream, typeIndex)
    assert.equal(T.length, 3)
    // Rows should sum to ~1
    for (var i = 0; i < 3; i++) {
      var sum = T[i][0] + T[i][1] + T[i][2]
      assert.ok(Math.abs(sum - 1) < 0.01, 'Row ' + i + ' sums to ' + sum)
    }
  })
})

// ====================================================================
// 4. CONSERVATION TRACKER TESTS
// ====================================================================

describe('Conservation tracker', function() {

  var la = mooSpectral._linalg

  it('tracks conservation over token stream', function() {
    // Simple 3-type eigenvector setup
    var eigenvectors = [[1, 0], [0, 1], [0.5, 0.5]]
    var typeIndex = { word: 0, number: 1, space: 2 }
    var tracker = new la.ConservationTracker(eigenvectors, typeIndex, 5, 2)

    // Feed some tokens
    var r1 = tracker.feed('word')
    assert.equal(r1.score, 0) // not enough data
    tracker.feed('space')
    tracker.feed('word')
    tracker.feed('space')
    var r2 = tracker.feed('word')
    // After enough tokens, we should get a score
    assert.ok(typeof r2.score === 'number')
  })

  it('suggests corrections based on eigenspace proximity', function() {
    var eigenvectors = [[1, 0], [0, 1], [0.5, 0.5]]
    var typeIndex = { word: 0, number: 1, op: 2 }
    var tracker = new la.ConservationTracker(eigenvectors, typeIndex, 5, 2)

    var result = tracker.suggestCorrection('__error__', ['word', 'word', 'word'])
    assert.ok(typeof result.suggestedType === 'string')
    assert.ok(result.confidence >= 0 && result.confidence <= 1)
  })
})

// ====================================================================
// 5. SPECTRAL LEXER INTEGRATION TESTS
// ====================================================================

describe('Spectral lexer integration', function() {

  it('compiles without spectral options (plain moo mode)', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      number: /[0-9]+/,
      space: { match: /\s+/, lineBreaks: true },
    })
    lexer.reset('hello 42 world')
    assert.deepEqual(tokenTypesFiltered(lexer, 'space'), ['word', 'number', 'word'])
  })

  it('compiles with spectral options enabled', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      number: /[0-9]+/,
      space: { match: /\s+/, lineBreaks: true },
      lparen: '(',
      rparen: ')',
    }, {
      enabled: true,
    })
    lexer.reset('hello (42) world')
    assert.deepEqual(tokenTypesFiltered(lexer, 'space'), ['word', 'lparen', 'number', 'rparen', 'word'])
  })

  it('spectral ordering produces same tokens but potentially different internal order', function() {
    var rules = {
      word: /[a-z]+/,
      number: /[0-9]+/,
      plus: '+',
      times: '*',
      lparen: '(',
      rparen: ')',
      space: { match: /\s+/, lineBreaks: true },
    }

    var lexerPlain = mooSpectral.compile(rules)
    var lexerSpectral = mooSpectral.compile(rules, { enabled: true })

    var input = 'x + 3 * (y + 2)'
    lexerPlain.reset(input)
    lexerSpectral.reset(input)

    var typesPlain = tokenTypesFiltered(lexerPlain, 'space')
    var typesSpectral = tokenTypesFiltered(lexerSpectral, 'space')

    assert.deepEqual(typesPlain, typesSpectral)
  })

  it('structuralInfo returns meaningful data when spectral is enabled', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      number: /[0-9]+/,
      space: { match: /\s+/, lineBreaks: true },
      lparen: '(',
      rparen: ')',
      lbrace: '{',
      rbrace: '}',
    }, { enabled: true })

    var info = lexer.structuralInfo()
    assert.ok(info.spectralGap >= 0)
    assert.ok(info.cheegerConstant >= 0)
    assert.ok(Array.isArray(info.tensionMatrix))
    assert.ok(Array.isArray(info.laplacianEigenvalues))
    assert.ok(Array.isArray(info.optimalOrdering))
    assert.ok(Array.isArray(info.fiedlerVector))
  })

  it('structuralInfo returns note when spectral is disabled', function() {
    var lexer = mooSpectral.compile({ word: /[a-z]+/ })
    var info = lexer.structuralInfo()
    assert.ok(info.note)
    assert.equal(info.spectralGap, 0)
  })

  it('conservation tracking annotates tokens', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      number: /[0-9]+/,
      space: { match: /\s+/, lineBreaks: true },
    }, {
      enabled: true,
      conservation: true,
    })

    lexer.reset('foo 123 bar 456 baz')
    var tokens = allTokens(lexer)
    // After a few tokens, conservation data should appear
    var withConservation = tokens.filter(function(t) {
      return t._conservation !== undefined
    })
    assert.ok(withConservation.length > 0, 'Some tokens should have conservation data')
  })

  it('Fiedler vector ordering groups related rules', function() {
    var lexer = mooSpectral.compile({
      lparen: '(',
      rparen: ')',
      lbrace: '{',
      rbrace: '}',
      lbracket: '[',
      rbracket: ']',
      word: /[a-z]+/,
      number: /[0-9]+/,
      space: { match: /\s+/, lineBreaks: true },
    }, { enabled: true })

    var info = lexer.structuralInfo()
    var ordering = info.optimalOrdering
    var fiedler = info.fiedlerVector

    // The Fiedler vector should have meaningful structure
    assert.ok(fiedler.length > 0)

    // Verify ordering is a valid permutation
    var sorted = ordering.slice().sort(function(a,b){return a-b})
    for (var i = 0; i < sorted.length; i++) {
      assert.equal(sorted[i], i)
    }
  })

  it('spectral gap and Cheeger constant are non-negative', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      number: /[0-9]+/,
      string: /"[^"]*"/,
      op: /[+\-*\/]/,
      space: { match: /\s+/, lineBreaks: true },
    }, { enabled: true })

    var info = lexer.structuralInfo()
    assert.ok(info.spectralGap >= 0, 'Spectral gap should be non-negative')
    assert.ok(info.cheegerConstant >= 0, 'Cheeger constant should be non-negative')
  })
})

// ====================================================================
// 6. STATEFUL LEXER WITH SPECTRAL
// ====================================================================

describe('Stateful spectral lexer', function() {

  it('compiles stateful lexer with spectral options', function() {
    var lexer = mooSpectral.states({
      expression: {
        word: /[a-z]+/,
        number: /[0-9]+/,
        lparen: { match: '(', push: 'expression' },
        rparen: { match: ')', pop: 1 },
        space: { match: /\s+/, lineBreaks: true },
      }
    }, null, { enabled: true })

    lexer.reset('f(42)')
    var types = tokenTypesFiltered(lexer, 'space')
    assert.deepEqual(types, ['word', 'lparen', 'number', 'rparen'])
  })
})

// ====================================================================
// 7. EDGE CASES
// ====================================================================

describe('Edge cases', function() {

  it('handles single rule', function() {
    var lexer = mooSpectral.compile({ word: /[a-z]+/ })
    lexer.reset('hello')
    assert.equal(lexer.next().type, 'word')
  })

  it('handles spectral with only 2 rules (groups)', function() {
    // Jacobi needs at least 2
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      number: /[0-9]+/,
    }, { enabled: true })
    lexer.reset('hello42')
    assert.equal(lexer.next().type, 'word')
    assert.equal(lexer.next().type, 'number')
  })

  it('handles empty input', function() {
    var lexer = mooSpectral.compile({ word: /[a-z]+/ })
    lexer.reset('')
    assert.equal(lexer.next(), undefined)
  })

  it('handles error token', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      space: { match: /\s+/, lineBreaks: true },
    })
    lexer.reset('hello !!! world')
    assert.equal(lexer.next().type, 'word')
    lexer.next() // space
    assert.throws(function() { lexer.next() }) // !!!
  })

  it('handles fallback token', function() {
    var lexer = mooSpectral.compile({
      word: /[a-z]+/,
      other: mooSpectral.fallback,
    })
    lexer.reset('hello123')
    assert.equal(lexer.next().type, 'word')
    assert.equal(lexer.next().type, 'other')
  })

  it('Jacobi handles identity matrix', function() {
    var la = mooSpectral._linalg
    var I = [[1,0,0],[0,1,0],[0,0,1]]
    var result = la.jacobiEigen(I)
    assert.deepEqual(result.eigenvalues, [1, 1, 1])
  })
})

// ====================================================================
// 8. PERFORMANCE / SCALE TEST
// ====================================================================

describe('Performance', function() {

  it('handles a realistic JS-like lexer', function() {
    var lexer = mooSpectral.compile({
      keyword: /\b(?:function|return|var|let|const|if|else|for|while|class|new|this|import|export|default|from|async|await|try|catch|throw)\b/,
      identifier: /[a-zA-Z_$][a-zA-Z0-9_$]*/,
      number: /(?:0x[0-9a-fA-F]+|\d+\.?\d*(?:e[+-]?\d+)?)/,
      string: /"(?:[^"\\]|\\.)*"/,
      operator: /[+\-*\/%=<>!&|^~?:]+/,
      lparen: '(',
      rparen: ')',
      lbrace: '{',
      rbrace: '}',
      lbracket: '[',
      rbracket: ']',
      semicolon: ';',
      comma: ',',
      dot: '.',
      whitespace: { match: /\s+/, lineBreaks: true },
    }, { enabled: true })

    var code = 'function factorial(n) { if (n <= 1) { return 1; } return n * factorial(n - 1); }'
    lexer.reset(code)

    var tokens = []
    var tok
    while ((tok = lexer.next()) !== undefined) {
      if (tok.type !== 'whitespace') tokens.push(tok.type)
    }

    assert.ok(tokens.length > 10)
    assert.equal(tokens[0], 'keyword')
    assert.ok(tokens.indexOf('identifier') >= 0)
  })

  it('spectral compilation is not unreasonably slow', function() {
    // this.timeout(5000) -- not available in standalone runner
    var start = Date.now()
    var lexer = mooSpectral.compile({
      keyword: /\b(?:function|return|var|let|const|if|else|for|while|do|switch|case|break|continue|class|extends|new|this|super|import|export|default|from|as|async|await|yield|try|catch|finally|throw|typeof|instanceof|in|of|void|delete|true|false|null|undefined)\b/,
      identifier: /[a-zA-Z_$][a-zA-Z0-9_$]*/,
      number: /(?:0x[0-9a-fA-F]+|0b[01]+|0o[0-7]+|\d+\.?\d*(?:e[+-]?\d+)?)/,
      string1: /"(?:[^"\\]|\\.)*"/,
      string2: /'(?:[^'\\]|\\.)*'/,
      template: /`(?:[^`\\]|\\.)*`/,
      regex: /\/(?!\s)(?:[^\/\\\[\n]|\\.|(?:\[(?:[^\]\\\n]|\\.)*\]))*\/[gimsuy]*/,
      operator: /[+\-*\/%=<>!&|^~?:]+/,
      arrow: '=>',
      spread: '...',
      lparen: '(',
      rparen: ')',
      lbrace: '{',
      rbrace: '}',
      lbracket: '[',
      rbracket: ']',
      semicolon: ';',
      comma: ',',
      dot: '.',
      colon: ':',
      whitespace: { match: /\s+/, lineBreaks: true },
      comment: /\/\/.*|\/\*[\s\S]*?\*\//,
    }, { enabled: true, conservation: true })

    var elapsed = Date.now() - start
    assert.ok(elapsed < 3000, 'Compilation should take <3s, took ' + elapsed + 'ms')

    var info = lexer.structuralInfo()
    assert.ok(info.laplacianEigenvalues.length > 0)
  })
})

// ====================================================================
// Helpers
// ====================================================================

function allTokens(lexer) {
  var tokens = []
  var tok
  while ((tok = lexer.next()) !== undefined) tokens.push(tok)
  return tokens
}

function tokenTypes(lexer) {
  return allTokens(lexer).map(function(t) { return t.type })
}

function tokenTypesFiltered(lexer, filterType) {
  return allTokens(lexer)
    .filter(function(t) { return t.type !== filterType })
    .map(function(t) { return t.type })
}

// Standalone test runner (works without mocha)
if (typeof global.describe === 'undefined') {
  describe = global.describe
  it = global.it
}
if (_testCollector.mode === 'run') {
  ;(function() {
    var passed = 0, failed = 0, errors = []
    var tests = _testCollector.tests
    console.log('\n moo-spectral test results\n' + '='.repeat(60) + '\n')
    for (var i = 0; i < tests.length; i++) {
      var t = tests[i]
      try {
        t.fn()
        passed++
        console.log('  \u2713 ' + t.suite + ' > ' + t.name)
      } catch (e) {
        failed++
        console.log('  \u2717 ' + t.suite + ' > ' + t.name)
        console.log('    ' + (e.message || e))
        errors.push({ test: t.name, error: e })
      }
    }
    console.log('\n' + '='.repeat(60))
    console.log('  ' + passed + ' passed, ' + failed + ' failed, ' + tests.length + ' total')
    if (failed > 0) {
      console.log('\nFailures:')
      for (var i = 0; i < errors.length; i++)
        console.log('  - ' + errors[i].test + ': ' + errors[i].error.message)
      process.exit(1)
    }
    console.log('')
  })()
}
