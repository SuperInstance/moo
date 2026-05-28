/**
 * examples/json.js — JSON lexer with constraints
 *
 * Balanced braces + no-adjacent-operators + type-conservation
 */

const cl = require('../constraint-lexer')

const lexer = cl.compile({
  rules: {
    STRING: /"(?:[^"\\]|\\.)*"/,
    NUMBER: /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/,
    TRUE: 'true',
    FALSE: 'false',
    NULL: 'null',
    LBRACE: '{',
    RBRACE: '}',
    LBRACKET: '[',
    RBRACKET: ']',
    COLON: ':',
    COMMA: ',',
    WS: { match: /\s+/, lineBreaks: true },
  },
  constraints: [
    // Balanced curly braces
    cl.constraints.balancedDelimiters('LBRACE', 'RBRACE'),
    // Balanced square brackets
    cl.constraints.balancedDelimiters('LBRACKET', 'RBRACKET'),
    // Type conservation: values and separators alternate predictably
    cl.constraints.typeConservation({
      STRING: 'value',
      NUMBER: 'value',
      TRUE: 'value',
      FALSE: 'value',
      NULL: 'value',
      COLON: 'operator',
      COMMA: 'operator',
    }),
  ],
})

// Demo
if (require.main === module) {
  const tests = [
    '{"name": "Alice", "age": 30, "active": true}',
    '{"broken": [1, 2, 3}',
    '{"oops": , "bad"}',
    '{"a": {"b": {"c": 1}}}',
  ]

  for (const input of tests) {
    console.log(`\n=== JSON: ${input} ===`)
    const result = lexer.tokenize(input)
    console.log(`Tokens: ${result.tokens.filter(t => t.type !== 'WS').map(t => `${t.type}(${t.value})`).join(' ')}`)
    console.log(`Conservation: ${result.conservationScore.toFixed(3)}`)
    if (result.violations.length > 0) {
      console.log(`Violations (${result.violations.length}):`)
      for (const v of result.violations) {
        console.log(`  [${v.severity}] ${v.message}`)
      }
    } else {
      console.log('No violations.')
    }
  }
}

module.exports = lexer
