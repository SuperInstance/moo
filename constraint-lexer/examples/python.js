/**
 * examples/python.js — Python subset lexer with constraints
 *
 * Indentation tracking + smooth transitions
 */

const cl = require('../constraint-lexer')

const lexer = cl.compile({
  rules: {
    // Keywords
    IF: 'if',
    ELSE: 'else',
    ELIF: 'elif',
    WHILE: 'while',
    FOR: 'for',
    IN: 'in',
    DEF: 'def',
    RETURN: 'return',
    CLASS: 'class',
    IMPORT: 'import',
    FROM: 'from',
    AS: 'as',
    PASS: 'pass',
    BREAK: 'break',
    CONTINUE: 'continue',
    TRY: 'try',
    EXCEPT: 'except',
    FINALLY: 'finally',
    WITH: 'with',
    RAISE: 'raise',
    YIELD: 'yield',
    LAMBDA: 'lambda',
    AND: 'and',
    OR: 'or',
    NOT: 'not',
    IS: 'is',
    NONE: 'None',
    TRUE: 'True',
    FALSE: 'False',

    // Operators
    OP: /[+\-*/%=<>!&|^~]+/,
    ARROW: '->',
    DOT: '.',

    // Delimiters
    LPAREN: '(',
    RPAREN: ')',
    LBRACKET: '[',
    RBRACKET: ']',
    LBRACE: '{',
    RBRACE: '}',

    // Symbols
    COMMA: ',',
    COLON: ':',
    SEMICOLON: ';',
    AT: '@',

    // Values
    NUMBER: /(?:0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+|[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)/,
    STRING: /(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|'''[\s\S]*?'''|"""[\s\S]*?""")/,
    IDENT: /[a-zA-Z_]\w*/,

    // Whitespace and indentation
    NEWLINE: { match: /\n/, lineBreaks: true },
    INDENT: { match: /[ \t]+/, lineBreaks: false },
    COMMENT: /#[^\n]*/,
  },
  constraints: [
    // Balanced parentheses
    cl.constraints.balancedDelimiters('LPAREN', 'RPAREN'),
    // Balanced brackets
    cl.constraints.balancedDelimiters('LBRACKET', 'RBRACKET'),
    // Balanced braces
    cl.constraints.balancedDelimiters('LBRACE', 'RBRACE'),
    // No adjacent operators
    cl.constraints.noAdjacent('OP'),
    // Smooth transitions between token types
    cl.constraints.smoothTransitions({ threshold: 2.0, windowSize: 15 }),
    // Keyword context: 'def' should be followed by IDENT, 'if' by an expression
    cl.constraints.keywordContext({
      'def': { followedBy: ['IDENT'] },
      'class': { followedBy: ['IDENT'] },
      'import': { followedBy: ['IDENT', 'FROM'] },
      'return': { followedBy: ['IDENT', 'NUMBER', 'STRING', 'NONE', 'TRUE', 'FALSE', 'LPAREN', 'LBRACKET', 'LBRACE', 'NOT', 'NEWLINE'] },
    }),
  ],
})

// Demo
if (require.main === module) {
  const tests = [
    'def hello(name):\n    return name',
    'if x > 0:\n    print(x)\nelse:\n    pass',
    'def (broken:\n    pass',
    'x = ((1 + 2) * 3',
  ]

  for (const input of tests) {
    console.log(`\n=== Python: ${input.replace(/\n/g, '\\n')} ===`)
    const result = lexer.tokenize(input)
    const nonWs = result.tokens.filter(t => !['NEWLINE', 'INDENT', 'COMMENT'].includes(t.type))
    console.log(`Tokens: ${nonWs.map(t => `${t.type}(${t.value})`).join(' ')}`)
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
