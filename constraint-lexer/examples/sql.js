/**
 * examples/sql.js — SQL lexer with constraints
 *
 * Balanced parens + keyword context preservation
 */

const cl = require('../constraint-lexer')

const lexer = cl.compile({
  rules: {
    // Keywords
    SELECT: 'SELECT',
    FROM: 'FROM',
    WHERE: 'WHERE',
    INSERT: 'INSERT',
    INTO: 'INTO',
    VALUES: 'VALUES',
    UPDATE: 'UPDATE',
    SET: 'SET',
    DELETE: 'DELETE',
    CREATE: 'CREATE',
    TABLE: 'TABLE',
    DROP: 'DROP',
    ALTER: 'ALTER',
    ADD: 'ADD',
    COLUMN: 'COLUMN',
    INDEX: 'INDEX',
    JOIN: 'JOIN',
    ON: 'ON',
    INNER: 'INNER',
    LEFT: 'LEFT',
    RIGHT: 'RIGHT',
    OUTER: 'OUTER',
    FULL: 'FULL',
    CROSS: 'CROSS',
    NATURAL: 'NATURAL',
    AND: 'AND',
    OR: 'OR',
    NOT: 'NOT',
    IN: 'IN',
    IS: 'IS',
    NULL: 'NULL',
    LIKE: 'LIKE',
    BETWEEN: 'BETWEEN',
    EXISTS: 'EXISTS',
    AS: 'AS',
    ORDER: 'ORDER',
    BY: 'BY',
    GROUP: 'GROUP',
    HAVING: 'HAVING',
    LIMIT: 'LIMIT',
    OFFSET: 'OFFSET',
    ASC: 'ASC',
    DESC: 'DESC',
    DISTINCT: 'DISTINCT',
    ALL: 'ALL',
    COUNT: 'COUNT',
    SUM: 'SUM',
    AVG: 'AVG',
    MIN: 'MIN',
    MAX: 'MAX',
    UNION: 'UNION',
    INTERSECT: 'INTERSECT',
    EXCEPT: 'EXCEPT',
    CASE: 'CASE',
    WHEN: 'WHEN',
    THEN: 'THEN',
    ELSE: 'ELSE',
    END: 'END',
    PRIMARY: 'PRIMARY',
    KEY: 'KEY',
    FOREIGN: 'FOREIGN',
    REFERENCES: 'REFERENCES',
    CONSTRAINT: 'CONSTRAINT',
    UNIQUE: 'UNIQUE',
    CHECK: 'CHECK',
    DEFAULT: 'DEFAULT',
    IF: 'IF',
    BEGIN: 'BEGIN',
    COMMIT: 'COMMIT',
    ROLLBACK: 'ROLLBACK',
    TRANSACTION: 'TRANSACTION',
    TYPE: 'TYPE',
    VARCHAR: 'VARCHAR',
    INTEGER: 'INTEGER',
    INT: 'INT',
    TEXT: 'TEXT',
    BOOLEAN: 'BOOLEAN',
    FLOAT: 'FLOAT',
    DOUBLE: 'DOUBLE',
    DATE: 'DATE',
    TIMESTAMP: 'TIMESTAMP',
    TRUE: 'TRUE',
    FALSE: 'FALSE',

    // Symbols
    STAR: '*',
    COMMA: ',',
    DOT: '.',
    SEMICOLON: ';',
    EQ: '=',
    NEQ: ['<>', '!='],
    LT: '<',
    GT: '>',
    LTE: '<=',
    GTE: '>=',
    LPAREN: '(',
    RPAREN: ')',

    // Values
    NUMBER: /[0-9]+(?:\.[0-9]+)?/,
    STRING: /'(?:[^']|'')*'/,
    IDENT: /[a-zA-Z_]\w*/,

    // Whitespace
    WS: { match: /\s+/, lineBreaks: true },
  },
  constraints: [
    // Balanced parentheses
    cl.constraints.balancedDelimiters('LPAREN', 'RPAREN'),
    // No adjacent comparison operators
    cl.constraints.noAdjacent(['EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE']),
    // Smooth transitions
    cl.constraints.smoothTransitions({ threshold: 1.5, windowSize: 10 }),
    // Keyword context: SQL keywords should be followed by appropriate tokens
    cl.constraints.keywordContext({
      'SELECT': { followedBy: ['STAR', 'IDENT', 'DISTINCT', 'ALL', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'WS'] },
      'FROM': { followedBy: ['IDENT', 'LPAREN', 'WS'] },
      'WHERE': { followedBy: ['IDENT', 'NOT', 'EXISTS', 'LPAREN', 'WS'] },
      'ORDER': { followedBy: ['BY', 'WS'] },
      'GROUP': { followedBy: ['BY', 'WS'] },
      'INSERT': { followedBy: ['INTO', 'WS'] },
      'JOIN': { followedBy: ['IDENT', 'LPAREN', 'WS'] },
      'ON': { followedBy: ['IDENT', 'WS'] },
    }),
  ],
})

// Demo
if (require.main === module) {
  const tests = [
    'SELECT name, age FROM users WHERE age > 18 ORDER BY name ASC;',
    'SELECT * FROM (SELECT id FROM products)',
    'SELECT * FROM products WHERE price = = 10;',
    'INSERT INTO users (name) VALUES (\'Alice\');',
  ]

  for (const input of tests) {
    console.log(`\n=== SQL: ${input} ===`)
    const result = lexer.tokenize(input)
    const nonWs = result.tokens.filter(t => t.type !== 'WS')
    console.log(`Tokens: ${nonWs.map(t => `${t.type}`).join(' ')}`)
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
