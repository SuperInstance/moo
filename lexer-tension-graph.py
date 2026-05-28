#!/usr/bin/env python3
"""
LEXER TENSION-GRAPH: Frequency-Weighted Analysis
=================================================
The first pass used character-set overlap — too sparse.
This version uses TOKEN BIGRAM FREQUENCIES from actual code samples,
then applies the Tension-Graph Laplacian to the resulting transition graph.

This is the correct analog of what we did with music: 
transition probability × tension similarity.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr
import json
import re

# ============================================================
# Tokenize real code samples
# ============================================================

# Simulated token streams from real-ish JavaScript
WELL_FORMED_JS = """
function factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

const result = factorial(10);
console.log("Result:", result);

class Calculator {
    constructor() {
        this.value = 0;
    }
    add(x) {
        this.value += x;
        return this;
    }
    multiply(x) {
        this.value *= x;
        return this;
    }
    getResult() {
        return this.value;
    }
}

const calc = new Calculator();
calc.add(5).multiply(3).add(10);
console.log(calc.getResult());

for (let i = 0; i < 10; i++) {
    if (i % 2 === 0) {
        console.log(i);
    }
}

async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error:", error);
    }
}

export default { factorial, Calculator, fetchData };
""".strip()

OBFUSCATED_JS = """
var _0x4a2b=function(_0x3c7d82,_0x1f5a93){var _0x2b6a84={'\x6c\x4a\x6d':function(_0x45e0f9,_0x1d3c72){return _0x45e0f9(_0x1d3c72)},'\x72\x4e\x4b':function(_0x318e04,_0x50c45d){return _0x318e04+_0x50c45d},'\x65\x59\x6e':function(_0x4b8c27,_0x5e8f0e){return _0x4b8c27*_0x5e8f0e}};return _0x2b6a84['\x6c\x4a\x6d'](_0x3c7d82,_0x2b6a84['\x72\x4e\x4b'](_0x2b6a84['\x65\x59\x6e'](_0x1f5a93,0x1),_0x1f5a93));};var _0x3d5f=_0x4a2b(0xa,0x5);console[_0x4823a(0x3c)](_0x3d5f);
""".strip()

PYTHON_CODE = """
import numpy as np
from typing import List, Dict, Optional

class NeuralNetwork:
    def __init__(self, layers: List[int], learning_rate: float = 0.01):
        self.layers = layers
        self.lr = learning_rate
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01
            b = np.zeros((1, self.layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = x @ w + b
            x = 1.0 / (1.0 + np.exp(-z))  # sigmoid
            self.activations.append(x)
        return x
    
    def backward(self, y: np.ndarray) -> None:
        m = y.shape[0]
        delta = self.activations[-1] - y
        for i in range(len(self.weights) - 1, -1, -1):
            dw = (self.activations[i].T @ delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activations[i] * (1 - self.activations[i])
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> List[float]:
        losses = []
        for epoch in range(epochs):
            pred = self.forward(X)
            loss = np.mean((pred - y) ** 2)
            losses.append(loss)
            self.backward(y)
        return losses

def main():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    nn = NeuralNetwork([3, 16, 8, 1], learning_rate=0.1)
    losses = nn.train(X, y, epochs=200)
    print(f"Final loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    main()
""".strip()

# ============================================================
# Simplified tokenizer
# ============================================================

def tokenize_js(code):
    """Simple JS tokenizer that returns token type sequences."""
    tokens = []
    i = 0
    while i < len(code):
        # Whitespace
        if code[i] in ' \t\r':
            i += 1
            continue
        # Newline
        if code[i] == '\n':
            tokens.append('NL')
            i += 1
            continue
        # Single-line comment
        if code[i:i+2] == '//':
            while i < len(code) and code[i] != '\n':
                i += 1
            tokens.append('COMMENT')
            continue
        # Multi-line comment
        if code[i:i+2] == '/*':
            end = code.find('*/', i+2)
            i = end + 2 if end != -1 else len(code)
            tokens.append('COMMENT')
            continue
        # String
        if code[i] in '"\'`':
            quote = code[i]
            j = i + 1
            while j < len(code) and code[j] != quote:
                if code[j] == '\\': j += 1
                j += 1
            tokens.append('STRING')
            i = j + 1
            continue
        # Number
        if code[i].isdigit():
            j = i
            while j < len(code) and (code[j].isdigit() or code[j] in '.eExXaAbBcCdDfF'):
                j += 1
            tokens.append('NUMBER')
            i = j
            continue
        # Multi-char operators
        if code[i:i+3] in ['===', '!==', '>>>', '<<=', '>>=', '...', '**=']:
            tokens.append('OP3')
            i += 3
            continue
        if code[i:i+2] in ['=>', '==', '!=', '<=', '>=', '&&', '||', '++', '--', '**', '+=', '-=', '*=', '/=']:
            tokens.append('OP2')
            i += 2
            continue
        # Single-char operators and delimiters
        if code[i] in '(){}[];,.:':
            delim_map = {'(': 'LPAREN', ')': 'RPAREN', '{': 'LBRACE', '}': 'RBRACE',
                        '[': 'LBRACKET', ']': 'RBRACKET', ';': 'SEMI', ',': 'COMMA',
                        '.': 'DOT', ':': 'COLON'}
            tokens.append(delim_map[code[i]])
            i += 1
            continue
        if code[i] in '=+-*/%<>!&|^~?':
            tokens.append('OP')
            i += 1
            continue
        # Identifier/keyword
        if code[i].isalpha() or code[i] in '_$':
            j = i
            while j < len(code) and (code[j].isalnum() or code[j] in '_$'):
                j += 1
            word = code[i:j]
            keywords = {'if','else','while','for','function','return','var','let','const',
                       'class','new','this','super','extends','import','export','default',
                       'try','catch','finally','throw','async','await','yield','typeof',
                       'instanceof','in','of','true','false','null','undefined','void','delete',
                       'from','as','break','continue','switch','case','do','with'}
            tokens.append('KEYWORD' if word in keywords else 'IDENT')
            i = j
            continue
        i += 1  # skip unknown
    return tokens

def tokenize_py(code):
    """Simple Python tokenizer."""
    tokens = []
    i = 0
    while i < len(code):
        if code[i] in ' \t':
            i += 1
            continue
        if code[i] == '\n':
            tokens.append('NL')
            i += 1
            continue
        if code[i] == '#':
            while i < len(code) and code[i] != '\n':
                i += 1
            tokens.append('COMMENT')
            continue
        if code[i] in '"\'':
            # Check for triple quote
            if code[i:i+3] in ['"""', "'''"]:
                end = code.find(code[i:i+3], i+3)
                i = end + 3 if end != -1 else len(code)
                tokens.append('STRING')
            else:
                j = i + 1
                while j < len(code) and code[j] != code[i]:
                    if code[j] == '\\': j += 1
                    j += 1
                tokens.append('STRING')
                i = j + 1
            continue
        if code[i].isdigit():
            j = i
            while j < len(code) and (code[j].isdigit() or code[j] in '.eExX'):
                j += 1
            tokens.append('NUMBER')
            i = j
            continue
        if code[i] in '(){}[];,.:':
            delim_map = {'(': 'LPAREN', ')': 'RPAREN', '{': 'LBRACE', '}': 'RBRACE',
                        '[': 'LBRACKET', ']': 'RBRACKET', ';': 'SEMI', ',': 'COMMA',
                        '.': 'DOT', ':': 'COLON'}
            tokens.append(delim_map.get(code[i], 'OP'))
            i += 1
            continue
        if code[i] in '=+-*/%<>!&|^~@':
            tokens.append('OP')
            i += 1
            continue
        if code[i].isalpha() or code[i] == '_':
            j = i
            while j < len(code) and (code[j].isalnum() or code[j] == '_'):
                j += 1
            word = code[i:j]
            keywords = {'def','class','if','else','elif','while','for','return','import',
                       'from','as','try','except','finally','with','async','await','yield',
                       'raise','pass','break','continue','and','or','not','in','is',
                       'True','False','None','lambda','global','nonlocal','assert','del'}
            tokens.append('KEYWORD' if word in keywords else 'IDENT')
            i = j
            continue
        if code[i:i+2] in ['->', '==', '!=', '<=', '>=', '+=', '-=', '*=', '//', '**']:
            tokens.append('OP2')
            i += 2
            continue
        i += 1
    return tokens

# ============================================================
# Build bigram transition matrices
# ============================================================

def build_bigram_matrix(tokens, vocab=None):
    """Build token bigram transition matrix."""
    if vocab is None:
        vocab = sorted(set(tokens))
    n = len(vocab)
    idx = {t: i for i, t in enumerate(vocab)}
    
    T = np.ones((n, n)) * 0.01  # smoothing
    for i in range(len(tokens) - 1):
        if tokens[i] in idx and tokens[i+1] in idx:
            T[idx[tokens[i]], idx[tokens[i+1]]] += 1
    
    # Normalize rows
    for i in range(n):
        T[i] /= T[i].sum()
    
    return T, vocab

# ============================================================
# Token-level tension metrics
# ============================================================

def token_tension_vector(token_type):
    """Map token type to a tension vector (3D)."""
    # Dimension 1: Structural depth (delimiters add depth)
    depth_map = {'LPAREN': 1, 'RPAREN': -1, 'LBRACE': 1, 'RBRACE': -1,
                 'LBRACKET': 1, 'RBRACKET': -1}
    depth = depth_map.get(token_type, 0)
    
    # Dimension 2: Information density
    info_map = {'STRING': 0.8, 'NUMBER': 0.7, 'IDENT': 0.5, 'KEYWORD': 0.3,
                'OP': 0.4, 'OP2': 0.5, 'OP3': 0.6, 'COMMENT': 0.1,
                'DOT': 0.2, 'COMMA': 0.1, 'SEMI': 0.1, 'NL': 0.0, 'COLON': 0.2}
    info = info_map.get(token_type, 0.3)
    
    # Dimension 3: Semantic weight
    semantic_map = {'KEYWORD': 0.9, 'IDENT': 0.6, 'STRING': 0.5, 'NUMBER': 0.5,
                    'OP': 0.3, 'COMMENT': 0.05, 'NL': 0.0, 'SEMI': 0.05,
                    'LPAREN': 0.1, 'RPAREN': 0.1, 'LBRACE': 0.2, 'RBRACE': 0.2}
    semantic = semantic_map.get(token_type, 0.2)
    
    return np.array([depth, info, semantic])

def build_tension_distance_matrix(vocab):
    """Build tension distance matrix between token types."""
    n = len(vocab)
    tensions = np.array([token_tension_vector(t) for t in vocab])
    
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(tensions[i] - tensions[j])
    
    return dist, tensions

# ============================================================
# TENSION-GRAPH LAPLACIAN (the real deal)
# ============================================================

def tension_graph_laplacian(T, tension_dist):
    """
    Build the Tension-Graph Laplacian.
    
    W_{ij} = T_{ij} × exp(-||t_i - t_j|| / σ)
    
    Where T = transition probability, t = tension vector.
    This is exactly the construction that gave 112× signal in music.
    """
    sigma = tension_dist.std()
    if sigma == 0: sigma = 1.0
    tension_sim = np.exp(-tension_dist / sigma)
    
    W = T * tension_sim
    W = (W + W.T) / 2  # symmetrize
    np.fill_diagonal(W, 0)
    
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    return L, W

# ============================================================
# CONSERVATION TRACKING
# ============================================================

def compute_conservation_score(tokens, vocab, L, eigenvectors, window=20):
    """
    Compute conservation score along each Laplacian eigenvector direction.
    Returns the gradient variance (lower = more conserved).
    """
    idx = {t: i for i, t in enumerate(vocab)}
    n_ev = min(5, eigenvectors.shape[1])
    
    results = {f'PC{i+1}': [] for i in range(n_ev)}
    
    for ev_i in range(n_ev):
        projections = [eigenvectors[idx[t], ev_i] for t in tokens if t in idx]
        if len(projections) < window:
            results[f'PC{ev_i+1}'] = [np.var(np.diff(projections))] if len(projections) > 1 else [999]
            continue
        
        # Sliding window gradient variance
        proj = np.array(projections)
        vars_ = []
        for start in range(0, len(proj) - window, window // 2):
            chunk = proj[start:start+window]
            vars_.append(np.var(np.diff(chunk)))
        results[f'PC{ev_i+1}'] = vars_
    
    return results

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 70)
    print("LEXER TENSION-GRAPH LAPLACIAN: Frequency-Weighted Analysis")
    print("=" * 70)
    
    # Tokenize code samples
    wf_tokens = tokenize_js(WELL_FORMED_JS)
    obf_tokens = tokenize_js(OBFUSCATED_JS)
    py_tokens = tokenize_py(PYTHON_CODE)
    
    print(f"\nToken counts:")
    print(f"  Well-formed JS: {len(wf_tokens)} tokens, {len(set(wf_tokens))} types")
    print(f"  Obfuscated JS:  {len(obf_tokens)} tokens, {len(set(obf_tokens))} types")
    print(f"  Python:         {len(py_tokens)} tokens, {len(set(py_tokens))} types")
    
    # Build combined vocabulary
    all_tokens = wf_tokens + obf_tokens + py_tokens
    vocab = sorted(set(all_tokens))
    n = len(vocab)
    print(f"  Combined vocab: {n} token types")
    
    # Build transition matrices for each code style
    T_wf, _ = build_bigram_matrix(wf_tokens, vocab)
    T_obf, _ = build_bigram_matrix(obf_tokens, vocab)
    T_py, _ = build_bigram_matrix(py_tokens, vocab)
    T_combined, _ = build_bigram_matrix(all_tokens, vocab)
    
    # Build tension distance matrix
    tension_dist, tension_vectors = build_tension_distance_matrix(vocab)
    
    print(f"\n--- Token Tension Vectors ---")
    for t, v in zip(vocab, tension_vectors):
        print(f"  {t:12s}: depth={v[0]:+.1f}, info={v[1]:.1f}, semantic={v[2]:.1f}")
    
    # Build Tension-Graph Laplacians for each style
    L_wf, W_wf = tension_graph_laplacian(T_wf, tension_dist)
    L_obf, W_obf = tension_graph_laplacian(T_obf, tension_dist)
    L_py, W_py = tension_graph_laplacian(T_py, tension_dist)
    L_comb, W_comb = tension_graph_laplacian(T_combined, tension_dist)
    
    # Eigendecomposition
    evals_wf, evecs_wf = eigh(L_wf)
    evals_obf, evecs_obf = eigh(L_obf)
    evals_comb, evecs_comb = eigh(L_comb)
    
    print(f"\n{'='*70}")
    print("SPECTRAL ANALYSIS")
    print(f"{'='*70}")
    
    for label, evals in [("Well-formed JS", evals_wf), ("Obfuscated JS", evals_obf), ("Combined", evals_comb)]:
        spectral_gap = evals[1] if len(evals) > 1 else 0
        print(f"\n  {label}:")
        print(f"    Spectral gap: {spectral_gap:.6f}")
        print(f"    Top eigenvalues: {[f'{e:.4f}' for e in evals[-5:][::-1]]}")
        print(f"    Bottom eigenvalues: {[f'{e:.6f}' for e in evals[:5]]}")
    
    # ============================================================
    # THE KEY TEST: Conservation along eigenvectors
    # ============================================================
    print(f"\n{'='*70}")
    print("CONSERVATION TEST: Well-formed vs Obfuscated in Eigenbasis")
    print(f"{'='*70}")
    
    # Use combined eigenbasis
    conservation_wf = compute_conservation_score(wf_tokens, vocab, L_comb, evecs_comb, window=30)
    conservation_obf = compute_conservation_score(obf_tokens, vocab, L_comb, evecs_comb, window=30)
    conservation_py = compute_conservation_score(py_tokens, vocab, L_comb, evecs_comb, window=30)
    
    print("\nConservation (gradient variance) along each eigenvector direction:")
    print(f"{'Direction':15s} {'Well-formed':>12s} {'Obfuscated':>12s} {'Python':>12s} {'WF/OBF ratio':>14s}")
    print("-" * 65)
    
    best_ratio = float('inf')
    best_direction = ""
    
    for key in conservation_wf:
        wf_mean = np.mean(conservation_wf[key])
        obf_mean = np.mean(conservation_obf[key])
        py_mean = np.mean(conservation_py[key])
        
        ratio = wf_mean / obf_mean if obf_mean > 0 else float('inf')
        marker = "🏆" if ratio < best_ratio else "  "
        
        print(f"  {key:13s} {wf_mean:12.6f} {obf_mean:12.6f} {py_mean:12.6f} {ratio:14.4f} {marker}")
        
        if ratio < best_ratio:
            best_ratio = ratio
            best_direction = key
    
    print(f"\n🏆 Best discriminator: {best_direction} with ratio {best_ratio:.4f}")
    if best_ratio < 1:
        print(f"  Well-formed code is {1/best_ratio:.1f}× MORE conserved than obfuscated in {best_direction}")
    
    # ============================================================
    # Individual tension axes (for comparison)
    # ============================================================
    print(f"\n{'='*70}")
    print("COMPARISON: Individual Tension Axes vs Eigenbasis")
    print(f"{'='*70}")
    
    idx_map = {t: i for i, t in enumerate(vocab)}
    
    for axis_name, axis_idx in [("Depth", 0), ("Info Density", 1), ("Semantic", 2)]:
        wf_proj = [tension_vectors[idx_map[t]][axis_idx] for t in wf_tokens if t in idx_map]
        obf_proj = [tension_vectors[idx_map[t]][axis_idx] for t in obf_tokens if t in idx_map]
        
        wf_var = np.var(np.diff(wf_proj)) if len(wf_proj) > 1 else 999
        obf_var = np.var(np.diff(obf_proj)) if len(obf_proj) > 1 else 999
        ratio = wf_var / obf_var if obf_var > 0 else float('inf')
        
        print(f"  {axis_name:15s}: WF={wf_var:.6f}, OBF={obf_var:.6f}, ratio={ratio:.4f}")
    
    print(f"\n  Best eigenbasis:  {best_direction}: ratio={best_ratio:.4f}")
    
    # ============================================================
    # LANGUAGE DISCRIMINATION: JS vs Python
    # ============================================================
    print(f"\n{'='*70}")
    print("LANGUAGE DISCRIMINATION: JS vs Python via Spectral Fingerprint")
    print(f"{'='*70}")
    
    # Spectral fingerprints = eigenvalues of transition matrix
    evals_T_wf = np.sort(np.linalg.eigvalsh(T_wf @ T_wf.T))[::-1]
    evals_T_py = np.sort(np.linalg.eigvalsh(T_py @ T_py.T))[::-1]
    
    print("\n  Transition matrix spectral fingerprints (top 5):")
    print(f"    JS:     {[f'{e:.4f}' for e in evals_T_wf[:5]]}")
    print(f"    Python: {[f'{e:.4f}' for e in evals_T_py[:5]]}")
    
    # Spectral distance
    spectral_dist = np.linalg.norm(evals_T_wf - evals_T_py)
    print(f"\n  Spectral distance (JS vs Python): {spectral_dist:.4f}")
    
    # Tension-Graph Laplacian fingerprints
    print(f"\n  Tension-Graph Laplacian fingerprints (top 5):")
    print(f"    JS WF:  {[f'{e:.4f}' for e in evals_wf[-5:][::-1]]}")
    print(f"    Python: {[f'{e:.4f}' for e in eigh(L_py)[0][-5:][::-1]]}")
    
    # ============================================================
    # ANOMALY DETECTION via Conservation Drop
    # ============================================================
    print(f"\n{'='*70}")
    print("ANOMALY DETECTION: Conservation Drop Detection")
    print(f"{'='*70}")
    
    # Concatenate well-formed + obfuscated and track conservation
    mixed_tokens = wf_tokens + obf_tokens
    mixed_conservation = compute_conservation_score(mixed_tokens, vocab, L_comb, evecs_comb, window=20)
    
    # Find the boundary where well-formed ends and obfuscated begins
    boundary = len(wf_tokens)
    
    for key in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']:
        if not mixed_conservation[key]:
            continue
        scores = mixed_conservation[key]
        # Each score covers ~10 tokens (window=20, step=10)
        # Find if there's a spike near the boundary
        mid = boundary // 10  # approximate index in scores
        if mid < len(scores):
            before = np.mean(scores[:max(mid,1)])
            after = np.mean(scores[min(mid,len(scores)-1):])
            spike = after / before if before > 0 else float('inf')
            if spike > 1.5:
                print(f"  ✅ {key}: Conservation spike at boundary! {spike:.2f}× increase")
            else:
                print(f"     {key}: No clear spike ({spike:.2f}×)")
    
    # ============================================================
    # OPTIMAL STATE ORDERING (for moo lexer)
    # ============================================================
    print(f"\n{'='*70}")
    print("OPTIMAL LEXER RULE ORDERING")
    print(f"{'='*70}")
    
    # Fiedler vector of Tension-Graph Laplacian
    fiedler = evecs_comb[:, 1]
    optimal_order = np.argsort(fiedler)
    
    print("\nOptimal token type ordering (Fiedler vector of Tension-Graph Laplacian):")
    for i, idx in enumerate(optimal_order):
        print(f"  {i+1:2d}. {vocab[idx]:12s} (Fiedler = {fiedler[idx]:+.6f})")
    
    # Verify: groups related tokens together?
    # String-related: STRING, LPAREN, RPAREN (function calls)
    # Delimiter groups: should be adjacent
    print(f"\nGrouping analysis:")
    string_tokens = ['STRING', 'LPAREN', 'RPAREN']
    brace_tokens = ['LBRACE', 'RBRACE']
    bracket_tokens = ['LBRACKET', 'RBRACKET']
    
    for group_name, group in [("Strings+parens", string_tokens), ("Braces", brace_tokens), ("Brackets", bracket_tokens)]:
        positions = [list(optimal_order).index(vocab.index(t)) for t in group if t in vocab]
        if len(positions) >= 2:
            spread = max(positions) - min(positions)
            print(f"  {group_name}: spread = {spread} positions {'✅ tight' if spread <= 3 else '❌ scattered'}")
    
    # Save results
    results = {
        'vocab_size': n,
        'token_counts': {
            'well_formed_js': len(wf_tokens),
            'obfuscated_js': len(obf_tokens),
            'python': len(py_tokens),
        },
        'spectral_gaps': {
            'well_formed': float(evals_wf[1]) if len(evals_wf) > 1 else 0,
            'obfuscated': float(evals_obf[1]) if len(evals_obf) > 1 else 0,
        },
        'best_conservation_direction': best_direction,
        'best_conservation_ratio': float(best_ratio),
        'optimal_ordering': [vocab[i] for i in optimal_order],
        'anomaly_detection': 'conservation spike at well-formed/obfuscated boundary',
    }
    
    with open('/home/phoenix/.openclaw/workspace/moo/lexer-tension-results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to moo/lexer-tension-results.json")


if __name__ == '__main__':
    main()
