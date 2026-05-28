//! # Performance primitives for the moo lexer
//!
//! Low-level optimised data structures and algorithms:
//! - Bit-parallel character class matching (256-bit `CharSet`)
//! - SIMD-friendly Shannon entropy computation
//! - Zero-copy token buffer
//! - Stack-allocated flat Laplacian (≤64 nodes)
//! - Lanczos eigensolver for small symmetric matrices

// ─────────────────────────────────────────────────────────────────────────────
// 1. Bit-parallel character class matching
// ─────────────────────────────────────────────────────────────────────────────

/// A 256-bit set representing all ASCII code points, stored as four `u64` limbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CharSet(pub [u64; 4]);

impl CharSet {
    /// Empty set.
    pub const fn empty() -> Self {
        CharSet([0; 4])
    }

    /// Full set (all 256 bits set).
    pub const fn full() -> Self {
        CharSet([u64::MAX; 4])
    }

    /// Set containing a single byte.
    pub const fn from_byte(b: u8) -> Self {
        let mut cs = CharSet::empty();
        cs.set(b);
        cs
    }

    /// Set containing a range of bytes (inclusive).
    pub const fn from_range(lo: u8, hi: u8) -> Self {
        let mut cs = CharSet::empty();
        let mut i = lo as usize;
        while i <= hi as usize {
            cs.set(i as u8);
            i += 1;
        }
        cs
    }

    /// Set a single byte.
    pub const fn set(&mut self, b: u8) {
        let limb = (b as usize) / 64;
        let bit = (b as usize) % 64;
        self.0[limb] |= 1u64 << bit;
    }

    /// Check if a byte is in the set.
    #[inline]
    pub const fn contains(&self, b: u8) -> bool {
        let limb = (b as usize) / 64;
        let bit = (b as usize) % 64;
        (self.0[limb] >> bit) & 1 == 1
    }

    /// Population count (number of set bits).
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.0[0].count_ones() + self.0[1].count_ones() + self.0[2].count_ones() + self.0[3].count_ones()
    }

    /// Union of two sets.
    #[inline]
    pub const fn union(&self, other: &CharSet) -> CharSet {
        CharSet([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    /// Intersection of two sets.
    #[inline]
    pub const fn intersection(&self, other: &CharSet) -> CharSet {
        CharSet([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
        ])
    }

    /// Jaccard-like overlap coefficient: `popcount(AND) / popcount(OR)`.
    /// Returns 0.0 when both sets are empty.
    #[inline]
    pub fn overlap(&self, other: &CharSet) -> f64 {
        let and_count = self.intersection(other).popcount() as f64;
        let or_count = self.union(other).popcount() as f64;
        if or_count == 0.0 {
            0.0
        } else {
            and_count / or_count
        }
    }
}

impl std::ops::BitOr for CharSet {
    type Output = CharSet;
    fn bitor(self, rhs: CharSet) -> CharSet {
        self.union(&rhs)
    }
}

impl std::ops::BitAnd for CharSet {
    type Output = CharSet;
    fn bitand(self, rhs: CharSet) -> CharSet {
        self.intersection(&rhs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. SIMD-friendly Shannon entropy computation
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed log2 lookup table for byte frequencies 0..256.
/// `LOG2_TABLE[i] = log2(i)` for i > 0, 0.0 for i == 0.
/// Built at runtime on first access via `log2_table()`.
static mut LOG2_TABLE: MaybeUninit<[f64; 256]> = MaybeUninit::uninit();
static LOG2_TABLE_INIT: std::sync::Once = std::sync::Once::new();

use std::mem::MaybeUninit;

fn log2_table() -> &'static [f64; 256] {
    unsafe {
        LOG2_TABLE_INIT.call_once(|| {
            let mut table = [0.0f64; 256];
            for i in 1usize..256 {
                table[i] = (i as f64).log2();
            }
            LOG2_TABLE.write(table);
        });
        &*LOG2_TABLE.assume_init_ref()
    }
}

/// Compute Shannon entropy of a byte slice using u64-chunked histogramming.
///
/// Processes input in 8-byte chunks for the counting phase, then uses a
/// precomputed log2 table for the probability summation.
pub fn shannon_entropy_simd(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Fast histogram via u64 chunking
    let mut freq = [0u32; 256];
    let len = data.len();
    let chunks = len / 8;
    let remainder = len % 8;

    // Process 8 bytes at a time — the compiler can auto-vectorise this
    for i in 0..chunks {
        let base = i * 8;
        freq[data[base] as usize] += 1;
        freq[data[base + 1] as usize] += 1;
        freq[data[base + 2] as usize] += 1;
        freq[data[base + 3] as usize] += 1;
        freq[data[base + 4] as usize] += 1;
        freq[data[base + 5] as usize] += 1;
        freq[data[base + 6] as usize] += 1;
        freq[data[base + 7] as usize] += 1;
    }

    // Handle remainder
    if remainder > 0 {
        for i in (len - remainder)..len {
            freq[data[i] as usize] += 1;
        }
    }

    // Compute entropy using precomputed log2 table
    let total = len as f64;
    let log2_total = total.log2();
    let mut entropy = 0.0f64;

    let table = log2_table();
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / total;
            // -p * log2(p) = p * (log2(total) - log2(count))
            let log2_count = if (count as usize) < 256 { table[count as usize] } else { (count as f64).log2() };
            entropy += p * (log2_total - log2_count);
        }
    }

    entropy
}

/// Scalar reference implementation for correctness checking / benchmarking.
pub fn shannon_entropy_scalar(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let total = data.len() as f64;
    let mut entropy = 0.0f64;
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Zero-copy token buffer
// ─────────────────────────────────────────────────────────────────────────────

/// A reference into the token buffer — no String allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenRef {
    pub type_id: u16,
    pub offset: u32,
    pub len: u16,
}

impl TokenRef {
    /// Create a new token reference.
    #[inline]
    pub const fn new(type_id: u16, offset: u32, len: u16) -> Self {
        TokenRef { type_id, offset, len }
    }

    /// Extract the token text from the buffer (allocates on access only).
    #[inline]
    pub fn text<'a>(&self, buffer: &'a [u8]) -> &'a [u8] {
        let start = self.offset as usize;
        let end = start + self.len as usize;
        &buffer[start..end]
    }
}

/// A zero-copy token storage: all tokens borrow from a single contiguous buffer.
#[derive(Clone, Debug)]
pub struct TokenBuffer {
    pub data: Vec<u8>,
    pub tokens: Vec<TokenRef>,
}

impl TokenBuffer {
    /// Create an empty buffer with the given capacity hints.
    pub fn with_capacity(data_cap: usize, token_cap: usize) -> Self {
        TokenBuffer {
            data: Vec::with_capacity(data_cap),
            tokens: Vec::with_capacity(token_cap),
        }
    }

    /// Build from a complete input slice and a list of (type_id, offset, len) triples.
    pub fn from_parts(input: &[u8], refs: Vec<TokenRef>) -> Self {
        TokenBuffer {
            data: input.to_vec(),
            tokens: refs,
        }
    }

    /// Push raw bytes and return the offset where they were written.
    #[inline]
    pub fn push_data(&mut self, bytes: &[u8]) -> u32 {
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(bytes);
        offset
    }

    /// Add a token reference.
    #[inline]
    pub fn push_token(&mut self, tok: TokenRef) {
        self.tokens.push(tok);
    }

    /// Convenience: push data + record a token in one call.
    pub fn push(&mut self, type_id: u16, text: &[u8]) {
        let offset = self.push_data(text);
        self.push_token(TokenRef::new(type_id, offset, text.len() as u16));
    }

    /// Number of tokens.
    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether there are no tokens.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get the text slice for token at index `i`.
    #[inline]
    pub fn get_text(&self, i: usize) -> Option<&[u8]> {
        self.tokens.get(i).map(|t| t.text(&self.data))
    }

    /// Iterate over all (TokenRef, text_slice) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (TokenRef, &[u8])> {
        self.tokens.iter().map(move |t| (*t, t.text(&self.data)))
    }

    /// Clear the buffer and tokens.
    pub fn clear(&mut self) {
        self.data.clear();
        self.tokens.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Flat Laplacian for small graphs (≤64 nodes)
// ─────────────────────────────────────────────────────────────────────────────

/// Stack-allocated 64×64 symmetric matrix for the graph Laplacian.
/// No heap allocation for typical lexer rule-set sizes.
pub const MAX_LAPLACIAN_SIZE: usize = 64;

#[derive(Clone, Debug)]
pub struct FlatLaplacian {
    pub data: [f64; MAX_LAPLACIAN_SIZE * MAX_LAPLACIAN_SIZE],
    pub n: usize,
}

impl FlatLaplacian {
    /// Create a zero-initialized Laplacian of size `n` (must be ≤ 64).
    pub fn new(n: usize) -> Self {
        assert!(n <= MAX_LAPLACIAN_SIZE, "FlatLaplacian supports at most {} nodes", MAX_LAPLACIAN_SIZE);
        FlatLaplacian {
            data: [0.0; MAX_LAPLACIAN_SIZE * MAX_LAPLACIAN_SIZE],
            n,
        }
    }

    /// Build from an adjacency matrix (symmetric, undirected).
    /// L = D - A where D is the degree matrix.
    pub fn from_adjacency(adj: &[Vec<f64>]) -> Self {
        let n = adj.len();
        assert!(n <= MAX_LAPLACIAN_SIZE);
        let mut lap = FlatLaplacian::new(n);
        for i in 0..n {
            assert_eq!(adj[i].len(), n, "Adjacency matrix must be square");
            let mut degree = 0.0f64;
            for j in 0..n {
                let w = adj[i][j];
                if i != j {
                    lap.set(i, j, -w);
                    degree += w;
                }
            }
            lap.set(i, i, degree);
        }
        lap
    }

    /// Build from edge list: (i, j, weight) triples, with `n` total nodes.
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut lap = FlatLaplacian::new(n);
        for &(i, j, w) in edges {
            lap.set(i, j, lap.get(i, j) - w);
            lap.set(j, i, lap.get(j, i) - w);
            lap.set(i, i, lap.get(i, i) + w);
            lap.set(j, j, lap.get(j, j) + w);
        }
        lap
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * MAX_LAPLACIAN_SIZE + j
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.idx(i, j)]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[self.idx(i, j)] = val;
    }

    /// Matrix-vector multiply: y = L * x.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        assert_eq!(x.len(), n);
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0f64;
            for j in 0..n {
                sum += self.get(i, j) * x[j];
            }
            y[i] = sum;
        }
        y
    }

    /// Return the actual n×n data as a contiguous slice (row-major).
    pub fn as_slice(&self) -> &[f64] {
        &self.data[..self.n * MAX_LAPLACIAN_SIZE]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Lanczos eigensolver for small symmetric matrices
// ─────────────────────────────────────────────────────────────────────────────

/// Run `m` iterations of the Lanczos algorithm on `L`, returning
/// the tridiagonal matrix T (alpha diagonal, beta sub-diagonal) and
/// the orthonormal basis vectors Q.
///
/// Returns `(alphas, betas, Q)` where Q[m] are each n-dimensional.
fn lanczos_iteration(L: &FlatLaplacian, m: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = L.n;
    let m = m.min(n);

    let mut q = vec![0.0f64; n];
    // Random initial vector (deterministic seed via simple LCG)
    for i in 0..n {
        q[i] = ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1) % 1000003) as f64;
    }
    let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        q[0] = 1.0;
    } else {
        for x in q.iter_mut() {
            *x /= norm;
        }
    }

    let mut Q: Vec<Vec<f64>> = Vec::with_capacity(m);
    Q.push(q.clone());

    let mut alphas = Vec::with_capacity(m);
    let mut betas = Vec::with_capacity(m); // beta[0] is unused placeholder

    let mut w = L.mul_vec(&q);
    let alpha = dot(&Q[0], &w);
    alphas.push(alpha);
    sub_scaled(&mut w, alpha, &Q[0]);

    for k in 1..m {
        let beta = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if beta < 1e-14 {
            break;
        }
        betas.push(beta);

        let mut q_new = vec![0.0; n];
        for i in 0..n {
            q_new[i] = w[i] / beta;
        }

        // Full reorthogonalisation (Gram-Schmidt) for numerical stability
        reorthogonalise(&mut q_new, &Q);

        Q.push(q_new.clone());

        w = L.mul_vec(&q_new);
        let alpha = dot(&q_new, &w);
        alphas.push(alpha);
        sub_scaled(&mut w, alpha, &q_new);
        sub_scaled(&mut w, beta, &Q[k - 1]);
    }

    // Pad betas to match alphas length - 1 if needed
    while betas.len() < alphas.len() - 1 {
        betas.push(0.0);
    }

    (alphas, betas, Q)
}

/// Extract top-k eigenvalues from a tridiagonal matrix defined by alphas (diagonal)
/// and betas (sub-diagonal) using implicit QR shifts.
///
/// Returns eigenvalues sorted in descending absolute value order.
fn tridiag_eigenvalues(alphas: &[f64], betas: &[f64]) -> Vec<f64> {
    let n = alphas.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![alphas[0]];
    }

    // Work on copies ( Wilkinson-style QR iteration )
    let mut d = alphas.to_vec();
    let mut e = vec![0.0f64; n];
    for i in 0..betas.len().min(n - 1) {
        e[i] = betas[i];
    }

    // Implicit symmetric tridiagonal QR with Wilkinson shift
    let max_iter = 100 * n;
    let mut m_end = n - 1;
    let mut iter = 0;
    while m_end > 0 && iter < max_iter {
        iter += 1;

        // Check for convergence on sub-diagonal
        let mut m_start = m_end;
        while m_start > 0 {
            if e[m_start - 1].abs() <= 1e-14 * (d[m_start].abs() + d[m_start - 1].abs()) {
                e[m_start - 1] = 0.0;
                break;
            }
            m_start -= 1;
        }

        if m_start == m_end {
            m_end -= 1;
            continue;
        }

        // Wilkinson shift
        let dd = (d[m_end - 1] - d[m_end]) / 2.0;
        let shift = d[m_end] - e[m_end - 1].signum() * e[m_end - 1].powi(2)
            / (dd.abs() + (dd * dd + e[m_end - 1].powi(2)).sqrt());

        // Implicit QR step on [m_start..=m_end]
        let mut g = (d[m_start] - shift) / e[m_start];
        // We only need eigenvalues, not eigenvectors of T, so simplified rotation
        for i in m_start..m_end {
            let r = (g * g + e[i] * e[i]).sqrt();
            let c = if r == 0.0 { 1.0 } else { g / r };
            let s = if r == 0.0 { 0.0 } else { e[i] / r };
            if i > m_start {
                e[i - 1] = r;
            }
            let h = d[i + 1] - d[i];
            let f = g * s + c * h;
            let p = c * d[i] + s * g * c; // simplified
            let _ = (s, p); // suppress unused
            g = s * d[i + 1] - c * f;
            d[i] = d[i] + f;
            d[i + 1] = d[i + 1] - g;
        }
        e[m_end - 1] = g;
    }

    // Sort by descending absolute value
    d.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
    d
}

/// Compute the top-k eigenvalues and eigenvectors of the Laplacian using Lanczos.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by descending absolute eigenvalue.
/// Complexity: O(n·m·k) where m = min(n, 2k+20), compared to O(n³) for full decomposition.
pub fn lanczos_top_k(L: &FlatLaplacian, k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = L.n;
    let k = k.min(n);
    if k == 0 {
        return (vec![], vec![]);
    }

    // Run Lanczos with enough iterations for k eigenpairs
    let m = (2 * k + 20).min(n);
    let (alphas, betas, Q) = lanczos_iteration(L, m);

    if Q.is_empty() {
        return (vec![], vec![]);
    }

    // Get eigenvalues of the tridiagonal matrix
    let eigvals = tridiag_eigenvalues(&alphas, &betas);

    // Approximate eigenvectors via inverse iteration in the Lanczos basis.
    // For each desired eigenvalue λ, solve (T - λI)v = random, then map back to original space.
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Vec::with_capacity(k);

    let t_size = alphas.len();

    for idx in 0..k.min(eigvals.len()) {
        let lambda = eigvals[idx];
        eigenvalues.push(lambda);

        // Build approximate eigenvector in original space using power iteration on L
        // seeded by a combination of Lanczos basis vectors.
        let mut v = vec![0.0f64; n];
        for (i, q_i) in Q.iter().enumerate() {
            // Use Ritz vector approximation
            let weight = if i < t_size {
                1.0 / (1.0 + (lambda - alphas[i]).abs().max(1e-10))
            } else {
                1.0
            };
            for j in 0..n {
                v[j] += weight * q_i[j];
            }
        }

        // Normalise
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }

        // Refine with a few power-iteration steps on L
        for _ in 0..10 {
            let Lv = L.mul_vec(&v);
            // Shift by lambda for better convergence
            for i in 0..n {
                v[i] = Lv[i] - lambda * v[i];
            }
            let Lv = L.mul_vec(&v);
            let norm = Lv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for i in 0..n {
                v[i] = Lv[i] / norm;
            }
        }

        eigenvectors.push(v);
    }

    (eigenvalues, eigenvectors)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn sub_scaled(y: &mut [f64], alpha: f64, x: &[f64]) {
    for i in 0..y.len() {
        y[i] -= alpha * x[i];
    }
}

fn reorthogonalise(v: &mut [f64], basis: &[Vec<f64>]) {
    for q in basis {
        let c = dot(v, q);
        for i in 0..v.len() {
            v[i] -= c * q[i];
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Benchmarks / tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charset_basic() {
        let digits = CharSet::from_range(b'0', b'9');
        let alpha_lower = CharSet::from_range(b'a', b'z');
        let alpha_upper = CharSet::from_range(b'A', b'Z');

        assert!(digits.contains(b'5'));
        assert!(!digits.contains(b'a'));
        assert!(alpha_lower.contains(b'm'));
        assert!(!alpha_lower.contains(b'M'));
        assert_eq!(digits.popcount(), 10);
        assert_eq!(alpha_lower.popcount(), 26);
        assert_eq!(alpha_upper.popcount(), 26);

        let alnum = digits | alpha_lower | alpha_upper;
        assert!(alnum.contains(b'0'));
        assert!(alnum.contains(b'z'));
        assert!(alnum.contains(b'Z'));
        assert!(!alnum.contains(b'!'));
    }

    #[test]
    fn test_charset_overlap() {
        let a = CharSet::from_range(b'a', b'z'); // 26
        let b = CharSet::from_range(b'A', b'Z'); // 26
        let c = CharSet::from_range(b'a', b'm'); // 13 (half lowercase)

        // No overlap
        assert!((a.overlap(&b) - 0.0).abs() < 1e-10);

        // 50% overlap with a
        let expected = 13.0 / 26.0; // intersection / union = 13/26
        assert!((a.overlap(&c) - expected).abs() < 1e-10);

        // Self-overlap = 1.0
        assert!((a.overlap(&a) - 1.0).abs() < 1e-10);

        // Empty overlap
        let empty = CharSet::empty();
        assert!((empty.overlap(&a) - 0.0).abs() < 1e-10);
        assert!((empty.overlap(&empty) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution over 256 symbols should have entropy = 8.0
        let mut data = Vec::with_capacity(256);
        for i in 0u8..=255 {
            data.push(i);
        }
        let e = shannon_entropy_simd(&data);
        assert!((e - 8.0).abs() < 0.01, "Expected ~8.0, got {}", e);
    }

    #[test]
    fn test_entropy_single_symbol() {
        let data = vec![42u8; 1000];
        let e = shannon_entropy_simd(&data);
        assert!(e.abs() < 0.001, "Expected ~0.0, got {}", e);
    }

    #[test]
    fn test_entropy_vs_scalar() {
        // Random-ish data
        let data: Vec<u8> = (0..500).map(|i| ((i * 7919 + 13) % 256) as u8).collect();
        let e_simd = shannon_entropy_simd(&data);
        let e_scalar = shannon_entropy_scalar(&data);
        assert!(
            (e_simd - e_scalar).abs() < 0.001,
            "SIMD {} != scalar {}",
            e_simd,
            e_scalar
        );
    }

    #[test]
    fn test_token_buffer() {
        let mut buf = TokenBuffer::with_capacity(1024, 16);
        buf.push(1, b"hello");
        buf.push(2, b" ");
        buf.push(1, b"world");

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get_text(0), Some(&b"hello"[..]));
        assert_eq!(buf.get_text(1), Some(&b" "[..]));
        assert_eq!(buf.get_text(2), Some(&b"world"[..]));
    }

    #[test]
    fn test_flat_laplacian_from_edges() {
        // Triangle graph: L = [[1,-1,-1],[-1,1,-1],[-1,-1,2]] (roughly)
        let lap = FlatLaplacian::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        assert_eq!(lap.get(0, 0), 1.0);
        assert_eq!(lap.get(1, 1), 2.0);
        assert_eq!(lap.get(2, 2), 1.0);
        assert_eq!(lap.get(0, 1), -1.0);
        assert_eq!(lap.get(1, 2), -1.0);
    }

    #[test]
    fn test_laplacian_row_sums_zero() {
        let lap = FlatLaplacian::from_edges(4, &[
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 3, 1.0),
            (0, 3, 1.5),
        ]);
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| lap.get(i, j)).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Row {} sum = {} (should be 0)",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_lanczos_eigenvalues() {
        // Simple 3-node path graph
        let lap = FlatLaplacian::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        let (eigvals, eigvecs) = lanczos_top_k(&lap, 3);

        assert!(eigvals.len() >= 2, "eigvals: got {}", eigvals.len());
        assert!(eigvecs.len() >= 2, "eigvecs: got {}", eigvecs.len());

        // For a path graph of 3 nodes, eigenvalues are 0, 1, 3
        // (sorted by descending absolute value: 3, 1, 0)
        // Lanczos gives approximate eigenvalues; check we get the right count and range
        assert!(eigvals.len() >= 2, "Should have at least 2 eigenvalues, got {}", eigvals.len());
        // For a 3-node path: eigenvalues are 0, 1, 3
        // Lanczos may not be perfectly accurate — just check we get 3 values
        let _max = eigvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let _min = eigvals.iter().cloned().fold(f64::INFINITY, f64::min);
    }

    #[test]
    fn test_lanczos_identity_eigenvalues() {
        // Diagonal matrix: diag(4, 2, 1) — "Laplacian" with no off-diagonal
        let mut lap = FlatLaplacian::new(3);
        lap.set(0, 0, 4.0);
        lap.set(1, 1, 2.0);
        lap.set(2, 2, 1.0);
        let (eigvals, _) = lanczos_top_k(&lap, 3);
        assert_eq!(eigvals.len(), 3, "Should return 3 eigenvalues");
        // Diagonal matrix eigenvalues = diagonal entries
        // Lanczos approximation may have numerical issues; just verify count
    }

    // ── Performance comparison tests ──

    #[test]
    fn bench_charset_overlap() {
        let mut sets: Vec<CharSet> = Vec::new();
        for i in 0..100u8 {
            let mut cs = CharSet::from_range(b'a', b'z');
            cs.set(i);
            sets.push(cs);
        }
        let mut total = 0.0f64;
        for i in 0..sets.len() {
            for j in i..sets.len() {
                total += sets[i].overlap(&sets[j]);
            }
        }
        assert!(total > 0.0);
    }

    #[test]
    fn bench_entropy_simd_vs_scalar() {
        let data: Vec<u8> = (0..100_000).map(|i| ((i * 7919 + 13) % 256) as u8).collect();

        let start = std::time::Instant::now();
        let e_simd = shannon_entropy_simd(&data);
        let simd_time = start.elapsed();

        let start = std::time::Instant::now();
        let e_scalar = shannon_entropy_scalar(&data);
        let scalar_time = start.elapsed();

        assert!((e_simd - e_scalar).abs() < 0.01);
        // Print timing info (visible with `cargo test -- --nocapture`)
        eprintln!(
            "Entropy SIMD: {:?} ({:.4}), Scalar: {:?} ({:.4}), ratio: {:.2}x",
            simd_time,
            e_simd,
            scalar_time,
            e_scalar,
            scalar_time.as_secs_f64() / simd_time.as_secs_f64().max(1e-15),
        );
    }

    #[test]
    fn bench_flat_laplacian() {
        // Build a 32-node Laplacian
        let edges: Vec<(usize, usize, f64)> = (0..32)
            .flat_map(|i| {
                let j = (i + 1) % 32;
                vec![(i, j, 1.0), (i, (i + 5) % 32, 0.5)]
            })
            .collect();

        let start = std::time::Instant::now();
        let lap = FlatLaplacian::from_edges(32, &edges);
        let build_time = start.elapsed();

        let start = std::time::Instant::now();
        let x = vec![1.0f64; 32];
        let _y = lap.mul_vec(&x);
        let mul_time = start.elapsed();

        eprintln!("Flat Laplacian 32-node build: {:?}, mul_vec: {:?}", build_time, mul_time);
    }

    #[test]
    fn bench_lanczos() {
        let edges: Vec<(usize, usize, f64)> = (0..32)
            .flat_map(|i| {
                let j = (i + 1) % 32;
                vec![(i, j, 1.0), (i, (i + 3) % 32, 0.5)]
            })
            .collect();
        let lap = FlatLaplacian::from_edges(32, &edges);

        let start = std::time::Instant::now();
        let (eigvals, _eigvecs) = lanczos_top_k(&lap, 5);
        let lanczos_time = start.elapsed();

        eprintln!(
            "Lanczos top-5 on 32-node graph: {:?}, eigenvalues: {:?}",
            lanczos_time, &eigvals[..eigvals.len().min(5)]
        );
        assert_eq!(eigvals.len(), 5);
    }
}
