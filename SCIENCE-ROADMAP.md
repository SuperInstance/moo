# MOO SCIENCE ROADMAP
## Building a Constraint-Native, Spectral-Aware Tokenizer

### The Science We're Bringing
1. **Tension-Graph Laplacian**: 112× conservation signal in eigenvectors
2. **Symplectic Phase Space**: 25× lower variance in common-practice (p = 10⁻¹⁰)
3. **Persistent Homology**: 96% classification accuracy
4. **Eigenbasis Hypothesis**: Conserved quantities visible only in structural eigenbasis

### Architecture Layers

#### L0: Core moo.js (upstream)
- Regex compilation, sticky matching, states, keywords
- ~600 lines, zero dependencies, battle-tested

#### L1: moo-spectral.js — Spectral Optimization
- Build state transition graph → Laplacian → Fiedler ordering
- Rule tension computation (character set overlap)
- Adaptive compilation from profiling data
- Conservation-aware error recovery
- Backward-compatible API

#### L2: moo-constraint/ — Constraint-Native DSL
- Specify structural properties, not just regex rules
- Compile constraints into optimized lexer states
- Runtime conservation tracking
- Spectral lexer generation from grammar descriptions

#### L3: topology-stream/ — Real-time Topological Analysis
- Streaming persistent homology on token graphs
- Wasserstein distance anomaly detection
- Spectral fingerprint language detection
- Conservation drop detection

#### L4: gpu-lexer/ — GPU-Accelerated Matching
- CUDA DFA matching kernel (RTX 4050)
- Batch file processing with tension-graph on GPU
- Node.js native bindings

#### L5: moo-rust/ — Native Performance Core
- Rust port with spectral optimization
- WASM target for browser deployment
- Constraint trait system
- 10-100× throughput improvement

### Spawn Queue (when slots available)
1. moo-spectral-lexer (Claude) — spectral optimization of moo.js
2. moo-constraint-dsl (Claude) — constraint-native DSL layer
3. moo-gpu-regex (Claude) — CUDA regex kernel
4. moo-topology-stream (Claude) — real-time topological analysis
5. moo-rust-core (Claude) — Rust port + WASM
