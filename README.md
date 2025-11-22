# ZDTP Chess

**Multi-Dimensional Decision Intelligence Using Applied Pathological Mathematics**

> *"Better math, less suffering"* - Chavez AI Labs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Research](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17574868-blue)](https://zenodo.org/records/17574868)

---

## What Makes This Different

Traditional chess engines evaluate positions with a **single number**. ZDTP evaluates positions across **three dimensional layers simultaneously**:

- **16D Tactical Layer** - Immediate threats, hanging pieces, forcing sequences
- **32D Positional Layer** - Piece coordination, pawn structure, gateway patterns
- **64D Strategic Layer** - Long-term planning, endgame evaluation, strategic depth

Each position is analyzed through six mathematical **gateways** (King, Queen, Knight, Bishop, Rook, Pawn) derived from zero divisor patterns in higher-dimensional algebras. When multiple gateways converge on the same evaluation, you've found something objectively strong across independent mathematical frameworks.

**This is infrastructure for AI systems, not a chess product.** Chess is the proof of concept for multi-dimensional decision intelligence.

---

## Features

### Multi-Dimensional Analysis
- **16D Tactical Layer** - Immediate threat detection, material balance, and tactical opportunities
- **32D Positional Layer** - Piece coordination, pawn structure, and center control through zero divisor gateway patterns
- **64D Strategic Layer** - Long-term planning, endgame evaluation, and strategic positioning

### Gateway Convergence Detection
Six independent mathematical "gateways" (King, Queen, Knight, Bishop, Rook, Pawn) analyze each position. When multiple gateways converge on the same evaluation and recommendation, the system identifies framework-independent optimal moves with mathematical certainty.

### Blunder Prevention
Industry-standard Static Exchange Evaluation (SEE) integrated with dimensional analysis to catch hanging pieces and material-losing moves before they happen.

### Educational Interface
Clear visualization of dimensional scores, gateway patterns, and convergence indicators help players understand not just *what* move to make, but *why* it's optimal across multiple mathematical frameworks.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Model Context Protocol (MCP) support

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/pchavez2029/zdtp-chess.git
cd zdtp-chess
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure as MCP server:**
Add to your MCP client configuration (e.g., Claude Desktop):
```json
{
  "mcpServers": {
    "zdtp-chess": {
      "command": "python",
      "args": ["-m", "zdtp_chess_mcp.zdtp_chess_server"]
    }
  }
}
```

### Requirements
- `python-chess>=1.999` - Chess move generation and board representation
- `anthropic-mcp` - Model Context Protocol server
- Additional dependencies listed in `requirements.txt`

---

## The Math Section

*"I was told there would be no math."* - Anonymous Student

Here's the minimal math you need to understand why this works:

### Zero Divisors: The "Impossible" Elements

In normal arithmetic, if **A × B = 0**, then either **A = 0** or **B = 0** (or both). This is the zero-product property you learned in algebra.

**But in higher-dimensional algebras, this breaks down beautifully.**

In 16D sedenions, 32D pathions, and 64D chingons, you can find non-zero elements **P** and **Q** where:

```
P × Q = 0

even though P ≠ 0 and Q ≠ 0
```

These are called **zero divisors**, and mathematicians traditionally dismissed them as "pathological"—algebraic structures that make systems "unusable."

### Why Zero Divisors Are Actually Useful

**Key insight:** Zero divisors encode information about **dimensional collapse** and **information loss** in algebraic systems. In decision-making, this maps to:

- **Tactical collapse** (16D) - Positions where forcing sequences eliminate options
- **Positional transformation** (32D) - How piece coordination changes across moves
- **Strategic encoding** (64D) - Long-term plan evaluation through dimensional reduction

When you analyze a chess position in 16D and transmit it to 32D via ZDTP, zero divisor patterns preserve the **decision-relevant structure** while collapsing irrelevant noise. This is lossless information movement between dimensional spaces.

### Example: The Canonical Six

We discovered six fundamental zero divisor patterns that appear consistently across 16D/32D/64D spaces:

```
(e₁ ± e₁₀) × (e₄ ∓ e₁₅) = 0   ← King Gateway pattern
(e₂ ± e₁₁) × (e₅ ∓ e₁₄) = 0   ← Queen Gateway pattern
(e₃ ± e₈)  × (e₆ ∓ e₁₃) = 0   ← Knight Gateway pattern
(e₁ ± e₁₄) × (e₄ ∓ e₁₁) = 0   ← Bishop Gateway pattern
(e₂ ± e₁₅) × (e₅ ∓ e₁₀) = 0   ← Rook Gateway pattern
(e₃ ± e₁₂) × (e₆ ∓ e₉) = 0    ← Pawn Gateway pattern
```

Each pattern provides a different "lens" for evaluating positions. When multiple patterns converge on the same evaluation, you've found framework-independent optimality.

**That's the math. The rest is engineering.**

---

## How It Works

### Zero Divisor Transmission Protocol (ZDTP)

ZDTP enables **lossless data movement** between higher-dimensional mathematical spaces:

1. **Encoding (16D)** - Chess position → 16D sedenion representation
   - Material balance, piece mobility, tactical threats
   - Encoded using basis elements e₀ through e₁₅

2. **Transmission (32D)** - 16D data → 32D pathion space via gateway patterns
   - Six independent gateways process position simultaneously
   - Zero divisor patterns preserve decision-relevant structure
   - Positional factors (coordination, structure) emerge in 32D

3. **Strategic Analysis (64D)** - 32D data → 64D chingon space
   - Long-term planning, endgame evaluation
   - Strategic depth analysis through dimensional expansion

4. **Convergence Detection** - Compare all gateway outputs
   - If multiple gateways agree (within threshold), move is framework-independent optimal
   - Disagreement indicates tactical complexity requiring deeper analysis

**Why this matters:** Traditional dimensional reduction (PCA, t-SNE, etc.) **loses information**. ZDTP uses zero divisor patterns to preserve decision-relevant structure across dimensional transformations. Information moves losslessly between 16D, 32D, and 64D spaces.

### Game Flow

1. **You play White** against a computer opponent (Black)
2. **After each move**, ZDTP analyzes the position through an adaptive gateway
3. **Dimensional scores** show tactical (16D), positional (32D), and strategic (64D) evaluation
4. **Positive scores** = advantage for White (you)
5. **Gateway convergence** alerts you when multiple frameworks independently agree

### MCP Tools

ZDTP Chess provides the following tools through the Model Context Protocol:

- **chess_new_game** - Start a new game (you play White or Black)
- **chess_make_move** - Execute a move (requires explicit user confirmation)
- **chess_analyze_move** - Preview move consequences without executing (what-if analysis)
- **chess_get_board** - Display current position and game state
- **chess_get_dimensional_analysis** - Detailed breakdown of current position
- **chess_check_gateway_convergence** - Check multiple gateways for framework-independent optimization

**Important:** Moves require explicit user permission. Use `chess_analyze_move` to explore options safely before committing with `chess_make_move`.

### Example Analysis Output

```
╔══════════════════════════════════════════════════════════════╗
║  BLACK RESPONDS: Nf6                                         ║
╚══════════════════════════════════════════════════════════════╝

Analysis Gateway: Knight (discontinuous gateway - non-linear patterns)
Position Evaluation: +0.94 (White's perspective)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MULTI-DIMENSIONAL POSITION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 16D TACTICAL LAYER (Immediate Threats & Material)
   Score: +0.87 (White's perspective)
   • Material balanced
   • White has better mobility (3 more legal moves)
   • No immediate threats detected

🏗️ 32D POSITIONAL LAYER (Structure & Coordination)
   Score: +2.91 (White's perspective)
   • Gateway Pattern: Discontinuous Gateway (Knight)
   • Strong center control for White
   • Well-coordinated piece development

🌟 64D STRATEGIC LAYER (Long-term Planning)
   Score: +1.23 (White's perspective)
   • White's central pawns provide middlegame flexibility
   • Favorable pawn structure approaching
   • Long-term endgame potential

CONSENSUS EVALUATION: +0.94 (slight advantage: WHITE)

💡 SUGGESTED MOVES: Nf3 (develop), c4 (Queen's Gambit), Bf4 (bishop out)
```

---

## Architecture

ZDTP Chess combines **standard chess engine components** with **novel multi-dimensional analysis**:

### Standard Components
- Legal move generation (python-chess library)
- Material evaluation
- Blunder detection using Static Exchange Evaluation (SEE)
- Basic tactical analysis

These components ensure ZDTP Chess meets baseline requirements for chess programming and can be validated against traditional engines.

### ZDTP Innovation
- **16D Tactical Analysis** - Sedenion-based immediate threat detection
- **32D Positional Analysis** - Pathion gateway patterns for structural evaluation
- **64D Strategic Analysis** - Chingon-based long-term planning
- **Gateway Convergence** - Framework-independent optimality detection

The ZDTP layers run after standard blunder detection, adding strategic insight beyond traditional evaluation functions.

---

## Mathematical Foundation

ZDTP Chess is built on research into Cayley-Dickson algebras and zero divisor patterns:

- **Sedenions (16D)** - Tactical complexity and immediate threats
- **Pathions (32D)** - Positional structures through six gateway patterns
- **Chingons (64D)** - Strategic planning and endgame evaluation

**Research Publication:** [Framework-Independent Zero Divisor Patterns in Higher-Dimensional Cayley-Dickson Algebras: Discovery and Verification of The Canonical Six](https://zenodo.org/records/17574868) - Zenodo DOI: 10.5281/zenodo.17574868

### The Six Gateways

Each gateway represents a different zero divisor pattern from 32D pathion algebra:

1. **King Gateway** - Master gateway, holistic evaluation
2. **Queen Gateway** - Multi-modal gateway, tactical complexity
3. **Knight Gateway** - Discontinuous gateway, non-linear patterns
4. **Bishop Gateway** - Diagonal gateway, long-range planning
5. **Rook Gateway** - Orthogonal gateway, file control
6. **Pawn Gateway** - Incremental gateway, structural analysis

When multiple gateways independently arrive at the same evaluation and recommendation, the move is considered framework-independent optimal.

---

## Applications

### Chess (Current Proof of Concept)
- Multi-dimensional position evaluation across tactical/positional/strategic layers
- Framework-independent move quality assessment through gateway convergence
- Real-time blunder detection with dimensional analysis
- Educational tool for understanding multi-perspective decision-making

### AI Infrastructure (Platform Vision)
- **Decision Intelligence** - Multi-framework validation for complex AI decisions
- **Quantitative Finance** - Portfolio analysis through dimensional risk assessment (CAILculator in development)
- **Medical Diagnostics** - Multi-framework symptom evaluation with convergence validation
- **Strategic Planning** - Business decisions analyzed across multiple independent frameworks
- **AI Safety** - Catching edge cases that single-model systems miss

### For Developers & Researchers
- **MCP Server Template** - Production-ready Model Context Protocol implementation
- **Multi-Framework Analysis** - Reference architecture for combining independent mathematical approaches
- **Applied Pathological Mathematics** - Demonstration that "unusable" mathematical structures have practical value
- **Framework-Independent Optimization** - Study convergence patterns across different algebraic systems

---

## Roadmap

### Phase 1: Chess (Complete - v1.0)
- ✅ Core dimensional analysis engine (16D/32D/64D)
- ✅ Six gateway patterns implemented
- ✅ ZDTP protocol for lossless dimensional transmission
- ✅ Gateway convergence detection
- ✅ Blunder detection with SEE integration
- ✅ MCP server with user confirmation safeguards

### Phase 2: Financial Infrastructure (In Development)
- **CAILculator** - Quantitative finance application
- Portfolio risk assessment across dimensional frameworks
- Multi-asset correlation analysis through gateway patterns
- Framework-independent optimization for trading strategies

### Phase 3: Platform Expansion
- Medical diagnostic decision support systems
- Strategic business planning tools
- AI safety validation frameworks
- Natural language processing with dimensional embeddings
- Extended dimensional analysis (128D, 256D layers)

### Chess Enhancements (Ongoing)
- Gateway selection strategy optimization
- Position-type adaptive gateway weighting
- Performance optimization (parallel gateway evaluation)
- PGN export with dimensional annotations

---

## About Chavez AI Labs

**Mission:** "Better math, less suffering"

Chavez AI Labs applies pathological mathematics - mathematical structures traditionally dismissed as unusable - to create practical AI systems and decision-making tools.

**Founder:** Paul Chavez
- 30+ years journalism experience (Associated Press, LA Times)
- UCLA alumnus (Political Science, 1989)
- Six years of research into higher-dimensional algebras
- Published research with CERN DOI on zero divisor patterns

**Products:**
- **CAILculator** - MCP server for high-dimensional mathematical analysis
- **ZDTP Chess** - Proof of concept for applied pathological mathematics
- Additional applications in quantitative finance, data analysis, and AI infrastructure

---

## Contributing

ZDTP Chess is currently in private beta. For collaboration inquiries, research partnerships, or commercial licensing:

- Contact: iknowpi@gmail.com
- Company: Chavez AI Labs (California-licensed AI company)
- Research: See published papers on framework-independent zero divisor patterns

---

## License

Apache License 2.0 with patent protection. See [LICENSE.txt](LICENSE.txt) for details.

---

## Acknowledgments

- **Python-chess library** - Foundation for chess logic and board representation
- **Anthropic MCP** - Model Context Protocol implementation
- **CERN** - Digital Object Identifier (DOI) for research publication
- **Chess community** - Feedback and validation during development

---

## Citation

If you use ZDTP Chess in academic research, please cite:

```
Chavez, P. (2025). ZDTP Chess: Multi-Dimensional Analysis Through Zero Divisor Patterns.
Chavez AI Labs. https://github.com/pchavez2029/zdtp-chess
```

---

**Built with pathological mathematics. Tested in battle. Ready for production.**

*Chavez AI Labs - Applied Pathological Mathematics*
