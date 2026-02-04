# ZDTP Chess

**Multi-Dimensional Decision Intelligence Using Applied Pathological Mathematics**

> *"Better math, less suffering"* - Chavez AI Labs

[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Research](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17402495-blue)](https://zenodo.org/records/17402495)

---

## What Makes This Different

Traditional chess engines evaluate positions with a **single number**. Zero Divisor Transmission Protocol (ZDTP) evaluates positions across **three dimensional layers simultaneously**:

- **16D Tactical Layer** - Immediate threats, hanging pieces, forcing sequences
- **32D Positional Layer** - Piece coordination, pawn structure, gateway patterns
- **64D Strategic Layer** - Long-term planning, endgame evaluation, strategic depth

Each position is analyzed through six mathematical **gateways** (King, Queen, Knight, Bishop, Rook, Pawn) derived from zero divisor patterns in higher-dimensional algebras. When multiple gateways converge on the same evaluation, you've found something objectively strong across independent mathematical frameworks.

**This is infrastructure for AI systems, not a chess product.** Chess is the proof of concept for multi-dimensional decision intelligence.

---

## Features

### Gateway Convergence Detection
Six independent mathematical "gateways" (King, Queen, Knight, Bishop, Rook, Pawn) analyze each position. When multiple gateways converge on the same evaluation and recommendation, the system identifies framework-independent optimal moves with mathematical certainty.

### Blunder Prevention
Industry-standard Static Exchange Evaluation (SEE) integrated with dimensional analysis to catch hanging pieces and catastrophic moves before they happen.

### Educational Interface
Clear visualization of dimensional scores, gateway patterns, and convergence indicators help players understand not just *what* move to make, but *why* it's optimal across multiple mathematical frameworks.

---

## Prerequisites

### Required Software

- **Python 3.10 - 3.13** - Python 3.14+ currently has compatibility issues
  - Download: https://www.python.org/downloads/
  - **Windows users:** During installation, check "Add Python to PATH"
- **Claude Desktop** with MCP support
  - Download: https://claude.ai/download

### Optional (Recommended)

- **Git** for cloning repository
  - Download: https://git-scm.com/downloads
  - **Alternative:** Download ZIP file directly from GitHub (see Installation)

### Verify Your Installation
```bash
# Check Python version (should show 3.10-3.13)
python --version

# Check pip is available
python -m pip --version
```

---

## Installation

### Quick Start (All Platforms)

1. Download ZDTP Chess
2. Install Python dependencies
3. Configure Claude Desktop
4. Restart Claude Desktop

Detailed platform-specific instructions below.

---

### Installation on Windows

#### Step 1: Download ZDTP Chess

**Option A: Using Git**
```powershell
git clone https://github.com/pchavez2029/zdtp-chess.git
cd zdtp-chess
```

**Option B: Download ZIP (No Git Required)**
1. Visit https://github.com/pchavez2029/zdtp-chess
2. Click the green **"Code"** button
3. Select **"Download ZIP"**
4. Extract to a permanent location (e.g., `C:\Users\YourName\Documents\zdtp-chess`)
5. Open PowerShell in the extracted folder:
   - Navigate to the folder in File Explorer
   - Hold **Shift + Right-click** in the folder
   - Select **"Open PowerShell window here"**

#### Step 2: Fix PowerShell Execution Policy (One-time Setup)

If you encounter "cannot be loaded because running scripts is disabled" errors:
```powershell
# Run PowerShell as Administrator
# Right-click PowerShell in Start Menu -> "Run as Administrator"

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Type 'Y' and press Enter when prompted
```

This is a one-time Windows security setting required for Python packages.

#### Step 3: Install Dependencies
```powershell
# Important: Use 'python -m pip' on Windows (not just 'pip')
python -m pip install -r requirements.txt
```

You may see warnings about scripts not on PATH - these are non-critical and can be ignored.

#### Step 4: Configure Claude Desktop MCP Server

1. Open **Claude Desktop**
2. Navigate to **Settings -> Developer -> Edit Config**
   - This opens `claude_desktop_config.json` in your default text editor
3. Add the ZDTP Chess configuration:
```json
{
  "mcpServers": {
    "zdtp-chess": {
      "command": "python",
      "args": ["-m", "zdtp_chess_mcp"],
      "cwd": "C:\\Users\\YourName\\Documents\\zdtp-chess",
      "env": {
        "PYTHONPATH": "C:\\Users\\YourName\\Documents\\zdtp-chess"
      }
    }
  }
}
```

**Critical Configuration Notes:**
- Replace `C:\\Users\\YourName\\Documents\\zdtp-chess` with your **actual installation path**
- Use **double backslashes** (`\\`) in Windows paths for JSON format
- Both `cwd` and `PYTHONPATH` must point to the **same directory**
- The args must be `["-m", "zdtp_chess_mcp"]` NOT `["-m", "zdtp_chess_mcp.zdtp_chess_server"]`
- If you have other MCP servers, add `zdtp-chess` inside the existing `mcpServers` object

4. **Save** the config file
5. **Completely close and restart Claude Desktop**
   - Quit the application entirely, don't just close the window
   - On Windows: Right-click system tray icon -> "Quit"

#### Step 5: Verify Installation

1. Open Claude Desktop
2. Go to **Settings -> Developer**
3. Check MCP Servers list:
   - `zdtp-chess` should show as **"connected"**
   - If it shows "failed", see Troubleshooting section below

---

### Installation on macOS/Linux

```bash
# Clone repository
git clone https://github.com/pchavez2029/zdtp-chess.git
cd zdtp-chess

# Install dependencies
pip install -r requirements.txt
```

Configure Claude Desktop by editing:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

Add to configuration:
```json
{
  "mcpServers": {
    "zdtp-chess": {
      "command": "python",
      "args": ["-m", "zdtp_chess_mcp"],
      "cwd": "/absolute/path/to/zdtp-chess",
      "env": {
        "PYTHONPATH": "/absolute/path/to/zdtp-chess"
      }
    }
  }
}
```

Replace `/absolute/path/to/zdtp-chess` with your actual installation path. Restart Claude Desktop completely.

### Requirements
- `python-chess>=1.999` - Chess move generation and board representation
- `mcp>=0.9.0` - Model Context Protocol server
- `hypercomplex>=0.3.4` - Hypercomplex number systems (Sedenions, Pathions, Chingons)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'zdtp_chess_mcp'"

**Causes & Solutions:**

1. **Missing `cwd` or `PYTHONPATH` in config**
   - Verify your config has BOTH `cwd` AND `PYTHONPATH` set to the installation directory

2. **Incorrect path format (Windows)**
   - Use double backslashes (`\\`) in JSON paths
   - Wrong: `"C:\Users\YourName\Documents\zdtp-chess"`
   - Correct: `"C:\\Users\\YourName\\Documents\\zdtp-chess"`

3. **Claude Desktop not restarted**
   - Completely quit and restart Claude Desktop (not just close window)

### "Server disconnected" Error

1. **Wrong Python version**
   - Check: `python --version` (must be 3.10-3.13, NOT 3.14+)
   - Solution: Install Python 3.13 from https://www.python.org/downloads/

2. **Dependencies not installed**
   - Run: `python -m pip install -r requirements.txt`

3. **Wrong command in config**
   - Use: `"args": ["-m", "zdtp_chess_mcp"]`
   - NOT: `"args": ["-m", "zdtp_chess_mcp.zdtp_chess_server"]`

4. **Python not in PATH**
   - Use full Python path in config:
```json
"command": "C:\\Users\\YourName\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"
```
   - Find your Python path: `where.exe python` (Windows) or `which python` (macOS/Linux)

### Multiple Python Installations

If packages install but you still get ModuleNotFoundError:

```powershell
# Windows - see all Python installations
where.exe python

# Check which Python pip uses
python -m pip --version
```

Use the full path to your Python 3.13 installation in the config:
```json
"command": "C:\\Users\\YourName\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"
```

### PowerShell "running scripts is disabled"

```powershell
# Open PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Type 'Y' and press Enter
```

### Can't Find Config File

**Easy method:** Settings -> Developer -> Edit Config (works on all operating systems)

**Manual paths:**
- Windows: `C:\Users\YourName\AppData\Roaming\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Note: On Windows, the `AppData` folder is hidden by default. Use Settings -> Developer -> Edit Config instead.

### Python 3.14 Compatibility

If you see:
```
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_backend'
```

Python 3.14 has breaking changes in setuptools. Use Python 3.13 or earlier:
1. Download Python 3.13 from https://www.python.org/downloads/
2. Install with "Add to PATH" checked
3. Reinstall dependencies: `python -m pip install -r requirements.txt`

### Getting Help

If you encounter issues not covered here:

1. **Check the logs:** Claude Desktop -> Settings -> Developer -> View logs
2. **Verify installation:**
```powershell
python --version
python -m pip list | findstr "chess mcp hypercomplex"
```
3. **Create a GitHub Issue:** https://github.com/pchavez2029/zdtp-chess/issues
   - Include your OS, Python version, and error messages from Claude Desktop logs

---

## The Math Section

*"I was told there would be no math."* - Anonymous Student

Sorry, but there's math. Here's the minimal math you need to understand how this works:

### Zero Divisors

In normal arithmetic, if **A × B = 0**, then either **A = 0** or **B = 0** (or both). This is the zero-product property you learned in algebra.

**In higher-dimensional algebras, however, this breaks down with amazing annihilation.**

Starting in 16D sedenions and continuing upward to 32D pathions, and 64D chingons, you can find non-zero elements **P** and **Q** where:

```
P × Q = 0

even though P ≠ 0 and Q ≠ 0
```

These are called **zero divisors**, and mathematicians traditionally have dismissed them as "pathological", wrongly demeaning them as algebraic structures that make systems "unusable."

### Why Zero Divisors Are Actually Useful

**Key insight:** Zero divisors can encode information about **dimensional collapse** and **information loss** in algebraic systems. In chess decision-making, this maps to:

- **Tactical collapse** (16D) - Positions where forcing sequences eliminate options
- **Positional transformation** (32D) - How piece coordination changes across moves
- **Strategic encoding** (64D) - Long-term plan evaluation through dimensional reduction

When you analyze a chess position in 16D and transmit it to 32D and 64D via ZDTP, zero divisor patterns preserve the **decision-relevant structure** with lossless information movement between dimensional spaces.

### Example: The Canonical Six

We discovered six fundamental zero divisor patterns that appear consistently across 16D/32D/64D spaces. (Reference: https://zenodo.org/records/17402495)

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

**Why this matters:** Traditional dimensional reduction **loses information**. ZDTP uses zero divisor patterns to preserve decision-relevant structure across dimensional transformations. Information moves losslessly between 16D, 32D, and 64D spaces.

### Game Flow

1. **You play White** against a computer opponent (Black)
2. **After each move**, ZDTP analyzes the position through an adaptive gateway
3. **Dimensional scores** show tactical (16D), positional (32D), and strategic (64D) evaluation
4. **Positive scores** = advantage for White (you)
5. **Gateway convergence** alerts you when multiple frameworks independently agree

### MCP Tools

ZDTP Chess provides the following tools through the Model Context Protocol:

- **chess_new_game** - Start a new game
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

**Research Publication:** [Framework-Independent Zero Divisor Patterns in Higher-Dimensional Cayley-Dickson Algebras: Discovery and Verification of The Canonical Six](https://zenodo.org/records/17402495) - Zenodo DOI: 10.5281/zenodo.17402495

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
- Zero Divisor Transmission Protocol moves position information with no data loss from 16D to 32D to 64D
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
- **CAILculator** - Quantitative finance application via MCP server
- Portfolio risk assessment across dimensional frameworks
- Multi-asset correlation analysis through gateway patterns
- Framework-independent optimization for trading strategies

### Phase 3: Platform Expansion
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
- Published research with CERN DOI on zero divisor patterns

**Products:**
- **CAILculator** - MCP server for high-dimensional mathematical analysis
- **ZDTP Chess** - Proof of concept for applied pathological mathematics
- Additional applications in quantitative finance, data analysis, and AI infrastructure

---

## Contributing

For collaboration inquiries, research partnerships, or commercial licensing:

- Contact: iknowpi@gmail.com
- Company: Chavez AI Labs (California-licensed AI company)
- Research: See published paper on framework-independent zero divisor patterns

---

## License

Apache License 2.0 with patent protection.

---

## Acknowledgments

- **Python-chess library** - Foundation for chess logic and board representation
- **Anthropic MCP** - Model Context Protocol implementation
- **CERN** - Digital Object Identifier (DOI) for research publication
- **Chess community** - Inspiration and education during development

---

## Citation

If you use ZDTP Chess in academic research, please cite:

```
Chavez, P. (2025). ZDTP Chess: Multi-Dimensional Analysis Through Zero Divisor Patterns.
Chavez AI Labs. https://github.com/pchavez2029/zdtp-chess
```

---

*Chavez AI Labs - Applied Pathological Mathematics*
