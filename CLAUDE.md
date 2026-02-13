# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZDTP Chess is an MCP server that evaluates chess positions across three dimensional layers (16D/32D/64D) using zero divisor patterns from higher-dimensional algebras. Chess is the proof of concept; the underlying architecture is multi-dimensional decision intelligence infrastructure.

## Commands

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Install dev dependencies (pytest, pytest-asyncio)
python -m pip install -e ".[dev]"

# Run MCP server standalone (stdio mode)
python -m zdtp_chess_mcp

# Run all tests (run individually due to Python 3.13 stdout encoding conflicts)
pytest test_candidate_suggester.py -v
python test_session_01_verification.py
python test_confirmation_validation.py

# Run a single test
pytest test_candidate_suggester.py::test_forcing_checks -v
```

Python 3.10-3.13 required (3.14+ has compatibility issues).

## Architecture

### Evaluation Pipeline

```
Position → 16D Sedenion Encoding → Gateway Selection → 32D Pathion → 64D Chingon → Consensus Score
```

1. **Encoding** (`dimensional_encoder.py`): Chess position → 16-coefficient Sedenion. Each dimension maps to a chess concept (e₀=material, e₁=pawn structure, e₂/e₃=king safety, e₄=center control, etc.)
2. **Transmission** (`dimensional_portal.py`): Sedenion × gateway zero divisor pattern → Pathion (32D) → Chingon (64D). Uses `cascade_16_32` and `cascade_32_64` methods.
3. **Evaluation** (`multidimensional_evaluator.py`): Combines 16D tactical, 32D positional, and 64D strategic scores into consensus. Includes Master Dampener for fortress draw detection.
4. **Move selection** (`zdtp_engine.py`): SEE-based blunder prevention + dimensional analysis for move ranking.

### The Gateway System

Six zero divisor patterns (the "Canonical Six") map to chess piece types. Each gateway evaluates a position from an independent mathematical framework. When multiple gateways converge, the move is framework-independent optimal.

- King (Pattern 18), Queen (59), Knight (84), Bishop (102), Rook (104), Pawn (124)
- Pattern definitions: `data/canonical_six_patterns.json`
- Gateway mapping: `gateway_patterns.py`

### MCP Server Structure

- **Server** (`zdtp_chess_server.py`): 9 MCP tools, game state in module-level `games: Dict[str, chess.Board]`
- **Tool registration**: Add to `handle_list_tools()` list, dispatch in `handle_call_tool()`, handler as async function returning `list[types.TextContent]`
- **Entry point**: `__main__.py` runs stdio server via `mcp.server.stdio`

### Key Modules

| Module | Role |
|---|---|
| `zdtp_engine.py` | SEE (line 43), piece values (line 61), blunder filter, adaptive gateway selection |
| `candidate_suggester.py` | Categorizes all legal moves: forcing → defensive → developing → quiet |
| `multidimensional_evaluator.py` | 16D/32D/64D scoring, Master Dampener (fortress detection) |
| `dimensional_portal.py` | 16D→32D→64D transmission via gateway conjugate pairs |
| `dimensional_encoder.py` | Position → Sedenion (16 coefficients) |
| `strategic_analyzer.py` | 64D Chingon analysis; Lean 4 theorem groundings in dims 44-55 |
| `stressor_positions.py` | Curated stress-test positions by category |
| `path_verifier.py` | Legal move validation encoded in dims 9-10 |

## Implementation Patterns

- **Game state**: Module-level `games` dict keyed by `game_id` string. Eval histories tracked in `eval_histories` dict (max 10 entries).
- **SEE piece values**: Knight=3.2, Bishop=3.3 (non-standard, for SEE consistency with dimensional analysis). Do not change without updating both `zdtp_engine.py` and tests.
- **Move confirmation**: `chess_make_move` requires `user_said` parameter containing confirmation keywords ('play', 'make', 'execute', etc.) validated via whole-word regex.
- **Test files**: Live at project root, use `sys.path.insert(0, 'zdtp_chess_mcp')` to import engine modules.
- **Defensive moves**: Only appear in candidate suggestions when pieces are actually attacked. Sort order: escapes → pawn_defends → piece_defends → counterattack, with secondary sort by piece value descending. Per-subcategory truncation limits: escape(5), pawn_defends(3), piece_defends(3), counterattack(2).
- **Developing subcategories**: castling → pawn_center → pawn_support_center (c/f files) → key_square (d4/e4/d5/e5) → minor_development → rook_open_file → rook_semi_open → centralization.
- **Development gate**: `_is_developing_move()` prevents developing moves from being reclassified as counterattacks. WARNING: duplicates developing checks in `categorize_move()` — must stay in sync.
- **Board perspective**: Evaluation scores are from White's perspective. When Black moves, scores are negated.

## Gotchas

- `chess.Board.gives_check(move)` is efficient (no board copy needed) — prefer it over push/pop patterns for check detection.
- Promotion moves that give check are classified as 'check' before 'promotion' in priority-based categorization.
- Test FENs: ensure kings don't block pawn promotion paths; ensure bishop diagonals are open (d-pawn must be moved for Bc1 tests).
- Master Dampener requires ≥4 entries in eval history for temporal stasis detection.
- Windows: test files set `sys.stdout` encoding to UTF-8 for chess unicode output. This causes pytest to crash when running multiple test files together — run them individually.
- **Test counts**: `test_candidate_suggester.py` has 19 tests (12 original + 7 Session 3 bug fix tests).
