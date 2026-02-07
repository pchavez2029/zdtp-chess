"""
ZDTP Chess MCP Server

MCP server that integrates ZDTP chess engine with Claude Desktop.

Provides tools for:
- Starting new games
- Making moves with ZDTP reasoning
- Getting dimensional analysis
- Viewing multi-dimensional evaluations
"""

import asyncio
import chess
from typing import Dict, Optional
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from .zdtp_engine import ZDTPEngine, get_best_move, _emergency_see_safety_check
from .dimensional_encoder import encode_position
from .dimensional_portal import full_cascade
from .multidimensional_evaluator import evaluate_position, FortressDampenerResult
from .opponent_response_analyzer import analyze_opponent_responses
from .zdtp_showcase import format_zdtp_showcase
from .stressor_positions import STRESSOR_LIBRARY, get_stressor_category, list_stressor_positions


# Game storage
games: Dict[str, chess.Board] = {}
eval_histories: Dict[str, list] = {}  # game_id -> list of consensus scores (Master Dampener)
game_counter = 0

# Master Dampener: max history entries to retain per game
MAX_EVAL_HISTORY = 10

# Engine instance
engine = ZDTPEngine(time_limit_ms=5000, max_candidates=15, gateway_strategy='adaptive')

# Create MCP server
server = Server("zdtp-chess")


def show_game_intro() -> str:
    """
    Display game introduction screen with instructions.

    Returns:
        Formatted intro screen text with game instructions
    """
    return """
╔══════════════════════════════════════════════════════════════╗
║      ZDTP CHESS - Applied Pathological Mathematics           ║
╚══════════════════════════════════════════════════════════════╝

YOU: Human player
OPPONENT: python-chess engine (consistent strength)

WHAT ZDTP PROVIDES:
ZDTP is an ANALYSIS TOOL that evaluates your moves using higher-
dimensional mathematics. It does NOT control the opponent's moves.

  • 16D Tactical Layer - immediate threats, material balance
  • 32D Positional Layer - piece coordination, pawn structure
  • 64D Strategic Layer - long-term planning, endgame evaluation
  • Gateway System - 6 mathematical perspectives (King/Queen/Knight/etc.)

HOW IT WORKS:
  • You make your move
  • Opponent (python-chess) responds automatically
  • ZDTP analyzes the resulting position using ONE gateway (adaptive)
  • Positive scores = advantage for you
  • Negative scores = advantage for opponent
  • Want deeper analysis? Use chess_check_gateway_convergence (multiple gateways)

GOAL: Learn chess through multi-dimensional mathematical analysis

RECOMMENDED WORKFLOW:
  1. chess_analyze_move - Explore moves safely (no execution)
  2. Compare options and ZDTP scores
  3. User decides which move to play
  4. chess_make_move - Execute the chosen move (requires explicit user command)

COMMANDS:
  chess_analyze_move     - Preview a move WITHOUT executing (what-if analysis)
  chess_make_move        - Execute your move (only when explicitly commanded!)
  chess_get_dimensional_analysis - Detailed position breakdown
  chess_check_gateway_convergence - Check multiple gateways
  chess_get_board        - Show current position

OPTIONAL PARAMETERS:
  verbose=true           - Show detailed tactical information
  show_dimensional_analysis=false - Hide analysis (faster play)
  threshold=0.3          - Convergence threshold (for convergence check)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to begin? Make your first move with chess_make_move!
"""


def detect_gateway_convergence(board: chess.Board, threshold: float = 0.1, game_id: str = None) -> dict:
    """
    Detect gateway convergence by evaluating position with multiple gateways.

    SESSION 1 UPGRADES (Gemini Peer Review):
    - Dimensional divergence detection (the "h3 Spike" pattern)
    - Dead Gateway weighting (Unified Field Theory / Option B)
    - Per-dimensional layer scores for divergence analysis

    SESSION 0.1 UPGRADE: Master Dampener (Fortress Draw Detection)
    - Passes eval_history to evaluator for temporal stasis detection
    - Records consensus scores for future temporal checks
    - Returns dampener diagnostics in result dict

    Args:
        board: Current board position
        threshold: Score difference threshold for convergence (default 0.1)
        game_id: Game ID for eval history lookup (Master Dampener)

    Returns:
        dict with convergence data, dimensional divergence, dead gateway info,
        and Master Dampener diagnostics
    """
    gateways = {
        'King': chess.KING,
        'Queen': chess.QUEEN,
        'Knight': chess.KNIGHT,
        'Bishop': chess.BISHOP,
        'Rook': chess.ROOK,
        'Pawn': chess.PAWN
    }

    # Look up eval history for Master Dampener
    history = eval_histories.get(game_id, []) if game_id else []

    # Evaluate position with each gateway, capturing per-layer scores
    scores = {}
    layer_scores = {}  # {gateway: {16d, 32d, 64d, consensus}}
    dampener_results = {}  # {gateway: FortressDampenerResult}
    for name, piece_type in gateways.items():
        try:
            state_16d = encode_position(board)
            cascade = full_cascade(state_16d, piece_type, board)
            analysis = evaluate_position(cascade, board, eval_history_64d=history)

            # Always use White's perspective
            consensus = analysis.consensus_score
            tactical = analysis.tactical_16d.score
            positional = analysis.positional_32d.score
            strategic = analysis.strategic_64d.score
            if board.turn == chess.BLACK:
                consensus *= -1
                tactical *= -1
                positional *= -1
                strategic *= -1

            scores[name] = consensus
            layer_scores[name] = {
                '16d': tactical,
                '32d': positional,
                '64d': strategic,
                'consensus': consensus
            }

            # Capture dampener result from first gateway (they share same structural signals)
            if analysis.dampener:
                dampener_results[name] = analysis.dampener
        except Exception as e:
            continue

    if len(scores) < 2:
        return {'converged': False, 'gateways': [], 'score': 0.0, 'confidence': 'NONE'}

    # ── Dead Gateway Detection (Unified Field Theory / Option B) ──
    # Immobilized pieces still contribute to algebraic gravity,
    # but we flag them for transparency.
    dead_gateways = _detect_dead_gateways(board)

    # ── Dimensional Divergence Detection (the "h3 Spike") ──
    divergence = _detect_dimensional_divergence(layer_scores)

    # Find groups of gateways with similar scores
    score_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    converged_gateways = [score_list[0][0]]
    consensus_score = score_list[0][1]

    for gateway_name, score in score_list[1:]:
        if abs(score - consensus_score) <= threshold:
            converged_gateways.append(gateway_name)

    # Determine confidence level
    num_converged = len(converged_gateways)
    if num_converged >= 5:
        confidence = 'VERY HIGH'
    elif num_converged >= 4:
        confidence = 'HIGH'
    elif num_converged >= 3:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    # ── Master Dampener: Record consensus for temporal tracking ──
    if game_id and game_id in eval_histories:
        # Record mean consensus across all gateways
        mean_consensus = sum(scores.values()) / len(scores)
        eval_histories[game_id].append(mean_consensus)
        # Trim to max history size
        if len(eval_histories[game_id]) > MAX_EVAL_HISTORY:
            eval_histories[game_id] = eval_histories[game_id][-MAX_EVAL_HISTORY:]

    # ── Master Dampener: Aggregate dampener results across gateways ──
    # Use the dampener from the first gateway that has structural signal
    # (structural signals come from the same 64D encoding regardless of gateway)
    primary_dampener = None
    for gw_name in ['King', 'Queen', 'Knight', 'Bishop', 'Rook', 'Pawn']:
        if gw_name in dampener_results and dampener_results[gw_name].structural_signal > 0:
            primary_dampener = dampener_results[gw_name]
            break
    if primary_dampener is None and dampener_results:
        primary_dampener = next(iter(dampener_results.values()))

    return {
        'converged': num_converged >= 3,
        'gateways': converged_gateways,
        'score': consensus_score,
        'confidence': confidence,
        'all_scores': scores,
        'layer_scores': layer_scores,
        'divergence': divergence,
        'dead_gateways': dead_gateways,
        'dampener': primary_dampener,
    }


def _detect_dead_gateways(board: chess.Board) -> dict:
    """
    Detect "dead" gateways — pieces that are immobilized or trapped.

    Per Gemini Session 0 conclusion (Unified Field Theory / Option B):
    Dead gateways STILL CONTRIBUTE to the algebraic gravity of the board,
    but we flag them for interpretive transparency.

    Returns:
        dict with 'dead_pieces' list and 'unified_field_note'
    """
    dead_pieces = []

    # Check each piece type for mobility
    piece_types = {
        'Rook': chess.ROOK,
        'Bishop': chess.BISHOP,
        'Knight': chess.KNIGHT,
    }

    for name, piece_type in piece_types.items():
        for color in [chess.WHITE, chess.BLACK]:
            for square in board.pieces(piece_type, color):
                # Count moves this specific piece can make
                piece_moves = [
                    m for m in board.legal_moves
                    if m.from_square == square
                ] if board.turn == color else []

                # For the non-moving side, check pseudo-legal moves
                if board.turn != color:
                    board_copy = board.copy()
                    board_copy.turn = color
                    piece_moves = [
                        m for m in board_copy.legal_moves
                        if m.from_square == square
                    ]

                color_name = "White" if color == chess.WHITE else "Black"
                if len(piece_moves) == 0:
                    dead_pieces.append({
                        'piece': f"{color_name} {name}",
                        'square': chess.square_name(square),
                        'mobility': 0,
                        'status': 'TRAPPED'
                    })
                elif len(piece_moves) <= 2:
                    dead_pieces.append({
                        'piece': f"{color_name} {name}",
                        'square': chess.square_name(square),
                        'mobility': len(piece_moves),
                        'status': 'RESTRICTED'
                    })

    return {
        'dead_pieces': dead_pieces,
        'unified_field_note': (
            "Per Unified Field Theory (Session 0): Dead gateways still "
            "contribute to algebraic gravity. Their patterns shape the "
            "manifold regardless of physical mobility."
        ) if dead_pieces else None
    }


def _detect_dimensional_divergence(layer_scores: dict) -> dict:
    """
    Detect the "h3 Spike" pattern: 16D fluctuating wildly while
    32D and 64D remain structurally stable.

    This validates ZDTP's ability to isolate "tactical noise" from
    "strategic signal" (confirmed in Gemini Session 0).

    Args:
        layer_scores: {gateway_name: {16d, 32d, 64d, consensus}}

    Returns:
        dict with divergence metrics and detected patterns
    """
    if len(layer_scores) < 3:
        return {'detected': False, 'pattern': None}

    # Collect per-layer scores across all gateways
    scores_16d = [ls['16d'] for ls in layer_scores.values()]
    scores_32d = [ls['32d'] for ls in layer_scores.values()]
    scores_64d = [ls['64d'] for ls in layer_scores.values()]

    def std_dev(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def score_range(values):
        return max(values) - min(values) if values else 0.0

    std_16d = std_dev(scores_16d)
    std_32d = std_dev(scores_32d)
    std_64d = std_dev(scores_64d)
    range_16d = score_range(scores_16d)
    range_32d = score_range(scores_32d)
    range_64d = score_range(scores_64d)

    # Detect h3 Spike: 16D volatile, 32D/64D stable
    h3_spike = (
        std_16d > 2.0 * max(std_32d, std_64d, 0.01)
        and max(std_32d, std_64d) < 1.0
    )

    # Detect strategic divergence: 64D disagrees with 16D+32D
    mean_16d = sum(scores_16d) / len(scores_16d)
    mean_32d = sum(scores_32d) / len(scores_32d)
    mean_64d = sum(scores_64d) / len(scores_64d)
    strategic_divergence = abs(mean_64d - (mean_16d + mean_32d) / 2) > 2.0

    # Detect pattern
    if h3_spike:
        pattern = 'h3_spike'
        description = (
            f"16D TACTICAL NOISE detected (σ={std_16d:.3f}). "
            f"32D (σ={std_32d:.3f}) and 64D (σ={std_64d:.3f}) remain "
            f"structurally stable. ZDTP successfully isolates tactical "
            f"fluctuation from strategic signal."
        )
    elif strategic_divergence:
        pattern = 'strategic_divergence'
        description = (
            f"64D STRATEGIC LAYER diverges from 16D/32D consensus. "
            f"Mean 16D={mean_16d:+.2f}, 32D={mean_32d:+.2f}, "
            f"64D={mean_64d:+.2f}. Position may have long-term "
            f"compensation not visible in tactical/positional layers."
        )
    else:
        pattern = 'unified'
        description = (
            f"All dimensional layers agree. "
            f"σ: 16D={std_16d:.3f}, 32D={std_32d:.3f}, 64D={std_64d:.3f}. "
            f"Canonical Six functioning as unified manifold."
        )

    return {
        'detected': h3_spike or strategic_divergence,
        'pattern': pattern,
        'description': description,
        'metrics': {
            'std_16d': std_16d,
            'std_32d': std_32d,
            'std_64d': std_64d,
            'range_16d': range_16d,
            'range_32d': range_32d,
            'range_64d': range_64d,
            'mean_16d': mean_16d,
            'mean_32d': mean_32d,
            'mean_64d': mean_64d,
        }
    }


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available ZDTP chess tools."""
    return [
        types.Tool(
            name="chess_new_game",
            description="Start a new chess game with ZDTP analysis. You play against python-chess engine while ZDTP provides dimensional analysis (16D/32D/64D) for your moves using zero divisor patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_color": {
                        "type": "string",
                        "enum": ["white", "black"],
                        "description": "Your color (white or black)",
                        "default": "white"
                    }
                }
            }
        ),
        types.Tool(
            name="chess_make_move",
            description="""Execute a chess move - REQUIRES EXPLICIT USER PERMISSION.

CRITICAL: ONLY call this tool when user explicitly commands you to play a move, such as:
- "play e4" / "make the move e4" / "execute e4"
- "do it" / "go ahead" (after suggesting a specific move)
- "I'll play d4" (user making their own move)

DO NOT call this during analysis or when user asks "what if" / "analyze" / "should I" questions.
For analysis, use chess_analyze_move instead.

MANDATORY CONFIRMATION: You MUST provide the exact phrase the user said in the user_said parameter.
This is validated to ensure moves are only executed with explicit permission.

Executing moves without permission violates user trust and ruins the game experience.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Game ID from chess_new_game"
                    },
                    "move": {
                        "type": "string",
                        "description": "Your move in UCI format (e.g., 'e2e4', 'e7e5', 'e1g1' for castling)"
                    },
                    "user_said": {
                        "type": "string",
                        "description": "REQUIRED: The exact phrase the user typed to authorize this move. Examples: 'play e4', 'execute Bf5', 'do it', 'go ahead', 'e4'. This must match the user's actual command."
                    },
                    "show_dimensional_analysis": {
                        "type": "boolean",
                        "description": "Include full 16D/32D/64D analysis in response",
                        "default": True
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Show detailed tactical information (hanging pieces, SEE, threats)",
                        "default": False
                    }
                },
                "required": ["game_id", "move", "user_said"]
            }
        ),
        types.Tool(
            name="chess_get_dimensional_analysis",
            description="Get multi-dimensional ZDTP analysis of current position",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Game ID"
                    },
                    "gateway_type": {
                        "type": "string",
                        "enum": ["king", "queen", "knight", "bishop", "rook", "pawn", "adaptive"],
                        "description": "Which dimensional gateway to use for analysis",
                        "default": "adaptive"
                    }
                },
                "required": ["game_id"]
            }
        ),
        types.Tool(
            name="chess_get_board",
            description="Get current board state and game info",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Game ID"
                    }
                },
                "required": ["game_id"]
            }
        ),
        types.Tool(
            name="chess_analyze_move",
            description="""Analyze a move WITHOUT executing it - shows ZDTP evaluation across all dimensions.

USE THIS TOOL when:
- User asks "what if I play...?" / "analyze this move" / "should I...?"
- Exploring options and comparing candidates
- User wants to see consequences before committing
- Learning mode - understanding position without making moves

This tool NEVER changes the game state. It's purely analytical.
After analysis, wait for user to explicitly command execution via chess_make_move.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Game ID"
                    },
                    "move": {
                        "type": "string",
                        "description": "Move to analyze in UCI format (e.g., 'e2e4', 'd2d8')"
                    },
                    "gateway_type": {
                        "type": "string",
                        "enum": ["king", "queen", "knight", "bishop", "rook", "pawn", "adaptive"],
                        "description": "Which dimensional gateway to use for analysis",
                        "default": "adaptive"
                    }
                },
                "required": ["game_id", "move"]
            }
        ),
        types.Tool(
            name="chess_check_gateway_convergence",
            description="Check gateway convergence by evaluating position with multiple gateways",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Game ID"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Convergence threshold (default 0.3 - gateways within this range are considered agreeing)",
                        "default": 0.3
                    },
                    "show_details": {
                        "type": "boolean",
                        "description": "Show detailed breakdown for each gateway",
                        "default": True
                    }
                },
                "required": ["game_id"]
            }
        ),
        types.Tool(
            name="chess_load_position",
            description="""Load a chess position from FEN string or from the Stressor Position Library.

Use this to set up specific positions for analysis, testing, or study.
Stressor positions are curated test cases from the Gemini Peer Review that
stress-test ZDTP's dimensional analysis (exchange sacs, locked chains,
fortress draws, dead gateways, perpetual patterns).

After loading, use chess_check_gateway_convergence for full dimensional analysis.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "FEN string to load (e.g. 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')"
                    },
                    "stressor_key": {
                        "type": "string",
                        "description": "Key from stressor library (e.g. 'french_fortress', 'exchange_sac_petrosian'). Use chess_list_stressors to see all keys."
                    },
                    "player_color": {
                        "type": "string",
                        "enum": ["white", "black"],
                        "description": "Which color you are playing (default: side to move in the FEN)",
                        "default": "white"
                    }
                }
            }
        ),
        types.Tool(
            name="chess_list_stressors",
            description="""List all available stressor positions from the ZDTP Stressor Library.

Shows curated positions organized by category:
- Exchange Sacrifices (16D crash / 32D bind)
- Locked Chains (French/KID gateway divergence)
- Perpetual Singularity (material deficit → draw)
- Topological Finality (fortress draws)
- Dead Gateway Paradox (immobilized piece contribution)

Use chess_load_position with a stressor_key to load any position.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "exchange_sacrifice", "locked_chain", "perpetual_singularity", "topological_finality", "dead_gateway"],
                        "description": "Filter by category (default: all)",
                        "default": "all"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict
) -> list[types.TextContent]:
    """Handle tool calls."""
    
    if name == "chess_new_game":
        return await chess_new_game(arguments)
    elif name == "chess_make_move":
        return await chess_make_move(arguments)
    elif name == "chess_get_dimensional_analysis":
        return await chess_get_dimensional_analysis(arguments)
    elif name == "chess_get_board":
        return await chess_get_board(arguments)
    elif name == "chess_analyze_move":
        return await chess_analyze_move(arguments)
    elif name == "chess_check_gateway_convergence":
        return await chess_check_gateway_convergence(arguments)
    elif name == "chess_load_position":
        return await chess_load_position(arguments)
    elif name == "chess_list_stressors":
        return await chess_list_stressors(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def chess_new_game(args: dict) -> list[types.TextContent]:
    """
    Start a new ZDTP chess game.

    ZDTP Chess provides advanced dimensional analysis (16D/32D/64D)
    for YOUR moves using zero divisor transmission protocols.

    Opponent: python-chess engine (consistent strength)
    """
    global game_counter

    player_color = args.get("player_color", "white")

    # Create new game
    game_id = f"zdtp_game_{game_counter}"
    game_counter += 1

    board = chess.Board()
    games[game_id] = board
    eval_histories[game_id] = []  # Master Dampener: fresh history

    # Build response with intro screen
    response = show_game_intro()

    response += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GAME STARTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Game ID: {game_id}
Your color: {player_color.upper()}
Opponent: python-chess engine

ZDTP provides multi-dimensional analysis for YOUR moves:
  • 16D Tactical Layer - immediate threats, material balance
  • 32D Positional Layer - piece coordination, pawn structure
  • 64D Strategic Layer - long-term planning, endgame evaluation
  • Gateway System - 6 different evaluation perspectives

Starting Position:

{board}

{'▶ You play first! Make your move with chess_make_move.' if player_color == 'white' else '▶ AI is thinking...'}
"""
    
    # If player is black, make AI's first move
    if player_color == "black":
        ai_response = engine.select_move(board)
        board.push(ai_response.best_move)

        response += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AI OPENING MOVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AI's move: {ai_response.best_move_san}
Gateway used: {ai_response.gateway_used}

Position after AI's move:

{board}

▶ Your turn! Make your move with chess_make_move.
"""
    
    return [types.TextContent(type="text", text=response)]


async def chess_make_move(args: dict) -> list[types.TextContent]:
    """
    Make a move and get AI response with dimensional reasoning.

    CRITICAL: This function should ONLY be called when user explicitly
    commands move execution. See tool description for details.
    """
    game_id = args["game_id"]
    move_uci = args["move"]
    user_said = args.get("user_said", "")
    show_analysis = args.get("show_dimensional_analysis", True)
    verbose = args.get("verbose", False)

    # VALIDATION: Check that user_said parameter is provided
    if not user_said or not user_said.strip():
        return [types.TextContent(type="text", text="""
╔══════════════════════════════════════════════════════════════╗
║  ❌ MOVE EXECUTION BLOCKED - MISSING CONFIRMATION           ║
╚══════════════════════════════════════════════════════════════╝

ERROR: The 'user_said' parameter is required but was not provided.

This parameter must contain the exact phrase the user typed to
authorize this move (e.g., "play e4", "execute Bf5", "do it").

REASON: This safeguard prevents accidental move execution during
analysis. Moves should only be executed when explicitly commanded
by the user.

TO FIX:
  1. Wait for user to explicitly command a move
  2. Include their exact phrase in the user_said parameter
  3. Example: chess_make_move(game_id="...", move="e2e4", user_said="play e4")

For analysis without execution, use chess_analyze_move instead.
""")]

    # VALIDATION: Check that user_said contains move-execution keywords
    user_said_lower = user_said.lower().strip()
    valid_keywords = ['play', 'make', 'execute', 'do it', 'go ahead', 'move']

    # Also accept just the move notation if it looks like a chess move
    is_move_notation = (
        len(user_said_lower) <= 8 and
        any(c in user_said_lower for c in ['e', 'a', 'b', 'c', 'd', 'f', 'g', 'h'])
    )

    # Use whole-word matching to avoid false positives like "playing" matching "play"
    import re
    has_valid_keyword = any(
        re.search(r'\b' + re.escape(keyword) + r'\b', user_said_lower)
        for keyword in valid_keywords
    )

    if not has_valid_keyword and not is_move_notation:
        return [types.TextContent(type="text", text=f"""
╔══════════════════════════════════════════════════════════════╗
║  ❌ MOVE EXECUTION BLOCKED - INVALID CONFIRMATION           ║
╚══════════════════════════════════════════════════════════════╝

ERROR: The user_said parameter doesn't match expected move authorization.

You provided: "{user_said}"

Expected phrases like:
  • "play e4" / "execute Bf5" / "make the move"
  • "do it" / "go ahead"
  • "e4" / "Nf3" (move notation)

REASON: This safeguard ensures moves are only executed when explicitly
commanded by the user, not during analysis or exploration.

If the user asked to analyze a move (e.g., "what if I play...?",
"analyze d6", "should I...?"), use chess_analyze_move instead.

Only use chess_make_move when the user explicitly commands execution.
""")]

    # Audit log - helps identify unauthorized move executions
    import sys
    print(f"[MOVE EXECUTION] Game {game_id}: {move_uci} | User said: '{user_said}'", file=sys.stderr)

    if game_id not in games:
        return [types.TextContent(type="text", text=f"Game {game_id} not found")]
    
    board = games[game_id]

    # Save board state before player's move (for ZDTP showcase)
    board_before_player = board.copy()

    # Parse and make player's move
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return [types.TextContent(type="text",
                text=f"Illegal move: {move_uci}. Try again.")]

        player_move_san = board.san(move) # Get SAN before pushing the move
        board.push(move)

        # Save board state after player's move (for ZDTP showcase)
        board_after_player = board.copy()
        
    except ValueError as e:
        return [types.TextContent(type="text", 
            text=f"Invalid move format: {move_uci}. Use UCI format like 'e2e4'.")]
    
    # Check if game over
    if board.is_game_over():
        result = "Checkmate!" if board.is_checkmate() else "Game over!"
        return [types.TextContent(type="text",
            text=f"Your move: {player_move_san}\n\n{board}\n\n{result}")]

    # BUG #3 FIX: Save turn before AI evaluation to correct perspective later
    turn_before_ai_move = board.turn

    # Get AI's response move
    ai_response = engine.select_move(board)
    board.push(ai_response.best_move)

    # BUG #3 FIX: Flip scores back to White's perspective for consistent display
    # Engine evaluates from current player's perspective (negates for Black)
    # But we ALWAYS want to display from White's perspective (+ = good for White)
    if turn_before_ai_move == chess.BLACK:
        ai_response.analysis.consensus_score *= -1
        ai_response.analysis.tactical_16d.score *= -1
        ai_response.analysis.tactical_16d.material_balance *= -1
        ai_response.analysis.tactical_16d.king_safety_diff *= -1
        ai_response.analysis.tactical_16d.mobility_diff *= -1
        ai_response.analysis.positional_32d.score *= -1
        ai_response.analysis.positional_32d.pawn_structure *= -1
        ai_response.analysis.positional_32d.center_control *= -1
        ai_response.analysis.strategic_64d.score *= -1
        ai_response.analysis.strategic_64d.pawn_advancement *= -1

    # Map gateway names to pattern explanations
    gateway_explanations = {
        'king': 'master gateway - holistic evaluation',
        'queen': 'multi-modal gateway - tactical complexity',
        'knight': 'discontinuous gateway - non-linear patterns',
        'bishop': 'diagonal gateway - long-range planning',
        'rook': 'orthogonal gateway - file control',
        'pawn': 'incremental gateway - structural analysis'
    }
    gateway_explain = gateway_explanations.get(ai_response.gateway_used.lower(), 'adaptive selection')

    # Build clean response
    response = f"""
╔══════════════════════════════════════════════════════════════╗
║  YOUR MOVE: {player_move_san:<50} ║
╚══════════════════════════════════════════════════════════════╝

{board}

╔══════════════════════════════════════════════════════════════╗
║  BLACK RESPONDS: {ai_response.best_move_san:<45} ║
╚══════════════════════════════════════════════════════════════╝

Analysis Gateway: {ai_response.gateway_used} ({gateway_explain})
Position Evaluation: {ai_response.analysis.consensus_score:+.2f} (White's perspective)

💡 Want to check all 6 gateways? Use chess_check_gateway_convergence
"""

    # Generate ZDTP Showcase (always show transmission fidelity and dimensional analysis)
    gateway_map = {
        'king': chess.KING,
        'queen': chess.QUEEN,
        'knight': chess.KNIGHT,
        'bishop': chess.BISHOP,
        'rook': chess.ROOK,
        'pawn': chess.PAWN
    }
    gateway_piece = gateway_map.get(ai_response.gateway_used.lower(), chess.KNIGHT)

    try:
        showcase = format_zdtp_showcase(
            move_san=player_move_san,
            board_before=board_before_player,
            board_after_our_move=board_after_player,
            board_after_opponent=board,  # Current board (after AI move)
            gateway_piece=gateway_piece,
            opponent_move_san=ai_response.best_move_san
        )
        response += "\n" + showcase + "\n"
    except Exception as e:
        # If showcase fails, don't crash - just log and continue
        response += f"\n(ZDTP showcase unavailable: {str(e)})\n"

    # Add verbose tactical details if requested
    if verbose:
        response += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VERBOSE TACTICAL ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        # Analyze hanging pieces
        hanging_pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                # Check if piece is attacked and not defended
                attackers = list(board.attackers(chess.BLACK, square))
                defenders = list(board.attackers(chess.WHITE, square))
                if attackers and not defenders:
                    hanging_pieces.append(f"   • {chess.piece_name(piece.piece_type).title()} on {chess.square_name(square)} (attacked by {len(attackers)} Black piece(s))")

        if hanging_pieces:
            response += "\n⚠️  HANGING PIECES DETECTED:\n" + "\n".join(hanging_pieces) + "\n"
        else:
            response += "\n✓ No hanging pieces detected\n"

        # Show threats
        if board.is_check():
            response += f"\n⚡ KING IN CHECK - {board.turn.name} must respond!\n"

        # Show material count
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        white_material = sum(piece_values.get(piece.piece_type, 0) for square in chess.SQUARES if (piece := board.piece_at(square)) and piece.color == chess.WHITE)
        black_material = sum(piece_values.get(piece.piece_type, 0) for square in chess.SQUARES if (piece := board.piece_at(square)) and piece.color == chess.BLACK)

        response += f"\n📊 Material Count: White {white_material} - Black {black_material} (Δ{white_material - black_material:+d})\n"

        response += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    if show_analysis:
        response += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MULTI-DIMENSIONAL POSITION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 16D TACTICAL LAYER (Immediate Threats & Material)
   Score: {ai_response.analysis.tactical_16d.score:+.2f}
   Material Balance: {ai_response.analysis.tactical_16d.material_balance:+.1f}

   Position after {ai_response.best_move_san}:
   {ai_response.analysis.tactical_16d.reasoning}

🏗️  32D POSITIONAL LAYER (Structure & Coordination)
   Score: {ai_response.analysis.positional_32d.score:+.2f}

   {ai_response.analysis.positional_32d.reasoning}

🌟 64D STRATEGIC LAYER (Long-term Planning)
   Score: {ai_response.analysis.strategic_64d.score:+.2f}

   {ai_response.analysis.strategic_64d.reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 OVERALL ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{ai_response.analysis.overall_assessment}
"""
    
    # Check if game over after AI move
    if board.is_game_over():
        if board.is_checkmate():
            response += "\n\n🏆 Checkmate! AI wins!"
        else:
            response += "\n\n🤝 Game over!"
    else:
        response += "\n\nYour turn!"
    
    return [types.TextContent(type="text", text=response)]


async def chess_get_dimensional_analysis(args: dict) -> list[types.TextContent]:
    """Get detailed dimensional analysis of current position."""
    game_id = args["game_id"]
    gateway_str = args.get("gateway_type", "adaptive")
    
    if game_id not in games:
        return [types.TextContent(type="text", text=f"Game {game_id} not found")]
    
    board = games[game_id]
    
    # Map gateway string to piece type
    gateway_map = {
        'king': chess.KING,
        'queen': chess.QUEEN,
        'knight': chess.KNIGHT,
        'bishop': chess.BISHOP,
        'rook': chess.ROOK,
        'pawn': chess.PAWN
    }
    
    if gateway_str == 'adaptive':
        # Use engine's adaptive selection
        temp_engine = ZDTPEngine(gateway_strategy='adaptive')
        gateway_piece = temp_engine._select_gateway(board)
    else:
        gateway_piece = gateway_map.get(gateway_str, chess.KNIGHT)
    
    # Look up eval history for Master Dampener
    history = eval_histories.get(game_id, [])

    # Encode and cascade (with intelligent tactical/strategic analysis!)
    state_16d = encode_position(board)
    cascade = full_cascade(state_16d, gateway_piece, board)
    analysis = evaluate_position(cascade, board, eval_history_64d=history)

    # Record consensus for temporal tracking
    if game_id in eval_histories:
        eval_histories[game_id].append(analysis.consensus_score)
        if len(eval_histories[game_id]) > MAX_EVAL_HISTORY:
            eval_histories[game_id] = eval_histories[game_id][-MAX_EVAL_HISTORY:]

    # Gateway explanation
    gateway_explanations = {
        'king': 'master gateway - holistic evaluation',
        'queen': 'multi-modal gateway - tactical complexity',
        'knight': 'discontinuous gateway - non-linear patterns',
        'bishop': 'diagonal gateway - long-range planning',
        'rook': 'orthogonal gateway - file control',
        'pawn': 'incremental gateway - structural analysis'
    }
    gateway_explain = gateway_explanations.get(chess.piece_name(gateway_piece).lower(), 'adaptive selection')

    response = f"""
╔══════════════════════════════════════════════════════════════╗
║  🔬 ZDTP DIMENSIONAL ANALYSIS                                ║
╚══════════════════════════════════════════════════════════════╝

Current Position:
{board}

Gateway: {chess.piece_name(gateway_piece)} ({gateway_explain})
Pattern ID: {cascade['portal_16_32']['gateway_pattern']['id']}
Transmission Fidelity: {cascade['overall_fidelity']:.0%}
Consensus Score: {analysis.consensus_score:+.2f} (White's perspective)

💡 Want to check all 6 gateways? Use chess_check_gateway_convergence

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DIMENSIONAL BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

16D TACTICAL LAYER
   Score: {analysis.tactical_16d.score:+.2f}
   Material: {analysis.tactical_16d.material_balance:+.1f}

   {analysis.tactical_16d.reasoning}

32D POSITIONAL LAYER
   Score: {analysis.positional_32d.score:+.2f}

   {analysis.positional_32d.reasoning}

64D STRATEGIC LAYER
   Score: {analysis.strategic_64d.score:+.2f}

   {analysis.strategic_64d.reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OVERALL ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{analysis.overall_assessment}
"""
    
    return [types.TextContent(type="text", text=response)]


async def chess_get_board(args: dict) -> list[types.TextContent]:
    """Get current board state."""
    game_id = args["game_id"]
    
    if game_id not in games:
        return [types.TextContent(type="text", text=f"Game {game_id} not found")]
    
    board = games[game_id]
    
    response = f"""
Current Position:

{board}

FEN: {board.fen()}
Turn: {'White' if board.turn == chess.WHITE else 'Black'}
Move number: {board.fullmove_number}

Legal moves: {len(list(board.legal_moves))}
"""
    
    if board.is_check():
        response += "\n⚠️ IN CHECK"
    
    if board.is_checkmate():
        response += "\n🏆 CHECKMATE"
    elif board.is_stalemate():
        response += "\n🤝 STALEMATE"
    elif board.is_insufficient_material():
        response += "\n🤝 INSUFFICIENT MATERIAL"

    return [types.TextContent(type="text", text=response)]


async def chess_analyze_move(args: dict) -> list[types.TextContent]:
    """
    Analyze a move WITHOUT executing it (P0 Critical Fix).

    Simulates the move and shows dimensional analysis of resulting position.
    """
    game_id = args["game_id"]
    move_uci = args["move"]
    gateway_str = args.get("gateway_type", "adaptive")

    if game_id not in games:
        return [types.TextContent(type="text", text=f"Game {game_id} not found")]

    board = games[game_id]

    # Parse the move
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return [types.TextContent(type="text",
                text=f"⚠️ ILLEGAL MOVE: {move_uci} is not legal in this position.\n\nLegal moves: {', '.join(board.san(m) for m in list(board.legal_moves)[:10])}...")]

        move_san = board.san(move)

    except ValueError as e:
        return [types.TextContent(type="text",
            text=f"❌ Invalid move format: {move_uci}. Use UCI format like 'e2e4' or 'd2d8'.")]

    # Emergency SEE: Check move safety BEFORE detailed analysis (integrated into engine)
    safety_result = _emergency_see_safety_check(board, move)

    # PRIORITY 1 FIX: Check opponent's response for tactical issues
    # This fixes the Ne5 repetition loop from Moves 14-16
    opponent_analysis = analyze_opponent_responses(board, move)

    # Create a copy of the board and simulate the move
    board_copy = board.copy()
    board_copy.push(move)

    # Map gateway string to piece type
    gateway_map = {
        'king': chess.KING,
        'queen': chess.QUEEN,
        'knight': chess.KNIGHT,
        'bishop': chess.BISHOP,
        'rook': chess.ROOK,
        'pawn': chess.PAWN
    }

    if gateway_str == 'adaptive':
        # Use engine's adaptive selection
        temp_engine = ZDTPEngine(gateway_strategy='adaptive')
        gateway_piece = temp_engine._select_gateway(board_copy)
    else:
        gateway_piece = gateway_map.get(gateway_str, chess.KNIGHT)

    # Look up eval history for Master Dampener
    history = eval_histories.get(game_id, [])

    # Encode and cascade to get analysis
    state_16d = encode_position(board_copy)
    cascade = full_cascade(state_16d, gateway_piece, board_copy)
    analysis = evaluate_position(cascade, board_copy, eval_history_64d=history)

    # Note: analyze_move does NOT record to eval_history
    # because it's hypothetical (move not executed)

    # BUG #3 FIX: Flip to White's perspective if user is Black
    # ALWAYS show scores from White's perspective (+ = good for White)
    if board.turn == chess.BLACK:
        analysis.consensus_score *= -1
        analysis.tactical_16d.score *= -1
        analysis.tactical_16d.material_balance *= -1
        analysis.tactical_16d.king_safety_diff *= -1
        analysis.tactical_16d.mobility_diff *= -1
        analysis.positional_32d.score *= -1
        analysis.positional_32d.pawn_structure *= -1
        analysis.positional_32d.center_control *= -1
        analysis.strategic_64d.score *= -1
        analysis.strategic_64d.pawn_advancement *= -1

    # Gateway explanation
    gateway_explanations = {
        'king': 'master gateway - holistic evaluation',
        'queen': 'multi-modal gateway - tactical complexity',
        'knight': 'discontinuous gateway - non-linear patterns',
        'bishop': 'diagonal gateway - long-range planning',
        'rook': 'orthogonal gateway - file control',
        'pawn': 'incremental gateway - structural analysis'
    }
    gateway_explain = gateway_explanations.get(chess.piece_name(gateway_piece).lower(), 'adaptive selection')

    # Build analysis response with safety warning if detected
    response = f"""
╔══════════════════════════════════════════════════════════════╗
║  YOUR ANALYSIS: {move_san:<48} ║
║  (HYPOTHETICAL - NOT EXECUTED)                               ║
╚══════════════════════════════════════════════════════════════╝
"""

    # EMERGENCY HOTFIX: Prepend blunder warning if detected
    if not safety_result['is_safe'] and safety_result['warning']:
        response += f"""
╔═══════════════════════════════════════════════════════════╗
║  🚨 CRITICAL BLUNDER - DO NOT PLAY THIS MOVE 🚨            ║
╚═══════════════════════════════════════════════════════════╝

{safety_result['warning']}

SEE (Static Exchange Evaluation): {safety_result['see_value']:.1f}
This move loses significant material through forced recapture!

🚫 STRONG RECOMMENDATION: DO NOT PLAY THIS MOVE

   CRITICAL: The 16D tactical layer has detected an immediate disaster.
   Positional and strategic gains are irrelevant if you lose major
   material. The consensus score reflects this catastrophic blunder.

"""

    # PRIORITY 1 FIX: Add opponent response warning if detected
    # This prevents the Ne5 repetition loop issue
    if not opponent_analysis.safe_after_response and opponent_analysis.tactical_warning:
        response += f"""
╔═══════════════════════════════════════════════════════════╗
║  ⚠️  TACTICAL WARNING - OPPONENT RESPONSE CHECK            ║
╚═══════════════════════════════════════════════════════════╝

{opponent_analysis.tactical_warning}

Immediate Safety: {'✓ Safe' if opponent_analysis.immediate_safe else '⚠️ Hangs immediately'}
After Opponent's Best Response: ⚠️ Tactical issue detected

Opponent can play:
"""
        for i, resp in enumerate(opponent_analysis.all_dangerous_responses[:3], 1):
            response += f"  {i}. {resp.move_san} ({resp.threat_type})"
            if resp.hangs_our_piece:
                piece_name = chess.piece_name(resp.hangs_our_piece[1])
                square_name = chess.square_name(resp.hangs_our_piece[0])
                response += f" → hangs {piece_name} on {square_name}"
            response += "\n"

        if opponent_analysis.worst_response and opponent_analysis.worst_response.material_loss > 1.0:
            response += f"\n⚠️ RECOMMENDATION: RISKY - Consider alternatives (loses {opponent_analysis.worst_response.material_loss:.0f} material)\n"
        else:
            response += f"\n⚠️ NOTE: Tactical complexity - opponent has counter-play\n"

        response += "\n"

    # After opponent response warning, add showcase:
    # If opponent response has tactical warning:
    if opponent_analysis and opponent_analysis.worst_response:
        # Create board after opponent's worst response
        board_after_opp = board_copy.copy()
        board_after_opp.push(opponent_analysis.worst_response.move)

        # Generate showcase
        showcase = format_zdtp_showcase(
            move_san=move_san,
            board_before=board,
            board_after_our_move=board_copy,
            board_after_opponent=board_after_opp,
            gateway_piece=gateway_piece,
            opponent_move_san=opponent_analysis.worst_response.move_san
        )

        response += "\n" + showcase + "\n"

    response += f"""
Position after {move_san}:

{board_copy}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MULTI-DIMENSIONAL EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analysis Gateway: {chess.piece_name(gateway_piece)} ({gateway_explain})
Consensus Score: {analysis.consensus_score:+.2f} (White's perspective)

🎯 16D TACTICAL LAYER
   Score: {analysis.tactical_16d.score:+.2f}
   Material: {analysis.tactical_16d.material_balance:+.1f}

   {analysis.tactical_16d.reasoning}

🏗️  32D POSITIONAL LAYER
   Score: {analysis.positional_32d.score:+.2f}

   {analysis.positional_32d.reasoning}

🌟 64D STRATEGIC LAYER
   Score: {analysis.strategic_64d.score:+.2f}

   {analysis.strategic_64d.reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 OVERALL ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{analysis.overall_assessment}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  THIS MOVE HAS NOT BEEN EXECUTED
Use chess_make_move to play it, or analyze other candidates.
"""

    return [types.TextContent(type="text", text=response)]


async def chess_check_gateway_convergence(args: dict) -> list[types.TextContent]:
    """
    Check gateway convergence by evaluating position with all 6 gateways.

    This is an expensive operation (6x evaluations) - use strategically!
    """
    game_id = args["game_id"]
    threshold = args.get("threshold", 0.3)
    show_details = args.get("show_details", True)

    if game_id not in games:
        return [types.TextContent(type="text", text=f"Game {game_id} not found")]

    board = games[game_id]

    # Run convergence check (with Master Dampener history)
    convergence = detect_gateway_convergence(board, threshold, game_id=game_id)

    # Build response
    response = f"""
╔══════════════════════════════════════════════════════════════╗
║  🎯 GATEWAY CONVERGENCE CHECK                                ║
╚══════════════════════════════════════════════════════════════╝

Position:
{board}

Threshold: {threshold} (gateways within ±{threshold} are considered agreeing)

"""

    if show_details and 'all_scores' in convergence:
        response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        response += "  INDIVIDUAL GATEWAY EVALUATIONS\n"
        response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        scores = convergence['all_scores']
        for gateway_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            response += f"  {gateway_name:8s}: {score:+.4f}\n"

        # Calculate statistics
        score_values = list(scores.values())
        mean = sum(score_values) / len(score_values)
        variance = sum((s - mean)**2 for s in score_values) / len(score_values)
        std_dev = variance ** 0.5
        score_range = max(score_values) - min(score_values)

        response += f"\n  Mean:     {mean:+.4f}\n"
        response += f"  Std Dev:  {std_dev:.4f}\n"
        response += f"  Range:    {score_range:.4f}\n"

    response += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    response += "  CONVERGENCE ANALYSIS\n"
    response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    if convergence['converged']:
        gateways_str = ', '.join(convergence['gateways'])
        response += f"✓ CONVERGENCE DETECTED\n\n"
        response += f"  Agreeing Gateways: {len(convergence['gateways'])}/6\n"
        response += f"  {gateways_str}\n\n"
        response += f"  Convergence Score: {convergence['score']:+.2f} (White's perspective)\n"
        response += f"  Confidence Level: {convergence['confidence']}\n\n"
        response += f"  → Framework-independent optimality confirmed\n"
        response += f"  → Multiple evaluation methods agree on position assessment\n"
    else:
        response += f"✗ NO CONVERGENCE\n\n"
        response += f"  Agreeing Gateways: {len(convergence['gateways'])}/6 (need ≥3 for convergence)\n"
        response += f"  Confidence Level: {convergence['confidence']}\n\n"
        response += f"  → Gateways disagree on this position\n"
        response += f"  → Consider trying different gateways for detailed analysis\n"
        response += f"  → Position may have complex trade-offs\n"

    # ── SESSION 1: Dimensional Divergence Detection ──
    divergence = convergence.get('divergence', {})
    if divergence and divergence.get('detected'):
        pattern = divergence.get('pattern', 'unknown')
        pattern_label = {
            'h3_spike': '⚡ h3 SPIKE — 16D Tactical Noise Detected',
            'strategic_divergence': '🌀 STRATEGIC DIVERGENCE — 64D Disagrees',
        }.get(pattern, f'🔍 {pattern.upper()}')

        response += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        response += f"  DIMENSIONAL DIVERGENCE (Session 1)\n"
        response += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        response += f"  {pattern_label}\n\n"
        response += f"  {divergence.get('description', '')}\n"

        metrics = divergence.get('metrics', {})
        if metrics:
            response += f"\n  Layer Volatility (σ across gateways):\n"
            response += f"    16D: σ={metrics.get('std_16d', 0):.4f}  range={metrics.get('range_16d', 0):.4f}  mean={metrics.get('mean_16d', 0):+.4f}\n"
            response += f"    32D: σ={metrics.get('std_32d', 0):.4f}  range={metrics.get('range_32d', 0):.4f}  mean={metrics.get('mean_32d', 0):+.4f}\n"
            response += f"    64D: σ={metrics.get('std_64d', 0):.4f}  range={metrics.get('range_64d', 0):.4f}  mean={metrics.get('mean_64d', 0):+.4f}\n"
    elif divergence:
        response += f"\n  ✓ Dimensional Layers Unified — Canonical Six functioning as unified manifold.\n"
        metrics = divergence.get('metrics', {})
        if metrics:
            response += f"    σ: 16D={metrics.get('std_16d', 0):.4f}, 32D={metrics.get('std_32d', 0):.4f}, 64D={metrics.get('std_64d', 0):.4f}\n"

    # ── SESSION 1: Per-Layer Gateway Scores ──
    layer_scores = convergence.get('layer_scores', {})
    if show_details and layer_scores:
        response += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        response += f"  PER-DIMENSIONAL LAYER SCORES (Session 1)\n"
        response += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        response += f"  {'Gateway':<10} {'16D':>8} {'32D':>8} {'64D':>8} {'Consensus':>10}\n"
        response += f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*10}\n"
        for gw_name, layers in sorted(layer_scores.items()):
            response += f"  {gw_name:<10} {layers['16d']:>+8.3f} {layers['32d']:>+8.3f} {layers['64d']:>+8.3f} {layers['consensus']:>+10.3f}\n"

    # ── SESSION 1: Dead Gateway Detection ──
    dead_gateways = convergence.get('dead_gateways', {})
    dead_pieces = dead_gateways.get('dead_pieces', []) if dead_gateways else []
    if dead_pieces:
        response += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        response += f"  👻 DEAD GATEWAY DETECTION (Session 1)\n"
        response += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        for dp in dead_pieces:
            status_emoji = '🔒' if dp['status'] == 'TRAPPED' else '⚠️'
            response += f"  {status_emoji} {dp['piece']} on {dp['square']} — {dp['status']} (mobility: {dp['mobility']})\n"
        if dead_gateways.get('unified_field_note'):
            response += f"\n  📐 {dead_gateways['unified_field_note']}\n"

    # ── SESSION 0.1: Master Dampener (Fortress Draw Detection) ──
    dampener = convergence.get('dampener')
    if dampener:
        response += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        response += f"  🏰 MASTER DAMPENER (Session 0.1 - Lean 4 Grounded)\n"
        response += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        if dampener.active:
            response += f"  ⚠️  DAMPENER ACTIVE — Fortress conditions detected\n\n"
            response += f"  Raw Consensus:      {dampener.raw_consensus:+.3f}\n"
            response += f"  Dampened Consensus: {dampener.dampened_consensus:+.3f}\n"
            response += f"  Fortress Signal:    {dampener.fortress_signal:.0%}\n\n"
        else:
            response += f"  ✓ Dampener inactive\n\n"

        response += f"  Structural Signal:  {dampener.structural_signal:.3f}"
        if dampener.structural_signal > 0.3:
            response += " (ABOVE threshold)"
        elif dampener.structural_signal > 0:
            response += " (below threshold 0.3)"
        response += f"\n"
        response += f"  Temporal Check:     {'CONFIRMED' if dampener.temporal_confirmed else 'not confirmed'}"
        response += f" ({dampener.eval_history_count} evals in history"
        if dampener.eval_history_count >= 4:
            response += f", delta={dampener.eval_stasis_delta:.3f} vs M={dampener.stability_constant_M}"
        response += f")\n\n"
        response += f"  {dampener.reason}\n"

        # Show formula
        if dampener.active:
            response += f"\n  Formula: Consensus × (1 - FortressSignal)\n"
            response += f"           {dampener.raw_consensus:+.3f} × (1 - {dampener.fortress_signal:.3f}) = {dampener.dampened_consensus:+.3f}\n"

    response += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    return [types.TextContent(type="text", text=response)]


async def chess_load_position(args: dict) -> list[types.TextContent]:
    """
    Load a chess position from FEN or stressor key.

    Session 1 Feature: Enables testing specific positions from the
    Stressor Position Library defined in Gemini Peer Review Session 0.
    """
    global game_counter

    fen = args.get("fen")
    stressor_key = args.get("stressor_key")
    player_color = args.get("player_color")

    # Must provide either FEN or stressor key
    if not fen and not stressor_key:
        return [types.TextContent(type="text", text="""
╔══════════════════════════════════════════════════════════════╗
║  ❌ ERROR: Must provide either 'fen' or 'stressor_key'       ║
╚══════════════════════════════════════════════════════════════╝

Examples:
  chess_load_position(fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
  chess_load_position(stressor_key="french_fortress")

Use chess_list_stressors to see all available stressor keys.
""")]

    # Load stressor position
    stressor_info = None
    if stressor_key:
        if stressor_key not in STRESSOR_LIBRARY:
            available = ', '.join(sorted(STRESSOR_LIBRARY.keys()))
            return [types.TextContent(type="text", text=f"""
╔══════════════════════════════════════════════════════════════╗
║  ❌ ERROR: Unknown stressor key '{stressor_key}'              ║
╚══════════════════════════════════════════════════════════════╝

Available keys:
  {available}

Use chess_list_stressors for full descriptions.
""")]
        stressor_info = STRESSOR_LIBRARY[stressor_key]
        fen = stressor_info["fen"]

    # Validate and create board from FEN
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return [types.TextContent(type="text", text=f"""
╔══════════════════════════════════════════════════════════════╗
║  ❌ ERROR: Invalid FEN string                                ║
╚══════════════════════════════════════════════════════════════╝

{str(e)}

FEN provided: {fen}
""")]

    # Create game
    game_id = f"zdtp_game_{game_counter}"
    game_counter += 1
    games[game_id] = board
    eval_histories[game_id] = []  # Master Dampener: fresh history for loaded position

    # Determine player color from FEN if not specified
    if not player_color:
        player_color = "white" if board.turn == chess.WHITE else "black"

    # Build response
    response = f"""
╔══════════════════════════════════════════════════════════════╗
║  📋 POSITION LOADED                                          ║
╚══════════════════════════════════════════════════════════════╝

Game ID: {game_id}
Your color: {player_color.upper()}
Side to move: {'White' if board.turn == chess.WHITE else 'Black'}
Move number: {board.fullmove_number}
"""

    if stressor_info:
        difficulty_emoji = {"medium": "🟡", "hard": "🟠", "expert": "🔴"}.get(
            stressor_info.get("difficulty", ""), "⚪"
        )
        response += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRESSOR: {stressor_info['name']}  {difficulty_emoji}
  Category: {stressor_info['category'].replace('_', ' ').title()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{stressor_info['description']}
"""
        # Show expected behavior hints
        for key, value in stressor_info.items():
            if key.startswith('expected_'):
                label = key.replace('expected_', '').replace('_', ' ').title()
                response += f"\n  Expected {label}: {value}"

        if stressor_info.get('test_move'):
            response += f"\n\n  💡 Suggested test move: {stressor_info['test_move']}"
            response += f"\n     Use chess_analyze_move to test without executing."

    response += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  POSITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{board}

FEN: {fen}
Legal moves: {len(list(board.legal_moves))}
"""

    if board.is_check():
        response += "\n⚠️ IN CHECK"

    response += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RECOMMENDED NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. chess_check_gateway_convergence - Full 6-gateway analysis
  2. chess_get_dimensional_analysis  - Single gateway deep dive
  3. chess_analyze_move              - Test specific moves
  4. chess_make_move                 - Play a move
"""

    return [types.TextContent(type="text", text=response)]


async def chess_list_stressors(args: dict) -> list[types.TextContent]:
    """
    List all stressor positions from the ZDTP Stressor Library.

    Session 1 Feature: Displays the curated test positions from
    the Gemini Peer Review Stressor Roadmap.
    """
    category = args.get("category", "all")

    if category == "all":
        response = list_stressor_positions()
    else:
        positions = get_stressor_category(category)
        if not positions:
            return [types.TextContent(type="text", text=f"No positions found for category: {category}")]

        category_titles = {
            "exchange_sacrifice": "💥 EXCHANGE SACRIFICES",
            "locked_chain": "🔒 LOCKED CHAINS",
            "perpetual_singularity": "♾️  PERPETUAL SINGULARITY",
            "topological_finality": "🏔️  TOPOLOGICAL FINALITY",
            "dead_gateway": "👻 DEAD GATEWAY PARADOX",
        }

        title = category_titles.get(category, category.upper())
        lines = []
        lines.append(f"╔══════════════════════════════════════════════════════════════╗")
        lines.append(f"║  {title:<58} ║")
        lines.append(f"╚══════════════════════════════════════════════════════════════╝")
        lines.append("")

        for pos in positions:
            difficulty_emoji = {"medium": "🟡", "hard": "🟠", "expert": "🔴"}.get(
                pos["difficulty"], "⚪"
            )
            lines.append(f"  {difficulty_emoji} {pos['key']}")
            lines.append(f"     {pos['name']}")
            lines.append(f"     {pos['description'][:80]}...")
            lines.append(f"     FEN: {pos['fen']}")
            if pos.get('test_move'):
                lines.append(f"     Test move: {pos['test_move']}")
            lines.append("")

        lines.append(f"Total: {len(positions)} positions in {category}")
        lines.append(f"\nLoad: chess_load_position(stressor_key='...')")
        response = "\n".join(lines)

    return [types.TextContent(type="text", text=response)]


async def run_test_move_14_ne5_analysis() -> list[types.TextContent]:
    """
    Dedicated test function to analyze Ne5 from Move 14 position.
    Sets up the board, calls chess_analyze_move, and returns the analysis.
    """
    test_game_id = "test_game_0"
    move_14_fen = "1rb1kb1r/ppqppppp/2p2n2/n2P4/Q7/2N2N2/PP2PPPP/1R2R1K1 w - - 0 14"
    move_to_analyze = "f3e5"

    # Create or reset the test game
    games[test_game_id] = chess.Board(move_14_fen)
    eval_histories[test_game_id] = []  # Master Dampener: fresh history

    # Call chess_analyze_move
    result = await chess_analyze_move(
        {"game_id": test_game_id, "move": move_to_analyze}
    )
    return result


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="zdtp-chess",
                server_version="2.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())