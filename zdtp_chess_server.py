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
from .multidimensional_evaluator import evaluate_position


# Game storage
games: Dict[str, chess.Board] = {}
game_counter = 0

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

YOU: White pieces (human player)
OPPONENT: Black pieces (computer AI)

YOUR ADVANTAGE: Multi-dimensional analysis tools
  • 16D Tactical Layer - immediate threats, material balance
  • 32D Positional Layer - piece coordination, pawn structure, center control
  • 64D Strategic Layer - long-term planning, endgame evaluation
  • Gateway Convergence - framework-independent optimal moves

HOW IT WORKS:
  • After each move, you'll see analysis from an adaptive gateway
  • Positive scores = advantage for White (you)
  • Negative scores = advantage for Black (opponent)
  • Gateway convergence shows when multiple frameworks agree

GOAL: Use higher-dimensional mathematics to defeat traditional chess AI

COMMANDS:
  chess_make_move        - Make a move (e.g., move="e2e4")
  chess_analyze_move     - Preview a move before playing
  chess_get_dimensional_analysis - Detailed position analysis
  chess_get_board        - Show current position

OPTIONAL PARAMETERS:
  verbose=true           - Show detailed tactical information
  show_dimensional_analysis=false - Hide analysis (faster play)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to begin? Make your first move with chess_make_move!
"""


def detect_gateway_convergence(board: chess.Board, threshold: float = 0.1) -> dict:
    """
    Detect gateway convergence by evaluating position with multiple gateways.

    Args:
        board: Current board position
        threshold: Score difference threshold for convergence (default 0.1)

    Returns:
        dict with 'converged' (bool), 'gateways' (list), 'score' (float), 'confidence' (str)
    """
    gateways = {
        'King': chess.KING,
        'Queen': chess.QUEEN,
        'Knight': chess.KNIGHT,
        'Bishop': chess.BISHOP,
        'Rook': chess.ROOK,
        'Pawn': chess.PAWN
    }

    # Evaluate position with each gateway
    scores = {}
    for name, piece_type in gateways.items():
        try:
            state_16d = encode_position(board)
            cascade = full_cascade(state_16d, piece_type, board)
            analysis = evaluate_position(cascade, board)

            # Always use White's perspective
            score = analysis.consensus_score
            if board.turn == chess.BLACK:
                score *= -1

            scores[name] = score
        except Exception as e:
            # Skip this gateway if evaluation fails
            continue

    if len(scores) < 2:
        return {'converged': False, 'gateways': [], 'score': 0.0, 'confidence': 'NONE'}

    # Find groups of gateways with similar scores
    score_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    converged_gateways = [score_list[0][0]]  # Start with highest-scoring gateway
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

    return {
        'converged': num_converged >= 3,
        'gateways': converged_gateways,
        'score': consensus_score,
        'confidence': confidence,
        'all_scores': scores
    }


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available ZDTP chess tools."""
    return [
        types.Tool(
            name="chess_new_game",
            description="Start a new ZDTP chess game",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_color": {
                        "type": "string",
                        "enum": ["white", "black"],
                        "description": "Your color (white or black)",
                        "default": "white"
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["novice", "intermediate", "advanced"],
                        "description": "ZDTP AI difficulty level",
                        "default": "intermediate"
                    }
                }
            }
        ),
        types.Tool(
            name="chess_make_move",
            description="Make a move and get ZDTP AI response with dimensional reasoning",
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
                "required": ["game_id", "move"]
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
            description="Analyze a move WITHOUT executing it - shows what would happen across all dimensions (P0 FIX)",
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
    else:
        raise ValueError(f"Unknown tool: {name}")


async def chess_new_game(args: dict) -> list[types.TextContent]:
    """Start a new ZDTP chess game."""
    global game_counter
    
    player_color = args.get("player_color", "white")
    difficulty = args.get("difficulty", "intermediate")
    
    # Create new game
    game_id = f"zdtp_game_{game_counter}"
    game_counter += 1
    
    board = chess.Board()
    games[game_id] = board

    # Build response with intro screen
    response = show_game_intro()

    response += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GAME STARTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Game ID: {game_id}
Your color: {player_color.upper()}
AI difficulty: {difficulty} (using Zero Divisor Traversal Protocol)

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
    """Make a move and get AI response with dimensional reasoning."""
    game_id = args["game_id"]
    move_uci = args["move"]
    show_analysis = args.get("show_dimensional_analysis", True)
    verbose = args.get("verbose", False)
    
    if game_id not in games:
        return [types.TextContent(type="text", text=f"Game {game_id} not found")]
    
    board = games[game_id]
    
    # Parse and make player's move
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return [types.TextContent(type="text", 
                text=f"Illegal move: {move_uci}. Try again.")]
        
        player_move_san = board.san(move) # Get SAN before pushing the move
        board.push(move)
        
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
    
    # Detect gateway convergence
    convergence = detect_gateway_convergence(board)

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
"""

    # Add convergence indicator if detected
    if convergence['converged']:
        gateways_str = ', '.join(convergence['gateways'])
        response += f"""
🎯 GATEWAY CONVERGENCE DETECTED!
   {len(convergence['gateways'])} gateways agree: {gateways_str}
   Convergence Score: {convergence['average_score']:+.2f} (White's perspective)
   Confidence: {convergence['confidence']}
   → Framework-independent optimality confirmed

   📊 Detailed analysis below uses {ai_response.gateway_used.title()} gateway
      (selected via adaptive gateway selection)
"""

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
    
    # Encode and cascade (with intelligent tactical/strategic analysis!)
    state_16d = encode_position(board)
    cascade = full_cascade(state_16d, gateway_piece, board)
    analysis = evaluate_position(cascade, board)

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

    # Detect gateway convergence
    convergence = detect_gateway_convergence(board)

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
"""

    # Add convergence indicator if detected
    if convergence['converged']:
        gateways_str = ', '.join(convergence['gateways'])
        response += f"""
🎯 GATEWAY CONVERGENCE DETECTED!
   {len(convergence['gateways'])} gateways agree: {gateways_str}
   Convergence Score: {convergence['average_score']:+.2f} (White's perspective)
   Confidence: {convergence['confidence']}
   → Framework-independent optimality confirmed

   📊 Detailed analysis below uses {chess.piece_name(gateway_piece).title()} gateway
"""

    response += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DIMENSIONAL BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    # Encode and cascade to get analysis
    state_16d = encode_position(board_copy)
    cascade = full_cascade(state_16d, gateway_piece, board_copy)
    analysis = evaluate_position(cascade, board_copy)

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
║  ⚠️  BLUNDER ALERT - EMERGENCY SAFETY CHECK                ║
╚═══════════════════════════════════════════════════════════╝

{safety_result['warning']}

SEE (Static Exchange Evaluation): {safety_result['see_value']:.1f}
This move loses significant material through recapture!

🚫 RECOMMENDATION: DO NOT PLAY THIS MOVE

"""

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