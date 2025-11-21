"""
Opponent Response Analyzer for ZDTP Chess

This module implements 1-ply opponent response prediction to detect
tactical issues that occur after the opponent's best reply.

Critical fix for:
- Move 14/16 Ne5 repetition loop (knight hangs after Qc7)
- False "completely safe" assessments
- Repetition loops caused by tactical blindness

Author: Chavez AI Labs
Date: 2025-11-18
"""

import chess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


@dataclass
class OpponentResponse:
    """A potential opponent response to our move."""
    move: chess.Move
    move_san: str
    threat_type: str  # 'capture', 'check', 'attack_piece', 'quiet'
    hangs_our_piece: Optional[Tuple[chess.Square, chess.PieceType]]  # (square, piece_type) if creates hanging
    material_loss: float  # Expected material loss if this response is played
    evaluation_drop: float  # How much position worsens (if we have evaluator)


@dataclass
class OpponentResponseAnalysis:
    """Analysis of all opponent responses to our candidate move."""
    our_move: chess.Move
    our_move_san: str
    immediate_safe: bool  # Is move safe immediately after we play it?
    safe_after_response: bool  # Is move safe after opponent's best response?
    worst_response: Optional[OpponentResponse]  # Opponent's most dangerous reply
    all_dangerous_responses: List[OpponentResponse]  # All responses that create threats
    tactical_warning: Optional[str]  # Human-readable warning
    recommended: bool  # Should we recommend this move?


def check_hanging_pieces(board: chess.Board, color: chess.Color) -> List[Tuple[chess.Square, chess.PieceType]]:
    """
    Check for hanging pieces (attacked and not defended) for given color.

    Args:
        board: Current board position
        color: Color to check for hanging pieces (chess.WHITE or chess.BLACK)

    Returns:
        List of (square, piece_type) tuples for hanging pieces
    """
    hanging = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            # Check if piece is attacked by opponent
            attackers = list(board.attackers(not color, square))
            if attackers:
                # Check if piece is defended by our pieces
                defenders = list(board.attackers(color, square))

                # Simple hanging check: attacked but not defended
                if not defenders:
                    hanging.append((square, piece.piece_type))

    return hanging


def analyze_opponent_responses(
    board: chess.Board,
    our_move: chess.Move,
    max_responses: int = 10
) -> OpponentResponseAnalysis:
    """
    Analyze opponent's potential responses to our candidate move.

    This is the core fix for the Ne5 repetition loop issue. It simulates
    the opponent's likely responses and checks if any of them create
    tactical problems for us.

    Args:
        board: Current board position (BEFORE our move)
        our_move: Our candidate move to analyze
        max_responses: Maximum number of opponent responses to check

    Returns:
        OpponentResponseAnalysis with tactical assessment
    """
    our_move_san = board.san(our_move)

    # Apply our move
    board_after_our_move = board.copy()
    board_after_our_move.push(our_move)

    # Check immediate safety (hanging pieces right after our move)
    immediate_hanging = check_hanging_pieces(board_after_our_move, board.turn)
    immediate_safe = len(immediate_hanging) == 0

    # Generate opponent's candidate responses
    # Prioritize: checks, captures, moves that attack our pieces
    opponent_responses = []

    for opponent_move in board_after_our_move.legal_moves:
        # Categorize response type
        threat_type = 'quiet'

        if board_after_our_move.gives_check(opponent_move):
            threat_type = 'check'
        elif board_after_our_move.is_capture(opponent_move):
            threat_type = 'capture'
        else:
            # Check if move attacks any of our pieces
            test_board = board_after_our_move.copy()
            test_board.push(opponent_move)
            hanging_after = check_hanging_pieces(test_board, board.turn)
            if hanging_after:
                threat_type = 'attack_piece'

        # Only analyze dangerous responses (limit to max_responses)
        if threat_type in ['check', 'capture', 'attack_piece']:
            opponent_responses.append((opponent_move, threat_type))

    # Sort by danger: checks first, then captures, then attacks
    priority_order = {'check': 0, 'capture': 1, 'attack_piece': 2}
    opponent_responses.sort(key=lambda x: priority_order[x[1]])
    opponent_responses = opponent_responses[:max_responses]

    # Analyze each opponent response
    dangerous_responses = []
    worst_material_loss = 0.0
    worst_response = None

    for opponent_move, threat_type in opponent_responses:
        opponent_move_san = board_after_our_move.san(opponent_move)

        # Apply opponent's response
        board_after_response = board_after_our_move.copy()
        board_after_response.push(opponent_move)

        # Check if opponent's move creates hanging pieces for us
        hanging_after_response = check_hanging_pieces(board_after_response, board.turn)

        material_loss = 0.0
        hangs_our_piece = None

        if hanging_after_response:
            # Calculate material loss (value of hanging piece)
            square, piece_type = hanging_after_response[0]  # Take first hanging piece
            material_loss = PIECE_VALUES.get(piece_type, 0)
            hangs_our_piece = (square, piece_type)

        # Create OpponentResponse object
        response = OpponentResponse(
            move=opponent_move,
            move_san=opponent_move_san,
            threat_type=threat_type,
            hangs_our_piece=hangs_our_piece,
            material_loss=material_loss,
            evaluation_drop=0.0  # Will be filled in if evaluator available
        )

        # Track dangerous responses
        if material_loss > 0 or threat_type == 'check':
            dangerous_responses.append(response)

            if material_loss > worst_material_loss:
                worst_material_loss = material_loss
                worst_response = response

    # Determine safety after opponent's best response
    safe_after_response = (worst_response is None)

    # Generate tactical warning
    tactical_warning = None
    if not safe_after_response and worst_response:
        piece_name = chess.piece_name(worst_response.hangs_our_piece[1])
        square_name = chess.square_name(worst_response.hangs_our_piece[0])
        tactical_warning = (
            f"After opponent plays {worst_response.move_san}, "
            f"your {piece_name} on {square_name} hangs! "
            f"(Loses {worst_response.material_loss:.0f} material)"
        )

    # Determine if move should be recommended
    # REFINED THRESHOLD (based on gameplay feedback 2025-11-18):
    # - Only flag as risky if piece worth ≥3 points hangs (minor piece or better)
    # - Allow moves that lose 1-2 pawns (common in openings/tactics)
    # This lets c4 pawn in Queen's Gambit be acceptable while catching Ne5 knight hanging
    recommended = safe_after_response or worst_material_loss < 3.0

    return OpponentResponseAnalysis(
        our_move=our_move,
        our_move_san=our_move_san,
        immediate_safe=immediate_safe,
        safe_after_response=safe_after_response,
        worst_response=worst_response,
        all_dangerous_responses=dangerous_responses,
        tactical_warning=tactical_warning,
        recommended=recommended
    )


def format_response_analysis(analysis: OpponentResponseAnalysis) -> str:
    """
    Format opponent response analysis for display to user.

    Args:
        analysis: OpponentResponseAnalysis to format

    Returns:
        Formatted string for display
    """
    output = []

    output.append(f"Move: {analysis.our_move_san}")
    output.append(f"Immediate Safety: {'[SAFE]' if analysis.immediate_safe else '[HANGS]'}")
    output.append(f"After Opponent Response: {'[SAFE]' if analysis.safe_after_response else '[TACTICAL ISSUE]'}")

    if analysis.tactical_warning:
        output.append(f"\n[!] TACTICAL WARNING:")
        output.append(f"   {analysis.tactical_warning}")

    if analysis.all_dangerous_responses:
        output.append(f"\nOpponent's Dangerous Responses ({len(analysis.all_dangerous_responses)}):")
        for i, response in enumerate(analysis.all_dangerous_responses[:3], 1):
            output.append(f"   {i}. {response.move_san} ({response.threat_type})")
            if response.hangs_our_piece:
                piece_name = chess.piece_name(response.hangs_our_piece[1])
                square_name = chess.square_name(response.hangs_our_piece[0])
                output.append(f"      -> Hangs {piece_name} on {square_name}")

    output.append(f"\nRecommendation: {'[OK] SAFE TO PLAY' if analysis.recommended else '[X] RISKY - Consider alternatives'}")

    return "\n".join(output)


# Convenience function for quick checks
def is_move_safe_after_response(board: chess.Board, move: chess.Move) -> bool:
    """
    Quick check: Is move safe after opponent's best response?

    Args:
        board: Current position
        move: Our candidate move

    Returns:
        True if safe, False if opponent can win material
    """
    analysis = analyze_opponent_responses(board, move)
    return analysis.safe_after_response


# Test function
if __name__ == "__main__":
    # Test with the problematic Ne5 position from Move 14
    print("Testing Opponent Response Analyzer")
    print("=" * 60)

    # Position before Move 14: 1rb1kb1r/ppqppppp/5n2/n2P4/Q7/2N2N2/PP2PPPP/1R2R1K1 w - - 0 14
    board = chess.Board("1rb1kb1r/ppqppppp/5n2/n2P4/Q7/2N2N2/PP2PPPP/1R2R1K1 w - - 0 14")

    print(f"Position:\n{board}\n")

    # Test Ne5 (the move that caused the repetition loop)
    ne5_move = chess.Move.from_uci("f3e5")

    print(f"Testing move: Nf3-e5")
    print("-" * 60)

    analysis = analyze_opponent_responses(board, ne5_move)
    print(format_response_analysis(analysis))

    print("\n" + "=" * 60)

    if not analysis.safe_after_response:
        print("[PASS] CORRECTLY DETECTED: Ne5 is unsafe after opponent's response!")
        print(f"  {analysis.tactical_warning}")
    else:
        print("[FAIL] Did not detect tactical issue with Ne5")
