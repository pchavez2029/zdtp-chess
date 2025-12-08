"""
ZDTP Chess Engine - Move Selection via Dimensional Reasoning

Uses the Zero Divisor Traversal Protocol to evaluate candidate moves
across 16D (tactical), 32D (positional), and 64D (strategic) spaces.

The engine:
1. Generates legal moves
2. For each move, encodes resulting position
3. Cascades through dimensions using appropriate gateway
4. Evaluates at all three levels
5. Selects move with best dimensional consensus

This is the first chess engine that uses zero divisor hyperwormholes
for position evaluation!
"""

import chess
import time
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from .dimensional_encoder import encode_position
    from .dimensional_portal import full_cascade
    from .multidimensional_evaluator import evaluate_position, MultiDimensionalAnalysis, TacticalEvaluation, PositionalEvaluation, StrategicEvaluation
    from .gateway_patterns import PIECE_TO_PATTERN_ID
    from .path_verifier import filter_legal_moves  # PHASE 0: Path verification
except ImportError:
    from dimensional_encoder import encode_position
    from dimensional_portal import full_cascade
    from multidimensional_evaluator import evaluate_position, MultiDimensionalAnalysis, TacticalEvaluation, PositionalEvaluation, StrategicEvaluation
    from gateway_patterns import PIECE_TO_PATTERN_ID
    from path_verifier import filter_legal_moves  # PHASE 0: Path verification


# ============================================================================
# Static Exchange Evaluation (SEE) - Integrated from emergency_hotfix.py
# Battle-tested in victory game 2025-11-16 (prevented 16 blunders)
# ============================================================================

def _emergency_see_safety_check(board: chess.Board, move: chess.Move) -> Dict:
    """
    Quick 2-ply Static Exchange Evaluation safety check.

    Evaluates whether a move loses material through recapture. Handles both
    captures and quiet moves that land on attacked squares.

    This is the battle-tested code that prevented 16 blunders in the
    2025-11-16 victory game. Integrated directly into engine for performance
    and code cleanliness.

    Args:
        board: Current position
        move: Move to check

    Returns:
        dict with 'is_safe', 'see_value', and 'warning' fields
    """
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.2,
        chess.BISHOP: 3.3,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0
    }

    target_square = move.to_square

    # BUG FIX (2025-11-15): Also check quiet moves that land on attacked squares
    # This prevents Qb5-style blunders where piece moves to attacked square
    is_capture = board.is_capture(move)

    if not is_capture:
        # For quiet moves: check if destination is attacked
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return {'is_safe': True, 'see_value': 0.0, 'warning': None}

        # Make move on copy to see if piece hangs
        board_copy = board.copy()
        board_copy.push(move)

        # Check if destination square is attacked by opponent
        opponent_attackers = list(board_copy.attackers(board_copy.turn, target_square))

        if not opponent_attackers:
            # Not attacked - safe quiet move
            return {'is_safe': True, 'see_value': 0.0, 'warning': None}

        # Piece lands on attacked square! Evaluate exchange
        # (fall through to SEE logic below)
        captured_value = 0.0  # Quiet move, no immediate capture
    else:
        # Capture move - get captured piece value
        captured_piece = board.piece_at(target_square)
        if not captured_piece:
            return {'is_safe': True, 'see_value': 0.0, 'warning': None}

        captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)

    # Get moving piece value
    if not is_capture:
        # For quiet moves, we already have moving_piece from above
        pass
    else:
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return {'is_safe': True, 'see_value': 0.0, 'warning': None}

    moving_piece_value = PIECE_VALUES.get(moving_piece.piece_type, 0)

    # Quick 2-ply SEE: Can opponent recapture?
    if not is_capture:
        # For quiet moves, we already made the board_copy above
        pass
    else:
        board_copy = board.copy()
        board_copy.push(move)
        opponent_attackers = list(board_copy.attackers(board_copy.turn, target_square))

    if opponent_attackers:
        # Net SEE: gained - lost
        see_value = captured_value - moving_piece_value

        # Find cheapest recapturer
        cheapest_recapturer_value = 999.0
        for attacker_square in opponent_attackers:
            piece = board_copy.piece_at(attacker_square)
            if piece:
                piece_value = PIECE_VALUES.get(piece.piece_type, 0)
                if piece_value < cheapest_recapturer_value:
                    cheapest_recapturer_value = piece_value

        is_safe = see_value >= -1.0

        warning = None
        if is_capture:
            if see_value < -5.0:
                warning = f"CRITICAL: Hangs {chess.piece_name(moving_piece.piece_type)} (loses {abs(see_value):.1f} material)"
            elif see_value < -2.0:
                warning = f"WARNING: Unfavorable exchange (loses {abs(see_value):.1f} material)"
        else:
            # Quiet move landing on attacked square
            if see_value < -5.0:
                warning = f"CRITICAL: {chess.piece_name(moving_piece.piece_type)} hangs on {chess.square_name(target_square)} (loses {abs(see_value):.1f} material)"
            elif see_value < -2.0:
                warning = f"WARNING: {chess.piece_name(moving_piece.piece_type)} poorly placed on {chess.square_name(target_square)} (loses {abs(see_value):.1f} material)"

        details = {
            'moving_piece': chess.piece_name(moving_piece.piece_type),
            'can_recapture': True,
            'move_type': 'capture' if is_capture else 'quiet'
        }
        if is_capture:
            details['captured'] = chess.piece_name(captured_piece.piece_type)

        return {
            'is_safe': is_safe,
            'see_value': see_value,
            'warning': warning,
            'details': details
        }
    else:
        # Safe capture - no recapture
        return {
            'is_safe': True,
            'see_value': captured_value,
            'warning': None,
            'details': {
                'captured': chess.piece_name(captured_piece.piece_type),
                'can_recapture': False
            }
        }


# ============================================================================
# Engine Data Structures
# ============================================================================

@dataclass
class MoveAnalysis:
    """Analysis of a candidate move."""
    move: chess.Move
    move_san: str
    analysis: MultiDimensionalAnalysis
    evaluation_time_ms: float


@dataclass
class EngineResponse:
    """Engine's response with selected move and analysis."""
    best_move: chess.Move
    best_move_san: str
    analysis: MultiDimensionalAnalysis
    candidates_evaluated: int
    total_time_ms: float
    gateway_used: str
    all_candidates: List[MoveAnalysis]  # For debugging/display


class ZDTPEngine:
    """
    Chess engine using Zero Divisor Traversal Protocol.    
    The engine evaluates positions by cascading them through dimensional
    spaces (16D→32D→64D) and using multi-dimensional consensus to select moves.
    """
    
    def __init__(
        self,
        time_limit_ms: int = 5000,
        max_candidates: int = 20,
        gateway_strategy: str = 'adaptive'
    ):
        """
        Initialize ZDTP engine.        
        Args:
            time_limit_ms: Maximum time per move in milliseconds
            max_candidates: Maximum candidate moves to evaluate
            gateway_strategy: 'adaptive', 'king', 'queen', etc.
        """
        self.time_limit_ms = time_limit_ms
        self.max_candidates = max_candidates
        self.gateway_strategy = gateway_strategy
        
        # Statistics
        self.positions_evaluated = 0
        self.total_evaluation_time = 0.0
    
    def select_move(self, board: chess.Board) -> EngineResponse:
        """
        Select the best move for current position.
        
        Args:
            board: Current chess board position            
        Returns:
            EngineResponse with selected move and analysis
        """
        start_time = time.perf_counter()
        
        if board.is_game_over():
            # Game is already over (checkmate, stalemate, etc.)
            # Return a dummy response indicating no moves
            dummy_analysis = MultiDimensionalAnalysis(
                tactical_16d=TacticalEvaluation(0.0, 0.0, 0.0, 0.0, [], None, "Game over."),
                positional_32d=PositionalEvaluation(0.0, 0.0, 0.0, 0.0, 0.0, None, "Game over."),
                strategic_64d=StrategicEvaluation(0.0, 0.0, 0.0, 0.0, 0.0, None, "Game over."),
                consensus_score=0.0,
                recommended_move=None,
                gateway_used="N/A",
                overall_assessment="Game is over."
            )
            return EngineResponse(
                best_move=None,
                best_move_san="N/A",
                analysis=dummy_analysis,
                candidates_evaluated=0,
                total_time_ms=0.0,
                gateway_used="N/A",
                all_candidates=[]
            )
        
        # Generate candidate moves
        candidates = self._generate_candidates(board)
        
        # Determine gateway to use
        gateway_piece = self._select_gateway(board)
        
        # Evaluate each candidate
        move_analyses = []
        for move in candidates:
            try:
                analysis = self._evaluate_move(board, move, gateway_piece)
                move_analyses.append(analysis)
            except Exception as e:
                print(f"Error evaluating move {board.san(move)}: {e}", file=sys.stderr)
            
            # Check time limit
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > self.time_limit_ms:
                break
        
        # Handle case where no moves could be analyzed
        if not move_analyses:
            raise ValueError("No legal moves could be analyzed within time limit or due to errors.")

        # INTEGRATION: Filter out tactical blunders using battle-tested SEE logic
        # This prevented 16 blunders in the 2025-11-16 victory game
        move_analyses = self._detect_blunders(board, move_analyses)

        # Select best move (from safe moves only)
        best_analysis = max(move_analyses, key=lambda ma: ma.analysis.consensus_score)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Update statistics
        self.positions_evaluated += len(move_analyses)
        self.total_evaluation_time += total_time
        
        return EngineResponse(
            best_move=best_analysis.move,
            best_move_san=best_analysis.move_san,
            analysis=best_analysis.analysis,
            candidates_evaluated=len(move_analyses),
            total_time_ms=total_time,
            gateway_used=chess.piece_name(gateway_piece),
            all_candidates=move_analyses
        )
    
    def _generate_candidates(self, board: chess.Board) -> List[chess.Move]:
        """
        Generate candidate moves to evaluate.

        PHASE 0: Uses path-verified legal moves to prevent illegal recommendations

        Strategy:
        1. If in check, evaluate all legal moves
        2. Otherwise, prioritize:
           - Captures
           - Checks
           - Central moves
           - Development moves
        3. Limit to max_candidates
        """
        # PHASE 0: Filter out moves with blocked paths or illegal patterns
        legal_moves = filter_legal_moves(board)

        # BUG #1 FIX: If in check, must evaluate ALL legal moves (don't truncate)
        # Truncating could cut off the only legal escape moves!
        if board.is_check():
            return legal_moves  # Return ALL moves, no limit
        
        # Prioritize captures and checks
        captures = []
        checks = []
        quiet_moves = []
        
        for move in legal_moves:
            if board.is_capture(move):
                captures.append(move)
            else:
                # Test if move gives check on a copy of the board
                board_after_move = board.copy()
                board_after_move.push(move)
                if board_after_move.is_check():
                    checks.append(move)
                else:
                    quiet_moves.append(move)
        
        # Combine prioritized moves
        candidates = captures + checks + quiet_moves
        
        return candidates[:self.max_candidates]
    
    def _select_gateway(self, board: chess.Board) -> chess.PieceType:
        """
        Select which gateway pattern to use for evaluation.        
        Strategies:
        - 'adaptive': Choose based on position type
        - 'king', 'queen', etc.: Always use that piece
        """
        if self.gateway_strategy == 'adaptive':
            # Adaptive strategy: choose gateway based on position            
            # Count material
            piece_count = len(board.piece_map())
            
            # Endgame (< 16 pieces): Use King gateway
            if piece_count < 16:
                return chess.KING
            
            # Opening (most pieces on back rank): Use Knight gateway
            elif piece_count > 28:
                return chess.KNIGHT
            
            # Check if queens on board
            white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
            black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
            
            if white_queens > 0 or black_queens > 0:
                # Queens present: Use Queen gateway
                return chess.QUEEN
            else:
                # No queens: Use Rook gateway
                return chess.ROOK
        
        else:
            # Fixed gateway strategy
            gateway_map = {
                'king': chess.KING,
                'queen': chess.QUEEN,
                'knight': chess.KNIGHT,
                'bishop': chess.BISHOP,
                'rook': chess.ROOK,
                'pawn': chess.PAWN
            }
            return gateway_map.get(self.gateway_strategy, chess.KNIGHT)
    
    def _evaluate_move(
        self,
        board: chess.Board,
        move: chess.Move,
        gateway_piece: chess.PieceType
    ) -> MoveAnalysis:
        """
        Evaluate a candidate move using ZDTP cascade.        
        Args:
            board: Current board position
            move: Candidate move to evaluate
            gateway_piece: Gateway pattern to use            
        Returns:
            MoveAnalysis with dimensional evaluation
        """
        eval_start = time.perf_counter()
        
        # Make move on copy of board
        board_copy = board.copy()
        board_copy.push(move)
        
        # Encode resulting position
        state_16d = encode_position(board_copy)

        # Cascade through dimensions (with intelligent tactical/strategic analysis!)
        cascade = full_cascade(state_16d, gateway_piece, board_copy)

        # Evaluate at all dimensional levels
        analysis = evaluate_position(cascade, board_copy)
        
        # Negate score if black to move (want score from current player's perspective)
        if board.turn == chess.BLACK:
            # Flip scores for black's perspective
            analysis = self._flip_perspective(analysis)
        
        eval_time = (time.perf_counter() - eval_start) * 1000
        
        return MoveAnalysis(
            move=move,
            move_san=board.san(move),
            analysis=analysis,
            evaluation_time_ms=eval_time
        )
    
    def _flip_perspective(self, analysis: MultiDimensionalAnalysis) -> MultiDimensionalAnalysis:
        """
        Flip evaluation scores for black's perspective.        
        ZDTP encodes positions from white's perspective.
        When black is to move, we need to negate scores.
        """
        # Negate tactical score
        analysis.tactical_16d.score *= -1
        analysis.tactical_16d.material_balance *= -1
        analysis.tactical_16d.king_safety_diff *= -1
        analysis.tactical_16d.mobility_diff *= -1
        
        # Negate positional score
        analysis.positional_32d.score *= -1
        analysis.positional_32d.pawn_structure *= -1
        analysis.positional_32d.center_control *= -1
        
        # Negate strategic score
        analysis.strategic_64d.score *= -1
        analysis.strategic_64d.pawn_advancement *= -1
        
        # Negate consensus
        analysis.consensus_score *= -1

        return analysis

    def _detect_blunders(
        self,
        board: chess.Board,
        move_analyses: List[MoveAnalysis]
    ) -> List[MoveAnalysis]:
        """
        Filter out moves that would be tactical blunders.

        Uses battle-tested SEE logic from emergency_hotfix.py (prevented 16 blunders
        in victory game 2025-11-16).

        Args:
            board: Current board position
            move_analyses: List of candidate moves with evaluations

        Returns:
            Filtered list with blunders removed (or all moves if all are blunders)
        """
        safe_moves = []
        blunders = []

        for analysis in move_analyses:
            # Run emergency SEE safety check (now integrated internally)
            safety_result = _emergency_see_safety_check(board, analysis.move)

            if safety_result['is_safe']:
                safe_moves.append(analysis)
            else:
                blunders.append((analysis, safety_result))

        # If ALL moves are blunders, return the "least bad" one
        # (Better to lose a pawn than lose the queen!)
        if not safe_moves:
            # Sort blunders by SEE value (least negative = least bad)
            blunders.sort(key=lambda x: x[1]['see_value'], reverse=True)
            print(f"WARNING: All moves are blunders! Choosing least bad: {blunders[0][0].move_san} (SEE: {blunders[0][1]['see_value']:.1f})", file=sys.stderr)
            return [blunders[0][0]]  # Return least bad blunder

        # Log filtered blunders
        if blunders:
            print(f"Filtered {len(blunders)} blunders:", file=sys.stderr)
            for analysis, safety in blunders:
                print(f"  - {analysis.move_san}: {safety['warning']}", file=sys.stderr)

        return safe_moves

    def get_statistics(self) -> Dict:
        """Get engine statistics."""
        return {
            'positions_evaluated': self.positions_evaluated,
            'total_time_ms': self.total_evaluation_time,
            'avg_time_per_position': (
                self.total_evaluation_time / self.positions_evaluated
                if self.positions_evaluated > 0 else 0
            )
        }


# Module-level convenience function

_engine = None

def get_best_move(
    board: chess.Board,
    time_limit_ms: int = 5000,
    gateway_strategy: str = 'adaptive'
) -> EngineResponse:
    """
    Get best move for position using ZDTP engine.    
    Convenience function for quick move selection.
    """
    engine = ZDTPEngine(
        time_limit_ms=time_limit_ms,
        gateway_strategy=gateway_strategy
    )
    return engine.select_move(board)


# Standalone test
if __name__ == "__main__":
    print("=" * 60)
    print("ZDTP CHESS ENGINE TEST")
    print("=" * 60)
    print()
    
    # Test game
    board = chess.Board()
    engine = ZDTPEngine(time_limit_ms=10000, max_candidates=10)
    
    print("Starting position:")
    print(board)
    print()
    
    # Get engine's first move
    print("Engine thinking...")
    response = engine.select_move(board)
    
    print(f"\nEngine selected: {response.best_move_san}")
    print(f"Evaluation: {response.analysis.consensus_score:.2f}")
    print(f"Gateway used: {response.gateway_used}")
    print(f"Candidates evaluated: {response.candidates_evaluated}")
    print(f"Time: {response.total_time_ms:.0f}ms")
    
    print(f"\n16D Tactical: {response.analysis.tactical_16d.reasoning}")
    print(f"32D Positional: {response.analysis.positional_32d.reasoning}")
    print(f"64D Strategic: {response.analysis.strategic_64d.reasoning}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
