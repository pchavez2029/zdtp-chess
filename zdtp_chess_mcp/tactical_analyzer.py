"""
Tactical Analysis Module - 32D Layer Intelligence

Provides tactical feature computation for the 32D positional layer:
- Static Exchange Evaluation (SEE)
- Hanging piece detection
- Pin and skewer analysis
- Fork threat detection

This module feeds dimensions 16-19 of the 32D Pathion representation.
"""

import chess
from typing import List, Dict, Optional, Tuple


class TacticalAnalyzer:
    """
    Analyzes tactical features of chess positions for 32D encoding.

    This is where the ZDTP gets tactical intelligence to prevent
    blunders like Rxd6 (trading rook for pawn).
    """

    # Piece values for exchange evaluation
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.2,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0  # King value is infinite, but 0 for SEE
    }

    def __init__(self):
        """Initialize tactical analyzer."""
        pass

    def compute_tactical_features(self, board: chess.Board) -> List[float]:
        """
        Compute 4 tactical dimensions for 32D space.

        Returns:
            [hanging_pieces, pin_analysis, fork_threats, exchange_eval]

        These populate dimensions 16-19 of the 32D Pathion.
        """

        # Dimension 16: Hanging pieces (undefended pieces under attack)
        hanging_score = self._analyze_hanging_pieces(board)

        # Dimension 17: Pins and skewers
        pin_score = self._analyze_pins(board)

        # Dimension 18: Fork threats (especially knight forks!)
        fork_score = self._analyze_fork_threats(board)

        # Dimension 19: Static Exchange Evaluation (SEE)
        # THIS IS THE CRITICAL ONE - would have prevented Rxd6!
        exchange_score = self._static_exchange_evaluation(board)

        return [hanging_score, pin_score, fork_score, exchange_score]

    def _get_piece_value(self, piece: Optional[chess.Piece]) -> float:
        """Get material value of a piece."""
        if piece is None:
            return 0.0
        return self.PIECE_VALUES.get(piece.piece_type, 0.0)

    def _analyze_hanging_pieces(self, board: chess.Board) -> float:
        """
        Detect hanging (undefended) pieces.

        Returns:
            Negative score if pieces are hanging (from White's perspective)
        """
        hanging_penalty = 0.0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Check if piece is attacked
            attackers = list(board.attackers(not piece.color, square))
            if len(attackers) == 0:
                continue  # Not under attack

            # Check if piece is defended
            defenders = list(board.attackers(piece.color, square))

            if len(defenders) == 0:
                # HANGING PIECE!
                piece_value = self._get_piece_value(piece)
                if piece.color == chess.WHITE:
                    hanging_penalty -= piece_value
                else:
                    hanging_penalty += piece_value

        return hanging_penalty

    def _analyze_pins(self, board: chess.Board) -> float:
        """
        Analyze pins and skewers.

        Returns:
            Score reflecting pin advantage/disadvantage
        """
        pin_score = 0.0

        # Check for pins along ranks, files, and diagonals
        # This is simplified - full implementation would check all directions

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                continue

            # Check if this piece is pinning an opponent piece
            # (Simplified implementation)
            attacks = board.attacks(square)
            for target_sq in attacks:
                target = board.piece_at(target_sq)
                if target and target.color != piece.color:
                    # Potential pin - would need to verify king is behind
                    if piece.color == chess.WHITE:
                        pin_score += 0.5
                    else:
                        pin_score -= 0.5

        return pin_score

    def _analyze_fork_threats(self, board: chess.Board) -> float:
        """
        Analyze fork threats (especially knights!).

        Returns:
            Score reflecting fork advantage
        """
        fork_score = 0.0

        # Check for knight forks (the most common and dangerous)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type != chess.KNIGHT:
                continue

            # Get all squares the knight attacks
            attacks = board.attacks(square)

            # Count valuable pieces attacked simultaneously
            attacked_pieces = []
            for target_sq in attacks:
                target = board.piece_at(target_sq)
                if target and target.color != piece.color:
                    attacked_pieces.append(target)

            # If attacking 2+ pieces, it's a fork!
            if len(attacked_pieces) >= 2:
                # Value the fork by the minimum piece value
                # (since you can only capture one)
                fork_value = min(self._get_piece_value(p) for p in attacked_pieces)

                if piece.color == chess.WHITE:
                    fork_score += fork_value * 0.5
                else:
                    fork_score -= fork_value * 0.5

        return fork_score

    def _static_exchange_evaluation(self, board: chess.Board) -> float:
        """
        Static Exchange Evaluation (SEE) - THE CRITICAL FEATURE!

        This evaluates all possible captures and their recapture sequences.
        This would have prevented Rxd6 (rook takes pawn, bishop recaptures).

        Returns:
            Most negative exchange value for the side to move.
            Negative value means a bad exchange exists!
        """
        worst_exchange = 0.0

        # Evaluate all capture moves
        for move in board.legal_moves:
            if board.is_capture(move):
                # Evaluate the exchange sequence for this capture
                exchange_value = self._evaluate_exchange_sequence(board, move)

                # Track the worst exchange
                if exchange_value < worst_exchange:
                    worst_exchange = exchange_value

        return worst_exchange

    def _evaluate_exchange_sequence(
        self,
        board: chess.Board,
        capture_move: chess.Move
    ) -> float:
        """
        Evaluate a single capture sequence using minimax.

        Example: Rxd6
        1. Rook (5) takes pawn (1) on d6
        2. Bishop (3) recaptures rook (5) on d6
        Result: Lost 5, gained 1 = -4 (BAD!)

        Returns:
            Net material change from the side to move's perspective
            (Negative = bad exchange for the moving side)
        """

        # Remember whose turn it is (they're making the capture)
        moving_side = board.turn

        # Make the capture
        captured_piece = board.piece_at(capture_move.to_square)
        capturing_piece = board.piece_at(capture_move.from_square)

        if captured_piece is None or capturing_piece is None:
            return 0.0

        # Initial gain: value of captured piece
        gain = [self._get_piece_value(captured_piece)]

        # Simulate the exchange sequence
        board.push(capture_move)

        # Find if opponent can recapture
        recapturers = list(board.attackers(
            not moving_side,  # Opponent's color
            capture_move.to_square
        ))

        if recapturers:
            # Find cheapest recapturer
            cheapest_recapturer = min(
                recapturers,
                key=lambda sq: self._get_piece_value(board.piece_at(sq))
            )

            # Value of piece that was captured (now being recaptured)
            recapture_value = self._get_piece_value(
                board.piece_at(capture_move.to_square)
            )
            gain.append(recapture_value)

            # Could continue deeper, but 2-ply is sufficient for most cases

        board.pop()

        # Calculate net gain (alternating signs for each side)
        # gain[0] = material gained by moving side
        # gain[1] = material lost to recapture
        # gain[2] = material gained by re-recapture, etc.
        net_gain = sum(gain[i] if i % 2 == 0 else -gain[i]
                      for i in range(len(gain)))

        # net_gain is already from the moving side's perspective
        # No additional adjustment needed!

        return net_gain


def compute_tactical_features(board: chess.Board) -> List[float]:
    """
    Convenience function for tactical feature computation.

    Args:
        board: Current chess position

    Returns:
        [hanging_pieces, pins, forks, exchange_eval]
    """
    analyzer = TacticalAnalyzer()
    return analyzer.compute_tactical_features(board)
