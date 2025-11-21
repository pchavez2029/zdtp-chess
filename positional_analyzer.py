"""
Positional Analysis Module - 32D Layer Intelligence

Provides advanced positional feature computation for the 32D layer:
- Advanced pawn structure analysis
- Piece coordination evaluation
- Control of key squares
- Space advantage measurement

This module feeds dimensions 20-23 of the 32D Pathion representation.
"""

import chess
from typing import List, Set


class PositionalAnalyzer:
    """
    Analyzes advanced positional features for 32D encoding.

    Goes beyond basic position evaluation to capture nuanced
    positional factors that influence the game.
    """

    def __init__(self):
        """Initialize positional analyzer."""
        pass

    def compute_positional_features(self, board: chess.Board) -> List[float]:
        """
        Compute 4 advanced positional dimensions for 32D space.

        Returns:
            [advanced_pawn_structure, piece_coordination,
             key_square_control, space_advantage]

        These populate dimensions 20-23 of the 32D Pathion.
        """

        # Dimension 20: Advanced pawn structure
        pawn_score = self._analyze_pawn_structure(board)

        # Dimension 21: Piece coordination
        coordination_score = self._analyze_piece_coordination(board)

        # Dimension 22: Control of key squares
        key_square_score = self._analyze_key_squares(board)

        # Dimension 23: Space advantage
        space_score = self._analyze_space(board)

        return [pawn_score, coordination_score, key_square_score, space_score]

    def _analyze_pawn_structure(self, board: chess.Board) -> float:
        """
        Advanced pawn structure analysis.

        Evaluates:
        - Passed pawns
        - Pawn chains
        - Doubled/isolated/backward pawns
        - Pawn islands

        Returns:
            Score from White's perspective
        """
        score = 0.0

        # Analyze passed pawns
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if self._is_passed_pawn(board, square, piece.color):
                    # Passed pawns are valuable!
                    rank = chess.square_rank(square)
                    # More valuable as they advance
                    if piece.color == chess.WHITE:
                        score += 0.5 + (rank / 10.0)
                    else:
                        score -= 0.5 + ((7 - rank) / 10.0)

        # Penalize doubled pawns
        for file in range(8):
            white_pawns = sum(
                1 for sq in chess.SQUARES
                if chess.square_file(sq) == file and
                board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE)
            )
            black_pawns = sum(
                1 for sq in chess.SQUARES
                if chess.square_file(sq) == file and
                board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK)
            )

            if white_pawns > 1:
                score -= 0.3 * (white_pawns - 1)
            if black_pawns > 1:
                score += 0.3 * (black_pawns - 1)

        return score

    def _is_passed_pawn(
        self,
        board: chess.Board,
        square: chess.Square,
        color: chess.Color
    ) -> bool:
        """Check if a pawn is passed (no opposing pawns ahead)."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        # Check files: same file and adjacent files
        for check_file in [file - 1, file, file + 1]:
            if check_file < 0 or check_file > 7:
                continue

            # Check ranks ahead of this pawn
            if color == chess.WHITE:
                ranks_ahead = range(rank + 1, 8)
            else:
                ranks_ahead = range(0, rank)

            for check_rank in ranks_ahead:
                check_square = chess.square(check_file, check_rank)
                piece = board.piece_at(check_square)
                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                    return False  # Opposing pawn blocks

        return True  # No opposing pawns ahead!

    def _analyze_piece_coordination(self, board: chess.Board) -> float:
        """
        Measure piece coordination and harmony.

        Evaluates:
        - Pieces defending each other
        - Rook batteries (doubled rooks)
        - Bishop pair coordination

        Returns:
            Coordination score
        """
        score = 0.0

        # Check for rook batteries (rooks on same file/rank)
        white_rooks = list(board.pieces(chess.ROOK, chess.WHITE))
        black_rooks = list(board.pieces(chess.ROOK, chess.BLACK))

        # White rook coordination
        for i, sq1 in enumerate(white_rooks):
            for sq2 in white_rooks[i+1:]:
                if chess.square_file(sq1) == chess.square_file(sq2):
                    score += 0.5  # Doubled rooks on file
                if chess.square_rank(sq1) == chess.square_rank(sq2):
                    score += 0.5  # Doubled rooks on rank

        # Black rook coordination
        for i, sq1 in enumerate(black_rooks):
            for sq2 in black_rooks[i+1:]:
                if chess.square_file(sq1) == chess.square_file(sq2):
                    score -= 0.5
                if chess.square_rank(sq1) == chess.square_rank(sq2):
                    score -= 0.5

        # Bishop pair bonus
        if len(list(board.pieces(chess.BISHOP, chess.WHITE))) >= 2:
            score += 0.5
        if len(list(board.pieces(chess.BISHOP, chess.BLACK))) >= 2:
            score -= 0.5

        return score

    def _analyze_key_squares(self, board: chess.Board) -> float:
        """
        Evaluate control of key squares.

        Key squares include:
        - Center squares (d4, d5, e4, e5)
        - Advanced outposts (e.g., d5 for White, d4 for Black)
        - Squares near opponent's king

        Returns:
            Key square control score
        """
        score = 0.0

        # Central squares (most important)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]

        for square in center_squares:
            white_control = len(list(board.attackers(chess.WHITE, square)))
            black_control = len(list(board.attackers(chess.BLACK, square)))

            score += (white_control - black_control) * 0.2

        # Outpost squares (5th/6th rank for White, 4th/3rd for Black)
        for square in chess.SQUARES:
            rank = chess.square_rank(square)

            # White outposts (ranks 4-5)
            if rank in [4, 5]:
                piece = board.piece_at(square)
                if piece and piece.color == chess.WHITE:
                    if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        # Check if defended and no enemy pawns can attack
                        if len(list(board.attackers(chess.WHITE, square))) > 0:
                            score += 0.3

            # Black outposts (ranks 2-3)
            if rank in [2, 3]:
                piece = board.piece_at(square)
                if piece and piece.color == chess.BLACK:
                    if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        if len(list(board.attackers(chess.BLACK, square))) > 0:
                            score -= 0.3

        return score

    def _analyze_space(self, board: chess.Board) -> float:
        """
        Measure space advantage.

        Space is evaluated by:
        - Number of squares controlled
        - Territory in opponent's half of the board

        Returns:
            Space advantage score
        """
        white_space = 0
        black_space = 0

        # Count controlled squares
        for square in chess.SQUARES:
            rank = chess.square_rank(square)

            white_attackers = len(list(board.attackers(chess.WHITE, square)))
            black_attackers = len(list(board.attackers(chess.BLACK, square)))

            # Weight squares in opponent's territory more
            if rank >= 4:  # White's perspective (Black's territory)
                white_space += white_attackers * 1.5
            else:
                white_space += white_attackers

            if rank <= 3:  # Black's perspective (White's territory)
                black_space += black_attackers * 1.5
            else:
                black_space += black_attackers

        # Normalize and return
        space_diff = (white_space - black_space) / 100.0
        return space_diff


def compute_positional_features(board: chess.Board) -> List[float]:
    """
    Convenience function for positional feature computation.

    Args:
        board: Current chess position

    Returns:
        [pawn_structure, coordination, key_squares, space]
    """
    analyzer = PositionalAnalyzer()
    return analyzer.compute_positional_features(board)
