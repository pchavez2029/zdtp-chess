"""
Strategic Analysis Module - 64D Layer Intelligence

Provides strategic feature computation for the 64D layer:
- Multi-move tactical sequences (2-3 ply)
- Game phase recognition
- Strategic imbalance analysis
- Gateway harmony measurement

This module feeds dimensions 32-63 of the 64D Chingon representation.
"""

import chess
from typing import List, Dict, Optional

try:
    from hypercomplex import Sedenion, Pathion, Chingon
except ImportError:
    # Fallback if hypercomplex not available
    pass


class StrategicAnalyzer:
    """
    Analyzes strategic features for 64D encoding.

    This is where the engine develops grandmaster-level strategic
    understanding through multi-dimensional reasoning.
    """

    # Piece values
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.2,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0
    }

    def __init__(self):
        """Initialize strategic analyzer."""
        pass

    def compute_strategic_features(
        self,
        board: chess.Board,
        state_32d: Optional['Pathion'] = None,
        gateway_32d: Optional[List[float]] = None,
        gateway_64d: Optional[List[float]] = None
    ) -> List[float]:
        """
        Compute 32 strategic dimensions for 64D space.

        Returns a list of 32 floats for dimensions 32-63.

        Args:
            board: Current position
            state_32d: 32D state (for gateway harmony)
            gateway_32d: Gateway features from 16→32 transition
            gateway_64d: Gateway features from 32→64 transition

        Returns:
            List of 32 strategic feature values
        """

        features = []

        # Dimensions 32-35: Multi-move sequences
        multi_move_features = self._analyze_multi_move_sequences(board)
        features.extend(multi_move_features)  # 4 values

        # Dimensions 36-39: Strategic planning
        plan_features = self._analyze_strategic_plans(board)
        features.extend(plan_features)  # 4 values

        # Dimensions 40-43: Game phase recognition
        phase_features = self._recognize_game_phase(board)
        features.extend(phase_features)  # 4 values

        # Dimensions 44-47: Positional patterns (placeholder)
        pattern_features = [0.0, 0.0, 0.0, 0.0]
        features.extend(pattern_features)  # 4 values

        # Dimensions 48-51: Long-term factors (placeholder)
        longterm_features = [0.0, 0.0, 0.0, 0.0]
        features.extend(longterm_features)  # 4 values

        # Dimensions 52-55: Strategic imbalances
        imbalance_features = self._analyze_strategic_imbalances(board)
        features.extend(imbalance_features)  # 4 values

        # Dimensions 56-59: Gateway harmony
        if gateway_32d and gateway_64d:
            harmony_features = self._compute_gateway_harmony(
                gateway_32d, gateway_64d, board
            )
        else:
            harmony_features = [0.0, 0.0, 0.0, 0.0]
        features.extend(harmony_features)  # 4 values

        # Dimensions 60-63: Meta-cognitive features
        meta_features = self._compute_meta_features(board)
        features.extend(meta_features)  # 4 values

        return features

    def _analyze_multi_move_sequences(self, board: chess.Board) -> List[float]:
        """
        Analyze tactical sequences 2-3 moves deep.

        THIS IS CRITICAL - would catch Rxd6 in context!

        Returns:
            [two_move_score, three_move_score, forcing_seq, quiet_potential]
        """

        # Get candidate moves (top 5 by simple heuristic)
        candidates = self._get_candidate_moves(board, max_moves=5)

        two_move_best = -999.0
        three_move_best = -999.0
        forcing_value = 0.0

        for move in candidates:
            # Evaluate 2-move sequence
            two_score = self._evaluate_2move_sequence(board, move)
            two_move_best = max(two_move_best, two_score)

            # Evaluate 3-move sequence (more expensive)
            three_score = self._evaluate_3move_sequence(board, move)
            three_move_best = max(three_move_best, three_score)

            # Track forcing moves (checks, captures)
            if board.gives_check(move) or board.is_capture(move):
                forcing_value += 0.5

        # Quiet move potential (non-forcing moves)
        quiet_potential = len(candidates) - forcing_value

        return [two_move_best, three_move_best, forcing_value, quiet_potential]

    def _evaluate_2move_sequence(
        self,
        board: chess.Board,
        move: chess.Move
    ) -> float:
        """Evaluate position after move + best opponent response."""
        board.push(move)

        # Find opponent's best tactical response
        opp_best_score = 999.0
        for opp_move in list(board.legal_moves)[:10]:  # Top 10 responses
            board.push(opp_move)
            score = self._simple_eval(board)
            opp_best_score = min(opp_best_score, score)
            board.pop()

        board.pop()
        return opp_best_score

    def _evaluate_3move_sequence(
        self,
        board: chess.Board,
        move: chess.Move
    ) -> float:
        """Evaluate position 3 moves deep."""
        board.push(move)

        best_score = -999.0
        for opp_move in list(board.legal_moves)[:5]:  # Top 5 opponent moves
            board.push(opp_move)

            # Your reply
            for reply in list(board.legal_moves)[:5]:
                board.push(reply)
                score = self._simple_eval(board)
                best_score = max(best_score, score)
                board.pop()

            board.pop()

        board.pop()
        return best_score

    def _get_candidate_moves(
        self,
        board: chess.Board,
        max_moves: int = 10
    ) -> List[chess.Move]:
        """Get candidate moves sorted by simple heuristic."""
        moves = list(board.legal_moves)

        # Prioritize: checks > captures > other
        def move_priority(move):
            score = 0
            if board.gives_check(move):
                score += 100
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    score += self.PIECE_VALUES.get(captured.piece_type, 0) * 10
            return -score  # Negative for descending sort

        moves.sort(key=move_priority)
        return moves[:max_moves]

    def _simple_eval(self, board: chess.Board) -> float:
        """Simple material-based evaluation."""
        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        return score

    def _analyze_strategic_plans(self, board: chess.Board) -> List[float]:
        """
        Analyze long-term strategic plans.

        Returns:
            [piece_majority, pawn_breakthrough, maneuvering, prophylaxis]
        """

        # Simplified placeholder
        piece_majority = 0.0
        pawn_breakthrough = 0.0
        maneuvering = 0.0
        prophylaxis = 0.0

        return [piece_majority, pawn_breakthrough, maneuvering, prophylaxis]

    def _recognize_game_phase(self, board: chess.Board) -> List[float]:
        """
        Recognize game phase and evaluate accordingly.

        Returns:
            [opening_score, middlegame_score, endgame_score, transition_value]
        """

        piece_count = len(board.piece_map())
        queen_count = (
            len(list(board.pieces(chess.QUEEN, chess.WHITE))) +
            len(list(board.pieces(chess.QUEEN, chess.BLACK)))
        )

        opening_score = 0.0
        middlegame_score = 0.0
        endgame_score = 0.0
        transition_value = 0.0

        # Opening phase (many pieces, queens on board)
        if piece_count > 28 and queen_count == 2:
            opening_score = 1.0
            # Evaluate development
            white_developed = self._count_developed_pieces(board, chess.WHITE)
            black_developed = self._count_developed_pieces(board, chess.BLACK)
            opening_score += (white_developed - black_developed) * 0.3

        # Endgame (few pieces or no queens)
        elif piece_count < 14 or queen_count == 0:
            endgame_score = 1.0
            # King activity matters in endgame
            white_king_sq = board.king(chess.WHITE)
            black_king_sq = board.king(chess.BLACK)
            if white_king_sq and black_king_sq:
                # Centralized king is good in endgame
                white_king_centralization = self._king_centralization(white_king_sq)
                black_king_centralization = self._king_centralization(black_king_sq)
                endgame_score += (white_king_centralization - black_king_centralization) * 0.5

        # Middlegame
        else:
            middlegame_score = 1.0

        return [opening_score, middlegame_score, endgame_score, transition_value]

    def _count_developed_pieces(self, board: chess.Board, color: chess.Color) -> int:
        """Count developed minor pieces."""
        developed = 0
        back_rank = 0 if color == chess.WHITE else 7

        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for square in board.pieces(piece_type, color):
                if chess.square_rank(square) != back_rank:
                    developed += 1

        return developed

    def _king_centralization(self, king_square: chess.Square) -> float:
        """Measure how centralized the king is (0-1)."""
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)

        # Distance from center (3.5, 3.5)
        dist_from_center = abs(file - 3.5) + abs(rank - 3.5)
        # Normalize to 0-1 (max distance is 7)
        centralization = 1.0 - (dist_from_center / 7.0)
        return centralization

    def _analyze_strategic_imbalances(self, board: chess.Board) -> List[float]:
        """
        Analyze strategic compensation and imbalances.

        Steinitz theory: material, time, space, structure can compensate.

        Returns:
            [space_vs_material, activity_vs_structure,
             attack_vs_defense, time_vs_position]
        """

        material = self._simple_eval(board) / 10.0
        space = self._space_evaluation(board)
        activity = self._activity_evaluation(board)
        structure = self._structure_evaluation(board)

        space_vs_mat = space - material
        activity_vs_struct = activity - structure
        attack_vs_def = 0.0  # Simplified
        time_vs_pos = 0.0  # Simplified

        return [space_vs_mat, activity_vs_struct, attack_vs_def, time_vs_pos]

    def _space_evaluation(self, board: chess.Board) -> float:
        """Evaluate space control."""
        white_space = sum(
            len(list(board.attackers(chess.WHITE, sq))) for sq in chess.SQUARES
        )
        black_space = sum(
            len(list(board.attackers(chess.BLACK, sq))) for sq in chess.SQUARES
        )
        return (white_space - black_space) / 100.0

    def _activity_evaluation(self, board: chess.Board) -> float:
        """Evaluate piece activity."""
        white_mobility = sum(
            1 for move in board.legal_moves if board.turn == chess.WHITE
        )
        board.push(chess.Move.null())
        black_mobility = sum(
            1 for move in board.legal_moves if board.turn == chess.BLACK
        )
        board.pop()
        return (white_mobility - black_mobility) / 20.0

    def _structure_evaluation(self, board: chess.Board) -> float:
        """Evaluate pawn structure quality."""
        # Simplified: count doubled pawns as penalty
        score = 0.0
        for file in range(8):
            white_count = sum(
                1 for sq in chess.SQUARES
                if chess.square_file(sq) == file and
                board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE)
            )
            black_count = sum(
                1 for sq in chess.SQUARES
                if chess.square_file(sq) == file and
                board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK)
            )
            if white_count > 1:
                score -= 0.2
            if black_count > 1:
                score += 0.2
        return score

    def _compute_gateway_harmony(
        self,
        gateway_32d: List[float],
        gateway_64d: List[float],
        board: chess.Board
    ) -> List[float]:
        """
        Measure harmony between dual gateways.

        Returns:
            [harmony, strategic_bias, resonance, pattern_match]
        """

        # Compute correlation between gateways
        if len(gateway_32d) < 4 or len(gateway_64d) < 4:
            return [0.0, 0.0, 0.0, 0.0]

        # Use first 4 elements for harmony calculation
        dot_product = sum(a * b for a, b in zip(gateway_32d[:4], gateway_64d[:4]))
        norm_32 = sum(x*x for x in gateway_32d[:4]) ** 0.5
        norm_64 = sum(x*x for x in gateway_64d[:4]) ** 0.5

        if norm_32 > 0 and norm_64 > 0:
            harmony = dot_product / (norm_32 * norm_64)
        else:
            harmony = 0.0

        strategic_bias = 0.0  # Placeholder
        resonance = 0.0  # Placeholder
        pattern_match = 0.0  # Placeholder

        return [harmony, strategic_bias, resonance, pattern_match]

    def _compute_meta_features(self, board: chess.Board) -> List[float]:
        """
        Compute meta-cognitive features.

        Returns:
            [sharpness, certainty, style_preference, zugzwang]
        """

        # Position sharpness (forcing vs quiet)
        total_moves = len(list(board.legal_moves))
        forcing_moves = sum(
            1 for move in board.legal_moves
            if board.gives_check(move) or board.is_capture(move)
        )
        sharpness = forcing_moves / total_moves if total_moves > 0 else 0.0

        certainty = 0.5  # Placeholder
        style_pref = 0.0  # Placeholder
        zugzwang = 0.0  # Placeholder

        return [sharpness, certainty, style_pref, zugzwang]


def compute_strategic_features(
    board: chess.Board,
    state_32d: Optional['Pathion'] = None,
    gateway_32d: Optional[List[float]] = None,
    gateway_64d: Optional[List[float]] = None
) -> List[float]:
    """
    Convenience function for strategic feature computation.

    Args:
        board: Current chess position
        state_32d: Optional 32D state
        gateway_32d: Optional gateway from 16→32
        gateway_64d: Optional gateway from 32→64

    Returns:
        List of 32 strategic features for dimensions 32-63
    """
    analyzer = StrategicAnalyzer()
    return analyzer.compute_strategic_features(board, state_32d, gateway_32d, gateway_64d)
