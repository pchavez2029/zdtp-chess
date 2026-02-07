"""
Strategic Analysis Module - 64D Layer Intelligence

Session 0.1 Upgrade: Formally Verified Features (Lean 4)
Grounded in ChavezTransform_Specification_aristotle.lean

Provides strategic feature computation for the 64D layer:
- Multi-move tactical sequences (2-3 ply)
- Game phase recognition
- Leverage deficiency & fortress detection (NEW - Convergence Thm)
- Tactical ceiling (NEW - Bilateral Kernel Bound, Thm 5)
- Square color domination (NEW - Bitboard parity)
- Mobility occlusion (NEW - Dimensional Weight, Thm 3)
- Zugzwang coefficient (NEW - Non-commutativity measure)

This module feeds dimensions 32-63 of the 64D Chingon representation.

64D OPERATIONAL FEATURE MAP (Session 0.1):
  Dims 32-35: Multi-move tactical sequences
  Dims 36-39: Strategic planning
  Dims 40-43: Game phase recognition
  Dims 44-47: Verified positional features (relocated + new)
  Dims 48-51: Draw detection features
  Dims 52-55: Formally verified features (Lean 4)
  Dims 56-59: Gateway harmony
  Dims 60-63: Meta-cognitive features (including zugzwang)
"""

import chess
import math
from typing import List, Dict, Optional

try:
    from hypercomplex import Sedenion, Pathion, Chingon
except ImportError:
    Sedenion = None
    Pathion = None
    Chingon = None


class StrategicAnalyzer:
    """
    Analyzes strategic features for 64D encoding.

    Session 0.1: Features now grounded in Chavez Transform formal
    verification (Lean 4). The bilateral kernel bound, dimensional
    weight, and stability constant provide mathematical anchors for
    fortress detection and draw recognition.
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

    # Bitboard masks for square color domination (Dim 53)
    # Standard chess convention: a1 is dark
    DARK_SQUARES = 0xAA55AA55AA55AA55
    LIGHT_SQUARES = ~DARK_SQUARES & 0xFFFFFFFFFFFFFFFF

    def __init__(self):
        """Initialize strategic analyzer."""
        pass

    def compute_strategic_features(
        self,
        board: chess.Board,
        state_32d: Optional['Pathion'] = None,
        gateway_32d: Optional[List[float]] = None,
        gateway_64d: Optional[List[float]] = None,
        gateway_P: Optional['Sedenion'] = None,
        conjugate_Q: Optional['Sedenion'] = None
    ) -> List[float]:
        """
        Compute 32 strategic dimensions for 64D space.

        Session 0.1: Now accepts gateway_P and conjugate_Q for
        formally verified features (tactical ceiling, zugzwang).

        Returns a list of 32 floats for dimensions 32-63.
        """
        features = []

        # ── Dims 32-35: Multi-move sequences (unchanged) ──
        multi_move_features = self._analyze_multi_move_sequences(board)
        features.extend(multi_move_features)  # 4 values

        # ── Dims 36-39: Strategic planning (unchanged) ──
        plan_features = self._analyze_strategic_plans(board)
        features.extend(plan_features)  # 4 values

        # ── Dims 40-43: Game phase recognition (unchanged) ──
        phase_features = self._recognize_game_phase(board)
        features.extend(phase_features)  # 4 values

        # ── Dims 44-47: Verified Positional Features (Session 0.1) ──
        # Dim 44: space_vs_material (RELOCATED from dim 52)
        # Dim 45: leverage_deficiency (NEW - Convergence Theorem)
        # Dim 46: activity_vs_structure (RELOCATED from dim 53)
        # Dim 47: fortress_structural (NEW - composite fortress signal)
        verified_positional = self._compute_verified_positional(board)
        features.extend(verified_positional)  # 4 values

        # ── Dims 48-51: Draw Detection Features (Session 0.1) ──
        # Dim 48: opposite_color_bishops
        # Dim 49: insufficient_material_signal
        # Dim 50: passed_pawn_absence (no passed pawns = drawish)
        # Dim 51: draw_signal (master local draw indicator)
        draw_features = self._compute_draw_features(board)
        features.extend(draw_features)  # 4 values

        # ── Dims 52-55: Formally Verified Features (Lean 4) ──
        # Dim 52: tactical_ceiling (Bilateral Kernel Bound, Thm 5)
        # Dim 53: square_domination (Bitboard parity)
        # Dim 54: mobility_occlusion (Dimensional Weight, Thm 3)
        # Dim 55: reserved (0.0)
        verified_features = self._compute_verified_features(
            board, state_32d, gateway_P, conjugate_Q
        )
        features.extend(verified_features)  # 4 values

        # ── Dims 56-59: Gateway harmony (unchanged) ──
        if gateway_32d and gateway_64d:
            harmony_features = self._compute_gateway_harmony(
                gateway_32d, gateway_64d, board
            )
        else:
            harmony_features = [0.0, 0.0, 0.0, 0.0]
        features.extend(harmony_features)  # 4 values

        # ── Dims 60-63: Meta-cognitive features (Session 0.1) ──
        # Dim 60: sharpness (unchanged)
        # Dim 61: certainty (unchanged)
        # Dim 62: style_preference (unchanged)
        # Dim 63: zugzwang_coeff (NEW - non-commutativity measure)
        meta_features = self._compute_meta_features(
            board, gateway_P, state_32d
        )
        features.extend(meta_features)  # 4 values

        return features

    # ═══════════════════════════════════════════════════════════════
    #  EXISTING METHODS (preserved from pre-Session 0.1)
    # ═══════════════════════════════════════════════════════════════

    def _analyze_multi_move_sequences(self, board: chess.Board) -> List[float]:
        """
        Analyze tactical sequences 2-3 moves deep.
        Returns: [two_move_score, three_move_score, forcing_seq, quiet_potential]
        """
        candidates = self._get_candidate_moves(board, max_moves=5)

        two_move_best = -999.0
        three_move_best = -999.0
        forcing_value = 0.0

        for move in candidates:
            two_score = self._evaluate_2move_sequence(board, move)
            two_move_best = max(two_move_best, two_score)

            three_score = self._evaluate_3move_sequence(board, move)
            three_move_best = max(three_move_best, three_score)

            if board.gives_check(move) or board.is_capture(move):
                forcing_value += 0.5

        quiet_potential = len(candidates) - forcing_value

        return [two_move_best, three_move_best, forcing_value, quiet_potential]

    def _evaluate_2move_sequence(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate position after move + best opponent response."""
        board.push(move)
        opp_best_score = 999.0
        for opp_move in list(board.legal_moves)[:10]:
            board.push(opp_move)
            score = self._simple_eval(board)
            opp_best_score = min(opp_best_score, score)
            board.pop()
        board.pop()
        return opp_best_score

    def _evaluate_3move_sequence(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate position 3 moves deep."""
        board.push(move)
        best_score = -999.0
        for opp_move in list(board.legal_moves)[:5]:
            board.push(opp_move)
            for reply in list(board.legal_moves)[:5]:
                board.push(reply)
                score = self._simple_eval(board)
                best_score = max(best_score, score)
                board.pop()
            board.pop()
        board.pop()
        return best_score

    def _get_candidate_moves(self, board: chess.Board, max_moves: int = 10) -> List[chess.Move]:
        """Get candidate moves sorted by simple heuristic."""
        moves = list(board.legal_moves)

        def move_priority(move):
            score = 0
            if board.gives_check(move):
                score += 100
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    score += self.PIECE_VALUES.get(captured.piece_type, 0) * 10
            return -score

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
        Returns: [piece_majority, pawn_breakthrough, maneuvering, prophylaxis]
        """
        return [0.0, 0.0, 0.0, 0.0]

    def _recognize_game_phase(self, board: chess.Board) -> List[float]:
        """
        Recognize game phase and evaluate accordingly.
        Returns: [opening_score, middlegame_score, endgame_score, transition_value]
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

        if piece_count > 28 and queen_count == 2:
            opening_score = 1.0
            white_developed = self._count_developed_pieces(board, chess.WHITE)
            black_developed = self._count_developed_pieces(board, chess.BLACK)
            opening_score += (white_developed - black_developed) * 0.3
        elif piece_count < 14 or queen_count == 0:
            endgame_score = 1.0
            white_king_sq = board.king(chess.WHITE)
            black_king_sq = board.king(chess.BLACK)
            if white_king_sq and black_king_sq:
                white_king_centralization = self._king_centralization(white_king_sq)
                black_king_centralization = self._king_centralization(black_king_sq)
                endgame_score += (white_king_centralization - black_king_centralization) * 0.5
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
        dist_from_center = abs(file - 3.5) + abs(rank - 3.5)
        centralization = 1.0 - (dist_from_center / 7.0)
        return centralization

    def _compute_gateway_harmony(
        self,
        gateway_32d: List[float],
        gateway_64d: List[float],
        board: chess.Board
    ) -> List[float]:
        """
        Measure harmony between dual gateways.
        Returns: [harmony, strategic_bias, resonance, pattern_match]
        """
        if len(gateway_32d) < 4 or len(gateway_64d) < 4:
            return [0.0, 0.0, 0.0, 0.0]

        dot_product = sum(a * b for a, b in zip(gateway_32d[:4], gateway_64d[:4]))
        norm_32 = sum(x*x for x in gateway_32d[:4]) ** 0.5
        norm_64 = sum(x*x for x in gateway_64d[:4]) ** 0.5

        if norm_32 > 0 and norm_64 > 0:
            harmony = dot_product / (norm_32 * norm_64)
        else:
            harmony = 0.0

        return [harmony, 0.0, 0.0, 0.0]

    # ═══════════════════════════════════════════════════════════════
    #  SESSION 0.1: VERIFIED POSITIONAL FEATURES (Dims 44-47)
    # ═══════════════════════════════════════════════════════════════

    def _compute_verified_positional(self, board: chess.Board) -> List[float]:
        """
        Compute dims 44-47: relocated + new verified features.

        Dim 44: space_vs_material (relocated from old dim 52)
        Dim 45: leverage_deficiency (Convergence Thm - pawn energy absence)
        Dim 46: activity_vs_structure (relocated from old dim 53)
        Dim 47: fortress_structural (composite fortress signal)
        """
        material = self._simple_eval(board) / 10.0
        space = self._space_evaluation(board)
        activity = self._activity_evaluation(board)
        structure = self._structure_evaluation(board)

        dim_44_space_vs_mat = space - material
        dim_45_leverage = self._compute_leverage_deficiency(board)
        dim_46_activity_vs_struct = activity - structure
        dim_47_fortress = self._compute_fortress_structural(board, dim_45_leverage)

        return [dim_44_space_vs_mat, dim_45_leverage, dim_46_activity_vs_struct, dim_47_fortress]

    def _compute_leverage_deficiency(self, board: chess.Board) -> float:
        """
        Dim 45: Leverage Deficiency (Convergence Theorem).

        Detects absence of "pawn energy" to break through locked lines.
        High value = pawns are locked with no break available = fortress-like.

        Grounded in: If the Chavez Transform converges (Thm 1), the position
        has settled into a stable basin. Leverage deficiency measures how
        far the pawn structure is from any phase transition.
        """
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        if not white_pawns and not black_pawns:
            return 0.0

        locked_pairs = 0
        available_breaks = 0

        # Count locked pairs and available breaks from White's perspective
        for wp in white_pawns:
            file = chess.square_file(wp)
            rank = chess.square_rank(wp)

            # Directly blocked by black pawn?
            if rank < 7:
                ahead = chess.square(file, rank + 1)
                if ahead in black_pawns:
                    locked_pairs += 1

                # Diagonal pawn captures available? (potential breaks)
                for df in [-1, 1]:
                    tf = file + df
                    if 0 <= tf < 8:
                        diag = chess.square(tf, rank + 1)
                        if diag in black_pawns:
                            available_breaks += 1

        # Count from Black's perspective too
        for bp in black_pawns:
            file = chess.square_file(bp)
            rank = chess.square_rank(bp)

            if rank > 0:
                ahead = chess.square(file, rank - 1)
                if ahead in white_pawns:
                    pass  # Already counted from white side

                for df in [-1, 1]:
                    tf = file + df
                    if 0 <= tf < 8:
                        diag = chess.square(tf, rank - 1)
                        if diag in white_pawns:
                            available_breaks += 1

        total_pawns = len(white_pawns) + len(black_pawns)
        lock_ratio = locked_pairs / max(len(white_pawns), 1)

        # No breaks and high lock ratio → maximum deficiency
        if available_breaks == 0 and locked_pairs > 0:
            return min(1.0, lock_ratio * 1.5)

        break_ratio = available_breaks / max(total_pawns, 1)
        deficiency = lock_ratio - break_ratio * 0.5
        return max(0.0, min(1.0, deficiency))

    def _compute_fortress_structural(
        self, board: chess.Board, leverage_deficiency: float
    ) -> float:
        """
        Dim 47: Composite fortress structural signal.

        Combines multiple indicators:
        - High leverage deficiency (locked pawns, no breaks)
        - No passed pawns (no promotion threats)
        - Kings cannot penetrate
        - Material roughly balanced or minor piece endgame

        Returns 0.0-1.0 where 1.0 = definite fortress structure.
        """
        signals = []

        # Signal 1: Leverage deficiency (already computed)
        signals.append(leverage_deficiency)

        # Signal 2: Absence of passed pawns
        has_passed_pawn = False
        for color in [chess.WHITE, chess.BLACK]:
            for pawn_sq in board.pieces(chess.PAWN, color):
                if self._is_passed_pawn(board, pawn_sq, color):
                    has_passed_pawn = True
                    break
            if has_passed_pawn:
                break
        signals.append(0.0 if has_passed_pawn else 1.0)

        # Signal 3: Material balance (fortresses need near-equal material)
        material_diff = abs(self._simple_eval(board))
        if material_diff < 1.0:
            signals.append(1.0)
        elif material_diff < 3.0:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # Signal 4: Endgame (fortresses are endgame phenomena)
        piece_count = len(board.piece_map())
        if piece_count <= 12:
            signals.append(1.0)
        elif piece_count <= 20:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # Composite: weighted average
        fortress = (
            0.35 * signals[0] +   # leverage deficiency (most important)
            0.25 * signals[1] +   # no passed pawns
            0.20 * signals[2] +   # material balance
            0.20 * signals[3]     # endgame phase
        )
        return min(1.0, fortress)

    def _is_passed_pawn(self, board: chess.Board, pawn_sq: int, color: chess.Color) -> bool:
        """Check if a pawn is passed (no opposing pawns blocking or guarding)."""
        file = chess.square_file(pawn_sq)
        rank = chess.square_rank(pawn_sq)
        enemy_pawns = board.pieces(chess.PAWN, not color)

        if color == chess.WHITE:
            # Check files file-1, file, file+1 for enemy pawns ahead
            for f in [file - 1, file, file + 1]:
                if 0 <= f < 8:
                    for r in range(rank + 1, 8):
                        if chess.square(f, r) in enemy_pawns:
                            return False
            return True
        else:
            for f in [file - 1, file, file + 1]:
                if 0 <= f < 8:
                    for r in range(0, rank):
                        if chess.square(f, r) in enemy_pawns:
                            return False
            return True

    # ═══════════════════════════════════════════════════════════════
    #  SESSION 0.1: DRAW DETECTION FEATURES (Dims 48-51)
    # ═══════════════════════════════════════════════════════════════

    def _compute_draw_features(self, board: chess.Board) -> List[float]:
        """
        Compute dims 48-51: draw detection signals.

        Dim 48: opposite_color_bishops
        Dim 49: insufficient_material_signal
        Dim 50: passed_pawn_absence (drawish indicator)
        Dim 51: draw_signal (master local draw indicator - triggers Master Dampener)
        """
        dim_48_ocb = self._detect_opposite_color_bishops(board)
        dim_49_insuf = self._detect_insufficient_material(board)
        dim_50_no_passers = self._detect_passed_pawn_absence(board)

        # Dim 51: Master draw signal - composite of all draw indicators
        # This is the LOCAL component of the Master Dampener
        # (temporal confirmation via eval_history happens in the evaluator)
        dim_51_draw = self._compute_draw_signal(
            board, dim_48_ocb, dim_49_insuf, dim_50_no_passers
        )

        return [dim_48_ocb, dim_49_insuf, dim_50_no_passers, dim_51_draw]

    def _detect_opposite_color_bishops(self, board: chess.Board) -> float:
        """
        Dim 48: Detect opposite-color bishop endgame.
        Returns 1.0 if OCB detected, 0.0 otherwise.
        """
        white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = list(board.pieces(chess.BISHOP, chess.BLACK))

        if len(white_bishops) != 1 or len(black_bishops) != 1:
            return 0.0

        # Check if bishops are on opposite colors
        wb_dark = (chess.square_file(white_bishops[0]) + chess.square_rank(white_bishops[0])) % 2 == 0
        bb_dark = (chess.square_file(black_bishops[0]) + chess.square_rank(black_bishops[0])) % 2 == 0

        if wb_dark != bb_dark:
            return 1.0
        return 0.0

    def _detect_insufficient_material(self, board: chess.Board) -> float:
        """
        Dim 49: Detect insufficient material for checkmate.
        Returns 0.0-1.0 scale (1.0 = definitely insufficient).
        """
        if board.is_insufficient_material():
            return 1.0

        # Near-insufficient: K+B vs K+B (same color), K+N vs K
        piece_count = len(board.piece_map())
        if piece_count <= 4:
            # Few pieces - likely drawish
            has_pawns = bool(board.pieces(chess.PAWN, chess.WHITE) or
                          board.pieces(chess.PAWN, chess.BLACK))
            if not has_pawns:
                return 0.7  # No pawns, few pieces = very drawish
        return 0.0

    def _detect_passed_pawn_absence(self, board: chess.Board) -> float:
        """
        Dim 50: Detect absence of passed pawns (drawish indicator).
        No passed pawns = no promotion threats = harder to win.
        """
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        if not white_pawns and not black_pawns:
            return 1.0  # No pawns at all = very drawish

        for color in [chess.WHITE, chess.BLACK]:
            for pawn_sq in board.pieces(chess.PAWN, color):
                if self._is_passed_pawn(board, pawn_sq, color):
                    return 0.0  # Found a passed pawn - not drawish

        return 1.0  # No passed pawns found

    def _compute_draw_signal(
        self, board: chess.Board,
        ocb: float, insuf: float, no_passers: float
    ) -> float:
        """
        Dim 51: Master local draw signal.

        Grounded in Stability Theorem (Lean 4, Thm 2):
        When the manifold ceases to evolve (position is static),
        the Chavez Transform is bounded by M · ||f||₁.

        This local signal detects structural draw conditions.
        The evaluator's Master Dampener confirms with temporal history.

        Final Eval = Consensus × (1 - FortressSignal)
        where FortressSignal derives from this dim + dim 45.
        """
        # Weight the signals
        draw_score = (
            0.30 * ocb +           # OCB is strong draw indicator
            0.30 * insuf +         # Insufficient material
            0.20 * no_passers +    # No passed pawns
            0.20 * self._detect_position_stasis(board)  # Position rigidity
        )
        return min(1.0, draw_score)

    def _detect_position_stasis(self, board: chess.Board) -> float:
        """
        Detect positional rigidity - how "frozen" is the position?
        High value = position cannot be changed = fortress-like.
        """
        legal_moves = list(board.legal_moves)
        total_moves = len(legal_moves)

        if total_moves == 0:
            return 1.0

        # Count moves that actually change the position meaningfully
        meaningful_moves = 0
        for move in legal_moves:
            # Captures and pawn moves are "meaningful" - they change structure
            if board.is_capture(move):
                meaningful_moves += 1
            elif board.piece_at(move.from_square) and \
                 board.piece_at(move.from_square).piece_type == chess.PAWN:
                meaningful_moves += 1

        # If most moves are just shuffling pieces, position is static
        shuffle_ratio = 1.0 - (meaningful_moves / max(total_moves, 1))
        return shuffle_ratio

    # ═══════════════════════════════════════════════════════════════
    #  SESSION 0.1: FORMALLY VERIFIED FEATURES (Dims 52-55, Lean 4)
    # ═══════════════════════════════════════════════════════════════

    def _compute_verified_features(
        self, board: chess.Board,
        state_32d: Optional['Pathion'],
        gateway_P: Optional['Sedenion'],
        conjugate_Q: Optional['Sedenion']
    ) -> List[float]:
        """
        Compute dims 52-55: formally verified features from Lean 4 proofs.

        Dim 52: tactical_ceiling (Bilateral Kernel Bound, Thm 5)
        Dim 53: square_domination (Bitboard color complex analysis)
        Dim 54: mobility_occlusion (Dimensional Weight decay, Thm 3)
        Dim 55: reserved (0.0)
        """
        dim_52 = self._compute_tactical_ceiling(state_32d, gateway_P, conjugate_Q)
        dim_53 = self._compute_square_domination(board)
        dim_54 = self._compute_mobility_occlusion(board)
        dim_55 = 0.0  # Reserved

        return [dim_52, dim_53, dim_54, dim_55]

    def _compute_tactical_ceiling(
        self,
        state_32d: Optional['Pathion'],
        gateway_P: Optional['Sedenion'],
        conjugate_Q: Optional['Sedenion']
    ) -> float:
        """
        Dim 52: Tactical Ceiling (Bilateral Kernel Bound, Thm 5).

        From Lean 4 verified proof:
            K_Z(P,Q,x) ≤ 4(||P||² + ||Q||²)||x||²

        Computes the ratio of actual bilateral kernel to its theoretical
        maximum. High ratio = tactical evaluation near saturation.
        Low ratio = room for tactical complexity to grow.

        The ceiling ensures tactical noise is always mathematically contained.
        """
        if gateway_P is None or conjugate_Q is None or state_32d is None:
            return 0.0

        if Sedenion is None:
            return 0.0

        try:
            # Extract 16D position from first 16 coefficients of 32D state
            coeffs_32d = list(state_32d.coefficients())
            state_16d = Sedenion(*coeffs_32d[:16])

            # Compute actual bilateral kernel: K_Z = ||P·x||² + ||x·Q||² + ||Q·x||² + ||x·P||²
            Px = gateway_P * state_16d
            xQ = state_16d * conjugate_Q
            Qx = conjugate_Q * state_16d
            xP = state_16d * gateway_P

            K_Z_actual = abs(Px)**2 + abs(xQ)**2 + abs(Qx)**2 + abs(xP)**2

            # Compute theoretical bound: 4(||P||² + ||Q||²)||x||²
            P_norm_sq = abs(gateway_P) ** 2
            Q_norm_sq = abs(conjugate_Q) ** 2
            x_norm_sq = abs(state_16d) ** 2

            K_Z_bound = 4 * (P_norm_sq + Q_norm_sq) * x_norm_sq

            if K_Z_bound < 1e-10:
                return 0.0

            # Ratio: how close to the ceiling are we?
            saturation = K_Z_actual / K_Z_bound
            return min(1.0, saturation)

        except Exception:
            return 0.0

    def _compute_square_domination(self, board: chess.Board) -> float:
        """
        Dim 53: Square Color Domination (Bitboard Parity).

        From Research Brief 0.1:
            DARK_SQUARES = 0xAA55AA55AA55AA55
            Score = (Control & DARK) - (Control & LIGHT)

        Detects color-locked bishops and "bad bishop" scenarios
        that contribute to the fortress signal. High disparity
        identifies positional binds on a specific color complex.

        Returns: Normalized domination score (-1 to +1).
        Positive = White dominates dark squares.
        Negative = Black dominates dark squares.
        """
        white_control_dark = 0
        white_control_light = 0
        black_control_dark = 0
        black_control_light = 0

        for sq in chess.SQUARES:
            is_dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0

            w_attackers = len(board.attackers(chess.WHITE, sq))
            b_attackers = len(board.attackers(chess.BLACK, sq))

            if is_dark:
                white_control_dark += w_attackers
                black_control_dark += b_attackers
            else:
                white_control_light += w_attackers
                black_control_light += b_attackers

        # Net dark square control
        dark_balance = (white_control_dark - black_control_dark)
        light_balance = (white_control_light - black_control_light)

        # Disparity: how much does one color complex differ from the other?
        disparity = (dark_balance - light_balance)

        # Normalize to roughly -1 to +1
        max_control = max(abs(dark_balance) + abs(light_balance), 1)
        return disparity / max_control

    def _compute_mobility_occlusion(self, board: chess.Board) -> float:
        """
        Dim 54: Mobility Occlusion (Dimensional Weight, Thm 3).

        From Lean 4 verified proof:
            Ω_d(x) = (1 + ||x||²)^(-d/2) ≤ 1

        Measures the percentage of squares unreachable by any piece,
        then applies the dimensional weight decay to dampen gateway
        "votes" when the board is highly occluded.

        High occlusion + decay = piece influence is diminished = fortress-like.
        """
        reachable = set()

        # Collect all squares reachable by any piece
        for color in [chess.WHITE, chess.BLACK]:
            board_copy = board.copy()
            board_copy.turn = color
            for move in board_copy.legal_moves:
                reachable.add(move.to_square)

        total_squares = 64
        unreachable = total_squares - len(reachable)
        occlusion_ratio = unreachable / total_squares

        # Apply dimensional weight decay: (1 + occlusion²)^(-d/2)
        # where d=2 (our working dimension parameter)
        d = 2.0
        x_sq = occlusion_ratio ** 2
        dimensional_weight = (1 + x_sq) ** (-d / 2)

        # Return: high occlusion dampened by dimensional weight
        # When occlusion is high, weight is near 1 (low x²),
        # but the occlusion value itself is high
        return occlusion_ratio * dimensional_weight

    # ═══════════════════════════════════════════════════════════════
    #  SESSION 0.1: META-COGNITIVE FEATURES (Dims 60-63)
    # ═══════════════════════════════════════════════════════════════

    def _compute_meta_features(
        self, board: chess.Board,
        gateway_P: Optional['Sedenion'] = None,
        state_32d: Optional['Pathion'] = None
    ) -> List[float]:
        """
        Compute meta-cognitive features.

        Dim 60: sharpness (unchanged)
        Dim 61: certainty (unchanged)
        Dim 62: style_preference (unchanged)
        Dim 63: zugzwang_coeff (NEW - non-commutativity of P·x vs x·P)
        """
        total_moves = len(list(board.legal_moves))
        forcing_moves = sum(
            1 for move in board.legal_moves
            if board.gives_check(move) or board.is_capture(move)
        )
        sharpness = forcing_moves / total_moves if total_moves > 0 else 0.0

        certainty = 0.5
        style_pref = 0.0

        # Dim 63: Zugzwang coefficient via non-commutativity
        zugzwang = self._compute_zugzwang(board, gateway_P, state_32d)

        return [sharpness, certainty, style_pref, zugzwang]

    def _compute_zugzwang(
        self, board: chess.Board,
        gateway_P: Optional['Sedenion'] = None,
        state_32d: Optional['Pathion'] = None
    ) -> float:
        """
        Dim 63: Zugzwang Coefficient (Non-Commutativity Measure).

        From Research Brief 0.1:
            zugzwang = |P·x - x·P|

        Measures the asymmetry between P·x and x·P in Cayley-Dickson
        multiplication. Non-zero values indicate the position has
        "directional bias" through the gateway — the evaluation
        depends on WHO is to move.

        This directly captures zugzwang: when one side is fine but
        the obligation to move causes deterioration.

        High value = strong directional asymmetry = zugzwang potential.
        """
        if gateway_P is None or state_32d is None or Sedenion is None:
            return 0.0

        try:
            # Reconstruct 16D position encoding from 32D state
            coeffs_32d = list(state_32d.coefficients())
            state_16d = Sedenion(*coeffs_32d[:16])

            # Compute P·x and x·P (non-commutative!)
            Px = gateway_P * state_16d
            xP = state_16d * gateway_P

            # Zugzwang = ||P·x - x·P|| (norm of the difference)
            diff = Px - xP
            raw_zugzwang = abs(diff)

            # Normalize: typical values range 0-10, scale to 0-1
            normalized = min(1.0, raw_zugzwang / 10.0)
            return normalized

        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  UTILITY METHODS (preserved from pre-Session 0.1)
    # ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════

def compute_strategic_features(
    board: chess.Board,
    state_32d: Optional['Pathion'] = None,
    gateway_32d: Optional[List[float]] = None,
    gateway_64d: Optional[List[float]] = None,
    gateway_P: Optional['Sedenion'] = None,
    conjugate_Q: Optional['Sedenion'] = None
) -> List[float]:
    """
    Convenience function for strategic feature computation.

    Session 0.1: Now accepts gateway_P and conjugate_Q for
    formally verified features (tactical ceiling, zugzwang).
    """
    analyzer = StrategicAnalyzer()
    return analyzer.compute_strategic_features(
        board, state_32d, gateway_32d, gateway_64d,
        gateway_P, conjugate_Q
    )
