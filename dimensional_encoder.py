"""
Dimensional Encoder - Chess Position to Sedenion Mapping

Encodes chess board positions as 16-dimensional sedenions for ZDTP gateway traversal.

PHASE 0 UPGRADE: Enhanced 16D structure with path verification
- Dimension 9: Path verification average (prevents illegal moves)
- Dimension 10: Piece reachability ratio (true mobility vs theoretical)

Week 2 Goal: Prove concept works with minimal viable encoding
Future: Optimize encoding for maximum information preservation
"""

import chess
import sys
import os

# Add CAILculator path for hypercomplex library access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'CAILculator'))

try:
    from hypercomplex import Sedenion
except ImportError:
    print("Warning: hypercomplex library not found. Install CAILculator.", file=sys.stderr)
    Sedenion = None

# Import Phase 0 path verification
try:
    from .path_verifier import calculate_dimension_9, calculate_dimension_10
except ImportError:
    from path_verifier import calculate_dimension_9, calculate_dimension_10


class DimensionalEncoder:
    """
    Encodes chess positions as 16D sedenions

    Encoding Strategy (Phase 0 Enhanced):
    - e_0: Material balance (scalar component)
    - e_1: Pawn structure score
    - e_2: King safety (white)
    - e_3: King safety (black)
    - e_4: Center control
    - e_5: Development (white)
    - e_6: Development (black)
    - e_7: Mobility (white)
    - e_8: Mobility (black)
    - e_9: Path verification average (PHASE 0 - prevents illegal moves!)
    - e_10: Piece reachability ratio (PHASE 0 - true vs theoretical mobility)
    - e_11: Pawn advancement (differential white - black)
    - e_12: Castling rights encoded
    - e_13: En passant target encoded
    - e_14: Halfmove clock (normalized)
    - e_15: Position complexity (normalized hash)
    """

    # Piece values for material counting
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0  # King has no material value
    }

    # Center squares for control evaluation
    CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]
    EXTENDED_CENTER = [
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6
    ]

    def __init__(self):
        """Initialize the encoder"""
        if Sedenion is None:
            raise ImportError("Sedenion class not available. Install hypercomplex library.")

    def encode_position(self, board: chess.Board) -> 'Sedenion':
        """
        Encode a chess position as a 16D sedenion

        Args:
            board: python-chess Board object

        Returns:
            Sedenion object representing the position
        """
        coeffs = [0.0] * 16

        # e_0: Material balance (positive = white advantage)
        coeffs[0] = self._material_balance(board)

        # e_1: Pawn structure score
        coeffs[1] = self._pawn_structure(board)

        # e_2, e_3: King safety for both sides
        coeffs[2] = self._king_safety(board, chess.WHITE)
        coeffs[3] = self._king_safety(board, chess.BLACK)

        # e_4: Center control
        coeffs[4] = self._center_control(board)

        # e_5, e_6: Development
        coeffs[5] = self._development(board, chess.WHITE)
        coeffs[6] = self._development(board, chess.BLACK)

        # e_7, e_8: Mobility (legal moves)
        coeffs[7] = self._mobility(board, chess.WHITE)
        coeffs[8] = self._mobility(board, chess.BLACK)

        # e_9: PHASE 0 - Path verification average (prevents illegal moves!)
        candidate_moves = list(board.legal_moves)
        coeffs[9] = calculate_dimension_9(board, candidate_moves)

        # e_10: PHASE 0 - Piece reachability ratio (true vs theoretical mobility)
        coeffs[10] = calculate_dimension_10(board)

        # e_11: Pawn advancement (shifted from e_10)
        coeffs[11] = self._pawn_advancement(board, chess.WHITE) - self._pawn_advancement(board, chess.BLACK)

        # e_12: Castling rights (encoded)
        coeffs[12] = self._castling_rights(board)

        # e_13: En passant target (encoded)
        coeffs[13] = self._en_passant_target(board)

        # e_14: Halfmove clock (normalized)
        coeffs[14] = board.halfmove_clock / 100.0  # Normalize to ~0-1 range

        # e_15: Position complexity (normalized hash)
        coeffs[15] = self._position_complexity(board)

        return Sedenion(*coeffs)

    def _material_balance(self, board: chess.Board) -> float:
        """Calculate material balance (positive = white advantage)"""
        balance = 0.0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            balance += (white_count - black_count) * self.PIECE_VALUES[piece_type]
        return balance

    def _pawn_structure(self, board: chess.Board) -> float:
        """
        Evaluate pawn structure
        Counts doubled, isolated, and passed pawns
        """
        score = 0.0

        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1.0 if color == chess.WHITE else -1.0
            pawns = board.pieces(chess.PAWN, color)

            # Count pawns per file
            file_counts = [0] * 8
            for square in pawns:
                file_counts[chess.square_file(square)] += 1

            # Penalize doubled pawns
            doubled = sum(1 for count in file_counts if count > 1)
            score += multiplier * (-0.5 * doubled)

            # Reward passed pawns (simplified check)
            for square in pawns:
                rank = chess.square_rank(square)
                if color == chess.WHITE and rank >= 5:
                    score += multiplier * 0.3
                elif color == chess.BLACK and rank <= 2:
                    score += multiplier * 0.3

        return score

    def _king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate king safety
        Considers castling and pawn shield
        """
        king_square = board.king(color)
        if king_square is None:
            return 0.0

        safety = 0.0

        # Has castled? (king on g1/g8 or c1/c8)
        if color == chess.WHITE:
            if king_square in [chess.G1, chess.C1]:
                safety += 1.0
        else:
            if king_square in [chess.G8, chess.C8]:
                safety += 1.0

        # Count pawns in front of king (pawn shield)
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)

        for file_offset in [-1, 0, 1]:
            target_file = king_file + file_offset
            if 0 <= target_file < 8:
                if color == chess.WHITE and king_rank < 7:
                    check_square = chess.square(target_file, king_rank + 1)
                    if board.piece_at(check_square) == chess.Piece(chess.PAWN, chess.WHITE):
                        safety += 0.3
                elif color == chess.BLACK and king_rank > 0:
                    check_square = chess.square(target_file, king_rank - 1)
                    if board.piece_at(check_square) == chess.Piece(chess.PAWN, chess.BLACK):
                        safety += 0.3

        return safety

    def _center_control(self, board: chess.Board) -> float:
        """Evaluate control of center squares"""
        control = 0.0

        for square in self.CENTER_SQUARES:
            # Check which side attacks this square more
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))
            control += (white_attackers - black_attackers) * 0.5

            # Bonus if occupied by our piece
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    control += 0.3
                else:
                    control -= 0.3

        return control

    def _development(self, board: chess.Board, color: chess.Color) -> float:
        """
        Measure piece development (knights and bishops off back rank)
        """
        developed = 0.0
        back_rank = 0 if color == chess.WHITE else 7

        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for square in board.pieces(piece_type, color):
                if chess.square_rank(square) != back_rank:
                    developed += 1.0

        # Normalize to 0-1 range (max 4 minor pieces)
        return developed / 4.0

    def _mobility(self, board: chess.Board, color: chess.Color) -> float:
        """
        Calculate mobility (number of legal moves)
        Normalized to roughly 0-1 range
        """
        # Save current turn
        original_turn = board.turn
        board.turn = color

        # Count legal moves
        move_count = len(list(board.legal_moves))

        # Restore turn
        board.turn = original_turn

        # Normalize (typical range 20-40 moves)
        return move_count / 40.0

    def _piece_activity(self, board: chess.Board) -> float:
        """
        Evaluate piece activity (pieces on good squares)
        """
        activity = 0.0

        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1.0 if color == chess.WHITE else -1.0

            # Knights on good squares
            for square in board.pieces(chess.KNIGHT, color):
                if square in self.EXTENDED_CENTER:
                    activity += multiplier * 0.5

            # Bishops on good diagonals
            for square in board.pieces(chess.BISHOP, color):
                if square in self.EXTENDED_CENTER:
                    activity += multiplier * 0.4

            # Rooks on open files
            for square in board.pieces(chess.ROOK, color):
                file = chess.square_file(square)
                # Check if file has no pawns (open file)
                has_pawn = any(
                    chess.square_file(sq) == file
                    for sq in board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
                )
                if not has_pawn:
                    activity += multiplier * 0.5

        return activity

    def _pawn_advancement(self, board: chess.Board, color: chess.Color) -> float:
        """
        Measure how advanced pawns are
        """
        advancement = 0.0
        pawns = board.pieces(chess.PAWN, color)

        for square in pawns:
            rank = chess.square_rank(square)
            if color == chess.WHITE:
                # White pawns advancing from rank 1 to rank 7
                advancement += rank / 7.0
            else:
                # Black pawns advancing from rank 6 to rank 0
                advancement += (7 - rank) / 7.0

        # Normalize by number of pawns (max 8)
        if len(pawns) > 0:
            return advancement / len(pawns)
        return 0.0

    def _castling_rights(self, board: chess.Board) -> float:
        """
        Encode castling rights as a number
        Each side gets 0.25 per castling right
        """
        rights = 0.0
        if board.has_kingside_castling_rights(chess.WHITE):
            rights += 0.25
        if board.has_queenside_castling_rights(chess.WHITE):
            rights += 0.25
        if board.has_kingside_castling_rights(chess.BLACK):
            rights -= 0.25
        if board.has_queenside_castling_rights(chess.BLACK):
            rights -= 0.25
        return rights

    def _en_passant_target(self, board: chess.Board) -> float:
        """
        Encode en passant target square
        Normalized to 0-1 range
        """
        if board.ep_square is None:
            return 0.0
        # Normalize square number (0-63) to 0-1
        return board.ep_square / 64.0

    def _position_complexity(self, board: chess.Board) -> float:
        """
        Estimate position complexity
        Based on piece count and position hash
        """
        # Count total pieces
        piece_count = len(board.piece_map())

        # Normalize (32 pieces max)
        complexity = piece_count / 32.0

        # Add normalized hash component for uniqueness
        # Use modulo to keep in reasonable range
        hash_component = (hash(board.fen()) % 1000) / 1000.0

        return (complexity + hash_component) / 2.0

    def decode_features(self, sedenion: 'Sedenion') -> dict:
        """
        Decode a sedenion back into interpretable features

        Args:
            sedenion: Sedenion representing a position

        Returns:
            Dictionary of decoded features
        """
        # Get coefficients - Sedenion has a coefficients() method
        if hasattr(sedenion, 'coefficients'):
            if callable(sedenion.coefficients):
                coeffs = sedenion.coefficients()
            else:
                coeffs = sedenion.coefficients
        else:
            # Try converting to list
            coeffs = [float(sedenion[i]) if hasattr(sedenion, '__getitem__') else 0.0 for i in range(16)]

        return {
            'material_balance': coeffs[0],
            'pawn_structure': coeffs[1],
            'king_safety_white': coeffs[2],
            'king_safety_black': coeffs[3],
            'center_control': coeffs[4],
            'development_white': coeffs[5],
            'development_black': coeffs[6],
            'mobility_white': coeffs[7],
            'mobility_black': coeffs[8],
            'piece_activity': coeffs[9],
            'pawn_advancement_white': coeffs[10],
            'pawn_advancement_black': coeffs[11],
            'castling_rights': coeffs[12],
            'en_passant': coeffs[13],
            'halfmove_clock': coeffs[14],
            'position_complexity': coeffs[15]
        }


# Module-level convenience functions
_encoder = None

def encode_position(board: chess.Board) -> 'Sedenion':
    """Encode a chess position as a 16D sedenion"""
    global _encoder
    if _encoder is None:
        _encoder = DimensionalEncoder()
    return _encoder.encode_position(board)

def decode_features(sedenion: 'Sedenion') -> dict:
    """Decode a sedenion into interpretable features"""
    global _encoder
    if _encoder is None:
        _encoder = DimensionalEncoder()
    return _encoder.decode_features(sedenion)


if __name__ == "__main__":
    # Test the encoder
    print("Testing Dimensional Encoder...")

    # Create a test board
    board = chess.Board()
    print(f"Starting position FEN: {board.fen()}")

    try:
        # Encode the position
        encoder = DimensionalEncoder()
        sedenion = encoder.encode_position(board)

        print(f"\nEncoded as 16D sedenion:")
        print(f"Coefficients: {sedenion}")

        # Decode back to features
        features = encoder.decode_features(sedenion)
        print(f"\nDecoded features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")

        print("\n[SUCCESS] Encoding test successful!")

    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
