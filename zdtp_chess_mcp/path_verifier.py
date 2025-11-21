"""
Path Verification Module - Phase 0: 16D Path Verification

Prevents illegal move recommendations by validating paths at the base dimensional layer.

This module implements:
- Enhanced 16D coefficient structure with path verification
- Move path validation for sliding pieces
- Legal move pre-filtering
- Reachability calculations

Expected impact: Reduces illegal move recommendations from ~30% to 0%
"""

import chess
from typing import List, Tuple, Optional


# CODE PIECE #1: Enhanced 16D Structure Definition
ENHANCED_16D_STRUCTURE = {
    0: "material_balance",           # Keep existing
    1: "king_safety",                # Keep existing
    2: "true_mobility",              # UPGRADE: Path-verified mobility
    3: "center_control",             # Keep existing
    4: "pawn_structure",             # Keep existing
    5: "king_activity",              # Keep existing
    6: "tactical_pins",              # Keep existing
    7: "tactical_forks",             # Keep existing
    8: "checks_available",           # Keep existing
    9: "path_verification_avg",      # NEW: Average path legality score
    10: "piece_reachability_ratio",  # NEW: True reachable squares
    11: "piece_coordination",        # Keep existing
    12: "open_files",                # Keep existing
    13: "bishop_pair_bonus",         # Keep existing
    14: "knight_outposts",           # Keep existing
    15: "passed_pawns"               # Keep existing
}


def is_valid_piece_pattern(from_square: chess.Square, to_square: chess.Square, piece_type: chess.PieceType) -> bool:
    """
    Check if a move follows the piece's legal movement pattern.

    Args:
        from_square: Starting square
        to_square: Destination square
        piece_type: Type of piece moving

    Returns:
        True if move pattern is legal for this piece type
    """
    from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
    to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)

    file_diff = abs(to_file - from_file)
    rank_diff = abs(to_rank - from_rank)

    if piece_type == chess.PAWN:
        # Pawns move forward 1 (or 2 from start), capture diagonally
        # Simplified - actual implementation needs direction and capture checking
        return True  # Defer to chess library's legal_moves

    elif piece_type == chess.KNIGHT:
        # L-shape: 2 squares in one direction, 1 in the other
        return (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2)

    elif piece_type == chess.BISHOP:
        # Diagonal movement only
        return file_diff == rank_diff and file_diff > 0

    elif piece_type == chess.ROOK:
        # Straight lines only (same rank or file)
        return (file_diff == 0 and rank_diff > 0) or (rank_diff == 0 and file_diff > 0)

    elif piece_type == chess.QUEEN:
        # Combines rook and bishop movement
        return is_valid_piece_pattern(from_square, to_square, chess.ROOK) or \
               is_valid_piece_pattern(from_square, to_square, chess.BISHOP)

    elif piece_type == chess.KING:
        # One square in any direction
        return file_diff <= 1 and rank_diff <= 1 and (file_diff > 0 or rank_diff > 0)

    return False


# CODE PIECE #2: Path Verification Functions
def verify_move_path(board: chess.Board, from_square: chess.Square, to_square: chess.Square, piece_type: chess.PieceType) -> float:
    """
    Core path verification for sliding pieces (Rook, Bishop, Queen)

    Args:
        board: Current board state (chess.Board object)
        from_square: Starting square (chess.Square)
        to_square: Destination square (chess.Square)
        piece_type: Type of piece (chess.ROOK, chess.BISHOP, etc.)

    Returns:
        float: 1.0 if path is clear and legal
               0.0 if path is blocked
              -1.0 if move violates piece movement rules
    """
    # Step 1: Validate piece can move in this pattern
    if not is_valid_piece_pattern(from_square, to_square, piece_type):
        return -1.0  # Illegal piece movement

    # Step 2: For sliding pieces, check intermediate squares
    if piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
        path_squares = get_path_squares(from_square, to_square, piece_type)

        # Check each intermediate square (exclude start and end)
        for square in path_squares[1:-1]:
            if board.piece_at(square) is not None:
                return 0.0  # Path blocked by piece

    # Step 3: Check destination square
    dest_piece = board.piece_at(to_square)
    if dest_piece and dest_piece.color == board.turn:
        return 0.0  # Can't capture own piece

    return 1.0  # Path is clear and legal


def get_path_squares(from_square: chess.Square, to_square: chess.Square, piece_type: chess.PieceType) -> List[chess.Square]:
    """
    Calculate all squares along the path between from and to

    Returns:
        list: Ordered list of squares including from_square and to_square
    """
    from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
    to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)

    path = []

    if piece_type == chess.ROOK:
        if from_file == to_file:  # Vertical movement
            step = 1 if to_rank > from_rank else -1
            for rank in range(from_rank, to_rank + step, step):
                path.append(chess.square(from_file, rank))
        elif from_rank == to_rank:  # Horizontal movement
            step = 1 if to_file > from_file else -1
            for file in range(from_file, to_file + step, step):
                path.append(chess.square(file, from_rank))

    elif piece_type == chess.BISHOP:
        file_step = 1 if to_file > from_file else -1
        rank_step = 1 if to_rank > from_rank else -1

        files = range(from_file, to_file + file_step, file_step)
        ranks = range(from_rank, to_rank + rank_step, rank_step)

        for file, rank in zip(files, ranks):
            path.append(chess.square(file, rank))

    elif piece_type == chess.QUEEN:
        # Queen moves like rook or bishop
        if from_file == to_file or from_rank == to_rank:
            return get_path_squares(from_square, to_square, chess.ROOK)
        else:
            return get_path_squares(from_square, to_square, chess.BISHOP)

    return path


def get_theoretical_squares(piece_type: chess.PieceType, from_square: chess.Square) -> List[chess.Square]:
    """
    Get all theoretically reachable squares for a piece (ignoring board state).

    Args:
        piece_type: Type of piece
        from_square: Current square

    Returns:
        List of theoretically reachable squares
    """
    from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
    theoretical = []

    if piece_type == chess.KNIGHT:
        # L-shaped moves
        offsets = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for df, dr in offsets:
            file, rank = from_file + df, from_rank + dr
            if 0 <= file <= 7 and 0 <= rank <= 7:
                theoretical.append(chess.square(file, rank))

    elif piece_type == chess.BISHOP:
        # Diagonals
        for direction in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for dist in range(1, 8):
                file, rank = from_file + direction[0] * dist, from_rank + direction[1] * dist
                if 0 <= file <= 7 and 0 <= rank <= 7:
                    theoretical.append(chess.square(file, rank))

    elif piece_type == chess.ROOK:
        # Ranks and files
        for file in range(8):
            if file != from_file:
                theoretical.append(chess.square(file, from_rank))
        for rank in range(8):
            if rank != from_rank:
                theoretical.append(chess.square(from_file, rank))

    elif piece_type == chess.QUEEN:
        # Combine rook and bishop
        theoretical.extend(get_theoretical_squares(chess.ROOK, from_square))
        theoretical.extend(get_theoretical_squares(chess.BISHOP, from_square))

    elif piece_type == chess.KING:
        # One square in any direction
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if df == 0 and dr == 0:
                    continue
                file, rank = from_file + df, from_rank + dr
                if 0 <= file <= 7 and 0 <= rank <= 7:
                    theoretical.append(chess.square(file, rank))

    elif piece_type == chess.PAWN:
        # Forward moves (simplified - ignores color)
        if from_rank < 7:
            theoretical.append(chess.square(from_file, from_rank + 1))
        if from_rank == 1:
            theoretical.append(chess.square(from_file, from_rank + 2))

    return theoretical


# CODE PIECE #3: Dimension 9 Calculator
def calculate_dimension_9(board: chess.Board, candidate_moves: List[chess.Move]) -> float:
    """
    Dimension 9: Average path verification score across all candidate moves

    Returns:
        float: Score between -1.0 and 1.0
    """
    if not candidate_moves:
        return 0.0

    total_score = 0.0
    for move in candidate_moves:
        piece = board.piece_at(move.from_square)
        if piece:
            path_score = verify_move_path(
                board,
                move.from_square,
                move.to_square,
                piece.piece_type
            )
            total_score += path_score

    return total_score / len(candidate_moves)


# CODE PIECE #4: Dimension 10 Calculator
def calculate_dimension_10(board: chess.Board) -> float:
    """
    Dimension 10: Ratio of truly reachable squares to theoretical maximum

    This prevents false mobility calculations

    Returns:
        float: Score between 0.0 and 1.0
    """
    color = board.turn
    pieces = board.pieces(chess.PAWN, color) | \
             board.pieces(chess.KNIGHT, color) | \
             board.pieces(chess.BISHOP, color) | \
             board.pieces(chess.ROOK, color) | \
             board.pieces(chess.QUEEN, color)

    total_reachable = 0
    total_theoretical = 0

    for from_square in pieces:
        piece = board.piece_at(from_square)
        theoretical_squares = get_theoretical_squares(piece.piece_type, from_square)

        for to_square in theoretical_squares:
            total_theoretical += 1

            # Verify path is actually clear
            path_score = verify_move_path(board, from_square, to_square, piece.piece_type)
            if path_score == 1.0:
                total_reachable += 1

    if total_theoretical == 0:
        return 0.0

    return total_reachable / total_theoretical


# CODE PIECE #5: Legal Move Pre-Filter
def filter_legal_moves(board: chess.Board) -> List[chess.Move]:
    """
    Pre-filter moves at 16D level before passing to 32D/64D

    This prevents illegal moves from ever being recommended

    Returns:
        list: Only moves with clear, legal paths
    """
    pseudo_legal_moves = list(board.legal_moves)  # Use chess library's legal moves
    verified_moves = []

    for move in pseudo_legal_moves:
        piece = board.piece_at(move.from_square)
        if piece:
            path_score = verify_move_path(
                board,
                move.from_square,
                move.to_square,
                piece.piece_type
            )

            if path_score == 1.0:  # Only include if path is clear
                verified_moves.append(move)

    return verified_moves
