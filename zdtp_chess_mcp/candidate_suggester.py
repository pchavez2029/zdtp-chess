"""
Candidate Move Suggester for ZDTP Chess

Categorizes ALL legal moves by type so the LLM sees the complete tactical
landscape before deciding. Fixes systematic blind spots where the LLM
misses pawn moves, defensive resources, and quiet positional moves.

Categories (priority order):
  1. FORCING  - checks, captures, promotions, threatens promotion
  2. DEFENSIVE - defends attacked piece, counterattacks (only when pieces ARE attacked)
  3. DEVELOPING - castling, center pawn advances, piece development, centralization
  4. QUIET - everything else (optional, off by default)

Author: Chavez AI Labs
Date: 2025-02-09
"""

import chess
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from .zdtp_engine import _emergency_see_safety_check
except ImportError:
    from zdtp_engine import _emergency_see_safety_check


# Piece values matching zdtp_engine.py line 61
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MoveEntry:
    """A single categorized move."""
    move_uci: str
    move_san: str
    category: str          # 'forcing', 'defensive', 'developing', 'quiet'
    subcategory: str       # e.g. 'check', 'capture', 'pawn_defense', 'castling'
    see_value: float       # Static Exchange Evaluation result
    see_safe: bool         # Whether SEE considers move safe
    see_warning: Optional[str]  # Warning text if unsafe
    target_info: str       # Human-readable detail (e.g. "captures queen on d5")
    piece_value: float = 0.0  # Value of moved piece (for secondary sorting)


@dataclass
class AttackedPiece:
    """A friendly piece under attack."""
    square: chess.Square
    piece_type: chess.PieceType
    value: float
    attacker_count: int
    defender_count: int
    is_hanging: bool       # attacked and not defended


@dataclass
class PositionSummary:
    """Overview of the current position."""
    side_to_move: str
    total_legal_moves: int
    white_material: float
    black_material: float
    material_balance: float
    in_check: bool
    attacked_pieces: List[AttackedPiece]


@dataclass
class CandidateSuggestion:
    """Complete suggestion response."""
    position_summary: PositionSummary
    forcing: List[MoveEntry] = field(default_factory=list)
    defensive: List[MoveEntry] = field(default_factory=list)
    developing: List[MoveEntry] = field(default_factory=list)
    quiet: List[MoveEntry] = field(default_factory=list)


# ============================================================================
# Helper Functions
# ============================================================================

def get_attacked_pieces(board: chess.Board, color: chess.Color) -> List[AttackedPiece]:
    """
    Find all friendly pieces under attack.

    Extends opponent_response_analyzer.py check_hanging_pieces() by also
    counting defenders and sorting by piece value descending.

    Args:
        board: Current position
        color: Side whose pieces to check

    Returns:
        List of AttackedPiece sorted by value descending
    """
    attacked = []
    opponent = not color

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            attackers = list(board.attackers(opponent, square))
            if attackers:
                defenders = list(board.attackers(color, square))
                value = PIECE_VALUES.get(piece.piece_type, 0.0)
                attacked.append(AttackedPiece(
                    square=square,
                    piece_type=piece.piece_type,
                    value=value,
                    attacker_count=len(attackers),
                    defender_count=len(defenders),
                    is_hanging=len(defenders) == 0,
                ))

    attacked.sort(key=lambda ap: ap.value, reverse=True)
    return attacked


def gives_check(board: chess.Board, move: chess.Move) -> bool:
    """Check if move gives check. Uses board.gives_check (efficient, no copy)."""
    return board.gives_check(move)


def threatens_promotion(board: chess.Board, move: chess.Move) -> bool:
    """Check if move is a promotion or moves pawn to 7th/2nd rank."""
    if move.promotion is not None:
        return True

    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(move.to_square)
        if piece.color == chess.WHITE and to_rank == 6:  # 7th rank (0-indexed)
            return True
        if piece.color == chess.BLACK and to_rank == 1:  # 2nd rank (0-indexed)
            return True

    return False


def defends_piece(
    board: chess.Board,
    move: chess.Move,
    attacked_squares: List[chess.Square]
) -> Optional[chess.Square]:
    """
    Check if a move defends any of the attacked squares.

    Push the move on a copy, then check if the moved piece now defends
    any attacked square via board.attackers().

    Returns:
        The defended square if found, else None
    """
    if not attacked_squares:
        return None

    board_copy = board.copy()
    board_copy.push(move)

    color = board.turn  # We just moved, so our pieces are this color

    for sq in attacked_squares:
        # After our move, does our side now defend this square?
        defenders = board_copy.attackers(color, sq)
        if move.to_square in defenders:
            return sq

    return None


def newly_attacks_piece(board: chess.Board, move: chess.Move) -> Optional[str]:
    """
    Check if the move creates a new attack on an enemy piece.

    Returns:
        Description of the attack, or None
    """
    board_copy = board.copy()
    board_copy.push(move)

    our_color = board.turn
    opponent = not our_color

    # Check what enemy pieces the moved piece now attacks
    for sq in chess.SQUARES:
        enemy = board_copy.piece_at(sq)
        if enemy and enemy.color == opponent and enemy.piece_type != chess.KING:
            if move.to_square in board_copy.attackers(our_color, sq):
                # Was this already attacked before the move?
                if move.from_square not in board.attackers(our_color, sq):
                    piece_name = chess.piece_name(enemy.piece_type)
                    return f"attacks {piece_name} on {chess.square_name(sq)}"

    return None


def build_position_summary(board: chess.Board) -> PositionSummary:
    """Compose position overview with material counts and attacked pieces."""
    white_mat = 0.0
    black_mat = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES.get(piece.piece_type, 0.0)
            if piece.color == chess.WHITE:
                white_mat += val
            else:
                black_mat += val

    color = board.turn
    attacked = get_attacked_pieces(board, color)

    return PositionSummary(
        side_to_move="white" if color == chess.WHITE else "black",
        total_legal_moves=len(list(board.legal_moves)),
        white_material=white_mat,
        black_material=black_mat,
        material_balance=white_mat - black_mat,
        in_check=board.is_check(),
        attacked_pieces=attacked,
    )


# ============================================================================
# Move Categorization
# ============================================================================

# The four key central squares (d4, e4, d5, e5) — used for outpost detection
KEY_CENTRAL_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}


def _is_developing_move(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if a move qualifies as developing (without full MoveEntry construction).

    Used to gate counterattack detection — developing moves should not be
    reclassified as counterattacks even when they happen to attack a piece.

    WARNING: This mirrors the developing checks in categorize_move(). If those
    checks change, this function must be updated to match.
    """
    piece = board.piece_at(move.from_square)
    piece_type = piece.piece_type if piece else None
    our_color = board.turn

    # Castling
    if board.is_castling(move):
        return True

    # Pawn center advance (d/e file)
    if piece_type == chess.PAWN:
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        if to_file in (3, 4):  # d and e files
            if (our_color == chess.WHITE and to_rank in (2, 3)) or \
               (our_color == chess.BLACK and to_rank in (4, 5)):
                return True
        # Pawn support center (c/f file)
        if to_file in (2, 5):  # c and f files
            if (our_color == chess.WHITE and to_rank in (2, 3)) or \
               (our_color == chess.BLACK and to_rank in (4, 5)):
                return True

    # Minor piece development from back rank
    if piece_type in (chess.KNIGHT, chess.BISHOP):
        from_rank = chess.square_rank(move.from_square)
        back_rank = 0 if our_color == chess.WHITE else 7
        if from_rank == back_rank:
            return True

    # Key central square
    if piece_type in (chess.KNIGHT, chess.BISHOP, chess.QUEEN):
        if move.to_square in KEY_CENTRAL_SQUARES:
            return True

    # Centralization (from outside center box to inside)
    if piece_type in (chess.KNIGHT, chess.BISHOP, chess.QUEEN):
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
            from_file = chess.square_file(move.from_square)
            from_rank = chess.square_rank(move.from_square)
            if not (2 <= from_file <= 5 and 2 <= from_rank <= 5):
                return True

    return False


def categorize_move(
    board: chess.Board,
    move: chess.Move,
    attacked_pieces: List[AttackedPiece]
) -> MoveEntry:
    """
    Classify a single move into category/subcategory.

    Priority: forcing > defensive > developing > quiet
    """
    move_san = board.san(move)
    move_uci = move.uci()

    # Run SEE
    see_result = _emergency_see_safety_check(board, move)
    see_value = see_result.get('see_value', 0.0)
    see_safe = see_result.get('is_safe', True)
    see_warning = see_result.get('warning', None)

    piece = board.piece_at(move.from_square)
    piece_type = piece.piece_type if piece else None
    piece_val = PIECE_VALUES.get(piece_type, 0.0) if piece_type else 0.0
    our_color = board.turn

    # --- FORCING ---
    if gives_check(board, move):
        return MoveEntry(
            move_uci=move_uci, move_san=move_san,
            category='forcing', subcategory='check',
            see_value=see_value, see_safe=see_safe, see_warning=see_warning,
            target_info=f"gives check", piece_value=piece_val
        )

    if move.promotion is not None:
        promo_piece = chess.piece_name(move.promotion)
        return MoveEntry(
            move_uci=move_uci, move_san=move_san,
            category='forcing', subcategory='promotion',
            see_value=see_value, see_safe=see_safe, see_warning=see_warning,
            target_info=f"promotes to {promo_piece}", piece_value=piece_val
        )

    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        # En passant: captured piece is not on to_square
        if captured:
            cap_name = chess.piece_name(captured.piece_type)
            cap_value = PIECE_VALUES.get(captured.piece_type, 0.0)
        else:
            cap_name = "pawn"
            cap_value = 1.0

        sub = 'pawn_capture' if piece_type == chess.PAWN else 'piece_capture'
        return MoveEntry(
            move_uci=move_uci, move_san=move_san,
            category='forcing', subcategory=sub,
            see_value=see_value, see_safe=see_safe, see_warning=see_warning,
            target_info=f"captures {cap_name} on {chess.square_name(move.to_square)}",
            piece_value=piece_val
        )

    if threatens_promotion(board, move):
        return MoveEntry(
            move_uci=move_uci, move_san=move_san,
            category='forcing', subcategory='threatens_promotion',
            see_value=see_value, see_safe=see_safe, see_warning=see_warning,
            target_info=f"pawn to 7th rank (threatens promotion)",
            piece_value=piece_val
        )

    # --- DEFENSIVE (only if we have attacked pieces) ---
    attacked_squares = [ap.square for ap in attacked_pieces]
    if attacked_squares:
        # 1. ESCAPE — any move of an attacked piece
        if move.from_square in attacked_squares:
            escaped_piece = board.piece_at(move.from_square)
            ep_name = chess.piece_name(escaped_piece.piece_type) if escaped_piece else "piece"
            ep_sq = chess.square_name(move.from_square)
            return MoveEntry(
                move_uci=move_uci, move_san=move_san,
                category='defensive', subcategory='escape',
                see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                target_info=f"moves {ep_name} from attacked {ep_sq}",
                piece_value=piece_val
            )

        # 2. DEFEND — piece moves to guard an attacked square
        defended_sq = defends_piece(board, move, attacked_squares)
        if defended_sq is not None:
            defended_piece = board.piece_at(defended_sq)
            if defended_piece:
                dp_name = chess.piece_name(defended_piece.piece_type)
                dp_sq = chess.square_name(defended_sq)
            else:
                dp_name = "piece"
                dp_sq = chess.square_name(defended_sq)

            if piece_type == chess.PAWN:
                sub = 'pawn_defends'
                info = f"pawn defends {dp_name} on {dp_sq}"
            else:
                sub = 'piece_defends'
                mover_name = chess.piece_name(piece_type) if piece_type else "piece"
                info = f"{mover_name} defends {dp_name} on {dp_sq}"

            return MoveEntry(
                move_uci=move_uci, move_san=move_san,
                category='defensive', subcategory=sub,
                see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                target_info=info, piece_value=piece_val
            )

        # 3. COUNTERATTACK — but not if the move is a developing move
        if not _is_developing_move(board, move):
            attack_info = newly_attacks_piece(board, move)
            if attack_info:
                return MoveEntry(
                    move_uci=move_uci, move_san=move_san,
                    category='defensive', subcategory='counterattack',
                    see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                    target_info=attack_info, piece_value=piece_val
                )

    # --- DEVELOPING ---
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > 4 else "queenside"
        return MoveEntry(
            move_uci=move_uci, move_san=move_san,
            category='developing', subcategory='castling',
            see_value=see_value, see_safe=see_safe, see_warning=see_warning,
            target_info=f"{side} castling", piece_value=piece_val
        )

    # Pawn center advance (d/e file pawn moving to rank 3-4 for white, 5-6 for black)
    if piece_type == chess.PAWN:
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        if to_file in (3, 4):  # d and e files
            if (our_color == chess.WHITE and to_rank in (2, 3)) or \
               (our_color == chess.BLACK and to_rank in (4, 5)):
                return MoveEntry(
                    move_uci=move_uci, move_san=move_san,
                    category='developing', subcategory='pawn_center',
                    see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                    target_info=f"center pawn advance", piece_value=piece_val
                )

    # Pawn support center (c/f file pawn moving to rank 3-4 for white, 5-6 for black)
    if piece_type == chess.PAWN:
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        if to_file in (2, 5):  # c and f files
            if (our_color == chess.WHITE and to_rank in (2, 3)) or \
               (our_color == chess.BLACK and to_rank in (4, 5)):
                return MoveEntry(
                    move_uci=move_uci, move_san=move_san,
                    category='developing', subcategory='pawn_support_center',
                    see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                    target_info=f"center support pawn advance",
                    piece_value=piece_val
                )

    # Minor piece development (N/B moving from back rank)
    if piece_type in (chess.KNIGHT, chess.BISHOP):
        from_rank = chess.square_rank(move.from_square)
        back_rank = 0 if our_color == chess.WHITE else 7
        if from_rank == back_rank:
            piece_name = chess.piece_name(piece_type)
            return MoveEntry(
                move_uci=move_uci, move_san=move_san,
                category='developing', subcategory='minor_development',
                see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                target_info=f"{piece_name} development", piece_value=piece_val
            )

    # Key central square (d4/e4/d5/e5) — fires before general centralization
    if piece_type in (chess.KNIGHT, chess.BISHOP, chess.QUEEN):
        if move.to_square in KEY_CENTRAL_SQUARES:
            piece_name = chess.piece_name(piece_type)
            return MoveEntry(
                move_uci=move_uci, move_san=move_san,
                category='developing', subcategory='key_square',
                see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                target_info=f"{piece_name} to key central square",
                piece_value=piece_val
            )

    # Rook to open/semi-open file
    if piece_type == chess.ROOK:
        to_file = chess.square_file(move.to_square)
        white_pawns_on_file = any(
            board.piece_at(chess.square(to_file, r)) == chess.Piece(chess.PAWN, chess.WHITE)
            for r in range(8)
        )
        black_pawns_on_file = any(
            board.piece_at(chess.square(to_file, r)) == chess.Piece(chess.PAWN, chess.BLACK)
            for r in range(8)
        )
        if not white_pawns_on_file and not black_pawns_on_file:
            return MoveEntry(
                move_uci=move_uci, move_san=move_san,
                category='developing', subcategory='rook_open_file',
                see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                target_info=f"rook to open file", piece_value=piece_val
            )
        elif (our_color == chess.WHITE and not white_pawns_on_file) or \
             (our_color == chess.BLACK and not black_pawns_on_file):
            return MoveEntry(
                move_uci=move_uci, move_san=move_san,
                category='developing', subcategory='rook_semi_open',
                see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                target_info=f"rook to semi-open file", piece_value=piece_val
            )

    # Centralization (piece moving toward center from outside center box)
    if piece_type in (chess.KNIGHT, chess.BISHOP, chess.QUEEN):
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
            from_file = chess.square_file(move.from_square)
            from_rank = chess.square_rank(move.from_square)
            if not (2 <= from_file <= 5 and 2 <= from_rank <= 5):
                piece_name = chess.piece_name(piece_type)
                return MoveEntry(
                    move_uci=move_uci, move_san=move_san,
                    category='developing', subcategory='centralization',
                    see_value=see_value, see_safe=see_safe, see_warning=see_warning,
                    target_info=f"{piece_name} centralization",
                    piece_value=piece_val
                )

    # --- QUIET ---
    info = ""
    if piece_type:
        info = f"{chess.piece_name(piece_type)} to {chess.square_name(move.to_square)}"
    else:
        info = f"move to {chess.square_name(move.to_square)}"

    return MoveEntry(
        move_uci=move_uci, move_san=move_san,
        category='quiet', subcategory='quiet',
        see_value=see_value, see_safe=see_safe, see_warning=see_warning,
        target_info=info, piece_value=piece_val
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def categorize_all_moves(
    board: chess.Board,
    max_per_category: int = 3,
    include_quiet: bool = False,
    min_eval_threshold: float = -2.0,
) -> CandidateSuggestion:
    """
    Categorize all legal moves and return structured suggestion.

    Args:
        board: Current position
        max_per_category: Max moves to return per category (0 = unlimited)
        include_quiet: Whether to include quiet moves in output
        min_eval_threshold: SEE threshold below which moves get warnings

    Returns:
        CandidateSuggestion with categorized moves
    """
    summary = build_position_summary(board)
    attacked_pieces = summary.attacked_pieces

    forcing = []
    defensive = []
    developing = []
    quiet = []

    for move in board.legal_moves:
        entry = categorize_move(board, move, attacked_pieces)

        if entry.category == 'forcing':
            forcing.append(entry)
        elif entry.category == 'defensive':
            defensive.append(entry)
        elif entry.category == 'developing':
            developing.append(entry)
        else:
            quiet.append(entry)

    # Sort forcing by subcategory priority + SEE value
    _subcategory_priority = {
        'check': 0,
        'promotion': 1,
        'piece_capture': 2,
        'pawn_capture': 3,
        'threatens_promotion': 4,
    }
    forcing.sort(key=lambda e: (
        _subcategory_priority.get(e.subcategory, 99),
        -e.see_value
    ))

    # Sort defensive: escapes first, then defenses, then counterattacks
    # Secondary sort by piece value descending (queen escapes before knight)
    _def_priority = {
        'escape': 0,
        'pawn_defends': 1,
        'piece_defends': 2,
        'counterattack': 3,
    }
    defensive.sort(key=lambda e: (
        _def_priority.get(e.subcategory, 99),
        -e.piece_value,
    ))

    # Sort developing by subcategory priority
    _dev_priority = {
        'castling': 0,
        'pawn_center': 1,
        'pawn_support_center': 2,
        'key_square': 3,
        'minor_development': 4,
        'rook_open_file': 5,
        'rook_semi_open': 6,
        'centralization': 7,
    }
    developing.sort(key=lambda e: _dev_priority.get(e.subcategory, 99))

    # Truncate to max_per_category
    if max_per_category > 0:
        forcing = forcing[:max_per_category]
        developing = developing[:max_per_category]
        quiet = quiet[:max_per_category]

        # Defensive: per-subcategory limits for guaranteed variety
        _sub_limits = {
            'escape': 5,
            'pawn_defends': 3,
            'piece_defends': 3,
            'counterattack': 2,
        }
        limited = []
        sub_counts: Dict[str, int] = {}
        for entry in defensive:  # already sorted
            count = sub_counts.get(entry.subcategory, 0)
            limit = _sub_limits.get(entry.subcategory, max_per_category)
            if count < limit:
                limited.append(entry)
                sub_counts[entry.subcategory] = count + 1
        defensive = limited

    result = CandidateSuggestion(
        position_summary=summary,
        forcing=forcing,
        defensive=defensive,
        developing=developing,
    )

    if include_quiet:
        result.quiet = quiet

    return result


# ============================================================================
# JSON Formatting
# ============================================================================

def format_suggestion_response(suggestion: CandidateSuggestion) -> dict:
    """Convert CandidateSuggestion to a JSON-serializable dict."""

    def format_move(entry: MoveEntry) -> dict:
        result = {
            'move_uci': entry.move_uci,
            'move_san': entry.move_san,
            'subcategory': entry.subcategory,
            'info': entry.target_info,
            'see_value': entry.see_value,
        }
        if entry.piece_value > 0:
            result['piece_value'] = entry.piece_value
        if not entry.see_safe:
            result['see_warning'] = entry.see_warning
        return result

    def format_attacked(ap: AttackedPiece) -> dict:
        return {
            'square': chess.square_name(ap.square),
            'piece': chess.piece_name(ap.piece_type),
            'value': ap.value,
            'attackers': ap.attacker_count,
            'defenders': ap.defender_count,
            'hanging': ap.is_hanging,
        }

    summary = suggestion.position_summary
    output = {
        'position': {
            'side_to_move': summary.side_to_move,
            'total_legal_moves': summary.total_legal_moves,
            'material': {
                'white': summary.white_material,
                'black': summary.black_material,
                'balance': summary.material_balance,
            },
            'in_check': summary.in_check,
        },
        'categories': {},
    }

    if summary.attacked_pieces:
        output['position']['attacked_pieces'] = [
            format_attacked(ap) for ap in summary.attacked_pieces
        ]

    if suggestion.forcing:
        output['categories']['forcing'] = [format_move(e) for e in suggestion.forcing]

    if suggestion.defensive:
        output['categories']['defensive'] = [format_move(e) for e in suggestion.defensive]

    if suggestion.developing:
        output['categories']['developing'] = [format_move(e) for e in suggestion.developing]

    if suggestion.quiet:
        output['categories']['quiet'] = [format_move(e) for e in suggestion.quiet]

    return output
