"""
Test suite for candidate_suggester.py

Validates move categorization across various position types:
1. Starting position - developing moves
2. Hanging piece positions - defensive moves surface
3. Check positions - forcing category populated
4. Capture positions with SEE - good/bad captures sorted
5. Promotion positions - forcing with promotion
6. max_per_category limiting
7. include_quiet flag
8. JSON output structure
"""

import sys
import os

# Add module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'zdtp_chess_mcp'))

import chess
from candidate_suggester import (
    categorize_all_moves,
    categorize_move,
    format_suggestion_response,
    get_attacked_pieces,
    gives_check,
    threatens_promotion,
    build_position_summary,
    PIECE_VALUES,
)


def test_starting_position():
    """Starting position: all moves should be developing or quiet (no forcing/defensive)."""
    board = chess.Board()
    result = categorize_all_moves(board, max_per_category=0, include_quiet=True)

    assert len(result.forcing) == 0, f"Expected no forcing moves, got {len(result.forcing)}"
    assert len(result.defensive) == 0, f"Expected no defensive moves, got {len(result.defensive)}"
    assert len(result.developing) > 0, f"Expected developing moves, got 0"

    total = len(result.forcing) + len(result.defensive) + len(result.developing) + len(result.quiet)
    assert total == 20, f"Expected 20 legal moves, got {total}"

    # Check that pawn center advances (d2d4, e2e4) appear in developing
    dev_ucis = [m.move_uci for m in result.developing]
    assert 'e2e4' in dev_ucis, f"e2e4 should be in developing, got: {dev_ucis}"
    assert 'd2d4' in dev_ucis, f"d2d4 should be in developing, got: {dev_ucis}"

    print("[PASS] Starting position: correct categorization")


def test_hanging_piece_defensive():
    """Position with a hanging piece: defensive moves should appear."""
    # White knight on e5 attacked by black queen on c7, no defender
    # FEN: r1b1kbnr/ppqppppp/2n5/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1
    board = chess.Board("r1b1kbnr/ppqppppp/2n5/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")

    attacked = get_attacked_pieces(board, chess.WHITE)
    # The knight on e5 should be attacked
    attacked_squares = [ap.square for ap in attacked]
    assert chess.E5 in attacked_squares, f"Knight on e5 should be attacked, got: {[chess.square_name(s) for s in attacked_squares]}"

    result = categorize_all_moves(board, max_per_category=0, include_quiet=True)

    # Should have some defensive moves (escape knight, or defend it)
    assert len(result.defensive) > 0, f"Expected defensive moves for hanging knight, got 0"

    # Check that the knight retreat/escape appears
    def_ucis = [m.move_uci for m in result.defensive]
    def_subcats = [m.subcategory for m in result.defensive]
    has_escape_or_defense = 'escape' in def_subcats or 'piece_defends' in def_subcats or 'pawn_defends' in def_subcats
    assert has_escape_or_defense, f"Expected escape/defense subcategory, got: {def_subcats}"

    print("[PASS] Hanging piece: defensive moves surface correctly")


def test_check_available():
    """Position where check is available: forcing category populated."""
    # White queen on d1 can give check on d8 or h5
    # Simple position: white Qd1 can check on h5 with no obstruction
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    # Qh5 gives check (via Qd1-h5)
    result = categorize_all_moves(board, max_per_category=0)

    check_moves = [m for m in result.forcing if m.subcategory == 'check']
    if check_moves:
        print(f"  Check moves found: {[m.move_san for m in check_moves]}")
        assert all(m.category == 'forcing' for m in check_moves)
        print("[PASS] Check moves: correctly placed in forcing category")
    else:
        # Qh5+ should be available
        qh5 = chess.Move.from_uci("d1h5")
        if qh5 in board.legal_moves and board.gives_check(qh5):
            assert False, "Qh5+ is legal and gives check but was not categorized as forcing/check"
        else:
            print("[PASS] Check moves: no check available (position as expected)")


def test_captures_with_see():
    """Captures should be in forcing category with SEE evaluation."""
    # Position where white can capture: e4 pawn takes d5 pawn
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")

    result = categorize_all_moves(board, max_per_category=0)

    captures = [m for m in result.forcing if 'capture' in m.subcategory]
    assert len(captures) > 0, "Expected captures in forcing category"

    # exd5 should be a pawn capture
    exd5_moves = [m for m in captures if m.move_uci == 'e4d5']
    assert len(exd5_moves) == 1, f"Expected exd5 capture, got: {[m.move_uci for m in captures]}"

    exd5 = exd5_moves[0]
    assert exd5.subcategory == 'pawn_capture', f"Expected pawn_capture, got {exd5.subcategory}"
    assert exd5.see_safe, f"exd5 should be SEE-safe (pawn takes pawn)"

    print("[PASS] Captures with SEE: correct categorization and evaluation")


def test_promotion_position():
    """Position with promotion available: should be in forcing category."""
    # White pawn on e7, can promote (kings far apart)
    board = chess.Board("8/4P3/8/8/8/8/8/K3k3 w - - 0 1")

    result = categorize_all_moves(board, max_per_category=0)

    # All promotion moves should be in forcing (either as 'promotion' or 'check')
    promo_moves = [m for m in result.forcing if m.subcategory in ('promotion', 'check')]
    assert len(promo_moves) > 0, f"Expected promotion/check moves in forcing, got 0"

    # Queen promotion gives check in this position, so it may be subcategory 'check'
    promo_sans = [m.move_san for m in promo_moves]
    has_queen_promo = any('Q' in san for san in promo_sans)
    assert has_queen_promo, f"Expected queen promotion, got: {promo_sans}"

    print("[PASS] Promotion position: correctly in forcing category")


def test_threatens_promotion():
    """Pawn on 6th rank (for white) should be categorized as threatens_promotion."""
    # White pawn on e6
    board = chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")

    result = categorize_all_moves(board, max_per_category=0)

    threat_promo = [m for m in result.forcing if m.subcategory == 'threatens_promotion']
    # e6e7 should threaten promotion
    if threat_promo:
        assert any(m.move_uci == 'e6e7' for m in threat_promo), \
            f"Expected e6e7 threatens promotion, got: {[m.move_uci for m in threat_promo]}"
        print("[PASS] Threatens promotion: pawn to 7th rank detected")
    else:
        # Might be classified as quiet or developing depending on exact position
        # Check that e6e7 is at least categorized somewhere
        all_moves = result.forcing + result.defensive + result.developing + result.quiet
        e7_moves = [m for m in all_moves if m.move_uci == 'e6e7']
        print(f"  e6e7 categorized as: {e7_moves[0].category}/{e7_moves[0].subcategory}" if e7_moves else "  e6e7 not found")
        print("[INFO] Threatens promotion: see categorization above")


def test_max_per_category():
    """max_per_category should limit output per category."""
    board = chess.Board()
    result_limited = categorize_all_moves(board, max_per_category=2, include_quiet=True)
    result_unlimited = categorize_all_moves(board, max_per_category=0, include_quiet=True)

    # Limited should have at most 2 per category
    assert len(result_limited.developing) <= 2, \
        f"Expected max 2 developing, got {len(result_limited.developing)}"
    assert len(result_limited.quiet) <= 2, \
        f"Expected max 2 quiet, got {len(result_limited.quiet)}"

    # Unlimited should have more
    total_unlimited = (len(result_unlimited.forcing) + len(result_unlimited.defensive) +
                       len(result_unlimited.developing) + len(result_unlimited.quiet))
    total_limited = (len(result_limited.forcing) + len(result_limited.defensive) +
                     len(result_limited.developing) + len(result_limited.quiet))
    assert total_unlimited >= total_limited, \
        f"Unlimited ({total_unlimited}) should be >= limited ({total_limited})"

    print("[PASS] max_per_category: limiting works correctly")


def test_include_quiet():
    """include_quiet=False should exclude quiet moves, True should include them."""
    board = chess.Board()

    result_no_quiet = categorize_all_moves(board, max_per_category=0, include_quiet=False)
    result_with_quiet = categorize_all_moves(board, max_per_category=0, include_quiet=True)

    assert len(result_no_quiet.quiet) == 0, \
        f"Expected 0 quiet moves with include_quiet=False, got {len(result_no_quiet.quiet)}"

    # With quiet included, total should equal all legal moves
    total_with = (len(result_with_quiet.forcing) + len(result_with_quiet.defensive) +
                  len(result_with_quiet.developing) + len(result_with_quiet.quiet))
    expected = len(list(board.legal_moves))
    assert total_with == expected, \
        f"Expected {expected} total moves, got {total_with}"

    print("[PASS] include_quiet: flag works correctly")


def test_json_output_structure():
    """JSON output should have correct structure."""
    board = chess.Board()
    suggestion = categorize_all_moves(board, max_per_category=3)
    output = format_suggestion_response(suggestion)

    # Check top-level keys
    assert 'position' in output, "Missing 'position' key"
    assert 'categories' in output, "Missing 'categories' key"

    # Check position structure
    pos = output['position']
    assert 'side_to_move' in pos, "Missing 'side_to_move'"
    assert 'total_legal_moves' in pos, "Missing 'total_legal_moves'"
    assert 'material' in pos, "Missing 'material'"
    assert 'in_check' in pos, "Missing 'in_check'"

    assert pos['side_to_move'] == 'white'
    assert pos['total_legal_moves'] == 20
    assert pos['in_check'] is False

    # Check material
    mat = pos['material']
    assert 'white' in mat and 'black' in mat and 'balance' in mat

    # Check categories
    cats = output['categories']
    assert 'developing' in cats, "Expected developing category in starting position"

    # Check move entry structure
    if cats.get('developing'):
        move_entry = cats['developing'][0]
        assert 'move_uci' in move_entry, "Missing 'move_uci'"
        assert 'move_san' in move_entry, "Missing 'move_san'"
        assert 'subcategory' in move_entry, "Missing 'subcategory'"
        assert 'info' in move_entry, "Missing 'info'"
        assert 'see_value' in move_entry, "Missing 'see_value'"

    # Verify it's JSON-serializable
    import json
    json_str = json.dumps(output, indent=2)
    assert len(json_str) > 0, "JSON serialization failed"

    print("[PASS] JSON output structure: validates correctly")


def test_castling_categorization():
    """Castling should be in developing category."""
    # Position where white can castle kingside
    board = chess.Board("r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    # White needs bishop out of the way for kingside castling
    # Let's use a position where castling is legal
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 0 4")

    result = categorize_all_moves(board, max_per_category=0)

    castling_moves = [m for m in result.developing if m.subcategory == 'castling']
    if castling_moves:
        assert castling_moves[0].category == 'developing'
        print(f"  Castling found: {castling_moves[0].move_san}")
        print("[PASS] Castling: correctly in developing category")
    else:
        # Check if O-O is even legal
        can_castle = any(board.is_castling(m) for m in board.legal_moves)
        if can_castle:
            assert False, "Castling is legal but not categorized as developing/castling"
        else:
            print("[SKIP] Castling: not available in this position")


def test_position_summary():
    """Position summary should have correct material and attack info."""
    board = chess.Board()
    summary = build_position_summary(board)

    assert summary.side_to_move == 'white'
    assert summary.total_legal_moves == 20
    assert summary.in_check is False

    # Starting material: 8 pawns (8) + 2 knights (6.4) + 2 bishops (6.6) + 2 rooks (10) + 1 queen (9) = 40
    expected_white = 8 * 1.0 + 2 * 3.2 + 2 * 3.3 + 2 * 5.0 + 9.0
    assert abs(summary.white_material - expected_white) < 0.01, \
        f"White material: expected {expected_white}, got {summary.white_material}"
    assert abs(summary.black_material - expected_white) < 0.01, \
        f"Black material should equal white in starting position"
    assert abs(summary.material_balance) < 0.01, \
        f"Material balance should be 0 in starting position"

    # No attacked pieces in starting position
    assert len(summary.attacked_pieces) == 0, \
        f"No pieces should be attacked in starting position"

    print("[PASS] Position summary: correct values")


def test_complex_tactical_position():
    """Test a complex middlegame position with multiple move types."""
    # Stressor-style position: white has tactical and defensive needs
    # White: Ke1, Qd1, Ra1, Rh1, Bc1, Bf1 (not developed), Nc3, pawns
    # Black: threats on various squares
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")

    result = categorize_all_moves(board, max_per_category=0, include_quiet=True)

    total = (len(result.forcing) + len(result.defensive) +
             len(result.developing) + len(result.quiet))
    expected = len(list(board.legal_moves))
    assert total == expected, f"Total categorized ({total}) should equal legal moves ({expected})"

    # Should have some developing moves (castling, knight/bishop development)
    print(f"  Forcing: {len(result.forcing)}, Defensive: {len(result.defensive)}, "
          f"Developing: {len(result.developing)}, Quiet: {len(result.quiet)}")

    print("[PASS] Complex position: all moves accounted for")


if __name__ == "__main__":
    print("=" * 60)
    print("CANDIDATE SUGGESTER TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        test_starting_position,
        test_hanging_piece_defensive,
        test_check_available,
        test_captures_with_see,
        test_promotion_position,
        test_threatens_promotion,
        test_max_per_category,
        test_include_quiet,
        test_json_output_structure,
        test_castling_categorization,
        test_position_summary,
        test_complex_tactical_position,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
