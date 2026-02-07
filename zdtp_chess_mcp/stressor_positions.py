"""
Stressor Position Library - ZDTP Chess Session 1

Curated positions for testing ZDTP's dimensional analysis under stress.
Based on the Stressor Roadmap from Gemini Peer Review Session 0 (Jan 6, 2026).

Categories:
1. Exchange Sacrifices - 16D crashes but 32D maintains "positional bind"
2. Locked Chains - French/KID structures forcing Bishop vs Knight gateway divergence
3. Perpetual Singularity - Queen Gateway bridging massive material deficit with strategic draw
4. Topological Finality - Fortress positions where 64D should identify draw despite 16D material imbalance
5. Dead Gateway Paradox - Immobilized pieces testing gateway contribution weighting

Reference: Gemini Research Brief, Session 0
Researcher: Paul Chavez (Founder, Chavez AI Labs)
"""

from typing import Dict, List, Optional


# Each stressor has: FEN, category, description, expected behavior, difficulty
STRESSOR_LIBRARY: Dict[str, Dict] = {

    # ========================================================================
    # CATEGORY 1: EXCHANGE SACRIFICES
    # Test: 16D crashes (material loss) but 32D sees "positional bind"
    # ========================================================================

    "exchange_sac_petrosian": {
        "fen": "r1b2rk1/pp3ppp/2n1pn2/q1bpN3/2P5/1PN1B3/P2QBPPP/R4RK1 w - - 0 13",
        "category": "exchange_sacrifice",
        "name": "Petrosian-Style Exchange Sacrifice",
        "description": (
            "White can sacrifice the exchange with Nxc6 bxc6, gaining a "
            "dominant knight vs bad bishop, ruined pawn structure, and lasting "
            "positional bind. 16D should show material deficit; 32D/64D should "
            "show compensation."
        ),
        "expected_16d": "Negative (material loss after exchange sac)",
        "expected_32d": "Positive (positional bind, structural damage)",
        "expected_64d": "Positive (long-term strategic compensation)",
        "difficulty": "hard",
        "test_move": "e5c6",
    },

    "exchange_sac_bind": {
        "fen": "r4rk1/1b2bppp/ppnppn2/q7/2P1P3/1PN1BN2/P2QBPPP/R4RK1 w - - 0 14",
        "category": "exchange_sacrifice",
        "name": "Positional Bind Exchange Sacrifice",
        "description": (
            "White has space advantage and can sacrifice exchange on c6 "
            "to destroy Black's pawn structure and create an outpost. "
            "Tests dimensional divergence between tactical and strategic layers."
        ),
        "expected_16d": "Negative (material deficit)",
        "expected_32d": "Slight positive (structural advantage)",
        "expected_64d": "Positive (outpost dominance)",
        "difficulty": "medium",
        "test_move": None,
    },

    # ========================================================================
    # CATEGORY 2: LOCKED CHAINS (French/KID)
    # Test: Bishop vs Knight gateway divergence in closed positions
    # ========================================================================

    "french_advance_chain": {
        "fen": "r1bqkb1r/pp3ppp/2n1pn2/2ppP3/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
        "category": "locked_chain",
        "name": "French Advance - Locked Pawn Chain",
        "description": (
            "Classic French Advance with locked e5/d4 vs e6/d5 chain. "
            "Bishop gateway should struggle (hemmed in by pawns); "
            "Knight gateway should thrive (can maneuver around chain). "
            "Tests the Dead Gateway Paradox for bishops."
        ),
        "expected_bishop_gateway": "Low score (bad bishop behind chain)",
        "expected_knight_gateway": "Higher score (knight maneuvers)",
        "expected_divergence": "High (Bishop vs Knight gateways should diverge)",
        "difficulty": "medium",
        "test_move": None,
    },

    "kid_locked_center": {
        "fen": "r1bq1rk1/pppn1pbp/3p1np1/3Pp3/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 9",
        "category": "locked_chain",
        "name": "King's Indian - Locked Center",
        "description": (
            "KID with locked d5/e4 center. White plays on queenside, "
            "Black on kingside. Bishop and Knight gateways should show "
            "different assessments of this mutually locked structure. "
            "64D should recognize the race dynamic."
        ),
        "expected_bishop_gateway": "Moderate (long diagonal potential)",
        "expected_knight_gateway": "Moderate (outpost on d5/f5)",
        "expected_divergence": "Medium",
        "difficulty": "hard",
        "test_move": None,
    },

    "french_fortress": {
        "fen": "5k2/pp3pp1/4p2p/3pP3/PP1P4/5P2/5KPP/8 w - - 0 30",
        "category": "locked_chain",
        "name": "French Defense Fortress (Endgame)",
        "description": (
            "Topological Finality test case from Gemini Session 0. "
            "Locked pawn chain creates a fortress. Despite potential "
            "16D material imbalance assessments, 64D should correctly "
            "identify this as a draw. Critical test for draw detection."
        ),
        "expected_16d": "Slight advantage for White (space)",
        "expected_64d": "Draw / near-zero (fortress recognized)",
        "difficulty": "expert",
        "test_move": None,
    },

    # ========================================================================
    # CATEGORY 3: PERPETUAL SINGULARITY
    # Test: Queen Gateway bridging material deficit with strategic draw
    # ========================================================================

    "perpetual_queen_chase": {
        "fen": "6k1/5ppp/8/8/8/8/1q3PPP/4R1K1 b - - 0 40",
        "category": "perpetual_singularity",
        "name": "Perpetual Queen Check Pattern",
        "description": (
            "Black has a queen vs rook (huge material advantage, ~+4). "
            "But White's queen-side threat of perpetual check via Qf2-Qg3 "
            "pattern should force a draw. Tests if Queen Gateway can bridge "
            "a massive material deficit (-10.0) with a strategic draw (0.00). "
            "The Perpetual Singularity from Gemini's roadmap."
        ),
        "expected_16d": "Large negative (material deficit for White)",
        "expected_queen_gateway": "Near-zero (perpetual draw detected)",
        "expected_divergence": "Extreme (16D vs 64D)",
        "difficulty": "expert",
        "test_move": None,
    },

    "perpetual_rook_endgame": {
        "fen": "8/8/8/5k2/8/4K3/r7/5R2 w - - 0 50",
        "category": "perpetual_singularity",
        "name": "Rook Endgame - Theoretical Draw",
        "description": (
            "Philidor-style rook endgame. Material is equal but position "
            "tests whether ZDTP correctly evaluates the drawing technique. "
            "64D should recognize the theoretical draw pattern."
        ),
        "expected_16d": "Near-zero (equal material)",
        "expected_64d": "Near-zero (theoretical draw)",
        "difficulty": "medium",
        "test_move": None,
    },

    # ========================================================================
    # CATEGORY 4: TOPOLOGICAL FINALITY
    # Test: 64D correctly identifies draws despite 16D material signals
    # ========================================================================

    "opposite_color_bishops": {
        "fen": "8/8/4k3/3b4/8/3B4/4K3/8 w - - 0 60",
        "category": "topological_finality",
        "name": "Opposite-Color Bishop Endgame",
        "description": (
            "Opposite-colored bishops with one extra pawn would normally "
            "show material advantage in 16D, but this is a well-known "
            "fortress draw. 64D should recognize topological finality."
        ),
        "expected_16d": "Near-zero (equal material)",
        "expected_64d": "Draw (opposite-color bishops fortress)",
        "difficulty": "medium",
        "test_move": None,
    },

    "wrong_color_bishop": {
        "fen": "7k/8/8/8/8/8/P7/KB6 w - - 0 1",
        "category": "topological_finality",
        "name": "Wrong-Color Bishop + Rook Pawn",
        "description": (
            "White has a bishop and rook pawn but the bishop is the wrong "
            "color (can't control the promotion square). This is a theoretical "
            "draw despite White's material advantage. 64D should detect the "
            "topological impossibility of winning."
        ),
        "expected_16d": "Positive (material advantage)",
        "expected_64d": "Draw (wrong-color bishop pattern)",
        "difficulty": "expert",
        "test_move": None,
    },

    # ========================================================================
    # CATEGORY 5: DEAD GATEWAY PARADOX
    # Test: Immobilized piece gateways still contribute algebraic gravity
    # ========================================================================

    "trapped_rook": {
        "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3",
        "category": "dead_gateway",
        "name": "Italian Game - Rook Still on a1",
        "description": (
            "Both rooks are on their starting squares and have no open files. "
            "The Rook Gateway (Pattern 104) evaluates 'dead' rooks. "
            "Per Unified Field Theory (Session 0 conclusion): these gateways "
            "should STILL contribute to the algebraic gravity of the board, "
            "even though the physical pieces are immobilized."
        ),
        "expected_rook_gateway": "Should still provide meaningful eval",
        "expected_convergence": "Rook gateway may diverge from Knight/Bishop",
        "difficulty": "medium",
        "test_move": None,
    },

    "trapped_bishop": {
        "fen": "rnbqk2r/ppppbppp/4pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        "category": "dead_gateway",
        "name": "Queen's Gambit - Hemmed-In Light Bishop",
        "description": (
            "Black's light-squared bishop on c8 is completely hemmed in by "
            "pawns on d7, e6. The Bishop Gateway should detect this 'dead' "
            "piece. Per Unified Field Theory: the gateway still contributes, "
            "but with reduced weight reflecting the piece's limitations."
        ),
        "expected_bishop_gateway": "Lower than other gateways (trapped piece)",
        "expected_convergence": "Bishop gateway should diverge",
        "difficulty": "medium",
        "test_move": None,
    },
}


def get_stressor_category(category: str) -> List[Dict]:
    """
    Get all stressor positions in a category.

    Args:
        category: One of 'exchange_sacrifice', 'locked_chain',
                 'perpetual_singularity', 'topological_finality', 'dead_gateway'

    Returns:
        List of stressor position dicts matching the category
    """
    return [
        {"key": key, **data}
        for key, data in STRESSOR_LIBRARY.items()
        if data["category"] == category
    ]


def list_stressor_positions() -> str:
    """
    Format a human-readable listing of all stressor positions.

    Returns:
        Formatted string showing all available stressor positions
    """
    output = []
    output.append("╔══════════════════════════════════════════════════════════════╗")
    output.append("║  📚 ZDTP STRESSOR POSITION LIBRARY                          ║")
    output.append("║  Session 1 - Gemini Peer Review Roadmap                      ║")
    output.append("╚══════════════════════════════════════════════════════════════╝")
    output.append("")

    categories = {
        "exchange_sacrifice": "💥 EXCHANGE SACRIFICES (16D crash, 32D bind)",
        "locked_chain": "🔒 LOCKED CHAINS (French/KID gateway divergence)",
        "perpetual_singularity": "♾️  PERPETUAL SINGULARITY (material deficit → draw)",
        "topological_finality": "🏔️  TOPOLOGICAL FINALITY (fortress draws)",
        "dead_gateway": "👻 DEAD GATEWAY PARADOX (immobilized piece contribution)",
    }

    for cat_key, cat_title in categories.items():
        positions = get_stressor_category(cat_key)
        if positions:
            output.append(f"\n{cat_title}")
            output.append("─" * 60)
            for pos in positions:
                difficulty_emoji = {"medium": "🟡", "hard": "🟠", "expert": "🔴"}.get(
                    pos["difficulty"], "⚪"
                )
                output.append(f"  {difficulty_emoji} {pos['key']}")
                output.append(f"     {pos['name']}")
                output.append(f"     FEN: {pos['fen'][:50]}...")
                output.append("")

    output.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    output.append(f"Total: {len(STRESSOR_LIBRARY)} stressor positions across {len(categories)} categories")
    output.append("")
    output.append("Load a position: chess_load_position(fen='...')")
    output.append("Or by key:       chess_load_position(stressor_key='french_fortress')")

    return "\n".join(output)
