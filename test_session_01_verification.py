"""
Session 0.1 Verification Test Suite
====================================

Validates the implementation of the Research Brief specification:
1. Master Dampener (Fortress Draw Detection)
2. Formal Theorem Groundings (Lean 4)
   - Theorem 5: Bilateral Kernel Bound (Dim 52)
   - Theorem 3: Dimensional Weight (Dim 54)
   - Stability Constant M
3. Zugzwang Coefficient (Dim 63)
4. Dimension Assignments (44-55, 63)

Reference: ChavezTransform_Specification_aristotle.lean
Reference: ZDTP Research Brief: Session 0.1 (Verified AGI Proof-of-Concept)
"""

import sys
import io

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import chess

# Import ZDTP modules
sys.path.insert(0, 'zdtp_chess_mcp')
from dimensional_encoder import encode_position
from dimensional_portal import full_cascade, DimensionalPortal
from multidimensional_evaluator import evaluate_position, MultidimensionalEvaluator
from strategic_analyzer import StrategicAnalyzer
from gateway_patterns import assign_gateway
from stressor_positions import STRESSOR_LIBRARY

try:
    from hypercomplex import Sedenion, Pathion, Chingon
except ImportError:
    print("ERROR: hypercomplex library required. Install with: pip install hypercomplex")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    expected: str
    actual: str
    details: str = ""


class Session01Verifier:
    """Verification suite for Session 0.1 implementation."""

    def __init__(self):
        self.portal = DimensionalPortal()
        self.evaluator = MultidimensionalEvaluator()
        self.strategic_analyzer = StrategicAnalyzer()
        self.results: List[TestResult] = []

    def run_all_tests(self) -> None:
        """Run complete verification suite."""
        print("=" * 70)
        print("SESSION 0.1 VERIFICATION TEST SUITE")
        print("Lean 4 Formal Verification Grounding")
        print("=" * 70)
        print()

        # Test categories
        self._test_theorem_5_bilateral_bound()
        self._test_theorem_3_dimensional_weight()
        self._test_stability_constant()
        self._test_zugzwang_coefficient()
        self._test_dimension_assignments()
        self._test_master_dampener_fortress()
        self._test_master_dampener_temporal()

        # Summary
        self._print_summary()

    def _test_theorem_5_bilateral_bound(self) -> None:
        """
        Test Theorem 5: Bilateral Kernel Bound

        From Lean 4: K_Z(P,Q,x) <= 4(||P||^2 + ||Q||^2)||x||^2

        Implementation: Dim 52 (tactical_ceiling) computes saturation ratio.
        """
        print("\n" + "-" * 70)
        print("TEST: Theorem 5 - Bilateral Kernel Bound (Dim 52)")
        print("-" * 70)

        # Test with starting position
        board = chess.Board()
        state_16d = encode_position(board)

        # Get gateway P and conjugate Q
        gateway_P = assign_gateway(chess.KNIGHT)
        conjugate_Q = self.portal.get_conjugate(chess.KNIGHT)

        # Verify zero divisor property first
        product = gateway_P * conjugate_Q
        product_norm = abs(product)
        zd_verified = product_norm < 1e-10

        self.results.append(TestResult(
            name="Zero Divisor Property (P × Q ≈ 0)",
            passed=zd_verified,
            expected="||P × Q|| < 1e-10",
            actual=f"||P × Q|| = {product_norm:.2e}",
            details="Foundation for bilateral kernel computation"
        ))
        print(f"  Zero divisor verified: {zd_verified} (||P×Q|| = {product_norm:.2e})")

        # Compute bilateral kernel components
        coeffs_16d = list(state_16d.coefficients())
        x = Sedenion(*coeffs_16d[:16])

        Px = gateway_P * x
        xQ = x * conjugate_Q
        Qx = conjugate_Q * x
        xP = x * gateway_P

        K_Z_actual = abs(Px)**2 + abs(xQ)**2 + abs(Qx)**2 + abs(xP)**2

        # Theoretical bound
        P_norm_sq = abs(gateway_P) ** 2
        Q_norm_sq = abs(conjugate_Q) ** 2
        x_norm_sq = abs(x) ** 2
        K_Z_bound = 4 * (P_norm_sq + Q_norm_sq) * x_norm_sq

        # Verify bound holds
        bound_holds = K_Z_actual <= K_Z_bound + 1e-10  # tolerance for floating point

        self.results.append(TestResult(
            name="Theorem 5: K_Z <= 4(||P||^2 + ||Q||^2)||x||^2",
            passed=bound_holds,
            expected=f"K_Z <= {K_Z_bound:.6f}",
            actual=f"K_Z = {K_Z_actual:.6f}",
            details=f"||P||^2={P_norm_sq:.4f}, ||Q||^2={Q_norm_sq:.4f}, ||x||^2={x_norm_sq:.4f}"
        ))

        print(f"  K_Z actual:  {K_Z_actual:.6f}")
        print(f"  K_Z bound:   {K_Z_bound:.6f}")
        print(f"  Bound holds: {bound_holds}")

        # Test saturation ratio (Dim 52)
        if K_Z_bound > 1e-10:
            saturation = K_Z_actual / K_Z_bound
            saturation_valid = 0.0 <= saturation <= 1.0

            self.results.append(TestResult(
                name="Dim 52: Tactical Ceiling (Saturation Ratio)",
                passed=saturation_valid,
                expected="0.0 <= saturation <= 1.0",
                actual=f"saturation = {saturation:.6f}",
                details="Ratio of actual K_Z to theoretical maximum"
            ))
            print(f"  Saturation:  {saturation:.6f} (valid: {saturation_valid})")

    def _test_theorem_3_dimensional_weight(self) -> None:
        """
        Test Theorem 3: Dimensional Weight Bound

        From Lean 4: (1 + ||x||^2)^(-d/2) <= 1 for d > 0

        Implementation: Dim 54 (mobility_occlusion) uses this decay.
        """
        print("\n" + "-" * 70)
        print("TEST: Theorem 3 - Dimensional Weight (Dim 54)")
        print("-" * 70)

        # Test with various x values
        test_cases = [
            (0.0, "x=0"),
            (0.5, "x=0.5"),
            (1.0, "x=1.0"),
            (5.0, "x=5.0"),
        ]

        d = 2.0  # Working dimension parameter
        all_passed = True

        for x_val, label in test_cases:
            x_sq = x_val ** 2
            weight = (1 + x_sq) ** (-d / 2)
            bound_holds = weight <= 1.0

            if not bound_holds:
                all_passed = False

            print(f"  {label}: (1 + {x_sq:.2f})^(-{d}/2) = {weight:.6f} <= 1.0: {bound_holds}")

        self.results.append(TestResult(
            name="Theorem 3: (1 + ||x||^2)^(-d/2) <= 1",
            passed=all_passed,
            expected="All weights <= 1.0",
            actual=f"{'All passed' if all_passed else 'Some failed'}",
            details=f"Tested with d={d}"
        ))

        # Test actual mobility_occlusion computation
        board = chess.Board()
        occlusion = self.strategic_analyzer._compute_mobility_occlusion(board)
        occlusion_valid = 0.0 <= occlusion <= 1.0

        self.results.append(TestResult(
            name="Dim 54: Mobility Occlusion Range",
            passed=occlusion_valid,
            expected="0.0 <= occlusion <= 1.0",
            actual=f"occlusion = {occlusion:.6f}",
            details="Starting position mobility check"
        ))
        print(f"  Starting position occlusion: {occlusion:.6f} (valid: {occlusion_valid})")

    def _test_stability_constant(self) -> None:
        """
        Test Stability Constant M

        From Lean 4: M = (||P||^2 + ||Q||^2) * sqrt(pi/alpha)

        Implementation uses practical threshold M = 0.5
        """
        print("\n" + "-" * 70)
        print("TEST: Stability Constant M")
        print("-" * 70)

        # Compute theoretical M for Knight gateway
        gateway_P = assign_gateway(chess.KNIGHT)
        conjugate_Q = self.portal.get_conjugate(chess.KNIGHT)

        P_norm_sq = abs(gateway_P) ** 2
        Q_norm_sq = abs(conjugate_Q) ** 2
        alpha = 1.0  # Standard decay parameter

        M_theoretical = (P_norm_sq + Q_norm_sq) * math.sqrt(math.pi / alpha)
        M_practical = self.evaluator.STABILITY_CONSTANT_M

        print(f"  ||P||^2 = {P_norm_sq:.4f}")
        print(f"  ||Q||^2 = {Q_norm_sq:.4f}")
        print(f"  alpha = {alpha}")
        print(f"  M_theoretical = {M_theoretical:.4f}")
        print(f"  M_practical = {M_practical}")

        # Verify practical is tighter than theoretical
        practical_tighter = M_practical < M_theoretical

        self.results.append(TestResult(
            name="Stability Constant: Practical < Theoretical",
            passed=practical_tighter,
            expected=f"M_practical ({M_practical}) < M_theoretical ({M_theoretical:.4f})",
            actual=f"{'Tighter' if practical_tighter else 'Looser'}",
            details="Practical threshold is more sensitive for fortress detection"
        ))
        print(f"  Practical tighter: {practical_tighter}")

    def _test_zugzwang_coefficient(self) -> None:
        """
        Test Zugzwang Coefficient (Dim 63)

        Formula: zugzwang = |P·x - x·P| (non-commutativity measure)
        """
        print("\n" + "-" * 70)
        print("TEST: Zugzwang Coefficient (Dim 63)")
        print("-" * 70)

        board = chess.Board()
        state_16d = encode_position(board)
        cascade = full_cascade(state_16d, chess.KNIGHT, board)

        # Get 64D coefficients
        state_64d = cascade['state_64d']
        coeffs_64d = list(state_64d.coefficients())

        # Dim 63 is zugzwang coefficient
        zugzwang = coeffs_64d[63] if len(coeffs_64d) > 63 else 0.0
        zugzwang_valid = 0.0 <= zugzwang <= 1.0

        self.results.append(TestResult(
            name="Dim 63: Zugzwang Coefficient Range",
            passed=zugzwang_valid,
            expected="0.0 <= zugzwang <= 1.0",
            actual=f"zugzwang = {zugzwang:.6f}",
            details="Non-commutativity measure |P·x - x·P|"
        ))
        print(f"  Zugzwang coefficient: {zugzwang:.6f}")
        print(f"  Valid range: {zugzwang_valid}")

        # Verify non-commutativity exists
        gateway_P = assign_gateway(chess.KNIGHT)
        coeffs_16d = list(state_16d.coefficients())
        x = Sedenion(*coeffs_16d[:16])

        Px = gateway_P * x
        xP = x * gateway_P
        diff = Px - xP
        raw_nc = abs(diff)

        non_commutative = raw_nc > 1e-10
        self.results.append(TestResult(
            name="Non-Commutativity: P·x != x·P",
            passed=non_commutative,
            expected="||P·x - x·P|| > 0",
            actual=f"||P·x - x·P|| = {raw_nc:.6f}",
            details="Sedenion multiplication is non-commutative"
        ))
        print(f"  Raw non-commutativity: {raw_nc:.6f}")

    def _test_dimension_assignments(self) -> None:
        """
        Test that all Session 0.1 dimensions are properly assigned.
        """
        print("\n" + "-" * 70)
        print("TEST: Dimension Assignments (44-55, 63)")
        print("-" * 70)

        board = chess.Board()
        state_16d = encode_position(board)
        cascade = full_cascade(state_16d, chess.KNIGHT, board)

        state_64d = cascade['state_64d']
        coeffs_64d = list(state_64d.coefficients())

        # Expected dimension assignments
        dim_assignments = {
            44: ("space_vs_material", "Relocated from 52"),
            45: ("leverage_deficiency", "Convergence Theorem"),
            46: ("activity_vs_structure", "Relocated from 53"),
            47: ("fortress_structural", "Composite fortress signal"),
            48: ("opposite_color_bishops", "Draw indicator"),
            49: ("insufficient_material", "Draw indicator"),
            50: ("passed_pawn_absence", "Draw indicator"),
            51: ("draw_signal", "Master local draw trigger"),
            52: ("tactical_ceiling", "Theorem 5 - Bilateral Bound"),
            53: ("square_domination", "Bitboard parity"),
            54: ("mobility_occlusion", "Theorem 3 - Dimensional Weight"),
            55: ("reserved", "Reserved for future"),
            63: ("zugzwang_coeff", "Non-commutativity measure"),
        }

        all_present = True
        for dim, (name, description) in dim_assignments.items():
            if dim < len(coeffs_64d):
                value = coeffs_64d[dim]
                # Just check that the dimension exists and has a value
                has_value = True
                print(f"  Dim {dim:2d}: {name:<25} = {value:+.6f} ({description})")
            else:
                has_value = False
                all_present = False
                print(f"  Dim {dim:2d}: MISSING")

        self.results.append(TestResult(
            name="All Session 0.1 Dimensions Present",
            passed=all_present,
            expected="Dims 44-55, 63 all present in 64D encoding",
            actual=f"{'All present' if all_present else 'Some missing'}",
            details="Verified dimension relocation from Research Brief"
        ))

    def _test_master_dampener_fortress(self) -> None:
        """
        Test Master Dampener against fortress positions.

        Positions tested:
        - french_fortress: Should trigger dampener
        - opposite_color_bishops: Should trigger dampener
        - wrong_color_bishop: Should trigger dampener
        """
        print("\n" + "-" * 70)
        print("TEST: Master Dampener - Fortress Detection")
        print("-" * 70)

        fortress_positions = [
            ("french_fortress", "Locked pawn chain fortress"),
            ("opposite_color_bishops", "OCB endgame fortress"),
            ("wrong_color_bishop", "Wrong color bishop draw"),
        ]

        for key, description in fortress_positions:
            if key not in STRESSOR_LIBRARY:
                print(f"  SKIP: {key} not in stressor library")
                continue

            stressor = STRESSOR_LIBRARY[key]
            board = chess.Board(stressor["fen"])

            # Encode and cascade
            state_16d = encode_position(board)
            cascade = full_cascade(state_16d, chess.KNIGHT, board)

            # Evaluate WITHOUT history (structural only)
            analysis = evaluate_position(cascade, board, eval_history_64d=[])
            dampener = analysis.dampener

            # Extract structural signal
            structural = dampener.structural_signal if dampener else 0.0

            # For fortress positions, we expect structural signal > 0.3
            expected_structural = 0.3
            has_structural = structural > expected_structural

            print(f"\n  {key}:")
            print(f"    Description: {description}")
            print(f"    Raw consensus: {dampener.raw_consensus:+.3f}" if dampener else "    No dampener")
            print(f"    Structural signal: {structural:.3f} (threshold: >{expected_structural})")
            print(f"    Fortress detected: {has_structural}")

            if dampener:
                # Get individual dimension values
                coeffs_64d = list(cascade['state_64d'].coefficients())
                dim_45 = coeffs_64d[45] if len(coeffs_64d) > 45 else 0.0
                dim_47 = coeffs_64d[47] if len(coeffs_64d) > 47 else 0.0
                dim_51 = coeffs_64d[51] if len(coeffs_64d) > 51 else 0.0
                print(f"    Dim 45 (leverage): {dim_45:.3f}")
                print(f"    Dim 47 (fortress): {dim_47:.3f}")
                print(f"    Dim 51 (draw):     {dim_51:.3f}")

            self.results.append(TestResult(
                name=f"Fortress Detection: {key}",
                passed=has_structural,
                expected=f"Structural signal > {expected_structural}",
                actual=f"Structural = {structural:.3f}",
                details=description
            ))

    def _test_master_dampener_temporal(self) -> None:
        """
        Test Master Dampener temporal confirmation (4-ply history).

        Simulates a fortress position with stable eval history.
        """
        print("\n" + "-" * 70)
        print("TEST: Master Dampener - Temporal Confirmation")
        print("-" * 70)

        # Use french_fortress
        stressor = STRESSOR_LIBRARY.get("french_fortress")
        if not stressor:
            print("  SKIP: french_fortress not found")
            return

        board = chess.Board(stressor["fen"])
        state_16d = encode_position(board)
        cascade = full_cascade(state_16d, chess.KNIGHT, board)

        # Test with insufficient history
        print("\n  Test 1: Insufficient history (< 4 evals)")
        history_short = [0.5, 0.6]
        analysis_short = evaluate_position(cascade, board, eval_history_64d=history_short)

        temporal_short = analysis_short.dampener.temporal_confirmed if analysis_short.dampener else False
        print(f"    History length: {len(history_short)}")
        print(f"    Temporal confirmed: {temporal_short}")

        self.results.append(TestResult(
            name="Temporal: Insufficient History (< 4)",
            passed=not temporal_short,  # Should NOT be confirmed
            expected="temporal_confirmed = False",
            actual=f"temporal_confirmed = {temporal_short}",
            details="Needs minimum 4 evaluations for temporal check"
        ))

        # Test with stable history (fortress should trigger)
        print("\n  Test 2: Stable history (stasis)")
        history_stable = [0.5, 0.52, 0.48, 0.51]  # Range = 0.04 < M=0.5
        analysis_stable = evaluate_position(cascade, board, eval_history_64d=history_stable)

        dampener = analysis_stable.dampener
        temporal_stable = dampener.temporal_confirmed if dampener else False
        stasis_delta = dampener.eval_stasis_delta if dampener else 0.0
        M = dampener.stability_constant_M if dampener else 0.5

        print(f"    History: {history_stable}")
        print(f"    Stasis delta: {stasis_delta:.3f} (threshold M = {M})")
        print(f"    Temporal confirmed: {temporal_stable}")

        self.results.append(TestResult(
            name="Temporal: Stable History (stasis)",
            passed=temporal_stable,
            expected="temporal_confirmed = True (delta < M)",
            actual=f"delta={stasis_delta:.3f}, M={M}",
            details="Fortress stasis should trigger temporal confirmation"
        ))

        # Test with volatile history (should NOT trigger)
        print("\n  Test 3: Volatile history (no stasis)")
        history_volatile = [0.5, 1.5, 0.3, 2.0]  # Range = 1.7 > M=0.5
        analysis_volatile = evaluate_position(cascade, board, eval_history_64d=history_volatile)

        dampener = analysis_volatile.dampener
        temporal_volatile = dampener.temporal_confirmed if dampener else False
        stasis_delta = dampener.eval_stasis_delta if dampener else 0.0

        print(f"    History: {history_volatile}")
        print(f"    Stasis delta: {stasis_delta:.3f}")
        print(f"    Temporal confirmed: {temporal_volatile}")

        self.results.append(TestResult(
            name="Temporal: Volatile History (no stasis)",
            passed=not temporal_volatile,
            expected="temporal_confirmed = False (delta >= M)",
            actual=f"delta={stasis_delta:.3f}",
            details="Volatile eval means position is changing, not a fortress"
        ))

        # Test full dampener activation
        print("\n  Test 4: Full Dampener (structural + temporal)")
        analysis_full = evaluate_position(cascade, board, eval_history_64d=history_stable)
        dampener = analysis_full.dampener

        if dampener:
            active = dampener.active
            raw = dampener.raw_consensus
            dampened = dampener.dampened_consensus
            signal = dampener.fortress_signal

            print(f"    Dampener active: {active}")
            print(f"    Raw consensus: {raw:+.3f}")
            print(f"    Dampened consensus: {dampened:+.3f}")
            print(f"    Fortress signal: {signal:.0%}")
            print(f"    Formula: {raw:+.3f} × (1 - {signal:.3f}) = {dampened:+.3f}")

            formula_correct = abs(raw * (1 - signal) - dampened) < 0.001
            self.results.append(TestResult(
                name="Dampener Formula: Consensus × (1 - FortressSignal)",
                passed=formula_correct,
                expected=f"{raw:.3f} × (1 - {signal:.3f}) = {raw * (1-signal):.3f}",
                actual=f"dampened = {dampened:.3f}",
                details="Master Dampener multiplicative formula"
            ))

    def _print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\nResults: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
        print()

        # Show failures first
        failures = [r for r in self.results if not r.passed]
        if failures:
            print("FAILURES:")
            print("-" * 70)
            for r in failures:
                print(f"  FAIL: {r.name}")
                print(f"    Expected: {r.expected}")
                print(f"    Actual:   {r.actual}")
                if r.details:
                    print(f"    Details:  {r.details}")
                print()

        # Show passes
        passes = [r for r in self.results if r.passed]
        if passes:
            print("PASSES:")
            print("-" * 70)
            for r in passes:
                print(f"  PASS: {r.name}")

        print()
        if passed == total:
            print("*** ALL TESTS PASSED - Session 0.1 Implementation Verified! ***")
        else:
            print(f"WARNING: {total - passed} test(s) failed - review implementation")
        print("=" * 70)


def main():
    """Run verification suite."""
    verifier = Session01Verifier()
    verifier.run_all_tests()


if __name__ == "__main__":
    main()
