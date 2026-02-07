"""
Multi-Dimensional Evaluator - ZDTP Chess Position Evaluation

Evaluates chess positions at 16D (tactical), 32D (positional), and 64D (strategic)
dimensional levels using the Zero Divisor Transmission Protocol.

Key Insight: Different dimensional spaces emphasize different chess features.

16D Tactical Layer:
- Focus: Immediate threats, material balance, king safety
- Weight: Material > Mobility > Threats

32D Positional Layer:  
- Focus: Pawn structure, piece coordination, central control
- Weight: Structure > Coordination > Space
- Added info: Gateway pattern influence in coefficients 16-31

64D Strategic Layer:
- Focus: Long-term planning, endgame potential, prophylaxis
- Weight: Advancement > Complexity > Gateway harmony
- Added info: Dual gateway influence in coefficients 32-63

Session 0.1 Upgrade: Master Dampener (Fortress Draw Detection)
- Grounded in Lean 4 Stability Theorem: |C[f]| <= M * ||f||_1
- When 64D evaluation stasis is detected (delta_V64 < M over 4 evals)
  AND structural fortress indicators fire (dims 45, 47, 51),
  the Master Dampener pulls consensus toward 0.0 (draw).
- Formula: Final Eval = Consensus * (1 - FortressSignal)

Reference: Paul Chavez's published research on framework-independent
zero divisor patterns and dimensional information theory.
"""

import chess
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

try:
    from hypercomplex import Sedenion, Pathion, Chingon
except ImportError as e:
    raise ImportError(
        "hypercomplex library not found. Install with: pip install hypercomplex>=0.3.4"
    ) from e

try:
    from .dimensional_encoder import decode_features
except ImportError:
    from dimensional_encoder import decode_features


@dataclass
class TacticalEvaluation:
    """16D Tactical layer evaluation result."""
    score: float
    material_balance: float
    king_safety_diff: float
    mobility_diff: float
    immediate_threats: List[str]
    best_move_hint: Optional[str]
    reasoning: str


@dataclass
class PositionalEvaluation:
    """32D Positional layer evaluation result."""
    score: float
    pawn_structure: float
    piece_coordination: float
    center_control: float
    gateway_influence: float
    best_move_hint: Optional[str]
    reasoning: str


@dataclass
class StrategicEvaluation:
    """64D Strategic layer evaluation result."""
    score: float
    pawn_advancement: float
    endgame_potential: float
    position_complexity: float
    gateway_harmony: float
    long_term_plan: Optional[str]
    reasoning: str


@dataclass
class FortressDampenerResult:
    """Result of Master Dampener (Fortress Draw Detection).
    
    Grounded in Lean 4 Stability Theorem (Thm 2):
    When the 64D manifold ceases to evolve, the Chavez Transform
    is bounded by the Stability Constant M = (||P||^2 + ||Q||^2) * sqrt(pi/alpha).
    
    The dampener fires when:
    1. Structural signals detect fortress topology (dims 45, 47, 51)
    2. Temporal confirmation: eval stasis over 4+ evaluations
    """
    active: bool = False                    # Is dampener firing?
    structural_signal: float = 0.0          # 0-1 from dims 45, 47, 51
    temporal_confirmed: bool = False         # 4-eval stasis check passed?
    fortress_signal: float = 0.0            # Final signal (structural * temporal)
    raw_consensus: float = 0.0              # Pre-dampener consensus
    dampened_consensus: float = 0.0         # Post-dampener consensus
    eval_history_count: int = 0             # Number of evals in history
    eval_stasis_delta: float = 0.0          # max-min of last 4 evals
    stability_constant_M: float = 0.5       # Threshold for stasis
    reason: str = ""                        # Human-readable explanation


@dataclass
class MultiDimensionalAnalysis:
    """Complete multi-dimensional analysis of a position."""
    tactical_16d: TacticalEvaluation
    positional_32d: PositionalEvaluation
    strategic_64d: StrategicEvaluation
    consensus_score: float
    recommended_move: Optional[str]
    gateway_used: str
    overall_assessment: str
    dampener: Optional[FortressDampenerResult] = None


class MultidimensionalEvaluator:
    """
    Evaluates chess positions across dimensional spaces.
    
    Uses ZDTP cascade results to provide tactical, positional, and
    strategic evaluations based on dimensional encodings.
    """
    
    # Evaluation weights for each dimensional layer
    TACTICAL_WEIGHTS = {
        'material': 3.0,
        'mobility': 2.0,
        'king_safety': 1.5,
        'development': 1.0
    }
    
    POSITIONAL_WEIGHTS = {
        'pawn_structure': 2.5,
        'center_control': 2.0,
        'piece_activity': 2.0,
        'gateway_influence': 1.0
    }
    
    STRATEGIC_WEIGHTS = {
        'pawn_advancement': 2.0,
        'complexity': 1.5,
        'gateway_harmony': 1.0
    }
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def evaluate_cascade(
        self,
        cascade_results: Dict,
        board: chess.Board,
        eval_history_64d: Optional[List[float]] = None
    ) -> MultiDimensionalAnalysis:
        """
        Evaluate a position using complete ZDTP cascade results.
        
        Args:
            cascade_results: Results from dimensional_portal.full_cascade()
            board: Current chess board position
            eval_history_64d: Previous consensus scores for Master Dampener
                              temporal confirmation. None or [] disables temporal check.
            
        Returns:
            MultiDimensionalAnalysis with all three dimensional evaluations
        """
        # Extract dimensional states
        state_16d = cascade_results['state_16d']
        state_32d = cascade_results['state_32d']
        state_64d = cascade_results['state_64d']
        gateway_piece = cascade_results['gateway_piece']
        
        # Evaluate at each dimensional level
        tactical = self._evaluate_16d_tactical(state_16d, state_32d, board)
        positional = self._evaluate_32d_positional(state_32d, board,
                                                   cascade_results['portal_16_32'])
        strategic = self._evaluate_64d_strategic(state_64d, board,
                                                 cascade_results['portal_32_64'])
        
        # Build consensus (now with Master Dampener)
        consensus_score, dampener = self._build_consensus(
            tactical, positional, strategic,
            state_64d=state_64d,
            eval_history_64d=eval_history_64d or []
        )
        
        # Determine recommended move
        recommended_move = self._recommend_move(tactical, positional, strategic, board)
        
        # Overall assessment (uses dampened consensus)
        assessment = self._generate_assessment(
            tactical, positional, strategic, consensus_score, dampener
        )
        
        return MultiDimensionalAnalysis(
            tactical_16d=tactical,
            positional_32d=positional,
            strategic_64d=strategic,
            consensus_score=consensus_score,
            recommended_move=recommended_move,
            gateway_used=gateway_piece,
            overall_assessment=assessment,
            dampener=dampener
        )
    
    def _evaluate_16d_tactical(
        self,
        state_16d: Sedenion,
        state_32d: Pathion,
        board: chess.Board
    ) -> TacticalEvaluation:
        """
        Evaluate position at 16D tactical level.

        Focus: Immediate threats, material, king safety, mobility

        BUG FIX (2025-11-15): Now integrates hanging piece detection from 32D encoding
        to prevent catastrophic blunders like Rb1, Qb5 that hung pieces.
        """
        # Decode features from 16D encoding
        features = decode_features(state_16d)

        # Material balance (primary tactical factor)
        material_balance = features['material_balance']

        # King safety differential
        king_safety_diff = features['king_safety_white'] - features['king_safety_black']

        # Mobility differential
        mobility_diff = features['mobility_white'] - features['mobility_black']

        # Development differential
        development_diff = features['development_white'] - features['development_black']

        # CRITICAL FIX: Extract hanging piece penalty from 32D encoding
        # Dims 16-19 of 32D pathion contain tactical features:
        # [hanging_pieces, pins, forks, exchange_eval]
        # This was computed but not used - now we integrate it!
        coeffs_32d = list(state_32d.coefficients())
        hanging_penalty = coeffs_32d[16]  # Dimension 16: hanging piece score

        # Compute tactical score (WITH HANGING PIECE DETECTION!)
        tactical_score = (
            self.TACTICAL_WEIGHTS['material'] * material_balance +
            self.TACTICAL_WEIGHTS['mobility'] * mobility_diff +
            self.TACTICAL_WEIGHTS['king_safety'] * king_safety_diff +
            self.TACTICAL_WEIGHTS['development'] * development_diff +
            5.0 * hanging_penalty  # CRITICAL: Heavy penalty for hanging pieces!
        )
        
        # Identify immediate threats
        threats = self._identify_threats(board, features)
        
        # Generate tactical reasoning
        reasoning = self._generate_tactical_reasoning(
            material_balance, king_safety_diff, mobility_diff, threats
        )
        
        # Best move hint (tactical focus)
        best_move_hint = self._tactical_move_hint(board, features, threats)
        
        return TacticalEvaluation(
            score=tactical_score,
            material_balance=material_balance,
            king_safety_diff=king_safety_diff,
            mobility_diff=mobility_diff,
            immediate_threats=threats,
            best_move_hint=best_move_hint,
            reasoning=reasoning
        )
    
    def _evaluate_32d_positional(
        self,
        state_32d: Pathion,
        board: chess.Board,
        portal_metadata: Dict
    ) -> PositionalEvaluation:
        """
        Evaluate position at 32D positional level.
        
        Focus: Pawn structure, piece coordination, center control
        Added dimension: Gateway pattern influence
        """
        # Extract coefficients from 32D pathion
        coeffs = list(state_32d.coefficients())
        
        # First 16 coefficients: Original chess features (preserved from 16D)
        chess_features = coeffs[:16]
        
        # Second 16 coefficients: Gateway pattern encoding
        gateway_features = coeffs[16:32]
        
        # Pawn structure (coefficient 1)
        pawn_structure = chess_features[1]
        
        # Center control (coefficient 4)
        center_control = chess_features[4]
        
        # Piece activity (coefficient 9)
        piece_coordination = chess_features[9]
        
        # Gateway influence (measure interaction between chess and gateway)
        gateway_influence = self._compute_gateway_influence(gateway_features)
        
        # Compute positional score
        positional_score = (
            self.POSITIONAL_WEIGHTS['pawn_structure'] * pawn_structure +
            self.POSITIONAL_WEIGHTS['center_control'] * center_control +
            self.POSITIONAL_WEIGHTS['piece_activity'] * piece_coordination +
            self.POSITIONAL_WEIGHTS['gateway_influence'] * gateway_influence
        )
        
        # Generate positional reasoning
        reasoning = self._generate_positional_reasoning(
            pawn_structure, center_control, piece_coordination, 
            portal_metadata['gateway_pattern']
        )
        
        # Best move hint (positional focus)
        best_move_hint = self._positional_move_hint(
            board, chess_features, gateway_features
        )
        
        return PositionalEvaluation(
            score=positional_score,
            pawn_structure=pawn_structure,
            piece_coordination=piece_coordination,
            center_control=center_control,
            gateway_influence=gateway_influence,
            best_move_hint=best_move_hint,
            reasoning=reasoning
        )
    
    def _evaluate_64d_strategic(
        self,
        state_64d: Chingon,
        board: chess.Board,
        portal_metadata: Dict
    ) -> StrategicEvaluation:
        """
        Evaluate position at 64D strategic level.
        
        Focus: Long-term planning, pawn advancement, endgame potential
        Added dimensions: Dual gateway harmony
        """
        # Extract coefficients from 64D chingon
        coeffs = list(state_64d.coefficients())
        
        # First 16: Original chess features
        chess_features = coeffs[:16]
        
        # 16-32: Gateway from 16D→32D transition
        gateway_32d = coeffs[16:32]
        
        # 32-64: Gateway from 32D→64D transition  
        gateway_64d = coeffs[32:64]
        
        # Pawn advancement differential (coefficients 10-11)
        pawn_advancement = chess_features[10] - chess_features[11]
        
        # Position complexity (coefficient 15)
        complexity = chess_features[15]
        
        # Gateway harmony (measure coherence between dual gateways)
        gateway_harmony = self._compute_gateway_harmony(gateway_32d, gateway_64d)
        
        # Endgame potential (based on material simplification)
        endgame_potential = self._compute_endgame_potential(chess_features)
        
        # Compute strategic score
        strategic_score = (
            self.STRATEGIC_WEIGHTS['pawn_advancement'] * pawn_advancement +
            self.STRATEGIC_WEIGHTS['complexity'] * complexity +
            self.STRATEGIC_WEIGHTS['gateway_harmony'] * gateway_harmony
        )
        
        # Generate strategic reasoning
        reasoning = self._generate_strategic_reasoning(
            pawn_advancement, complexity, endgame_potential
        )
        
        # Long-term plan
        long_term_plan = self._identify_strategic_plan(board, chess_features)
        
        return StrategicEvaluation(
            score=strategic_score,
            pawn_advancement=pawn_advancement,
            endgame_potential=endgame_potential,
            position_complexity=complexity,
            gateway_harmony=gateway_harmony,
            long_term_plan=long_term_plan,
            reasoning=reasoning
        )
    
    # ═══════════════════════════════════════════════════════════════
    #  MASTER DAMPENER CONSTANTS (Session 0.1, Lean 4 Grounded)
    # ═══════════════════════════════════════════════════════════════

    # Stability Constant M: threshold for eval stasis detection.
    # From Lean 4 Stability Theorem: |C[f]| <= M * ||f||_1
    # M = (||P||^2 + ||Q||^2) * sqrt(pi/alpha)
    # For typical ZDTP gateway norms ~1.0 and alpha=1.0:
    #   M ≈ 2.0 * sqrt(pi) ≈ 3.54
    # We use a practical threshold: if the range of the last 4
    # consensus scores is less than M, the position is "static".
    STABILITY_CONSTANT_M = 0.5  # Practical threshold (tighter than theoretical)

    # Minimum evaluations before temporal check can activate
    MIN_EVALS_FOR_DAMPENER = 4

    # Maximum history entries to retain
    MAX_EVAL_HISTORY = 10

    # Minimum tactical score to consider dampening
    # (only dampen when one side appears to have advantage)
    DAMPENER_TACTICAL_THRESHOLD = 1.5

    def _build_consensus(
        self,
        tactical: TacticalEvaluation,
        positional: PositionalEvaluation,
        strategic: StrategicEvaluation,
        state_64d: Optional[Chingon] = None,
        eval_history_64d: Optional[List[float]] = None
    ) -> Tuple[float, FortressDampenerResult]:
        """
        Build consensus score from three dimensional evaluations,
        then apply Master Dampener if fortress conditions are met.

        Session 0.1 Upgrade: Master Dampener (Fortress Draw Detection)

        The dampener has two independent signal paths:
        1. STRUCTURAL: Dims 45 (leverage_deficiency), 47 (fortress_structural),
           51 (draw_signal) from the 64D Chingon encoding
        2. TEMPORAL: Eval stasis over the last 4 evaluations (delta < M)

        Both must fire for full dampening. Structural alone gives partial.

        Formula: Final = Consensus × (1 - FortressSignal)

        Weights: 16D (50%), 32D (30%), 64D (20%)

        CRITICAL SAFETY OVERRIDE (2025-11-20):
        If tactical score is catastrophically bad (< -10), consensus cannot
        be positive.
        """
        # ── Step 1: Raw consensus (unchanged formula) ──
        raw_consensus = (
            0.5 * tactical.score +
            0.3 * positional.score +
            0.2 * strategic.score
        )

        # SAFETY OVERRIDE: If tactical disaster, cap consensus
        if tactical.score < -10.0:
            raw_consensus = min(raw_consensus, tactical.score)

        # ── Step 2: Extract fortress structural signals from 64D ──
        structural_signal = 0.0
        dim_45 = 0.0  # leverage_deficiency
        dim_47 = 0.0  # fortress_structural
        dim_51 = 0.0  # draw_signal

        if state_64d is not None:
            try:
                coeffs = list(state_64d.coefficients())
                # Dims 44-47: Verified positional features
                # dim 44 = space_vs_material, dim 45 = leverage_deficiency
                # dim 46 = activity_vs_structure, dim 47 = fortress_structural
                dim_45 = coeffs[45] if len(coeffs) > 45 else 0.0
                dim_47 = coeffs[47] if len(coeffs) > 47 else 0.0
                # Dims 48-51: Draw detection features
                # dim 48 = OCB, dim 49 = insufficient, dim 50 = no passers
                # dim 51 = draw_signal (master local draw indicator)
                dim_51 = coeffs[51] if len(coeffs) > 51 else 0.0

                # Structural signal: weighted combination
                # leverage_deficiency is the strongest indicator
                structural_signal = (
                    0.40 * dim_45 +   # Pawn energy absence
                    0.35 * dim_47 +   # Composite fortress structure
                    0.25 * dim_51     # Draw indicators (OCB, insuf, etc.)
                )
                structural_signal = max(0.0, min(1.0, structural_signal))
            except Exception:
                structural_signal = 0.0

        # ── Step 3: Temporal confirmation (eval stasis check) ──
        temporal_confirmed = False
        eval_stasis_delta = float('inf')
        history_count = len(eval_history_64d) if eval_history_64d else 0

        if eval_history_64d and len(eval_history_64d) >= self.MIN_EVALS_FOR_DAMPENER:
            last_n = eval_history_64d[-self.MIN_EVALS_FOR_DAMPENER:]
            eval_stasis_delta = max(last_n) - min(last_n)
            # Stasis: evaluation range is below stability constant
            temporal_confirmed = eval_stasis_delta < self.STABILITY_CONSTANT_M

        # ── Step 4: Compute fortress signal and apply dampener ──
        # Fortress signal requires:
        #   - Structural signal > 0.3 (meaningful fortress indicators)
        #   - abs(raw_consensus) > threshold (there IS an advantage to dampen)
        fortress_signal = 0.0
        dampener_active = False
        reason = ""

        if structural_signal > 0.3 and abs(raw_consensus) > self.DAMPENER_TACTICAL_THRESHOLD:
            if temporal_confirmed:
                # FULL DAMPENER: Both structural and temporal confirmed
                fortress_signal = structural_signal
                dampener_active = True
                reason = (
                    f"FORTRESS DETECTED: Structural signal {structural_signal:.2f} "
                    f"(dim45={dim_45:.2f}, dim47={dim_47:.2f}, dim51={dim_51:.2f}) "
                    f"+ Temporal stasis confirmed (delta={eval_stasis_delta:.3f} < "
                    f"M={self.STABILITY_CONSTANT_M}). "
                    f"Master Dampener active: eval pulled toward draw."
                )
            elif history_count < self.MIN_EVALS_FOR_DAMPENER:
                # PARTIAL: Structural fires but insufficient history
                # Apply half-strength dampening as a warning signal
                fortress_signal = structural_signal * 0.5
                dampener_active = True
                reason = (
                    f"FORTRESS PROBABLE: Structural signal {structural_signal:.2f} "
                    f"but insufficient eval history ({history_count}/{self.MIN_EVALS_FOR_DAMPENER}). "
                    f"Half-strength dampener active. Run more evaluations to confirm."
                )
            else:
                # Structural fires but temporal does NOT confirm (eval is changing)
                fortress_signal = 0.0
                reason = (
                    f"Fortress structure detected (signal={structural_signal:.2f}) "
                    f"but position is evolving (delta={eval_stasis_delta:.3f} >= "
                    f"M={self.STABILITY_CONSTANT_M}). Dampener NOT active."
                )
        elif structural_signal > 0.0:
            reason = (
                f"Mild fortress indicators (signal={structural_signal:.2f}) "
                f"below activation threshold (0.3). No dampening."
            )
        else:
            reason = "No fortress indicators detected."

        # Apply: Final Eval = Consensus × (1 - FortressSignal)
        dampened_consensus = raw_consensus * (1.0 - fortress_signal)

        dampener = FortressDampenerResult(
            active=dampener_active,
            structural_signal=structural_signal,
            temporal_confirmed=temporal_confirmed,
            fortress_signal=fortress_signal,
            raw_consensus=raw_consensus,
            dampened_consensus=dampened_consensus,
            eval_history_count=history_count,
            eval_stasis_delta=eval_stasis_delta if eval_stasis_delta != float('inf') else 0.0,
            stability_constant_M=self.STABILITY_CONSTANT_M,
            reason=reason
        )

        return dampened_consensus, dampener
    
    def _recommend_move(
        self,
        tactical: TacticalEvaluation,
        positional: PositionalEvaluation,
        strategic: StrategicEvaluation,
        board: chess.Board
    ) -> Optional[str]:
        """
        Recommend a move based on dimensional consensus.
        
        Priority:
        1. If tactical threats exist, address them
        2. Otherwise, follow positional recommendation
        3. If unclear, use strategic guidance
        """
        # Check for immediate tactical threats
        if tactical.immediate_threats:
            return tactical.best_move_hint
        
        # No immediate threats - use positional recommendation
        if positional.best_move_hint:
            return positional.best_move_hint
        
        # Fall back to strategic if available
        return strategic.long_term_plan
    
    def _generate_assessment(
        self,
        tactical: TacticalEvaluation,
        positional: PositionalEvaluation,
        strategic: StrategicEvaluation,
        consensus_score: float,
        dampener: Optional[FortressDampenerResult] = None
    ) -> str:
        """Generate overall position assessment."""
        # Determine who's better
        if consensus_score > 2.0:
            advantage = "White has a significant advantage"
        elif consensus_score > 0.5:
            advantage = "White has a slight advantage"
        elif consensus_score < -2.0:
            advantage = "Black has a significant advantage"
        elif consensus_score < -0.5:
            advantage = "Black has a slight advantage"
        else:
            advantage = "Position is approximately equal"

        assessment = f"{advantage}. {tactical.reasoning}"

        # Add dampener note if active
        if dampener and dampener.active:
            raw = dampener.raw_consensus
            final = dampener.dampened_consensus
            signal = dampener.fortress_signal
            assessment += (
                f" [MASTER DAMPENER: Raw eval {raw:+.2f} → {final:+.2f} "
                f"(fortress signal {signal:.0%})]"
            )

        return assessment
    
    # Helper methods
    
    def _identify_threats(self, board: chess.Board, features: Dict) -> List[str]:
        """Identify immediate tactical threats."""
        threats = []
        
        # Check for checks
        if board.is_check():
            threats.append("Check")
        
        # Check for attacks on high-value pieces
        # (simplified - would need board analysis)
        
        # Material imbalance indicates captures
        if abs(features['material_balance']) > 1.0:
            if features['material_balance'] > 0:
                threats.append("Material advantage (White)")
            else:
                threats.append("Material advantage (Black)")
        
        return threats
    
    def _generate_tactical_reasoning(
        self,
        material: float,
        king_safety: float,
        mobility: float,
        threats: List[str]
    ) -> str:
        """Generate human-readable tactical reasoning."""
        parts = []

        # P0-3 FIX: Clear material labeling with perspective
        if abs(material) > 0.5:
            if material > 0:
                parts.append(f"Material: {material:+.1f} (White advantage)")
            else:
                parts.append(f"Material: {material:+.1f} (Black advantage)")

        if abs(mobility) > 0.1:
            if mobility > 0:
                parts.append("White has better mobility")
            else:
                parts.append("Black has better mobility")

        if threats:
            parts.append(f"Threats: {', '.join(threats)}")

        if not parts:
            parts.append("Balanced tactical position")
        
        return ". ".join(parts)
    
    def _tactical_move_hint(
        self,
        board: chess.Board,
        features: Dict,
        threats: List[str]
    ) -> Optional[str]:
        """Generate tactical move hint."""
        # Simplified - would analyze specific moves
        if "Check" in threats:
            return "Respond to check"
        elif features['mobility_white'] - features['mobility_black'] > 0.2:
            return "Maximize mobility advantage"
        return None
    
    def _compute_gateway_influence(self, gateway_features: List[float]) -> float:
        """
        Compute influence of gateway pattern on position. 
        
        Gateway features are in coefficients 16-31 of 32D pathion.
        Measure their magnitude and coherence.
        """
        # Simple metric: average magnitude of gateway coefficients
        if not gateway_features:
            return 0.0
        
        magnitude = sum(abs(f) for f in gateway_features) / len(gateway_features)
        return magnitude
    
    def _generate_positional_reasoning(
        self,
        pawn_structure: float,
        center_control: float,
        piece_coordination: float,
        gateway_pattern: Dict
    ) -> str:
        """Generate human-readable positional reasoning."""
        parts = []
        
        if pawn_structure > 0.2:
            parts.append("Good pawn structure")
        elif pawn_structure < -0.2:
            parts.append("Weak pawn structure")
        
        if center_control > 0.3:
            parts.append("Strong center control")
        
        if piece_coordination > 0.2:
            parts.append("Well-coordinated pieces")
        
        parts.append(f"Gateway: {gateway_pattern['name']}")
        
        return ". ".join(parts)
    
    def _positional_move_hint(
        self,
        board: chess.Board,
        chess_features: List[float],
        gateway_features: List[float]
    ) -> Optional[str]:
        """Generate positional move hint."""
        # Simplified
        if chess_features[4] < 0:  # center_control
            return "Improve center control"
        return "Maintain positional advantage"
    
    def _compute_gateway_harmony(
        self,
        gateway_32d: List[float],
        gateway_64d: List[float]
    ) -> float:
        """
        Measure harmony between dual gateway patterns. 
        
        High harmony = gateways reinforce each other
        Low harmony = gateways in tension
        """
        # Simplified: correlation between gateway vectors
        if not gateway_32d or not gateway_64d:
            return 0.0
        
        # Dot product normalized
        dot_product = sum(a * b for a, b in zip(gateway_32d, gateway_64d))
        norm_32d = sum(a*a for a in gateway_32d) ** 0.5
        norm_64d = sum(b*b for b in gateway_64d) ** 0.5
        
        if norm_32d == 0 or norm_64d == 0:
            return 0.0
        
        correlation = dot_product / (norm_32d * norm_64d)
        return correlation
    
    def _compute_endgame_potential(self, chess_features: List[float]) -> float:
        """
        Estimate endgame potential based on material simplification.

        BUG FIX (2025-11-20): This function was using material_balance (difference)
        instead of total material. This caused "Approaching endgame" on move 1
        because material_balance = 0 at start.

        Since chess_features doesn't contain total material, we use complexity
        (coefficient 15) as a proxy - complex positions have more pieces.
        """
        # Use position complexity as proxy for material count
        # High complexity (0.7-1.0) = opening/middlegame (many pieces)
        # Low complexity (0.0-0.3) = endgame (few pieces)
        complexity = chess_features[15]

        # Inverse: low complexity = high endgame potential
        # complexity 0.9 → endgame_potential 0.1 (not endgame)
        # complexity 0.3 → endgame_potential 0.7 (approaching endgame)
        endgame_potential = max(0, 1.0 - complexity)
        return endgame_potential
    
    def _generate_strategic_reasoning(
        self,
        pawn_advancement: float,
        complexity: float,
        endgame_potential: float
    ) -> str:
        """Generate human-readable strategic reasoning."""
        parts = []
        
        if pawn_advancement > 0.2:
            parts.append("White's pawns are advancing")
        elif pawn_advancement < -0.2:
            parts.append("Black's pawns are advancing")
        
        if endgame_potential > 0.6:
            parts.append("Approaching endgame")
        
        if complexity > 0.7:
            parts.append("Complex position with many possibilities")
        
        if not parts:
            parts.append("Balanced strategic position")
        
        return ". ".join(parts)
    
    def _identify_strategic_plan(
        self,
        board: chess.Board,
        chess_features: List[float]
    ) -> Optional[str]:
        """Identify long-term strategic plan."""
        # Simplified - would require deeper analysis
        endgame_potential = self._compute_endgame_potential(chess_features)
        
        if endgame_potential > 0.6:
            return "Simplify to favorable endgame"
        elif chess_features[10] > chess_features[11]:  # pawn advancement
            return "Push passed pawns"
        else:
            return "Improve piece positioning"


# Module-level convenience functions

_evaluator = None

def evaluate_position(
    cascade_results: Dict,
    board: chess.Board,
    eval_history_64d: Optional[List[float]] = None
) -> MultiDimensionalAnalysis:
    """
    Evaluate a position using ZDTP cascade results.
    
    Convenience function for quick evaluation.
    
    Args:
        cascade_results: Results from dimensional_portal.full_cascade()
        board: Current chess board position
        eval_history_64d: Previous consensus scores for Master Dampener.
                          Pass the game's eval history list for fortress detection.
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = MultidimensionalEvaluator()
    return _evaluator.evaluate_cascade(cascade_results, board, eval_history_64d)


# Standalone test
if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-DIMENSIONAL EVALUATOR TEST")
    print("=" * 60)
    print()
    
    import chess
    from dimensional_encoder import encode_position
    from dimensional_portal import full_cascade
    
    # Test with starting position
    print("Test: Starting position evaluation")
    board = chess.Board()
    
    # Encode and cascade (with intelligent tactical/strategic features!)
    state_16d = encode_position(board)
    cascade = full_cascade(state_16d, chess.KNIGHT, board)
    
    # Evaluate
    evaluator = MultidimensionalEvaluator()
    analysis = evaluator.evaluate_cascade(cascade, board)
    
    print(f"\n16D Tactical Layer:")
    print(f"  Score: {analysis.tactical_16d.score:.2f}")
    print(f"  {analysis.tactical_16d.reasoning}")
    
    print(f"\n32D Positional Layer:")
    print(f"  Score: {analysis.positional_32d.score:.2f}")
    print(f"  {analysis.positional_32d.reasoning}")
    
    print(f"\n64D Strategic Layer:")
    print(f"  Score: {analysis.strategic_64d.score:.2f}")
    print(f"  {analysis.strategic_64d.reasoning}")
    
    print(f"\nConsensus: {analysis.consensus_score:.2f}")
    print(f"Overall: {analysis.overall_assessment}")
    print(f"Gateway: {analysis.gateway_used}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
