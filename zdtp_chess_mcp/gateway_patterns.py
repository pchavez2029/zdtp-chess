"""
Gateway Patterns Module - The Canonical Six

Maps chess piece types to framework-independent zero divisor patterns
discovered and verified in Paul Chavez's research (2025).

The Canonical Six are the only known zero divisor patterns that work
identically in both Cayley-Dickson (non-associative) and Clifford
(associative) algebraic frameworks, verified across 16D-256D with
machine precision (~10^-15).

Reference: "Framework-Independent Zero Divisor Patterns in
Higher-Dimensional Cayley-Dickson Algebras" - Chavez (2025)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import chess

# Add CAILculator MCP to path (most updated version with Clifford Element updates)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "cailculator-mcp"))

try:
    from hypercomplex import Sedenion
except ImportError as e:
    raise ImportError(
        "CAILculator hypercomplex library not found. "
        "Ensure CAILculator project is in parent directory."
    ) from e


class GatewayPatternLoader:
    """Loads and manages the Canonical Six gateway patterns."""

    def __init__(self, patterns_file: Optional[str] = None):
        """
        Initialize pattern loader.

        Args:
            patterns_file: Path to canonical_six_patterns.json
                          If None, uses default location in data/
        """
        if patterns_file is None:
            # Default: look in data/ at package root
            base_path = Path(__file__).parent.parent
            patterns_file = base_path / "data" / "canonical_six_patterns.json"

        self.patterns_file = Path(patterns_file)
        self.patterns_data = None
        self.canonical_six = {}

        self._load_patterns()

    def _load_patterns(self):
        """Load patterns from JSON file."""
        if not self.patterns_file.exists():
            raise FileNotFoundError(
                f"Canonical Six patterns file not found: {self.patterns_file}\n"
                f"Expected location: mcp-server/data/canonical_six_patterns.json"
            )

        with open(self.patterns_file, 'r') as f:
            self.patterns_data = json.load(f)

        # Build pattern dictionary
        for pattern_id, pattern_data in self.patterns_data['patterns'].items():
            self.canonical_six[int(pattern_id)] = pattern_data

    def get_pattern(self, pattern_id: int) -> Dict:
        """Get pattern data by ID (18, 59, 84, 102, 104, 124)."""
        if pattern_id not in self.canonical_six:
            raise ValueError(
                f"Pattern {pattern_id} not found. "
                f"Valid IDs: {list(self.canonical_six.keys())}"
            )
        return self.canonical_six[pattern_id]

    def get_metadata(self) -> Dict:
        """Get pattern catalog metadata."""
        return self.patterns_data['metadata']


# Mapping: chess piece type → Canonical Six pattern ID
PIECE_TO_PATTERN_ID = {
    chess.KING:   18,   # Master Gateway
    chess.QUEEN:  59,   # Multi-Modal Gateway
    chess.KNIGHT: 84,   # Discontinuous Gateway
    chess.BISHOP: 102,  # Conjugate Pair Gateway
    chess.ROOK:   104,  # Linear Gateway
    chess.PAWN:   124   # Transformation Gateway
}

# Reverse mapping
PATTERN_ID_TO_PIECE = {v: k for k, v in PIECE_TO_PATTERN_ID.items()}

# Global loader instance (singleton pattern)
_pattern_loader = None


def get_pattern_loader() -> GatewayPatternLoader:
    """Get or create the global pattern loader instance."""
    global _pattern_loader
    if _pattern_loader is None:
        _pattern_loader = GatewayPatternLoader()
    return _pattern_loader


def assign_gateway(piece_type: chess.PieceType) -> Sedenion:
    """
    Get the Gateway Pattern (as 16D Sedenion) for a chess piece type.

    Args:
        piece_type: chess.KING, chess.QUEEN, etc.

    Returns:
        16D Sedenion representing the gateway pattern

    Example:
        >>> pattern = assign_gateway(chess.KNIGHT)
        >>> # Returns Pattern 84: (e4 + e11) × (e6 + e9) = 0

    Raises:
        ValueError: If piece_type not recognized
    """
    if piece_type not in PIECE_TO_PATTERN_ID:
        raise ValueError(
            f"Unknown piece type: {piece_type}. "
            f"Valid types: {list(PIECE_TO_PATTERN_ID.keys())}"
        )

    pattern_id = PIECE_TO_PATTERN_ID[piece_type]
    loader = get_pattern_loader()
    pattern_data = loader.get_pattern(pattern_id)

    # Construct sedenion from pattern
    # Pattern structure: (ea + s1*eb) × (ec + s2*ed) = 0
    # We return the first term: ea + s1*eb

    indices = pattern_data['indices']  # [a, b, c, d]
    signs = pattern_data['signs']      # [s1, s2]

    # Create basis elements
    # For e_i, create a sedenion with 1.0 at position i, zeros elsewhere
    coeffs_a = [0.0] * 16
    coeffs_a[indices[0]] = 1.0
    ea = Sedenion(*coeffs_a)

    coeffs_b = [0.0] * 16
    coeffs_b[indices[1]] = 1.0
    eb = Sedenion(*coeffs_b)

    # Construct: ea + s1*eb
    gateway = ea + (signs[0] * eb)

    return gateway


def get_pattern_info(piece_type: chess.PieceType) -> Dict:
    """
    Get human-readable information about a piece's gateway pattern.

    Args:
        piece_type: Chess piece type

    Returns:
        Dictionary with pattern details:
            - 'id': Pattern ID (18, 59, 84, 102, 104, 124)
            - 'name': Gateway name (e.g., "Discontinuous Gateway")
            - 'formula': Mathematical formula
            - 'indices': Basis element indices [a, b, c, d]
            - 'signs': Sign values [s1, s2]
            - 'piece': Chess piece name

    Example:
        >>> info = get_pattern_info(chess.KNIGHT)
        >>> print(info['name'])
        "Discontinuous Gateway"
        >>> print(info['formula'])
        "(e4 + e11) × (e6 + e9) = 0"
    """
    pattern_id = PIECE_TO_PATTERN_ID[piece_type]
    loader = get_pattern_loader()
    pattern_data = loader.get_pattern(pattern_id)

    return {
        'id': pattern_id,
        'name': pattern_data['name'],
        'formula': pattern_data['formula'],
        'indices': pattern_data['indices'],
        'signs': pattern_data['signs'],
        'piece': pattern_data['chess_piece']
    }


def validate_canonical_six() -> bool:
    """
    Verify that loaded patterns have expected mathematical properties.

    Tests:
        1. All 6 patterns load successfully
        2. Each pattern has valid indices (0-15 for 16D sedenions)
        3. Each pattern has valid signs (+1 or -1)
        4. Patterns map to all 6 chess piece types

    Returns:
        True if all validations pass

    Raises:
        AssertionError: If critical properties violated
    """
    loader = get_pattern_loader()

    # Test 1: All 6 patterns present
    expected_ids = [18, 59, 84, 102, 104, 124]
    assert len(loader.canonical_six) == 6, \
        f"Expected 6 patterns, found {len(loader.canonical_six)}"

    for pattern_id in expected_ids:
        assert pattern_id in loader.canonical_six, \
            f"Missing pattern {pattern_id}"

    # Test 2: Valid indices (0-15 for sedenions)
    for pattern_id, pattern_data in loader.canonical_six.items():
        indices = pattern_data['indices']
        assert len(indices) == 4, \
            f"Pattern {pattern_id}: Expected 4 indices, got {len(indices)}"

        for idx in indices:
            assert 0 <= idx <= 15, \
                f"Pattern {pattern_id}: Invalid index {idx} (must be 0-15)"

    # Test 3: Valid signs (+1 or -1)
    for pattern_id, pattern_data in loader.canonical_six.items():
        signs = pattern_data['signs']
        assert len(signs) == 2, \
            f"Pattern {pattern_id}: Expected 2 signs, got {len(signs)}"

        for sign in signs:
            assert sign in [-1, 1], \
                f"Pattern {pattern_id}: Invalid sign {sign} (must be ±1)"

    # Test 4: All piece types mapped
    piece_types = [chess.KING, chess.QUEEN, chess.KNIGHT,
                   chess.BISHOP, chess.ROOK, chess.PAWN]

    for piece_type in piece_types:
        assert piece_type in PIECE_TO_PATTERN_ID, \
            f"Piece type {piece_type} not mapped to pattern"

    return True


# Module-level convenience functions
def load_canonical_six() -> Dict[int, Sedenion]:
    """
    Load all six gateway patterns as Sedenions.

    Returns:
        Dictionary mapping pattern IDs to their Sedenion representations

    Example:
        >>> patterns = load_canonical_six()
        >>> knight_pattern = patterns[84]
    """
    loader = get_pattern_loader()
    result = {}

    for pattern_id in [18, 59, 84, 102, 104, 124]:
        pattern_data = loader.get_pattern(pattern_id)
        indices = pattern_data['indices']
        signs = pattern_data['signs']

        # Construct first term: ea + s1*eb
        coeffs_a = [0.0] * 16
        coeffs_a[indices[0]] = 1.0
        ea = Sedenion(*coeffs_a)

        coeffs_b = [0.0] * 16
        coeffs_b[indices[1]] = 1.0
        eb = Sedenion(*coeffs_b)

        result[pattern_id] = ea + (signs[0] * eb)

    return result


if __name__ == "__main__":
    """Test the gateway patterns module."""

    print("=" * 60)
    print("GATEWAY PATTERNS MODULE TEST")
    print("=" * 60)
    print()

    # Test 1: Load patterns
    print("Test 1: Loading Canonical Six patterns...")
    try:
        loader = get_pattern_loader()
        metadata = loader.get_metadata()
        print(f"Loaded {len(loader.canonical_six)} patterns")
        print(f"  Source: {metadata['source']}")
        print(f"  Precision: {metadata['precision']}")
        print()
    except Exception as e:
        print(f"Failed to load patterns: {e}")
        sys.exit(1)

    # Test 2: Validate patterns
    print("Test 2: Validating pattern properties...")
    try:
        validate_canonical_six()
        print("All validations passed")
        print()
    except AssertionError as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

    # Test 3: Test piece assignments
    print("Test 3: Testing piece -> pattern assignments...")
    piece_types = [
        (chess.KING, "King"),
        (chess.QUEEN, "Queen"),
        (chess.KNIGHT, "Knight"),
        (chess.BISHOP, "Bishop"),
        (chess.ROOK, "Rook"),
        (chess.PAWN, "Pawn")
    ]

    for piece_type, piece_name in piece_types:
        info = get_pattern_info(piece_type)
        print(f"{piece_name:8s} -> Pattern {info['id']:3d} ({info['name']})")
        print(f"           Formula: {info['formula']}")
    print()

    # Test 4: Construct sedenions
    print("Test 4: Constructing gateway sedenions...")
    try:
        for piece_type, piece_name in piece_types:
            gateway = assign_gateway(piece_type)
            print(f"{piece_name}: {gateway}")
        print()
    except Exception as e:
        print(f"Failed to construct sedenion: {e}")
        sys.exit(1)

    # Test 5: Load all patterns
    print("Test 5: Loading all patterns as sedenions...")
    try:
        all_patterns = load_canonical_six()
        print(f"Loaded {len(all_patterns)} gateway patterns")
        print()
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("Gateway Patterns module is ready for integration")
    print("=" * 60)
