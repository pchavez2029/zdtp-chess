"""
MULTID_CHESS MCP Server

Multi-dimensional chess AI using Zero Divisor Transmission Protocol (ZDTP)
"""

__version__ = "0.1.0"
__author__ = "Paul Chavez - Chavez AI Labs"

from .zdtp_chess_server import server

__all__ = ["server"]
