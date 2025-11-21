"""
Main entry point for MULTID_CHESS MCP Server

Run with: python -m multid_chess_mcp
"""

import asyncio
import sys
from mcp.server.stdio import stdio_server
from .zdtp_chess_server import server

async def run_server():
    """Async runner for MCP server"""
    print("MULTID_CHESS server starting...", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        print("MULTID_CHESS server running", file=sys.stderr)
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

def main():
    """Run the MCP server"""
    try:
        asyncio.run(run_server())
    except Exception as e:
        print(f"MULTID_CHESS server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
