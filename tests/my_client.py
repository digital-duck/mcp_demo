import asyncio
from fastmcp import Client
from pathlib import Path

def get_mcp_server_path(server_filename="mcp_server.py"):
    """
    Get absolute path to MCP server file in the same directory as the current script.
    
    Args:
        server_filename (str): Name of the MCP server file (default: "mcp_server.py")
    
    Returns:
        Path: Absolute path to the MCP server file
    
    Raises:
        FileNotFoundError: If the server file doesn't exist
        RuntimeError: If unable to determine current script directory
    """
    try:
        # Method 1: Use __file__ if available (works in most cases)
        if '__file__' in globals():
            current_script_dir = Path(__file__).parent.resolve()
        else:
            # Method 2: Fallback for interactive environments
            current_script_dir = Path.cwd()
        
        # Construct the server path
        server_path = current_script_dir / server_filename
        
        # Verify the file exists
        if not server_path.exists():
            raise FileNotFoundError(f"MCP server file not found: {server_path}")
        
        return server_path.resolve()  # Return absolute path
    
    except Exception as e:
        raise RuntimeError(f"Failed to determine MCP server path: {e}")


server_path = str(get_mcp_server_path("my_server.py"))
print(f"📡 Connecting to MCP server: {server_path}")

client = Client(server_path)

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)

asyncio.run(call_tool("Ford"))