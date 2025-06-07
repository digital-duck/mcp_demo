import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
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


async def run_mcp_client():
    server_params = StdioServerParameters(
        command="python", 
        # args=[str(get_mcp_server_path("my_server.py"))]
        args=["my_server.py"]  # Assuming the server script is in the same directory)]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # Call tool
            result = await session.call_tool("greet", {"name": "Wen"})
            print(result)

asyncio.run(run_mcp_client())