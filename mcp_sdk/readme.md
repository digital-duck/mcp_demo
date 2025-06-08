Ported FastMCP implementation to the official MCP Python SDK. 

## Key Changes Made:

### **Server (mcp_server.py)**
1. **Replaced FastMCP with Official SDK**: Uses `mcp.server.Server` instead of `FastMCP`
2. **New Handler Structure**: 
   - `@server.list_tools()` for tool discovery
   - `@server.call_tool()` for tool execution
   - `@server.list_resources()` and `@server.read_resource()` for resources
3. **Proper Schema Definitions**: Tools now have full JSON schemas with validation
4. **Stdio Transport**: Uses `stdio_server()` for communication
5. **Response Format**: Returns `TextContent` objects with JSON strings

### **Client (st_mcp_app.py)**
1. **Official SDK Integration**: Uses `ClientSession` and `stdio_client`
2. **Proper Async Context**: Each operation creates fresh connection
3. **Updated Path Helper**: Uses your helper function for cross-platform paths
4. **Enhanced Error Handling**: Better connection management
5. **All RAG Features Preserved**: Complete semantic search and LLM integration

## Installation Requirements:

```bash
conda create -n mcp
conda activate mcp
git clone https://github.com/digital-duck/mcp_demo.git

cd mcp_demo/mcp_sdk
pip install -r requirements.txt

# in 1st terminal
python mcp_server.py


# in 2nd terminal
streamlit run st_mcp_app.py
```

## Usage:

1. **Start the server**: `python mcp_server.py`
2. **Run the client**: `streamlit run st_mcp_app.py`

## Key Benefits of Official SDK:

- ✅ **Windows Compatibility**: No more subprocess issues
- ✅ **Better Error Handling**: Proper connection lifecycle
- ✅ **Schema Validation**: Input validation built-in
- ✅ **Future-Proof**: Official support and updates
- ✅ **Standards Compliant**: Full MCP protocol implementation

The implementation maintains all your original functionality (calculator, trig, stock quotes, health checks, echo) plus the complete RAG system with semantic search and LLM integration. The client interface remains exactly the same for users!