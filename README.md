
# Model Context Protocol (MCP)

## Modelcontextprotocol.io

- [docs](https://modelcontextprotocol.io/introduction)
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [python-sdk source](https://github.com/modelcontextprotocol/python-sdk)

## MCP Reference Implementations

### FastMCP

- [source](https://github.com/jlowin/fastmcp)
- [docs](https://gofastmcp.com/getting-started/welcome)

The fast, Pythonic way to build MCP servers and clients

### Fast Agent

- [source](https://github.com/evalstate/fast-agent)
- [story](https://llmindset.co.uk/resources/fast-agent/)

Create and interact with sophisticated Agents and Workflows in minutes. It is the first framework with complete, end-to-end tested MCP Feature support including Sampling. Both Anthropic (Haiku, Sonnet, Opus) and OpenAI models (gpt-4o/gpt-4.1 family, o1/o3 family) are supported

## MCP Demos

### FastMCP - Gemini

Here's a complete set of files for a `fastmcp` tutorial using Gemini.

- `mcp_server.py`: server with tools: calculator, trigonometry function, yahoo finance
- MCP clients:
    - `mcp_client_simple.py`: simple client with rule-based routing
    - `mcp_client_llm.py`: client with LLM parsing and calling tools
    - `mcp_client_llm_resource.py`: client with LLM parsing and calling tools, resources (see `readme_tool_resource.md` for details)

see `readme_gemini.md` for more details

#### Setup
```bash
conda create -n mcp
conda activate mcp
git clone https://github.com/digital-duck/mcp_demo.git

cd mcp_demo
pip install -r requirements.txt

# in 1st terminal
python mcp_server.py

# in 2nd terminal
# python mcp_client_simple.py
python mcp_client_llm.py
# python mcp_client_llm_resource.py
```

### FastMCP - AWS

see `readme_aws.md` for more details
