
# Model Context Protocol (MCP)

## Modelcontextprotocol.io

- [docs](https://modelcontextprotocol.io/introduction)
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [python-sdk source](https://github.com/modelcontextprotocol/python-sdk)

## Reference Implementations

### FastMCP

- [source](https://github.com/jlowin/fastmcp)
- [docs](https://gofastmcp.com/getting-started/welcome)

A fast, Pythonic way to build MCP servers and clients

### Fast Agent

- [source](https://github.com/evalstate/fast-agent)
- [story](https://llmindset.co.uk/resources/fast-agent/)

A framework with complete, end-to-end tested MCP Feature support including Sampling, which helps create and interact with sophisticated Agents and Workflows in minutes. 
Both Anthropic (Haiku, Sonnet, Opus) and OpenAI models (gpt-4o/gpt-4.1 family, o1/o3 family) are supported

## Demos

### FastMCP - Gemini

A `fastmcp` tutorial using Gemini.

- MCP server:
    - `mcp_server.py`: implements tools/resources like calculator, trigonometry function, yahoo finance API call
- MCP clients:
    - `mcp_client_simple.py`: simple client with rule-based query parsing
    - `mcp_client_llm.py`: client with LLM-based query parsing
    - `mcp_client_llm_resource.py`: client with LLM parsing and calling tool/resource (see `readme_tool_resource.md` for details)

see [readme_gemini.md](https://github.com/digital-duck/mcp_demo/blob/main/readme_gemini.md) for more details

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

see [readme_aws.md](https://github.com/digital-duck/mcp_demo/blob/main/readme_aws.md) for more details
