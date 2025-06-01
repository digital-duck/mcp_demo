
# Model Context Protocol (MCP) Demo with fastmcp and Gemini

This project demonstrates the Model Context Protocol (MCP) using the `fastmcp` Python library for the server, a custom Python client, and Google Gemini for natural language understanding and tool call translation.

It showcases how a Large Language Model (LLM) can interpret a user's plain English request, determine which tool to use, extract the necessary parameters, and then format that into an MCP-compliant call to a server hosting those tools.

## Features

* **MCP Server (`fastmcp`):**
    * Exposes a `calculator` tool for basic arithmetic operations (addition, subtraction).
    * Exposes a `yahoo_finance` tool to fetch current stock quotes for a given ticker.
    * Built using `fastmcp` for easy tool definition and JSON-RPC 2.0 handling.
    * Runs on `uvicorn` for a robust HTTP server.
* **MCP Client:**
    * Connects to the `fastmcp` server to discover available tools.
    * Sends MCP `Context.call_tool` requests to the server.
* **Gemini Integration:**
    * Uses the Google Gemini API to translate natural language user queries into structured tool calls.
    * Dynamically identifies the correct tool and extracts parameters based on the user's intent.

## Prerequisites

Before you begin, ensure you have:

* Python 3.8+ installed.
* A Google Gemini API Key. You can obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## Setup

1.  **Clone the repository (or create the files manually):**
    Create a new directory for your project and place the following files inside it:
    * `README.md` (this file)
    * `requirements.txt`
    * `mcp_server_llm.py`
    * `mcp_client.py`

2.  **Install Dependencies:**
    Navigate to your project directory in the terminal and install the required Python packages using the provided `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Gemini API Key:**
    Open `mcp_client.py` and replace `"YOUR_GEMINI_API_KEY"` with your actual Google Gemini API key:

    ```python
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # <--- REPLACE THIS
    ```

## Running the Demo

This demo involves running two separate Python scripts concurrently: the MCP server and the MCP client.

### Step 1: Run the MCP Server

Open your first terminal window, navigate to your project directory, and run the server:

```bash
python mcp_server.py
```

You should see output from `uvicorn` indicating that the server is running, typically on `http://127.0.0.1:5000/`. The MCP endpoint will be at `http://127.0.0.1:5000/mcp`.

### Step 2: Run the MCP Client with Gemini Integration

Open your second terminal window, navigate to your project directory, and run the client:

```bash
python mcp_client.py
```

The client will first discover the tools available on the server. Then, it will prompt you to enter natural language queries.

**Example Queries you can try:**

* **Calculator:**
    * `What is 10 plus 5?`
    * `Can you add 12.5 and 7.3?`
    * `I need to calculate the sum of 100 and 200.`
    * `Subtract 5 from 10.`
    * `What is 25 minus 12?`
* **Yahoo Finance:**
    * `What is the current stock price of Google?`
    * `Get me the quote for Apple stock.`
    * `How much is TSLA right now?`
    * `What's the latest price for Microsoft?`
* **Non-Tool Queries:**
    * `Tell me a joke.` (Should not result in a tool call)
    * `What is the capital of France?` (Should not result in a tool call)

Observe the output in both terminals. The client terminal will show Gemini's interpretation and the MCP request sent, along with the result from the server. The server terminal will show logs of the incoming MCP requests and their processing.

## How it Works

1.  **Tool Definition (`mcp_server.py`):**
    * We define our `calculator` and `yahoo_finance` tools using `fastmcp`'s `@tool` decorator and Pydantic models for input/output schemas. This automatically generates the MCP-compliant tool definitions.
    * The `MCPServer` instance then serves these tools via a JSON-RPC 2.0 endpoint (`/mcp`).

2.  **Tool Discovery (Client Startup):**
    * The `MCPClient` in `mcp_client.py` makes an initial `Context.get_tool_definitions` call to the `fastmcp` server to retrieve the schema of all available tools.

3.  **Gemini's Role (`mcp_client.py`):**
    * The discovered tool definitions are provided to the Google Gemini model when it's initialized.
    * When you type a natural language query, Gemini uses its function-calling capabilities to analyze your intent. If it determines that a query can be fulfilled by one of the provided tools, it generates a `FunctionCall` object (e.g., `name='calculator', args={'operation': 'add', 'num1': 10, 'num2': 5}`).

4.  **MCP Request Formatting (`mcp_client.py`):**
    * The `FunctionCall` object from Gemini is then translated into the exact JSON-RPC 2.0 format required by the MCP `Context.call_tool` method.

5.  **Tool Execution (`mcp_server.py`):**
    * The `MCPClient` sends this formatted MCP request to the `fastmcp` server.
    * The `fastmcp` server receives the request, validates it against the tool's schema, executes the corresponding Python function (`calculator_tool` or `yahoo_finance_tool`), and returns the result in an MCP-compliant JSON-RPC 2.0 response.

## Future Enhancements

* **More Operations:** Extend the `calculator` tool to include multiplication, division, etc.
* **Error Handling:** Implement more sophisticated error handling and user feedback for tool failures.
* **Complex Tool Parameters:** Explore how to handle more complex data structures (e.g., lists, nested objects) as tool parameters.
* **Asynchronous Client:** For real-world applications, consider using an asynchronous HTTP client (like `httpx` or `aiohttp`) in the client script for better performance.
* **Context Management:** Explore MCP's `Context.update_resource` and `Context.get_resource` methods to manage shared state or data between tool calls.
* **Multi-Agent Orchestration:** Build a more complex system where multiple AI agents interact with each other and tools via MCP.
* **Deployment:** Deploy the `fastmcp` server to a cloud platform (e.g., Google Cloud Run, Azure App Service) for a production environment.
