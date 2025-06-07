#!/usr/bin/env python3
"""
MCP Server using the Official MCP Python SDK
Ported from FastMCP implementation
"""

import asyncio
import logging
import math
import yfinance as yf
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource
)
from mcp.server import NotificationOptions
from pydantic import AnyUrl
import mcp.types as types

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create MCP server instance
server = Server("Demo 🚀")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="calculator",
            description="Performs arithmetic operations on two numbers. Supports: add, subtract, multiply, divide, power",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "power"],
                        "description": "The arithmetic operation to perform"
                    },
                    "num1": {
                        "type": "number",
                        "description": "First number"
                    },
                    "num2": {
                        "type": "number", 
                        "description": "Second number"
                    }
                },
                "required": ["operation", "num1", "num2"]
            }
        ),
        Tool(
            name="trig",
            description="Performs trigonometric operations, with input/output angle (in unit of degree or radian). Supports: sine, cosine, tangent, arc sine, arc cosine, arc tangent",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["sine", "cosine", "tangent", "arc sine", "arc cosine", "arc tangent"],
                        "description": "The trigonometric operation to perform"
                    },
                    "num1": {
                        "type": "number",
                        "description": "Input number/angle"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["degree", "radian"],
                        "description": "Unit for angle measurement",
                        "default": "degree"
                    }
                },
                "required": ["operation", "num1", "unit"]
            }
        ),
        Tool(
            name="stock_quote",
            description="Retrieves live stock data for a given ticker symbol. Example: AAPL, GOOGL, MSFT, TSLA",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
                    }
                },
                "required": ["ticker"]
            }
        ),
        Tool(
            name="health",
            description="Simple health check to verify server is running",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="echo",
            description="Echo back the provided message. Useful for testing",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo back"
                    }
                },
                "required": ["message"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls."""
    if arguments is None:
        arguments = {}
    
    logging.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "calculator":
            result = await calculator_tool(**arguments)
        elif name == "trig":
            result = await trig_tool(**arguments)
        elif name == "stock_quote":
            result = await stock_quote_tool(**arguments)
        elif name == "health":
            result = await health_tool()
        elif name == "echo":
            result = await echo_tool(**arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        # Convert result to JSON string and return as TextContent
        import json
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logging.error(f"Error in tool {name}: {e}")
        error_result = {"error": f"Tool execution failed: {str(e)}"}
        import json
        return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]

async def calculator_tool(operation: str, num1: float, num2: float) -> dict:
    """Calculator tool implementation."""
    logging.info(f"Calculator called: {operation} {num1} {num2}")
    
    try:
        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return {"error": "Cannot divide by zero"}
            result = num1 / num2
        elif operation == "power":
            if num1 == 0:
                return {"error": "Base cannot be 0"}
            result = pow(num1, num2)
        else:
            return {"error": f"Unsupported operation: {operation}. Use: add, subtract, multiply, divide, power"}
        
        return {
            "operation": operation,
            "num1": num1,
            "num2": num2,
            "result": result,
            "expression": f"{num1} {operation} {num2} = {result}"
        }
    except Exception as e:
        logging.error(f"Calculator error: {e}")
        return {"error": str(e)}

async def trig_tool(operation: str, num1: float, unit: str) -> dict:
    """Trigonometry tool implementation."""
    logging.info(f"Trig called: {operation} {num1} in {unit}")
    
    try:
        unit = unit.lower()
        if operation == "sine":
            arg1 = num1 if unit.startswith("rad") else math.radians(num1)
            result = math.sin(arg1)
        elif operation == "cosine":
            arg1 = num1 if unit.startswith("rad") else math.radians(num1)
            result = math.cos(arg1)
        elif operation == "tangent":
            arg1 = num1 if unit.startswith("rad") else math.radians(num1)
            result = math.tan(arg1)
        elif operation == "arc sine":
            result = math.asin(num1)
            if not unit.startswith("rad"):
                result = math.degrees(result)
        elif operation == "arc cosine":
            result = math.acos(num1)
            if not unit.startswith("rad"):
                result = math.degrees(result)
        elif operation == "arc tangent":
            result = math.atan(num1)
            if not unit.startswith("rad"):
                result = math.degrees(result)
        else:
            return {"error": f"Unsupported operation: {operation}. Use: sine, cosine, tangent, arc sine, arc cosine, arc tangent"}
        
        return {
            "operation": operation,
            "num1": num1,
            "unit": unit,
            "result": result,
            "expression": f"{operation} ( {num1} ) = {result}"
        }
    except Exception as e:
        logging.error(f"Trig error: {e}")
        return {"error": str(e)}

async def stock_quote_tool(ticker: str) -> dict:
    """Stock quote tool implementation."""
    logging.info(f"Stock quote requested for: {ticker}")
    
    try:
        ticker_data = yf.Ticker(ticker.upper())
        info = ticker_data.info
        
        # Get current price - try multiple fields
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        
        # Fallback to recent historical data
        if current_price is None:
            try:
                hist = ticker_data.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
            except Exception:
                pass
        
        if current_price is None:
            return {
                "ticker": ticker.upper(),
                "error": f"Could not retrieve price data for {ticker}. Ticker may be invalid."
            }
        
        return {
            "ticker": ticker.upper(),
            "current_price": round(float(current_price), 2),
            "currency": info.get('currency', 'USD'),
            "company_name": info.get('longName', info.get('shortName', 'Unknown')),
            "market_cap": info.get('marketCap'),
            "previous_close": info.get('previousClose'),
            "volume": info.get('volume'),
            "day_high": info.get('dayHigh'),
            "day_low": info.get('dayLow')
        }
        
    except Exception as e:
        logging.error(f"Yahoo Finance error for {ticker}: {e}")
        return {
            "ticker": ticker.upper(),
            "error": f"Failed to get stock data: {str(e)}"
        }

async def health_tool() -> dict:
    """Health check tool implementation."""
    return {
        "status": "healthy",
        "message": "MCP Server is running properly",
        "available_tools": ["calculator", "trig", "stock_quote", "health", "echo"],
        "server_name": "Demo 🚀"
    }

async def echo_tool(message: str) -> dict:
    """Echo tool implementation."""
    return {
        "original_message": message,
        "echo": f"Echo: {message}",
        "length": len(message),
        "timestamp": "2025-06-04"
    }

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources."""
    return [
        types.Resource(
            uri=AnyUrl("info://server"),
            name="Server Information",
            description="Provides basic information about this MCP server",
            mimeType="text/plain"
        ),
        types.Resource(
            uri=AnyUrl("stock://template"),
            name="Stock Information Template", 
            description="Template for stock information resources",
            mimeType="text/plain"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Handle resource reading."""
    uri_str = str(uri)
    logging.info(f"Resource requested: {uri_str}")
    
    if uri_str == "info://server":
        return "This is a demo MCP server built with the official MCP Python SDK. It provides calculator, trigonometry, and stock quote tools."
    elif uri_str.startswith("stock://"):
        # Extract ticker from URI like stock://AAPL
        if uri_str == "stock://template":
            return "Stock resource template: Use stock://TICKER format (e.g., stock://AAPL)"
        
        ticker = uri_str.replace("stock://", "").upper()
        try:
            ticker_data = yf.Ticker(ticker)
            info = ticker_data.info
            company_name = info.get('longName', ticker)
            return f"{company_name} ({ticker}) - A publicly traded company"
        except Exception:
            return f"Stock ticker: {ticker}"
    else:
        raise ValueError(f"Unknown resource: {uri_str}")

async def main():
    """Main server entry point."""
    logging.info("Starting MCP Server with Official MCP SDK...")
    print("🚀 MCP Server Starting...")
    print("📊 Available tools:")
    print("   • calculator - Perform arithmetic operations") 
    print("   • trig - Perform trigonometric operations") 
    print("   • stock_quote - Get stock price data")
    print("   • health - Server health check")
    print("   • echo - Echo messages for testing")
    print("📚 Available resources:")
    print("   • info://server - Server information")
    print("   • stock://TICKER - Stock information (e.g., stock://AAPL)")
    print("✅ Server ready! Starting stdio server...")
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, 
            write_stream, 
            InitializationOptions(
                server_name="Demo MCP Server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())