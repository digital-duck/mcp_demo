import streamlit as st
import asyncio
import json
import logging
import os
import sqlite3
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from fastmcp import Client
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit page config
st.set_page_config(
    page_title="MCP Client Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .tool-call {
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
    .error-message {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
    .debug-info {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# LLM Models and Provider Selection
LLM_MODELS = ["openai", "anthropic", "ollama", "gemini", "bedrock"]
DATABASE_FILE = "mcp_chat_history.db"

# --- Database Operations ---
class ChatHistoryDB:
    def __init__(self, db_file: str = DATABASE_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    llm_provider TEXT,
                    model_name TEXT,
                    parsing_mode TEXT,
                    user_query TEXT NOT NULL,
                    parsed_action TEXT,
                    tool_name TEXT,
                    resource_uri TEXT,
                    parameters TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    response_data TEXT,
                    formatted_response TEXT,
                    elapsed_time_ms INTEGER,
                    error_message TEXT,
                    success BOOLEAN NOT NULL DEFAULT 1
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON chat_history(session_id)")
            conn.commit()
    
    def insert_chat_entry(self, entry: Dict[str, Any]) -> int:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (
                    session_id, timestamp, llm_provider, model_name, parsing_mode,
                    user_query, parsed_action, tool_name, resource_uri, parameters,
                    confidence, reasoning, response_data, formatted_response,
                    elapsed_time_ms, error_message, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.get('session_id'), entry.get('timestamp'), entry.get('llm_provider'),
                entry.get('model_name'), entry.get('parsing_mode'), entry.get('user_query'),
                entry.get('parsed_action'), entry.get('tool_name'), entry.get('resource_uri'),
                entry.get('parameters'), entry.get('confidence'), entry.get('reasoning'),
                entry.get('response_data'), entry.get('formatted_response'),
                entry.get('elapsed_time_ms'), entry.get('error_message'), entry.get('success', True)
            ))
            entry_id = cursor.lastrowid
            conn.commit()
            return entry_id
    
    def get_chat_history(self, limit: int = 100, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM chat_history"
            params = []
            
            if filters and filters.get('session_id'):
                query += " WHERE session_id = ?"
                params.append(filters['session_id'])
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

# Initialize session state
def init_session_state():
    if 'chat_history_db' not in st.session_state:
        st.session_state.chat_history_db = ChatHistoryDB()
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(f"{datetime.now()}{os.getpid()}".encode()).hexdigest()[:8]
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "anthropic"
    if 'use_llm' not in st.session_state:
        st.session_state.use_llm = True
    if 'server_connected' not in st.session_state:
        st.session_state.server_connected = False
    if 'available_tools' not in st.session_state:
        st.session_state.available_tools = []
    if 'available_resources' not in st.session_state:
        st.session_state.available_resources = []
    if 'last_parsed_query' not in st.session_state:
        st.session_state.last_parsed_query = None

# --- LLM Query Parser ---
class LLMQueryParser:
    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.client = None
        self.model_name = None
        self.setup_llm_client()
    
    def setup_llm_client(self):
        try:
            if self.provider == "anthropic":
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    self.model_name = "claude-3-5-sonnet-20241022"
            
            elif self.provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.model_name = "gpt-4o-mini"
            
            elif self.provider == "gemini":
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel("gemini-1.5-flash")
                    self.model_name = "gemini-1.5-flash"
                
        except Exception as e:
            st.error(f"Failed to initialize {self.provider}: {e}")
            self.client = None
    
    def parse_query_sync(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        
        system_prompt = """You are a tool selection assistant. Respond with ONLY a JSON object:

{
    "action": "tool",
    "tool": "tool_name_or_null",
    "params": {"param1": "value1"},
    "confidence": 0.95,
    "reasoning": "Brief explanation"
}

Available tools:
- calculator: operation (add/subtract/multiply/divide/power), num1, num2
- trig: operation (sine/cosine/tangent), num1, unit (degree/radian)
- health: no parameters
- echo: message

Examples:
"15 plus 27" -> {"action": "tool", "tool": "calculator", "params": {"operation": "add", "num1": 15, "num2": 27}, "confidence": 0.98, "reasoning": "Simple addition"}
"sine of 30 degrees" -> {"action": "tool", "tool": "trig", "params": {"operation": "sine", "num1": 30, "unit": "degree"}, "confidence": 0.95, "reasoning": "Trigonometric calculation"}

Respond with ONLY the JSON object."""
        
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=300,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": query}]
                )
                llm_response = response.content[0].text.strip()
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                llm_response = response.choices[0].message.content.strip()
            
            elif self.provider == "gemini":
                response = self.client.generate_content(
                    f"{system_prompt}\n\nUser: {query}",
                    generation_config={"temperature": 0.1, "max_output_tokens": 300}
                )
                llm_response = response.text.strip()
            
            # Clean and parse JSON
            if llm_response.startswith("```json"):
                llm_response = llm_response.replace("```json", "").replace("```", "").strip()
            elif llm_response.startswith("```"):
                llm_response = llm_response.replace("```", "").strip()
            
            parsed_response = json.loads(llm_response)
            
            if parsed_response.get("action") and parsed_response.get("confidence", 0) >= 0.5:
                return parsed_response
            
        except Exception as e:
            st.error(f"LLM parsing error: {e}")
        
        return None

# --- Rule-based Parser ---
class RuleBasedQueryParser:
    @staticmethod
    def parse_query(query: str) -> Optional[Dict[str, Any]]:
        import re
        query_lower = query.lower().strip()
        
        # Health check
        if any(word in query_lower for word in ["health", "status", "ping"]):
            return {"action": "tool", "tool": "health", "params": {}, "confidence": 0.9, "reasoning": "Health check request"}
        
        # Echo command
        if query_lower.startswith("echo "):
            return {"action": "tool", "tool": "echo", "params": {"message": query[5:].strip()}, "confidence": 0.95, "reasoning": "Echo command"}
        
        # Calculator
        calc_patterns = [
            ("add", ["plus", "add", "+", "sum"]),
            ("subtract", ["minus", "subtract", "-"]),
            ("multiply", ["times", "multiply", "*", "×"]),
            ("divide", ["divide", "divided by", "/"]),
            ("power", ["power", "to the power", "^"])
        ]
        
        for operation, keywords in calc_patterns:
            for keyword in keywords:
                if keyword in query_lower:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if len(numbers) >= 2:
                        return {
                            "action": "tool",
                            "tool": "calculator", 
                            "params": {"operation": operation, "num1": float(numbers[0]), "num2": float(numbers[1])},
                            "confidence": 0.9,
                            "reasoning": f"Calculator operation: {operation}"
                        }
        
        # Trig functions
        trig_patterns = [
            ("sine", ["sine", "sin"]),
            ("cosine", ["cosine", "cos"]),
            ("tangent", ["tangent", "tan"])
        ]
        
        for operation, keywords in trig_patterns:
            for keyword in keywords:
                if keyword in query_lower:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if numbers:
                        unit = "radian" if any(word in query_lower for word in ["radian", "rad"]) else "degree"
                        return {
                            "action": "tool",
                            "tool": "trig",
                            "params": {"operation": operation, "num1": float(numbers[0]), "unit": unit},
                            "confidence": 0.9,
                            "reasoning": f"Trigonometry: {operation}"
                        }
        
        return None

# --- Utility Functions ---
def extract_result_data(result):
    try:
        if isinstance(result, list) and len(result) > 0:
            content_item = result[0]
            if hasattr(content_item, 'text'):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return {"text": content_item.text}
            else:
                return {"content": str(content_item)}
        elif hasattr(result, 'content') and result.content:
            content_item = result.content[0]
            if hasattr(content_item, 'text'):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return {"text": content_item.text}
            else:
                return {"content": str(content_item)}
        else:
            return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as e:
        return {"error": f"Could not parse result: {e}"}

def format_result_for_display(tool_name: str, result: Dict) -> str:
    if isinstance(result, dict) and "error" in result:
        return f"❌ **Error:** {result['error']}"
    
    if tool_name == "calculator":
        expression = result.get('expression', f"{result.get('num1', '?')} {result.get('operation', '?')} {result.get('num2', '?')} = {result.get('result', '?')}")
        return f"🧮 **Calculator:** {expression}"
    
    elif tool_name == "trig":
        expression = result.get('expression', f"{result.get('operation', '?')}({result.get('num1', '?')}) = {result.get('result', '?')}")
        return f"📐 **Trigonometry:** {expression}"
    
    elif tool_name == "health":
        return f"✅ **Health:** {result.get('message', 'Server is healthy')}"
    
    elif tool_name == "echo":
        return f"🔊 **Echo:** {result.get('echo', result.get('message', str(result)))}"
    
    return f"✅ **Result:** {json.dumps(result, indent=2)}"

# --- CACHED MCP Operations using st.cache_resource ---
@st.cache_resource  
def get_mcp_server_info():
    """Get cached server info (tools/resources) - cached across reruns"""
    async def _discover():
        async with Client("mcp_server.py") as client:
            # Get tools
            tools = await client.list_tools()
            available_tools = [{"name": tool.name, "description": tool.description} for tool in tools] if tools else []
            
            # Get resources
            try:
                resources = await client.list_resources()
                available_resources = [{"uri": resource.uri, "description": resource.description} for resource in resources] if resources else []
            except:
                available_resources = []
            
            return available_tools, available_resources
    
    return asyncio.run(_discover())

async def execute_mcp_query_async(parsed_query):
    """Execute MCP query with proper async context manager"""
    start_time = time.time()
    
    action = parsed_query.get("action")
    tool_name = parsed_query.get("tool")
    parameters = parsed_query.get("params", {})
    
    results = []
    
    if action == "tool" and tool_name:
        try:
            # Use proper async context manager for each query
            async with Client("mcp_server.py") as client:
                tool_result = await client.call_tool(tool_name, parameters)
                tool_data = extract_result_data(tool_result)
                results.append({
                    "type": "tool",
                    "name": tool_name,
                    "data": tool_data,
                    "success": "error" not in tool_data
                })
        except Exception as e:
            results.append({
                "type": "error",
                "message": f"Tool call error: {e}",
                "success": False
            })
    
    elapsed_time = int((time.time() - start_time) * 1000)
    return results, elapsed_time

# --- Main App ---
def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 MCP Client Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"📍 **Session ID:** `{st.session_state.session_id}`")
        
        # LLM Provider Selection
        st.session_state.llm_provider = st.selectbox(
            "🤖 LLM Provider",
            LLM_MODELS,
            index=LLM_MODELS.index(st.session_state.llm_provider)
        )
        
        # Parsing Mode
        st.session_state.use_llm = st.checkbox(
            "🧠 Use LLM Parsing",
            value=st.session_state.use_llm
        )
        
        # API Keys Status
        st.subheader("🔑 API Keys Status")
        api_keys_status = {
            "OpenAI": "✅" if os.getenv("OPENAI_API_KEY") else "❌",
            "Anthropic": "✅" if os.getenv("ANTHROPIC_API_KEY") else "❌",
            "Gemini": "✅" if os.getenv("GEMINI_API_KEY") else "❌",
        }
        
        for provider, status in api_keys_status.items():
            st.write(f"{status} {provider}")
        
        # Server Connection using st.cache_resource for discovery only!
        st.subheader("🔌 Server Status")
        
        # Try to get cached server info (tools/resources)
        try:
            # This will use cached discovery if available
            tools, resources = get_mcp_server_info()
            
            # If we get here, server is reachable
            st.session_state.server_connected = True
            st.session_state.available_tools = tools
            st.session_state.available_resources = resources
            
        except Exception as e:
            st.session_state.server_connected = False
            st.session_state.available_tools = []
            st.session_state.available_resources = []
        
        if st.button("🔄 Refresh Server Discovery"):
            # Clear cache and rediscover
            st.cache_resource.clear()
            st.rerun()
        
        # Connection Status
        if st.session_state.server_connected:
            st.success("🟢 Server Connected (Discovery Cached)")
            
            # Show tools and resources if connected
            if st.session_state.available_tools:
                with st.expander("🔧 Available Tools"):
                    for tool in st.session_state.available_tools:
                        st.write(f"• **{tool['name']}**: {tool['description']}")
            
            if st.session_state.available_resources:
                with st.expander("📚 Available Resources"):
                    for resource in st.session_state.available_resources:
                        st.write(f"• **{resource['uri']}**: {resource['description']}")
        else:
            st.error("🔴 Server Disconnected")
            st.info("💡 Make sure mcp_server.py is running, then click 'Refresh Server Discovery'")
        
        # Example queries
        st.subheader("💡 Example Queries")
        if st.button("15 + 27"):
            st.session_state.example_query = "15 + 27"
        if st.button("sine of 30 degrees"):
            st.session_state.example_query = "sine of 30 degrees"
        if st.button("health check"):
            st.session_state.example_query = "health check"
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Query Interface")
        
        # Query Input - SIMPLE!
        default_query = st.session_state.get('example_query', '')
        user_query = st.text_input(
            "🎯 Enter your query:",
            value=default_query,
            placeholder="15 + 27"
        )
        
        # Clear example after using it
        if 'example_query' in st.session_state:
            del st.session_state.example_query
        
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submit_button = st.button("🚀 Submit Query", type="primary")
        with col_clear:
            if st.button("🗑️ Clear Session"):
                st.session_state.session_id = hashlib.md5(f"{datetime.now()}{os.getpid()}".encode()).hexdigest()[:8]
                st.session_state.last_parsed_query = None
                st.success("✅ New session started!")
                st.rerun()
        
        # Process Query using cached resources!
        if submit_button and user_query:
            try:
                # Parse query
                parsed_query = None
                model_name = None
                
                if st.session_state.use_llm:
                    parser = LLMQueryParser(st.session_state.llm_provider)
                    if parser.client:
                        parsed_query = parser.parse_query_sync(user_query)
                        model_name = parser.model_name
                    else:
                        st.warning("🔄 LLM not available, using rule-based parsing")
                        parsed_query = RuleBasedQueryParser.parse_query(user_query)
                else:
                    parsed_query = RuleBasedQueryParser.parse_query(user_query)
                
                if parsed_query:
                    # Store for debug display
                    st.session_state.last_parsed_query = parsed_query
                    
                    # Execute query with proper async context manager
                    results, elapsed_time = asyncio.run(execute_mcp_query_async(parsed_query))
                    
                    # Auto-update connection status if successful
                    if results and any(r.get('success', True) for r in results):
                        st.session_state.server_connected = True
                    
                    # Save to database
                    db_entry = {
                        'session_id': st.session_state.session_id,
                        'timestamp': datetime.now(),
                        'llm_provider': st.session_state.llm_provider if st.session_state.use_llm else None,
                        'model_name': model_name,
                        'parsing_mode': 'LLM' if st.session_state.use_llm else 'Rule-based',
                        'user_query': user_query,
                        'parsed_action': parsed_query.get('action'),
                        'tool_name': parsed_query.get('tool'),
                        'parameters': json.dumps(parsed_query.get('params', {})),
                        'confidence': parsed_query.get('confidence'),
                        'reasoning': parsed_query.get('reasoning'),
                        'elapsed_time_ms': elapsed_time,
                        'success': all(result.get('success', True) for result in results)
                    }
                    
                    entry_id = st.session_state.chat_history_db.insert_chat_entry(db_entry)
                    
                    # Display results
                    st.success(f"✅ Query processed in {elapsed_time}ms (Entry ID: {entry_id})")
                    
                    for result in results:
                        if result['type'] == 'tool':
                            formatted_display = format_result_for_display(result['name'], result['data'])
                            st.markdown(f'<div class="tool-call">{formatted_display}</div>', unsafe_allow_html=True)
                        elif result['type'] == 'error':
                            st.markdown(f'<div class="error-message">❌ {result["message"]}</div>', unsafe_allow_html=True)
                    
                else:
                    st.error("❓ I couldn't understand your query. Please try rephrasing.")
                    
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.info("💡 Try clicking 'Refresh MCP Connection' if connection issues persist")
    
    with col2:
        st.subheader("📊 Query Analysis")
        
        # Display debug info
        if st.session_state.last_parsed_query:
            parsed_query = st.session_state.last_parsed_query
            
            st.markdown('<div class="debug-info">', unsafe_allow_html=True)
            st.markdown("**🔍 Debug - Parsed Query:**")
            debug_info = {
                "Action": parsed_query.get('action'),
                "Tool": parsed_query.get('tool'),
                "Parameters": parsed_query.get('params', {}),
                "Confidence": parsed_query.get('confidence'),
                "Reasoning": parsed_query.get('reasoning')
            }
            st.json(debug_info)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Session stats
        try:
            recent_entries = st.session_state.chat_history_db.get_chat_history(
                limit=5, 
                filters={'session_id': st.session_state.session_id}
            )
            
            if recent_entries:
                latest_entry = recent_entries[0]
                st.info(f"🔍 **Parser:** {latest_entry['parsing_mode']}")
                if latest_entry['model_name']:
                    st.info(f"🤖 **Model:** {latest_entry['model_name']}")
                
                if len(recent_entries) > 1:
                    successful = sum(1 for entry in recent_entries if entry['success'])
                    avg_time = sum(entry['elapsed_time_ms'] or 0 for entry in recent_entries) / len(recent_entries)
                    
                    st.markdown("**Session Statistics:**")
                    st.metric("Queries", len(recent_entries))
                    st.metric("Success Rate", f"{(successful/len(recent_entries)*100):.1f}%")
                    st.metric("Avg Response Time", f"{avg_time:.0f}ms")
            else:
                st.info("💡 No queries in this session yet. Try asking something!")
                
        except Exception as e:
            st.error(f"Error loading query analysis: {e}")

if __name__ == "__main__":
    main()