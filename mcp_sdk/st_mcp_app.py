import streamlit as st
import asyncio
import json
import logging
import os
import sqlite3
import pandas as pd
import time
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import numpy as np
from pathlib import Path

# MCP Official SDK imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util
    from sklearn.metrics.pairwise import cosine_similarity
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.warning("⚠️ RAG features disabled. Install: pip install sentence-transformers scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit page config
st.set_page_config(
    page_title="MCP Client with RAG (Official SDK)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAMPLE_QUERIES = """
- 15 + 27
- sine of 30 degrees
- compute square root of 2
- health check
- arc tangent of 1.0
- get stock price of GOOG
- tell me about company of ticker symbol of AAPL
- server diagnostics
- repeat this message: hello MCP server
"""

CUSTOM_CSS_STYLE = """
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
    .rag-match {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    .similarity-score {
        background-color: #d4edda;
        color: #155724;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
"""

# Custom CSS
st.markdown(CUSTOM_CSS_STYLE, unsafe_allow_html=True)

# LLM Models and Provider Selection
LLM_MODELS = ["openai", "anthropic", "ollama", "gemini", "bedrock"]
DATABASE_FILE = "mcp_chat_history.db"

# --- Helper function to get server path ---
def get_mcp_server_path(server_filename="mcp_server.py"):
    """Get absolute path to MCP server file."""
    try:
        if '__file__' in globals():
            current_script_dir = Path(__file__).parent.resolve()
        else:
            current_script_dir = Path.cwd()
        
        server_path = current_script_dir / server_filename
        
        if not server_path.exists():
            raise FileNotFoundError(f"MCP server file not found: {server_path}")
        
        return server_path.resolve()
    
    except Exception as e:
        raise RuntimeError(f"Failed to determine MCP server path: {e}")

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
                    rag_matches TEXT,
                    similarity_scores TEXT,
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
                    confidence, reasoning, rag_matches, similarity_scores, response_data, 
                    formatted_response, elapsed_time_ms, error_message, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.get('session_id'), entry.get('timestamp'), entry.get('llm_provider'),
                entry.get('model_name'), entry.get('parsing_mode'), entry.get('user_query'),
                entry.get('parsed_action'), entry.get('tool_name'), entry.get('resource_uri'),
                entry.get('parameters'), entry.get('confidence'), entry.get('reasoning'),
                entry.get('rag_matches'), entry.get('similarity_scores'), entry.get('response_data'),
                entry.get('formatted_response'), entry.get('elapsed_time_ms'), 
                entry.get('error_message'), entry.get('success', True)
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

# --- RAG System for MCP Tools/Resources ---
class MCPRAGSystem:
    def __init__(self):
        self.model = None
        self.tool_embeddings = None
        self.resource_embeddings = None
        self.tool_contexts = []
        self.resource_contexts = []
        
        if RAG_AVAILABLE:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("✅ RAG System initialized with all-MiniLM-L6-v2")
        except Exception as e:
            logging.error(f"❌ Failed to initialize RAG model: {e}")
            self.model = None
    
    def build_rich_context(self, item: Dict, item_type: str) -> str:
        """Build rich context for tools and resources with examples and synonyms"""
        if item_type == "tool":
            name = item.get('name', '')
            desc = item.get('description', '')
            
            # Enhanced context with usage examples and synonyms
            context_map = {
                'calculator': """
Tool: calculator
Description: Performs mathematical arithmetic operations
Type: computation tool
Usage examples: 
- Basic math: "15 plus 27", "multiply 8 by 4", "divide 100 by 5"
- Advanced: "what's 2 to the power of 3", "square root calculation"
- Keywords: add, subtract, multiply, divide, power, math, compute, calculate
Synonyms: math, arithmetic, computation, calculate, compute
                """,
                'trig': """
Tool: trig  
Description: Trigonometric functions (sine, cosine, tangent)
Type: mathematical tool
Usage examples:
- "sine of 30 degrees", "cosine of 45", "tangent of 60 degrees"
- "sin(π/4)", "cos(0)", "tan(90 degrees)"
- Unit support: degrees, radians
Keywords: trigonometry, sine, cosine, tangent, sin, cos, tan, angle
Synonyms: trigonometry, trig functions, angles, geometry
                """,
                'health': """
Tool: health
Description: Server health check and status monitoring  
Type: diagnostic tool
Usage examples:
- "health check", "server status", "is server running"
- "ping server", "system status", "connectivity test"
Keywords: health, status, ping, check, monitor, diagnostic
Synonyms: status, ping, check, monitor, diagnostic, connectivity
                """,
                'echo': """
Tool: echo
Description: Echo back messages for testing
Type: utility tool  
Usage examples:
- "echo hello world", "repeat this message", "say hello"
- Testing connectivity and response
Keywords: echo, repeat, say, message, test
Synonyms: repeat, say, message, test, respond
                """,
                'stock_quote': """
Tool: stock_quote
Description: Get live stock market data and quotes
Type: financial tool
Usage examples:
- "get stock price of AAPL", "GOOGL stock quote", "Tesla stock price"
- "what's the current price of Microsoft stock"
Keywords: stock, quote, price, ticker, market, finance
Synonyms: stock price, share price, market data, ticker
                """
            }
            
            return context_map.get(name, f"""
Tool: {name}
Description: {desc}
Type: generic tool
Usage: General purpose tool for {name} operations
Keywords: {name}
            """).strip()
            
        elif item_type == "resource":
            uri_raw = item.get('uri', '')
            try:
                uri = str(uri_raw) if hasattr(uri_raw, '__str__') else uri_raw
            except Exception:
                uri = 'unknown_resource'
            
            desc = item.get('description', '')
            
            try:
                uri_lower = uri.lower() if isinstance(uri, str) else str(uri).lower()
                
                if 'stock' in uri_lower:
                    return f"""
Resource: {uri}
Description: {desc}
Type: financial data resource
Usage examples:
- Stock information, financial data, company details
- Market data, stock prices, financial analysis  
Keywords: stock, finance, market, company, financial, investment
Synonyms: stocks, shares, equity, financial data, market data
                    """.strip()
                else:
                    try:
                        resource_name = uri.split('/')[-1] if '/' in str(uri) else str(uri)
                    except Exception:
                        resource_name = 'resource'
                    
                    return f"""
Resource: {uri}
Description: {desc}
Type: data resource
Usage: Access to {uri} data and information
Keywords: {resource_name}
                    """.strip()
            except Exception as e:
                logging.warning(f"Failed to process resource URI: {e}")
                return f"""
Resource: {uri}
Description: {desc}
Type: data resource
Usage: General data resource
Keywords: data, resource
                """.strip()
        
        return f"{item_type}: {item}"
    
    def build_embeddings(self, tools: List[Dict], resources: List[Dict]):
        """Build embeddings for all tools and resources"""
        if not self.model:
            return
        
        try:
            # Build rich contexts with error handling
            self.tool_contexts = []
            for tool in tools:
                try:
                    context = self.build_rich_context(tool, 'tool')
                    self.tool_contexts.append({
                        'name': tool.get('name', ''),
                        'description': tool.get('description', ''),
                        'context': context,
                        'type': 'tool'
                    })
                except Exception as e:
                    logging.warning(f"Failed to build context for tool {tool.get('name', 'unknown')}: {e}")
                    self.tool_contexts.append({
                        'name': tool.get('name', 'unknown'),
                        'description': tool.get('description', ''),
                        'context': f"Tool: {tool.get('name', 'unknown')}\nDescription: {tool.get('description', '')}",
                        'type': 'tool'
                    })
            
            self.resource_contexts = []
            for resource in resources:
                try:
                    context = self.build_rich_context(resource, 'resource')
                    uri_raw = resource.get('uri', '')
                    uri = str(uri_raw) if uri_raw else 'unknown_resource'
                    
                    self.resource_contexts.append({
                        'uri': uri,
                        'description': resource.get('description', ''),
                        'context': context,
                        'type': 'resource'
                    })
                except Exception as e:
                    logging.warning(f"Failed to build context for resource {resource.get('uri', 'unknown')}: {e}")
                    uri_raw = resource.get('uri', 'unknown_resource')
                    uri = str(uri_raw) if uri_raw else 'unknown_resource'
                    self.resource_contexts.append({
                        'uri': uri,
                        'description': resource.get('description', ''),
                        'context': f"Resource: {uri}\nDescription: {resource.get('description', '')}",
                        'type': 'resource'
                    })
            
            # Create embeddings with error handling
            if self.tool_contexts:
                try:
                    tool_texts = [item['context'] for item in self.tool_contexts]
                    self.tool_embeddings = self.model.encode(tool_texts)
                    logging.info(f"✅ Built embeddings for {len(self.tool_contexts)} tools")
                except Exception as e:
                    logging.error(f"❌ Failed to encode tool embeddings: {e}")
                    self.tool_embeddings = None
            
            if self.resource_contexts:
                try:
                    resource_texts = [item['context'] for item in self.resource_contexts]
                    self.resource_embeddings = self.model.encode(resource_texts)
                    logging.info(f"✅ Built embeddings for {len(self.resource_contexts)} resources")
                except Exception as e:
                    logging.error(f"❌ Failed to encode resource embeddings: {e}")
                    self.resource_embeddings = None
                
        except Exception as e:
            logging.error(f"❌ Failed to build embeddings: {e}")
            self.tool_contexts = []
            self.resource_contexts = []
            self.tool_embeddings = None
            self.resource_embeddings = None
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform optimized semantic search across tools and resources"""
        if not self.model:
            return []
        
        try:
            query_embedding = self.model.encode([query])
            
            all_embeddings = []
            all_contexts = []
            
            if self.tool_embeddings is not None and len(self.tool_embeddings) > 0:
                all_embeddings.append(self.tool_embeddings)
                all_contexts.extend([(ctx, 'tool') for ctx in self.tool_contexts])
            
            if self.resource_embeddings is not None and len(self.resource_embeddings) > 0:
                all_embeddings.append(self.resource_embeddings)
                all_contexts.extend([(ctx, 'resource') for ctx in self.resource_contexts])
            
            if not all_embeddings:
                return []
            
            # Concatenate all embeddings into single corpus
            corpus_embeddings = np.concatenate(all_embeddings, axis=0)
            
            # Use optimized semantic_search from sentence-transformers
            search_results = st_util.semantic_search(
                query_embedding, 
                corpus_embeddings, 
                top_k=top_k,
                score_function=st_util.cos_sim
            )[0]  # Get results for first (and only) query
            
            # Map results back to our context format
            results = []
            for hit in search_results:
                corpus_id = hit['corpus_id']
                similarity = float(hit['score'])
                
                # Only include results above minimum threshold
                if similarity > 0.1:
                    context_item, item_type = all_contexts[corpus_id]
                    results.append({
                        'item': context_item,
                        'similarity': similarity,
                        'type': item_type
                    })
            
            logging.info(f"✅ Semantic search found {len(results)} relevant items for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logging.error(f"❌ Optimized semantic search failed: {e}")
            return self._fallback_semantic_search(query, top_k)
    
    def _fallback_semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Fallback to original semantic search method if optimized version fails"""
        if not self.model:
            return []
        
        try:
            query_embedding = self.model.encode([query])
            results = []
            
            # Search tools using original method
            if self.tool_embeddings is not None and len(self.tool_embeddings) > 0:
                tool_similarities = cosine_similarity(query_embedding, self.tool_embeddings)[0]
                
                for i, similarity in enumerate(tool_similarities):
                    if similarity > 0.1:  # Minimum similarity threshold
                        results.append({
                            'item': self.tool_contexts[i],
                            'similarity': float(similarity),
                            'type': 'tool'
                        })
            
            # Search resources using original method
            if self.resource_embeddings is not None and len(self.resource_embeddings) > 0:
                resource_similarities = cosine_similarity(query_embedding, self.resource_embeddings)[0]
                
                for i, similarity in enumerate(resource_similarities):
                    if similarity > 0.1:  # Minimum similarity threshold
                        results.append({
                            'item': self.resource_contexts[i],
                            'similarity': float(similarity),
                            'type': 'resource'
                        })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            logging.warning(f"⚠️ Used fallback search method for query: '{query[:50]}...'")
            return results[:top_k]
            
        except Exception as e:
            logging.error(f"❌ Fallback semantic search also failed: {e}")
            return []
    
    def build_dynamic_prompt(self, relevant_items: List[Dict], query: str) -> str:
        """Build dynamic system prompt based on relevant items"""
        if not relevant_items:
            return "You are a tool selection assistant. Respond with ONLY a JSON object with action, tool, params, confidence, and reasoning fields."
        
        # Build tools section
        tools_section = "Available tools:\n"
        for item in relevant_items:
            if item['type'] == 'tool':
                tool_info = item['item']
                similarity = item['similarity']
                tools_section += f"- {tool_info['name']}: {tool_info['description']} (relevance: {similarity:.2f})\n"
        
        # Build resources section
        resources_section = "\nAvailable resources:\n"
        for item in relevant_items:
            if item['type'] == 'resource':
                resource_info = item['item']
                similarity = item['similarity']
                resources_section += f"- {resource_info['uri']}: {resource_info['description']} (relevance: {similarity:.2f})\n"
        
        # Examples based on most relevant items
        examples_section = "\nExamples based on available tools:\n"
        
        # Add specific examples for discovered tools
        for item in relevant_items[:3]:  # Top 3 most relevant
            if item['type'] == 'tool':
                tool_name = item['item']['name']
                if tool_name == 'calculator':
                    examples_section += '- "15 plus 27" -> {"action": "tool", "tool": "calculator", "params": {"operation": "add", "num1": 15, "num2": 27}, "confidence": 0.98, "reasoning": "Simple addition"}\n'
                elif tool_name == 'trig':
                    examples_section += '- "sine of 30 degrees" -> {"action": "tool", "tool": "trig", "params": {"operation": "sine", "num1": 30, "unit": "degree"}, "confidence": 0.95, "reasoning": "Trigonometric calculation"}\n'
                elif tool_name == 'health':
                    examples_section += '- "health check" -> {"action": "tool", "tool": "health", "params": {}, "confidence": 0.9, "reasoning": "Server health check"}\n'
                elif tool_name == 'echo':
                    examples_section += '- "echo hello" -> {"action": "tool", "tool": "echo", "params": {"message": "hello"}, "confidence": 0.95, "reasoning": "Echo command"}\n'
                elif tool_name == 'stock_quote':
                    examples_section += '- "get stock price of AAPL" -> {"action": "tool", "tool": "stock_quote", "params": {"ticker": "AAPL"}, "confidence": 0.9, "reasoning": "Stock price query"}\n'
        
        system_prompt = f"""You are an intelligent tool selection assistant. Analyze the user query and respond with ONLY a JSON object:

{{
    "action": "tool",
    "tool": "tool_name_or_null",
    "params": {{"param1": "value1"}},
    "confidence": 0.95,
    "reasoning": "Brief explanation"
}}

{tools_section}{resources_section}{examples_section}

Instructions:
- Only use tools/resources listed above
- Consider the relevance scores when making decisions
- Set confidence based on query clarity and tool match
- If no tool matches well (all relevance < 0.3), set tool to null
- Respond with ONLY the JSON object, no other text."""

        return system_prompt

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
    if 'use_rag' not in st.session_state:
        st.session_state.use_rag = True
    if 'last_parsed_query' not in st.session_state:
        st.session_state.last_parsed_query = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MCPRAGSystem()
    if 'last_rag_matches' not in st.session_state:
        st.session_state.last_rag_matches = []

# --- Enhanced LLM Query Parser with RAG ---
class LLMQueryParser:
    def __init__(self, provider: str = "gemini"):
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
    
    def parse_query_with_rag(self, query: str, rag_system: MCPRAGSystem) -> Optional[Dict[str, Any]]:
        """Parse query using RAG-enhanced semantic search"""
        if not self.client or not rag_system.model:
            return None
        
        try:
            # Perform semantic search
            relevant_items = rag_system.semantic_search(query, top_k=5)
            
            # Store RAG matches for debugging
            st.session_state.last_rag_matches = relevant_items
            
            if not relevant_items:
                # Fallback to standard parsing if no relevant items found
                return self.parse_query_sync(query)
            
            # Build dynamic prompt based on relevant items
            system_prompt = rag_system.build_dynamic_prompt(relevant_items, query)
            
            # Get LLM response with dynamic prompt
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=400,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": f"Query: {query}"}]
                )
                llm_response = response.content[0].text.strip()
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Query: {query}"}
                    ],
                    temperature=0.1,
                    max_tokens=400
                )
                llm_response = response.choices[0].message.content.strip()
            
            elif self.provider == "gemini":
                response = self.client.generate_content(
                    f"{system_prompt}\n\nUser Query: {query}",
                    generation_config={"temperature": 0.1, "max_output_tokens": 400}
                )
                llm_response = response.text.strip()
            
            # Clean and parse JSON
            if llm_response.startswith("```json"):
                llm_response = llm_response.replace("```json", "").replace("```", "").strip()
            elif llm_response.startswith("```"):
                llm_response = llm_response.replace("```", "").strip()
            
            parsed_response = json.loads(llm_response)
            
            # Add RAG metadata
            parsed_response['rag_enhanced'] = True
            parsed_response['rag_matches'] = len(relevant_items)
            
            if parsed_response.get("action") and parsed_response.get("confidence", 0) >= 0.3:
                return parsed_response
            
        except Exception as e:
            logging.error(f"RAG-enhanced parsing error: {e}")
            # Fallback to standard parsing
            return self.parse_query_sync(query)
        
        return None
    
    def parse_query_sync(self, query: str) -> Optional[Dict[str, Any]]:
        """Legacy parsing method with hardcoded examples"""
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
- stock_quote: ticker (e.g., AAPL, GOOGL)
- health: no parameters
- echo: message

Examples:
"15 plus 27" -> {"action": "tool", "tool": "calculator", "params": {"operation": "add", "num1": 15, "num2": 27}, "confidence": 0.98, "reasoning": "Simple addition"}
"sine of 30 degrees" -> {"action": "tool", "tool": "trig", "params": {"operation": "sine", "num1": 30, "unit": "degree"}, "confidence": 0.95, "reasoning": "Trigonometric calculation"}
"get stock price of AAPL" -> {"action": "tool", "tool": "stock_quote", "params": {"ticker": "AAPL"}, "confidence": 0.9, "reasoning": "Stock price query"}

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
            parsed_response['rag_enhanced'] = False
            
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
            return {"action": "tool", "tool": "health", "params": {}, "confidence": 0.9, "reasoning": "Health check request", "rag_enhanced": False}
        
        # Echo command
        if query_lower.startswith("echo "):
            return {"action": "tool", "tool": "echo", "params": {"message": query[5:].strip()}, "confidence": 0.95, "reasoning": "Echo command", "rag_enhanced": False}
        
        # Stock quote
        stock_keywords = ["stock", "price", "quote", "ticker"]
        if any(keyword in query_lower for keyword in stock_keywords):
            # Look for ticker symbols (3-5 uppercase letters)
            ticker_match = re.search(r'\b[A-Z]{1,5}\b', query.upper())
            if ticker_match:
                ticker = ticker_match.group()
                return {"action": "tool", "tool": "stock_quote", "params": {"ticker": ticker}, "confidence": 0.85, "reasoning": f"Stock quote for {ticker}", "rag_enhanced": False}
        
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
                            "reasoning": f"Calculator operation: {operation}",
                            "rag_enhanced": False
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
                            "reasoning": f"Trigonometry: {operation}",
                            "rag_enhanced": False
                        }
        
        return None

# --- Utility Functions ---
def extract_result_data(result):
    """Extract result data from MCP response"""
    try:
        if hasattr(result, 'content') and result.content:
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
    """Format result for display"""
    if isinstance(result, dict) and "error" in result:
        return f"❌ [Error] {result['error']}"
    
    if tool_name == "calculator":
        expression = result.get('expression', f"{result.get('num1', '?')} {result.get('operation', '?')} {result.get('num2', '?')} = {result.get('result', '?')}")
        return f"🧮 [Calculator] {expression}"
    
    elif tool_name == "trig":
        expression = result.get('expression', f"{result.get('operation', '?')}({result.get('num1', '?')}) = {result.get('result', '?')}")
        return f"📐 [Trigonometry] {expression}"
    
    elif tool_name == "stock_quote":
        if "error" in result:
            return f"❌ [Stock] {result['error']}"
        ticker = result.get('ticker', 'Unknown')
        price = result.get('current_price', 'N/A')
        company = result.get('company_name', 'Unknown Company')
        return f"📈 [Stock] {company} ({ticker}): ${price}"
    
    elif tool_name == "health":
        return f"✅ [Health] {result.get('message', 'Server is healthy')}"
    
    elif tool_name == "echo":
        return f"🔊 [Echo] {result.get('echo', result.get('message', str(result)))}"
    
    return f"✅ [Result] {json.dumps(result, indent=2)}"

# --- CACHED MCP Operations using st.cache_resource ---
@st.cache_resource(show_spinner=False)
def get_mcp_server_info():
    """Get cached server info (tools/resources) - cached across reruns"""
    async def _discover():
        try:
            server_path = str(get_mcp_server_path("mcp_server.py"))
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_path],
                env=dict(os.environ)  # Pass current environment
            )
            
            # Set a shorter timeout for Windows
            async with asyncio.timeout(10):  # 10 second timeout
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        # Initialize the session
                        init_result = await session.initialize()
                        logging.info(f"Session initialized: {init_result}")
                        
                        # Get tools
                        tools_result = await session.list_tools()
                        available_tools = []
                        if tools_result and tools_result.tools:
                            available_tools = [
                                {"name": tool.name, "description": tool.description} 
                                for tool in tools_result.tools
                            ]
                        
                        # Get resources
                        try:
                            resources_result = await session.list_resources()
                            available_resources = []
                            if resources_result and resources_result.resources:
                                available_resources = [
                                    {"uri": str(resource.uri), "description": resource.description} 
                                    for resource in resources_result.resources
                                ]
                        except Exception as e:
                            logging.warning(f"Failed to get resources: {e}")
                            available_resources = []
                        
                        logging.info(f"Discovered {len(available_tools)} tools and {len(available_resources)} resources")
                        return available_tools, available_resources
                        
        except asyncio.TimeoutError:
            logging.error("Server discovery timed out - is mcp_server.py running?")
            raise Exception("Server discovery timed out. Make sure mcp_server.py is running.")
        except Exception as e:
            logging.error(f"Failed to discover server info: {e}")
            raise e
    
    return asyncio.run(_discover())

async def execute_mcp_query_async(parsed_query):
    """Execute MCP query with proper async context manager using official SDK"""
    start_time = time.time()
    
    action = parsed_query.get("action")
    tool_name = parsed_query.get("tool")
    parameters = parsed_query.get("params", {})
    
    results = []
    
    if action == "tool" and tool_name:
        try:
            server_path = str(get_mcp_server_path("mcp_server.py"))
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_path],
                env=dict(os.environ)  # Pass current environment
            )
            
            # Set timeout for tool execution
            async with asyncio.timeout(30):  # 30 second timeout for tool calls
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        # Initialize the session
                        await session.initialize()
                        
                        # Call the tool
                        logging.info(f"Calling tool {tool_name} with params: {parameters}")
                        tool_result = await session.call_tool(tool_name, parameters)
                        tool_data = extract_result_data(tool_result)
                        
                        results.append({
                            "type": "tool",
                            "name": tool_name,
                            "data": tool_data,
                            "success": "error" not in tool_data
                        })
        except asyncio.TimeoutError:
            results.append({
                "type": "error",
                "message": f"Tool call timed out after 30 seconds",
                "success": False
            })
        except Exception as e:
            logging.error(f"Tool call error: {e}")
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
    st.markdown('<h1 class="main-header">🧠 MCP Client with RAG (Official SDK)</h1>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"📍 [Session ID] `{st.session_state.session_id}`")
        
        # RAG System Status
        st.subheader("🧠 RAG System")
        if RAG_AVAILABLE and st.session_state.rag_system.model:
            st.success("✅ RAG System Active")
            st.info("🔍 Semantic search enabled")
            
            # RAG settings
            st.session_state.use_rag = st.checkbox(
                "🎯 Use RAG-Enhanced Parsing",
                value=st.session_state.use_rag,
                help="Use semantic search to find relevant tools dynamically"
            )
        else:
            st.error("❌ RAG System Disabled")
            st.warning("Install: `pip install sentence-transformers scikit-learn`")
            st.session_state.use_rag = False
        
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
        
        # Parsing Mode Display
        if st.session_state.use_llm and st.session_state.use_rag and RAG_AVAILABLE:
            st.info("🎯 [Mode] RAG-Enhanced LLM")
        elif st.session_state.use_llm:
            st.info("🤖 [Mode] Legacy LLM")
        else:
            st.info("📝 [Mode] Rule-based")
        
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
            with st.spinner("🔍 Discovering MCP server..."):
                # This will use cached discovery if available
                tools, resources = get_mcp_server_info()
                
                # If we get here, server is reachable
                st.session_state.server_connected = True
                st.session_state.available_tools = tools
                st.session_state.available_resources = resources
                
                # Build RAG embeddings when tools/resources are available
                if RAG_AVAILABLE and st.session_state.rag_system.model:
                    st.session_state.rag_system.build_embeddings(tools, resources)
            
        except Exception as e:
            st.session_state.server_connected = False
            st.session_state.available_tools = []
            st.session_state.available_resources = []
            
            # Show the specific error to help debugging
            if "timed out" in str(e).lower():
                st.error("🔴 Server Connection Timeout")
                st.info("💡 Make sure `python mcp_server.py` is running in another terminal")
            else:
                st.error(f"🔴 Server Error: {str(e)[:100]}...")
        
        if st.button("🔄 Refresh Server Discovery"):
            # Clear cache and rediscover
            st.cache_resource.clear()
            st.rerun()
        
        # Connection Status
        if st.session_state.server_connected:
            st.success("🟢 Server Connected")
            
            # Show tools and resources if connected
            if st.session_state.available_tools:
                with st.expander("🔧 Available Tools"):
                    for tool in st.session_state.available_tools:
                        st.write(f"• [{tool['name']}] {tool['description']}")
            
            if st.session_state.available_resources:
                with st.expander("📚 Available Resources"):
                    for resource in st.session_state.available_resources:
                        st.write(f"• [{resource['uri']}] {resource['description']}")
            
            # RAG Embeddings Status
            if RAG_AVAILABLE and st.session_state.rag_system.model:
                tool_count = len(st.session_state.rag_system.tool_contexts)
                resource_count = len(st.session_state.rag_system.resource_contexts)
                st.info(f"🎯 RAG: {tool_count} tools, {resource_count} resources indexed")
        else:
            st.error("🔴 Server Disconnected")
            st.info("💡 Make sure mcp_server.py is running, then click 'Refresh Server Discovery'")
        
        # Example queries
        st.subheader("💡 Example Queries")
        st.markdown(SAMPLE_QUERIES)

    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Query Interface")
        
        # Query Input
        default_query = st.session_state.get('example_query', '')
        user_query = st.text_input(
            "🎯 Enter your query:",
            value=default_query,
            placeholder="compute the square root of 144"
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
                st.session_state.last_rag_matches = []
                st.success("✅ New session started!")
                st.rerun()
        
        # Process Query using RAG!
        if submit_button and user_query:
            try:
                # Parse query
                parsed_query = None
                model_name = None
                rag_matches = []
                
                if st.session_state.use_llm:
                    parser = LLMQueryParser(st.session_state.llm_provider)
                    if parser.client:
                        # Use RAG-enhanced parsing if available
                        if st.session_state.use_rag and RAG_AVAILABLE and st.session_state.rag_system.model:
                            parsed_query = parser.parse_query_with_rag(user_query, st.session_state.rag_system)
                            rag_matches = st.session_state.last_rag_matches
                        else:
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
                    st.session_state.elapsed_time = elapsed_time

                    # Auto-update connection status if successful
                    if results and any(r.get('success', True) for r in results):
                        st.session_state.server_connected = True
                    
                    # Save to database with RAG data
                    db_entry = {
                        'session_id': st.session_state.session_id,
                        'timestamp': datetime.now(),
                        'llm_provider': st.session_state.llm_provider if st.session_state.use_llm else None,
                        'model_name': model_name,
                        'parsing_mode': 'RAG-Enhanced' if parsed_query.get('rag_enhanced') else ('LLM' if st.session_state.use_llm else 'Rule-based'),
                        'user_query': user_query,
                        'parsed_action': parsed_query.get('action'),
                        'tool_name': parsed_query.get('tool'),
                        'parameters': json.dumps(parsed_query.get('params', {})),
                        'confidence': parsed_query.get('confidence'),
                        'reasoning': parsed_query.get('reasoning'),
                        'rag_matches': json.dumps([{
                            'name': m['item'].get('name', m['item'].get('uri', '')),
                            'similarity': m['similarity'],
                            'type': m['type']
                        } for m in rag_matches]) if rag_matches else None,
                        'similarity_scores': json.dumps([m['similarity'] for m in rag_matches]) if rag_matches else None,
                        'elapsed_time_ms': elapsed_time,
                        'success': all(result.get('success', True) for result in results)
                    }
                    
                    entry_id = st.session_state.chat_history_db.insert_chat_entry(db_entry)
                    st.session_state.entry_id = entry_id

                    # Show RAG matches if available
                    if rag_matches:
                        st.markdown("### 🎯 RAG Search Results:")
                        for i, match in enumerate(rag_matches[:3]):  # Show top 3
                            item = match['item']
                            similarity = match['similarity']
                            item_type = match['type']
                            
                            if item_type == 'tool':
                                name = item.get('name', 'Unknown')
                                desc = item.get('description', '')
                            else:
                                name = item.get('uri', 'Unknown')
                                desc = item.get('description', '')
                            
                            st.markdown(f"""
                            <div class="rag-match">
                                <strong>#{i+1} {item_type.title()}:</strong> {name}<br>
                                <small>{desc}</small><br>
                                <span class="similarity-score">Similarity: {similarity:.3f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
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
                st.info("💡 Try clicking 'Refresh Server Discovery' if connection issues persist")
    
    with col2:
        st.subheader("📊 Query Analysis")

        # Display debug info
        if st.session_state.last_parsed_query:
            parsed_query = st.session_state.last_parsed_query
            parsing_mode = "RAG-Enhanced" if parsed_query.get('rag_enhanced') else "Legacy"
            st.success(f"✅ Query processed in {st.session_state.elapsed_time}ms using {parsing_mode} parsing (Entry ID: {st.session_state.entry_id})")
            
            st.markdown('<div class="debug-info">', unsafe_allow_html=True)
            st.markdown("🔍 Debug - Parsed Query:")
            debug_info = {
                "Action": parsed_query.get('action'),
                "Tool": parsed_query.get('tool'),
                "Parameters": parsed_query.get('params', {}),
                "Confidence": parsed_query.get('confidence'),
                "Reasoning": parsed_query.get('reasoning'),
                "RAG Enhanced": parsed_query.get('rag_enhanced', False),
                "RAG Matches": parsed_query.get('rag_matches', 0)
            }
            st.json(debug_info)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # RAG matches details
        if st.session_state.last_rag_matches:
            with st.expander("🎯 Detailed RAG Matches"):
                for i, match in enumerate(st.session_state.last_rag_matches):
                    item = match['item']
                    similarity = match['similarity']
                    item_type = match['type']
                    
                    st.write(f"[Match #{i+1} ({item_type})]")
                    st.write(f"• Similarity: {similarity:.4f}")
                    if item_type == 'tool':
                        st.write(f"• Tool: {item.get('name', 'Unknown')}")
                        st.write(f"• Description: {item.get('description', 'No description')}")
                    else:
                        st.write(f"• Resource: {item.get('uri', 'Unknown')}")
                        st.write(f"• Description: {item.get('description', 'No description')}")
                    st.write("---")
        
        # Session stats
        try:
            recent_entries = st.session_state.chat_history_db.get_chat_history(
                limit=5, 
                filters={'session_id': st.session_state.session_id}
            )
            
            if recent_entries:
                latest_entry = recent_entries[0]
                st.info(f"🔍 [Parser] {latest_entry['parsing_mode']}")
                if latest_entry['model_name']:
                    st.info(f"🤖 [Model] {latest_entry['model_name']}")
                
                if len(recent_entries) > 1:
                    successful = sum(1 for entry in recent_entries if entry['success'])
                    avg_time = sum(entry['elapsed_time_ms'] or 0 for entry in recent_entries) / len(recent_entries)
                    rag_enhanced_count = sum(1 for entry in recent_entries if entry['parsing_mode'] == 'RAG-Enhanced')
                    
                    st.markdown("[Session Statistics]")
                    st.metric("Queries", len(recent_entries))
                    st.metric("Success Rate", f"{(successful/len(recent_entries)*100):.1f}%")
                    st.metric("Avg Response Time", f"{avg_time:.0f}ms")
                    st.metric("RAG-Enhanced", f"{rag_enhanced_count}/{len(recent_entries)}")
            else:
                st.info("💡 No queries in this session yet. Try asking something!")
                
        except Exception as e:
            st.error(f"Error loading query analysis: {e}")

# Entry point
if __name__ == "__main__":
    main()