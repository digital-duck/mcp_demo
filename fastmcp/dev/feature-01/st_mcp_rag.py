import streamlit as st
import asyncio
import json
import logging
import os
import sqlite3
import pandas as pd
import time
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union
from fastmcp import Client
from datetime import datetime
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures

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
    page_title="Enhanced MCP Client with RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

SAMPLE_QUERIES = """
**Single Operations:**
- 15 + 27
- convert 5 feet to meters  
- analyze this text: Hello world
- sine of 30 degrees

**Batch Operations:**
- Calculate 15 + 27, then find sine of that result
- Convert 100 km to miles, then multiply by 1.5
- Analyze text 'Hello world', then echo the word count
- Calculate 2^3, then convert that inches to cm

**Custom Tool Examples:**
- Convert 5.5 feet to centimeters
- How many words in: The quick brown fox jumps
- Convert 100 kilometers to miles
- Text statistics for: Lorem ipsum dolor sit amet
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
    .batch-operation {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    .custom-tool {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
</style>
"""

# LLM Models and Provider Selection
LLM_PROVIDER_MAP = {
    "google": [
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "openai": [
        "gpt-4o-mini", 
        "gpt-4o", 
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022", 
        "claude-3-7-sonnet",
    ],
}

MCP_SERVER_PATH = "mcp_server.py"
SQLITE_DB_FILE = "mcp_chat_history.db"

TABLE_CHAT_HISTORY = "chat_history"
CHAT_HISTORY_DDL = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_CHAT_HISTORY} (
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
        success BOOLEAN NOT NULL DEFAULT 1,
        is_batch BOOLEAN DEFAULT 0,
        operation_count INTEGER DEFAULT 1
    )
"""

# ============================================================================
# DYNAMIC TOOL REGISTRATION SYSTEM
# ============================================================================

@dataclass
class ToolParameter:
    """Define a tool parameter with validation"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

@dataclass
class CustomToolDefinition:
    """Complete tool definition for registration"""
    name: str
    description: str
    category: str
    parameters: List[ToolParameter]
    endpoint: str  # URL or local function reference
    method: str = "POST"  # HTTP method or "FUNCTION" for local functions
    examples: List[str] = None
    tags: List[str] = None
    author: str = None
    version: str = "1.0.0"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.examples is None:
            self.examples = []
        if self.tags is None:
            self.tags = []

class DynamicToolRegistry:
    """Registry for managing custom tools"""
    
    def __init__(self, registry_file: str = "custom_tools.json"):
        self.registry_file = registry_file
        self.tools: Dict[str, CustomToolDefinition] = {}
        self.local_functions: Dict[str, callable] = {}
        self.load_registry()
    
    def register_tool(self, tool_def: CustomToolDefinition) -> bool:
        """Register a new tool"""
        try:
            # Validate tool definition
            if not self._validate_tool_definition(tool_def):
                return False
            
            # Store tool
            self.tools[tool_def.name] = tool_def
            
            # Save to file
            self.save_registry()
            
            logging.info(f"✅ Registered tool: {tool_def.name}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to register tool {tool_def.name}: {e}")
            return False
    
    def register_local_function(self, tool_name: str, function: callable):
        """Register a local Python function as a tool"""
        self.local_functions[tool_name] = function
        logging.info(f"✅ Registered local function: {tool_name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Remove a tool from registry"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            if tool_name in self.local_functions:
                del self.local_functions[tool_name]
            self.save_registry()
            logging.info(f"🗑️ Unregistered tool: {tool_name}")
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[CustomToolDefinition]:
        """Get tool definition by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[CustomToolDefinition]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tools_by_category(self, category: str) -> List[CustomToolDefinition]:
        """Get tools filtered by category"""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def search_tools(self, query: str) -> List[CustomToolDefinition]:
        """Search tools by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower() or
                any(query_lower in tag.lower() for tag in tool.tags)):
                results.append(tool)
        
        return results
    
    def _validate_tool_definition(self, tool_def: CustomToolDefinition) -> bool:
        """Validate tool definition"""
        if not tool_def.name or not tool_def.description:
            logging.error("Tool name and description are required")
            return False
        
        if tool_def.name in self.tools:
            logging.warning(f"Tool {tool_def.name} already exists, will overwrite")
        
        # Validate parameters
        for param in tool_def.parameters:
            if param.type not in ["string", "number", "boolean", "array", "object"]:
                logging.error(f"Invalid parameter type: {param.type}")
                return False
        
        return True
    
    def save_registry(self):
        """Save registry to file"""
        try:
            registry_data = {
                name: {
                    **asdict(tool_def),
                    'created_at': tool_def.created_at.isoformat()
                }
                for name, tool_def in self.tools.items()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save registry: {e}")
    
    def load_registry(self):
        """Load registry from file"""
        try:
            if Path(self.registry_file).exists():
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for name, tool_data in registry_data.items():
                    # Convert back to proper objects
                    tool_data['created_at'] = datetime.fromisoformat(tool_data['created_at'])
                    tool_data['parameters'] = [
                        ToolParameter(**param) for param in tool_data['parameters']
                    ]
                    self.tools[name] = CustomToolDefinition(**tool_data)
                
                logging.info(f"✅ Loaded {len(self.tools)} custom tools from registry")
                
        except Exception as e:
            logging.error(f"Failed to load registry: {e}")
            self.tools = {}

# ============================================================================
# BATCH OPERATIONS SYSTEM
# ============================================================================

@dataclass
class BatchOperation:
    """Single operation in a batch"""
    id: str
    tool: str
    params: Dict[str, Any]
    depends_on: Optional[List[str]] = None
    variable_name: Optional[str] = None

@dataclass
class BatchRequest:
    """Complete batch request"""
    operations: List[BatchOperation]
    parallel: bool = False
    fail_fast: bool = True
    timeout: int = 300

@dataclass
class BatchResult:
    """Result of batch execution"""
    operation_id: str
    tool: str
    success: bool
    result: Any = None
    error: str = None
    execution_time_ms: int = 0
    dependencies_resolved: List[str] = None

class BatchProcessor:
    """Process batch operations with dependency resolution"""
    
    def __init__(self, custom_registry: DynamicToolRegistry):
        self.custom_registry = custom_registry
        self.variables = {}
    
    async def execute_batch(self, batch_request: BatchRequest) -> List[BatchResult]:
        """Execute a batch of operations"""
        results = []
        completed_ops = set()
        self.variables = {}  # Reset variables for each batch
        
        try:
            if batch_request.parallel:
                results = await self._execute_parallel(batch_request, completed_ops)
            else:
                results = await self._execute_sequential(batch_request, completed_ops)
                
        except Exception as e:
            logging.error(f"Batch execution failed: {e}")
            results.append(BatchResult(
                operation_id="batch_error",
                tool="batch",
                success=False,
                error=str(e)
            ))
        
        return results
    
    async def _execute_sequential(self, batch_request: BatchRequest, completed_ops: set) -> List[BatchResult]:
        """Execute operations sequentially with dependency resolution"""
        results = []
        operations = batch_request.operations.copy()
        
        while operations and len(completed_ops) < len(batch_request.operations):
            # Find operations that can be executed
            ready_ops = [
                op for op in operations 
                if not op.depends_on or all(dep in completed_ops for dep in op.depends_on)
            ]
            
            if not ready_ops:
                remaining_ops = [op.id for op in operations]
                error_result = BatchResult(
                    operation_id="dependency_error",
                    tool="batch",
                    success=False,
                    error=f"Circular or missing dependencies for operations: {remaining_ops}"
                )
                results.append(error_result)
                break
            
            # Execute first ready operation
            operation = ready_ops[0]
            result = await self._execute_single_operation(operation, batch_request.fail_fast)
            results.append(result)
            
            if result.success:
                completed_ops.add(operation.id)
                if operation.variable_name:
                    self.variables[operation.variable_name] = result.result
            elif batch_request.fail_fast:
                break
            
            operations.remove(operation)
        
        return results
    
    async def _execute_parallel(self, batch_request: BatchRequest, completed_ops: set) -> List[BatchResult]:
        """Execute operations in parallel where dependencies allow"""
        results = []
        operations = batch_request.operations.copy()
        
        while operations and len(completed_ops) < len(batch_request.operations):
            ready_ops = [
                op for op in operations 
                if not op.depends_on or all(dep in completed_ops for dep in op.depends_on)
            ]
            
            if not ready_ops:
                break
            
            tasks = [
                self._execute_single_operation(op, batch_request.fail_fast)
                for op in ready_ops
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    result = BatchResult(
                        operation_id=ready_ops[i].id,
                        tool=ready_ops[i].tool,
                        success=False,
                        error=str(result)
                    )
                
                results.append(result)
                
                if result.success:
                    completed_ops.add(ready_ops[i].id)
                    if ready_ops[i].variable_name:
                        self.variables[ready_ops[i].variable_name] = result.result
                elif batch_request.fail_fast:
                    return results
            
            for op in ready_ops:
                operations.remove(op)
        
        return results
    
    async def _execute_single_operation(self, operation: BatchOperation, fail_fast: bool) -> BatchResult:
        """Execute a single operation"""
        start_time = time.time()
        
        try:
            resolved_params = self._resolve_parameters(operation.params)
            
            custom_tool = self.custom_registry.get_tool(operation.tool)
            if custom_tool:
                result = await self._execute_custom_tool(custom_tool, resolved_params)
            else:
                result = await self._execute_mcp_tool(operation.tool, resolved_params)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return BatchResult(
                operation_id=operation.id,
                tool=operation.tool,
                success=True,
                result=result,
                execution_time_ms=execution_time,
                dependencies_resolved=operation.depends_on or []
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return BatchResult(
                operation_id=operation.id,
                tool=operation.tool,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
                dependencies_resolved=operation.depends_on or []
            )
    
    def _resolve_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve variable references in parameters"""
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                if var_name in self.variables:
                    resolved[key] = self.variables[var_name]
                else:
                    raise ValueError(f"Variable '{var_name}' not found")
            else:
                resolved[key] = value
        
        return resolved
    
    async def _execute_custom_tool(self, tool_def: CustomToolDefinition, params: Dict[str, Any]) -> Any:
        """Execute a custom tool"""
        if tool_def.method == "FUNCTION":
            if tool_def.name in self.custom_registry.local_functions:
                func = self.custom_registry.local_functions[tool_def.name]
                if asyncio.iscoroutinefunction(func):
                    return await func(**params)
                else:
                    return func(**params)
            else:
                raise ValueError(f"Local function not found: {tool_def.name}")
        else:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    tool_def.method,
                    tool_def.endpoint,
                    json=params
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise ValueError(f"HTTP {response.status}: {await response.text()}")
    
    async def _execute_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a standard MCP tool"""
        async with Client(MCP_SERVER_PATH) as client:
            result = await client.call_tool(tool_name, params)
            return extract_result_data(result)

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

class ChatHistoryDB:
    def __init__(self, db_file: str = SQLITE_DB_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(CHAT_HISTORY_DDL)
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_session_id ON {TABLE_CHAT_HISTORY}(session_id)")
            conn.commit()
    
    def insert_chat_entry(self, entry: Dict[str, Any]) -> int:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {TABLE_CHAT_HISTORY} (
                    session_id, timestamp, llm_provider, model_name, parsing_mode,
                    user_query, parsed_action, tool_name, resource_uri, parameters,
                    confidence, reasoning, rag_matches, similarity_scores, response_data, 
                    formatted_response, elapsed_time_ms, error_message, success, is_batch,
                    operation_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.get('session_id'), entry.get('timestamp'), entry.get('llm_provider'),
                entry.get('model_name'), entry.get('parsing_mode'), entry.get('user_query'),
                entry.get('parsed_action'), entry.get('tool_name'), entry.get('resource_uri'),
                entry.get('parameters'), entry.get('confidence'), entry.get('reasoning'),
                entry.get('rag_matches'), entry.get('similarity_scores'), entry.get('response_data'),
                entry.get('formatted_response'), entry.get('elapsed_time_ms'), 
                entry.get('error_message'), entry.get('success', True), entry.get('is_batch', False),
                entry.get('operation_count', 1)
            ))
            entry_id = cursor.lastrowid
            conn.commit()
            return entry_id
    
    def get_chat_history(self, limit: int = 100, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            query = f"SELECT * FROM {TABLE_CHAT_HISTORY}"
            params = []
            
            if filters and filters.get('session_id'):
                query += " WHERE session_id = ?"
                params.append(filters['session_id'])
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

# ============================================================================
# ENHANCED RAG SYSTEM
# ============================================================================

class EnhancedMCPRAGSystem:
    """Enhanced RAG system that includes custom tools"""
    
    def __init__(self, custom_registry: DynamicToolRegistry):
        self.custom_registry = custom_registry
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
            logging.info("✅ Enhanced RAG System initialized with all-MiniLM-L6-v2")
        except Exception as e:
            logging.error(f"❌ Failed to initialize RAG model: {e}")
            self.model = None
    
    def build_embeddings(self, tools: List[Dict], resources: List[Dict]):
        """Build embeddings including custom tools"""
        if not self.model:
            return
        
        try:
            # Add custom tools to the standard tools
            custom_tools = []
            for tool_def in self.custom_registry.list_tools():
                custom_tools.append({
                    'name': tool_def.name,
                    'description': tool_def.description,
                    'category': tool_def.category,
                    'tags': tool_def.tags,
                    'examples': tool_def.examples,
                    'is_custom': True
                })
            
            # Combine standard and custom tools
            all_tools = tools + custom_tools
            
            # Build contexts
            self.tool_contexts = []
            for tool in all_tools:
                try:
                    context = self.build_rich_context(tool, 'tool')
                    self.tool_contexts.append({
                        'name': tool.get('name', ''),
                        'description': tool.get('description', ''),
                        'context': context,
                        'type': 'tool',
                        'is_custom': tool.get('is_custom', False)
                    })
                except Exception as e:
                    logging.warning(f"Failed to build context for tool {tool.get('name', 'unknown')}: {e}")
            
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
                    logging.warning(f"Failed to build context for resource: {e}")
            
            # Create embeddings
            if self.tool_contexts:
                try:
                    tool_texts = [item['context'] for item in self.tool_contexts]
                    self.tool_embeddings = self.model.encode(tool_texts)
                    logging.info(f"✅ Built embeddings for {len(self.tool_contexts)} tools (including {len(custom_tools)} custom)")
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
            logging.error(f"❌ Failed to build enhanced embeddings: {e}")
    
    def build_rich_context(self, item: Dict, item_type: str) -> str:
        """Enhanced context building for custom tools"""
        if item_type == "tool":
            name = item.get('name', '')
            desc = item.get('description', '')
            is_custom = item.get('is_custom', False)
            
            if is_custom:
                # Custom tool context
                category = item.get('category', '')
                tags = item.get('tags', [])
                examples = item.get('examples', [])
                
                context = f"""
Tool: {name}
Description: {desc}
Category: {category}
Type: custom tool
Usage examples: {' | '.join(examples) if examples else 'No examples provided'}
Keywords: {name}, {category}, {', '.join(tags)}
Tags: {', '.join(tags)}
                """.strip()
                return context
            else:
                # Standard tool context
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
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search across tools and resources"""
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
            
            corpus_embeddings = np.concatenate(all_embeddings, axis=0)
            
            search_results = st_util.semantic_search(
                query_embedding, 
                corpus_embeddings, 
                top_k=top_k,
                score_function=st_util.cos_sim
            )[0]
            
            results = []
            for hit in search_results:
                corpus_id = hit['corpus_id']
                similarity = float(hit['score'])
                
                if similarity > 0.1:
                    context_item, item_type = all_contexts[corpus_id]
                    results.append({
                        'item': context_item,
                        'similarity': similarity,
                        'type': item_type
                    })
            
            logging.info(f"✅ Enhanced semantic search found {len(results)} relevant items")
            return results
            
        except Exception as e:
            logging.error(f"❌ Enhanced semantic search failed: {e}")
            return []
    
    def build_dynamic_prompt(self, relevant_items: List[Dict], query: str) -> str:
        """Build dynamic system prompt based on relevant items"""
        if not relevant_items:
            return """
You are a tool selection assistant. 
Respond with ONLY a JSON object with action, tool, params, confidence, and reasoning fields.
"""
        
        tools_section = "Available tools:\n"
        for item in relevant_items:
            if item['type'] == 'tool':
                tool_info = item['item']
                similarity = item['similarity']
                is_custom = tool_info.get('is_custom', False)
                tool_type = "custom" if is_custom else "standard"
                tools_section += f"- {tool_info['name']} ({tool_type}): {tool_info['description']} (relevance: {similarity:.2f})\n"
        
        resources_section = "\nAvailable resources:\n"
        for item in relevant_items:
            if item['type'] == 'resource':
                resource_info = item['item']
                similarity = item['similarity']
                resources_section += f"- {resource_info['uri']}: {resource_info['description']} (relevance: {similarity:.2f})\n"
        
        examples_section = "\nExamples based on available tools:\n"
        
        for item in relevant_items[:3]:
            if item['type'] == 'tool':
                tool_name = item['item']['name']
                if tool_name == 'calculator':
                    examples_section += '- "15 plus 27" -> {"action": "tool", "tool": "calculator", "params": {"operation": "add", "num1": 15, "num2": 27}, "confidence": 0.98, "reasoning": "Simple addition"}\n'
                elif tool_name == 'trig':
                    examples_section += '- "sine of 30 degrees" -> {"action": "tool", "tool": "trig", "params": {"operation": "sine", "num1": 30, "unit": "degree"}, "confidence": 0.95, "reasoning": "Trigonometric calculation"}\n'
                elif tool_name == 'unit_converter':
                    examples_section += '- "convert 5 feet to meters" -> {"action": "tool", "tool": "unit_converter", "params": {"value": 5, "from_unit": "ft", "to_unit": "m"}, "confidence": 0.95, "reasoning": "Unit conversion"}\n'
                elif tool_name == 'text_analyzer':
                    examples_section += '- "analyze this text: Hello world" -> {"action": "tool", "tool": "text_analyzer", "params": {"text": "Hello world"}, "confidence": 0.9, "reasoning": "Text analysis"}\n'
        
        system_prompt = f"""
You are an intelligent tool selection assistant. 
Analyze the user query and respond with ONLY a JSON object:

{{
    "action": "tool",
    "tool": "tool_name_or_null",
    "params": {{"param1": "value1"}},
    "confidence": 0.95,
    "reasoning": "Brief explanation"
}}

{tools_section}

{resources_section}

{examples_section}

Instructions:
- Only use tools/resources listed above
- Consider the relevance scores when making decisions
- Set confidence based on query clarity and tool match
- If no tool matches well (all relevance < 0.3), set tool to null
- Respond with ONLY the JSON object, no other text.
"""

        return system_prompt

# ============================================================================
# LLM INTEGRATION
# ============================================================================

def ask_llm(provider, client, model_name, query, system_prompt, max_tokens=300, temperature=0.1):
    """Get LLM response with dynamic prompt"""
    if provider == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Query: {query}"}]
        )
        llm_response = response.content[0].text.strip()
    
    elif provider == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        llm_response = response.choices[0].message.content.strip()
    
    elif provider == "google":
        response = client.generate_content(
            f"{system_prompt}\n\nUser Query: {query}",
            generation_config={
                "temperature": temperature, 
                "max_output_tokens": max_tokens,
            }
        )
        llm_response = response.text.strip()
    else:
        st.error(f"Unsupported LLM provider: {provider}")
        return None
    
    # Clean and parse JSON
    if llm_response.startswith("```json"):
        llm_response = llm_response.replace("```json", "").replace("```", "").strip()
    elif llm_response.startswith("```"):
        llm_response = llm_response.replace("```", "").strip()
    
    parsed_response = json.loads(llm_response)
    return parsed_response

class LLMQueryParser:
    def __init__(self, provider: str = "google"):
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
                    self.model_name = st.session_state.llm_model_name
            
            elif self.provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.model_name = st.session_state.llm_model_name
            
            elif self.provider == "google":
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model_name = st.session_state.llm_model_name
                    self.client = genai.GenerativeModel(self.model_name)
                
        except Exception as e:
            st.error(f"Failed to initialize {self.provider}: {e}")
            self.client = None
    
    def parse_query_with_rag(self, query: str, rag_system: EnhancedMCPRAGSystem) -> Optional[Dict[str, Any]]:
        """Parse query using RAG-enhanced semantic search"""
        if not self.client or not rag_system.model:
            return None
        
        try:
            relevant_items = rag_system.semantic_search(query, top_k=5)
            st.session_state.last_rag_matches = relevant_items
            
            if not relevant_items:
                return self.parse_query_sync(query)
            
            system_prompt = rag_system.build_dynamic_prompt(relevant_items, query)
            
            parsed_response = ask_llm(self.provider, self.client, self.model_name, query, system_prompt)
            if not parsed_response:
                return None

            parsed_response['rag_enhanced'] = True
            parsed_response['rag_matches'] = len(relevant_items)
            
            if parsed_response.get("action") and parsed_response.get("confidence", 0) >= 0.3:
                return parsed_response
            
        except Exception as e:
            logging.error(f"RAG-enhanced parsing error: {e}")
            return self.parse_query_sync(query)
        
        return None
    
    def parse_query_sync(self, query: str) -> Optional[Dict[str, Any]]:
        """Legacy parsing method with hardcoded examples"""
        if not self.client:
            return None
        
        system_prompt = """
You are a tool selection assistant. Respond with ONLY a JSON object:

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

Respond with ONLY the JSON object.
"""
        
        try:
            parsed_response = ask_llm(self.provider, self.client, self.model_name, query, system_prompt)
            if not parsed_response:
                return None

            parsed_response['rag_enhanced'] = False
            
            if parsed_response.get("action") and parsed_response.get("confidence", 0) >= 0.5:
                return parsed_response
            
        except Exception as e:
            st.error(f"LLM parsing error: {e}")
        
        return None

class EnhancedQueryParser:
    """Enhanced parser that can handle batch operations and custom tools"""
    
    def __init__(self, registry: DynamicToolRegistry, llm_parser: LLMQueryParser):
        self.registry = registry
        self.llm_parser = llm_parser
    
    def parse_batch_query(self, query: str) -> Optional[BatchRequest]:
        """Parse queries that request batch operations"""
        query_lower = query.lower()
        
        batch_indicators = [
            "and then", "then", "after that", "followed by",
            "also calculate", "also find", "batch", "multiple",
            "calculate all", "find all"
        ]
        
        if not any(indicator in query_lower for indicator in batch_indicators):
            return None
        
        try:
            custom_tools_list = [tool.name for tool in self.registry.list_tools()]
            
            system_prompt = f"""
You are a batch operation parser. Parse the user query into multiple operations.
Return a JSON object with this structure:

{{
    "is_batch": true,
    "parallel": false,
    "operations": [
        {{
            "id": "op1",
            "tool": "calculator",
            "params": {{"operation": "add", "num1": 15, "num2": 27}},
            "variable_name": "sum_result"
        }},
        {{
            "id": "op2", 
            "tool": "trig",
            "params": {{"operation": "sine", "num1": "${{sum_result}}", "unit": "degree"}},
            "depends_on": ["op1"]
        }}
    ]
}}

Available standard tools: calculator, trig, health, echo
Available custom tools: {custom_tools_list}

Rules:
- Each operation needs a unique id
- Use variable_name to store results for later reference
- Use ${{variable_name}} to reference previous results
- Use depends_on to specify operation dependencies
- Set parallel=true only if operations can run simultaneously
"""

            parsed_response = ask_llm(
                self.llm_parser.provider, 
                self.llm_parser.client,
                self.llm_parser.model_name,
                query,
                system_prompt,
                max_tokens=800
            )
            
            if parsed_response and parsed_response.get("is_batch"):
                operations = [
                    BatchOperation(**op_data) 
                    for op_data in parsed_response["operations"]
                ]
                
                return BatchRequest(
                    operations=operations,
                    parallel=parsed_response.get("parallel", False)
                )
                
        except Exception as e:
            logging.error(f"Batch parsing error: {e}")
        
        return None

# ============================================================================
# RULE-BASED PARSER
# ============================================================================

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

# ============================================================================
# SAMPLE CUSTOM TOOLS
# ============================================================================

def create_sample_custom_tools(registry: DynamicToolRegistry):
    """Create some sample custom tools for demonstration"""
    
    # 1. Unit Converter Tool
    unit_converter = CustomToolDefinition(
        name="unit_converter",
        description="Convert between different units of measurement",
        category="conversion",
        parameters=[
            ToolParameter("value", "number", "Value to convert", required=True),
            ToolParameter("from_unit", "string", "Source unit", required=True,
                         enum=["m", "ft", "in", "cm", "mm", "km", "miles"]),
            ToolParameter("to_unit", "string", "Target unit", required=True,
                         enum=["m", "ft", "in", "cm", "mm", "km", "miles"]),
        ],
        endpoint="FUNCTION",
        examples=[
            "convert 5 feet to meters",
            "convert 100 km to miles", 
            "convert 12 inches to cm"
        ],
        tags=["conversion", "units", "measurement"],
        author="System"
    )
    
    def unit_converter_func(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        to_meters = {
            "m": 1, "ft": 0.3048, "in": 0.0254, "cm": 0.01, 
            "mm": 0.001, "km": 1000, "miles": 1609.34
        }
        
        meters = value * to_meters[from_unit]
        result = meters / to_meters[to_unit]
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": round(result, 6),
            "converted_unit": to_unit,
            "expression": f"{value} {from_unit} = {round(result, 6)} {to_unit}"
        }
    
    registry.register_tool(unit_converter)
    registry.register_local_function("unit_converter", unit_converter_func)
    
    # 2. Text Analysis Tool
    text_analyzer = CustomToolDefinition(
        name="text_analyzer",
        description="Analyze text for word count, character count, and basic statistics",
        category="text",
        parameters=[
            ToolParameter("text", "string", "Text to analyze", required=True),
            ToolParameter("include_spaces", "boolean", "Include spaces in character count", default=True),
        ],
        endpoint="FUNCTION",
        examples=[
            "analyze this text: Hello world",
            "count words in: The quick brown fox",
            "text stats for: Lorem ipsum dolor sit amet"
        ],
        tags=["text", "analysis", "statistics", "nlp"],
        author="System"
    )
    
    def text_analyzer_func(text: str, include_spaces: bool = True) -> Dict[str, Any]:
        words = text.split()
        chars = len(text) if include_spaces else len(text.replace(" ", ""))
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        return {
            "text": text,
            "word_count": len(words),
            "character_count": chars,
            "sentence_count": max(1, sentences),
            "average_word_length": round(sum(len(word) for word in words) / len(words), 2) if words else 0,
            "includes_spaces": include_spaces
        }
    
    registry.register_tool(text_analyzer)
    registry.register_local_function("text_analyzer", text_analyzer_func)
    
    logging.info("✅ Created sample custom tools: unit_converter, text_analyzer")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
        return f"❌ [Error] {result['error']}"
    
    if tool_name == "calculator":
        expression = result.get('expression', f"{result.get('num1', '?')} {result.get('operation', '?')} {result.get('num2', '?')} = {result.get('result', '?')}")
        return f"🧮 [Calculator] {expression}"
    
    elif tool_name == "trig":
        expression = result.get('expression', f"{result.get('operation', '?')}({result.get('num1', '?')}) = {result.get('result', '?')}")
        return f"📐 [Trigonometry] {expression}"
    
    elif tool_name == "health":
        return f"✅ [Health] {result.get('message', 'Server is healthy')}"
    
    elif tool_name == "echo":
        return f"🔊 [Echo] {result.get('echo', result.get('message', str(result)))}"
    
    return f"✅ [Result] {json.dumps(result, indent=2)}"

def enhanced_format_result_for_display(result: Dict) -> str:
    """Enhanced result formatting for custom tools and batch operations"""
    
    if result.get("type") == "custom_tool":
        tool_name = result.get("name", "Unknown")
        category = result.get("category", "custom")
        data = result.get("data", {})
        
        if isinstance(data, dict) and "error" in data:
            return f"❌ [Custom Tool Error] {data['error']}"
        
        if category == "conversion":
            if "expression" in data:
                return f"🔄 [Converter] {data['expression']}"
            else:
                return f"🔄 [Converter] {tool_name}: {json.dumps(data, indent=2)}"
        
        elif category == "text":
            if "word_count" in data:
                return f"📝 [Text Analysis] Words: {data['word_count']}, Characters: {data['character_count']}, Sentences: {data['sentence_count']}"
            else:
                return f"📝 [Text Tool] {tool_name}: {json.dumps(data, indent=2)}"
        
        else:
            return f"🔧 [Custom {category.title()}] {tool_name}: {json.dumps(data, indent=2)}"
    
    elif result.get("type") == "tool":
        tool_name = result.get("name")
        data = result.get("data", {})
        
        if "operation_id" in result:
            op_id = result["operation_id"]
            exec_time = result.get("execution_time_ms", 0)
            deps = result.get("dependencies", [])
            
            formatted = format_result_for_display(tool_name, data)
            batch_info = f" [Batch: {op_id}, {exec_time}ms"
            if deps:
                batch_info += f", deps: {', '.join(deps)}"
            batch_info += "]"
            
            return formatted + batch_info
        else:
            return format_result_for_display(tool_name, data)
    
    elif result.get("type") == "error":
        op_id = result.get("operation_id")
        if op_id:
            return f"❌ [Batch Error - {op_id}] {result.get('message', 'Unknown error')}"
        else:
            return f"❌ [Error] {result.get('message', 'Unknown error')}"
    
    else:
        return f"ℹ️ [Result] {json.dumps(result, indent=2)}"

# ============================================================================
# CACHED MCP OPERATIONS
# ============================================================================

@st.cache_resource  
def get_mcp_server_info():
    """Get cached server info (tools/resources) - cached across reruns"""
    async def _discover():
        async with Client(MCP_SERVER_PATH) as client:
            tools = await client.list_tools()
            available_tools = [{"name": tool.name, "description": tool.description} for tool in tools] if tools else []
            
            try:
                resources = await client.list_resources()
                available_resources = [{"uri": resource.uri, "description": resource.description} for resource in resources] if resources else []
            except:
                available_resources = []
            
            return available_tools, available_resources
    
    return asyncio.run(_discover())

# ============================================================================
# ENHANCED QUERY EXECUTION
# ============================================================================

async def enhanced_execute_query(user_query: str) -> Tuple[List[Dict], int, bool]:
    """Enhanced query execution with batch and custom tool support"""
    start_time = time.time()
    
    try:
        # Check if it's a batch operation
        batch_request = st.session_state.enhanced_parser.parse_batch_query(user_query)
        
        if batch_request:
            # Execute batch operations
            batch_results = await st.session_state.batch_processor.execute_batch(batch_request)
            
            results = []
            is_batch = True
            
            for batch_result in batch_results:
                if batch_result.success:
                    results.append({
                        "type": "tool",
                        "name": batch_result.tool,
                        "data": batch_result.result,
                        "success": True,
                        "operation_id": batch_result.operation_id,
                        "execution_time_ms": batch_result.execution_time_ms,
                        "dependencies": batch_result.dependencies_resolved
                    })
                else:
                    results.append({
                        "type": "error", 
                        "message": f"Operation {batch_result.operation_id} failed: {batch_result.error}",
                        "success": False,
                        "operation_id": batch_result.operation_id
                    })
            
            elapsed_time = int((time.time() - start_time) * 1000)
            return results, elapsed_time, is_batch
        
        else:
            # Single operation
            parsed_query = None
            
            if st.session_state.use_llm and st.session_state.use_rag and RAG_AVAILABLE:
                tools, resources = get_mcp_server_info()
                st.session_state.enhanced_rag_system.build_embeddings(tools, resources)
                
                parser = LLMQueryParser(st.session_state.llm_provider)
                if parser.client:
                    parsed_query = parser.parse_query_with_rag(user_query, st.session_state.enhanced_rag_system)
            
            if not parsed_query:
                if st.session_state.use_llm:
                    parser = LLMQueryParser(st.session_state.llm_provider)
                    if parser.client:
                        parsed_query = parser.parse_query_sync(user_query)
                    else:
                        parsed_query = RuleBasedQueryParser.parse_query(user_query)
                else:
                    parsed_query = RuleBasedQueryParser.parse_query(user_query)
            
            if not parsed_query:
                elapsed_time = int((time.time() - start_time) * 1000)
                return [], elapsed_time, False
            
            # Store for debug display
            st.session_state.last_parsed_query = parsed_query
            
            # Execute single operation
            results = []
            tool_name = parsed_query.get("tool")
            parameters = parsed_query.get("params", {})
            
            if tool_name:
                custom_tool = st.session_state.custom_registry.get_tool(tool_name)
                
                if custom_tool:
                    # Execute custom tool
                    try:
                        if custom_tool.method == "FUNCTION":
                            func = st.session_state.custom_registry.local_functions.get(tool_name)
                            if func:
                                if asyncio.iscoroutinefunction(func):
                                    tool_result = await func(**parameters)
                                else:
                                    tool_result = func(**parameters)
                                
                                results.append({
                                    "type": "custom_tool",
                                    "name": tool_name,
                                    "data": tool_result,
                                    "success": True,
                                    "category": custom_tool.category
                                })
                            else:
                                results.append({
                                    "type": "error",
                                    "message": f"Custom tool function not found: {tool_name}",
                                    "success": False
                                })
                        else:
                            # HTTP endpoint execution
                            import aiohttp
                            async with aiohttp.ClientSession() as session:
                                async with session.request(
                                    custom_tool.method,
                                    custom_tool.endpoint,
                                    json=parameters
                                ) as response:
                                    if response.status == 200:
                                        tool_result = await response.json()
                                        results.append({
                                            "type": "custom_tool",
                                            "name": tool_name,
                                            "data": tool_result,
                                            "success": True,
                                            "category": custom_tool.category
                                        })
                                    else:
                                        error_text = await response.text()
                                        results.append({
                                            "type": "error",
                                            "message": f"HTTP {response.status}: {error_text}",
                                            "success": False
                                        })
                    
                    except Exception as e:
                        results.append({
                            "type": "error",
                            "message": f"Custom tool execution error: {e}",
                            "success": False
                        })
                
                else:
                    # Execute standard MCP tool
                    try:
                        async with Client(MCP_SERVER_PATH) as client:
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
                            "message": f"MCP tool error: {e}",
                            "success": False
                        })
            
            elapsed_time = int((time.time() - start_time) * 1000)
            return results, elapsed_time, False
    
    except Exception as e:
        elapsed_time = int((time.time() - start_time) * 1000)
        return [{
            "type": "error",
            "message": f"Query execution error: {e}",
            "success": False
        }], elapsed_time, False

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    if 'chat_history_db' not in st.session_state:
        st.session_state.chat_history_db = ChatHistoryDB()
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(f"{datetime.now()}{os.getpid()}".encode()).hexdigest()[:8]
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "google"
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
    if 'last_rag_matches' not in st.session_state:
        st.session_state.last_rag_matches = []
    if 'show_tool_management' not in st.session_state:
        st.session_state.show_tool_management = False
    
    # Enhanced session state
    if 'custom_registry' not in st.session_state:
        st.session_state.custom_registry = DynamicToolRegistry()
        create_sample_custom_tools(st.session_state.custom_registry)
    
    if 'batch_processor' not in st.session_state:
        st.session_state.batch_processor = BatchProcessor(
            custom_registry=st.session_state.custom_registry
        )
    
    if 'enhanced_rag_system' not in st.session_state:
        st.session_state.enhanced_rag_system = EnhancedMCPRAGSystem(
            custom_registry=st.session_state.custom_registry
        )
    
    if 'enhanced_parser' not in st.session_state:
        llm_parser = LLMQueryParser(st.session_state.llm_provider)
        st.session_state.enhanced_parser = EnhancedQueryParser(
            registry=st.session_state.custom_registry,
            llm_parser=llm_parser
        )

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_tool_management_ui(registry: DynamicToolRegistry):
    """Render UI for managing custom tools"""
    
    st.subheader("🔧 Tool Management")
    
    tab1, tab2, tab3 = st.tabs(["📋 View Tools", "➕ Add Tool", "🗑️ Remove Tool"])
    
    with tab1:
        custom_tools = registry.list_tools()
        if custom_tools:
            st.write(f"**{len(custom_tools)} Custom Tools Registered:**")
            
            for tool in custom_tools:
                with st.expander(f"🔧 {tool.name} ({tool.category})"):
                    st.write(f"**Description:** {tool.description}")
                    st.write(f"**Author:** {tool.author or 'Unknown'}")
                    st.write(f"**Version:** {tool.version}")
                    st.write(f"**Created:** {tool.created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    if tool.tags:
                        st.write(f"**Tags:** {', '.join(tool.tags)}")
                    
                    if tool.examples:
                        st.write("**Examples:**")
                        for example in tool.examples:
                            st.write(f"  • {example}")
                    
                    st.write("**Parameters:**")
                    for param in tool.parameters:
                        required_text = "✅ Required" if param.required else "⭕ Optional"
                        st.write(f"  • `{param.name}` ({param.type}): {param.description} - {required_text}")
        else:
            st.info("No custom tools registered yet.")
    
    with tab2:
        st.write("**Create a New Custom Tool**")
        
        with st.form("add_tool_form"):
            name = st.text_input("Tool Name", help="Unique identifier for the tool")
            description = st.text_area("Description", help="What does this tool do?")
            category = st.selectbox("Category", 
                                   ["calculation", "conversion", "text", "data", "utility", "other"])
            author = st.text_input("Author", value="User")
            
            st.write("**Parameters:**")
            param_count = st.number_input("Number of Parameters", min_value=0, max_value=10, value=1)
            
            parameters = []
            for i in range(int(param_count)):
                st.write(f"Parameter {i+1}:")
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    param_name = st.text_input(f"Name {i+1}", key=f"param_name_{i}")
                    param_desc = st.text_input(f"Description {i+1}", key=f"param_desc_{i}")
                
                with col2:
                    param_type = st.selectbox(f"Type {i+1}", 
                                            ["string", "number", "boolean"], key=f"param_type_{i}")
                
                with col3:
                    param_required = st.checkbox(f"Required {i+1}", value=True, key=f"param_req_{i}")
                
                if param_name and param_desc:
                    parameters.append(ToolParameter(
                        name=param_name,
                        type=param_type,
                        description=param_desc,
                        required=param_required
                    ))
            
            examples_text = st.text_area("Examples (one per line)", 
                                       help="Provide example queries that would use this tool")
            tags_text = st.text_input("Tags (comma-separated)", 
                                     help="Keywords to help find this tool")
            
            endpoint = st.text_input("Endpoint", value="FUNCTION", 
                                   help="Use 'FUNCTION' for local Python functions, or provide HTTP URL")
            
            submitted = st.form_submit_button("Register Tool")
            
            if submitted and name and description:
                examples = [ex.strip() for ex in examples_text.split('\n') if ex.strip()]
                tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
                
                tool_def = CustomToolDefinition(
                    name=name,
                    description=description,
                    category=category,
                    parameters=parameters,
                    endpoint=endpoint,
                    examples=examples,
                    tags=tags,
                    author=author
                )
                
                if registry.register_tool(tool_def):
                    st.success(f"✅ Tool '{name}' registered successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to register tool. Check the logs for details.")
    
    with tab3:
        custom_tools = registry.list_tools()
        if custom_tools:
            tool_names = [tool.name for tool in custom_tools]
            selected_tool = st.selectbox("Select tool to remove:", tool_names)
            
            if st.button("🗑️ Remove Tool", type="secondary"):
                if registry.unregister_tool(selected_tool):
                    st.success(f"✅ Tool '{selected_tool}' removed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to remove tool.")
        else:
            st.info("No custom tools to remove.")

def render_batch_operations_ui():
    """Render UI for batch operations"""
    
    st.subheader("🔄 Batch Operations")
    
    with st.expander("ℹ️ Batch Operations Help"):
        st.markdown("""
        **Batch operations** allow you to chain multiple tool calls together:
        
        **Example queries:**
        - "Calculate 15 + 27, then find the sine of that result"
        - "Convert 5 feet to meters, then multiply by 2"
        - "Analyze this text: 'Hello world', then echo the word count"
        
        **Features:**
        - **Sequential execution** with dependency resolution
        - **Variable references** using ${variable_name}
        - **Parallel execution** when operations are independent
        - **Error handling** with fail-fast or continue options
        """)
    
    st.write("**Try these batch operation examples:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Math Chain Example"):
            st.session_state.example_query = "Calculate 15 plus 27, then find sine of that result in degrees"
    
    with col2:
        if st.button("🔄 Conversion Chain Example"):
            st.session_state.example_query = "Convert 5 feet to meters, then multiply that by 3.14"

def enhanced_sidebar():
    """Enhanced sidebar with tool management features"""
    
    with st.sidebar:
        st.header("⚙️ Enhanced Configuration")
        st.info(f"📍 [Session ID] `{st.session_state.session_id}`")
        
        # RAG System Status
        st.subheader("🧠 RAG System")
        if RAG_AVAILABLE and st.session_state.enhanced_rag_system.model:
            st.success("✅ RAG System Active")
            st.info("🔍 Semantic search enabled")
            
            st.session_state.use_rag = st.checkbox(
                "🎯 Use RAG-Enhanced Parsing",
                value=st.session_state.use_rag,
                help="Use semantic search to find relevant tools dynamically"
            )
        else:
            st.error("❌ RAG System Disabled")
            st.warning("Install: `pip install sentence-transformers scikit-learn`")
            st.session_state.use_rag = False
        
        # LLM Provider/Model Selection
        c1, c2 = st.columns([2,3])
        with c1:
            LLM_PROVIDER_LIST = list(LLM_PROVIDER_MAP.keys())
            st.session_state.llm_provider = st.selectbox(
                "🤖 LLM Provider",
                LLM_PROVIDER_LIST,
                index=LLM_PROVIDER_LIST.index("google")
            )

        with c2:
            LLM_MODEL_NAME_LIST = LLM_PROVIDER_MAP.get(st.session_state.llm_provider)
            st.session_state.llm_model_name = st.selectbox(
                "Model Name",
                LLM_MODEL_NAME_LIST,
                index=0
            )

        # Parsing Mode
        st.session_state.use_llm = st.checkbox(
            "🧠 Use LLM Parsing",
            value=st.session_state.use_llm
        )
        
        # Parsing Mode Display
        if st.session_state.use_llm and st.session_state.use_rag and RAG_AVAILABLE:
            st.info("🎯 [Mode] RAG-enhanced LLM")
        elif st.session_state.use_llm:
            st.info("🤖 [Mode] LLM-based")
        else:
            st.info("📝 [Mode] Rule-based")
        
        # API Keys Status
        st.subheader("🔑 API Keys Status")
        api_keys_status = {
            "OpenAI": "✅" if os.getenv("OPENAI_API_KEY") else "❌",
            "Anthropic": "✅" if os.getenv("ANTHROPIC_API_KEY") else "❌",
            "Google": "✅" if os.getenv("GEMINI_API_KEY") else "❌",
        }
        
        for provider, status in api_keys_status.items():
            st.write(f"{status} {provider}")
        
        # Server Connection
        st.subheader("🔌 Server Status")
        
        try:
            tools, resources = get_mcp_server_info()
            
            st.session_state.server_connected = True
            st.session_state.available_tools = tools
            st.session_state.available_resources = resources
            
            if RAG_AVAILABLE and st.session_state.enhanced_rag_system.model:
                st.session_state.enhanced_rag_system.build_embeddings(tools, resources)
            
        except Exception as e:
            st.session_state.server_connected = False
            st.session_state.available_tools = []
            st.session_state.available_resources = []
        
        if st.button("🔄 Refresh Server Discovery"):
            st.cache_resource.clear()
            st.rerun()
        
        if st.session_state.server_connected:
            st.success("🟢 Server Connected")
            
            if st.session_state.available_tools:
                with st.expander("🔧 Available Tools"):
                    for tool in st.session_state.available_tools:
                        st.write(f"• [{tool['name']}] {tool['description']}")
            
            if st.session_state.available_resources:
                with st.expander("📚 Available Resources"):
                    for resource in st.session_state.available_resources:
                        st.write(f"• [{resource['uri']}] {resource['description']}")
            
            if RAG_AVAILABLE and st.session_state.enhanced_rag_system.model:
                tool_count = len(st.session_state.enhanced_rag_system.tool_contexts)
                resource_count = len(st.session_state.enhanced_rag_system.resource_contexts)
                st.info(f"🎯 RAG: {tool_count} tools, {resource_count} resources indexed")
        else:
            st.error("🔴 Server Disconnected")
            st.info(f"💡 Make sure {MCP_SERVER_PATH} is running")
        
        st.divider()
        
        # Custom Tools Section
        st.subheader("🔧 Custom Tools")
        custom_tools = st.session_state.custom_registry.list_tools()
        
        if custom_tools:
            st.success(f"✅ {len(custom_tools)} custom tools registered")
            
            categories = {}
            for tool in custom_tools:
                categories[tool.category] = categories.get(tool.category, 0) + 1
            
            for category, count in categories.items():
                st.write(f"  • {category.title()}: {count} tools")
            
            if len(custom_tools) > 3:
                search_query = st.text_input("🔍 Search tools:", key="tool_search")
                if search_query:
                    found_tools = st.session_state.custom_registry.search_tools(search_query)
                    st.write(f"Found {len(found_tools)} tools:")
                    for tool in found_tools[:3]:
                        st.write(f"  • {tool.name} ({tool.category})")
        else:
            st.info("No custom tools yet")
        
        if st.button("🔧 Manage Tools"):
            st.session_state.show_tool_management = True
        
        # Batch Operations Section
        st.subheader("🔄 Batch Operations")
        
        if hasattr(st.session_state, 'last_batch_info'):
            batch_info = st.session_state.last_batch_info
            st.info(f"Last batch: {batch_info.get('operation_count', 0)} operations in {batch_info.get('total_time_ms', 0)}ms")
        
        st.write("**Quick Examples:**")
        if st.button("📊 Math Chain", key="batch_math"):
            st.session_state.example_query = "Calculate 15 + 27, then find sine of that result"
        
        if st.button("🔄 Convert Chain", key="batch_convert"):
            st.session_state.example_query = "Convert 5 feet to meters, then multiply by 2"
        
        # Performance metrics
        st.divider()
        st.subheader("📈 Performance")
        
        try:
            recent_entries = st.session_state.chat_history_db.get_chat_history(limit=20)
            if recent_entries:
                tool_usage = {}
                for entry in recent_entries:
                    tool = entry.get('tool_name')
                    if tool:
                        tool_usage[tool] = tool_usage.get(tool, 0) + 1
                
                if tool_usage:
                    st.write("**Tool Usage (Last 20):**")
                    for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"  • {tool}: {count}x")
        except:
            pass

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    init_session_state()
    
    # Custom CSS
    st.markdown(CUSTOM_CSS_STYLE, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Enhanced MCP Client with RAG</h1>', unsafe_allow_html=True)
    
    # Check if we should show tool management
    if st.session_state.get('show_tool_management', False):
        render_tool_management_ui(st.session_state.custom_registry)
        render_batch_operations_ui()
        
        if st.button("◀️ Back to Main"):
            st.session_state.show_tool_management = False
            st.rerun()
        
        enhanced_sidebar()
        return
    
    # Sidebar Configuration
    enhanced_sidebar()

    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Query Interface")
        
        # Query Input
        default_query = st.session_state.get('example_query', '')
        user_query = st.text_input(
            "🎯 Enter your query:",
            value=default_query,
            placeholder="convert 5 feet to meters, then multiply by 2"
        )
        
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
        
        # Process Query
        if submit_button and user_query:
            try:
                # Execute enhanced query
                results, elapsed_time, is_batch = asyncio.run(enhanced_execute_query(user_query))
                st.session_state.elapsed_time = elapsed_time
                
                # Store batch info for sidebar display
                if is_batch:
                    st.session_state.last_batch_info = {
                        'operation_count': len(results),
                        'total_time_ms': elapsed_time,
                        'timestamp': datetime.now()
                    }
                
                # Auto-update connection status if successful
                if results and any(r.get('success', True) for r in results):
                    st.session_state.server_connected = True
                
                # Save to database
                parsing_mode = 'RAG-Enhanced' if st.session_state.last_parsed_query and st.session_state.last_parsed_query.get('rag_enhanced') else ('LLM' if st.session_state.use_llm else 'Rule-based')
                
                db_entry = {
                    'session_id': st.session_state.session_id,
                    'timestamp': datetime.now(),
                    'llm_provider': st.session_state.llm_provider if st.session_state.use_llm else None,
                    'model_name': getattr(st.session_state, 'llm_model_name', None),
                    'parsing_mode': parsing_mode,
                    'user_query': user_query,
                    'parsed_action': st.session_state.last_parsed_query.get('action') if st.session_state.last_parsed_query else None,
                    'tool_name': st.session_state.last_parsed_query.get('tool') if st.session_state.last_parsed_query else None,
                    'parameters': json.dumps(st.session_state.last_parsed_query.get('params', {})) if st.session_state.last_parsed_query else None,
                    'confidence': st.session_state.last_parsed_query.get('confidence') if st.session_state.last_parsed_query else None,
                    'reasoning': st.session_state.last_parsed_query.get('reasoning') if st.session_state.last_parsed_query else None,
                    'rag_matches': json.dumps([{
                        'name': m['item'].get('name', m['item'].get('uri', '')),
                        'similarity': m['similarity'],
                        'type': m['type']
                    } for m in st.session_state.last_rag_matches]) if st.session_state.last_rag_matches else None,
                    'similarity_scores': json.dumps([m['similarity'] for m in st.session_state.last_rag_matches]) if st.session_state.last_rag_matches else None,
                    'elapsed_time_ms': elapsed_time,
                    'success': all(result.get('success', True) for result in results),
                    'is_batch': is_batch,
                    'operation_count': len(results) if is_batch else 1
                }
                
                entry_id = st.session_state.chat_history_db.insert_chat_entry(db_entry)
                st.session_state.entry_id = entry_id

                # Show RAG matches if available
                if st.session_state.last_rag_matches:
                    st.markdown("### 🎯 RAG Search Results:")
                    for i, match in enumerate(st.session_state.last_rag_matches[:3]):
                        item = match['item']
                        similarity = match['similarity']
                        item_type = match['type']
                        
                        if item_type == 'tool':
                            name = item.get('name', 'Unknown')
                            desc = item.get('description', '')
                            is_custom = item.get('is_custom', False)
                            tool_badge = "🔧 Custom" if is_custom else "⚙️ Standard"
                        else:
                            name = item.get('uri', 'Unknown')
                            desc = item.get('description', '')
                            tool_badge = "📚 Resource"
                        
                        st.markdown(f"""
                        <div class="rag-match">
                            <strong>#{i+1} {tool_badge}:</strong> {name}<br>
                            <small>{desc}</small><br>
                            <span class="similarity-score">Similarity: {similarity:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display results
                if is_batch:
                    st.markdown("### 🔄 Batch Operation Results:")
                
                for result in results:
                    formatted_display = enhanced_format_result_for_display(result)
                    
                    if result.get('success', True):
                        if result.get("type") == "custom_tool":
                            st.markdown(f'<div class="custom-tool">{formatted_display}</div>', unsafe_allow_html=True)
                        elif "operation_id" in result:
                            st.markdown(f'<div class="batch-operation">{formatted_display}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="tool-call">{formatted_display}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-message">{formatted_display}</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.info("💡 Try clicking 'Refresh Server Discovery' if connection issues persist")
    
    with col2:
        st.subheader("📊 Query Analysis")

        # Display debug info
        if st.session_state.last_parsed_query:
            parsed_query = st.session_state.last_parsed_query
            parsing_mode = "RAG-Enhanced" if parsed_query.get('rag_enhanced') else "Legacy"
            st.success(f"✅ Query processed in {st.session_state.elapsed_time}ms using {parsing_mode} parsing")
            
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
                        st.write(f"• Custom: {item.get('is_custom', False)}")
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
                with st.expander("[Session Statistics]"):
                    latest_entry = recent_entries[0]
                    st.info(f"🔍 [Parser] {latest_entry['parsing_mode']}")
                    if latest_entry['model_name']:
                        st.info(f"🤖 [Model] {latest_entry['model_name']}")
                    
                    if len(recent_entries) > 1:
                        successful = sum(1 for entry in recent_entries if entry['success'])
                        avg_time = sum(entry['elapsed_time_ms'] or 0 for entry in recent_entries) / len(recent_entries)
                        rag_enhanced_count = sum(1 for entry in recent_entries if entry['parsing_mode'] == 'RAG-Enhanced')
                        batch_count = sum(1 for entry in recent_entries if entry.get('is_batch'))
                        
                        st.metric("Queries", len(recent_entries))
                        st.metric("Success Rate", f"{(successful/len(recent_entries)*100):.1f}%")
                        st.metric("Avg Response Time", f"{avg_time:.0f}ms")
                        st.metric("RAG-Enhanced", f"{rag_enhanced_count}/{len(recent_entries)}")
                        st.metric("Batch Operations", f"{batch_count}")
            else:
                st.info("💡 No queries in this session yet. Try asking something!")
                
        except Exception as e:
            st.error(f"Error loading query analysis: {e}")
        
        # Sample queries
        st.subheader("💡 Example Queries")
        st.markdown(SAMPLE_QUERIES)

if __name__ == "__main__":
    main()