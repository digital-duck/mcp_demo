# Enhanced MCP Client with Dynamic Tool Registration and Batch Operations
# Add these classes and modifications to your existing code

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import concurrent.futures
from datetime import datetime
import logging

# ============================================================================
# 1. DYNAMIC TOOL REGISTRATION SYSTEM
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
# 2. BATCH OPERATIONS SYSTEM
# ============================================================================

@dataclass
class BatchOperation:
    """Single operation in a batch"""
    id: str
    tool: str
    params: Dict[str, Any]
    depends_on: Optional[List[str]] = None  # List of operation IDs this depends on
    variable_name: Optional[str] = None  # Store result in variable for reference

@dataclass
class BatchRequest:
    """Complete batch request"""
    operations: List[BatchOperation]
    parallel: bool = False  # Execute operations in parallel where possible
    fail_fast: bool = True  # Stop on first error
    timeout: int = 300  # Total timeout in seconds

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
    
    def __init__(self, mcp_client_func, custom_registry: DynamicToolRegistry):
        self.mcp_client_func = mcp_client_func
        self.custom_registry = custom_registry
        self.variables = {}  # Store results for variable references
    
    async def execute_batch(self, batch_request: BatchRequest) -> List[BatchResult]:
        """Execute a batch of operations"""
        results = []
        completed_ops = set()
        
        try:
            if batch_request.parallel:
                results = await self._execute_parallel(batch_request, completed_ops)
            else:
                results = await self._execute_sequential(batch_request, completed_ops)
                
        except Exception as e:
            logging.error(f"Batch execution failed: {e}")
            # Return partial results with error
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
            # Find operations that can be executed (dependencies met)
            ready_ops = [
                op for op in operations 
                if not op.depends_on or all(dep in completed_ops for dep in op.depends_on)
            ]
            
            if not ready_ops:
                # Circular dependency or missing dependency
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
                # Store result in variables if specified
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
            # Find all operations that can be executed in parallel
            ready_ops = [
                op for op in operations 
                if not op.depends_on or all(dep in completed_ops for dep in op.depends_on)
            ]
            
            if not ready_ops:
                break
            
            # Execute ready operations in parallel
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
            
            # Remove completed operations
            for op in ready_ops:
                operations.remove(op)
        
        return results
    
    async def _execute_single_operation(self, operation: BatchOperation, fail_fast: bool) -> BatchResult:
        """Execute a single operation"""
        start_time = time.time()
        
        try:
            # Resolve variable references in parameters
            resolved_params = self._resolve_parameters(operation.params)
            
            # Check if it's a custom tool
            custom_tool = self.custom_registry.get_tool(operation.tool)
            if custom_tool:
                result = await self._execute_custom_tool(custom_tool, resolved_params)
            else:
                # Execute via standard MCP
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
                # Variable reference
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
            # Local function execution
            if tool_def.name in self.custom_registry.local_functions:
                func = self.custom_registry.local_functions[tool_def.name]
                if asyncio.iscoroutinefunction(func):
                    return await func(**params)
                else:
                    return func(**params)
            else:
                raise ValueError(f"Local function not found: {tool_def.name}")
        else:
            # HTTP endpoint execution
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
        return await self.mcp_client_func(tool_name, params)

# ============================================================================
# 3. ENHANCED QUERY PARSER FOR BATCH OPERATIONS
# ============================================================================

class EnhancedQueryParser:
    """Enhanced parser that can handle batch operations and custom tools"""
    
    def __init__(self, registry: DynamicToolRegistry, llm_parser: 'LLMQueryParser'):
        self.registry = registry
        self.llm_parser = llm_parser
    
    def parse_batch_query(self, query: str) -> Optional[BatchRequest]:
        """Parse queries that request batch operations"""
        query_lower = query.lower()
        
        # Detect batch operation patterns
        batch_indicators = [
            "and then", "then", "after that", "followed by",
            "also calculate", "also find", "batch", "multiple",
            "calculate all", "find all"
        ]
        
        if not any(indicator in query_lower for indicator in batch_indicators):
            return None
        
        try:
            # Enhanced prompt for batch parsing
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
Available custom tools: {[tool.name for tool in self.registry.list_tools()]}

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
# 4. ENHANCED RAG SYSTEM FOR CUSTOM TOOLS
# ============================================================================

class EnhancedMCPRAGSystem(MCPRAGSystem):
    """Enhanced RAG system that includes custom tools"""
    
    def __init__(self, custom_registry: DynamicToolRegistry):
        super().__init__()
        self.custom_registry = custom_registry
    
    def build_embeddings(self, tools: List[Dict], resources: List[Dict]):
        """Build embeddings including custom tools"""
        # Add custom tools to the standard tools
        custom_tools = []
        for tool_def in self.custom_registry.list_tools():
            custom_tools.append({
                'name': tool_def.name,
                'description': tool_def.description,
                'category': tool_def.category,
                'tags': tool_def.tags,
                'examples': tool_def.examples
            })
        
        # Combine standard and custom tools
        all_tools = tools + custom_tools
        
        # Call parent method with enhanced tool list
        super().build_embeddings(all_tools, resources)
    
    def build_rich_context(self, item: Dict, item_type: str) -> str:
        """Enhanced context building for custom tools"""
        if item_type == "tool" and 'category' in item:
            # This is a custom tool
            name = item.get('name', '')
            desc = item.get('description', '')
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
            # Use parent method for standard tools/resources
            return super().build_rich_context(item, item_type)

# ============================================================================
# 5. EXAMPLE USAGE AND SAMPLE TOOLS
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
        endpoint="FUNCTION",  # Local function
        examples=[
            "convert 5 feet to meters",
            "convert 100 km to miles", 
            "convert 12 inches to cm"
        ],
        tags=["conversion", "units", "measurement"],
        author="System"
    )
    
    # Local function for unit converter
    def unit_converter_func(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        # Simple conversion factors to meters
        to_meters = {
            "m": 1, "ft": 0.3048, "in": 0.0254, "cm": 0.01, 
            "mm": 0.001, "km": 1000, "miles": 1609.34
        }
        
        # Convert to meters first, then to target unit
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
# 6. STREAMLIT UI COMPONENTS FOR TOOL MANAGEMENT
# ============================================================================

def render_tool_management_ui(registry: DynamicToolRegistry):
    """Render UI for managing custom tools"""
    
    st.subheader("🔧 Tool Management")
    
    # Tabs for different tool operations
    tab1, tab2, tab3 = st.tabs(["📋 View Tools", "➕ Add Tool", "🗑️ Remove Tool"])
    
    with tab1:
        # List existing tools
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
        # Add new tool form
        st.write("**Create a New Custom Tool**")
        
        with st.form("add_tool_form"):
            name = st.text_input("Tool Name", help="Unique identifier for the tool")
            description = st.text_area("Description", help="What does this tool do?")
            category = st.selectbox("Category", 
                                   ["calculation", "conversion", "text", "data", "utility", "other"])
            author = st.text_input("Author", value="User")
            
            # Parameters section
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
            
            # Examples and tags
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
        # Remove tools
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
    
    # Batch operation examples
    st.write("**Try these batch operation examples:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Math Chain Example"):
            st.session_state.example_query = "Calculate 15 plus 27, then find sine of that result in degrees"
    
    with col2:
        if st.button("🔄 Conversion Chain Example"):
            st.session_state.example_query = "Convert 5 feet to meters, then multiply that by 3.14"

# Example integration into your main Streamlit app:
def enhanced_main():
    """Enhanced main function with dynamic tools and batch operations"""
    
    # Initialize enhanced systems
    if 'custom_registry' not in st.session_state:
        st.session_state.custom_registry = DynamicToolRegistry()
        # Create sample tools on first run
        create_sample_custom_tools(st.session_state.custom_registry)
    
    if 'batch_processor' not in st.session_state:
        # You'll need to adapt this to your MCP client function
        async def mcp_client_func(tool_name, params):
            async with Client(MCP_SERVER_PATH) as client:
                result = await client.call_tool(tool_name, params)
                return extract_result_data(result)
        
        st.session_state.batch_processor = BatchProcessor(
            mcp_client_func=mcp_client_func,
            custom_registry=st.session_state.custom_registry
        )
    
    if 'enhanced_parser' not in st.session_state:
        # Initialize with your existing LLM parser
        llm_parser = LLMQueryParser(st.session_state.llm_provider)
        st.session_state.enhanced_parser = EnhancedQueryParser(
            registry=st.session_state.custom_registry,
            llm_parser=llm_parser
        )
    
    # Enhanced RAG system
    if 'enhanced_rag_system' not in st.session_state:
        st.session_state.enhanced_rag_system = EnhancedMCPRAGSystem(
            custom_registry=st.session_state.custom_registry
        )

# ============================================================================
# 7. INTEGRATION WITH EXISTING QUERY PROCESSING
# ============================================================================

async def enhanced_execute_query(user_query: str) -> Tuple[List[Dict], int, bool]:
    """Enhanced query execution with batch and custom tool support"""
    start_time = time.time()
    
    try:
        # First, check if it's a batch operation
        batch_request = st.session_state.enhanced_parser.parse_batch_query(user_query)
        
        if batch_request:
            # Execute batch operations
            batch_results = await st.session_state.batch_processor.execute_batch(batch_request)
            
            # Convert batch results to standard format
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
            # Single operation - use existing logic but check for custom tools first
            parsed_query = None
            
            # Try RAG-enhanced parsing with custom tools
            if st.session_state.use_llm and st.session_state.use_rag and RAG_AVAILABLE:
                # Rebuild embeddings to include custom tools
                tools, resources = get_mcp_server_info()
                st.session_state.enhanced_rag_system.build_embeddings(tools, resources)
                
                # Use enhanced RAG system
                parser = LLMQueryParser(st.session_state.llm_provider)
                if parser.client:
                    parsed_query = parser.parse_query_with_rag(user_query, st.session_state.enhanced_rag_system)
            
            # Fallback to standard parsing
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
            
            # Execute single operation
            results = []
            tool_name = parsed_query.get("tool")
            parameters = parsed_query.get("params", {})
            
            if tool_name:
                # Check if it's a custom tool
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
# 8. ENHANCED RESULT FORMATTING
# ============================================================================

def enhanced_format_result_for_display(result: Dict) -> str:
    """Enhanced result formatting for custom tools and batch operations"""
    
    if result.get("type") == "custom_tool":
        tool_name = result.get("name", "Unknown")
        category = result.get("category", "custom")
        data = result.get("data", {})
        
        if isinstance(data, dict) and "error" in data:
            return f"❌ [Custom Tool Error] {data['error']}"
        
        # Format based on tool category
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
        # Use existing formatting for standard tools
        tool_name = result.get("name")
        data = result.get("data", {})
        
        if "operation_id" in result:
            # This is from a batch operation
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
# 9. ENHANCED SIDEBAR WITH TOOL MANAGEMENT
# ============================================================================

def enhanced_sidebar():
    """Enhanced sidebar with tool management features"""
    
    with st.sidebar:
        st.header("⚙️ Enhanced Configuration")
        
        # Existing configuration sections...
        # (Keep your existing sidebar code)
        
        # Add new sections for enhanced features
        st.divider()
        
        # Custom Tools Section
        st.subheader("🔧 Custom Tools")
        custom_tools = st.session_state.custom_registry.list_tools()
        
        if custom_tools:
            st.success(f"✅ {len(custom_tools)} custom tools registered")
            
            # Show tool categories
            categories = {}
            for tool in custom_tools:
                categories[tool.category] = categories.get(tool.category, 0) + 1
            
            for category, count in categories.items():
                st.write(f"  • {category.title()}: {count} tools")
            
            # Quick tool search
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
        
        # Batch operation status
        if hasattr(st.session_state, 'last_batch_info'):
            batch_info = st.session_state.last_batch_info
            st.info(f"Last batch: {batch_info.get('operation_count', 0)} operations in {batch_info.get('total_time_ms', 0)}ms")
        
        # Batch operation examples
        st.write("**Quick Examples:**")
        if st.button("📊 Math Chain", key="batch_math"):
            st.session_state.example_query = "Calculate 15 + 27, then find sine of that result"
        
        if st.button("🔄 Convert Chain", key="batch_convert"):
            st.session_state.example_query = "Convert 5 feet to meters, then multiply by 2"
        
        # Performance metrics
        st.divider()
        st.subheader("📈 Performance")
        
        # Show RAG effectiveness
        if hasattr(st.session_state, 'rag_stats'):
            rag_stats = st.session_state.rag_stats
            st.metric("RAG Accuracy", f"{rag_stats.get('accuracy', 0):.1%}")
            st.metric("Avg Similarity", f"{rag_stats.get('avg_similarity', 0):.3f}")
        
        # Show tool usage stats
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
# 10. USAGE EXAMPLES
# ============================================================================

# Example configuration for your Streamlit app
ENHANCED_SAMPLE_QUERIES = """
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

# Integration example for your main app:
def integrate_enhancements():
    """How to integrate these enhancements into your existing app"""
    
    # 1. Replace your query execution with enhanced version
    # In your submit button handler:
    if submit_button and user_query:
        results, elapsed_time, is_batch = asyncio.run(enhanced_execute_query(user_query))
        
        # Store batch info for sidebar display
        if is_batch:
            st.session_state.last_batch_info = {
                'operation_count': len(results),
                'total_time_ms': elapsed_time,
                'timestamp': datetime.now()
            }
        
        # Use enhanced result formatting
        for result in results:
            formatted_display = enhanced_format_result_for_display(result)
            
            if result.get('success', True):
                st.markdown(f'<div class="tool-call">{formatted_display}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-message">{formatted_display}</div>', unsafe_allow_html=True)
    
    # 2. Add tool management UI
    if st.session_state.get('show_tool_management', False):
        render_tool_management_ui(st.session_state.custom_registry)
        render_batch_operations_ui()
        
        if st.button("◀️ Back to Main"):
            st.session_state.show_tool_management = False
            st.rerun()
    
    # 3. Use enhanced sidebar
    enhanced_sidebar()
    
    # 4. Update sample queries
    st.markdown("### 💡 Enhanced Example Queries")
    st.markdown(ENHANCED_SAMPLE_QUERIES)

# ============================================================================
# 11. YAML CONFIGURATION SUPPORT
# ============================================================================

def export_tools_to_yaml(registry: DynamicToolRegistry, filename: str = "custom_tools.yaml"):
    """Export custom tools to YAML for easy sharing"""
    tools_data = {}
    
    for name, tool in registry.tools.items():
        tools_data[name] = {
            'description': tool.description,
            'category': tool.category,
            'parameters': [asdict(param) for param in tool.parameters],
            'endpoint': tool.endpoint,
            'method': tool.method,
            'examples': tool.examples,
            'tags': tool.tags,
            'author': tool.author,
            'version': tool.version
        }
    
    with open(filename, 'w') as f:
        yaml.dump(tools_data, f, default_flow_style=False, indent=2)
    
    return filename

def import_tools_from_yaml(registry: DynamicToolRegistry, filename: str):
    """Import custom tools from YAML file"""
    try:
        with open(filename, 'r') as f:
            tools_data = yaml.safe_load(f)
        
        imported_count = 0
        for name, tool_data in tools_data.items():
            # Convert parameters back to ToolParameter objects
            parameters = [ToolParameter(**param) for param in tool_data['parameters']]
            
            tool_def = CustomToolDefinition(
                name=name,
                description=tool_data['description'],
                category=tool_data['category'],
                parameters=parameters,
                endpoint=tool_data['endpoint'],
                method=tool_data.get('method', 'POST'),
                examples=tool_data.get('examples', []),
                tags=tool_data.get('tags', []),
                author=tool_data.get('author', 'Imported'),
                version=tool_data.get('version', '1.0.0')
            )
            
            if registry.register_tool(tool_def):
                imported_count += 1
        
        return imported_count
        
    except Exception as e:
        logging.error(f"Failed to import tools from YAML: {e}")
        return 0