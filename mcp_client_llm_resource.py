import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from fastmcp import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LLM Integration Options - Enhanced with Claude 3.5 Sonnet and GPT-4o-mini
LLM_MODELS = ["openai", "anthropic", "ollama", "gemini", "bedrock"]
LLM_PROVIDER = "anthropic"  # Default to Claude 3.5 Sonnet

# --- Enhanced LLM Query Parser ---
class LLMQueryParser:
    """Use LLM to parse natural language queries into tool calls"""
    
    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.setup_llm_client()
    
    def setup_llm_client(self):
        """Setup the appropriate LLM client based on provider"""
        if self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    # Test with a simple completion to verify API access
                    try:
                        test_response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=1
                        )
                        logging.info("✅ OpenAI client initialized with GPT-4o-mini")
                        self.model_name = "gpt-4o-mini"
                    except Exception as e:
                        logging.warning(f"GPT-4o-mini not accessible, falling back to gpt-3.5-turbo: {e}")
                        self.model_name = "gpt-3.5-turbo"
                else:
                    logging.error("❌ OPENAI_API_KEY not set")
                    self.client = None
            except ImportError:
                logging.error("❌ OpenAI library not installed. Run: pip install openai")
                self.client = None
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    # Test with Claude 3.5 Sonnet
                    try:
                        test_response = self.client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=1,
                            messages=[{"role": "user", "content": "test"}]
                        )
                        logging.info("✅ Anthropic client initialized with Claude 3.5 Sonnet")
                        self.model_name = "claude-3-5-sonnet-20241022"
                    except Exception as e:
                        logging.warning(f"Claude 3.5 Sonnet not accessible, trying Claude 3 Haiku: {e}")
                        try:
                            test_response = self.client.messages.create(
                                model="claude-3-haiku-20240307",
                                max_tokens=1,
                                messages=[{"role": "user", "content": "test"}]
                            )
                            self.model_name = "claude-3-haiku-20240307"
                            logging.info("✅ Anthropic client initialized with Claude 3 Haiku")
                        except Exception as e2:
                            logging.error(f"❌ Anthropic API error: {e2}")
                            self.client = None
                else:
                    logging.error("❌ ANTHROPIC_API_KEY not set")
                    self.client = None
            except ImportError:
                logging.error("❌ Anthropic library not installed. Run: pip install anthropic")
                self.client = None
        
        elif self.provider == "ollama":
            try:
                import requests
                # Test if Ollama is running
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    logging.info("✅ Ollama server detected")
                    self.client = "ollama"  # Use string indicator
                    self.model_name = "llama3.2"
                else:
                    logging.error("❌ Ollama server not responding")
                    self.client = None
            except Exception as e:
                logging.error(f"❌ Ollama connection failed: {e}")
                logging.error("Make sure Ollama is installed and running: https://ollama.ai")
                self.client = None
        
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    # Updated Gemini models list
                    gemini_models = [
                        "gemini-2.0-flash-exp",
                        "gemini-exp-1206", 
                        "gemini-2.0-flash-thinking-exp-1219",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                    ]
                    
                    # Try models in order of preference
                    for model_name in gemini_models:
                        try:
                            self.client = genai.GenerativeModel(model_name)
                            # Test the model
                            test_response = self.client.generate_content("test", generation_config={"max_output_tokens": 1})
                            self.model_name = model_name
                            logging.info(f"✅ Gemini client initialized with {model_name}")
                            break
                        except Exception as e:
                            logging.warning(f"Model {model_name} not available: {e}")
                            continue
                    else:
                        logging.error("❌ No Gemini models available")
                        self.client = None
                else:
                    logging.error("❌ GEMINI_API_KEY not set")
                    self.client = None
            except ImportError:
                logging.error("❌ Gemini library not installed. Run: pip install google-generativeai")
                self.client = None
        
        elif self.provider == "bedrock":
            try:
                import boto3
                from botocore.exceptions import NoCredentialsError, PartialCredentialsError
                
                region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
                
                try:
                    self.client = boto3.client(
                        service_name='bedrock-runtime',
                        region_name=region
                    )
                    
                    # Updated model ID for Claude 3.5 Sonnet v2
                    self.bedrock_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
                    self.model_name = self.bedrock_model_id
                    logging.info(f"✅ AWS Bedrock client initialized (region: {region})")
                    logging.info(f"📋 Using model: {self.bedrock_model_id}")
                    
                except (NoCredentialsError, PartialCredentialsError) as e:
                    logging.error(f"❌ AWS credentials not configured: {e}")
                    self.client = None
                    
            except ImportError:
                logging.error("❌ Boto3 library not installed. Run: pip install boto3")
                self.client = None
    
    def get_system_prompt(self, available_tools: List[Dict], available_resources: List[Dict] = None) -> str:
        """Create system prompt with available tools and resources"""
        tools_desc = "\n".join([
            f"- {tool['name']}: {tool.get('description', 'No description')}"
            for tool in available_tools
        ])
        
        resources_desc = ""
        if available_resources:
            resources_desc = "\n\nAvailable resources:\n" + "\n".join([
                f"- {resource['uri']}: {resource.get('description', 'No description')}"
                for resource in available_resources
            ])
        
        return f"""You are a tool and resource selection assistant. Given a user query, you must decide whether to use a tool, read a resource, or both.

Available tools:
{tools_desc}{resources_desc}

For each user query, respond with ONLY a JSON object in this exact format:
{{
    "action": "tool|resource|both",
    "tool": "tool_name_or_null",
    "resource_uri": "resource_uri_or_null",
    "params": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this action was chosen"
}}

Tool-specific parameter requirements:
- calculator: operation (add/subtract/multiply/divide/power), num1 (number), num2 (number)
- trig: operation (sine/cosine/tangent/arc sine/arc cosine/arc tangent), num1 (float), unit (degree/radian)
- stock_quote: ticker (stock symbol like AAPL, MSFT)
- health: no parameters needed
- echo: message (text to echo back)

Resource-specific patterns:
- info://server: Server information (no parameters)
- stock://{{ticker}}: Stock information for specific ticker

If you cannot determine which action to take, respond with:
{{
    "action": null,
    "tool": null,
    "resource_uri": null,
    "params": {{}},
    "confidence": 0.0,
    "reasoning": "Could not parse the query"
}}

Examples:
User: "What is 15 plus 27?"
Response: {{"action": "tool", "tool": "calculator", "resource_uri": null, "params": {{"operation": "add", "num1": 15, "num2": 27}}, "confidence": 0.98, "reasoning": "Clear arithmetic addition request"}}

User: "Find sine of 30 degrees"
Response: {{"action": "tool", "tool": "trig", "resource_uri": null, "params": {{"operation": "sine", "num1": 30, "unit": "degree"}}, "confidence": 0.95, "reasoning": "Trigonometric function request with angle in degrees"}}

User: "Tell me about Apple as a company"
Response: {{"action": "resource", "tool": null, "resource_uri": "stock://AAPL", "params": {{}}, "confidence": 0.90, "reasoning": "Request for Apple company information from stock resource"}}

User: "Get Apple stock price and company info"
Response: {{"action": "both", "tool": "stock_quote", "resource_uri": "stock://AAPL", "params": {{"ticker": "AAPL"}}, "confidence": 0.95, "reasoning": "Request for both current price and company information"}}

User: "server info"
Response: {{"action": "resource", "tool": null, "resource_uri": "info://server", "params": {{}}, "confidence": 0.95, "reasoning": "Request for server information"}}

Remember: Respond with ONLY the JSON object, no additional text."""

    async def parse_query_with_llm(self, query: str, available_tools: List[Dict], available_resources: List[Dict] = None) -> Optional[Dict[str, Any]]:
        """Use LLM to parse the query"""
        if not self.client:
            logging.error("❌ LLM client not available, falling back to rule-based parsing")
            return None
        
        system_prompt = self.get_system_prompt(available_tools, available_resources)
        
        try:
            if self.provider == "openai":
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
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=300,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": query}]
                )
                llm_response = response.content[0].text.strip()
            
            elif self.provider == "ollama":
                import requests
                payload = {
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nUser: {query}\nResponse:",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 300
                    }
                }
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
                if response.status_code == 200:
                    llm_response = response.json()["response"].strip()
                else:
                    logging.error(f"Ollama API error: {response.status_code}")
                    return None
            
            elif self.provider == "gemini":
                response = self.client.generate_content(
                    f"{system_prompt}\n\nUser: {query}",
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 300
                    }
                )
                llm_response = response.text.strip()
            
            elif self.provider == "bedrock":
                import json as json_lib
                
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "temperature": 0.1,
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                }
                
                response = self.client.invoke_model(
                    modelId=self.bedrock_model_id,
                    body=json_lib.dumps(request_body),
                    contentType="application/json",
                    accept="application/json"
                )
                
                response_body = json_lib.loads(response['body'].read())
                llm_response = response_body['content'][0]['text'].strip()
                
                logging.debug(f"Bedrock response usage: {response_body.get('usage', {})}")
            
            # Parse the JSON response
            logging.debug(f"LLM Response: {llm_response}")
            
            # Clean up the response - sometimes LLMs add extra text
            if llm_response.startswith("```json"):
                llm_response = llm_response.replace("```json", "").replace("```", "").strip()
            elif llm_response.startswith("```"):
                llm_response = llm_response.replace("```", "").strip()
            
            # Handle case where LLM wraps response in additional text
            try:
                parsed_response = json.loads(llm_response)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    parsed_response = json.loads(json_match.group())
                else:
                    raise
            
            # Validate the response structure
            if not isinstance(parsed_response, dict):
                logging.error("LLM response is not a dictionary")
                return None
            
            if parsed_response.get("action") is None:
                logging.warning(f"LLM could not parse query: {parsed_response.get('reasoning', 'Unknown reason')}")
                return None
            
            confidence = parsed_response.get("confidence", 0.0)
            if confidence < 0.5:
                logging.warning(f"Low confidence ({confidence}) in LLM parsing: {parsed_response.get('reasoning')}")
                return None
            
            logging.info(f"🤖 LLM ({self.provider}:{getattr(self, 'model_name', 'unknown')}) parsed query: {parsed_response.get('reasoning')} (confidence: {confidence})")
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM JSON response: {e}")
            logging.error(f"Raw LLM response: {llm_response}")
            return None
        except Exception as e:
            logging.error(f"LLM query parsing error: {e}")
            return None

# --- Enhanced Rule-Based Parser with Resource Support ---
class RuleBasedQueryParser:
    """Enhanced rule-based query parser with resource support"""
    
    @staticmethod
    def parse_calculator_query(query: str) -> Optional[Dict[str, Any]]:
        import re
        query = query.lower().strip()
        patterns = [
            ("add", ["plus", "add", "+", "sum"]),
            ("subtract", ["minus", "subtract", "-", "difference"]),
            ("multiply", ["times", "multiply", "*", "×", "product"]),
            ("divide", ["divide", "divided by", "/", "÷"]),
            ("power", ["power", "to the power", "raised to", "^", "**"]),
        ]
        
        for operation, keywords in patterns:
            for keyword in keywords:
                if keyword in query:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if len(numbers) >= 2:
                        try:
                            return {
                                "action": "tool",
                                "tool": "calculator",
                                "resource_uri": None,
                                "params": {
                                    "operation": operation,
                                    "num1": float(numbers[0]),
                                    "num2": float(numbers[1])
                                }
                            }
                        except ValueError:
                            pass
        return None
    
    @staticmethod
    def parse_trig_query(query: str) -> Optional[Dict[str, Any]]:
        import re
        query_lower = query.lower().strip()
        
        # Trigonometric operations
        trig_patterns = [
            ("sine", ["sine", "sin"]),
            ("cosine", ["cosine", "cos"]),
            ("tangent", ["tangent", "tan"]),
            ("arc sine", ["arc sine", "arcsin", "asin", "inverse sine"]),
            ("arc cosine", ["arc cosine", "arccos", "acos", "inverse cosine"]),
            ("arc tangent", ["arc tangent", "arctan", "atan", "inverse tangent"]),
        ]
        
        for operation, keywords in trig_patterns:
            for keyword in keywords:
                if keyword in query_lower:
                    # Extract number
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if numbers:
                        # Determine unit
                        unit = "degree"  # default
                        if any(word in query_lower for word in ["radian", "rad"]):
                            unit = "radian"
                        
                        try:
                            return {
                                "action": "tool",
                                "tool": "trig",
                                "resource_uri": None,
                                "params": {
                                    "operation": operation,
                                    "num1": float(numbers[0]),
                                    "unit": unit
                                }
                            }
                        except ValueError:
                            pass
        return None
    
    @staticmethod
    def parse_stock_query(query: str) -> Optional[Dict[str, Any]]:
        import re
        query_lower = query.lower().strip()
        
        # Check for company info requests
        info_keywords = ["about", "company", "info", "information", "describe", "tell me about", "what is"]
        is_info_request = any(keyword in query_lower for keyword in info_keywords)
        
        # Check for price requests
        price_keywords = ["stock", "price", "quote", "ticker", "share", "trading", "cost", "current"]
        is_price_request = any(keyword in query_lower for keyword in price_keywords)
        
        # Extract ticker symbols
        tickers = re.findall(r'\b[A-Z]{2,5}\b', query.upper())
        excluded_words = {"GET", "THE", "FOR", "AND", "BUT", "NOT", "YOU", "ALL", "CAN", "STOCK", "PRICE", "WHAT", "TELL", "ABOUT"}
        valid_tickers = [t for t in tickers if t not in excluded_words]
        
        # Check for common company names and map to tickers
        company_mapping = {
            "apple": "AAPL",
            "google": "GOOGL", 
            "alphabet": "GOOGL",
            "microsoft": "MSFT",
            "tesla": "TSLA",
            "amazon": "AMZN",
            "meta": "META",
            "facebook": "META",
            "nvidia": "NVDA",
            "netflix": "NFLX",
            "salesforce": "CRM"
        }
        
        for company, ticker in company_mapping.items():
            if company in query_lower:
                valid_tickers.append(ticker)
        
        if valid_tickers:
            ticker = valid_tickers[0]
            
            if is_info_request and is_price_request:
                return {
                    "action": "both",
                    "tool": "stock_quote",
                    "resource_uri": f"stock://{ticker}",
                    "params": {"ticker": ticker}
                }
            elif is_info_request:
                return {
                    "action": "resource",
                    "tool": None,
                    "resource_uri": f"stock://{ticker}",
                    "params": {}
                }
            elif is_price_request:
                return {
                    "action": "tool",
                    "tool": "stock_quote",
                    "resource_uri": None,
                    "params": {"ticker": ticker}
                }
        
        return None
    
    @staticmethod
    def parse_server_info_query(query: str) -> Optional[Dict[str, Any]]:
        query_lower = query.lower().strip()
        server_keywords = ["server", "info", "information", "about server", "server status", "server details"]
        
        if any(keyword in query_lower for keyword in server_keywords):
            return {
                "action": "resource",
                "tool": None,
                "resource_uri": "info://server",
                "params": {}
            }
        return None
    
    @staticmethod
    def parse_query(query: str) -> Optional[Dict[str, Any]]:
        # Health check
        if any(word in query.lower() for word in ["health", "status", "ping", "alive"]):
            return {
                "action": "tool",
                "tool": "health", 
                "resource_uri": None,
                "params": {}
            }
        
        # Echo command
        if query.lower().startswith("echo "):
            message = query[5:].strip()
            return {
                "action": "tool",
                "tool": "echo",
                "resource_uri": None, 
                "params": {"message": message}
            }
        
        # Server info resource
        server_result = RuleBasedQueryParser.parse_server_info_query(query)
        if server_result:
            return server_result
        
        # Trigonometric functions
        trig_result = RuleBasedQueryParser.parse_trig_query(query)
        if trig_result:
            return trig_result
        
        # Calculator
        calc_result = RuleBasedQueryParser.parse_calculator_query(query)
        if calc_result:
            return calc_result
        
        # Stock queries (including resources)
        stock_result = RuleBasedQueryParser.parse_stock_query(query)
        if stock_result:
            return stock_result
        
        return None

# --- Resource Handling Functions ---
def extract_resource_data(result):
    """Extract data from resource result"""
    try:
        if isinstance(result, list) and len(result) > 0:
            content_item = result[0]
            if hasattr(content_item, 'text'):
                return content_item.text
            else:
                return str(content_item)
        elif hasattr(result, 'content') and result.content:
            content_item = result.content[0]
            if hasattr(content_item, 'text'):
                return content_item.text
            else:
                return str(content_item)
        else:
            return str(result)
    except Exception as e:
        logging.error(f"Error extracting resource data: {e}")
        return f"Could not parse resource: {e}"

def format_resource_result(resource_uri: str, content: str) -> str:
    """Format resource results for display"""
    if resource_uri.startswith("info://server"):
        return f"🖥️  Server Info: {content}"
    elif resource_uri.startswith("stock://"):
        ticker = resource_uri.split("://")[1]
        return f"🏢 Company Info ({ticker}): {content}"
    else:
        return f"📄 Resource ({resource_uri}): {content}"

# --- Tool Result Extraction and Formatting ---
def extract_result_data(result):
    """Extract actual data from FastMCP result object"""
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
            if isinstance(result, dict):
                return result
            else:
                return {"result": str(result)}
    except Exception as e:
        logging.error(f"Error extracting result data: {e}")
        return {"error": f"Could not parse result: {e}"}

def format_result(tool_name: str, result: Dict) -> str:
    """Format tool results for display"""
    if isinstance(result, dict) and "error" in result:
        return f"❌ Error: {result['error']}"
    
    if tool_name == "calculator":
        if "result" in result:
            expression = result.get('expression')
            if expression:
                return f"🧮 {expression}"
            else:
                num1 = result.get('num1', '?')
                num2 = result.get('num2', '?')
                operation = result.get('operation', '?')
                calc_result = result.get('result', '?')
                return f"🧮 {num1} {operation} {num2} = {calc_result}"
        elif "error" in result:
            return f"❌ Calculator Error: {result['error']}"
    
    elif tool_name == "trig":
        if "result" in result:
            expression = result.get('expression')
            if expression:
                return f"📐 {expression}"
            else:
                operation = result.get('operation', '?')
                num1 = result.get('num1', '?')
                unit = result.get('unit', '?')
                trig_result = result.get('result', '?')
                return f"📐 {operation}({num1}°) = {trig_result} [{unit}]"
        elif "error" in result:
            return f"❌ Trig Error: {result['error']}"
    
    elif tool_name == "stock_quote":
        if "current_price" in result:
            ticker = result.get('ticker', 'Unknown')
            name = result.get('company_name', ticker)
            price = result.get('current_price', 0)
            currency = result.get('currency', 'USD')
            extra_info = []
            if result.get('volume'):
                extra_info.append(f"Vol: {result['volume']:,}")
            if result.get('day_high') and result.get('day_low'):
                extra_info.append(f"Range: {result['day_low']}-{result['day_high']}")
            extra = f" ({', '.join(extra_info)})" if extra_info else ""
            return f"📈 {name} ({ticker}): {currency} {price}{extra}"
        elif "error" in result:
            return f"❌ Stock Error: {result['error']}"
    
    elif tool_name == "health":
        if isinstance(result, dict):
            return f"✅ {result.get('message', 'Server is healthy')}"
    
    elif tool_name == "echo":
        if isinstance(result, dict):
            return f"🔊 {result.get('echo', result.get('message', str(result)))}"
    
    try:
        return f"✅ Result: {json.dumps(result, indent=2)}"
    except (TypeError, ValueError):
        return f"✅ Result: {str(result)}"

# --- Main Demo Function ---
async def run_enhanced_demo():
    print("🚀 Enhanced LLM-powered MCP client demo starting...")
    print(f"🤖 Using LLM Provider: {LLM_PROVIDER}")
    
    # Initialize parsers
    llm_parser = LLMQueryParser(LLM_PROVIDER)
    fallback_parser = RuleBasedQueryParser()
    
    server_path = "mcp_server.py"
    
    try:
        print(f"📡 Connecting to MCP server: {server_path}")
        
        async with Client(server_path) as client:
            print("✅ Connected to MCP server!")
            
            # Discover available tools and resources
            available_tools = []
            available_resources = []
            
            try:
                # Get tools
                tools = await client.list_tools()
                if tools:
                    available_tools = [
                        {"name": tool.name, "description": tool.description} for tool in tools
                    ]
                    print(f"✅ Found {len(available_tools)} tools")
                
                # Get resources
                try:
                    resources = await client.list_resources()
                    if resources:
                        available_resources = [
                            {"uri": resource.uri, "description": resource.description}
                            for resource in resources
                        ]
                        print(f"✅ Found {len(available_resources)} resources")
                except Exception as e:
                    logging.warning(f"Resource discovery failed: {e}")
                    available_resources = []
                
                # Add known dynamic resources that don't appear in discovery
                dynamic_resources = [
                    {"uri": "stock://{ticker}", "description": "Stock company information for any ticker"}
                ]
                available_resources.extend(dynamic_resources)
                
                if not available_tools:
                    # Fallback tool definitions
                    available_tools = [
                        {"name": "calculator", "description": "Perform arithmetic operations"},
                        {"name": "trig", "description": "Performs trigonometric operations"},
                        {"name": "stock_quote", "description": "Get stock price data"},
                        {"name": "health", "description": "Check server health"},
                        {"name": "echo", "description": "Echo back messages"}
                    ]
                
                if not available_resources:
                    # Fallback resource definitions
                    available_resources = [
                        {"uri": "info://server", "description": "Server information"},
                        {"uri": "stock://{ticker}", "description": "Stock company information for any ticker"}
                    ]
                    
            except Exception as e:
                logging.error(f"Tool/Resource discovery failed: {e}")
            
            print(f"\n{'='*70}")
            print("🎯 Enhanced LLM-Powered MCP Client Demo!")
            print(f"🤖 Model: {getattr(llm_parser, 'model_name', 'Unknown')} ({LLM_PROVIDER})")
            print("\n📝 Try natural language queries:")
            print("   Tool examples:")
            print("   • 'What's fifteen plus twenty seven?'")
            print("   • 'Can you multiply 12 by 8?'")
            print("   • 'Find sine of 30 degrees'")
            print("   • 'Calculate cosine of pi/4 radians'")
            print("   • 'I need the current Apple stock price'")
            print("   • 'How much is Tesla trading for?'")
            print("   • 'Please echo back: Hello AI!'")
            print("   • 'Is the server working properly?'")
            print("\n   Resource examples:")
            print("   • 'Tell me about Apple as a company'")
            print("   • 'What is Microsoft?'")
            print("   • 'Give me server information'")
            print("   • 'Show me info about Tesla'")
            print("\n   Combined examples:")
            print("   • 'Get Apple stock price and company info'")
            print("   • 'I want both Tesla's price and company details'")
            print("\n💡 Commands:")
            print("   • 'tools' - List available tools")
            print("   • 'resources' - List available resources") 
            print("   • 'switch' - Switch parsing mode")
            print("   • 'provider [name]' - Switch LLM provider")
            print("   • 'exit' - Quit the demo")
            print(f"{'='*70}")
            
            use_llm = llm_parser.client is not None
            
            while True:
                try:
                    user_input = input(f"\n💬 Your query {'🤖' if use_llm else '🔧'}: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ('exit', "quit", "bye", "q"):
                        print("👋 Goodbye!")
                        break
                    
                    if user_input.lower() == 'switch':
                        use_llm = not use_llm
                        mode = "LLM" if use_llm else "Rule-based"
                        print(f"🔄 Switched to {mode} parsing mode")
                        continue
                    
                    if user_input.lower().startswith('provider '):
                        new_provider = user_input[9:].strip()
                        if new_provider in LLM_MODELS:
                            print(f"🔄 Switching to {new_provider}...")
                            llm_parser = LLMQueryParser(new_provider)
                            use_llm = llm_parser.client is not None
                            print(f"✅ Now using {new_provider}" if use_llm else f"❌ Failed to initialize {new_provider}")
                        else:
                            print(f"❌ Unknown provider. Available: {', '.join(LLM_MODELS)}")
                        continue
                    
                    if user_input.lower() == 'tools':
                        print("\n🔧 Available tools:")
                        for tool in available_tools:
                            print(f"   • {tool['name']}: {tool['description']}")
                        continue
                    
                    if user_input.lower() == 'resources':
                        print("\n📚 Available resources:")
                        for resource in available_resources:
                            print(f"   • {resource['uri']}: {resource['description']}")
                        continue
                    
                    # Parse the query
                    parsed_query = None
                    
                    if use_llm and llm_parser.client:
                        parsed_query = await llm_parser.parse_query_with_llm(user_input, available_tools, available_resources)
                    
                    # Fallback to rule-based if LLM fails
                    if not parsed_query:
                        if use_llm:
                            print("🔄 LLM parsing failed, trying rule-based parser...")
                        parsed_query = fallback_parser.parse_query(user_input)
                    
                    if not parsed_query:
                        print("❓ I couldn't understand your query. Try rephrasing or being more specific.")
                        continue
                    
                    # Execute based on action type
                    action = parsed_query.get("action")
                    tool_name = parsed_query.get("tool")
                    resource_uri = parsed_query.get("resource_uri")
                    parameters = parsed_query.get("params", {})
                    
                    print(f"🎯 Action: {action}")
                    if parsed_query.get("reasoning"):
                        print(f"💭 Reasoning: {parsed_query['reasoning']}")
                    
                    # Execute tool call if needed
                    if action in ["tool", "both"] and tool_name:
                        print(f"🔧 Calling tool: {tool_name}")
                        if parameters:
                            print(f"📝 Parameters: {json.dumps(parameters, indent=2)}")
                        
                        try:
                            tool_result = await client.call_tool(tool_name, parameters)
                            tool_data = extract_result_data(tool_result)
                            print(format_result(tool_name, tool_data))
                        except Exception as e:
                            print(f"❌ Error calling tool: {e}")
                            logging.error(f"Tool call error: {e}", exc_info=True)
                    
                    # Execute resource read if needed
                    if action in ["resource", "both"] and resource_uri:
                        print(f"📚 Reading resource: {resource_uri}")
                        
                        try:
                            resource_result = await client.read_resource(resource_uri)
                            resource_content = extract_resource_data(resource_result)
                            print(format_resource_result(resource_uri, resource_content))
                        except Exception as e:
                            print(f"❌ Error reading resource: {e}")
                            logging.error(f"Resource read error: {e}", exc_info=True)
                    
                    # Handle case where no valid action was determined
                    if not action or action not in ["tool", "resource", "both"]:
                        print("❓ Could not determine appropriate action for your query.")
                
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    logging.error(f"Unexpected error: {e}", exc_info=True)
                    print(f"❌ Unexpected error: {e}")
                    
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print("\nMake sure the server file exists and FastMCP is installed:")
        print("  pip install fastmcp yfinance")
        print(f"  Ensure {server_path} exists in the current directory")

def main():
    """Run the enhanced LLM-powered async demo"""
    print("🤖 Enhanced LLM-Powered MCP Client")
    print("=" * 50)
    print(f"Selected LLM Provider: {LLM_PROVIDER}")
    
    print("\n📋 Setup Instructions:")
    print("Available LLM providers:")
    print("  • OpenAI (GPT-4o-mini): Set OPENAI_API_KEY")
    print("  • Anthropic (Claude 3.5 Sonnet): Set ANTHROPIC_API_KEY")
    print("  • Google Gemini: Set GEMINI_API_KEY")
    print("  • Ollama (Local): Install and run Ollama")
    print("  • AWS Bedrock: Configure AWS credentials")
    
    print("=" * 50)
    
    asyncio.run(run_enhanced_demo())

if __name__ == '__main__':
    main()