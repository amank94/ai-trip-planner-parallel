from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import time
import random
from datetime import datetime

# Load environment variables
load_dotenv()

# Arize and tracing imports
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Import Airtable integration
from airtable_integration import airtable_logger

# Configure LiteLLM
import litellm
litellm.drop_params = True

# Global tracer
tracer = trace.get_tracer(__name__)

# Initialize LLM - use OpenAI or fallback to Groq
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=2000
    )
    print("✅ Using OpenAI GPT-3.5-turbo")
elif os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "your_groq_api_key_here":
    llm = ChatOpenAI(
        model="groq/llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.7
    )
    print("✅ Using Groq Llama3")
else:
    raise ValueError("No valid API key found. Please set OPENAI_API_KEY or GROQ_API_KEY in .env")

# Initialize search tools
search_tools = []
if os.getenv("TAVILY_API_KEY"):
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        search_tools.append(TavilySearchResults(max_results=5))
        print("✅ Tavily search tool initialized")
    except:
        print("⚠️ Tavily API key provided but initialization failed")

# Initialize Arize tracing
def setup_tracing():
    """Set up Arize tracing for the application"""
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        
        if not space_id or not api_key:
            print("⚠️ Arize credentials not configured. Tracing disabled.")
            return None
            
        # Register with Arize
        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name="ai-trip-planner-supervisor-hierarchy"
        )
        
        # Instrument LangChain
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            include_tools=True,
            include_chains=True,
            include_agents=True
        )
        
        # Instrument LiteLLM
        LiteLLMInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )
        
        print("✅ Arize tracing initialized successfully")
        print(f"📊 Project: ai-trip-planner-supervisor-hierarchy")
        print(f"🔗 Space ID: {space_id[:10]}...")
        print(f"🌐 View traces at: https://app.arize.com/")
        
        return tracer_provider
        
    except Exception as e:
        print(f"⚠️ Arize tracing setup failed: {str(e)}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    setup_tracing()
    yield

app = FastAPI(title="Trip Planner API with Proper Tracing Hierarchy", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None

class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []

# ========== RESEARCH TOOLS ==========

@tool
def search_destination_info(destination: str) -> str:
    """Search for general destination information using web search."""
    if search_tools:
        search_result = search_tools[0].invoke(f"{destination} travel guide attractions weather culture")
        return f"Search results for {destination}: {str(search_result)[:500]}"
    return f"General info for {destination}: Popular tourist destination with varied attractions and cultural sites."

@tool
def get_destination_weather(destination: str) -> str:
    """Get weather information for the destination."""
    time.sleep(random.uniform(0.5, 1.5))
    return f"Weather in {destination}: Average temperature 20-25°C, moderate rainfall. Best time: April-October."

@tool
def research_visa_requirements(destination: str) -> str:
    """Research visa requirements for the destination."""
    time.sleep(random.uniform(0.5, 1.5))
    countries = {"Japan": "Visa on arrival for US citizens (30 days), fee: $50",
                 "France": "No visa required for US citizens (90 days)",
                 "Thailand": "Visa on arrival (30 days), fee: $35"}
    return f"Visa for {destination}: {countries.get(destination, 'Check embassy requirements')}"

@tool
def find_local_events(destination: str) -> str:
    """Find local events and festivals happening at the destination."""
    time.sleep(random.uniform(0.5, 1.5))
    events = ["Cultural Festival (weekends)", "Night Market (daily)", "Art Exhibition (various locations)"]
    return f"Events in {destination}: {', '.join(events)}"

# ========== BUDGET TOOLS ==========

@tool
def calculate_accommodation_cost(destination: str, duration: str, level: str = "mid-range") -> str:
    """Calculate accommodation costs for the trip."""
    costs = {"budget": "$30-50/night", "mid-range": "$80-150/night", "luxury": "$200-500/night"}
    return f"Accommodation in {destination} ({level}) for {duration}: {costs.get(level, costs['mid-range'])}"

@tool
def calculate_food_cost(destination: str, duration: str, style: str = "mixed") -> str:
    """Calculate food and dining costs."""
    costs = {"street_food": "$15-25/day", "mixed": "$40-60/day", "fine_dining": "$100-200/day"}
    return f"Food costs in {destination} ({style}) for {duration}: {costs.get(style, costs['mixed'])}"

@tool
def calculate_transport_cost(destination: str, duration: str, mode: str = "public") -> str:
    """Calculate transportation costs."""
    costs = {"public": "$5-15/day", "taxi": "$30-50/day", "rental_car": "$40-80/day"}
    return f"Transport in {destination} ({mode}) for {duration}: {costs.get(mode, costs['public'])}"

@tool
def get_attraction_prices(destination: str, attractions: List[str] = None) -> str:
    """Get entrance fees for specific attractions."""
    if not attractions:
        attractions = ["Major Museum", "Historical Site", "Theme Park"]
    prices = [f"{attr}: ${random.randint(10, 50)}" for attr in attractions]
    return f"Attraction prices in {destination}: {', '.join(prices)}"

# ========== LOCAL EXPERIENCES TOOLS ==========

@tool
def find_local_restaurants(destination: str, cuisine_type: Optional[str] = None) -> str:
    """Find local restaurants and eateries."""
    cuisine = cuisine_type or "local"
    return f"Restaurants in {destination} ({cuisine}): 1. Casa Local (authentic), 2. Street Food Market (budget), 3. Hidden Gem Bistro"

@tool
def find_cultural_activities(destination: str, interests: Optional[str] = None) -> str:
    """Find cultural activities and experiences."""
    return f"Cultural activities in {destination}: Traditional dance show, Cooking class, Artisan workshop, Local festival"

@tool
def find_hidden_gems(destination: str, category: Optional[str] = None) -> str:
    """Find off-the-beaten-path attractions and hidden gems."""
    return f"Hidden gems in {destination}: Secret garden cafe, Underground art gallery, Local's favorite viewpoint"

@tool
def get_local_customs(destination: str) -> str:
    """Get information about local customs and etiquette."""
    return f"Customs in {destination}: Tipping 10-15%, Remove shoes indoors, Greet with handshake, Dress modestly at temples"

# ========== ITINERARY TOOLS ==========

@tool
def create_daily_schedule(destination: str, day_number: int, interests: List[str] = None) -> str:
    """Create a detailed daily schedule."""
    return f"Day {day_number} in {destination}: 9AM Breakfast, 10AM Museum visit, 1PM Local lunch, 3PM Walking tour, 7PM Dinner"

@tool
def calculate_travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Calculate travel time between locations."""
    time.sleep(random.uniform(0.5, 1.5))
    return f"Travel from {from_location} to {to_location} by {mode}: {random.randint(15, 60)} minutes"

@tool
def book_recommendations(destination: str, duration: str) -> str:
    """Get recommendations for what needs advance booking."""
    return f"Must book in {destination}: Popular restaurants (2 weeks ahead), Tours (1 week ahead), Shows (3 days ahead)"

@tool
def create_packing_list(destination: str, duration: str, activities: List[str] = None) -> str:
    """Create a customized packing list."""
    return f"Packing for {destination}: Weather-appropriate clothing, Comfortable walking shoes, Camera, Adapter, Sunscreen"

# Define the state for our graph
class TripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research_data: Optional[str]
    budget_data: Optional[str]
    local_data: Optional[str]
    final_result: Optional[str]
    tool_calls_made: Annotated[List[Dict[str, Any]], operator.add]
    supervisor_decision: Optional[str]

# ========== HELPER FUNCTION FOR AGENT EXECUTION ==========

def execute_agent_with_tools(agent_name: str, agent_tools: List, prompt: str, messages: List[BaseMessage], trip_req: Dict) -> Dict:
    """Execute an agent with proper tracing hierarchy"""
    
    # Create agent span as child of current span
    with tracer.start_as_current_span(f"{agent_name}_agent") as agent_span:
        try:
            destination = trip_req["destination"]
            
            # Set span attributes
            agent_span.set_attribute("destination", destination)
            agent_span.set_attribute("agent", agent_name)
            agent_span.set_attribute("operation", "agent_execution")
            
            print(f"🤖 {agent_name} agent starting for {destination}")
            
            # Create LLM with tools
            agent_llm = llm.bind_tools(agent_tools)
            
            # Invoke LLM
            response = agent_llm.invoke(messages)
            
            # Track tool calls
            tool_calls = []
            result_messages = [response]
            
            # Process tool calls if any
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_calls.append({
                        "agent": agent_name,
                        "tool": tool_call["name"],
                        "args": tool_call.get("args", {})
                    })
                
                # Execute tools with individual tool spans
                tool_node = ToolNode(agent_tools)
                
                # Create tool execution span
                with tracer.start_as_current_span(f"{agent_name}_tool_execution") as tool_span:
                    tool_span.set_attribute("tool_count", len(response.tool_calls))
                    tool_span.set_attribute("agent", agent_name)
                    
                    for i, tool_call in enumerate(response.tool_calls):
                        with tracer.start_as_current_span(f"tool_{tool_call['name']}") as individual_tool_span:
                            individual_tool_span.set_attribute("tool_name", tool_call["name"])
                            individual_tool_span.set_attribute("tool_args", str(tool_call.get("args", {})))
                            individual_tool_span.set_attribute("agent", agent_name)
                    
                    tool_results = tool_node.invoke({"messages": [response]})
                    result_messages.extend(tool_results["messages"])
                    
                    tool_span.set_status(Status(StatusCode.OK))
                
                # Get final summary from LLM
                final_response = agent_llm.invoke(messages + result_messages)
                result_messages.append(final_response)
            
            # Extract agent data
            agent_data = result_messages[-1].content if result_messages else f"{agent_name} agent completed"
            
            print(f"✅ {agent_name} agent completed with {len(tool_calls)} tool calls")
            agent_span.set_attribute("tool_calls_count", len(tool_calls))
            agent_span.set_status(Status(StatusCode.OK))
            
            return {
                "messages": result_messages,
                f"{agent_name}_data": agent_data,
                "tool_calls_made": tool_calls
            }
            
        except Exception as e:
            print(f"❌ {agent_name} agent error: {str(e)}")
            agent_span.set_status(Status(StatusCode.ERROR, str(e)))
            agent_span.record_exception(e)
            return {
                "messages": [HumanMessage(content=f"{agent_name} failed: {str(e)}")],
                f"{agent_name}_data": f"{agent_name} failed: {str(e)}"
            }

# ========== SUPERVISOR AGENT ==========

def supervisor_agent(state: TripPlannerState) -> TripPlannerState:
    """Supervisor agent that orchestrates all other agents"""
    with tracer.start_as_current_span("supervisor_agent") as supervisor_span:
        try:
            trip_req = state["trip_request"]
            destination = trip_req["destination"]
            duration = trip_req["duration"]
            
            # Set supervisor span attributes
            supervisor_span.set_attribute("destination", destination)
            supervisor_span.set_attribute("duration", duration)
            supervisor_span.set_attribute("agent", "supervisor")
            supervisor_span.set_attribute("operation", "orchestration")
            
            print(f"👨‍💼 Supervisor agent orchestrating trip planning for {destination}")
            
            # Initialize results
            all_messages = []
            all_tool_calls = []
            agent_results = {}
            
            # Execute research agent
            research_tools = [search_destination_info, get_destination_weather, research_visa_requirements, find_local_events]
            research_prompt = f"""You are a destination research specialist. 
            Research {destination} and gather essential travel information.
            Use 1-2 of your available tools to get key information about:
            - General destination overview
            - Weather conditions
            - Visa requirements
            - Local events
            
            Be selective and efficient - only call the most relevant tools."""
            
            research_messages = [
                SystemMessage(content=research_prompt),
                HumanMessage(content=f"Research {destination} for a trip")
            ]
            
            research_result = execute_agent_with_tools(
                "research", research_tools, research_prompt, research_messages, trip_req
            )
            all_messages.extend(research_result.get("messages", []))
            all_tool_calls.extend(research_result.get("tool_calls_made", []))
            agent_results["research_data"] = research_result.get("research_data", "")
            
            # Execute budget agent
            budget_tools = [calculate_accommodation_cost, calculate_food_cost, calculate_transport_cost, get_attraction_prices]
            budget_prompt = f"""You are a travel budget analyst.
            Analyze costs for a {duration} trip to {destination} with budget: {trip_req.get('budget', 'flexible')}.
            Use 1-2 of your available tools to calculate:
            - Accommodation costs
            - Food expenses
            - Transportation costs
            - Attraction fees
            
            Focus on the most important cost categories for this trip."""
            
            budget_messages = [
                SystemMessage(content=budget_prompt),
                HumanMessage(content=f"Calculate budget for {destination} trip")
            ]
            
            budget_result = execute_agent_with_tools(
                "budget", budget_tools, budget_prompt, budget_messages, trip_req
            )
            all_messages.extend(budget_result.get("messages", []))
            all_tool_calls.extend(budget_result.get("tool_calls_made", []))
            agent_results["budget_data"] = budget_result.get("budget_data", "")
            
            # Execute local experiences agent
            local_tools = [find_local_restaurants, find_cultural_activities, find_hidden_gems, get_local_customs]
            local_prompt = f"""You are a local experiences curator.
            Find authentic experiences in {destination} for someone interested in: {trip_req.get('interests', 'general exploration')}.
            Use 1-2 of your available tools to discover:
            - Local restaurants
            - Cultural activities
            - Hidden gems
            - Local customs
            
            Focus on unique, authentic experiences that match the traveler's interests."""
            
            local_messages = [
                SystemMessage(content=local_prompt),
                HumanMessage(content=f"Find local experiences in {destination}")
            ]
            
            local_result = execute_agent_with_tools(
                "local", local_tools, local_prompt, local_messages, trip_req
            )
            all_messages.extend(local_result.get("messages", []))
            all_tool_calls.extend(local_result.get("tool_calls_made", []))
            agent_results["local_data"] = local_result.get("local_data", "")
            
            # Execute itinerary agent
            itinerary_tools = [create_daily_schedule, calculate_travel_time, book_recommendations, create_packing_list]
            itinerary_prompt = f"""You are a trip itinerary planner.
            Create a {duration} itinerary for {destination} ({trip_req.get('travel_style', 'standard')} style).
            
            Available information:
            Research: {agent_results.get('research_data', '')[:300]}
            Budget: {agent_results.get('budget_data', '')[:300]}
            Local: {agent_results.get('local_data', '')[:300]}
            
            Use 1-2 of your tools to create:
            - Daily schedules
            - Travel logistics
            - Booking recommendations
            - Packing list
            
            Focus on creating a practical, enjoyable itinerary."""
            
            itinerary_messages = [
                SystemMessage(content=itinerary_prompt),
                HumanMessage(content=f"Create itinerary for {destination}")
            ]
            
            itinerary_result = execute_agent_with_tools(
                "itinerary", itinerary_tools, itinerary_prompt, itinerary_messages, trip_req
            )
            all_messages.extend(itinerary_result.get("messages", []))
            all_tool_calls.extend(itinerary_result.get("tool_calls_made", []))
            
            # Create final comprehensive trip plan
            final_result = f"""### Comprehensive Trip Plan for {destination}
            
**Duration:** {duration}
**Travel Style:** {trip_req.get('travel_style', 'standard')}

**Research Information:**
{agent_results.get('research_data', 'No research data available')[:500]}

**Budget Analysis:**
{agent_results.get('budget_data', 'No budget data available')[:500]}

**Local Experiences:**
{agent_results.get('local_data', 'No local data available')[:500]}

**Detailed Itinerary:**
{itinerary_result.get('itinerary_data', 'Itinerary planning completed')[:500]}

**Total Tools Used:** {len(all_tool_calls)} across all agents
**Agents Coordinated:** Research, Budget, Local Experiences, Itinerary
"""
            
            print(f"✅ Supervisor agent completed coordination with {len(all_tool_calls)} total tool calls")
            supervisor_span.set_attribute("total_tool_calls", len(all_tool_calls))
            supervisor_span.set_attribute("agents_coordinated", 4)
            supervisor_span.set_status(Status(StatusCode.OK))
            
            return {
                "messages": all_messages,
                "final_result": final_result,
                "tool_calls_made": all_tool_calls,
                "supervisor_decision": f"Coordinated {len(all_tool_calls)} tool calls across 4 specialized agents",
                **agent_results
            }
            
        except Exception as e:
            print(f"❌ Supervisor agent error: {str(e)}")
            supervisor_span.set_status(Status(StatusCode.ERROR, str(e)))
            supervisor_span.record_exception(e)
            return {
                "messages": [HumanMessage(content=f"Supervisor coordination failed: {str(e)}")],
                "final_result": f"Trip planning failed: {str(e)}"
            }

# Build the graph with supervisor pattern
def create_supervisor_trip_planning_graph():
    """Create and compile the trip planning graph with supervisor-agent architecture"""
    
    # Create the state graph
    workflow = StateGraph(TripPlannerState)
    
    # Add only the supervisor node - it coordinates everything
    workflow.add_node("supervisor", supervisor_agent)
    
    # Simple linear flow: START -> supervisor -> END
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# API Routes
@app.get("/")
async def root():
    return {"message": "Trip Planner API with Supervisor-Agent Hierarchy is running!"}

@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using supervisor-agent workflow with proper tracing hierarchy"""
    
    # Start main span for the entire trip planning
    with tracer.start_as_current_span("plan_trip") as main_span:
        try:
            # Set main span attributes
            main_span.set_attribute("destination", trip_request.destination)
            main_span.set_attribute("duration", trip_request.duration)
            main_span.set_attribute("budget", trip_request.budget or "flexible")
            main_span.set_attribute("operation", "trip_planning_endpoint")
            main_span.set_attribute("architecture", "supervisor_hierarchy")
            
            # Log to Airtable if available
            trace_id = None
            if airtable_logger:
                trace_id = airtable_logger.log_request(trip_request.model_dump())
            
            # Create the graph
            graph = create_supervisor_trip_planning_graph()
            
            # Prepare initial state
            initial_state = {
                "messages": [],
                "trip_request": trip_request.model_dump(),
                "research_data": None,
                "budget_data": None,
                "local_data": None,
                "final_result": None,
                "tool_calls_made": [],
                "supervisor_decision": None
            }
            
            # Execute the workflow
            config = {
                "configurable": {
                    "thread_id": f"supervised_trip_{trip_request.destination}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            }
            
            print(f"🚀 Starting supervised trip planning for {trip_request.destination} ({trip_request.duration})")
            start_time = time.time()
            
            # Invoke the graph
            output = graph.invoke(initial_state, config)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            print(f"✅ Supervised trip planning completed in {latency_ms:.0f}ms with {len(output.get('tool_calls_made', []))} tool calls")
            
            # Log result to Airtable
            if airtable_logger and output:
                airtable_logger.log_response(
                    trip_request.model_dump(),
                    output.get("final_result", "No result"),
                    output.get("tool_calls_made", [])
                )
            
            # Set success status
            main_span.set_attribute("tool_calls_count", len(output.get("tool_calls_made", [])))
            main_span.set_attribute("latency_ms", latency_ms)
            main_span.set_attribute("supervisor_decision", output.get("supervisor_decision", ""))
            main_span.set_status(Status(StatusCode.OK))
            
            # Return the final result
            return TripResponse(
                result=output.get("final_result", "Trip planning completed."),
                tool_calls=output.get("tool_calls_made", [])
            )
            
        except Exception as e:
            print(f"❌ Trip planning error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Set error status on span
            main_span.set_status(Status(StatusCode.ERROR, str(e)))
            main_span.record_exception(e)
            
            # Log error to Airtable
            if airtable_logger:
                airtable_logger.log_error(trip_request.model_dump(), str(e))
            
            raise HTTPException(status_code=500, detail=f"Trip planning failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)