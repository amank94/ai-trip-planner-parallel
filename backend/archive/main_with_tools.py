from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import asyncio
import random
import time

# Load environment variables from .env file
load_dotenv()

# Airtable integration
from airtable_integration import airtable_logger

# Arize and tracing imports
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation import using_prompt_template

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Configure LiteLLM
import litellm
litellm.drop_params = True  # Drop unsupported parameters automatically

# Initialize Arize tracing
def setup_tracing():
    """Set up Arize tracing for the application"""
    try:
        # Check if required environment variables are set
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        
        if not space_id or not api_key:
            print("⚠️ Arize credentials not configured. Tracing disabled.")
            print("📝 Please set ARIZE_SPACE_ID and ARIZE_API_KEY environment variables.")
            print("📝 Copy backend/env_example.txt to backend/.env and update with your credentials.")
            return None
            
        # Register with Arize
        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name="ai-trip-planner-with-tools-v2"
        )
        
        # Instrument LangChain for comprehensive tracing
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            include_tools=True,
            include_chains=True,
            include_agents=True
        )
        
        # Instrument LiteLLM for LLM call tracing
        LiteLLMInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )
        
        print("✅ Arize tracing initialized successfully")
        print(f"📊 Project: ai-trip-planner-with-tools-v2")
        print(f"🔗 Space ID: {space_id[:8]}...")
        print(f"🌐 View traces at: https://app.arize.com/")
        
        return tracer_provider
        
    except Exception as e:
        print(f"⚠️ Arize tracing setup failed: {str(e)}")
        print("📝 Continuing without tracing - check your ARIZE_SPACE_ID and ARIZE_API_KEY")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Setup tracing before anything else
    setup_tracing()
    yield

app = FastAPI(title="Trip Planner API with Tools V2", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None

class TripResponse(BaseModel):
    result: str
    tool_calls: Optional[List[Dict[str, Any]]] = []  # Track tool calls for evaluation

# Initialize the LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=2000,
    timeout=30
)

# Initialize search tool if available
search_tools = []
if os.getenv("TAVILY_API_KEY"):
    search_tools.append(TavilySearchResults(max_results=5))

# ========== RESEARCH TOOLS ==========

@tool
def search_destination_info(destination: str, query: str) -> str:
    """Search for specific information about a destination.
    
    Args:
        destination: The destination to search about
        query: Specific query (e.g., "weather in summer", "top attractions")
    """
    if search_tools:
        search_tool = search_tools[0]
        results = search_tool.invoke(f"{destination} {query}")
        return f"Search results for '{query}' in {destination}: {str(results)[:500]}"
    return f"Mock search data for {query} in {destination}: Weather is pleasant, attractions include historical sites."

@tool
def get_destination_weather(destination: str, month: Optional[str] = None) -> str:
    """Get weather information for a destination.
    
    Args:
        destination: The destination
        month: Specific month (optional)
    """
    month_text = month or "year-round"
    return f"Weather in {destination} ({month_text}): Average 20-25°C, moderate rainfall, best time April-October"

@tool
def research_visa_requirements(destination: str, nationality: str = "US") -> str:
    """Research visa requirements for a destination.
    
    Args:
        destination: The destination country
        nationality: Traveler's nationality
    """
    return f"Visa for {destination} (from {nationality}): Visa on arrival for 30 days, $50 fee"

@tool
def find_local_events(destination: str, dates: Optional[str] = None) -> str:
    """Find local events and festivals.
    
    Args:
        destination: The destination
        dates: Travel dates (optional)
    """
    return f"Events in {destination}: Cultural Festival (weekends), Night Market (daily), Art Exhibition"

# ========== BUDGET TOOLS ==========

@tool
def calculate_accommodation_cost(destination: str, duration: str, level: str = "mid-range") -> str:
    """Calculate accommodation costs for the trip.
    
    Args:
        destination: The destination
        duration: Duration of stay
        level: Accommodation level (budget/mid-range/luxury)
    """
    costs = {"budget": "$30-50/night", "mid-range": "$80-150/night", "luxury": "$200-500/night"}
    return f"Accommodation in {destination} ({level}) for {duration}: {costs.get(level, costs['mid-range'])}"

@tool
def calculate_food_cost(destination: str, duration: str, style: str = "mixed") -> str:
    """Calculate food and dining costs.
    
    Args:
        destination: The destination
        duration: Trip duration
        style: Dining style (street_food/mixed/fine_dining)
    """
    costs = {"street_food": "$15-25/day", "mixed": "$40-60/day", "fine_dining": "$100-200/day"}
    return f"Food costs in {destination} ({style}) for {duration}: {costs.get(style, costs['mixed'])}"

@tool
def calculate_transport_cost(destination: str, duration: str, mode: str = "public") -> str:
    """Calculate transportation costs.
    
    Args:
        destination: The destination
        duration: Trip duration
        mode: Transport mode (public/taxi/rental_car)
    """
    costs = {"public": "$5-15/day", "taxi": "$30-50/day", "rental_car": "$40-80/day"}
    return f"Transport in {destination} ({mode}) for {duration}: {costs.get(mode, costs['public'])}"

@tool
def get_attraction_prices(destination: str, attractions: List[str]) -> str:
    """Get entrance fees for specific attractions.
    
    Args:
        destination: The destination
        attractions: List of attractions to check
    """
    if not attractions:
        attractions = ["Major Museum", "Historical Site", "Theme Park"]
    prices = [f"{attr}: ${random.randint(10, 50)}" for attr in attractions]
    return f"Attraction prices in {destination}: {', '.join(prices)}"

# ========== LOCAL EXPERIENCES TOOLS ==========

@tool
def find_local_restaurants(destination: str, cuisine_type: Optional[str] = None, budget: Optional[str] = None) -> str:
    """Find local restaurants and eateries.
    
    Args:
        destination: The destination
        cuisine_type: Type of cuisine (optional)
        budget: Budget level (optional)
    """
    cuisine = cuisine_type or "local"
    return f"Restaurants in {destination} ({cuisine}): 1. Casa Local (authentic), 2. Street Food Market (budget), 3. Hidden Gem Bistro"

@tool
def find_cultural_activities(destination: str, interests: Optional[str] = None) -> str:
    """Find cultural activities and experiences.
    
    Args:
        destination: The destination
        interests: Specific interests (optional)
    """
    return f"Cultural activities in {destination}: Traditional dance show, Cooking class, Artisan workshop, Local festival"

@tool
def find_hidden_gems(destination: str, category: Optional[str] = None) -> str:
    """Find off-the-beaten-path attractions and hidden gems.
    
    Args:
        destination: The destination
        category: Category of hidden gems (optional)
    """
    return f"Hidden gems in {destination}: Secret garden cafe, Underground art gallery, Local's favorite viewpoint"

@tool
def get_local_customs(destination: str) -> str:
    """Get information about local customs and etiquette.
    
    Args:
        destination: The destination
    """
    return f"Customs in {destination}: Tipping 10-15%, Remove shoes indoors, Greet with handshake, Dress modestly at temples"

# ========== ITINERARY TOOLS ==========

@tool
def create_daily_schedule(destination: str, day_number: int, interests: List[str]) -> str:
    """Create a detailed daily schedule.
    
    Args:
        destination: The destination
        day_number: Which day of the trip
        interests: List of interests to incorporate
    """
    return f"Day {day_number} in {destination}: 9AM Breakfast, 10AM Museum visit, 1PM Local lunch, 3PM Walking tour, 7PM Dinner"

@tool
def calculate_travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Calculate travel time between locations.
    
    Args:
        from_location: Starting location
        to_location: Destination location
        mode: Transportation mode
    """
    time.sleep(random.randint(1, 3)) # Simulate network delay
    return f"Travel from {from_location} to {to_location} by {mode}: {random.randint(15, 60)} minutes"

@tool
def book_recommendations(destination: str, duration: str, must_book: bool = True) -> str:
    """Get recommendations for what needs advance booking.
    
    Args:
        destination: The destination
        duration: Trip duration
        must_book: Only show must-book items
    """
    return f"Must book in {destination}: Popular restaurants (2 weeks ahead), Tours (1 week ahead), Shows (3 days ahead)"

@tool
def create_packing_list(destination: str, duration: str, activities: List[str]) -> str:
    """Create a customized packing list.
    
    Args:
        destination: The destination
        duration: Trip duration
        activities: Planned activities
    """
    return f"Packing for {destination}: Weather-appropriate clothing, Comfortable walking shoes, Camera, Adapter, Sunscreen"

# Define the state for our graph
class TripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    final_result: Optional[str]
    tool_calls_made: List[Dict[str, Any]]  # Track all tool calls

# Define the supervisor node that can make multiple tool calls
def supervisor_node(state: TripPlannerState) -> TripPlannerState:
    """Supervisor that uses multiple tools to plan trips comprehensively"""
    
    trip_req = state["trip_request"]
    
    supervisor_prompt = f"""
    You are a comprehensive trip planning assistant with access to many specialized tools.
    Plan a {trip_req['duration']} trip to {trip_req['destination']}.
    
    Trip requirements:
    - Destination: {trip_req['destination']}
    - Duration: {trip_req['duration']}
    - Budget: {trip_req.get('budget', 'Flexible')}
    - Interests: {trip_req.get('interests', 'General sightseeing')}
    - Travel style: {trip_req.get('travel_style', 'Standard')}
    
    Use multiple tools to gather comprehensive information:
    1. Research the destination (weather, visa, events)
    2. Calculate detailed budget breakdown (accommodation, food, transport)
    3. Find local experiences (restaurants, cultural activities, hidden gems)
    4. Create a day-by-day itinerary with specific recommendations
    
    Make sure to use various tools to provide a complete trip plan.
    """
    
    messages = [SystemMessage(content=supervisor_prompt)]
    messages.extend(state.get("messages", []))
    
    # Bind all tools to the LLM
    all_tools = [
        # Research tools
        search_destination_info,
        get_destination_weather,
        research_visa_requirements,
        find_local_events,
        # Budget tools
        calculate_accommodation_cost,
        calculate_food_cost,
        calculate_transport_cost,
        get_attraction_prices,
        # Local experience tools
        find_local_restaurants,
        find_cultural_activities,
        find_hidden_gems,
        get_local_customs,
        # Itinerary tools
        create_daily_schedule,
        calculate_travel_time,
        book_recommendations,
        create_packing_list
    ]
    
    if search_tools:
        all_tools.extend(search_tools)
    
    supervisor_llm = llm.bind_tools(all_tools)
    
    response = supervisor_llm.invoke(messages)
    
    return {
        "messages": [response]
    }

# Check if supervisor should call tools or finish
def should_continue(state: TripPlannerState) -> str:
    """Determine if supervisor should call tools or finish"""
    messages = state.get("messages", [])
    if not messages:
        return "tools"
    
    last_message = messages[-1]
    
    # If last message has tool calls, go to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Track the tool calls
        if "tool_calls_made" not in state:
            state["tool_calls_made"] = []
        for tool_call in last_message.tool_calls:
            state["tool_calls_made"].append({
                "tool": tool_call["name"],
                "args": tool_call.get("args", {})
            })
        return "tools"
    
    # If we have a comprehensive response, we're done
    if hasattr(last_message, 'content') and last_message.content:
        # Check if we have enough content for a complete itinerary
        content_lower = last_message.content.lower()
        if any(word in content_lower for word in ["day", "itinerary", "schedule", "morning", "afternoon"]):
            return END
    
    # If we've made at least some tool calls, we can finish
    if state.get("tool_calls_made") and len(state.get("tool_calls_made", [])) > 3:
        return END
    
    # Otherwise continue with tools
    return "tools"

# Build the graph with multiple tool calling capability
def create_trip_planning_graph():
    """Create and compile the trip planning graph with multiple tool calls"""
    
    # Create the state graph
    workflow = StateGraph(TripPlannerState)
    
    # Add nodes - supervisor and tools
    workflow.add_node("supervisor", supervisor_node)
    
    # Create tool node with all tools
    all_tools = [
        # Research tools
        search_destination_info,
        get_destination_weather,
        research_visa_requirements,
        find_local_events,
        # Budget tools
        calculate_accommodation_cost,
        calculate_food_cost,
        calculate_transport_cost,
        get_attraction_prices,
        # Local experience tools
        find_local_restaurants,
        find_cultural_activities,
        find_hidden_gems,
        get_local_customs,
        # Itinerary tools
        create_daily_schedule,
        calculate_travel_time,
        book_recommendations,
        create_packing_list
    ]
    
    if search_tools:
        all_tools.extend(search_tools)
    
    tool_node = ToolNode(all_tools)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    
    # Conditional edge from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Tools always go back to supervisor
    workflow.add_edge("tools", "supervisor")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# API Routes
@app.get("/")
async def root():
    return {"message": "Trip Planner API with Multiple Tools is running!"}

@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using LangGraph workflow with multiple tool calls"""
    try:
        # Log to Airtable if available
        if airtable_logger:
            airtable_logger.log_request(trip_request.model_dump())
        
        # Create the graph
        graph = create_trip_planning_graph()
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=f"Plan a comprehensive trip to {trip_request.destination} for {trip_request.duration}")],
            "trip_request": trip_request.model_dump(),
            "final_result": None,
            "tool_calls_made": []
        }
        
        # Execute the workflow
        config = {"configurable": {"thread_id": f"trip_{trip_request.destination}_{trip_request.duration}_{random.randint(1000, 9999)}"}}
        
        print(f"🚀 Starting trip planning for {trip_request.destination} ({trip_request.duration})")
        
        output = graph.invoke(initial_state, config)
        
        print(f"✅ Trip planning completed with {len(output.get('tool_calls_made', []))} tool calls")
        
        # Log result to Airtable
        if airtable_logger and output:
            airtable_logger.log_response(
                trip_request.model_dump(),
                output.get("messages", [])[-1].content if output.get("messages") else "No result",
                output.get("tool_calls_made", [])
            )
        
        # Return the final result with tool call tracking
        if output and output.get("messages") and len(output.get("messages")) > 0:
            last_message = output.get("messages")[-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            return TripResponse(
                result=content,
                tool_calls=output.get("tool_calls_made", [])
            )
        
        return TripResponse(
            result="Trip planning completed but no detailed results available.",
            tool_calls=output.get("tool_calls_made", [])
        )
        
    except Exception as e:
        print(f"❌ Trip planning error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Log error to Airtable
        if airtable_logger:
            airtable_logger.log_error(trip_request.model_dump(), str(e))
        
        raise HTTPException(status_code=500, detail=f"Trip planning failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-with-tools-v2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)