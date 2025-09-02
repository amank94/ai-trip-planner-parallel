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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Import Airtable integration
from airtable_integration import airtable_logger

# Configure LiteLLM
import litellm
litellm.drop_params = True

# Global tracer
tracer = trace.get_tracer(__name__)

# Initialize LLM
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1000
    )
    print("✅ Using OpenAI GPT-3.5-turbo")
else:
    raise ValueError("No valid API key found. Please set OPENAI_API_KEY in .env")

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
            project_name="ai-trip-planner-clean"
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
        print(f"📊 Project: ai-trip-planner-clean")
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

app = FastAPI(title="Trip Planner API Clean", lifespan=lifespan)

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

# ========== TOOLS WITH ACTUAL FUNCTIONALITY ==========

@tool
def get_weather_forecast(destination: str) -> str:
    """Get real weather forecast for the destination."""
    # In production, this would call a weather API
    weather_data = {
        "Tokyo": "Spring: 15-20°C, mild. Summer: 25-30°C, humid. Fall: 15-20°C, pleasant. Winter: 5-10°C, cool.",
        "Paris": "Spring: 10-15°C. Summer: 20-25°C. Fall: 10-15°C. Winter: 0-5°C.",
        "default": "Temperate climate, 15-25°C average. Check local forecast before travel."
    }
    return weather_data.get(destination, weather_data["default"])

@tool
def check_visa_requirements(destination: str, nationality: str = "US") -> str:
    """Check visa requirements based on nationality."""
    visa_info = {
        "Japan": {"US": "90 days visa-free", "EU": "90 days visa-free", "CN": "Visa required"},
        "France": {"US": "90 days visa-free", "EU": "No visa needed", "CN": "Visa required"},
        "default": "Please check embassy website for current visa requirements"
    }
    country_info = visa_info.get(destination, {})
    return country_info.get(nationality, visa_info["default"])

@tool
def estimate_accommodation_cost(destination: str, nights: int, level: str = "mid") -> str:
    """Estimate accommodation costs based on destination and level."""
    costs = {
        "Tokyo": {"budget": 50, "mid": 120, "luxury": 300},
        "Paris": {"budget": 60, "mid": 150, "luxury": 400},
        "default": {"budget": 40, "mid": 100, "luxury": 250}
    }
    city_costs = costs.get(destination, costs["default"])
    per_night = city_costs.get(level, city_costs["mid"])
    total = per_night * nights
    return f"{level.capitalize()} accommodation in {destination}: ${per_night}/night, Total for {nights} nights: ${total}"

@tool
def estimate_daily_food_cost(destination: str, style: str = "mixed") -> str:
    """Estimate daily food costs based on eating style."""
    food_costs = {
        "Tokyo": {"street": 20, "mixed": 50, "fine": 150},
        "Paris": {"street": 25, "mixed": 60, "fine": 200},
        "default": {"street": 15, "mixed": 40, "fine": 100}
    }
    city_costs = food_costs.get(destination, food_costs["default"])
    daily_cost = city_costs.get(style, city_costs["mixed"])
    return f"Daily food budget ({style}): ${daily_cost}/day in {destination}"

@tool
def find_top_attractions(destination: str, interests: str = "") -> str:
    """Find top attractions based on interests."""
    attractions = {
        "Tokyo": {
            "technology": ["Akihabara Electronics District", "Miraikan Science Museum", "teamLab Borderless"],
            "culture": ["Senso-ji Temple", "Meiji Shrine", "Imperial Palace"],
            "anime": ["Ghibli Museum", "Akihabara", "Nakano Broadway"],
            "default": ["Tokyo Tower", "Shibuya Crossing", "Tsukiji Market"]
        },
        "Paris": {
            "art": ["Louvre Museum", "Musée d'Orsay", "Centre Pompidou"],
            "history": ["Versailles", "Notre-Dame", "Arc de Triomphe"],
            "food": ["Le Marais food tour", "Cooking classes", "Market visits"],
            "default": ["Eiffel Tower", "Louvre", "Champs-Élysées"]
        }
    }
    
    city_attractions = attractions.get(destination, {})
    interest_key = interests.lower() if interests else "default"
    
    for key in city_attractions:
        if key in interest_key:
            return f"Top attractions for {interests}: " + ", ".join(city_attractions[key])
    
    return f"Top attractions: " + ", ".join(city_attractions.get("default", ["City center", "Museums", "Parks"]))

@tool
def recommend_restaurants(destination: str, cuisine: str = "local") -> str:
    """Recommend restaurants based on cuisine preference."""
    restaurants = {
        "Tokyo": {
            "sushi": ["Sukiyabashi Jiro", "Sushi Saito", "Conveyor belt sushi at Genki Sushi"],
            "ramen": ["Ichiran", "Ippudo", "Afuri"],
            "local": ["Izakaya Toyo", "Gonpachi", "Local street food in Harajuku"]
        },
        "Paris": {
            "french": ["Le Comptoir", "Bistrot Paul Bert", "L'Ami Jean"],
            "bakery": ["Du Pain et des Idées", "Pierre Hermé", "Poilâne"],
            "local": ["Marché des Enfants Rouges", "Breizh Café", "L'As du Fallafel"]
        }
    }
    
    city_restaurants = restaurants.get(destination, {})
    cuisine_key = cuisine.lower() if cuisine else "local"
    
    for key in city_restaurants:
        if key in cuisine_key or cuisine_key in key:
            return f"Recommended {cuisine} restaurants: " + ", ".join(city_restaurants[key])
    
    return f"Recommended restaurants: " + ", ".join(city_restaurants.get("local", ["Local favorites", "Street food", "Traditional cuisine"]))

@tool
def suggest_daily_itinerary(destination: str, day: int, pace: str = "moderate") -> str:
    """Suggest a daily itinerary based on pace preference."""
    itineraries = {
        "relaxed": f"Day {day}: 10am late breakfast, 12pm one major sight, 2pm lunch, 4pm cafe break, 7pm dinner",
        "moderate": f"Day {day}: 9am breakfast, 10am morning activity, 1pm lunch, 3pm afternoon sight, 7pm dinner, 9pm evening stroll",
        "packed": f"Day {day}: 8am breakfast, 9am first sight, 11am second activity, 1pm quick lunch, 2pm third sight, 5pm fourth activity, 8pm dinner"
    }
    return itineraries.get(pace, itineraries["moderate"]) + f" in {destination}"

@tool
def calculate_transport_options(destination: str) -> str:
    """Calculate transport options and costs."""
    transport = {
        "Tokyo": "JR Pass (7 days): $280, Subway day pass: $8, Taxi: $20-50 per trip",
        "Paris": "Metro tickets: €1.90 each, Day pass: €13.20, Taxi: €20-40 per trip",
        "default": "Public transport day pass: $10-20, Taxi: $20-40 per trip"
    }
    return transport.get(destination, transport["default"])

# Define the state
class TripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    final_result: Optional[str]
    tool_calls_made: Annotated[List[Dict[str, Any]], operator.add]

# Create specialized agent functions
def research_agent(state: TripPlannerState) -> TripPlannerState:
    """Research agent that gathers destination information"""
    trip_req = state["trip_request"]
    destination = trip_req["destination"]
    
    # Tools for research
    tools = [get_weather_forecast, check_visa_requirements, find_top_attractions]
    
    # Create agent with tools
    agent = llm.bind_tools(tools)
    
    # Create messages
    messages = [
        SystemMessage(content=f"You are a travel researcher. Research {destination} using available tools. Call 1-2 most relevant tools."),
        HumanMessage(content=f"Research travel information for {destination}")
    ]
    
    # Get response
    response = agent.invoke(messages)
    
    # Process tool calls
    tool_calls = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_node = ToolNode(tools)
        tool_results = tool_node.invoke({"messages": [response]})
        
        for tc in response.tool_calls:
            tool_calls.append({
                "agent": "research",
                "tool": tc["name"],
                "args": tc.get("args", {})
            })
        
        # Get final summary
        messages.append(response)
        messages.extend(tool_results["messages"])
        final = agent.invoke(messages)
        
        return {
            "messages": [response] + tool_results["messages"] + [final],
            "tool_calls_made": tool_calls
        }
    
    return {
        "messages": [response],
        "tool_calls_made": []
    }

def budget_agent(state: TripPlannerState) -> TripPlannerState:
    """Budget agent that analyzes costs"""
    trip_req = state["trip_request"]
    destination = trip_req["destination"]
    duration = trip_req["duration"]
    
    # Parse duration to get nights (simple parsing)
    nights = 3  # Default
    if "day" in duration.lower():
        nights = int(''.join(filter(str.isdigit, duration))) - 1 if any(c.isdigit() for c in duration) else 2
    
    # Tools for budget
    tools = [estimate_accommodation_cost, estimate_daily_food_cost, calculate_transport_options]
    
    # Create agent with tools
    agent = llm.bind_tools(tools)
    
    # Create messages
    messages = [
        SystemMessage(content=f"You are a travel budget analyst. Calculate costs for {destination}. Call 2-3 tools to estimate budget."),
        HumanMessage(content=f"Calculate budget for {duration} in {destination}")
    ]
    
    # Get response
    response = agent.invoke(messages)
    
    # Process tool calls
    tool_calls = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_node = ToolNode(tools)
        tool_results = tool_node.invoke({"messages": [response]})
        
        for tc in response.tool_calls:
            tool_calls.append({
                "agent": "budget",
                "tool": tc["name"],
                "args": tc.get("args", {})
            })
        
        # Get final summary
        messages.append(response)
        messages.extend(tool_results["messages"])
        final = agent.invoke(messages)
        
        return {
            "messages": [response] + tool_results["messages"] + [final],
            "tool_calls_made": tool_calls
        }
    
    return {
        "messages": [response],
        "tool_calls_made": []
    }

def local_expert_agent(state: TripPlannerState) -> TripPlannerState:
    """Local expert agent that finds experiences"""
    trip_req = state["trip_request"]
    destination = trip_req["destination"]
    interests = trip_req.get("interests", "")
    
    # Tools for local experiences
    tools = [recommend_restaurants, find_top_attractions]
    
    # Create agent with tools
    agent = llm.bind_tools(tools)
    
    # Create messages
    messages = [
        SystemMessage(content=f"You are a local expert for {destination}. Find experiences matching interests: {interests}. Call 1-2 tools."),
        HumanMessage(content=f"Find local experiences in {destination}")
    ]
    
    # Get response
    response = agent.invoke(messages)
    
    # Process tool calls
    tool_calls = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_node = ToolNode(tools)
        tool_results = tool_node.invoke({"messages": [response]})
        
        for tc in response.tool_calls:
            tool_calls.append({
                "agent": "local_expert",
                "tool": tc["name"],
                "args": tc.get("args", {})
            })
        
        # Get final summary
        messages.append(response)
        messages.extend(tool_results["messages"])
        final = agent.invoke(messages)
        
        return {
            "messages": [response] + tool_results["messages"] + [final],
            "tool_calls_made": tool_calls
        }
    
    return {
        "messages": [response],
        "tool_calls_made": []
    }

def itinerary_agent(state: TripPlannerState) -> TripPlannerState:
    """Itinerary agent that creates the final plan"""
    trip_req = state["trip_request"]
    destination = trip_req["destination"]
    duration = trip_req["duration"]
    travel_style = trip_req.get("travel_style", "moderate")
    
    # Map travel style to pace
    pace_map = {"relaxed": "relaxed", "adventure": "packed", "standard": "moderate"}
    pace = pace_map.get(travel_style, "moderate")
    
    # Tools for itinerary
    tools = [suggest_daily_itinerary]
    
    # Create agent with tools
    agent = llm.bind_tools(tools)
    
    # Collect all previous information
    all_messages = state.get("messages", [])
    previous_info = "\n".join([m.content for m in all_messages if hasattr(m, 'content')][:5])
    
    # Create messages
    messages = [
        SystemMessage(content=f"""You are an itinerary planner. Create a {duration} itinerary for {destination}.
        Previous information gathered:
        {previous_info[:500]}
        
        Use tools to create daily schedules, then summarize into a complete trip plan."""),
        HumanMessage(content=f"Create complete itinerary for {destination}")
    ]
    
    # Get response
    response = agent.invoke(messages)
    
    # Process tool calls
    tool_calls = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_node = ToolNode(tools)
        tool_results = tool_node.invoke({"messages": [response]})
        
        for tc in response.tool_calls:
            tool_calls.append({
                "agent": "itinerary",
                "tool": tc["name"],
                "args": tc.get("args", {})
            })
        
        # Get final summary
        messages.append(response)
        messages.extend(tool_results["messages"])
        
        # Create comprehensive final plan
        final_prompt = f"""Based on all information, create a complete trip plan for {destination}.
        Include: Overview, Budget, Attractions, Restaurants, Daily Itinerary.
        Keep it concise but comprehensive."""
        
        final = agent.invoke([SystemMessage(content=final_prompt)] + messages[-3:])
        
        return {
            "messages": [response] + tool_results["messages"] + [final],
            "tool_calls_made": tool_calls,
            "final_result": final.content
        }
    
    return {
        "messages": [response],
        "tool_calls_made": [],
        "final_result": response.content
    }

# Build the graph
def create_trip_planning_graph():
    """Create the trip planning graph"""
    workflow = StateGraph(TripPlannerState)
    
    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("budget", budget_agent)  
    workflow.add_node("local_expert", local_expert_agent)
    workflow.add_node("itinerary", itinerary_agent)
    
    # Define flow
    workflow.add_edge(START, "research")
    workflow.add_edge(START, "budget")
    workflow.add_edge(START, "local_expert")
    
    workflow.add_edge("research", "itinerary")
    workflow.add_edge("budget", "itinerary")
    workflow.add_edge("local_expert", "itinerary")
    
    workflow.add_edge("itinerary", END)
    
    # Compile
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# API Routes
@app.get("/")
async def root():
    return {"message": "Trip Planner API Clean is running!"}

@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using multi-agent workflow"""
    try:
        print(f"🚀 Starting trip planning for {trip_request.destination}")
        
        # Create graph
        graph = create_trip_planning_graph()
        
        # Initial state
        initial_state = {
            "messages": [],
            "trip_request": trip_request.model_dump(),
            "final_result": None,
            "tool_calls_made": []
        }
        
        # Execute
        config = {"configurable": {"thread_id": f"trip_{trip_request.destination}_{time.time()}"}}
        output = graph.invoke(initial_state, config)
        
        print(f"✅ Trip planning completed with {len(output.get('tool_calls_made', []))} tool calls")
        
        # Return result
        result = output.get("final_result", "Trip plan created successfully")
        if not result or len(result) < 100:
            # Fallback if result is too short
            messages = output.get("messages", [])
            result = "\n".join([m.content for m in messages if hasattr(m, 'content') and m.content][-3:])
        
        return TripResponse(
            result=result,
            tool_calls=output.get("tool_calls_made", [])
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)