from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

# Arize and tracing imports
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation import using_prompt_template
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
from langchain_community.tools.tavily_search import TavilySearchResults

# Configure LiteLLM
import litellm
litellm.drop_params = True  # Drop unsupported parameters automatically

# Global tracer to use throughout the application
tracer = trace.get_tracer(__name__)

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
            project_name="ai-trip-planner-fresh-7-7"
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
        print(f"📊 Project: ai-trip-planner")
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

app = FastAPI(title="Trip Planner API", lifespan=lifespan)

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

# Define the state for our graph
class TripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    final_result: Optional[str]

# Initialize the LLM - Using GPT-4.1 for production
# Note: This should be initialized after instrumentation setup
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",  # GPT-4o-mini
    temperature=0,
    max_tokens=2000,
    timeout=30
)

# Initialize search tool if available
search_tools = []
if os.getenv("TAVILY_API_KEY"):
    search_tools.append(TavilySearchResults(max_results=5))

# Define trip planning tools with proper trace context
@tool
def research_destination(destination: str, duration: str) -> str:
    """Research a destination comprehensively for trip planning.
    
    Args:
        destination: The destination to research
        duration: Duration of the trip
    """
    # System message for constraints
    system_prompt = "You are a concise travel researcher. CRITICAL: Your response must be under 150 words and 800 characters. Focus on key facts only."
    
    # Use search tool if available, otherwise use LLM knowledge
    if search_tools:
        search_tool = search_tools[0]
        search_results = search_tool.invoke(f"{destination} travel guide {duration} trip attractions weather")
        
        prompt_template = """Research {destination} for {duration} trip based on: {search_results}

Key info only:
- Weather/best time
- Top 3 attractions  
- Transport options
- Cultural notes
- Safety basics"""
        
        prompt_template_variables = {
            "destination": destination,
            "duration": duration,
            "search_results": str(search_results)[:500]  # Limit search results length
        }
    else:
        prompt_template = """Research {destination} for {duration} trip.

Key info only:
- Weather/best time
- Top 3 attractions
- Transport options  
- Cultural notes
- Safety basics

Note: Using general knowledge."""
        
        prompt_template_variables = {
            "destination": destination,
            "duration": duration
        }
    
    # Track prompt template for observability
    with using_prompt_template(
        template=prompt_template,
        variables=prompt_template_variables,
        version="research-v1.0"
    ):
        formatted_prompt = prompt_template.format(**prompt_template_variables)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_prompt)
        ])
    
    return response.content

@tool
def analyze_budget(destination: str, duration: str, budget: str = None) -> str:
    """Analyze budget requirements for a trip.
    
    Args:
        destination: The destination
        duration: Duration of the trip
        budget: Target budget (optional)
    """
    budget_text = budget or "provide options for different budget levels"
    
    # Use system message for strict constraints
    system_prompt = "You are a concise travel budget analyst. CRITICAL: Your response must be under 100 words and 500 characters. No exceptions."
    
    prompt_template = """Budget for {duration} trip to {destination}. Target: {budget}

Include only:
- Accommodation range
- Transport costs  
- Food budget
- Activities cost
- Total estimate

Be extremely concise."""
    
    prompt_template_variables = {
        "destination": destination,
        "duration": duration,
        "budget": budget_text
    }
    
    # Track prompt template for observability
    with using_prompt_template(
        template=prompt_template,
        variables=prompt_template_variables,
        version="budget-v1.0"
    ):
        formatted_prompt = prompt_template.format(**prompt_template_variables)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_prompt)
        ])
    
    return response.content

@tool
def curate_local_experiences(destination: str, interests: str = None) -> str:
    """Curate authentic local experiences and hidden gems.
    
    Args:
        destination: The destination
        interests: Traveler interests (optional)
    """
    interests_text = interests or "general exploration and cultural immersion"
    
    # System message for constraints
    system_prompt = "You are a local experience curator. CRITICAL: Your response must be under 100 words and 600 characters. Focus on authentic, specific recommendations only."
    
    prompt_template = """Local experiences in {destination} for: {interests}

Recommend only:
- 2 hidden gem restaurants  
- 1 cultural activity
- 1 off-path location
- 1 local market/workshop
- Cultural etiquette tip

Be specific and concise."""
    
    prompt_template_variables = {
        "destination": destination,
        "interests": interests_text
    }
    
    # Track prompt template for observability
    with using_prompt_template(
        template=prompt_template,
        variables=prompt_template_variables,
        version="local-v1.0"
    ):
        formatted_prompt = prompt_template.format(**prompt_template_variables)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_prompt)
        ])
    
    return response.content

@tool
def create_itinerary(destination: str, duration: str, research: str, budget_info: str, local_info: str, travel_style: str = None) -> str:
    """Create a comprehensive day-by-day itinerary.
    
    Args:
        destination: The destination
        duration: Duration of the trip
        research: Destination research information
        budget_info: Budget analysis information
        local_info: Local experiences information
        travel_style: Travel style preferences (optional)
    """
    style_text = travel_style or "Standard"
    
    # System message for constraints
    system_prompt = "You are a concise trip planner. CRITICAL: Your response must be under 200 words and 1200 characters. Create day-by-day format with times, activities, and costs only."
    
    prompt_template = """{duration} itinerary for {destination} ({travel_style} style):

Research: {research}
Budget: {budget_info}
Local: {local_info}

Format: Day X: Time - Activity - Cost
Include top attractions, meals, transport between locations.
Be concise."""
    
    prompt_template_variables = {
        "destination": destination,
        "duration": duration,
        "travel_style": style_text,
        "research": research[:200],  # Limit input length
        "budget_info": budget_info[:200],
        "local_info": local_info[:200]
    }
    
    # Track prompt template for observability
    with using_prompt_template(
        template=prompt_template,
        variables=prompt_template_variables,
        version="itinerary-v1.0"
    ):
        formatted_prompt = prompt_template.format(**prompt_template_variables)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_prompt)
        ])
    
    return response.content

# Enhanced state to track parallel data
class EfficientTripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research_data: Optional[str]
    budget_data: Optional[str]
    local_data: Optional[str]
    final_result: Optional[str]

# Define more efficient nodes for parallel execution
def research_node(state: EfficientTripPlannerState) -> EfficientTripPlannerState:
    """Research destination in parallel"""
    with tracer.start_as_current_span("research_node") as span:
        try:
            trip_req = state["trip_request"]
            # Add span attributes for better tracing
            span.set_attribute("destination", trip_req.get("destination", "Unknown"))
            span.set_attribute("duration", trip_req.get("duration", "Unknown"))
            
            print(f"🔍 Starting research for {trip_req.get('destination', 'Unknown')}")
            
            research_result = research_destination.invoke({
                "destination": trip_req["destination"], 
                "duration": trip_req["duration"]
            })
            
            print(f"✅ Research completed for {trip_req.get('destination', 'Unknown')}")
            span.set_status(Status(StatusCode.OK))
            return {
                "messages": [HumanMessage(content=f"Research completed: {research_result}")],
                "research_data": research_result
            }
        except Exception as e:
            print(f"❌ Research node error: {str(e)}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            return {
                "messages": [HumanMessage(content=f"Research failed: {str(e)}")],
                "research_data": f"Research failed: {str(e)}"
            }

def budget_node(state: EfficientTripPlannerState) -> EfficientTripPlannerState:
    """Analyze budget in parallel"""
    with tracer.start_as_current_span("budget_node") as span:
        try:
            trip_req = state["trip_request"]
            # Add span attributes
            span.set_attribute("destination", trip_req.get("destination", "Unknown"))
            span.set_attribute("duration", trip_req.get("duration", "Unknown"))
            span.set_attribute("budget", trip_req.get("budget", "Not specified"))
            
            print(f"💰 Starting budget analysis for {trip_req.get('destination', 'Unknown')}")
            
            budget_result = analyze_budget.invoke({
                "destination": trip_req["destination"], 
                "duration": trip_req["duration"], 
                "budget": trip_req.get("budget")
            })
            
            print(f"✅ Budget analysis completed for {trip_req.get('destination', 'Unknown')}")
            span.set_status(Status(StatusCode.OK))
            return {
                "messages": [HumanMessage(content=f"Budget analysis completed: {budget_result}")],
                "budget_data": budget_result
            }
        except Exception as e:
            print(f"❌ Budget node error: {str(e)}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            return {
                "messages": [HumanMessage(content=f"Budget analysis failed: {str(e)}")],
                "budget_data": f"Budget analysis failed: {str(e)}"
            }

def local_experiences_node(state: EfficientTripPlannerState) -> EfficientTripPlannerState:
    """Curate local experiences in parallel"""
    with tracer.start_as_current_span("local_experiences_node") as span:
        try:
            trip_req = state["trip_request"]
            # Add span attributes
            span.set_attribute("destination", trip_req.get("destination", "Unknown"))
            span.set_attribute("interests", trip_req.get("interests", "Not specified"))
            
            print(f"🍽️ Starting local experiences curation for {trip_req.get('destination', 'Unknown')}")
            
            local_result = curate_local_experiences.invoke({
                "destination": trip_req["destination"], 
                "interests": trip_req.get("interests")
            })
            
            print(f"✅ Local experiences completed for {trip_req.get('destination', 'Unknown')}")
            span.set_status(Status(StatusCode.OK))
            return {
                "messages": [HumanMessage(content=f"Local experiences curated: {local_result}")],
                "local_data": local_result
            }
        except Exception as e:
            print(f"❌ Local experiences node error: {str(e)}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            return {
                "messages": [HumanMessage(content=f"Local experiences failed: {str(e)}")],
                "local_data": f"Local experiences failed: {str(e)}"
            }

def itinerary_node(state: EfficientTripPlannerState) -> EfficientTripPlannerState:
    """Create final itinerary using all gathered data"""
    with tracer.start_as_current_span("itinerary_node") as span:
        try:
            trip_req = state["trip_request"]
            # Add span attributes
            span.set_attribute("destination", trip_req.get("destination", "Unknown"))
            span.set_attribute("duration", trip_req.get("duration", "Unknown"))
            span.set_attribute("travel_style", trip_req.get("travel_style", "Standard"))
            
            print(f"📅 Starting itinerary creation for {trip_req.get('destination', 'Unknown')}")
            
            # Get data from previous nodes
            research_data = state.get("research_data", "")
            budget_data = state.get("budget_data", "")
            local_data = state.get("local_data", "")
            
            print(f"📊 Data available - Research: {len(research_data) if research_data else 0} chars, Budget: {len(budget_data) if budget_data else 0} chars, Local: {len(local_data) if local_data else 0} chars")
            
            itinerary_result = create_itinerary.invoke({
                "destination": trip_req["destination"],
                "duration": trip_req["duration"],
                "research": research_data,
                "budget_info": budget_data,
                "local_info": local_data,
                "travel_style": trip_req.get("travel_style")
            })
            
            print(f"✅ Itinerary creation completed for {trip_req.get('destination', 'Unknown')}")
            span.set_status(Status(StatusCode.OK))
            return {
                "messages": [HumanMessage(content=itinerary_result)],
                "final_result": itinerary_result
            }
        except Exception as e:
            print(f"❌ Itinerary node error: {str(e)}")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            return {
                "messages": [HumanMessage(content=f"Itinerary creation failed: {str(e)}")],
                "final_result": f"Itinerary creation failed: {str(e)}"
            }

# Build the optimized graph with parallel execution
def create_efficient_trip_planning_graph():
    """Create and compile the optimized trip planning graph with parallel execution"""
    
    # Create the state graph
    workflow = StateGraph(EfficientTripPlannerState)
    
    # Add parallel processing nodes
    workflow.add_node("research", research_node)
    workflow.add_node("budget", budget_node)
    workflow.add_node("local_experiences", local_experiences_node)
    workflow.add_node("itinerary", itinerary_node)
    
    # Start all research tasks in parallel
    workflow.add_edge(START, "research")
    workflow.add_edge(START, "budget")
    workflow.add_edge(START, "local_experiences")
    
    # All parallel tasks feed into itinerary creation
    workflow.add_edge("research", "itinerary")
    workflow.add_edge("budget", "itinerary")
    workflow.add_edge("local_experiences", "itinerary")
    
    # Itinerary is the final step
    workflow.add_edge("itinerary", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# API Routes
@app.get("/")
async def root():
    return {"message": "Trip Planner API is running with simplified LangGraph!"}


@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using optimized parallel LangGraph workflow"""
    with tracer.start_as_current_span("plan_trip_request") as span:
        # Add request attributes to span
        span.set_attribute("destination", trip_request.destination)
        span.set_attribute("duration", trip_request.duration)
        span.set_attribute("budget", trip_request.budget or "Not specified")
        span.set_attribute("interests", trip_request.interests or "Not specified")
        span.set_attribute("travel_style", trip_request.travel_style or "Standard")
        
        try:
            # Create the efficient graph
            graph = create_efficient_trip_planning_graph()
            
            # Prepare initial state with the new structure
            initial_state = {
                "messages": [],
                "trip_request": trip_request.model_dump(),
                "research_data": None,
                "budget_data": None,
                "local_data": None,
                "final_result": None
            }
            
            # Execute the workflow with parallel processing
            config = {"configurable": {"thread_id": f"trip_{trip_request.destination.replace(' ', '_')}_{trip_request.duration.replace(' ', '_')}"}}
            
            print(f"🚀 Starting trip planning for {trip_request.destination} ({trip_request.duration})")
            
            output = graph.invoke(initial_state, config)
            
            print(f"✅ Trip planning completed. Output keys: {list(output.keys()) if output else 'None'}")
            
            # Return the final result
            if output and output.get("final_result"):
                span.set_status(Status(StatusCode.OK))
                return TripResponse(result=output.get("final_result"))
            elif output and output.get("messages") and len(output.get("messages")) > 0:
                # Fallback to last message if final_result is not available
                last_message = output.get("messages")[-1]
                content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                span.set_status(Status(StatusCode.OK))
                return TripResponse(result=content)
            
            span.set_status(Status(StatusCode.ERROR, "No result generated"))
            return TripResponse(result="Trip planning completed but no detailed results available.")
            
        except Exception as e:
            print(f"❌ Trip planning error: {str(e)}")
            import traceback
            traceback.print_exc()
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=f"Trip planning failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-backend-simplified"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)    

