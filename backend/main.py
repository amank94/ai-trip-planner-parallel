from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Minimal tools (deterministic for tutorials)
@tool
def essential_info(destination: str) -> str:
    """Return essential destination info like weather, sights, and etiquette."""
    # Real-world destination data
    info_db = {
        "Prague": "Weather: 5-15°C spring/fall, 20-25°C summer. Top sights: Prague Castle, Charles Bridge, Old Town Square. Etiquette: Remove shoes indoors, quiet on public transport.",
        "Bangkok": "Weather: 25-35°C year-round, monsoon May-Oct. Top sights: Grand Palace, Wat Pho, Chatuchak Market. Etiquette: Remove shoes in temples, dress modestly, don't touch heads.",
        "Dubai": "Weather: 25-45°C, hottest Jun-Sep. Top sights: Burj Khalifa, Dubai Mall, Gold Souk. Etiquette: Modest dress in public, no PDA, respect Ramadan customs.",
        "Barcelona": "Weather: 10-25°C, mild winters. Top sights: Sagrada Familia, Park Güell, Las Ramblas. Etiquette: Late dining (9-10pm), siesta time respected.",
        "Tokyo": "Weather: 5-30°C seasonal variation. Top sights: Senso-ji, Shibuya, Mount Fuji day trip. Etiquette: No eating while walking, bow for greetings, quiet on trains.",
        "Rome": "Weather: 8-30°C Mediterranean climate. Top sights: Colosseum, Vatican, Trevi Fountain. Etiquette: Dress modestly in churches, no touching artifacts.",
        "Lisbon": "Weather: 10-28°C, mild year-round. Top sights: Belém Tower, Jerónimos, Alfama. Etiquette: Greet shopkeepers, late dinners common.",
        "Marrakech": "Weather: 8-38°C, hot summers. Top sights: Jemaa el-Fnaa, Majorelle Garden, Souks. Etiquette: Haggling expected, modest dress, ask before photos.",
        "New York": "Weather: -3-30°C seasonal extremes. Top sights: Statue of Liberty, Central Park, Times Square. Etiquette: Walk fast, stand right on escalators, tip 20%."
    }
    
    # Extract city name from destination string
    city = destination.split(',')[0].strip()
    for key in info_db:
        if key.lower() in city.lower() or city.lower() in key.lower():
            return info_db[key]
    
    return f"Essential info for {destination}: Check seasonal weather patterns, research top attractions, respect local customs and dress codes."


@tool
def budget_basics(destination: str, duration: str) -> str:
    """Return high-level budget categories for a given destination and duration."""
    # Extract days from duration string
    import re
    days_match = re.search(r'(\d+)', duration)
    days = int(days_match.group(1)) if days_match else 5
    
    # Budget data per day (USD)
    budget_db = {
        "Prague": {"lodging": 60, "food": 30, "transit": 10, "attractions": 25},
        "Bangkok": {"lodging": 40, "food": 20, "transit": 8, "attractions": 20},
        "Dubai": {"lodging": 120, "food": 50, "transit": 20, "attractions": 40},
        "Barcelona": {"lodging": 80, "food": 40, "transit": 12, "attractions": 30},
        "Tokyo": {"lodging": 90, "food": 45, "transit": 15, "attractions": 35},
        "Rome": {"lodging": 85, "food": 40, "transit": 10, "attractions": 30},
        "Lisbon": {"lodging": 65, "food": 30, "transit": 8, "attractions": 20},
        "Marrakech": {"lodging": 50, "food": 25, "transit": 10, "attractions": 20},
        "New York": {"lodging": 150, "food": 60, "transit": 15, "attractions": 40}
    }
    
    city = destination.split(',')[0].strip()
    for key in budget_db:
        if key.lower() in city.lower() or city.lower() in key.lower():
            daily = budget_db[key]
            total = sum(daily.values()) * days
            breakdown = f"Lodging: ${daily['lodging']*days}, Food: ${daily['food']*days}, Transit: ${daily['transit']*days}, Attractions: ${daily['attractions']*days}"
            return f"Budget for {destination} over {duration} (~${total} total): {breakdown}"
    
    return f"Budget for {destination} over {duration}: Lodging $50-150/day, Food $30-60/day, Transit $10-20/day, Attractions $20-40/day"


@tool
def local_flavor(destination: str, interests: Optional[str] = None) -> str:
    """Suggest authentic local experiences matching optional interests."""
    experiences_db = {
        "Prague": "Try goulash at Lokál, explore Petřín Tower at sunset, Czech beer tasting in microbreweries, classical concerts in historic churches",
        "Bangkok": "Street food at Chinatown, sunrise at Wat Arun, Thai cooking class, Chao Phraya river dinner cruise, Muay Thai match",
        "Dubai": "Desert safari with Bedouin dinner, Gold Souk haggling, Friday brunch culture, abra boat across Dubai Creek",
        "Barcelona": "Tapas crawl in El Born, flamenco show in Gothic Quarter, beach volleyball at Barceloneta, Boqueria Market morning visit",
        "Tokyo": "Tsukiji outer market breakfast, karaoke in Shibuya, onsen experience, robot restaurant show, cherry blossom hanami parties",
        "Rome": "Aperitivo in Trastevere, early morning Vatican visit, carbonara at local trattoria, evening stroll at Villa Borghese",
        "Lisbon": "Fado music in Alfama, pastel de nata at Belém, sunset at Miradouro, tram 28 ride through historic neighborhoods",
        "Marrakech": "Hammam spa experience, tagine cooking class, sunset at Jemaa el-Fnaa, mint tea in traditional riad",
        "New York": "Jazz at Blue Note, food tour in Queens, High Line walk, Broadway show, Brooklyn flea markets"
    }
    
    city = destination.split(',')[0].strip()
    for key in experiences_db:
        if key.lower() in city.lower() or city.lower() in key.lower():
            base_exp = experiences_db[key]
            if interests:
                return f"Local experiences for {destination} based on {interests}: {base_exp}"
            return f"Local experiences for {destination}: {base_exp}"
    
    return f"Local experiences for {destination}: Explore local markets, try regional cuisine, visit cultural sites, experience traditional entertainment"


@tool
def day_plan(destination: str, day: int) -> str:
    """Return a simple day plan outline for a specific day number."""
    return f"Day {day} in {destination}: breakfast, highlight visit, lunch, afternoon walk, dinner."


# Additional simple tools per agent (to mirror original multi-tool behavior)
@tool
def weather_brief(destination: str) -> str:
    """Return a brief weather summary for planning purposes."""
    weather_db = {
        "Prague": "Continental climate: Cold winters (-2 to 4°C), warm summers (15-25°C). Best: May-Sept. Pack layers.",
        "Bangkok": "Tropical: Hot year-round (25-35°C), monsoon May-Oct. Best: Nov-Feb. Light clothes, umbrella needed.",
        "Dubai": "Desert climate: Very hot summers (40-45°C), mild winters (15-25°C). Best: Nov-Mar. AC everywhere.",
        "Barcelona": "Mediterranean: Mild winters (8-15°C), hot summers (20-28°C). Best: Apr-Jun, Sept-Oct.",
        "Tokyo": "Four seasons: Cold winters (0-10°C), hot humid summers (25-35°C). Best: Spring (cherry blossoms) or Fall.",
        "Rome": "Mediterranean: Mild winters (8-15°C), hot summers (20-32°C). Best: Apr-May, Sept-Oct.",
        "Lisbon": "Atlantic climate: Mild year-round (10-28°C), rainy Nov-Mar. Best: May-Oct. Always pleasant.",
        "Marrakech": "Semi-arid: Hot summers (20-38°C), cool winters (8-20°C). Best: Mar-May, Sept-Nov.",
        "New York": "Four seasons: Cold winters (-5 to 5°C), hot summers (20-30°C). Best: Apr-Jun, Sept-Nov."
    }
    
    city = destination.split(',')[0].strip()
    for key in weather_db:
        if key.lower() in city.lower() or city.lower() in key.lower():
            return weather_db[key]
    
    return f"Weather in {destination}: Check seasonal patterns, pack appropriate clothing, consider best travel months"


@tool
def visa_brief(destination: str) -> str:
    """Return a brief visa guidance placeholder for tutorial purposes."""
    visa_db = {
        "Prague": "Czech Republic (Schengen): EU/US/UK citizens visa-free 90 days. Others may need Schengen visa.",
        "Bangkok": "Thailand: Many nationalities get 30-day visa on arrival. US/EU/UK can extend to 60 days.",
        "Dubai": "UAE: Most visitors get 30-90 day visa on arrival. Check UAE immigration for your nationality.",
        "Barcelona": "Spain (Schengen): EU citizens no visa. US/UK/CAN visa-free 90 days in 180.",
        "Tokyo": "Japan: Many countries visa-free 90 days (US/EU/UK). Work/study requires proper visa.",
        "Rome": "Italy (Schengen): EU no visa needed. US/UK/CAN/AUS visa-free 90 days in Schengen zone.",
        "Lisbon": "Portugal (Schengen): EU citizens free movement. Most others 90 days visa-free in 180.",
        "Marrakech": "Morocco: US/EU/UK visa-free 90 days. Some nationalities need visa in advance.",
        "New York": "USA: ESTA required for visa waiver countries. Others need B1/B2 tourist visa. Apply early."
    }
    
    city = destination.split(',')[0].strip()
    for key in visa_db:
        if key.lower() in city.lower() or city.lower() in key.lower():
            return visa_db[key]
    
    return f"Visa for {destination}: Check embassy requirements for your nationality. Most tourists get 30-90 days visa-free or on arrival"


@tool
def attraction_prices(destination: str, attractions: Optional[List[str]] = None) -> str:
    """Return rough placeholder prices for attractions."""
    prices_db = {
        "Prague": {"Prague Castle": "$14", "Jewish Quarter": "$20", "Petrin Tower": "$8", "Museums": "$8-12"},
        "Bangkok": {"Grand Palace": "$15", "Wat Pho": "$6", "Wat Arun": "$3", "Museums": "$5-10"},
        "Dubai": {"Burj Khalifa": "$40-60", "Dubai Museum": "$1", "Dubai Frame": "$15", "Desert Safari": "$50-80"},
        "Barcelona": {"Sagrada Familia": "$30", "Park Güell": "$13", "Casa Batlló": "$35", "Picasso Museum": "$14"},
        "Tokyo": {"Tokyo Tower": "$12", "Senso-ji": "Free", "teamLab": "$35", "Museums": "$10-15"},
        "Rome": {"Colosseum": "$20", "Vatican Museums": "$25", "Borghese Gallery": "$17", "Pantheon": "Free"},
        "Lisbon": {"Belém Tower": "$8", "Jerónimos": "$12", "Castle": "$13", "Tram 28": "$3"},
        "Marrakech": {"Majorelle Garden": "$8", "Bahia Palace": "$8", "Saadian Tombs": "$6", "Museums": "$5-10"},
        "New York": {"Empire State": "$44", "MET Museum": "$30", "MoMA": "$25", "Statue of Liberty": "$24"}
    }
    
    city = destination.split(',')[0].strip()
    for key in prices_db:
        if key.lower() in city.lower() or city.lower() in key.lower():
            price_list = ", ".join([f"{k}: {v}" for k, v in prices_db[key].items()])
            return f"Attraction prices in {destination}: {price_list}"
    
    items = attractions or ["Museums", "Historic Sites", "Viewpoints"]
    priced = ", ".join(f"{a}: $10-30" for a in items)
    return f"Attraction prices in {destination}: {priced}"


@tool
def local_customs(destination: str) -> str:
    """Return simple etiquette reminders for the destination."""
    customs_db = {
        "Prague": "Say 'Dobrý den' (hello), remove shoes in homes, quiet on public transport, tip 10%, avoid loud tourist behavior",
        "Bangkok": "Wai greeting (palms together), never touch heads, feet are lowest, remove shoes in temples, King is revered",
        "Dubai": "No public affection, modest dress, no alcohol in public, left hand unclean, respect Ramadan fasting",
        "Barcelona": "Kiss both cheeks greeting, late dining (9-10pm), respect siesta, try Catalan phrases, don't call it Spain only",
        "Tokyo": "Bow don't shake hands, no tips ever, quiet on trains, slurping noodles is polite, business card ritual important",
        "Rome": "Dress modestly in Vatican, no touching artifacts, coperto (cover charge) normal, cappuccino only before 11am",
        "Lisbon": "Greet shopkeepers always, don't compare to Spain, fado is sacred, dinner after 8pm, queue politely",
        "Marrakech": "Right hand for eating, haggling expected, ask before photos, dress conservatively, Friday is holy day",
        "New York": "Walk fast or move aside, tip 20%, direct communication style, stand right on escalators, respect personal space"
    }
    
    city = destination.split(',')[0].strip()
    for key in customs_db:
        if key.lower() in city.lower() or city.lower() in key.lower():
            return f"Customs in {destination}: {customs_db[key]}"
    
    return f"Customs in {destination}: Be respectful, dress modestly in religious sites, learn basic greetings, follow local dining etiquette"


@tool
def hidden_gems(destination: str) -> str:
    """Return a few off-the-beaten-path ideas."""
    gems_db = {
        "Prague": "Vrtba Garden (baroque terrace garden), Speculum Alchemiae (alchemy museum), Riegrovy Sady beer garden, Wallenstein Garden peacocks",
        "Bangkok": "Talad Rot Fai night market, Erawan Museum, Bang Krachao green lung, Artist's House puppet shows, Airplane Graveyard",
        "Dubai": "Al Seef heritage district, Coffee Museum, Ras Al Khor flamingo sanctuary, Al Qudra Lakes, Persian restaurants in Deira",
        "Barcelona": "Bunkers del Carmel viewpoint, Hospital de Sant Pau, Mercat de Sant Antoni, Labyrinth Park, El Nacional food hall",
        "Tokyo": "Yanaka cemetery walks, Golden Gai tiny bars, Koenji vintage shopping, Todoroki Valley, teamLab Borderless",
        "Rome": "Quartiere Coppedè architecture, Protestant Cemetery, Centrale Montemartini museum, Gianicolo Hill sunset, Via Margutta art street",
        "Lisbon": "LX Factory creative hub, Calouste Gulbenkian gardens, Pink Street nightlife, Feira da Ladra flea market, Estufa Fria greenhouse",
        "Marrakech": "Le Jardin Secret, Dar Si Said Museum, Mellah spice market, Cyber Park, rooftop cafes in the Medina",
        "New York": "Elevated Acre park, Morgan Library, Brooklyn Botanic Garden, Smorgasburg food market, Green-Wood Cemetery views"
    }
    
    city = destination.split(',')[0].strip()
    for key in gems_db:
        if key.lower() in city.lower() or city.lower() in key.lower():
            return f"Hidden gems in {destination}: {gems_db[key]}"
    
    return f"Hidden gems in {destination}: Explore quiet neighborhoods, local artist galleries, neighborhood markets, lesser-known parks and viewpoints"


@tool
def travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Return an approximate travel time placeholder."""
    return f"Travel from {from_location} to {to_location} by {mode}: ~20-60 minutes."


@tool
def packing_list(destination: str, duration: str, activities: Optional[List[str]] = None) -> str:
    """Return a generic packing list summary."""
    acts = ", ".join(activities or ["walking", "sightseeing"]) 
    return f"Packing for {destination} ({duration}): comfortable shoes, layers, adapter; for {acts}."


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research: Optional[str]
    budget: Optional[str]
    local: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def research_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    prompt_t = (
        "You are a research assistant.\n"
        "Gather essential information about {destination}.\n"
        "Use at most one tool if needed."
    )
    vars_ = {"destination": destination}
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        agent = llm.bind_tools([essential_info, weather_brief, visa_brief])
        res = agent.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    out = res.content
    calls: List[Dict[str, Any]] = []
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode([essential_info, weather_brief, visa_brief])
        tr = tool_node.invoke({"messages": [res]})
        msgs = tr["messages"]
        out = msgs[-1].content if msgs else out

    return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def budget_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination, duration = req["destination"], req["duration"]
    prompt_t = (
        "You are a budget analyst.\n"
        "Summarize high-level costs for {destination} over {duration}."
    )
    vars_ = {"destination": destination, "duration": duration}
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        agent = llm.bind_tools([budget_basics, attraction_prices])
        res = agent.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    out = res.content
    calls: List[Dict[str, Any]] = []
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode([budget_basics, attraction_prices])
        tr = tool_node.invoke({"messages": [res]})
        msgs = tr["messages"]
        out = msgs[-1].content if msgs else out

    return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def local_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    interests = req.get("interests", "local culture")
    prompt_t = (
        "You are a local guide.\n"
        "Suggest authentic experiences in {destination} for interests: {interests}."
    )
    vars_ = {"destination": destination, "interests": interests}
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        agent = llm.bind_tools([local_flavor, local_customs, hidden_gems])
        res = agent.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    out = res.content
    calls: List[Dict[str, Any]] = []
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "local", "tool": c["name"], "args": c.get("args", {})})
        tool_node = ToolNode([local_flavor, local_customs, hidden_gems])
        tr = tool_node.invoke({"messages": [res]})
        msgs = tr["messages"]
        out = msgs[-1].content if msgs else out

    return {"messages": [SystemMessage(content=out)], "local": out, "tool_calls": calls}


def itinerary_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    duration = req["duration"]
    travel_style = req.get("travel_style", "standard")
    prompt_t = (
        "Create a {duration} itinerary for {destination} ({travel_style}).\n\n"
        "Inputs:\nResearch: {research}\nBudget: {budget}\nLocal: {local}\n"
    )
    vars_ = {
        "duration": duration,
        "destination": destination,
        "travel_style": travel_style,
        "research": (state.get("research") or "")[:400],
        "budget": (state.get("budget") or "")[:400],
        "local": (state.get("local") or "")[:400],
    }
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_graph():
    g = StateGraph(TripState)
    g.add_node("research_agent", research_agent)
    g.add_node("budget_agent", budget_agent)
    g.add_node("local_agent", local_agent)
    g.add_node("itinerary_agent", itinerary_agent)

    # Run research, budget, and local agents in parallel
    g.add_edge(START, "research_agent")
    g.add_edge(START, "budget_agent")
    g.add_edge(START, "local_agent")
    
    # All three agents feed into the itinerary agent
    g.add_edge("research_agent", "itinerary_agent")
    g.add_edge("budget_agent", "itinerary_agent")
    g.add_edge("local_agent", "itinerary_agent")
    
    g.add_edge("itinerary_agent", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="AI Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-trip-planner"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="ai-trip-planner")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):

    graph = build_graph()
    # Only include necessary fields in initial state
    # Agent outputs (research, budget, local, final) will be added during execution
    state = {
        "messages": [],
        "trip_request": req.model_dump(),
        "tool_calls": [],
    }
    # No config needed without checkpointer
    out = graph.invoke(state)
    return TripResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
