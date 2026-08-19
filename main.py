
# LangGraph Multi-Agent Travel Booking System with Long-Term Memory

import os
import json
from typing import TypedDict, Annotated
import operator
import asyncio
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_openai import ChatOpenAI

# from tools.tavily_tool import tavily_search

#from mcp_client import tavily_mcp_search

from mcp_client import (
    tavily_mcp_search,
    get_airports,
    get_airlines,
    aviation_mcp_call,extract_destination,forecast_mcp_search,weather_mcp_search,
    sightseeing_mcp_search,
)


#from tools.flight_tool import search_flights


from dotenv import load_dotenv
#load_dotenv()
load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")

# LangSmith tracing — activates automatically once LANGCHAIN_API_KEY is set in .env
# (get a free key at https://smith.langchain.com/settings). No-op until then.
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "ai-travel-planner")

# LLM — routed through OpenRouter (avoids Groq's low free-tier TPM limits)
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")


def _build_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


llm = _build_llm(DEFAULT_MODEL)


def set_model(model: str):
    """Swap the active model — node functions read the module-level `llm`
    at call time, so reassigning it here takes effect on the next graph run."""
    global llm
    llm = _build_llm(model)

def _usage_tokens(response) -> int:
    meta = getattr(response, "usage_metadata", None) or {}
    return meta.get("total_tokens", 0)


# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    total_tokens: int
    weather_results: str
    destination_city: str
    sightseeing_results: str

# Flight Agent
# def flight_agent(state: TravelState):
#     query = state["user_query"]
#     flight_data = search_flights(query)
#     return {
#         "flight_results": flight_data,
#         "messages": [
#             AIMessage(content=f"Flight results fetched")
#         ],
#         "llm_calls": state.get("llm_calls", 0) + 1
#     }


# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""



# Flight Agent
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content
        tokens = _usage_tokens(response)

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"
        tokens = 0

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "total_tokens": state.get("total_tokens", 0) + tokens,
    }




# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    #hotel_results = tavily_search(query)

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )
    except Exception as e:
        hotel_results = f"Hotel information unavailable: {str(e)}"

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }







def _parse_mcp_json(result):
    """MCP tool results can be a plain string or a list of content blocks
    (e.g. [{'type': 'text', 'text': '<json string>'}]) — extract and parse the JSON payload."""
    text = None
    if isinstance(result, str):
        text = result
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
                break
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _format_current_weather(current: dict | None) -> str:
    if not current or "temperature_c" not in current:
        return "_Current weather unavailable._"
    return (
        f"**{current.get('city', 'Unknown')}** — {current['temperature_c']}°C "
        f"(feels like {current['feels_like_c']}°C)\n\n"
        f"{current['condition'].capitalize()} · Humidity {current['humidity']}% "
        f"· Wind {current['wind_speed']} m/s"
    )


def _format_forecast(forecast: dict | None) -> str:
    entries = forecast.get("forecast") if forecast else None
    if not entries:
        return "_Forecast unavailable._"
    lines = ["| Time | Temp | Condition |", "|---|---|---|"]
    for e in entries:
        lines.append(f"| {e['datetime']} | {e['temperature']}°C | {e['weather'].capitalize()} |")
    return "\n".join(lines)


def weather_agent(state: TravelState):

    dest_tokens = 0
    try:
        city, dest_tokens = extract_destination(state["user_query"])
        weather_raw = asyncio.run(
            weather_mcp_search(city)
        )
        forecast_raw = asyncio.run(
            forecast_mcp_search(city)
        )
        current_text = _format_current_weather(_parse_mcp_json(weather_raw))
        forecast_text = _format_forecast(_parse_mcp_json(forecast_raw))
    except Exception as e:
        city = ""
        current_text = f"Weather information unavailable: {str(e)}"
        forecast_text = ""

    return {
        "weather_results": f"""**Current Weather**

{current_text}

**Forecast**

{forecast_text}
""",
        "destination_city": city,
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ],
        "total_tokens": state.get("total_tokens", 0) + dest_tokens,
    }


def _format_attractions(data) -> str:
    attractions = data.get("attractions") if data else None
    if not attractions:
        return "_No notable attractions found._"
    lines = ["| Attraction | Category | Rating |", "|---|---|---|"]
    for a in attractions[:12]:
        name = a.get("name") or "Unknown"
        kind = (a.get("kinds") or "").split(",")[0].replace("_", " ").title() or "—"
        rate = a.get("rate") or 0
        stars = "⭐" * min(int(rate), 5) if isinstance(rate, (int, float)) else "—"
        lines.append(f"| {name} | {kind} | {stars} |")
    return "\n".join(lines)


def sightseeing_agent(state: TravelState):

    city = state.get("destination_city") or "the destination"

    try:
        raw = asyncio.run(sightseeing_mcp_search(city))
        sightseeing_text = _format_attractions(_parse_mcp_json(raw))
    except Exception as e:
        sightseeing_text = f"Sightseeing information unavailable: {str(e)}"

    return {
        "sightseeing_results": sightseeing_text,
        "messages": [
            AIMessage(
                content="Sightseeing attractions fetched"
            )
        ],
    }



# Itinerary Agent
def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}

    Weather Information:
    {state['weather_results']}

    Sightseeing / Attractions:
    {state.get('sightseeing_results', '')}

    Use the named attractions above to build a concrete day-by-day sightseeing plan
    rather than generic suggestions, where relevant.
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "total_tokens": state.get("total_tokens", 0) + _usage_tokens(response),
    }







graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("sightseeing_agent", sightseeing_agent)
graph.add_node("itinerary_agent", itinerary_agent)


graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "sightseeing_agent")
graph.add_edge("sightseeing_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", END)


# Persistent connection so both CLI and Streamlit can share the compiled app
_conn = psycopg.connect(DATABASE_URL, autocommit=True)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)


# ── Search history (latest searches: model, tokens, response) ──────────────────
with _conn.cursor() as _cur:
    _cur.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            user_id TEXT,
            query TEXT,
            model TEXT,
            tokens_used INTEGER,
            start_date DATE,
            end_date DATE,
            response TEXT
        )
    """)


def log_search(user_id: str, query: str, model: str, tokens_used: int,
                response: str, start_date=None, end_date=None):
    with _conn.cursor() as cur:
        cur.execute(
            """INSERT INTO search_history
               (user_id, query, model, tokens_used, start_date, end_date, response)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (user_id, query, model, tokens_used, start_date, end_date, response),
        )


def get_recent_searches(limit: int = 20):
    with _conn.cursor() as cur:
        cur.execute(
            """SELECT created_at, user_id, query, model, tokens_used,
                      start_date, end_date, response
               FROM search_history ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


if __name__ == "__main__":
    # config = {
    #     "configurable": {
    #         "thread_id": "user_aarohi"
    #     }
    # }

    # every run starts fresh.
    import uuid
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }


    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
            "total_tokens": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)
