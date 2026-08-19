# AI Travel Planning App using LangGraph, LangSmith and MCP

A multi-agent travel planning system built with [LangGraph](https://github.com/langchain-ai/langgraph). Specialized agents for flights, hotels, weather, and sightseeing run as a graph, each pulling live data through [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) servers, and hand off to an itinerary agent that synthesizes everything into a day-by-day plan. Conversation state is persisted in PostgreSQL so plans can be resumed across sessions.



## Architecture

```
START -> flight_agent -> hotel_agent -> weather_agent -> sightseeing_agent -> itinerary_agent -> END
```
<img width="1442" height="583" alt="image" src="https://github.com/user-attachments/assets/7122c563-ec7e-4166-8c1d-7be96a6b8bff" />
<img width="1470" height="733" alt="image" src="https://github.com/user-attachments/assets/f366cd21-30e1-4959-a530-a703d72ab4b1" />


- **flight_agent** - queries the AviationStack MCP server for airport/airline data, then asks the LLM for route and pricing guidance.
- **hotel_agent** - searches for hotels via the Tavily MCP server.
- **weather_agent** - extracts the destination city with the LLM, then fetches current conditions and a forecast from a custom OpenWeatherMap MCP server.
- **sightseeing_agent** - fetches nearby attractions from a custom OpenTripMap MCP server.
- **itinerary_agent** - combines all of the above into a final itinerary.

<img width="1600" height="3000" alt="image" src="./AI-Travel-Booking-System-08-19-2026_05_18_PM.png" />
 

**Observability**
<img width="1470" height="733" alt="image" src="./AI-Travel-Booking-System-08-19-2026_05_20_PM.png" />
 

**Observability with Langsmith**
<img width="1470" height="733" alt="image" src="./ai-travel-planner-LangSmith-08-19-2026_05_21_PM.png" />
 

State (including token/LLM-call counters and search history) is checkpointed to PostgreSQL via `langgraph-checkpoint-postgres`.

A full architecture diagram is available in [Travel_Planner_Architecture.pptx](./Travel_Planner_Architecture.pptx).

## Features

- Multi-agent architecture built with LangGraph
- PostgreSQL-backed conversation memory and search history
- Live flight data via an AviationStack MCP server
- Live hotel search via the Tavily MCP server
- Live weather + forecast via a custom OpenWeatherMap MCP server
- Live sightseeing/attractions via a custom OpenTripMap MCP server
- LLM access through OpenRouter (configurable model)
- Streamlit web UI with PDF/DOCX itinerary export
- Optional LangSmith tracing

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | LangGraph graph definition, agent nodes, and CLI entry point |
| `mcp_client.py` | MCP client setup and tool-calling helpers for all servers |
| `frontend.py` | Streamlit web application |
| `export_utils.py` | Markdown -> PDF/DOCX export helpers for itineraries |
| `custom_weather_mcp_server.py` | MCP server exposing current weather + forecast (OpenWeatherMap) |
| `custom_sightseeing_mcp_server.py` | MCP server exposing nearby attractions (OpenTripMap) |
| `aviationstack-mcp/` | AviationStack MCP server (local dependency, run via its own venv) |

## Requirements

### APIs

- [OpenRouter](https://openrouter.ai/) - LLM access
- [Tavily](https://www.tavily.com/) - hotel/web search MCP
- [AviationStack](https://aviationstack.com/) - flight data
- [OpenWeatherMap](https://openweathermap.org/) - weather data
- [OpenTripMap](https://opentripmap.io/) - sightseeing data
- [LangSmith](https://smith.langchain.com/) - optional, for tracing

### Tools

- Python 3.11+
- [PostgreSQL](https://www.postgresql.org/download/) - persistent memory/checkpointing
- [uv](https://docs.astral.sh/uv/) - used to install/run the AviationStack MCP server

## Setup

### 1. Create a Python environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up PostgreSQL

Install PostgreSQL, then create a database:

```sql
CREATE DATABASE langgraph_memory_demo;
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
OPENWEATHER_API_KEY=your_openweathermap_api_key
OPENTRIPMAP_API_KEY=your_opentripmap_api_key

DATABASE_URL=postgresql://postgres:postgres@localhost:5432/langgraph_memory_demo

# Optional - enables LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=ai-travel-planner
```

### 5. Set up the AviationStack MCP server

This runs as a local stdio MCP server with its own virtual environment.

```bash
git clone https://github.com/Pradumnasaraf/aviationstack-mcp.git
cd aviationstack-mcp

pip install uv        # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                # creates aviationstack-mcp/.venv and installs dependencies
```

Add `AVIATION_STACK_API_KEY=your_api_key_here` to `aviationstack-mcp/.env`.

`mcp_client.py` invokes this server directly via `aviationstack-mcp/.venv/bin/python`, so no separate process needs to be started manually - the graph launches it on demand.

The weather and sightseeing MCP servers (`custom_weather_mcp_server.py`, `custom_sightseeing_mcp_server.py`) are also launched automatically as stdio subprocesses; nothing extra needs to be run for them.

## Running the App

### CLI

```bash
python main.py
```

### Streamlit web app

```bash
streamlit run frontend.py
```

## Example Prompt

```
Plan a complete 7 day Japan trip including flights, hotels and sightseeing under 2 lakhs.
```
