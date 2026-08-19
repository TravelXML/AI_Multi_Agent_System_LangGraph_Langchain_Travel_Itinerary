import os
import asyncio
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

#load_dotenv()
load_dotenv(override=True)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY")

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
AVIATIONSTACK_MCP_PYTHON = os.path.join(
    REPO_DIR, "aviationstack-mcp", ".venv", "bin", "python"
)
WEATHER_MCP_SCRIPT = os.path.join(REPO_DIR, "custom_weather_mcp_server.py")
SIGHTSEEING_MCP_SCRIPT = os.path.join(REPO_DIR, "custom_sightseeing_mcp_server.py")

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },

        "aviationstack": {
            "transport": "stdio",
            "command": AVIATIONSTACK_MCP_PYTHON,
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run"
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
            }
        },

        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                WEATHER_MCP_SCRIPT
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
            }
        },

        "sightseeing": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                SIGHTSEEING_MCP_SCRIPT
            ],
            "env": {
                "OPENTRIPMAP_API_KEY": OPENTRIPMAP_API_KEY
            }
        }



    }

)


# tools discovery
# async def main():

#     tools = await client.get_tools()

#     print("\nAvailable MCP Tools:\n")

#     for tool in tools:
#         print(tool.name)


# async def main():
#     tools = await client.get_tools()

#     search_tool = next(
#         tool
#         for tool in tools
#         if tool.name == "tavily_search"
#     )

#     result = await search_tool.ainvoke(
#         {
#             "query": "Best hotels in Delhi"
#         }
#     )

#     print(result)

# asyncio.run(main()) 


# search_tool = None

# async def initialize_mcp():
#     global search_tool
#     if search_tool is not None:
#         return

#     tools = await client.get_tools()
#     print("\nAvailable MCP Tools:")

#     for tool in tools:
#         print(tool.name)

#     search_tool = next(
#         tool
#         for tool in tools
#         if tool.name == "tavily_search"
#     )



search_tool = None
aviation_tools = {}

async def initialize_mcp():

    global search_tool
    global aviation_tools

    if search_tool is not None and aviation_tools:
        return

    tools = await asyncio.wait_for(client.get_tools(), timeout=MCP_CALL_TIMEOUT)

    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(tool.name)

    search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )

    aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name != "tavily_search"
    }





MCP_CALL_TIMEOUT = 30


async def tavily_mcp_search(query: str):
    await initialize_mcp()
    result = await asyncio.wait_for(
        search_tool.ainvoke({"query": query}),
        timeout=MCP_CALL_TIMEOUT
    )
    return result




async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):

    tools = await asyncio.wait_for(
        client.get_tools(server_name="aviationstack"),
        timeout=MCP_CALL_TIMEOUT
    )

    tool = next(
        t for t in tools
        if t.name == tool_name
    )

    result = await asyncio.wait_for(
        tool.ainvoke(tool_args or {}),
        timeout=MCP_CALL_TIMEOUT
    )

    return result



async def get_airports():

    await initialize_mcp()

    tool = aviation_tools.get("list_airports")

    if not tool:
        return "Airport tool unavailable"

    result = await tool.ainvoke({})

    return result


async def get_airlines():

    await initialize_mcp()

    tool = aviation_tools.get("list_airlines")

    if not tool:
        return "Airline tool unavailable"

    result = await tool.ainvoke({})

    return result





weather_tool = None
forecast_tool = None


async def initialize_weather_tools():

    global weather_tool, forecast_tool

    if weather_tool is not None:
        return

    tools = await asyncio.wait_for(
        client.get_tools(server_name="weather"),
        timeout=MCP_CALL_TIMEOUT
    )

    weather_tool = next(
        t for t in tools
        if t.name == "get_current_weather"
    )

    forecast_tool = next(
        t for t in tools
        if t.name == "get_forecast"
    )


async def weather_mcp_search(city: str):

    await initialize_weather_tools()

    return await asyncio.wait_for(
        weather_tool.ainvoke({"city": city}),
        timeout=MCP_CALL_TIMEOUT
    )


async def forecast_mcp_search(city: str):

    await initialize_weather_tools()

    return await asyncio.wait_for(
        forecast_tool.ainvoke({"city": city}),
        timeout=MCP_CALL_TIMEOUT
    )


sightseeing_tool = None


async def initialize_sightseeing_tools():

    global sightseeing_tool

    if sightseeing_tool is not None:
        return

    tools = await asyncio.wait_for(
        client.get_tools(server_name="sightseeing"),
        timeout=MCP_CALL_TIMEOUT
    )

    sightseeing_tool = next(
        t for t in tools
        if t.name == "get_attractions"
    )


async def sightseeing_mcp_search(city: str):

    await initialize_sightseeing_tools()

    return await asyncio.wait_for(
        sightseeing_tool.ainvoke({"city": city}),
        timeout=MCP_CALL_TIMEOUT
    )




from langchain_openai import ChatOpenAI

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
    global llm
    llm = _build_llm(model)

###################################
# Destination Extractor
###################################

def extract_destination(query: str):

    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)
    meta = getattr(response, "usage_metadata", None) or {}

    return response.content.strip(), meta.get("total_tokens", 0)


if __name__ == "__main__":
    asyncio.run(main())