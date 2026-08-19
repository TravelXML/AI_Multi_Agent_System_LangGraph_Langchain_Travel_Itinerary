# pip install mcp requests
# Free tourist-attraction data via OpenTripMap (https://opentripmap.io) — 5,000 req/day free tier.

from mcp.server.fastmcp import FastMCP
import requests
import os

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Sightseeing Server")

OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY")
BASE_URL = "https://api.opentripmap.com/0.1/en/places"


@mcp.tool()
def get_attractions(city: str):

    geo = requests.get(
        f"{BASE_URL}/geoname",
        params={"name": city, "apikey": OPENTRIPMAP_API_KEY},
    ).json()

    if "lat" not in geo or "lon" not in geo:
        return {"error": f"Could not locate '{city}'"}

    places = requests.get(
        f"{BASE_URL}/radius",
        params={
            "radius": 15000,
            "lon": geo["lon"],
            "lat": geo["lat"],
            "kinds": "interesting_places",
            "rate": 3,
            "format": "json",
            "limit": 15,
            "apikey": OPENTRIPMAP_API_KEY,
        },
    ).json()

    attractions = [
        {
            "name": p.get("name"),
            "kinds": p.get("kinds"),
            "rate": p.get("rate"),
        }
        for p in places
        if p.get("name")
    ]

    return {"city": geo.get("name", city), "attractions": attractions}


if __name__ == "__main__":
    mcp.run()
