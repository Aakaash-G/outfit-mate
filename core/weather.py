"""Weather context from OpenWeatherMap, with a safe default when no key is set."""
from datetime import date

import requests

from .config import OPENWEATHER_API_KEY

DEFAULT = {"temp_c": 28.0, "rain_prob": 0.2, "description": "no forecast (default values)", "source": "default"}


def get_weather(city: str, day: date) -> dict:
    if not OPENWEATHER_API_KEY:
        return dict(DEFAULT)
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                         params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}, timeout=8)
        r.raise_for_status()
        entries = [e for e in r.json()["list"] if e["dt_txt"].startswith(day.isoformat())]
        if not entries:  # forecast covers ~5 days only
            return dict(DEFAULT, description="date outside forecast range (default values)")
        # use the daytime reading closest to 1 pm
        e = min(entries, key=lambda e: abs(int(e["dt_txt"][11:13]) - 13))
        return {
            "temp_c": round(float(e["main"]["temp"]), 1),
            "rain_prob": float(e.get("pop", 0.0)),
            "description": e["weather"][0]["description"],
            "source": "openweathermap",
        }
    except Exception as ex:  # network / key problems should never break the app
        return dict(DEFAULT, description=f"weather unavailable ({type(ex).__name__})")