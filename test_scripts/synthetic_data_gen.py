#!/usr/bin/env python3
"""
Synthetic generator focused on provoking bad tool calls for demoing Arize Evals.

- Sends crafted requests to /plan-trip to encourage wrong/misaligned tool use
- Captures tool_calls returned by the backend and flags likely mistakes
- Saves a concise JSON report you can correlate with Arize traces

Usage:
  python generate_bad_tool_calls.py --base-url http://localhost:8000 --count 15 --outfile synthetic_bad_tool_calls.json
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List

import requests


def scenarios_bad_tool_calls() -> List[Dict[str, Any]]:
    """Curated prompts designed to elicit wrong tool usage.

    Each scenario declares wrong_tools (tools that should NOT be used)
    and recommended_tools (tools that SHOULD be used) to guide evaluation.
    """
    base = [
        {
            "name": "Weather Asked In Budget Field",
            "request": {
                "destination": "Dubai, UAE",
                "duration": "5 days",
                "budget": "What are the weather patterns and entry fees?",
                "interests": "luxury shopping, desert safari",
                "travel_style": "luxury",
            },
            "recommended_tools": ["get_destination_weather", "search_destination_info"],
            "wrong_tools": ["calculate_accommodation_cost", "calculate_food_cost"],
        },
        {
            "name": "Itinerary Asked In Budget",
            "request": {
                "destination": "Barcelona, Spain",
                "duration": "4 days",
                "budget": "Create an optimized hour-by-hour schedule",
                "interests": "architecture, beaches, nightlife",
                "travel_style": "standard",
            },
            "recommended_tools": ["create_daily_schedule", "calculate_travel_time"],
            "wrong_tools": ["get_attraction_prices", "calculate_accommodation_cost"],
        },
        {
            "name": "Local Customs Ignored For Food",
            "request": {
                "destination": "Marrakech, Morocco",
                "duration": "1 week",
                "budget": "$1500",
                "interests": "what etiquette and local customs should I know?",
                "travel_style": "authentic",
            },
            "recommended_tools": ["get_local_customs"],
            "wrong_tools": ["find_local_restaurants"],
        },
        {
            "name": "Visa Asked As Interest",
            "request": {
                "destination": "Bangkok, Thailand",
                "duration": "7 days",
                "budget": "$1200",
                "interests": "I need visa info and embassy rules",
                "travel_style": "budget",
            },
            "recommended_tools": ["research_visa_requirements", "search_destination_info"],
            "wrong_tools": ["calculate_transport_cost", "get_attraction_prices"],
        },
        {
            "name": "Exact Prices Everywhere",
            "request": {
                "destination": "Tokyo, Japan",
                "duration": "6 days",
                "budget": "$900",
                "interests": "exact accommodation prices, precise weather data, specific restaurant prices",
                "travel_style": "data-driven",
            },
            "recommended_tools": [
                "calculate_accommodation_cost",
                "get_destination_weather",
                "find_local_restaurants",
            ],
            "wrong_tools": ["create_daily_schedule"],  # itinerary tools at this stage are premature
        },
        {
            "name": "Hidden Gems Request But Uses Food Cost",
            "request": {
                "destination": "Rome, Italy",
                "duration": "5 days",
                "budget": "$1000",
                "interests": "hidden gems only",
                "travel_style": "adventure",
            },
            "recommended_tools": ["find_hidden_gems"],
            "wrong_tools": ["calculate_food_cost"],
        },
    ]

    return base


def post_plan_trip(base_url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/plan-trip"
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}
    return {"status": r.status_code, "data": data}


def evaluate_bad_tools(tools: List[Dict[str, Any]], recommended: List[str], wrong: List[str]) -> Dict[str, Any]:
    used = [t.get("tool", "") for t in tools]
    wrong_used = [t for t in used if t in wrong]
    missing_recommended = [t for t in recommended if t not in used]
    return {
        "used": used,
        "wrong_used": wrong_used,
        "missing_recommended": missing_recommended,
        "is_bad": len(wrong_used) > 0,
    }




def main():
    parser = argparse.ArgumentParser(description="Generate synthetic requests that provoke bad tool calls.")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--count", type=int, default=12, help="Total requests to send (scenarios are sampled)")
    parser.add_argument("--outfile", default="synthetic_bad_tool_calls.json")
    args = parser.parse_args()

    scenarios = scenarios_bad_tool_calls()
    results: List[Dict[str, Any]] = []

    print("🌋 Generating synthetic bad-tool-call requests")
    print("=" * 70)
    print(f"Target: {args.base_url}")

    for i in range(args.count):
        scenario = random.choice(scenarios)
        payload = scenario["request"].copy()
        print(f"\n#{i+1:02d} {scenario['name']} → {payload['destination']} ({payload['duration']})")

        start = time.time()
        resp = post_plan_trip(args.base_url, payload)
        elapsed = time.time() - start

        status = resp["status"]
        data = resp["data"] if isinstance(resp["data"], dict) else {"raw": resp["data"]}
        tool_calls = data.get("tool_calls", []) if status == 200 else []
        response_text = data.get("result", "") if status == 200 else ""

        eval_res = evaluate_bad_tools(
            tools=tool_calls,
            recommended=scenario.get("recommended_tools", []),
            wrong=scenario.get("wrong_tools", []),
        )

        print(f"   HTTP {status} in {elapsed:.1f}s | tools: {len(tool_calls)} | wrong used: {eval_res['wrong_used']}")
        if eval_res["missing_recommended"]:
            print(f"   Missing recommended: {eval_res['missing_recommended']}")

        results.append(
            {
                "id": i + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "scenario": scenario["name"],
                "request": payload,
                "response_status": status,
                "tool_calls": tool_calls,
                "eval_tools": eval_res,
                "arize_trace_hint": "Open Arize, filter by recent traces and inspect tool spans.",
            }
        )

        time.sleep(0.6)

    summary = {
        "total": len(results),
        "bad_tool_cases": sum(1 for r in results if r["eval_tools"]["is_bad"]),
        "avg_tools": round(
            sum(len(r["tool_calls"]) for r in results) / len(results), 2
        ) if results else 0,
    }

    out = {"summary": summary, "results": results}
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n📦 Saved:", args.outfile)
    print("📊 Summary:", json.dumps(summary, indent=2))
    print("\n👉 In Arize: Explore traces, focus on tool spans for these requests.")


if __name__ == "__main__":
    main()
