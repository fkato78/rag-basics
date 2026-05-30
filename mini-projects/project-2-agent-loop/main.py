import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)


def get_weather(location: str) -> str:
    weather_data = {
        "San Francisco, CA": {"temp": 15, "unit": "celsius", "condition": "foggy"},
        "New York, NY": {"temp": 22, "unit": "celsius", "condition": "sunny"},
        "London, UK": {"temp": 11, "unit": "celsius", "condition": "rainy"},
    }
    data = weather_data.get(
        location, {"temp": 20, "unit": "celsius", "condition": "unknown"}
    )
    return json.dumps(data)


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    if from_unit == "celsius" and to_unit == "fahrenheit":
        result = (value * 9 / 5) + 32
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        result = (value - 32) * 5 / 9
    else:
        result = value
    return json.dumps({"value": round(result, 1), "unit": to_unit})


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "Convert a temperature value between celsius and fahrenheit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Temperature value"},
                    "from_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "to_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["value", "from_unit", "to_unit"],
                "additionalProperties": False,
            },
        },
    },
]

available_functions = {
    "get_weather": lambda args: get_weather(args["location"]),
    "convert_temperature": lambda args: convert_temperature(
        args["value"], args["from_unit"], args["to_unit"]
    ),
}


def agent_loop(user_message: str, max_iterations: int = 10) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when needed.",
        },
        {"role": "user", "content": user_message},
    ]
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-5-mini", messages=messages, tools=tools
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        if not assistant_message.tool_calls:
            return assistant_message.content
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            func = available_functions.get(func_name)
            result = func(func_args) if func else json.dumps({"error": "unknown"})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
    return "Max iterations reached."


if __name__ == "__main__":
    answer = agent_loop("What's the weather in San Francisco in Fahrenheit?")
    print(answer)
