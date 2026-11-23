# Kimi Multi-Turn Tools

A Python helper for multi-turn function calling with Kimi K3.1 Pro through CometAPI. It preserves complete assistant messages—including tool-call IDs and other returned fields—when replaying conversation history.

> Confirm the current model ID and API behavior in the CometAPI documentation. This example requires live API testing before production use.

## Setup

```bash
pip install openai pytest
export COMETAPI_KEY="your_api_key_here"
```

## Helper

Create `kimi_tools.py`:

```python
import json
import os
from collections.abc import Callable
from typing import Any

from openai import OpenAI

client = OpenAI(
    base_url="https://api.cometapi.com/v1",
    api_key=os.environ["COMETAPI_KEY"],
)


def function_tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def run_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    functions: dict[str, Callable[..., Any]],
    model: str = "kimi-k3.1-pro",
    max_steps: int = 10,
) -> str:
    conversation = list(messages)

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            tools=tools,
        )

        message = response.choices[0].message

        # Replay every field returned in the assistant message.
        conversation.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or ""

        for call in message.tool_calls:
            try:
                name = call.function.name
                arguments = json.loads(call.function.arguments or "{}")

                if name not in functions:
                    raise ValueError(f"Unknown tool: {name}")
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")

                result = functions[name](**arguments)
                payload = {"ok": True, "result": result}
            except Exception as exc:
                payload = {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }

            conversation.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(payload, default=str),
            })

    raise RuntimeError(f"Tool loop exceeded {max_steps} steps")
```

## Demo

Create `demo.py`:

```python
from kimi_tools import function_tool, run_with_tools

STOCK = {"A-100": 12, "B-200": 0}
PRICES = {"A-100": 19.99, "B-200": 34.50}


def get_stock(sku):
    return {"sku": sku, "units_in_stock": STOCK.get(sku, 0)}


def get_price(sku):
    if sku not in PRICES:
        raise KeyError(f"Unknown SKU: {sku}")
    return {"sku": sku, "price_usd": PRICES[sku]}


TOOLS = [
    function_tool(
        "get_stock",
        "Return the number of units currently in stock for a SKU.",
        {"sku": {"type": "string"}},
        ["sku"],
    ),
    function_tool(
        "get_price",
        "Return the current price in US dollars for a SKU.",
        {"sku": {"type": "string"}},
        ["sku"],
    ),
]

FUNCTIONS = {
    "get_stock": get_stock,
    "get_price": get_price,
}

print(run_with_tools(
    messages=[{
        "role": "user",
        "content": (
            "Check A-100 and B-200. Tell me which products are available "
            "and the total price of one unit of each available product."
        ),
    }],
    tools=TOOLS,
    functions=FUNCTIONS,
))
```

Run it:

```bash
python demo.py
```

## Why complete message replay matters

This loses tool calls and other returned fields:

```python
messages.append({
    "role": "assistant",
    "content": message.content,
})
```

Preserve the complete message instead:

```python
messages.append(
    message.model_dump(exclude_none=True)
)
```

Tool results refer to IDs in the assistant message. Removing those fields can break later turns.

## Testing

Use mocked responses for unit tests so routine testing does not consume API credits. Keep live CometAPI tests separate and run them only when explicitly enabled:

```bash
pytest -q
RUN_LIVE_TESTS=1 pytest -q -m integration
```

Do not describe the project as “tested” until its tests exist and pass against the current CometAPI gateway.

## Production safety

- Validate all tool arguments as untrusted input.
- Expose only an explicit allowlist of tools.
- Set timeouts, resource limits and a maximum number of steps.
- Require human approval for payments, deletion, deployment and messaging.
- Never expose unrestricted shell or filesystem access.
- Redact secrets and sensitive exception details from logs and model responses.
- Verify CometAPI’s current model ID, pricing, cache behavior and message requirements.

## Links

- [Kimi K3.1 Pro on CometAPI](https://www.cometapi.com/models/moonshotai/kimi-k3-1-pro/?utm_source=github&utm_medium=distribution&utm_campaign=bom&utm_content=kimi-k3-1-pro)
- [CometAPI documentation](https://apidoc.cometapi.com)

## License

MIT. Add a `LICENSE` file containing the MIT License before publishing.
