import ollama

# 1. Define your tools
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: first number
        b: second number
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: first number
        b: second number
    """
    return a * b

# 2. Map function names to their Python callables
AVAILABLE_TOOLS = {
    "add": add,
    "multiply": multiply,
}

# List of tool functions passed to Ollama
TOOLS_LIST = list(AVAILABLE_TOOLS.values())

# Maintain chat memory across turns
messages = [
    {"role": "system", "content": "You are a helpful assistant with access to local tools."}
]

print("--- Chat Loop Started (type 'exit' or 'quit' to stop) ---")

while True:
    user_input = input("\nUser: ").strip()
    if user_input.lower() in ["exit", "quit", "q"]:
        break

    if not user_input:
        continue

    # Add user message to conversation history
    messages.append({"role": "user", "content": user_input})

    # Turn processing loop: repeat until the LLM produces a final text response
    while True:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=messages,
            tools=TOOLS_LIST
        )

        # Append LLM's response (tool call request or final text) to history
        messages.append(response.message)

        # If the LLM requested tool calls, execute them and stay in the inner loop
        if response.message.tool_calls:
            for call in response.message.tool_calls:
                fn_name = call.function.name
                fn_args = call.function.arguments

                print(f"  [Executing Tool] -> {fn_name}(**{fn_args})")

                if fn_name in AVAILABLE_TOOLS:
                    try:
                        result = AVAILABLE_TOOLS[fn_name](**fn_args)
                    except Exception as e:
                        result = f"Error executing {fn_name}: {e}"
                else:
                    result = f"Error: Tool '{fn_name}' is not registered."

                # Append tool result back to history
                messages.append({
                    "role": "tool",
                    "tool_name": fn_name,
                    "content": str(result),
                })
        else:
            # No tool calls requested; LLM gave a final response
            print(f"Assistant: {response.message.content}")
            break