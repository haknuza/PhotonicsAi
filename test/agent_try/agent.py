import json
from datetime import datetime
from pathlib import Path

import ollama

from check_claim import check_claim
from run_spectrum import run_spectrum, find_resonances, measure_finesse

MODEL = "qwen2.5:3b"
TOOLS = {f.__name__: f for f in (run_spectrum, find_resonances, measure_finesse)}
MAX_STEPS = 12      # tool-calling steps inside one attempt
MAX_ATTEMPTS = 5    # how many times we retry after a FAIL
TARGET_NM = 1550
TOL_NM = 2

SYSTEM = (
    "You are a photonics research assistant controlling a Fabry-Perot "
    "simulation through tools. "
    "Rules: use the tools for every number and never invent results. "
    "Call ONE tool at a time and wait for its result. "
    "The cavity is in vacuum (n=1). Resonances are at "
    "lambda = 2*L*1000/m nm (L in micrometers, m an integer). "
    "Workflow: "
    "1) run_spectrum. "
    "2) find_resonances (it takes no arguments). "
    "3) Compare the resonances with the target wavelength. If none is within "
    "the tolerance, change the cavity length L and repeat steps 1-3. "
    "4) Only when a resonance is within the tolerance, call measure_finesse "
    "(it takes no arguments). "
    "5) Give a short final report with L and the finesse."
)

GOAL = ("Find a cavity length L between 5 and 15 micrometers, with R = 0.9, "
        "that has a resonance within 2 nm of 1550 nm. Report L and the finesse.")


def to_dict(m):
    """Ollama messages are objects; the ones we built ourselves are dicts."""
    return m if isinstance(m, dict) else m.model_dump()


def run_agent(goal, attempt=1):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": goal}]
    last_L = None
    final_text = None

    for step in range(MAX_STEPS):
        reply = ollama.chat(model=MODEL, messages=messages,
                            tools=list(TOOLS.values()),
                            options={"temperature": 0})
        calls = (reply.message.tool_calls or [])[:1]   # enforce ONE tool per reply
        reply.message.tool_calls = calls
        messages.append(reply.message)

        if not calls:                                   # no tool -> final answer
            final_text = reply.message.content
            break

        call = calls[0]
        name, args = call.function.name, call.function.arguments
        try:
            result = TOOLS[name](**args)
        except Exception as e:                          # unknown tool, bad arguments...
            result = f"Error calling {name}: {e}"

        # remember L only if the simulation really ran
        if name == "run_spectrum" and not str(result).startswith("Error"):
            try:
                last_L = float(args["length_um"])
            except (KeyError, TypeError, ValueError):
                pass

        print(f"[{step}] {name}({args}) -> {result}")
        messages.append({"role": "tool", "tool_name": name,
                         "content": str(result)})

    print("FINAL:", final_text or "(no final answer, MAX_STEPS reached)")

    # independent check (this is NOT shown to the model during the attempt)
    if last_L is None:
        lambda_m, dist, passed = None, None, False
        print("CHECK: no successful run_spectrum call -> FAIL")
    else:
        lambda_m, dist, passed = check_claim(last_L, TARGET_NM, TOL_NM)
        print(f"CHECK: L={last_L} um, nearest resonance {lambda_m:.2f} nm, "
              f"distance {dist:.2f} nm -> {'PASS' if passed else 'FAIL'}")

    # save the log
    log_path = Path("logs") / f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}_attempt{attempt}.json"
    log_path.parent.mkdir(exist_ok=True)
    log = {"model": MODEL, "attempt": attempt, "goal": goal,
           "last_L": last_L, "passed": passed,
           "final_text": final_text,
           "messages": [to_dict(m) for m in messages]}
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    return {"last_L": last_L, "passed": passed, "final_text": final_text,
            "nearest_nm": lambda_m, "distance_nm": dist}


def run_with_retries(goal):
    feedback = ""
    results = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n=== ATTEMPT {attempt} ===")
        result = run_agent(goal + feedback, attempt)
        results.append(result)
        if result["passed"]:
            break
        if result["last_L"] is None:
            feedback = ("\n\nPrevious attempt failed: no simulation was run "
                        "successfully. Start by calling run_spectrum.")
        else:
            feedback = (f"\n\nPrevious attempt failed: with L={result['last_L']} um "
                        f"the nearest resonance was {result['nearest_nm']:.2f} nm, "
                        f"which is {result['distance_nm']:.2f} nm from the target. "
                        f"Use this to choose a better L.")

    first = results[0]["passed"]
    final = results[-1]["passed"]
    print(f"\nSUMMARY: first attempt {'PASS' if first else 'FAIL'}; "
          f"after {len(results)} attempt(s) {'PASS' if final else 'FAIL'}")
    return results


if __name__ == "__main__":
    run_with_retries(GOAL)