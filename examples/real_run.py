"""Real run against the Claude API. Requires ANTHROPIC_API_KEY in the environment.

    python examples/real_run.py

Copy the output (the cost/token trace) into the README as an end-to-end proof.
"""
import logging

from llm_client import ProductionLLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    client = ProductionLLMClient()
    res = client.complete(
        system="Be concise.",
        messages=[{"role": "user", "content": "Say 'hello' in a single word."}],
        max_tokens=20,
    )
    print("-" * 40)
    print("status     :", res.status)
    print("model      :", res.model)
    print("text       :", res.text)
    print("in/out tok :", res.input_tokens, "/", res.output_tokens)
    print("latency_s  :", round(res.latency_s, 3))
    print("cost_usd   :", res.cost_usd)


if __name__ == "__main__":
    main()
