"""Ejecución real contra la Claude API. Requiere ANTHROPIC_API_KEY en el entorno.

    python examples/real_run.py

Copia la salida (el trace de coste/tokens) al README como prueba end-to-end.
"""
import logging

from llm_client import ProductionLLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    client = ProductionLLMClient()
    res = client.complete(
        system="Eres conciso.",
        messages=[{"role": "user", "content": "Di 'hola' en una sola palabra."}],
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
