# llm-client

Capa de cliente LLM lista para producción sobre el SDK de Anthropic. No es un
`messages.create()` suelto: es lo que lo rodea —streaming, retries, fallback,
timeouts y observabilidad de coste/tokens— que hace que un workflow no se caiga
cuando el proveedor devuelve un 429/529 bajo carga.

## Por qué existe

Todo sistema GenAI productivo se apoya en una capa de cliente fiable y observable.
Sin ella, RAG, agentes y evals se construyen sobre arena. Este repo es el primer
ladrillo de un futuro *model gateway*.

## Instalar y correr

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q          # 6 tests, sin gastar API (SDK mockeado)
```

Ejecución real (requiere `ANTHROPIC_API_KEY`):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python examples/real_run.py
```

## Contrato

`ProductionLLMClient.complete()` **nunca** propaga una excepción de red/SDK hacia
arriba. Devuelve siempre un `LLMResult` con `status`:

| status | significado |
|---|---|
| `ok` | respuesta válida (primario o fallback) |
| `error` | error **permanente** (400/401/403/404/422): la request hay que corregirla |
| `unavailable` | retries de primario **y** fallback agotados: degradación elegante |

## Decisiones y trade-offs

- **Solo se reintentan los transitorios** (429, 529, errores de conexión). Los
  permanentes fallan rápido: reintentarlos da el mismo error y quema latencia/dinero.
- **El fallback se dispara solo en transitorios**, no en permanentes (otro modelo
  no arregla un prompt mal formado).
- **`max_retries` configurable (default 3).** Más retries = más resiliencia pero
  peor latencia de cola (p99): cada retry suma espera + nueva llamada.
- **Timeout explícito (default 30s).** Corto protege p99 pero mata llamadas lentas
  legítimas; largo evita falsos negativos pero deja colgado el workflow.
- **Coste con precios fechados** (`pricing.py`, verificados 2026-06-30). Modelo
  desconocido → coste 0 (no inventamos precios). Los precios cambian: revísalos.

## Riesgos conocidos (lo que aún NO resuelve)

- **El fallback a modelo más barato puede degradar calidad en silencio.** Para un
  *quality gate* (p.ej. code review) esto es peligroso: preferible `unavailable` +
  revisión humana que una review peor tratada como fiable. Esta política de fallback
  por-tarea es trabajo futuro.
- **El trace registra el `LLMResult` completo**: no debe incluir PII en `text` en un
  entorno real (pendiente para C-E6).
- Coste por *request*, no por *workflow*: la agregación por unidad de negocio se
  construye encima (C-F1).

## Cómo se evalúa

- `tests/test_retries.py` — transitorio en primario → fallback → `ok`.
- `tests/test_degradation.py` — primario + fallback caen → `unavailable` (sin excepción);
  permanente → `error` sin intentar fallback.
- `tests/test_cost.py` — coste correcto; lectura de cache = 0.1× del input base.
- CI (`.github/workflows/ci.yml`) corre la suite en cada push/PR.

## Ejecución real (trace end-to-end)

> Pega aquí la salida de `python examples/real_run.py` tras configurar la key.

```text
(pendiente: ejecutar y pegar el trace de status/model/tokens/latency/cost)
```

## Siguiente versión

Política de fallback por-tarea, async para concurrencia, prompt caching
instrumentado de verdad (C-A4), y redacción de PII en el trace (C-E6).
