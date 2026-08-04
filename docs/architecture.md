# Arquitectura de Jaull

Aquest document descriu **les capes** en què està organitzat el projecte i **les regles de dependència** entre elles. `docs/Workflow.md` explica *què fa* el pipeline; aquest document explica *com està muntat* i per què les fronteres estan on estan.

Jaull segueix sent avui un **monòlit modular Python**: no hi ha workers, ni Docker, ni `llama.cpp` real, ni un benchmark worker. El propòsit d’aquesta arquitectura és deixar la casa endreçada perquè el següent cicle (integració real de runtimes, execució remota) pugui afegir-se sense arrossegar cicles.

## Diagrama de capes

```
                     ┌──── CLI ────┐         ┌──── TUI ────┐
                     └──────┬──────┘         └──────┬──────┘
                            └─────── AdvisorService ─────────┐
                                     │                       │
                    Workflow (orquestrador guiat) ───────────┤
                                     │                       │
             ┌─────── Discovery ─────┼─── Recommendation ────┤
             │              (comparteixen contractes via)     │
             └────────────────── Domain  ────────────────────┘
                                     ▲
              Estimator · Metadata · HuggingFace · Hardware · Runtime
                                     ▲
                     Reporting · Diagnostics · Presentation
```

Les fletxes representen imports permesos, no fluxos de dades.

## Paquets

| Paquet | Responsabilitat | Depèn de |
|---|---|---|
| `domain/` | Models Pydantic i enums compartits; polítiques constants; heurístiques pures (families, licenses) | — |
| `hardware/` | Detecció local (psutil, NVML) | `domain/` |
| `huggingface/` | Client HTTP contra el Hub i parseig d’URLs | `domain/` |
| `metadata/` | Lectura de headers safetensors i GGUF | `domain/`, `huggingface/` |
| `estimator/` | Càlcul de memòria, selecció de variant, compatibilitat | `domain/`, `metadata/`, `huggingface/`, `runtime/` |
| `runtime/` | Recomanació de runtime (llama.cpp / Transformers / vLLM) | `domain/` |
| `discovery/` | Consulta al Hub, filtratge, enriquiment, agrupació en sèries | `domain/`, `huggingface/`, `estimator/`, `metadata/` |
| `recommendation/` | Scoring, ranking, explicacions, capability | `domain/`, `estimator/` |
| `workflow/` | Orquestrador del guided run (síncron, amb progrés i cancel·lació) | `domain/`, `discovery/`, `recommendation/`, `estimator/`, `hardware/`, `huggingface/`, `metadata/` |
| `reporting/` | Serialització JSON i Markdown dels resultats | `domain/`, `workflow.state` |
| `diagnostics/` | Comprovacions d’entorn (Python, xarxa, HF, NVML, cache) | `domain/`, `hardware/` |
| `advisor/` | Fachada d’aplicació que empaqueta els serveis anteriors | tot el que hi ha per sota |
| `presentation/` | Renderitzat Rich (taules, panells) | `domain/`, `reporting/` |
| `cli/` | Subcomandes Typer, entrypoint | `advisor/`, `presentation/`, `domain/` |
| `tui/` | Pantalles Textual, entrypoint | `advisor/`, `domain/` |

## Regles de dependència (dures)

1. **`domain/` no importa mai res d’una capa superior.** És el fons de la pila.
2. **`discovery/` i `recommendation/` no s’importen mútuament.** Els contractes que necessiten compartir (candidats, polítiques, families, licenses) viuen a `domain/`.
3. **`discovery/` i `recommendation/` no importen `workflow/`.** Reben `UserRequirements` (a `domain/`) i retornen resultats; l’orquestració és responsabilitat de `workflow/`.
4. **`recommendation/` no importa `presentation/`.** La serialització viu a `reporting/`; el renderitzat Rich viu a `presentation/`; la lògica de ranking no coneix cap dels dos.
5. **`cli/` i `tui/` no s’importen l’un a l’altre.** L’única fachada compartida és `AdvisorService`.
6. **`workflow/` pot orquestrar;** `advisor/` és qui la CLI i la TUI toquen — mai construeixen `HfClient()`, `detect_hardware`, `estimate_memory` o `collect_diagnostics` directament.

Aquestes regles es poden verificar amb `grep`:

```bash
# Cap import creuat entre discovery i recommendation
grep -rn "from jaull.recommendation" src/jaull/discovery/
grep -rn "from jaull.discovery"      src/jaull/recommendation/

# Ni workflow des de discovery/recommendation
grep -rn "from jaull.workflow"       src/jaull/discovery/ src/jaull/recommendation/

# Ni presentation des de recommendation
grep -rn "from jaull.presentation"   src/jaull/recommendation/

# Ni cli des de tui
grep -rn "from jaull.cli"            src/jaull/tui/
```

Totes aquestes consultes han de retornar zero coincidències.

## `AdvisorService`

`src/jaull/advisor/service.py` conté la fachada única que **tota** la CLI i **totes** les pantalles TUI utilitzen. Els seus mètodes cobreixen les cinc operacions de l’aplicació:

- `scan_hardware(on_progress=None)` — perfil local, opcionalment amb progrés per passes.
- `diagnostics()` — llista de `DiagnosticResult`.
- `inspect_model(repo_id)` — anàlisi d’un repositori.
- `estimate_model(analysis, hardware, inference_cfg, ...)` — estimació de memòria completa.
- `recommend(answers, hardware=None, on_progress=None, is_cancelled=None)` — guided run end-to-end.

Dues fàbriques:

- `AdvisorService.default()` — muntatge de producció (`ServiceContainer.default()`).
- `AdvisorService.build(hf_client=..., detect_hardware=..., inspect_model=..., estimate_memory=..., collect_diagnostics=...)` — muntatge de test, amb tots els serveis com a callables injectats.

Els screens TUI accedeixen a l’advisor via `self.app.advisor`; les funcions CLI l’accepten com a paràmetre opcional (`advisor: AdvisorService | None = None`) i cauen a `AdvisorService.default()` quan no se’n rep cap.

## Composició de dependències

`workflow/container.py::ServiceContainer` continua sent el contenidor de serveis que fa servir el `workflow` per parametritzar HTTP client, capability analyzer, range client factory, etc. L’`AdvisorService` **el conté**, no el substitueix — d’aquesta manera els tests que abans construïen un `ServiceContainer` de mentida per exercir el guided run segueixen funcionant sense canvis, i els que ara construeixen un `AdvisorService` de test poden usar `AdvisorService.build(...)`.

## Reporting i serialització

`jaull.reporting.estimation.estimate_to_json_dict` és el **únic** productor de la representació JSON d’una `MemoryEstimate`. `presentation/estimation_report.py` la re-exporta per compatibilitat, però ja no en conté una còpia.

`jaull.reporting.recommendation.report_to_json`/`report_to_markdown` són les úniques funcions que construeixen l’informe complet del guided run. `recommendation/report.py` només és un shim de compatibilitat que re-exporta d’allà.

Els contractes de compatibilitat són **byte-idèntics**: `tests/test_reporting_regression.py` compara la sortida JSON i Markdown contra `tests/snapshots/report.json` i `tests/snapshots/report.md`. Qualsevol canvi que trenqui la igualtat byte a byte ha d’actualitzar `REPORT_SCHEMA_VERSION` explícitament.

## Convencions

- **Sense excepció `Exception` nua.** Sempre atrapem un tipus específic de `jaull.exceptions` o d’una llibreria concreta (`OSError`, `ImportError`, …).
- **Python 3.12+**: `X | None`, `list[str]`, `type` genèric, sense `Union`/`Optional` de `typing`.
- **Pydantic v2 frozen models** a `domain/`. Cap classe muta l’estat després de la construcció.
- **Sense singletons globals mutables**: el contenidor de serveis i l’advisor es construeixen al punt d’entrada i s’injecten cap avall.

## Treball pendent (fora d’aquest cicle)

- Docker / Docker Compose.
- Integració real amb `llama.cpp`, descàrrega de pesos i benchmark dels finalistes.
- HTTP intern entre `Advisor` i un `Executor` remot.
- Capturas de `docs/assets/tui-*.png` amb el nom nou (es regeneraran quan la TUI s’estabilitzi).
