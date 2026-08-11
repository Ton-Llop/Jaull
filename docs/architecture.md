# Arquitectura de Jaull

Aquest document descriu **les capes** en què està organitzat el projecte i **les regles de dependència** entre elles. `docs/Workflow.md` explica *què fa* el pipeline; aquest document explica *com està muntat* i per què les fronteres estan on estan.

Jaull segueix sent avui un **monòlit modular Python**: no hi ha workers, ni Docker, ni benchmark worker. La recomanació guiada continua sent metadata-only, però ja existeix un camí explícit d’execució local: `jaull run` resol, descarrega, verifica i executa artefactes GGUF single-file amb `llama-cli`. El propòsit d’aquesta arquitectura és deixar la casa endreçada perquè els següents cicles (benchmarks, streaming de descàrrega, execució remota) puguin afegir-se sense arrossegar cicles.

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
       Estimator · Metadata · HuggingFace · Hardware · Runtime · Artifacts · Execution
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
| `runtime/` | Recomanació de runtime i runner local de llama.cpp | `domain/`, `execution/` |
| `artifacts/` | Resolució, descàrrega, emmagatzematge i verificació d’artefactes executables | `domain/`, `huggingface/` |
| `execution/` | Contractes d’execució i backend host per llançar processos locals | `domain/` |
| `discovery/` | Consulta al Hub, filtratge, enriquiment, agrupació en sèries | `domain/`, `huggingface/`, `estimator/`, `metadata/` |
| `recommendation/` | Scoring, ranking, explicacions, capability | `domain/`, `estimator/` |
| `workflow/` | Orquestrador del guided run (síncron, amb progrés i cancel·lació) | `domain/`, `discovery/`, `recommendation/`, `estimator/`, `hardware/`, `huggingface/`, `metadata/` |
| `reporting/` | Serialització JSON i Markdown dels resultats | `domain/`, `workflow.state` |
| `diagnostics/` | Comprovacions d’entorn (Python, xarxa, HF, NVML, cache) | `domain/`, `hardware/` |
| `advisor/` | Fachada d’aplicació que empaqueta els serveis anteriors | tot el que hi ha per sota |
| `presentation/` | Renderitzat Rich (taules, panells) | `domain/`, `reporting/` |
| `cli/` | Subcomandes Typer, entrypoint; `run` també composa el runner local | `advisor/`, `presentation/`, `domain/`, `runtime/`, `execution/` |
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

`src/jaull/advisor/service.py` conté la fachada que la CLI i les pantalles TUI utilitzen per accedir als serveis d’aplicació. Els seus mètodes cobreixen les operacions principals:

- `scan_hardware(on_progress=None)` — perfil local, opcionalment amb progrés per passes.
- `diagnostics()` — llista de `DiagnosticResult`.
- `inspect_model(repo_id)` — anàlisi d’un repositori.
- `estimate_model(analysis, hardware, inference_cfg, ...)` — estimació de memòria completa.
- `recommend(answers, hardware=None, on_progress=None, is_cancelled=None)` — guided run end-to-end.
- `resolve_artifact(repo_id, quantization=None, revision=None)` — tria un fitxer GGUF executable.
- `download_artifact(artifact)` — baixa l’artefacte al layout local.
- `verify_artifact(artifact, full=False)` — comprova mida i SHA-256.
- `run_artifact(artifact=..., prompt=..., runtime=...)` — executa via el runner configurat.

Dues fàbriques:

- `AdvisorService.default()` — muntatge de producció (`ServiceContainer.default()`).
- `AdvisorService.build(hf_client=..., detect_hardware=..., inspect_model=..., estimate_memory=..., collect_diagnostics=...)` — muntatge de test, amb tots els serveis com a callables injectats.

Els screens TUI accedeixen a l’advisor via `self.app.advisor`; les funcions CLI l’accepten com a paràmetre opcional (`advisor: AdvisorService | None = None`) i cauen a `AdvisorService.default()` quan no se’n rep cap. `cli/run.py` fa servir l’advisor per resoldre/descarregar/verificar artefactes i instancia el runner local amb les opcions específiques de CLI (`--llama-cli`, `--timeout-seconds`, `--ctx-size`, `--n-gpu-layers`).

## Artefactes i execució local

El camí `run` és deliberadament separat de l’estimador i del workflow guiat:

```text
cli/run.py
   ├── normalize_repo_id()
   ├── AdvisorService.resolve_artifact()
   ├── AdvisorService.download_artifact()   # només si falta el fitxer
   ├── AdvisorService.verify_artifact()
   └── LlamaCppRunner(HostExecutionBackend).run()
```

`artifacts/` tradueix un repositori abstracte a un `ModelArtifact` concret i verificat. En aquesta fase només accepta GGUF single-file; repositoris Transformers i GGUF multipart es rebutgen amb errors específics.

`execution/` no coneix Hugging Face ni models: només rep una `ExecutionRequest` immutable i retorna un `ExecutionResult`. Aquest resultat conté stdout/stderr i una `ExecutionObservation`, que és la font de veritat sobre què ha passat realment durant el procés: durada, exit status, peak RSS del procés principal i peak VRAM atribuïda via NVML quan és possible. Això permet provar el runner sense invocar cap binari real i manté separades predicció (`MemoryEstimate`) i observació.

`runtime/llama_cpp_runner.py` valida que l’artefacte sigui GGUF, descarregat, verificat i present en disc abans de construir el comandament `llama-cli --single-turn`.

## Prediction validation

`jaull.evaluation.comparison.compare_prediction` compara una predicció de Jaull
amb una execució real ja observada:

```text
MemoryEstimate + ExecutionObservation -> PredictionComparison
```

La comparació no modifica l'estimador ni calibra fórmules. `MemoryEstimate`
continua representant la predicció, `ExecutionObservation` continua representant
la realitat mesurada i `PredictionComparison` és una tercera peça derivada.

La convenció d'error és única:

```text
error_bytes = measured_bytes - predicted_bytes
error_percent = (measured_bytes - predicted_bytes) / predicted_bytes * 100
```

Un error positiu significa que Jaull ha infraestimat el consum real. Un error
negatiu significa que Jaull ha sobreestimat el consum real.

La comparació RAM només es calcula quan la configuració executada és CPU-only o
no offload. En aquest cas la predicció comparable és la suma dels components que
representen consum de procés (`weights + kv_cache + runtime_overhead`), excloent
`device_reserve` i `safety_margin`, perquè aquests últims són polítiques de
capacitat i no RSS observat. Quan el runtime usa GPU offload, Jaull encara no
conserva un breakdown host/device; per tant `ram.predicted_bytes` queda `null` i
la comparació es marca com `methodologically_unavailable`.

La comparació VRAM també queda `methodologically_unavailable` en aquesta fase:
el model d'estimació actual no conserva VRAM atribuïda al PID/configuració
executada. Si NVML no exposa memòria de procés, `peak_vram_bytes = null` no es
tracta com zero.

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
- Streaming de descàrrega i progrés byte-level per a artefactes grans.
- Benchmarks reals del camí `run` (tokens/s, latència de primer token).
- HTTP intern entre `Advisor` i un `Executor` remot.
- Capturas de `docs/assets/tui-*.png` amb el nom nou (es regeneraran quan la TUI s’estabilitzi).
