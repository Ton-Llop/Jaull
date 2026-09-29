# RTX 4060 — rèplica de referència B001-R10

Campanya iniciada el 29/09/2026; execució Windows local el 30/09/2026 (UTC+02).
Aquest informe distingeix resultats persistits, límits metodològics i incidències.
No s'ha canviat cap fórmula, ranking, fitxer de `src/` ni test.

## Font i entorn

- HEAD inicial i final: `f14b3e9eb8d8c906bad4efb1484727cc86ee47a3`.
- Estat inicial: working tree net. No s'ha trobat `AGENTS.md` al repositori ni als
  directoris pare comprovats. S'han llegit `docs/experimental-validation.md` i el
  document del baseline B001-R10 abans d'executar l'experiment.
- Jaull 0.1.0, Windows 11 natiu, build 10.0.26200 AMD64. WSL Ubuntu està instal·lat
  però no és l'entorn d'aquesta execució; no s'han emprat executables Windows com
  a substituts d'un runtime Linux.
- Ryzen 7 5800X, 8 nuclis / 16 threads, RAM total 34282192896 bytes.
- RTX 4060, UUID `GPU-bbbc059d-9014-f50c-72d1-511fe91342e6`, driver 616.92,
  CUDA del driver 13.4. VRAM total NVML 8585740288 bytes (8188 MiB).
  Disponible inicial 6673 MiB; snapshot congelat del pla 7128145920 bytes
  (6798 MiB). Són snapshots diferents, no memòria disponible constant.
- Entorn Python inicial 3.14.7 preservat. Entorn experimental a
  `.venv/rtx4060-py312`, Python 3.12.14 i dependències del lockfile.
- `scan-ready.log` i `doctor-ready.log`: exit 0. Doctor confirma runtime i
  dispositiu CUDA. El primer doctor, abans d'instal·lar el runtime, informava
  `runtime_missing`.

## Runtime verificat

llama.cpp oficial b10357, commit
`689e227db485c6b33d061555e74034c93a867649`, versió 10357. Binari Windows x86_64
compilat amb Clang 20.1.8; distribució CUDA 13.3. El baseline té el mateix commit
i versió, però binari Linux compilat amb GNU 13.3.0: no és el mateix build binari.

Paths inicials: `.venv/rtx4060-runtime-b10357/bin/llama-cli.exe` i
`llama-bench.exe`. `--list-devices` del CLI exposa CUDA0 RTX 4060; el probe buit
del bench carrega `ggml-cuda.dll` i imprimeix `build: 689e227db (10357)`.
`llama-bench --version` no és un flag admès en aquest build (exit 1); es conserva
la sortida, sense interpretar-la com una mesura.

SHA-256 dels executables inicials:

- CLI: `75533f89185faffe35eff096365512742bf3f4eb7a80208bd5699651b55a4f3a`.
- Bench: `9dcfdd1a968646e09926e663e67b1e04af9d26e0629bdc23352a86c3a2aeb94f`.
- CUDA DLL: `55336c75802c5fce8a5ea1c06efce71f8dc81b462e59f04d1300debf3aafe043`.

Les descàrregues, certificats i incidència de hardlinks OneDrive estan als logs
`setup*.log`. S'ha usat el magatzem de certificats Windows i mode copy; no s'ha
desactivat la verificació TLS. No s'ha requerit administrador ni fet neteja.

## Artefacte congelat

- Repo `bartowski/Qwen2.5-7B-Instruct-GGUF`.
- Revisió `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, contrastada amb metadata
  local i consulta HF fixada (`pinned-hub-metadata.log`).
- `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, quantització Q4_K_M, 4683074240 bytes.
- SHA-256 local recalculat abans de Validate:
  `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.
- Path real:
  `C:\Users\PC\AppData\Local\jaull\models\bartowski\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-Q4_K_M.gguf`.

Configuració d'inferència copiada del baseline: context 4096, concurrència 1,
KV float16, batch declarat 1. Anàlisi recalculada amb serveis reals i congelada a
`analysis.json`; **no** és el snapshot històric d'anàlisi B001-R4 del baseline.

## UI i flags efectius

S'ha obert `uv run jaull ui`, però el model exacte no apareixia als cinc resultats
del flux guiat. Les accions finals Validate i Benchmark s'han executat sobre
les pantalles de producció amb Textual headless Pilot i `AdvisorService.default`:
cap mock, runner substituït ni observació sintètica. Aquesta és una desviació del
procediment manual de selecció, explícita a `run-index.json`. Les captures SVG i
`run-reference-ui.py` documenten els controls; s'ha activat Save raw runtime logs.
La recomanació d'un únic candidat fix no és una afirmació de ranking global.

Validate executa el CLI amb el path real i:

```text
--ctx-size 4096 --n-gpu-layers 29 --device CUDA0
--no-display-prompt --color off --no-show-timings --simple-io
--single-turn --verbose --prompt "Explain briefly what artificial intelligence is."
```

El pla mostra `--batch-size 1`, però la invocació real l'omet; el log informa
`n_batch=2048`, `n_ubatch=512`. No es presenta batch 1 com a flag aplicat.
No s'ha fixat una llavor: l'output generat no és determinista.

## Records i resultats reals

Experiment ID `exp-54063df9-685d-4b86-a481-acc98e768567`.
Original a `%LOCALAPPDATA%/jaull/experiments/<ID>.json`; còpia a
`records/exp-54063df9-685d-4b86-a481-acc98e768567.json`, sidecar a
`logs/exp-54063df9-685d-4b86-a481-acc98e768567.runtime-log`.
Validate: success true, exit 0, 5.8941184 s; un únic prompt, no test de càrrega
concurrent. CUDA observat a la traça del runtime.

PredictionComparison preservat:

| Mètrica | Predicció | Observació | Comparació |
|---|---:|---:|---|
| VRAM física comparada | 5923133600 B | 4748056986 B | −19.8388%, available |
| Pesos GPU | 4683074240 B | 4370559140 B | −6.6733%, direct |
| KV GPU | 234881024 B | 234881024 B | 0%, direct |
| Overhead / compute | 1005178336 B | 142616822 B | −85.8118%, **proxy**, no calibració |
| RAM | null | RSS 4836134912 B | methodologically_unavailable |

VRAM observada és `runtime_reported_allocation`, `driver_confirmed=false`;
NVML process allocation és null, no zero. Sota WDDM no s'ha obtingut memòria
atribuïda al procés. RAM RSS inclou pàgines mmap del model i no mesura la demanda
del host segons placement. Reserva i marge són pressupost, no allocation física.
Compatibilitat prevista `tight`, runnable true, resultat `correct_success`.

Primer Benchmark ID `bench-491757e3-204b-4956-b89e-61d39ce2f76b`.
Original a `%LOCALAPPDATA%/jaull/benchmarks/<ID>.json`; còpia a `records/<ID>.json`
i raw log a `logs/<ID>.runtime-log`.
Metodologia `llama_bench_v1`, `-dev CUDA0 -ngl 29 -p 128,512 -n 64 -r 3`.
Context 4096 és **provenance**, no s'ha passat `--ctx-size` al bench.
Record: success true, exit 0, durada 73.5882845 s. Sortida **incompleta**:
pp128 2207.29 ± 87.84 t/s; pp512 2406.60 ± 45.78 t/s; manca tg64 i footer.
No hi ha mesura de generació, TTFT, model load o temps de warmup; tots resten
absents. No s'assimila aquesta durada al throughput ni al temps de Validate.

## Cas i bundle inicials

ExperimentalCase `case-0e4d5682-34bb-4609-ac99-1936b9988f49`, manifest exportat a
`bundle/case.json`, bundle a `bundle/`, manifest d'integritat `bundle/bundle.json`.
Inclou els dos IDs anteriors i els dos logs. Comandos del document executats:
case create, case validate, case export i case bundle validate, tots exit 0.
Resultat de cas i bundle: **partial**, perquè falta el runtime build al
BenchmarkRecord. Integritat `valid`, tots els fitxers declarats verificats.
Sortides natives a `case-*.stdout.json` i `bundle-validate.stdout.json`.
No s'han emplenat manualment camps absents per convertir partial en valid.
Logs revisats: prompt fix d'IA i output innocu; cerca automàtica de credencials
sense coincidències, no garantia absoluta. Hashes a `log-review.json`.

## Gates (Python 3.12.14)

| Gate | Resultat real |
|---|---|
| `uv run --python 3.12 ruff check .` | FAIL: primer 2 B009; final 5 infraccions (2 B009 + 3 E501), només als helpers nous de la campanya |
| `uv run --python 3.12 mypy src` | PASS: 237 fitxers |
| `uv run --python 3.12 pytest` | FAIL: 1753 pass, 1 fail, 129.93 s |
| Repetició completa única | FAIL: 1751 pass, 3 fail, 109.27 s |
| `python -m compileall -q src` | PASS, Python 3.12.14 |
| `git diff --check` | PASS |

Els quatre architecture tests passen dins de les suites. El test de navegació
fallit és `test_details_show_plan_assessment_instead_of_legacy_score`; passa sol
(1 test, 1.99 s), però falla a les dues suites. La repetició afegeix dues fallades
`test_guided_flow_stays_usable_at[size0/size2]` (botó wizard fora del viewport).
La navegació repetida mostra `NoMatches` a `WorkflowHeader.on_mount` cercant
`#workflow-machine`. Problemes de UI/layout/muntatge, no de CUDA; intermitència
observada, causa definitiva no provada. No s'han relaxat tests ni corregit el
producte per ocultar-los. Ruff és una fallada del scaffolding nou, no de hardware.

Els primers intents 3.12 van fallar perquè faltava l'intèrpret (entorn).
La comprovació suplementària inicial 3.14 va passar 1754 tests en 116.63 s;
no substitueix els gates exigits amb 3.12. Logs de tots els intents conservats.

## Límits de comparació amb RTX 2060

Baseline: commit `d2ec8cf1ac7f0340aeeded6b47cc3f646fa546ae`, WSL2/Linux,
Python 3.12.3, Ryzen 5 3600, RAM ~8 GiB, GPU 6 GiB amb 3.39 GiB disponibles,
ngl14 (offload parcial), Validate 16.05 s, runtime allocation 2459.61 MiB.
Aquí: Windows natiu, Python 3.12.14, CPU/RAM diferents, més VRAM disponible,
ngl29, anàlisi reconstruïda i commit Jaull diferent. Encara que el commit
llama.cpp i el driver coincideixen, compilador, OS i distribució CUDA del build
difereixen. No s'atribueixen les diferències de temps/memòria exclusivament a la
GPU. La comparació VRAM del baseline és methodologically_unavailable: no se'n
dedueix un error percentual ni una millora percentual equivalent.

## Incidència i control del runtime

Després dels probes i del primer Benchmark, `llama-bench.exe` ja no era al path
original. No s'ha executat cap ordre d'esborrament. Consultes de Defender no han
aportat evidència d'una causa; no s'afirma quarantena. Restaurar només el fitxer
al mateix path falla amb UnauthorizedAccessException; `direct-benchmark-*.log`
conserva també FileNotFoundError. No s'han canviat ACL ni proteccions.

Extracció normal dels mateixos arxius verificats a
`.venv/rtx4060-runtime-control-b10357` reeixida; SHA-256 del bench idèntic,
probe CUDA i footer de build reeixits (`runtime-control-*.log`). Un segon
Benchmark per la pantalla real amb el mateix pla està documentat separadament;
mai substitueix ni reescriu el primer record.

## Resultat del control i lliurament final

Segon Benchmark ID `bench-12a23300-b538-457f-9091-c5550c132e00`.
Còpia a `records/bench-12a23300-b538-457f-9091-c5550c132e00.json`, raw log a
`logs/bench-12a23300-b538-457f-9091-c5550c132e00.runtime-log`, original a
`%LOCALAPPDATA%/jaull/benchmarks/<ID>.json`.
Mateix pla congelat i mateixa metodologia/repeticions que el primer intent;
només canvia el path del binari verificat idèntic recuperat. El hardware del
BenchmarkRecord és el snapshot congelat del pla, no un nou scan al segon intent.

Resultat real: **success false**, exit 1, failure_reason `timeout`, durada
900.490 s. La UI conserva pp128 2253.33 ± 94.70 t/s i pp512 2458.59 ± 38.45 t/s
com a sortida parcial. No són un benchmark complet reeixit; torna a faltar tg64
i el footer. La causa del bloqueig de generació no s'ha demostrat. El runner ha
terminat el procés en arribar al timeout configurat. No s'ha desinstal·lat ni
esborrat el runtime. El procés experimental ja no està actiu.

Cas final `case-e82d832e-d5ff-4ace-bcc4-8ce5abb844e3`, inclou l'ExperimentRecord
i **tots dos** BenchmarkRecords (inclòs el fallit), així com els tres logs.
Manifest a `bundle-final/case.json`, bundle a `bundle-final/`, índex d'integritat
a `bundle-final/bundle.json`. El primer cas i bundle segueixen intactes.
Creació, validació de cas, exportació i validació de bundle han retornat exit 0.
Estat final **partial** per manca de build en els BenchmarkRecords; integritat
**valid**, tots els fitxers declarats verificats. Això no certifica que el
benchmark hagi completat la generació. Sortides a `final-*.stdout.json`, revisió
manual dels tres raw logs i hashes a `log-review-final.json`.

## Git i preservació

`git diff --stat`: buit, perquè no hi ha canvis tracked. `git diff --cached --stat`:
buit. `git status --short`:

```text
?? validation/rtx4060-b001-r10-20260929/
```

Només evidència i helpers nous sense stage. L'evidència històrica de `validation/`
no s'ha modificat ni esborrat. Entorns, arxius descarregats i models preservats.

NO git add, NO git commit, NO git push, NO git stash, NO git reset,
NO git checkout, NO branch change.
