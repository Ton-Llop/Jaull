# Mini-benchmark agèntic amb tasques de Jaull

La prova 3 (`agentic-pruebas-biosfer.md`) és un estudi de cas: una funcionalitat, un projecte i un model, jutjat a mà.
Aquest benchmark hi afegeix una mesura que es pot repetir. Un agent de programació rep tasques reals de l'historial
de Jaull amb la solució amagada, i uns tests decideixen si les ha resolt. Com que Jaull és codi propi i públic, la
mateixa tasca es pot donar a un model local i a un del núvol, cosa que amb el codi de Biosfer no es pot fer.

Estats: **VALIDATED** (comprovat), **UNVERIFIED** (hipòtesi) i **PENDENT** (no fet).

## Pregunta

En aquestes tasques, quantes en resol el model local recomanat per a la B60 (Qwen3.6-35B-A3B) dins d'OpenCode, amb
quin temps i amb quantes intervencions, i com es compara amb un model del núvol amb el mateix arnès i el mateix
enunciat?

## Tasques

Cada tasca és un commit de correcció de Jaull. L'espai de treball és el codi del commit anterior, amb els fitxers de
`tests/` que la correcció va afegir o canviar. Els tests nous fallen fins que l'agent fa la correcció.

| Tasca | Commit | Què cal resoldre | Codi de la solució original | Dificultat (estimada) |
|---|---|---|---|---|
| T1 | `9aa4abb` | Un fitxer dins d'una carpeta del repo fa fallar `exists()` i amaga totes les rutes del model | 1 fitxer, 12 línies | Fàcil |
| T2 | `ce29a4c` | Els noms Dozen (WSL2) porten el nom real entre parèntesis i les GPU no es distingeixen | 1 fitxer, 46 línies | Mitjana |
| T3 | `06ebdbf` | Un checkpoint AWQ amb l'etiqueta `gguf` es tracta com un GGUF | 1 fitxer de més de 1.000 línies, 12 línies | Fàcil-mitjana |
| T4 | `b29601f` | Una quantització publicada sencera i en parts se suma dues vegades | 2 fitxers, 29 línies | Mitjana |
| T5 | `14e13fa` | La pestanya Memory del Benchmark compara la RAM del procés amb la VRAM estimada | 1 fitxer de la TUI, 50 línies | Mitjana-alta |
| T6 | `1d12f63` | El `batch_size` de l'estimador es passa a llama.cpp com a `--batch-size` | 3 fitxers, 19 línies | Mitjana |
| T7 | `e800e57` | Balanced ha de preferir una quantització còmoda a una d'ajustada | 1 fitxer, 64 línies | Alta |

La selecció per defecte és T1, T2, T4, T5, T6 i T7, amb T3 de reserva. Les dificultats són una estimació feta a
partir de la mida i el tipus de canvi, no una mesura.

**Validació de les tasques, VALIDATED (2026-11-13).** Per a cada commit, el test nou falla amb el codi d'abans i
passa amb el del commit. A més, el circuit sencer (`prepare.sh`, aplicar la solució original, `grade.sh`) dona verd
a totes set, i un espai sense cap canvi dona vermell (vegeu «Validació del circuit»).

## Format

- L'agent rep l'enunciat (`TASK.md`) **amb els tests nous visibles**. Diversos tests comproven textos exactes (per
  exemple «VRAM estimated» o un motiu que contingui «runtime default»), que no es poden endevinar a partir de
  l'enunciat. Amb els tests amagats, com a SWE-bench, T5, T6 i T7 fallarien pel text i no pel codi.
- L'agent no pot tocar `tests/`. `grade.sh` ho comprova i, a més, torna `tests/` a la versió base abans de corregir,
  de manera que canviar un test no pot fer passar una tasca.
- Una tasca és **resolta** si passen els fitxers de test de la tasca **i** la suite sencera.
- No hi ha cap fitxer d'instruccions per a agents a l'espai (`AGENTS.md`, `CLAUDE.md`): no n'hi havia a cap dels
  commits.

## Condicions fixes

- **Arnès:** OpenCode, la mateixa versió per a tots els models. Anotar-la (`opencode --version`).
- **Prompt:** sempre el mateix, en anglès com l'enunciat:

  > Read TASK.md and solve the task. Do not modify any file under tests/. When you are done, run the task's tests
  > and the full test suite, and explain what you changed.

- **Permisos de `bash`** a `opencode.json`: `uv run pytest*`, `git diff*`, `git status*` i `git log*`. Res més.
- **Un intent per tasca i model**, en una sessió nova d'OpenCode.
- **Intervencions:** només es pot escriure «continue» si l'agent s'atura abans d'acabar, o respondre a una pregunta
  seva amb «decide yourself». Cap pista sobre la solució. Es compten totes.
- **Temps màxim:** 45 minuts per tasca. Si s'arriba al límit, es para i es corregeix el que hi hagi.
- **Model local:** `llama-server.exe` b11889 amb Vulkan natiu a Windows, la mateixa configuració que la prova 3:

  ```
  ./llama-server.exe -m 'C:\llama-win\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf' -dev Vulkan1 -ngl 99 -c 131072 -np 1 -lm dio --sleep-idle-seconds 45 --cache-ram 0 --jinja --host 127.0.0.1 --port 8082
  ```

  amb el pont `ssh.exe -R 8081:127.0.0.1:8082` d'`entorno-biosfer.md`.
- **Model del núvol:** **PENDENT** de triar (Claude o GPT). Es pot executar des del portàtil, perquè Jaull és públic.

## Com s'executa a monk-e

Un cop, per preparar-ho:

```bash
git clone https://github.com/Ton-Llop/Jaull.git ~/jaull-bench/jaull-src
```

`prepare.sh` el clona sol si no hi és. Els scripts són a `scripts/agentic-bench/` del mateix repositori.

Per a cada tasca (l'exemple és T1 amb el model local):

```bash
~/jaull-bench/jaull-src/scripts/agentic-bench/prepare.sh T1 qwen36
cd ~/jaull-bench/runs/T1-qwen36 && opencode
```

A OpenCode: triar el model, enganxar el prompt fix i anotar l'hora d'inici. En acabar, `/export` i desar el fitxer a
`~/jaull-bench/results/` amb el nom de la tasca i el model. Després, fora d'OpenCode:

```bash
~/jaull-bench/jaull-src/scripts/agentic-bench/grade.sh T1 qwen36
```

`grade.sh` desa el canvi de l'agent a `~/jaull-bench/results/T1-qwen36.patch`, el registre de la suite a
`T1-qwen36.suite.log`, i afegeix una fila a `~/jaull-bench/results.csv`.

## Què s'anota per tasca

Automàtic, a `results.csv`: si passen els tests de la tasca, si passa la suite, si l'agent va tocar `tests/`, i
fitxers i línies canviades fora de `tests/`.

A mà, de l'`/export` d'OpenCode: durada, tokens de context al final, compactacions, ordres repetides i intervencions.

Revisió del patch, a mà: si la solució és raonable o només fa passar els tests (per exemple, T4 hauria d'invalidar la
memòria cau d'anàlisis, que el test no comprova).

## Validació del circuit

Execució al portàtil, amb la solució original de cada commit aplicada sobre l'espai preparat. **PENDENT** d'apuntar el
resultat.

## Límits

- Set tasques, un intent per model: no hi ha variància mesurada.
- Les tasques són correccions petites d'un sol projecte, en Python. No representen totes les feines d'un
  desenvolupador ni el codi de Biosfer.
- Els tests visibles fan la tasca més fàcil que un informe d'error real.
- Jaull és públic a GitHub i els commits són d'agost a novembre de 2026. Un model del núvol entrenat després podria
  haver vist les solucions. El Qwen3.6-35B-A3B es va publicar a l'abril de 2026 i no les pot haver vist.
- El model local i el del núvol no tenen el mateix context ni la mateixa velocitat; el temps mesura la combinació de
  model, servidor i màquina.
