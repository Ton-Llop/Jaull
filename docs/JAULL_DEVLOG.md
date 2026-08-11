# Jaull — DevLog

> Registre simplificat del desenvolupament de Jaull.  
> Les dates són aproximades, ja que al principi no portava un seguiment diari del projecte.

---

## 29/07/2026 — Idea inicial

Començo a definir la idea del projecte.

L'objectiu inicial és crear una eina capaç d'analitzar el hardware d'un ordinador i determinar quins models d'IA local s'hi podrien executar.

La idea neix arran del TFG i de la necessitat de saber quin hardware seria necessari per executar models locals dins d'una empresa.

Primer objectiu:

- detectar CPU, RAM i GPU;
- buscar models;
- estimar quanta memòria necessiten;
- recomanar quins models poden funcionar.

En aquest moment el projecte encara es diu aproximadament `local-ai-checker`.

---

## 30/07/2026 — Detecció de hardware

Començo a implementar la detecció automàtica del sistema.

Jaull pot obtenir informació com:

- CPU;
- memòria RAM;
- GPU NVIDIA;
- VRAM;
- sistema operatiu;
- informació bàsica d'emmagatzematge.

Creo una estructura `HardwareProfile` per tenir tota aquesta informació normalitzada i poder-la utilitzar després en les estimacions.

---

## 31/07/2026 — Integració amb Hugging Face

Afegeixo la connexió amb Hugging Face Hub.

L'eina ja pot:

- buscar models;
- obtenir informació d'un repositori;
- llegir metadata;
- detectar formats com GGUF o Safetensors;
- consultar configuracions dels models.

També començo a separar la lògica de cerca de la lògica d'anàlisi dels models.

El projecte passa a dir-se **Jaull**.

---

## 01/08/2026 — Estimació de memòria

Començo una de les parts principals del projecte: calcular si un model pot cabre al hardware disponible.

L'estimador té en compte:

- mida dels pesos;
- precisió / quantització;
- RAM;
- VRAM;
- KV cache;
- mida de context;
- overhead del runtime;
- marge de seguretat.

A partir d'això, Jaull pot classificar una configuració aproximadament com:

- comfortable;
- compatible;
- tight;
- offloading required;
- insufficient.

També començo a tenir en compte diversos usuaris simultanis.

---

## 02/08/2026 — Sistema de recomanació

Jaull ja no només calcula si un model cap o no.

Començo a construir un sistema de ranking per recomanar els models més adequats.

Es tenen en compte factors com:

- compatibilitat amb el hardware;
- memòria disponible;
- ús que vol donar l'usuari al model;
- idioma;
- llicència;
- qualitat de la informació disponible;
- popularitat;
- format del model;
- possibilitat real d'executar-lo.

També creo un workflow guiat on l'usuari indica què necessita:

- chat general;
- programació;
- documents;
- prioritat entre qualitat, velocitat i memòria;
- context;
- concurrència.

---

## 03/08/2026 — Refactor d'arquitectura

El projecte comença a créixer bastant i faig una reorganització important.

Separo responsabilitats entre diferents blocs:

```text
hardware
huggingface
analyzers
estimator
recommendation
workflow
runtime
presentation
cli
tui
```

Creo `AdvisorService` com a façana principal del projecte.

La idea és que la CLI i la TUI no hagin de conèixer tota la implementació interna.

Per exemple:

```python
advisor.scan_hardware()
advisor.inspect_model()
advisor.estimate_model()
advisor.recommend()
```

També reforço tests, Ruff i mypy.

---

## 04/08/2026 — TUI i workflow guiat

Començo a treballar més seriosament en la interfície TUI amb Textual.

El flux principal passa a ser aproximadament:

```text
Jaull
  ↓
Hardware
  ↓
Necessitats de l'usuari
  ↓
Cerca de models
  ↓
Anàlisi
  ↓
Recomanacions
```

L'objectiu és que l'usuari no necessiti conèixer Hugging Face, GGUF, quantitzacions o VRAM per poder obtenir una recomanació.

---

## 05/08/2026 — Millores del ranking

Faig diverses proves amb hardware real i detecto problemes en algunes recomanacions.

Per exemple, models AWQ podien aparèixer com a bones opcions encara que després no fossin fàcilment executables amb el runtime disponible.

Afegeixo més informació sobre:

- executabilitat real;
- formats disponibles;
- quantitzacions;
- nombre de paràmetres;
- compatibilitat amb CPU/GPU;
- confiança de les estimacions.

L'objectiu és evitar que Jaull recomani una configuració que teòricament encaixa però que després no es pot executar.

---

## 06/08/2026 — Gestió d'artefactes

Creo `ArtifactService`.

Fins ara Jaull podia recomanar models, però no els descarregava.

Ara el flux pot ser:

```text
Model recommendation
      ↓
Resolve artifact
      ↓
Download GGUF
      ↓
Verify
```

Creo `ModelArtifact`, que guarda informació com:

- repositori;
- revision;
- filename;
- format;
- quantització;
- mida;
- ruta local;
- SHA-256;
- estat de verificació.

Els models es guarden localment a:

```text
~/.local/share/jaull/models/
```

---

## 07/08/2026 — Descàrrega real des de Hugging Face

Provo `ArtifactService` contra Hugging Face real.

Faig servir:

```text
TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
Q4_K_M
```

Jaull resol correctament:

- revision;
- filename;
- quantització;
- mida.

Després descarrega aproximadament 669 MB, calcula SHA-256 i verifica l'artefacte.

Primer flux real complet de:

```text
Hugging Face
→ resolve
→ download
→ local storage
→ verify
```

---

## 08/08/2026 — Backend d'execució

Començo a preparar Jaull perquè no només recomani models, sinó que també els pugui executar.

Creo una abstracció d'execució:

```text
ExecutionBackend
```

i una primera implementació:

```text
HostExecutionBackend
```

que executa processos locals mitjançant `subprocess`.

També creo:

```text
LlamaCppRunner
```

que construeix els comandos necessaris per executar un GGUF amb `llama-cli`.

Primer faig proves amb un executable fals per comprovar el pipeline.

El resultat funciona:

```text
ModelArtifact
→ LlamaCppRunner
→ HostExecutionBackend
→ executable
→ stdout
```

---

## 09/08/2026 — Primera prova amb llama.cpp

Intento compilar `llama.cpp` al portàtil.

La compilació falla perquè WSL es queda sense memòria i l'OOM killer mata `cc1plus`.

Decideixo no continuar forçant aquest ordinador i preparar la prova real en un PC amb GPU NVIDIA.

Aquest problema també ajuda a entendre millor les limitacions que Jaull haurà de detectar i explicar.

---

## 10/08/2026 — llama.cpp + CUDA

Faig el setup de `llama.cpp` al PC amb NVIDIA.

Compilo:

```text
llama-cli
```

amb suport CUDA.

El build acaba correctament:

```text
[100%] Built target llama-cli
```

També documento el setup de:

- CUDA;
- CMake;
- compilació;
- PATH;
- execució de `llama-cli`.

---

## 10/08/2026 — Primera inferència real

Descarrego TinyLlama des del mateix Jaull i l'executo amb `llama-cli`.

Flux:

```text
TinyLlama GGUF
→ llama.cpp
→ CUDA
→ inferència
→ text
```

La primera resposta és bastant divertida perquè TinyLlama decideix que GGUF significa alguna mena de competició de MMA, però tècnicament la inferència funciona correctament.

També detecto que `llama-cli` queda en mode interactiu després de generar.

Afegeixo:

```text
--single-turn
```

al `LlamaCppRunner`.

Ara el procés genera una resposta i finalitza correctament.

---

## 10/08/2026 — Primer end-to-end real de Jaull

Executo:

```text
jaull run
```

utilitzant:

- model real;
- GGUF real;
- ArtifactService;
- SHA verification;
- LlamaCppRunner;
- HostExecutionBackend;
- llama.cpp;
- CUDA.

Flux complet validat:

```text
Hugging Face
      ↓
ArtifactService
      ↓
GGUF
      ↓
AdvisorService
      ↓
LlamaCppRunner
      ↓
ExecutionBackend
      ↓
llama-cli
      ↓
CUDA
      ↓
Generated text
```

Aquest és un dels milestones més importants del projecte: Jaull deixa de limitar-se a predir si un model hauria de funcionar i passa a executar-lo realment.

---

## 10/08/2026 — Unificació de l'execució

Detecto que la CLI estava creant directament:

```text
LlamaCppRunner
HostExecutionBackend
```

tot i que `AdvisorService` ja tenia `run_artifact()`.

Refactoritzo perquè només existeixi una via:

```text
CLI
 ↓
AdvisorService
 ↓
LlamaCppRunner
 ↓
ExecutionBackend
 ↓
llama-cli
```

Això evita duplicar lògica i manté `AdvisorService` com la façana principal de Jaull.

---

## 11/08/2026 — Integració i millora de la TUI

Integro tot el flux de Jaull dins de la TUI, permetent fer:

```text
Hardware
→ Necessitats
→ Recomanacions
→ Seleccionar model
→ Descarregar/verificar GGUF
→ Escriure prompt
→ Executar amb llama.cpp
→ Veure resposta

També milloro la pantalla d'execució amb diversos prompts consecutius i historial visual.

Soluciono diversos errors de Textual relacionats amb DuplicateIds i amb el lifecycle dels workers, que provocaven bloquejos durant els tests.

# Estat actual

Actualment Jaull ja pot:

- detectar hardware;
- consultar Hugging Face;
- analitzar models;
- llegir metadata GGUF;
- estimar RAM i VRAM;
- tenir en compte KV cache i context;
- filtrar models incompatibles;
- rankejar models;
- recomanar configuracions;
- generar reports;
- descarregar GGUF;
- verificar artefactes;
- executar `llama.cpp`;
- utilitzar CUDA;
- generar inferències reals;
- executar models des de CLI;
- executar models des de TUI;
- reutilitzar models descarregats;
- executar diversos prompts des de la TUI.

Arquitectura principal:

```text
                Jaull

Hardware ─┐
HF ───────┤
Estimator ├──→ Recommendation
Runtime ──┘          │
                     ↓
               AdvisorService
                /         \
               ↓           ↓
          ArtifactService  LlamaCppRunner
               ↓                ↓
             GGUF        ExecutionBackend
                                ↓
                            llama.cpp
                                ↓
                              CUDA
```

---

# Següents passos

Ara mateix la prioritat és acabar de polir la TUI/UX.

Després, el següent gran bloc del projecte serà començar a comparar les prediccions de Jaull amb execucions reals.

Per exemple:

```text
JAULL PREDIU              EXECUCIÓ REAL

RAM estimada       ↔      RAM utilitzada
VRAM estimada      ↔      VRAM utilitzada
model compatible   ↔      arrenca o no
GPU offload        ↔      offload real
                   +
                   tokens/s
                   latència
```

Això permetrà passar de:

> "Jaull creu que aquest model funcionarà"

a:

> "Jaull ho ha executat, ho ha mesurat i pot calcular l'error de la seva pròpia estimació."

Aquesta serà una de les parts principals a estudiar dins del TFG.
