# Glossari tècnic de `local-ai-check`

Diccionari de consulta ràpida per recordar els conceptes principals del projecte. Les definicions estan orientades al context de `local-ai-check`, no a cobrir tota la teoria d’intel·ligència artificial.

---

## A

### Adapter
Fitxer o conjunt de pesos petits que modifica el comportament d’un model base sense copiar tots els pesos originals.

Un adapter normalment no es pot executar sol: necessita saber quin és el seu `base_model`.

### API
Interfície que permet que un programa parli amb un altre servei.

`local-ai-check` utilitza l’API de Hugging Face per cercar models, consultar repositoris i obtenir metadades.

### Architecture / arquitectura
Estructura interna del model: nombre de capes, tipus d’atenció, `hidden_size`, caps, etc.

Exemples de famílies d’arquitectures:

- Llama;
- Qwen2;
- Mistral;
- Gemma;
- Phi.

Dos repositoris diferents poden utilitzar la mateixa arquitectura.

### Artifact / artefacte
Fitxer executable o distribuïble concret derivat d’un model.

Exemples:

- un `.gguf` Q4_K_M;
- diversos `.safetensors` FP16;
- un model ONNX;
- un model AWQ.

El **model** és el concepte o família. L’**artefacte** és la representació concreta que es descarrega i executa.

### Attention / atenció
Mecanisme que permet al model relacionar tokens entre si durant la inferència.

La configuració d’atenció afecta especialment el KV cache.

### AWQ
Mètode de quantització orientat a reduir la memòria dels pesos mantenint la qualitat.

Un repositori AWQ no és automàticament executable amb qualsevol runtime. Cal comprovar el backend, el hardware i el format concret.

---

## B

### Backend
Implementació tècnica que executa el model.

En aquest projecte, `llama.cpp`, Transformers i vLLM es poden considerar backends o runtimes d’inferència.

### Base model
Model original del qual deriva un altre repositori.

Exemple:

```text
Model original: Qwen/Qwen2.5-1.5B-Instruct
Conversió: algun-autor/Qwen2.5-1.5B-Instruct-GGUF
```

La relació s’hauria de demostrar amb metadades, no únicament pel nom.

### Batch size
Nombre de seqüències processades conjuntament en una operació.

No és exactament el mateix que nombre d’usuaris simultanis.

En `local-ai-check`, el `batch_size` influeix en el càlcul del KV cache.

### Benchmark
Prova controlada que mesura el comportament real.

Mètriques típiques:

- tokens per segon;
- temps fins al primer token;
- memòria màxima;
- latència;
- consum energètic;
- degradació amb concurrència.

Una estimació teòrica no substitueix un benchmark.

### BF16 / bfloat16
Format numèric de 16 bits.

Té una precisió diferent de FP16 i és habitual en models moderns. Ocupa aproximadament 2 bytes per paràmetre abans d’overheads.

### Bits per parameter
Nombre mitjà de bits utilitzats per guardar cada paràmetre.

Exemples teòrics:

- FP32: 32 bits;
- FP16/BF16: 16 bits;
- INT8: 8 bits;
- INT4: 4 bits.

Els formats reals poden afegir escales, blocs, metadades i overhead.

---

## C

### Cache
Còpia temporal d’un resultat per evitar repetir una operació.

El workflow manté cachés en memòria durant una execució per no tornar a inspeccionar o estimar el mateix repositori.

### Candidate / candidat
Model trobat a la cerca que podria arribar a ser recomanació.

Hi ha dos nivells:

- `ModelCandidate`: metadades preliminars i barates;
- `EvaluatedCandidate`: model inspeccionat, estimat i puntuat.

### Candidate discovery
Procés de cercar, deduplicar, filtrar i seleccionar models del Hugging Face Hub.

### CLI
**Command-Line Interface**.

Interfície basada en comandes:

```bash
local-ai-check estimate MODEL
```

És adequada per scripting, automatització i resultats reproduïbles.

### Compatibility / compatibilitat
Classificació de si la memòria estimada cap als recursos detectats.

Estats del projecte:

- `comfortable`;
- `compatible`;
- `tight`;
- `offloading_required`;
- `insufficient`;
- `unknown`.

Compatibilitat de memòria no significa necessàriament rendiment adequat.

### Concurrency / concurrència
Nombre de peticions o usuaris actius al mateix temps.

Els pesos del model normalment es comparteixen, però cada sessió pot necessitar KV cache i buffers addicionals.

### Config.json
Fitxer de configuració típic dels repositoris Transformers.

Pot contenir:

- arquitectura;
- capes;
- `hidden_size`;
- caps d’atenció;
- caps KV;
- context màxim;
- dtype original.

### Confidence / confiança
Valor que indica fins a quin punt l’estimació està fonamentada en dades fiables.

Exemples:

- **alta:** mida real d’un fitxer GGUF i configuració completa;
- **mitjana:** paràmetres coneguts i memòria derivada;
- **baixa:** metadades incompletes o quantització teòrica;
- **desconeguda:** falten dades essencials.

### Context length / longitud de context
Nombre màxim o seleccionat de tokens que el model pot mantenir dins de la conversa o entrada.

Augmentar el context augmenta especialment el KV cache.

No s’ha de confondre la longitud de context amb el nombre total de documents d’un sistema RAG.

### CPU offloading
Tècnica per guardar o executar part del model a la RAM/CPU quan no cap completament a la VRAM.

Pot permetre executar un model més gran, però acostuma a reduir la velocitat.

### CUDA
Plataforma de NVIDIA per executar càlculs a la GPU.

Que el driver informi d’una versió CUDA no garanteix que totes les biblioteques instal·lades siguin compatibles amb aquella versió.

---

## D

### Dense model / model dens
Model on, en general, tots els paràmetres principals de cada capa participen en cada token.

Contrasta amb un model MoE, que activa només una part dels experts.

### Dependency injection / injecció de dependències
Passar els serveis que una classe necessita des de fora en lloc de construir-los internament.

Exemple conceptual:

```python
LocalAiCheckApp(services=fake_services)
```

Això facilita tests sense Internet ni hardware real.

### Device reserve
Memòria que l’estimador decideix deixar lliure per al driver, el sistema, kernels i altres processos.

És una assumpció conservadora, no una dada exacta.

### Diffusers
Biblioteca i estructura de repositoris habitual per models de difusió, especialment generació d’imatges.

El workflow guiat actual està centrat en models de text.

### Download count
Nombre de descàrregues declarat per Hugging Face.

És una senyal de popularitat, no una mesura directa de qualitat.

### Dtype
Tipus de dades numèric utilitzat pels pesos o altres tensors.

Exemples:

- float32;
- float16;
- bfloat16;
- int8;
- int4.

---

## E

### Embedding
Vector numèric que representa significat o similitud.

És essencial en sistemes RAG, però el workflow actual només recomana el model de text; encara no implementa embeddings ni vector store.

### Enrichment / enriquiment
Procés d’afegir informació que no estava directament al repositori analitzat.

En GGUF, pot significar:

- llegir la capçalera;
- resoldre el model base;
- obtenir el seu `config.json`;
- fusionar les dades.

### Estimate / estimació
Càlcul teòric basat en metadades, fórmules i heurístiques.

No és una mesura real d’execució.

### Executability / executabilitat
Grau de confiança que una combinació model + artefacte + runtime + hardware es pot executar realment.

Un model pot cabre teòricament en memòria i, tot i així, no tenir una ruta d’execució compatible confirmada.

---

## F

### Family / família de model
Conjunt de repositoris o artefactes que deriven del mateix model base.

Exemple:

```text
Qwen2.5-1.5B-Instruct
├── Safetensors
├── GGUF Q4_K_M
├── GGUF Q5_K_M
└── altres conversions
```

El resultat final idealment recomana una família i selecciona un artefacte concret.

### Fine-tuning
Entrenament addicional sobre un model base per adaptar-lo a una tasca, estil o domini.

### FP16 / float16
Format de coma flotant de 16 bits.

Ocupa aproximadament 2 bytes per paràmetre abans d’overheads.

### FP32 / float32
Format de coma flotant de 32 bits.

Ocupa aproximadament 4 bytes per paràmetre i normalment necessita molta més memòria que FP16 o una quantització.

---

## G

### Gated model
Model públicament visible, però que necessita acceptar condicions o obtenir permís abans de descarregar-lo.

Pot requerir un `HF_TOKEN` amb accés.

### GGML
Projecte i format històric relacionat amb l’ecosistema de `llama.cpp`.

GGUF és el format modern que el substitueix en gran part.

### GGUF
Format binari pensat per distribuir i executar models, especialment amb `llama.cpp`.

Pot incloure:

- pesos quantitzats;
- tokenizer;
- arquitectura;
- context;
- metadades del model;
- informació de quantització.

Un repositori GGUF pot publicar diverses variants alternatives. Només se’n carrega una a la vegada.

### GQA
**Grouped-Query Attention**.

Diversos caps de consulta comparteixen un nombre menor de caps Key/Value.

Això redueix la memòria del KV cache respecte de MHA.

### GPU
Processador paral·lel utilitzat per accelerar inferència.

En el projecte, les GPU NVIDIA es detecten mitjançant NVML.

---

## H

### Hardware fit
Subscore que representa com encaixa la memòria estimada del candidat al maquinari detectat.

No mesura la qualitat de les respostes.

### Head / cap d’atenció
Unitat paral·lela dins del mecanisme d’atenció.

Camps relacionats:

- `num_attention_heads`;
- `num_key_value_heads`;
- `head_dim`.

### Hidden size
Mida principal de les representacions internes del model.

Sovint permet calcular:

```text
head_dim = hidden_size / num_attention_heads
```

### HF_TOKEN
Token d’autenticació de Hugging Face.

Permet accedir a repositoris privats o gated autoritzats.

No s’ha de mostrar en logs, reports, captures ni commits.

### Hugging Face Hub
Plataforma on es publiquen models, datasets, demos i metadades.

`local-ai-check` utilitza el Hub com a catàleg de candidats.

---

## I

### Inference / inferència
Procés d’utilitzar un model ja entrenat per generar una resposta.

El projecte estima recursos per inferència, no per entrenament.

### Inference configuration
Configuració concreta que s’estima:

- precisió;
- quantització;
- context;
- batch;
- dispositiu;
- dtype del KV cache;
- marges.

### INT4
Representació aproximada de 4 bits per paràmetre.

Redueix molt la memòria, però la qualitat, compatibilitat i rendiment depenen del mètode concret de quantització.

### INT8
Representació aproximada de 8 bits per paràmetre.

Ocupa aproximadament 1 byte per paràmetre abans d’overheads.

### Inspect / inspecció
Operació que analitza un repositori concret i retorna un `ModelAnalysis`.

No executa el model.

---

## K

### KV cache
Memòria que guarda les claus i valors d’atenció dels tokens ja processats.

Creix aproximadament amb:

- nombre de capes;
- caps KV;
- dimensió dels caps;
- context;
- batch o sessions;
- bytes del dtype del KV cache.

Els pesos es carreguen una vegada, però el KV cache pot créixer per sessió.

---

## L

### Latency / latència
Temps que tarda el sistema a respondre.

Pot referir-se al temps total o al temps per token.

### License / llicència
Condicions legals d’ús del model.

Exemples coneguts:

- MIT;
- Apache-2.0;
- BSD;
- llicències personalitzades de fabricants.

`local-ai-check` fa una classificació conservadora, però no ofereix assessorament legal.

### Likes
Nombre de “m’agrada” d’un repositori al Hub.

És una senyal secundària de popularitat.

### llama.cpp
Runtime C/C++ optimitzat per executar models GGUF en CPU, GPU o combinació de totes dues.

És la ruta d’execució principal que proposa el projecte per GGUF.

### LLM
**Large Language Model**.

Model de llenguatge amb molts paràmetres capaç de generar o transformar text.

El terme no implica una mida exacta.

### LoRA
Tècnica d’adaptació eficient que entrena matrius petites en lloc de modificar tots els pesos.

Normalment es distribueix com un adapter que necessita el model base.

---

## M

### MHA
**Multi-Head Attention**.

Cada cap de consulta té els seus caps Key/Value corresponents.

Normalment consumeix més KV cache que GQA o MQA.

### Metadata / metadades
Informació descriptiva o tècnica sobre el model:

- arquitectura;
- llicència;
- idiomes;
- tags;
- model base;
- fitxers;
- context;
- dtype.

Les metadades poden ser exactes, declarades per l’autor, derivades o assumides.

### Model card
README especial d’un model de Hugging Face.

Pot contenir text i una capçalera YAML amb:

- llicència;
- idiomes;
- pipeline;
- tags;
- model base;
- datasets;
- mètriques.

La qualitat de la model card depèn de l’autor.

### Model parameters / paràmetres
Valors numèrics apresos durant l’entrenament.

La quantitat de paràmetres ajuda a estimar la memòria, però no determina per si sola la qualitat.

### MoE
**Mixture of Experts**.

Arquitectura amb múltiples experts, dels quals només se n’activa una part per token.

Cal distingir:

- paràmetres totals;
- paràmetres actius;
- memòria dels pesos;
- cost de computació.

### MQA
**Multi-Query Attention**.

Molts caps de consulta comparteixen un únic conjunt de caps Key/Value.

Redueix encara més el KV cache.

### Multimodal
Model que treballa amb més d’un tipus de dada, per exemple text + imatge.

El workflow actual filtra molts models multimodals perquè està limitat a text-generation.

### Mypy
Comprovador estàtic de tipus de Python.

El projecte l’executa en mode `strict`.

---

## N

### NVML
**NVIDIA Management Library**.

API utilitzada per consultar GPU, VRAM i driver NVIDIA.

El paquet Python utilitzat és `nvidia-ml-py`.

---

## O

### Offloading
Moure part dels pesos o càlculs fora de la GPU, habitualment cap a la RAM/CPU.

Permet executar models que no caben sencers en VRAM, però pot reduir molt el rendiment.

### ONNX
Format interoperable per representar models.

Pot ser executat per diversos runtimes, però el suport i les optimitzacions depenen de l’artefacte concret.

### Orchestrator / orquestrador
Component que decideix l’ordre d’execució dels serveis.

`workflow/orchestrator.py` coordina cerca, filtratge, avaluació i rànquing, però no implementa totes les fórmules internament.

### Overhead
Memòria addicional que no correspon directament als pesos o al KV cache.

Inclou:

- buffers;
- kernels;
- allocator;
- estructures internes;
- memòria temporal.

---

## P

### Parameter count hint
Estimació aproximada de la mida del model extreta del nom, com `7B`.

En el projecte només s’utilitza per prioritzar el shortlist. No és una dada prou fiable per mostrar-la com a recompte real.

### Pipeline tag
Etiqueta de Hugging Face que indica la tasca principal.

Exemples:

- `text-generation`;
- `image-text-to-text`;
- `automatic-speech-recognition`.

### Popularity score
Subscore derivat de descàrregues i likes.

Està comprimit logarítmicament perquè els models molt populars no dominin tot el rànquing.

### Precision / precisió
Format numèric dels pesos o tensors.

Més precisió acostuma a implicar més memòria, però la relació amb la qualitat depèn del model i la quantització.

### Prompt
Text d’entrada enviat al model.

Els tokens del prompt també ocupen KV cache.

### Pydantic
Biblioteca utilitzada per definir i validar els models de domini.

### Pytest
Framework de tests del projecte.

---

## Q

### Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0
Noms comuns de quantitzacions GGUF.

Orientació general:

- número més baix → menys memòria i, sovint, més pèrdua;
- número més alt → més memòria i, sovint, més fidelitat;
- `K` indica esquemes de quantització per blocs de l’ecosistema llama.cpp;
- `S`, `M` o altres sufixos indiquen variants internes.

No s’ha d’ordenar la qualitat únicament pel nom sense validar el model i el runtime.

### Quantization / quantització
Reducció de la precisió dels pesos per disminuir memòria i, segons el backend, accelerar la inferència.

Una quantització és una transformació concreta, no només dir “INT4”.

### Query builder
Component que transforma requisits de l’usuari en diverses consultes al Hugging Face Hub.

---

## R

### RAG
**Retrieval-Augmented Generation**.

Arquitectura que:

1. cerca fragments rellevants en documents;
2. els afegeix al prompt;
3. demana al model que respongui amb aquell context.

El projecte actual pot recomanar un model de text per a un futur RAG, però encara no implementa el cercador documental.

### RAM
Memòria principal del sistema.

En CPU inference o offloading, és crítica.

### Ranking
Ordenació dels candidats segons el score i les regles de desempat.

### Repo ID
Identificador canònic d’un repositori de Hugging Face:

```text
organització/model
```

Exemple:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

### Repository / repositori
Contenidor publicat al Hugging Face Hub.

Pot contenir pesos, configuració, tokenizer, README, quantitzacions o adapters.

Un repositori no sempre equival a una família de model única.

### Round-robin interleaving
Forma de barrejar resultats agafant un element de cada consulta per torns.

Evita que la primera consulta monopolitzi tot el límit de candidats.

### Ruff
Linter i format checker utilitzat per detectar errors i problemes d’estil.

### Runtime
Programa o biblioteca que carrega i executa el model.

Exemples:

- llama.cpp;
- Transformers;
- vLLM.

---

## S

### Safetensors
Format binari segur i eficient per guardar tensors.

Avantatges respecte a formats basats en `pickle`:

- no executa codi en carregar;
- permet lectura eficient;
- facilita metadades i sharding.

### Score
Puntuació composta del recomanador.

En l’estat actual representa **adequació segons les regles**, no qualitat absoluta del model.

### Score breakdown
Desglossament del score en components:

- hardware fit;
- task match;
- language match;
- license;
- metadata quality;
- popularity.

### Service container
Objecte que agrupa dependències i serveis.

Evita que cada pantalla construeixi directament clients reals.

### Shard
Una part d’un model dividit en diversos fitxers.

Exemple:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
...
```

Tots els shards formen conjuntament un artefacte.

### Shortlist
Subconjunt de candidats que passa de la cerca barata a la inspecció profunda.

Es limita perquè consultar configuracions i metadades detallades és més lent.

### Sliding window attention
Atenció limitada a una finestra recent de tokens en determinades capes o arquitectures.

Pot reduir el KV cache efectiu, però el càlcul depèn de la implementació exacta.

### sdist
**Source distribution**.

Paquet amb el codi font preparat per distribuir-se i construir-se.

### Streaming HTTP
Lectura d’una resposta de xarxa per fragments, sense carregar-la sencera a memòria.

S’utilitza per llegir una part limitada de la capçalera GGUF.

---

## T

### Task match
Subscore que intenta decidir si el model sembla adequat per al cas d’ús.

Actualment utilitza sobretot pipeline, tags i paraules clau. No és un benchmark de qualitat.

### Text generation
Tasca on el model rep text i genera continuació o resposta.

És el pipeline principal suportat pel workflow guiat actual.

### Textual
Framework Python utilitzat per construir la TUI.

### Throughput
Quantitat de treball processat per unitat de temps.

En LLM se sol expressar en tokens/s totals o peticions simultànies.

### Token
Unitat en què el tokenizer divideix el text.

Una paraula pot ser un token, diversos tokens o compartir token amb altres fragments.

### Tokenizer
Component que converteix text en IDs de tokens i al revés.

### Tokens per second / tokens/s
Velocitat de generació.

No es pot deduir de manera fiable només amb VRAM i nombre de paràmetres; cal benchmark.

### Training / entrenament
Procés d’ajustar els paràmetres del model amb dades.

Necessita molta més memòria i computació que la inferència. El projecte actual no dimensiona entrenament.

### Transformers
Biblioteca de Hugging Face per carregar i executar models.

També és el nom informal del tipus de repositori basat en `config.json`, tokenizer i pesos com Safetensors.

### TTFT
**Time To First Token**.

Temps des que s’envia la petició fins que apareix el primer token.

És una mètrica important d’experiència d’usuari.

### TUI
**Terminal User Interface**.

Interfície visual dins de la terminal, implementada amb Textual.

### Typer
Biblioteca utilitzada per construir la CLI a partir de funcions Python tipades.

---

## U

### `uv`
Gestor de projectes, entorns i dependències Python.

Comandes habituals:

```bash
uv sync
uv run pytest
uv run local-ai-check
uv build
```

### `uv.lock`
Fitxer que fixa les versions exactes resoltes per `uv`.

---

## V

### vLLM
Runtime orientat a servir LLM amb alt throughput, continuous batching i gestió eficient de memòria.

Està més orientat a servidor que a una simple execució local. La compatibilitat depèn de l’arquitectura, GPU, dtype i quantització.

### VRAM
Memòria de la GPU.

Ha de contenir pesos, KV cache, buffers i altres reserves quan el model s’executa completament a GPU.

---

## W

### Warning
Avís que no invalida necessàriament el resultat, però en redueix la confiança o en limita la interpretació.

Exemples:

- llicència desconeguda;
- configuració incompleta;
- quantització teòrica;
- concurrència no modelada;
- runtime no confirmat.

### Weight / pes
Paràmetre après del model.

La memòria dels pesos és sovint la part principal, però no és l’única.

### Weight precision
Precisió utilitzada per guardar els pesos.

### Wheel
Paquet binari/distribuïble de Python amb extensió `.whl`.

Ha d’incloure el codi i recursos com `styles.tcss`.

### Workflow
Seqüència completa de passos del mode guiat.

En aquest projecte:

```text
hardware → requisits → cerca → filtratge → estimació → rànquing → resultat
```

### WSL2
Entorn Linux virtualitzat dins de Windows.

Pot tenir límits de RAM o diferències de dispositius respecte al Windows host.

---

# Conceptes que es confonen fàcilment

## Model vs repositori vs artefacte

```text
Model/família
└── idea i pesos conceptuals

Repositori
└── contenidor publicat al Hub

Artefacte
└── fitxer concret executable, per exemple Q5_K_M.gguf
```

## Cabre en memòria vs funcionar bé

```text
Cabre
└── estimació de RAM/VRAM

Funcionar
└── runtime compatible i càrrega correcta

Funcionar bé
└── latència i throughput acceptables, verificats amb benchmark
```

## Score vs qualitat

```text
Score actual
└── adequació segons metadades, memòria, llicència i popularitat

Qualitat real
└── rendiment del model en tasques i ús real
```

## Batch size vs concurrència

```text
Batch size
└── seqüències processades conjuntament

Concurrència
└── usuaris o peticions actives alhora
```

## Context del model vs mida de la base documental

```text
Context
└── tokens que entren en una petició

Base documental RAG
└── pot tenir milions de tokens, però només se’n recuperen fragments
```

---

# Fórmules que convé recordar

## Pesos teòrics

```text
memòria_pesos ≈ paràmetres × bits_per_parameter / 8
```

## KV cache simplificat

```text
KV ≈ 2 × capes × caps_KV × head_dim × context × batch × bytes_element
```

## Memòria total del projecte

```text
total ≈ pesos + KV + overhead + reserva + marge
```

Aquestes fórmules són aproximacions. El consum real depèn del runtime i del maquinari.

---

# Fitxers a recordar segons el terme

| Concepte | Fitxer principal |
|---|---|
| Workflow | `workflow/orchestrator.py` |
| Preguntes i requisits | `workflow/models.py`, `workflow/requirements.py` |
| Cerca de models | `discovery/query_builder.py`, `discovery/search_client.py` |
| Shortlist | `discovery/candidate_filter.py` |
| Inspecció d’un model | `huggingface/repository.py` |
| Tipus de repositori | `huggingface/classifiers.py` |
| Config Transformers | `analyzers/transformers.py` |
| GGUF | `analyzers/gguf.py`, `metadata/*` |
| Pesos | `estimator/weights.py` |
| KV cache | `estimator/kv_cache.py` |
| Compatibilitat | `estimator/compatibility.py` |
| Quantització automàtica | `recommendation/configuration.py` |
| Score | `recommendation/scoring.py` |
| Pesos del score | `recommendation/policies.py` |
| Rànquing | `recommendation/ranker.py` |
| Explicacions | `recommendation/explanations.py` |
| Runtime | `runtime/service.py` |
| TUI | `tui/app.py`, `tui/screens/` |
