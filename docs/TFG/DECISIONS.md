# Jaull: registre de decisions de disseny

**Data de tall:** 11 d'octubre de 2026.

**Historial de Git revisat:** des de `326068f` (2 d'agost de 2026) fins a
`330bf0f` (11 d'octubre), que integra l'agregació de velocitat i les revisions D066-D068.

**Cobertura addicional:** cap. Els canvis que en una primera redacció eren locals
(agregació de velocitat i revisions de l'avaluació) ja són a `330bf0f`.

## 1. Objectiu i criteri de lectura

Aquest document recull les decisions que expliquen **per què Jaull té la forma actual**.
Està pensat com a material de treball per redactar la memòria del TFG: problema,
alternatives, metodologia, arquitectura, resultats i limitacions. No és un manual d'ús
ni una transcripció de totes les converses amb els assistents.

La reconstrucció es basa en l'historial de Git, el codi, els plans, el diari de
desenvolupament i els informes d'experiments del repositori. Les motivacions no sempre
es van deixar escrites en el moment de decidir. Quan només podem explicar el benefici
de l'opció implementada, **no afirmem que es fes una comparació formal d'alternatives**.

Les dates del diari inicial són aproximades. Les anotacions de juliol descriuen
l'origen del projecte, però el primer commit conservat és del 2 d'agost. La data
d'autoria d'un commit tampoc no fixa necessàriament quan es va integrar el canvi.

### Estats utilitzats

- **Vigent:** implementat a l'historial de Git i encara aplicable.
- **Substituïda:** decisió anterior que ajuda a entendre l'evolució, però no descriu
  la política actual.
- **Local, sense commit:** implementació present al directori de treball, no una
  versió publicada ni una garantia de validació a Windows.
- **Pendent:** proposta o límit conegut; no s'ha de presentar com una funcionalitat feta.

### Precaució amb les fonts

Alguns documents descriuen el moment en què es van redactar. Per exemple, el diari
del 10 d'octubre encara deixa Speed i Balanced en shadow, mentre que `491b6fc` ja
els activa. El document del pilot inicial descriu HellaSwag, no tot el flux posterior
d'IFEval. El catàleg publicat continua sent diagnòstic, però això **no vol dir que la
qualitat local no pugui ordenar**.

En aquestes discrepàncies cal contrastar la data, el commit i el codi corresponent.
La política de soroll va canviar durant la redacció (D066-D068); totes aquestes
revisions queden fixades a `330bf0f`.

### Índex de lectura

- [Producte i tecnologia](#3-producte-abast-i-tecnologia),
  [arquitectura](#4-arquitectura-i-contractes) i
  [metadades i integritat](#5-metadades-identitat-i-integritat).
- [Estimació i HFA](#6-estimació-i-collocació-de-memòria).
- [Descoberta i recomanació](#7-descoberta-i-recomanació),
  [qualitat](#8-qualitat-referències-pilot-i-política-activa) i
  [velocitat/Balanced](#9-velocitat-balanced-i-soroll-experimental).
- [Execució i metodologia experimental](#10-execució-observació-i-reproducció-experimental).
- [Interfície, rendiment i proves](#11-interfície-rendiment-i-verificació-del-producte).
- [Resultats experimentals](#12-resultats-que-van-justificar-decisions).
- [Pendents i tancament](#13-decisions-de-tancament-i-treball-pendent) i
  [decisions substituïdes](#14-què-sha-substituït-i-no-sha-de-redactar-com-a-actual).
- [Guia per a la memòria](#15-guia-per-redactar-la-memòria) i
  [fonts i manteniment](#16-índex-de-fonts-i-versions).

## 2. Cronologia resumida

| Període | Fita | Referències de Git |
|---|---|---|
| 2 d'agost | Estructura inicial: inspecció, maquinari i estimació explicable, sense executar models | `326068f` |
| 3-4 d'agost | Nom Jaull, flux guiat, separació de capes i preparació d'artefactes | `e228038`, `f44d9c9`, `b347466`, `25e2017` |
| 10-15 d'agost | Execució real, TUI, observacions, experiments, Transformers i benchmarks persistits | `309b74d`, `900ed05`, `9d411ce`, `01e80fd`, `db34670` |
| 17-22 d'agost | Caché, descoberta concurrent, diversitat, ranking per plans i independència dels runtimes instal·lats | `048e197`, `2679f64`, `452df33`, `b71bd24`, `67b85b0` |
| 24 d'agost - 10 de setembre | Hardware Fit Analyzer, KV per col·locació, límits de pesos no repetits i diagnòstics d'offload | `8fea9e9`, `285ad9d`, `d904e6e`, `607bf80`, `7333a57` |
| 12-14 de setembre | Casos reproduïbles, bundles, mesures de la 2060, política tensor-exacta i contracte d'observació | `a57229c`, `9918cea`, `6cc7c82`, `3a6dc26` |
| 16-30 de setembre | Correcció de capability i tiers, retirada de compatibilitat antiga, integritat i campanya 4060 | `89a30c7`, `bb1a7a9`, `ce44cde`, `a603eea`, `1d12f63`, `0474edf` |
| 1-7 d'octubre | Catàleg publicat, pilot de qualitat Docker, registres locals i avaluació opcional | `af01ed3`, `89db042`, `8a69267`, `da3e41e`, `24b774c`, `982d408` |
| 8-10 d'octubre | IFEval auditat, shadow, Evaluate candidates, Recompare i activació de Quality | treball integrat a `7025bb3` |
| 10 d'octubre | `llama_bench_v2`, velocitat aplicable i activació de Fastest/Balanced | `491b6fc` |
| 11 d'octubre | Esperes de TUI, campanya de soroll, agregació per mediana i comparació per parelles | `5ef3635`, `330bf0f` |

## 3. Producte, abast i tecnologia

### D001. Qualificar un desplegament local, no fer un altre chatbot

**Decisió:** Jaull ajuda a passar de maquinari i necessitats a un model, un artefacte
i un pla d'execució justificats. Poder enviar prompts serveix per validar el pla,
no converteix l'aplicació en el producte conversacional final.

**Motiu i conseqüència:** el problema del TFG és triar i validar infraestructura local
amb evidència. La integració amb els fluxos de Biosfer és un treball posterior i
separat. Això evita que la interfície de conversa substitueixi la pregunta experimental.

**Estat i fonts:** vigent. Primer README a `326068f`, [README actual](../../README.md),
[protocol Biosfer](../experimental-protocol-biosfer.md).

### D002. Començar amb metadades i ajornar l'execució

**Decisió:** la primera versió feia `scan`, `inspect`, `estimate` i `doctor`, sense
descarregar pesos ni executar inferència. Després es van afegir Run, Validate,
Benchmark i Evaluate com a accions explícites.

**Motiu i alternativa:** fer estimacions abans de gastar amplada de banda, disc o GPU
permet descartar opcions amb un cost menor. Es va ampliar la capacitat inicial,
però no es va convertir Search en un benchmark automàtic.

**Estat i fonts:** abast inicial ampliat; la separació continua vigent. `326068f`,
`309b74d`, `ecf4659`, `8a69267`; [limitacions](../limitations.md).

### D003. Python 3.12 i eines especialitzades, no infraestructura pròpia

**Decisió:** Python amb Pydantic per als contractes, Typer per a la CLI, Rich per a
la presentació, Textual per a la TUI, `huggingface-hub` per a metadades, `psutil`
per al sistema i NVML per a NVIDIA. `httpx` es declara explícitament per als Range
reads. `uv`, Hatchling, Ruff, mypy i pytest formen la cadena de desenvolupament.

**Motiu reconstruït:** reutilitzar biblioteques existents concentra l'esforç en la
qualificació i la traçabilitat. No consta un estudi formal que descartés altres
llenguatges o una aplicació web. La llicència del projecte és MIT.

**Conseqüència:** el runtime llama.cpp és extern. Tot i que l'execució és opcional,
el `pyproject.toml` actual inclou Torch, Transformers, Accelerate i bitsandbytes com
a dependències normals; no s'ha d'afirmar que el paquet actual només instal·la eines
lleugeres de metadades.

**Estat i fonts:** vigent, amb dependències ampliades durant el projecte. `326068f`,
`01e80fd`; [configuració del paquet](../../pyproject.toml), [lock](../../uv.lock).

### D004. CLI i TUI com a dues entrades al mateix producte

**Decisió:** mantenir ordres individuals per a usuaris tècnics i un recorregut guiat
Hardware → Your needs → Search → Results per a qui no coneix repositoris,
quantitzacions o flags.

**Motiu i conseqüència:** la CLI facilita automatització i reproducció; la TUI fa
accessible la decisió. No s'han de duplicar estimadors o polítiques a cada frontend.
El nom inicial `local-ai-check` es va substituir per Jaull, sense canviar el problema
que resolia l'eina.

**Navegació revisable, 14 d'octubre:** Search és una pantalla transitòria i
Results la substitueix, no l'apila. Tornar des dels resultats recupera el mateix
formulari de necessitats amb les respostes conservades. Canviar Quality per Speed
o una altra resposta i confirmar inicia una cerca nova amb el maquinari detectat;
no rellança res només per tornar ni modifica la recomanació anterior.

**Estat i fonts:** vigent. `e228038`, `f44d9c9`, `e52ba83`;
[CLI](../cli.md), [interfície](../ui.md).

### D005. Text generation i consciència explícita dels límits

**Decisió:** el flux principal cobreix LLMs de text. Detectar Diffusers, ONNX,
adaptadors o arquitectures especials no implica estimar-los i executar-los tots.
`document_qa` no és una implementació de RAG. La finestra de context
representa el text que rep el model, no la mida de tota una col·lecció
de documents; un sistema RAG haurà de seleccionar fragments abans.

**Motiu i conseqüència:** un suport parcial declarat és preferible a extrapolar
fórmules de transformers densos a qualsevol model. Embeddings, recuperació,
vector store, multimodalitat i servei empresarial no entren automàticament al nucli.

**Estat i fonts:** vigent. [Limitacions](../limitations.md),
[arquitectura](../../ARCHITECTURE.md), [protocol Biosfer](../experimental-protocol-biosfer.md).

### D006. Monòlit modular, no microserveis ni scheduler

**Decisió:** una aplicació local amb mòduls separats. No afegir un servidor remot,
una cua distribuïda, un scheduler o una nova base de dades per implementar les polítiques.
Docker queda acotat al pilot de qualitat sol·licitat per l'usuari.

**Motiu i conseqüència:** menys infraestructura, menys estats i menys cost de manteniment
per al TFG. El processament aïllat d'un runtime no exigeix convertir l'aplicació en
un sistema distribuït.

**Estat i fonts:** vigent. [Arquitectura](../../ARCHITECTURE.md),
[pla vFinal](../recommendation-vfinal-plan.md).

## 4. Arquitectura i contractes

### D007. Separar domini, adaptadors, casos d'ús i presentació

**Decisió:** el domini no depèn de Textual, Rich, Typer o clients externs. Els
adaptadors obtenen dades; els casos d'ús i workflow coordinen; els frontends mostren
i demanen accions. `AdvisorService` actua com a façana per a CLI/TUI.

**Motiu i alternativa:** evita que una pantalla decideixi com estimar o construeixi
clients del Hub. La refactorització inicial va separar responsabilitats abans
d'introduir més execució real.

**Estat i fonts:** vigent. `b347466`, `4682519`, `ca63ca1`;
[arquitectura](../../ARCHITECTURE.md).

### D008. Contractes tipats, validats i majoritàriament immutables

**Decisió:** representar identitats, estimacions, plans i evidència amb models de
domini Pydantic, conservant camps desconeguts i estats explícits. Els registres
experimentals persistits són immutables.

**Motiu i conseqüència:** la validació és una frontera de confiança, no només
comoditat per a la UI. `None` no és zero, ni un comodí que confirmi una coincidència.
Els canvis de schema han de conservar lectura històrica o declarar-ne els límits.

**Estat i fonts:** vigent des de `326068f`;
[domini](../../src/jaull/domain), [evidència](../evidence.md).

### D009. Comprovar les fronteres arquitectòniques amb tests

**Decisió:** proves d'importacions i AST impedeixen dependències prohibides entre
capes i l'ús de dades d'HFA per decidir flags del runtime. Les convencions importants
no queden només en un diagrama.

**Motiu i conseqüència:** els assistents i les refactoritzacions poden reintroduir
acoblaments aparentment útils. La regla automàtica protegeix els límits que realment
comprova; no prova per si sola tota l'arquitectura ni tot el comportament.

**Estat i fonts:** vigent. `898c01f`, `1220fb7`;
[tests d'arquitectura](../../tests/test_architecture_dependencies.py).

### D010. Un model lògic no és un repositori ni un fitxer

**Decisió:** distingir `ModelIdentity`, `ArtifactVariant` i `ExecutionPlan`.
El primer identifica el model; el segon una representació concreta; el tercer combina
artefacte, runtime, backend, preparació i predicció de memòria.

**Motiu i conseqüència:** un model pot tenir diverses quantitzacions i repositoris,
i una mateixa representació pot tenir diferents camins d'execució. Recomanar només
un nom de repositori no especifica què es carregarà.

**Estat i fonts:** vigent. `452df33`, `6eff792`;
[plans](../evidence.md), [construcció de plans](../../src/jaull/execution_plans).

### D011. Un camí compartit de planificació i preparació

**Decisió:** reutilitzar la construcció del pla i la resolució del llançament a Run,
Validate i Benchmark, en lloc de reconstruir flags independentment a cada pantalla.

**Motiu i conseqüència:** evita que una recomanació s'estimi amb una configuració
però es validi amb una altra. Les especialitzacions metodològiques del benchmark
continuen sent explícites: compartir el pla no converteix totes les execucions
en el mateix experiment.

**Estat i fonts:** vigent. `6eff792`, `1c341c4`;
[arquitectura](../../ARCHITECTURE.md), [evidència](../evidence.md).

### D012. La confiança, la viabilitat i la preparació no són el mateix

**Decisió:** separar qualitat de la informació, compatibilitat de memòria,
coherència tècnica del pla i disponibilitat operativa del runtime. Corregir els
tiers que quedaven sempre en best-effort per una heurística d'overhead amb confiança baixa.

**Motiu i conseqüència:** una estimació conservadora no prova que el model sigui dolent
o inexecutable. Tampoc s'ha d'ocultar que una part del càlcul és assumida.
Els títols de recomanació són afirmacions del producte, no només decoració.

**Estat i fonts:** vigent. `bb1a7a9`;
[tiers](../../src/jaull/recommendation/tier.py), [diari](../JAULL_DEVLOG.md).

## 5. Metadades, identitat i integritat

### D013. Descoberta al Hub oficial, sense descarregar pesos

**Decisió:** utilitzar l'API de Hugging Face, consultes complementàries, intercalació
round-robin i deduplicació. Search consulta metadades i capçaleres limitades, no
fitxers complets de pesos. Un token, si existeix, no es registra ni s'exporta.

**Motiu i conseqüència:** el conjunt de candidats pot evolucionar amb el Hub sense
una llista tancada de models. Això no elimina el biaix de cerca, etiquetes i popularitat,
ni converteix el catàleg d'avaluacions publicades en un actualitzador automàtic.

**Estat i fonts:** vigent. `e228038`, `af01ed3`;
[descoberta](../recommendation.md).

### D014. Llegir només la capçalera GGUF necessària

**Decisió:** HTTP Range inicial de 256 KiB, creixement limitat a 8 MiB i lectura
streaming acotada. Si el servidor ignora Range, no s'acaba descarregant tot el cos.
Errors i capçaleres massa grans degraden l'enriquiment amb un motiu.

**Motiu i conseqüència:** obtenir arquitectura i dades del fitxer amb un cost
controlat. Una estimació incompleta és una resposta vàlida; no justifica descarregar
gigabytes en una operació que l'usuari creu que només inspecciona.

**Estat i fonts:** vigent des de la base inicial;
[estimació](../estimation.md), [lector Range](../../src/jaull/metadata/range_reader.py).

### D015. Confirmar el linatge amb dades, no amb sufixos del nom

**Decisió:** per resoldre un model base s'utilitzen declaracions de la model card,
metadades GGUF i fonts amb confiança registrada. El nom d'un repositori pot ser
una pista, però no confirma per si mateix que `X-GGUF` sigui exactament `X`.

**Motiu i conseqüència:** un repack, fine-tune o model semblant no hereta automàticament
la identitat ni els resultats d'un altre. Per atribuir evidència publicada cal comparar
el valor declarat, no només el tipus d'evidència, plegant majúscules però sense
retallar sufixos per confirmar el subjecte.

**Estat i fonts:** vigent, endurit a `982d408`;
[contracte d'evidència publicada](../model-capability-evidence.md), [estimació](../estimation.md).

### D016. La representació que s'executa té prioritat sobre el model base

**Decisió:** les dades específiques del GGUF governen els camps que declaren;
el config del model base completa informació absent. Els conflictes queden
registrats, juntament amb la procedència de cada valor.

**Motiu i conseqüència:** la capacitat de context o la precisió d'un original
no s'han d'atribuir silenciosament a una conversió concreta. Les entrades de l'usuari
també s'han d'interpretar distingint la petició de la capacitat declarada.

**Estat i fonts:** vigent. [Enriquiment i precedència](../estimation.md),
[metadades](../../src/jaull/metadata).

### D017. SHA-256 exacte per reutilitzar mesures d'un artefacte

**Decisió:** qualitat local i velocitat es vinculen als bytes del fitxer, no només
al nom del model, format o etiqueta Q4/Q5/Q6. Es resol una revisió concreta quan
és possible i es recalcula el SHA abans de les campanyes.

**Motiu i conseqüència:** Q4_K_M, Q5_K_M i Q6_K són experiments diferents. Un
resultat excel·lent d'un Q5 no prova el resultat del Q6 seleccionat per Search.
Una revisió flotant i una revisió fixada poden reconèixer el mateix fitxer quan
repositori, nom de fitxer i SHA coincideixen; això no relaxa la resta del protocol.

**Estat i fonts:** vigent. `a603eea`, `982d408`, `491b6fc`;
[artefactes](../evidence.md), [evidència local](../../src/jaull/recommendation/local_evidence.py).

### D018. Descàrrega verificada i sidecars que no puguin certificar altres bytes

**Decisió:** separar resolució, descàrrega i verificació; rebutjar noms i rutes
insegures; comprovar mida i digests. Abans de substituir un fitxer es retira
el sidecar anterior. Un sidecar no pot contradir un SHA conegut de l'artefacte.

**Motiu i conseqüència:** una descàrrega interrompuda no ha de deixar una antiga
verificació que sembli vàlida. El mode de verificació completa recalcula el hash;
confiar en un sidecar no és el mateix nivell de comprovació. Un checksum comprova
integritat, no l'autenticitat o honestedat de qui aporta el fitxer.

**Estat i fonts:** vigent. `25e2017`, `a603eea`, `8423fd7`;
[servei d'artefactes](../../src/jaull/artifacts), [evidència](../evidence.md).

### D019. No ocultar formats i quantitzacions no suportats

**Decisió:** el resolver GGUF local rebutja multipart, inclosa una shard
reanomenada. Transformers té un camí propi. AWQ/GPTQ conserven la seva semàntica;
int4/int8 no es tradueixen a un fals `torch_dtype=int8`, sinó a una configuració
de quantització compatible quan el runtime la suporta.

**Motiu i conseqüència:** estimar pesos petits i executar float16 invalidaria
l'experiment. Si falten els components necessaris, s'ha de bloquejar l'acció
amb un motiu, no substituir silenciosament la representació.

**Estat i fonts:** vigent. `df21246`, `78ecf2e`;
[execució](../evidence.md), [limitacions](../limitations.md).

## 6. Estimació i col·locació de memòria

### D020. Descompondre la memòria i declarar la procedència

**Decisió:** separar pesos, KV cache, overhead, reserva del dispositiu i marge de
seguretat. Les fonts `EXACT`, `METADATA`, `DERIVED`, `ASSUMED` i `UNKNOWN` no són
intercanviables; la confiança global reflecteix les limitacions dels components.

**Motiu i conseqüència:** el total pot explicar-se i revisar-se sense inventar
precisió. La mida de la quantització GGUF seleccionada és preferible a la suma
de tots els fitxers del repositori; un dtype teòric no incorpora necessàriament
scales, alineació i metadades d'un format real.

**Estat i fonts:** vigent des de `326068f`; [estimació](../estimation.md).

### D021. Fórmula de KV amb context, seqüències i concurrència

**Decisió:** per a arquitectures suportades, modelar:

```text
KV = 2 × capes × caps KV × head_dim × context
     × seqüències del batch × usuaris concurrents × bytes per element
```

**Motiu i conseqüència:** GQA utilitza caps KV, no necessàriament tots els caps
d'atenció. Cada sessió es pressuposa amb el seu context retingut. És un model de
capacitat, no una mesura de throughput ni una prova que el runtime executi aquell
nombre d'usuaris simultanis.

**Estat i fonts:** vigent. [Estimació](../estimation.md),
[estimador KV](../../src/jaull/estimator/kv_cache.py).

### D022. Una sliding window declarada no prova que estigui activada

**Decisió:** no limitar el context perquè simplement existeixi `sliding_window`.
Cal interpretar `use_sliding_window` i les semàntiques conegudes de l'arquitectura;
una configuració explícitament desactivada no redueix el càlcul.

**Motiu i conseqüència:** aplicar sempre el mínim podia subestimar la KV de models
que declaren una mida però no utilitzen atenció de finestra. Amb semàntica desconeguda
no s'inventa un estalvi.

**Estat i fonts:** substitueix la simplificació inicial; vigent. `285ad9d`;
[implementació](../../src/jaull/estimator/kv_cache.py),
[regressions](../../tests/test_kv_cache_estimator.py).

### D023. Desconegut abans que una xifra fabricada

**Decisió:** MoE, MLA, composites multimodals i configuracions no suportades poden
quedar amb memòria o KV desconegudes. Un pla sense runtime determinable s'explica
com a no confirmat, no com a incompatibilitat demostrada.

**Motiu i conseqüència:** el nombre total de paràmetres no resol per si sol el
comportament d'arquitectures especials. El pool pot perdre un model viable; és una
limitació explícita de cobertura, no evidència que no càpiga físicament.

**Estat i fonts:** vigent, amb una excepció posterior: la KV d'un MoE ja no és desconeguda (D102); no hi ha suport general MoE. [Limitacions](../limitations.md),
[auditoria del pool](../recommendation-vfinal-plan.md).

### D024. Overhead i marges són polítiques, no mesures

**Decisió:** conservar una heurística centralitzada d'overhead i components
separats per a reserva i seguretat. No ajustar les constants per fer encaixar
un únic model en una única 2060.

**Motiu i conseqüència:** protegir una execució i predir una assignació física
són objectius diferents. La regla general d'overhead i el headroom específic
del llançament no s'han de confondre ni reduir només perquè un buffer observat
sigui menor.

**Estat i fonts:** vigent. [Polítiques de l'estimador](../../src/jaull/estimator/policies.py),
[B001-R5](../qwen2.5-tests/b001-r5-launch-policy-recheck.md),
[contracte d'observació](../qwen2.5-tests/b001-r7-observation-contract.md).

### D025. Pressupost de capacitat no és assignació física

**Decisió:** distingir `gpu_required_bytes`, que inclou reserva i marge,
de `gpu_physical_bytes`, que els exclou. Només el segon pot comparar-se amb
una observació d'assignacions, si passen les portes metodològiques.

**Motiu i conseqüència:** cap procés assigna el marge de seguretat del planificador.
Incloure'l al costat predit inflaria l'error i mesclaria política de prudència
amb precisió del model de memòria.

**Estat i fonts:** vigent. `d1d878c` i enduriments posteriors;
[comparació](../evidence.md), [limitacions](../limitations.md).

### D026. HFA amb pools separats, no una suma RAM + VRAM

**Decisió:** afegir Hardware Fit Analyzer amb `GPU_RESIDENT`, `GPU_OFFLOAD`,
`CPU_RAM` i `TOO_LARGE`. Comprovar per separat els requisits del dispositiu i
del host, amb desglossament i mètode de col·locació.

**Motiu i alternativa:** el veredicte antic basat en memòria combinada no
explicava on hi havia el coll d'ampolla. L'offload és una opció real amb un
cost, no una extensió transparent de la VRAM. Alguns estats antics es mantenen
per compatibilitat, però no substitueixen l'anàlisi de col·locació.

**Estat i fonts:** vigent; substitueix la simplificació inicial. `8fea9e9`;
[HFA i llmfit](../hardware-fit-vs-llmfit.md), [estimació](../estimation.md).

### D027. La KV acompanya els blocs en l'offload genèric

**Decisió:** repartir KV, overhead i marge segons la col·locació, amb arrodoniment
conservador a GPU i resta exacta a RAM. No carregar tota la KV a VRAM mentre
el host no en rep cap.

**Motiu i conseqüència:** conservar bytes entre pools i reflectir millor
l'execució observada. És una hipòtesi genèrica contrastada en un cas, no una
garantia per a qualsevol runtime amb paging, KV quantitzada o col·locació especial.

**Estat i fonts:** vigent. `607bf80`;
[distribució KV](../estimation.md), [validació](../../validation/README.md).

### D028. Separar pesos repetits i no repetits, amb límits conservadors

**Decisió:** estimar blocs transformer i pesos no repetits a partir del config
quan hi ha informació suficient. Per a la part no repetida d'ubicació desconeguda,
HFA comprova els extrems més exigents de GPU i RAM.

**Motiu i conseqüència:** embeddings i output head no són un bloc uniforme més.
Els dos màxims poden correspondre a col·locacions hipotètiques diferents:
no se sumen com si fossin una sola assignació. Aquesta descomposició és una
projecció de paràmetres a bytes, no una lectura de tensors GGUF.

**Estat i fonts:** vigent. `8b40376`, `7333a57`;
[límits de col·locació](../estimation.md).

### D029. Explicar la frontera d'offload, no només donar un nombre

**Decisió:** conservar el recompte seleccionat, el primer recompte superior
rebutjat quan s'ha provat, el sostre de cerca i els bytes de dèficit o headroom.

**Motiu i conseqüència:** permet respondre «per què 18 i no 19?» amb els components
que decideixen. Si no s'ha avaluat un candidat superior, no s'ha d'inventar un
rebuig. Aquests diagnòstics són de planificació, no mesures de CUDA.

**Estat i fonts:** vigent. `91f6305`;
[diagnòstics HFA](../estimation.md).

### D030. HFA no decideix `--n-gpu-layers`

**Decisió:** els blocs d'HFA són unitats independents del runtime. La política
de llançament de llama.cpp decideix les seves pròpies unitats, que inclouen
l'output layer i poden sumar `n_layer + 1`.

**Motiu i conseqüència:** convertir directament blocs d'HFA en flags del runtime
pot col·locar pesos diferents dels previstos. Les proves arquitectòniques
prohibeixen aquesta dependència fins i tot si sembla una drecera pràctica.

**Estat i fonts:** vigent. `d2ec8cf`, `9918cea`;
[arquitectura](../../ARCHITECTURE.md), [B001-R6](../qwen2.5-tests/b001-r6-tensor-policy.md).

### D031. Refinar el llançament amb descriptors de tensors del fitxer local

**Decisió:** per al build i layout verificats de llama.cpp, llegir els descriptors
GGUF locals i sumar els pesos emmagatzemats dels darrers blocs i de l'output.
Validar tipus, límits, alineació, offsets i solapaments sense llegir els payloads.

**Motiu i alternativa:** les aproximacions uniformes de bytes per capa deixaven
GPU infrautilitzada. La regla verificada per dense `qwen2`, build `689e227db`
i una CUDA discreta no es generalitza automàticament a MoE, altres builds,
output lligat, multi-GPU o altres backends.

**Conseqüència:** els embeddings poden continuar al host fins i tot amb offload
complet. Si no es pot demostrar el layout, s'utilitza el fallback amb un motiu.
Els overrides explícits de l'usuari tenen precedència. El refinament modifica
el llançament preparat, no reescriu HFA, l'estimació o un registre antic.

**Estat i fonts:** vigent. `9918cea`;
[política tensor-exacta](../qwen2.5-tests/b001-r6-tensor-policy.md), [estimació](../estimation.md).

### D032. Capacitat física per filtrar; disponibilitat observada per executar ara

**Decisió:** la preselecció pot usar capacitat per no perdre classes de models;
l'estimació final i HFA consumeixen memòria disponible observada. No substituir
disponibilitat desconeguda per capacitat total.

**Motiu i conseqüència:** «aquesta màquina podria executar-lo» i «pot arrencar
ara» són preguntes diferents. Altres processos poden alterar el veredicte;
la discrepància entre la memòria vista per NVML i pel runtime també importa
a prop de la frontera.

**Estat i fonts:** vigent, amb límits de planificació encara oberts. `d904e6e`;
[limitacions de maquinari](../limitations.md).

### D033. Memòria unificada no és una segona reserva independent

**Decisió:** no sumar una GPU de memòria compartida a la RAM com si fossin dos
pools. Descartar renderers software com llvmpipe de la selecció d'acceleració útil.

**Motiu i conseqüència:** detectar Vulkan no prova ni VRAM dedicada ni capacitat
de còmput adequada. La identitat, el tipus de memòria i les dades disponibles
governen què pot afirmar Jaull.

**Estat i fonts:** vigent. `96517a2`, `24d6c26`, correccions posteriors;
[limitacions](../limitations.md).

### D034. Batch de seqüències no és batch de tokens del runtime

**Decisió:** deixar de traduir el `batch_size` de l'estimador a `--batch-size`
de llama.cpp. El primer compta contexts retinguts; el segon governa tokens
processats en una passada. Els flags antics explícits es conserven com a procedència.

**Motiu i conseqüència:** es corregia una falsa equivalència entre capacitat
i configuració de kernels. No s'han de reetiquetar experiments antics com si
haguessin aplicat la política corregida.

**Estat i fonts:** vigent. `1d12f63`;
[estimació](../estimation.md), [execució](../evidence.md).

## 7. Descoberta i recomanació

### D035. Auditar el pool abans de millorar el ranking

**Decisió:** aplicar filtres preliminars abans de consumir pressupostos i
examinar quins candidats arriben a inspecció. Treure adaptadors no executables
que ocupaven places; explicar exclusions de models no estimables.

**Motiu i conseqüència:** un ranker perfecte no recupera un bon model que
ja s'ha perdut abans. Els límits actuals són 20 resultats per consulta,
40 repositoris elegibles, 12 inspeccions profundes, concurrència 4, pressupost
6 per descoberta de variants i fins a 5 recomanacions. No són recall exhaustiu.

**Estat i fonts:** vigent. `09bd390`, `af01ed3`, `7025bb3`;
[polítiques](../../src/jaull/application/recommendation/policies.py),
[auditoria vFinal](../recommendation-vfinal-plan.md).

### D036. Preselecció conscient de maquinari, però encara barata

**Decisió:** ordenar la cua d'inspecció amb pistes preliminars de task, mida
i col·locació possible, deixant entrar opcions GPU-resident, offload i CPU.
Una mida extreta del nom només serveix com a pista de cua.

**Motiu i conseqüència:** inspeccionar totes les shards de tots els repositoris
és massa costós. Aquesta heurística no es publica com a mesura ni substitueix
`MemoryEstimate` i HFA després d'inspeccionar.

**Estat i fonts:** vigent. `d904e6e`;
[descoberta](../recommendation.md).

### D037. Retirar l'score global com a autoritat del flux guiat

**Decisió:** passar d'una suma ponderada de memòria, task, idioma, llicència,
metadades i popularitat a `PlanAssessment` i claus lexicogràfiques per prioritat.
L'score antic que encara apareix en contractes legacy no decideix el flux amb maquinari.

**Motiu i alternativa:** un nombre global barrejava dimensions i amagava
restriccions darrere de pesos arbitraris. També podia mostrar una explicació
numèrica que no correspondia a l'ordre real. La TUI va deixar de mostrar-lo
com a indicador de superioritat.

**Estat i fonts:** vigent; score inicial substituït en el flux principal.
`452df33`, `b71bd24`, `a3cc9b5`, `ce44cde`;
[motor](../../src/jaull/recommendation/engine_v2.py), [recomanació](../recommendation.md).

### D038. Capability és un prior, no intel·ligència mesurada

**Decisió:** no duplicar popularitat, metadata quality, llicència o idioma
dins de capability. El senyal de mida i artefacte és un fallback de capacitat
estimada, no una nota científica de qualitat.

**Motiu i conseqüència:** el model no ha de guanyar dues vegades pel mateix
senyal ni perdre capacitat aparent perquè li falti una etiqueta d'idioma.
Més paràmetres, una família nova o més descàrregues no demostren millor resposta.

**Estat i fonts:** vigent. `89a30c7`;
[capability](../../src/jaull/recommendation/capability.py), [diari](../JAULL_DEVLOG.md).

### D039. Restriccions dures separades de preferències i incertesa

**Decisió:** descartar incompatibilitats demostrades d'artefacte/runtime,
memòria insuficient i requisits incompatibles. No considerar tota metadada
absent com a incompatibilitat; conservar l'estat desconegut i explicar-lo.

**Motiu i conseqüència:** és diferent «la llicència prohibeix l'ús demanat»,
«no hem confirmat la llicència» i «falta runtime a la màquina». Llicències
personalitzades no es declaren comercials sense dades suficients. La classificació
és conservadora i informativa, no assessorament legal.

**Estat i fonts:** vigent. [Restriccions i llicències](../recommendation.md),
[motor](../../src/jaull/recommendation/engine_v2.py).

### D040. No empitjorar recomanacions perquè encara falta instal·lar el runtime

**Decisió:** `executability` és coherència tècnica i afecta la qualificació;
`runtime_readiness` és operativa i només governa les accions.

**Motiu i alternativa:** abans, sense llama.cpp podien desaparèixer tots els
GGUF, o una GPU amb un build CPU-only podia obtenir pitjors recomanacions
que un portàtil CPU. La instal·lació és un pas resoluble, no qualitat del model.
La descàrrega tampoc no necessita que el runtime estigui preparat.

**Estat i fonts:** vigent. `67b85b0`;
[regressió d'independència](../../tests/test_recommendation_runtime_agnostic.py).

### D041. Cinc opcions útils, no cinc quantitzacions del mateix model

**Decisió:** col·lapsar plans del mateix model lògic, conservar alternatives
i afavorir diversitat entre candidats de signatura comparable. Família,
escala i perfil d'execució poden aportar varietat sense inventar nous guanyadors.

**Motiu i conseqüència:** el Top 5 ha de ser una selecció navegable. La diversitat
pot explicar una entrada o sortida del Top 5; no s'ha d'atribuir tot canvi a
una mesura de qualitat o velocitat. El resultat pot tenir menys de cinc opcions.

**Estat i fonts:** vigent. `2679f64`, `452df33`;
[diversitat](../../src/jaull/recommendation/diversity.py), [recomanació](../recommendation.md).

### D042. Task, mode de treball i desplegament són dimensions diferents

**Decisió:** task indica què es fa; `interactive`/`batch` indica com arriben
les peticions. Un servei persistent és una altra dimensió. `WorkloadProfile`
pot registrar context, concurrència, llargades i objectius TPS/TTFT.

**Motiu i conseqüència:** coding pot ser interactiu o batch; documents pot
ser un chatbot o un processament offline. Un SLO demanat no és una observació,
i el perfil no implica que hi hagi un servidor o un generador de càrrega.
Valors serialitzats històrics continuen llegint-se encara que el wizard ja
no els ofereixi.

**Estat i fonts:** vigent. `a285acf`, `ce44cde`;
[requisits](../recommendation.md), [arquitectura](../../ARCHITECTURE.md).

### D043. Determinisme i explicacions basades en regles

**Decisió:** desempats estables per identitat i configuració, i explicacions
generades per les regles que han decidit l'ordre. No utilitzar un altre LLM
per redactar justificacions o triar models.

**Motiu i conseqüència:** el mateix input congelat i la mateixa evidència
han de donar la mateixa decisió, independentment de l'ordre del store.
Això no promet que una cerca nova al Hub viu torni el mateix pool un mes després.

**Estat i fonts:** vigent. [Motor i explicacions](../recommendation.md),
[fidelitat de l'informe](../../tests/test_report_fidelity_regression.py).

## 8. Qualitat: referències, pilot i política activa

### D044. Evidència publicada i mesura local són seccions diferents

**Decisió:** mantenir un contracte de capability publicada amb subjecte,
benchmark, mètrica, versió, variant, revisió, precisió, protocol i font.
Mostrar-lo separat de la qualitat mesurada en l'artefacte exacte.

**Motiu i conseqüència:** un resultat del publisher pot ser rellevant, però
no prova que el GGUF actual reprodueixi aquell valor. Una revisió o precisió
desconeguda es conserva com a desconeguda; no es completa amb el HEAD actual
del Hub. Validar la transcripció no valida la veracitat de la publicació.

**Estat i fonts:** vigent. `89db042`, `24b774c`, `982d408`;
[contracte publicat](../model-capability-evidence.md).

### D045. Catàleg versionat i revisat, no scraping automàtic de leaderboards

**Decisió:** guardar fonts transcrites i validades amb versió i hash del
catàleg. Rebutjar JSON ambigu, identificadors duplicats, xifres no finites
i URLs amb credencials. No afegir un model a discovery només perquè tingui
una entrada al catàleg.

**Motiu i conseqüència:** evita inventar protocols llegint només una puntuació
de leaderboard. La descoberta de models és dinàmica, però **el catàleg publicat
actual s'actualitza manualment**. L'objectiu que Jaull segueixi models nous
no significa que ja hi hagi un actualitzador mensual d'avaluacions.

**Estat i fonts:** vigent; automatització pendent. `89db042`;
[catàleg i cobertura](../model-capability-evidence.md),
[catàleg JSON](../../src/jaull/recommendation/capability_catalog.json).

### D046. No aplicar cegament «local > publicat > prior»

**Decisió:** `QualityAssessment` projecta evidència pertinent per task i
idioma, identitat, cobertura, protocol, comparabilitat i limitacions. La
procedència local no garanteix més valor que una publicació adequada.

**Motiu i alternativa:** HellaSwag local amb 100 mostres no ha d'ordenar una
petició coding per davant d'una evidència coding pertinent només perquè és
local. Tampoc es tracta l'absència de mesures com una qualitat de zero.
Completitud, confiança i percentatge d'encerts són coses diferents.

**Estat i fonts:** vigent. `7025bb3`;
[pla vFinal](../recommendation-vfinal-plan.md),
[assessment de qualitat](../../src/jaull/recommendation/quality.py).

### D047. Reutilitzar lm-eval amb Docker i un contracte auditat

**Decisió:** el pilot executa un harness fixat contra un `llama-server`
controlat, amb imatge, codi, dataset, binari, suite i artefacte identificats.
No escriure un avaluador general nou ni confiar en una etiqueta `latest`.

**Motiu i conseqüència:** reutilitzar un harness no elimina la necessitat
d'auditar l'adaptador. Es van trobar versions que ignoraven kwargs, límits
que no arribaven a HTTP i operacions no implementades. Una imatge antiga
es rebutja amb un motiu i es reconstrueix amb una etiqueta nova; no es
substitueix silenciosament una imatge que va produir evidència antiga.

**Límit:** el pilot host/contenidor requereix Linux o WSL i un checkout de
confiança; executar aquell checkout és executar codi local. Que la TUI
funcioni a Windows no prova que el pilot funcioni a Windows natiu.

**Estat i fonts:** vigent. `8a69267`, `fc663bf`, `6bf4d56`;
[pilot](../quality-evaluation-pilot.md),
[productor](../../pilot/quality_eval), [runner](../../src/jaull/runtime/quality_eval_runner.py).

### D048. Classificar smoke, diagnòstic i avaluació completa

**Decisió:** separar `plumbing` per comprovar infraestructura, `limited`
per comparació exploratòria i `full` per una execució que cobreix el contracte
complet. La classificació es verifica contra les mostres, no es creu per etiqueta.

**Motiu i conseqüència:** tres prompts no donen una nota general de qualitat;
100 exemples de HellaSwag tampoc no es converteixen en el benchmark complet.
Es poden comparar diagnòstics del mateix protocol sense autoritzar que ordenin
la recomanació. No s'uneixen suites diferents per augmentar artificialment N.

**Estat i fonts:** vigent. `fc663bf`, `da3e41e`, `7025bb3`;
[pilot](../quality-evaluation-pilot.md), [registres](../../src/jaull/evaluation/quality_records.py).

### D049. HellaSwag primer per infraestructura; IFEval per instruccions

**Decisió:** començar amb scoring de continuacions de HellaSwag, i després
afegir IFEval per avaluar seguiment d'instruccions de chat. IFEval utilitza
541 prompts i 834 instruccions del split fixat; la mètrica principal és
l'encert estricte per prompt.

**Motiu i conseqüència:** triar una continuació plausible i obeir una petició
de l'usuari són capacitats diferents. La suite es declara abans de mirar
els resultats, en lloc d'escollir després el benchmark on un model guanya.
L'IFEval actual és pertinent per a chat general en anglès, no per demostrar
qualitat de codi, català, traducció o RAG.

**Estat i fonts:** vigent. `8a69267`, `7025bb3`;
[pla i auditoria IFEval](../recommendation-vfinal-plan.md),
[productor IFEval](../../pilot/quality_eval/ifeval.py).

### D050. Contracte IFEval fix: template propi i raonament desactivat

**Decisió:** utilitzar la plantilla de chat del GGUF, registrar-ne la identitat,
desactivar raonament, fixar context 4096, màxim de generació 1280 i condicions
deterministes, inclosa la llavor del detector d'idioma.

**Motiu i alternativa:** un prompt raw comú no equival al mode instruct del
model; una plantilla pròpia arbitrària tampoc. Si el template oficial afegeix
un system prompt, això forma part del protocol i de les seves limitacions.
La decisió no aïlla l'efecte del template entre models.

**Conseqüència:** és «Jaull IFEval» amb aquest contracte, no una xifra
intercanviable amb qualsevol resultat publicat d'IFEval. TinyLlama, amb
context 2048, no es va adaptar abaixant només el seu context; es va excloure
d'aquella comparació i es va utilitzar un candidat que admetia el contracte.

**Estat i fonts:** vigent. `7025bb3`;
[suite](../../pilot/quality_eval/suite_ifeval.yaml),
[auditoria del protocol](../recommendation-vfinal-plan.md), [diari](../JAULL_DEVLOG.md).

### D051. Validar el que arriba al servidor i les respostes reals

**Decisió:** verificar peticions i respostes HTTP, rebutjar parsejos invàlids
i aturar una execució si l'entrada i el límit de sortida no caben al context.
Les respostes que arriben al límit de generació es compten i continuen al
denominador. No convertir un error de parseig en una resposta buida vàlida.

**Motiu i conseqüència:** un flag acceptat pel CLI no prova que s'hagi aplicat.
Ocultar truncaments o errors distorsionaria el resultat. Els logs poden
tolerar bytes UTF-8 tallats pel runtime sense relaxar el contracte dels
resultats estructurats.

**Estat i fonts:** vigent. `7025bb3`;
[productor IFEval](../../pilot/quality_eval/ifeval.py),
[tests del productor](../../tests/test_quality_eval_ifeval.py).

### D052. Desactivar caches que poden contaminar una comparació

**Decisió:** el pilot inicial desactiva cache de respostes i prompt caching
segons el contracte auditat. El key genèric de cache de lm-eval no identificava
l'artefacte, i es va observar que un prefix cached canviava logprobs.

**Motiu i conseqüència:** l'optimització no ha de reutilitzar la resposta
d'un altre model ni canviar silenciosament el protocol. Els resultats de
HellaSwag també van mostrar canvis de logprobs amb offload CPU/parcial/complet;
la col·locació queda dins de la identitat de qualitat.

**Estat i fonts:** vigent dins dels protocols auditats, no una prohibició de
qualsevol cache futura. [Pilot i prova de placement](../quality-evaluation-pilot.md),
[diari històric del pilot](../history/quality-pilot-journal-2026-10.md).

### D053. Registre generatiu propi i importació que rebutja execucions parcials

**Decisió:** ampliar el contracte de qualitat a schema 2 per als resultats
generatius, amb respostes, samples, mètriques, outcome i digests validats.
No importar bundles interromputs o sense comprovacions finals com si fossin full.

**Motiu i conseqüència:** scoring de continuacions i generació necessiten
dades i comparadors diferents. L'execució que es va interrompre després de
58/541 respostes no es podia reconstruir com un resultat complet.
Desar fitxers parcials conserva diagnòstic, no autoritza el ranking.

**Estat i fonts:** vigent. `fc663bf`, `7025bb3`;
[registres](../../src/jaull/evaluation/quality_records.py),
[comparador](../../src/jaull/evaluation/quality_comparison.py).

### D054. Repeticions idèntiques compten com un resultat de qualitat

**Decisió:** per una mateixa identitat completa, les mateixes samples,
results i outcome compten com un resultat encara que hi hagi diverses execucions.
Només s'ignora la data irrellevant de lm-eval en aquesta equivalència.
El representant és determinista, amb el digest menor.

**Motiu i alternativa:** inicialment qualsevol duplicat feia abstenir el
lookup. Dos runs del Qwen amb 541 respostes idèntiques van justificar
deduplicar sense perdre traçabilitat. Si els resultats discrepen, no se'n
tria el millor, el més recent ni una mitjana improvisada; el protocol
actual s'absté. Un rerun no arbitra automàticament el conflicte.

**Conseqüència:** store, assessment i selecció de candidats utilitzen la
mateixa decisió. És la política del perfil determinista actual, no una
prohibició universal d'avaluacions amb múltiples seeds.

**Estat i fonts:** vigent. `7025bb3`;
[qualitat](../../src/jaull/recommendation/quality.py),
[store](../../src/jaull/evaluation/quality_storage.py),
[tests d'aplicabilitat](../../tests/test_ifeval_applicability.py).

### D055. Aplicabilitat exigeix pins auditats, no només el nom «IFEval full»

**Decisió:** exigir identitat exacta, task/idioma pertinent, split complet
i pins auditats de suite, dataset, evaluator i runtime. Distingir
`controlled_artifact` de `current_configuration`.

**Motiu i conseqüència:** un registre ben format però d'un protocol arbitrari
no ha de passar només perquè diu IFEval i full. La qualitat controlada d'un
artefacte no prova qualitat mesurada a qualsevol context i placement
de la configuració demanada per l'usuari.

**Estat i fonts:** vigent. `7025bb3`;
[pla vFinal](../recommendation-vfinal-plan.md),
[aplicabilitat](../../src/jaull/recommendation/quality.py).

### D056. IFEval no és intrínsecament GGUF; el productor actual sí que ho és

**Decisió:** limitar la primera implementació a un GGUF únic amb el
`llama-server` auditat. No estendre la mateixa evidència a Transformers
o vLLM perquè el nom del model coincideixi.

**Motiu i conseqüència:** comparar representacions i runtimes és possible
com a disseny experimental, però necessita controlar prompt/template,
tokenització, generació, precisió, backend i mètrica. Encara no existeix
un protocol de qualitat multiruntime general. vLLM és una possible recomanació
de runtime, no un runner de qualitat ja implementat.

**Estat i fonts:** vigent; extensió pendent. [Limitacions](../limitations.md),
[pilot](../quality-evaluation-pilot.md), [pla vFinal](../recommendation-vfinal-plan.md).

### D057. Shadow abans d'activar una política nova

**Decisió:** calcular l'ordre proposat sobre el mateix pool i comparar-lo
amb l'ordre actiu sense substituir-lo. Registrar moviments, entrada/sortida
del Top 5, evidència, fallback i restriccions. Un error del shadow no fa
perdre la recomanació antiga.

**Motiu i conseqüència:** permet revisar la política amb tests sintètics
i casos reals abans d'autoritzar-la. Shadow no és una nova suite ni una
avaluació amb GPU. Després d'aprovar la fase, la mateixa regla s'activa
per a Quality i posteriorment per a Fastest/Balanced; no es manté per
sempre només com una demostració desconnectada del producte.

**Estat i fonts:** vigent com a mecanisme de revisió; polítiques ja activades.
`7025bb3`, `491b6fc`; [polítiques compartides](../../src/jaull/recommendation/shadow.py).

### D058. Mesurar no dona un bonus ni permet saltar restriccions

**Decisió:** reordenar només les posicions dels membres mesurats d'una
cohort comparable dins d'un tram contigu amb el mateix estrat: viabilitat
confirmada, rebuig, confirmació comercial, suitability, llicència i idioma.
Els plans sense dades mantenen les seves posicions en aquesta etapa.

**Motiu i conseqüència:** una mesura aïllada no promou un model per existir.
Dos plans del mateix estrat, separats per un altre estrat, no s'intercanvien
saltant-hi per sobre. Per això LFM2.5 no intercanvia llocs amb Qwen en els
casos observats amb llicències de classificació diferent, encara que les
seves xifres siguin millors.

**Estat i fonts:** vigent. `7025bb3`, `491b6fc`;
[shadow](../../src/jaull/recommendation/shadow.py),
[tests de polítiques](../../tests/test_shadow_policies.py).

### D059. Quality activa amb fallback exacte i abast declarat

**Decisió:** Quality ordena pels resultats IFEval aplicables dins dels
grups permesos. En empats conserva l'ordre base; sense evidència suficient,
el resultat és el del fallback. Memory no rep aquesta etapa mesurada.

**Motiu i conseqüència:** activació funcional, però no una afirmació que
tots els models estiguin mesurats o que aquests siguin els cinc millors
del món. El catàleg publicat actual continua com a referència, no s'ha
convertit automàticament en ranking quantitatiu.

**Estat i fonts:** vigent. `7025bb3`;
[activació i regressions](../../tests/test_measured_policy_activation.py),
[pla vFinal](../recommendation-vfinal-plan.md).

## 9. Velocitat, Balanced i soroll experimental

### D060. Separar prefill, decode, latència i temps total d'avaluació

**Decisió:** `pp512` i `tg128` són mesures diferents; Fastest usa tg128
com a mètrica principal i pp512 com a informació secundària. No triar
el màxim dels dos ni tractar la durada d'IFEval com a velocitat d'inferència.

**Motiu i conseqüència:** dos tests sintètics no són una conversa de
512 tokens seguida d'una resposta de 128. Throughput no és TTFT ni
capacitat sostenible d'un servei concurrent. Tampoc no s'ha afegit
un predictor de rendiment calibrat amb aquestes mesures.

**Estat i fonts:** vigent. `ecf4659`, `491b6fc`;
[benchmarks](../evidence.md), [pla Fastest/Balanced](../fastest-balanced-plan.md).

### D061. Registrar el protocol reportat, no deduir-lo de la taula Markdown

**Decisió:** afegir JSONL de llama-bench per conservar build, threads,
batches, caches K/V, flash attention i load mode. Camps absents o
contradictoris queden desconeguts. Els registres antics carreguen sense
protocol i segueixen visibles com a referència.

**Motiu i conseqüència:** la taula Markdown omet valors per defecte i no identifica
tot el protocol. Afegir un format de sortida no canvia per si sol la
metodologia v1. Warmup no surt com a camp reportat del binari i s'ha
de distingir del que controla el runner.

**Estat i fonts:** vigent, integrat en l'evolució posterior;
[parser JSONL](../../src/jaull/runtime/llama_bench_jsonl.py),
[contracte de benchmark](../../src/jaull/domain/benchmarks.py).

### D062. `llama_bench_v2` amb condicions explícites i comprovades

**Decisió:** fixar cinc repeticions, `-b 512 -ub 512 -fa off -ctk f16 -ctv f16
-lm mmap` i threads iguals als nuclis físics. Respectar el `-ngl` del pla,
no forçar offload complet per conveniència del benchmark. Comprovar command,
build i valors reportats contra el contracte.

**Motiu i alternativa:** `auto` no demostrava si flash attention es va
aplicar; `off` evita aquesta ambigüitat per a la v2. `-lm` substitueix
el flag obsolet `-mmp` del build utilitzat. Els límits de batch són explícits,
però el runtime pot retallar-los de manera determinista segons el test:
tg128 no demostra un batch aplicat de 512.

**Conseqüència:** una execució completada que no compleix la v2 es conserva
com a referència. No es reescriu un v1 com a v2. Que la v2 desactivi FA
també limita extrapolar-ne el rendiment a una producció amb altres flags.

**Estat i fonts:** vigent. `491b6fc`;
[pla](../fastest-balanced-plan.md), [runner](../../src/jaull/runtime/llama_bench_runner.py),
[tests v2](../../tests/test_llama_bench_v2.py).

### D063. Cohorts de velocitat i comprovació del dispositiu real

**Decisió:** l'aplicabilitat comprova SHA, pla, placement, workload,
runtime, build, màquina, backend, dispositiu i protocol. Les cohorts
inclouen driver i CUDA per separar entorns diferents; cada mesura ha
de coincidir amb el seu propi artefacte, no amb el SHA dels altres models.

**Motiu i conseqüència:** una 4090 no ordena directament plans per a
una 2060. Una sola GPU declarada no prova el dispositiu del runtime:
el banner de llama.cpp ha d'identificar el nom i índex corresponents
sense ambigüitat. Diverses GPU iguals o un índex contradictori bloquegen
la reutilització. L'empremta usada per emparellar evidència arrodoneix
RAM a 256 MiB per tolerar petites variacions de WSL; els manifests
experimentals conserven els bytes exactes.

**Estat i fonts:** vigent. `491b6fc`;
[aplicabilitat local](../../src/jaull/recommendation/local_evidence.py),
[regressions v2](../../tests/test_fastest_v2_applicability.py).

### D064. Una única autoritat de velocitat, sense filtracions a l'ordre base

**Decisió:** `assess_speed` decideix què és aplicable. El ranking no torna
a llegir tok/s crues del benchmark ni usa `performance_evidence` com
a bonus a la clau base o a la signatura de diversitat.

**Motiu i conseqüència:** llegir una velocitat aplicable en la clau base
feia que una única mesura alterés l'ordre abans de formar cohorts,
fins i tot en Memory. La velocitat només reordena mitjançant la política
mesurada. Un benchmark v1 pot continuar aportant evidència d'execució
o memòria quan pertoqui; això no autoritza usar-ne throughput per Fastest.

**Correcció del fallback Speed, 14 d'octubre:** sense evidència comparable,
l'ordre és adequació a la tasca → viabilitat d'execució → coherència del pla
→ empremta de memòria. Un model base petit no ha de guanyar a un instruct
adequat per a xat només per mida. La preferència Q4 de la configuració GGUF
es manté dins de cada repositori, però deixa de ser un avantatge global davant
d'un altre model Q5 o Transformers. Els plans amb offload es distingeixen dels
normalment executables abans de premiar una memòria menor. No és una predicció
de tok/s ni canvia la política d'evidència mesurada, Quality, Balanced o Memory.

**Estat i fonts:** vigent. `491b6fc`;
[motor](../../src/jaull/recommendation/engine_v2.py),
[activació des del motor complet](../../tests/test_measured_policy_activation.py).

### D065. Balanced no té un benchmark propi ni una suma 50/50

**Decisió:** Balanced consumeix qualitat i velocitat aplicables del mateix
pla. La versió inicial registrada a Git ordena per fronts de Pareto dins dels
grups comparables i conserva l'ordre base dins de cada front.

**Motiu i alternativa:** no sumar percentatges de qualitat i tok/s com
si compartissin escala. L'esborrany d'una porta `min_generation_tps`
es va deixar fora d'aquesta activació. Un model més ràpid i un altre
més capaç poden constituir un trade-off sense guanyador únic.

**Conseqüència:** els dos eixos són necessaris; qualitat sola no ordena
Balanced. El parell Qwen petit/ràpid i Qwen gran/més qualitat pot conservar
l'ordre sense que sigui un error. La dominància es prova també amb casos
sintètics, no fabricant un moviment a les mesures reals.

**Estat i fonts:** els fronts de Pareto estan substituïts a `330bf0f` per la
comparació per parelles de D068; la necessitat dels dos eixos continua vigent;
[pla vFinal](../recommendation-vfinal-plan.md),
[pla Fastest/Balanced](../fastest-balanced-plan.md).

### D066. Agregar crides diferents en lloc de fer guanyar l'última

**Decisió:** seleccionar la cohort aplicable més recent i agregar
totes les crides vàlides i diferents d'aquella cohort: mediana tg128,
mediana pp512 quan hi ha dades, nombre de crides, min/max observats i tots
els IDs. Sense finestra temporal ni exclusions perquè un valor és baix.

**Motiu i alternativa:** la campanya intercalada va mostrar que l'ordre
basat en l'última crida pot variar entre sessions. Cinc repeticions dins
d'una crida no equivalen a cinc sessions independents. La desviació
interna no es converteix en una falsa incertesa agregada.

**Conseqüència:** menys de tres crides vàlides es mostren però no ordenen.
Tres és un mínim conservador de producte, no una demostració estadística.
Els IDs duplicats no multipliquen N; continguts diferents amb el mateix
ID es bloquegen. Un benchmark complet dins d'una campanya interrompuda
continua comptant, inclòs el valor 218,6 del Qwen petit. El min/max només
pot créixer en afegir crides de la mateixa cohort: un valor atípic pot
impedir distingir dos plans indefinidament. Encara no hi ha un rang robust.

**Estat i fonts:** vigent a `330bf0f`;
[agregador](../../src/jaull/recommendation/local_evidence.py),
[tests d'agregació](../../tests/test_speed_aggregation.py),
[campanya de soroll](../qwen2.5-tests/rtx2060-tg128-noise-20261011.md).

### D067. Primera revisió del soroll: bandes connexes de rangs solapats

**Decisió local inicial:** agrupar rangs de velocitat que se solapen o
es toquen, incloses cadenes A~B~C, en bandes que mantenen l'ordre base.
Balanced exigia dominància entre bandes completes.

**Motiu i alternativa:** evitar comparadors d'epsilon per parelles amb
empats no transitius i evitar un llindar universal del 10%. Min/max és
rang observat, no interval de confiança. Aquesta opció era conservadora,
però una cadena podia amagar una diferència directa entre A i C.

**Estat i fonts:** substituïda per D068 abans del commit; no hi ha cap commit
d'aquesta etapa intermèdia (`330bf0f` ja inclou D068).
[Pla Fastest/Balanced](../fastest-balanced-plan.md), [recomanació](../recommendation.md).

### D068. Diferències per parelles i Wilson en Balanced

**Decisió:** `shadow.py` manté la mediana i el
mínim de tres crides, però ja no fusiona cadenes de solapaments. Fastest
compara rangs directament: només hi ha precedència de velocitat quan
són disjunts. Balanced afegeix intervals Wilson del 95% sobre encerts
i samples de qualitat i considera precedència quan cap eix és
demostrablement pitjor i almenys un és millor segons aquestes regles.

**Motiu visible al codi:** permetre que una qualitat clarament superior
decideixi quan la velocitat se solapa, sense deduir A~C només de A~B i
B~C. L'ordenació intenta respectar les precedències i conservar l'ordre
base tant com sigui possible; té un fallback si apareix un cicle.

**Cautela acadèmica:** «no demostrablement pitjor» no és una prova
d'equivalència ni de no-inferioritat. Un interval Wilson pressuposa
un model estadístic sobre prompts: les xifres són exactes per als 541
prompts processats, mentre que extrapolar-les a una població requereix
justificar representativitat i dependència. Rangs de velocitat i intervals
de proporcions no són la mateixa mena d'incertesa.

**Estat i fonts:** vigent a `330bf0f`, amb proves de regressió per a
solapaments encadenats i per a Start amb 0, 1, 2 i 3 crides.
El pla Fastest/Balanced la documenta com a revisió amb Ton de l'11 d'octubre.
Ton confirma posteriorment que l'últim experiment de velocitat d'avui està
acabat i que ha seguit el criteri Wilson. Wilson s'aplica al component
de qualitat de Balanced, no als tok/s. La lectura feta aquí no certifica
absència de cicles, correcció estadística o CI d'aquesta revisió.
[Implementació llegida](../../src/jaull/recommendation/shadow.py),
[pla actualitzat](../fastest-balanced-plan.md).

### D096. Predictor de tg128 experimental, auditat abans d'arribar al producte

**Decisió:** provar fora del producte un predictor de tg128 per a models
densos amb full offload: `segons/token = bytes del GGUF / ample de banda + c`,
on `c` és un residual empíric ajustat per màquina i protocol. Es congela la
predicció abans de mesurar i només puntuen mesures posteriors (D071). No es
mostra a la interfície ni ordena Fastest.

**Motiu:** la mateixa comprovació posa la baseline `0,55 × ample de banda / bytes`
entre +4 % i +388 %. No és una execució de llmfit complet, que també
incorpora altres dades i correccions. L'eficiència observada respecte a `ample de banda / bytes` no és
constant en models petits. `ample de banda / bytes` tampoc no és un sostre:
el fitxer no són els bytes llegits per token.

**Resultat observat i correcció de l'auditoria:** a la RTX 2060 (build 10357,
`llama_bench_v2`), amb `c` ajustat amb tres artefactes, quatre artefactes
reservats donen errors entre −6,1 % i +10,5 %. Però la calibració es va fer
amb WSL `6.18.40.1` i les mesures amb `6.18.54.1`: no són la mateixa cohort
segons el contracte de Jaull. Les xifres són observacions de transferència
entre entorns, no precisió validada dins de la mateixa cohort. L'scorer
corregit rebutja la barreja; prediccions, scores i logs originals es conserven.
Dos artefactes reservats pertanyen a models Qwen presents a la calibració;
no són quatre models nous independents.

**Cautela:** una GPU, un build, artefactes fins a 1,3 GB i set plans. És
un experiment exploratori, no una garantia d'error. Ajustar `c` amb mesures
s'aparta de D079, que demana que els predictors derivin de metadades i
les mesures els jutgin.

**Ampliació de presentació, 14 d'octubre, a petició de Ton:** mostrar la lectura
a Results i un botó `Speed` al costat que obre el benchmark del pla exacte,
sense executar-lo automàticament. Es distingeixen `Measured`,
`Est. ... (experimental)` i `Speed unknown`. La predicció es recalcula amb
registres locals compatibles, no amb un coeficient fix de la campanya antiga:
una sola cohort, almenys dos altres artefactes amb tres crides cadascun,
mateixa màquina, kernel, driver i build. L'abast inicial de presentació és
GGUF dens Qwen2/Llama amb offload CUDA complet i una petició interactiva.
Transformers, offload parcial i condicions desconegudes no reben una xifra.
Les referències històriques només es mostren als detalls de procedència.

**Ampliació heurística aprovada el mateix dia:** no exigir mesures locals per
a una primera aproximació. Si no hi ha mesura aplicable ni calibració vàlida,
Results pot mostrar `Est. ... (heuristic)` per a arquitectures denses conegudes
de llama.cpp i Transformers. Utilitza els bytes de pesos estimats del pla
(amb la seva precisió/quantització), no la mida del repositori ni el pressupost
de memòria amb reserves. El model és
`0,5 / (bytes_GPU / BW_GPU + bytes_host / BW_RAM)`.

El `0,5` és una hipòtesi d'escenari sense validar, no una eficiència mesurada
ni una correcció que s'hagi demostrat millor que `0,55`. L'ample de banda de
GPU ve de la taula de fitxes tècniques; els `50 GB/s` de RAM són una hipòtesi
explícita, no una lectura del maquinari. CPU i offload Transformers poden rebre
aquesta estimació. En offload es condiciona a la col·locació de planificació
d'HFA, no es confirma la col·locació del runtime; si hi ha dos extrems de
pesos no-bloc, es mostra el rang derivat, que no és un interval de confiança.
L'offload GGUF parcial continua sense estimació: no es converteixen blocs HFA
en unitats de llançament de llama.cpp. MoE, arquitectura desconeguda, manca
de pesos/col·locació, pla que no cap i càrrega batch/multiusuari s'abstenen.
No s'hereta mai la calibració de llama.cpp en un pla Transformers.

Aquesta ampliació prioritza utilitat orientativa amb procedència explícita;
**no afirma precisió validada**, no modela per separat context/KV, còmput,
desquantització o transferències i no ordena recomanacions. El botó `Speed`
continua sent una comprovació opcional amb el pla exacte.

**Contrast real del fallback Transformers, 14 d'octubre:** Qwen2.5-0.5B FP16
va donar 18,36 tok/s al benchmark de l'usuari davant d'uns 170 estimats.
Un diagnòstic amb els mateixos pesos entre mètodes confirma tots els paràmetres
a CUDA i FP16: el runner dona 19,63 tok/s i `model.generate` 17,87, amb el
mateix recorregut de tokens, en dues passades intercalades. No és una
penalització de 9x exclusiva de Jaull; és una refutació clara de la precisió
de la heurística en aquest cas. No s'ha ajustat l'eficiència per encaixar-hi.
No identifica encara el coll d'ampolla ni el màxim rendiment possible de la
GPU. [Diagnòstic i originals](../../validation/qwen2.5-0.5b-transformers-decode-2060-20261014/README.md).

**Retirada de les prediccions de Results, aprovada el mateix dia:** després
d'aquest contrast, es retiren tant la heurística sense calibrar com la predicció
calibrada de la interfície. La procedència explícita no compensa mostrar una
xifra sense prou precisió validada. Results mostra només `Measured ... tok/s`
quan la mesura és aplicable segons el contracte existent. Si no ho és, no es
mostra cap etiqueta de velocitat; es conserva el botó `Speed`. Les mesures
d'altres condicions continuen com a referències
històriques als detalls, no com a velocitat atribuïda al pla actual.

L'excepció de presentació a D079 queda retirada: el predictor es conserva només
com a experiment fora del producte, juntament amb els resultats originals i el
diagnòstic. No es modifica el ranking, el model de memòria, la cerca guardada,
els registres ni els informes exportats. Les ampliacions anteriors d'aquest
apartat documenten el procés de decisió, no el comportament vigent.

**Visibilitat dels benchmarks Transformers guardats:** es distingeix una mesura
aplicable al ranking d'un resultat històric. Results també pot mostrar
`Last benchmark ... tok/s (Transformers)` per a l'última execució correcta
amb generació de 128 tokens i metadades de repositori, runtime, precisió,
backend i màquina coincidents. Un conflicte conegut de SHA impedeix mostrar-la.
La data, el protocol, les repeticions i l'ID consten als detalls. No verifica
els bytes exactes ni l'execució actual, no substitueix una mesura aplicable i
no entra al ranking Fastest/Balanced. Així es fan visibles les mesures ja
guardades sense convertir una revisió mutable com `main` en una identitat immutable.

**Estat i fonts:** experiment del 14 d'octubre, sense commit en redactar-se.
[Informe](../qwen2.5-tests/rtx2060-speed-predictor-20261014.md),
[script](../../scripts/speed_predictor.py),
[lectura de presentació](../../src/jaull/presentation/speed.py),
[prediccions congelades](../../validation/speed-predictor-2060-20261014/frozen-predictions.json),
[reavaluació auditada](../../validation/speed-predictor-2060-20261014/scores-audited.json).

## 10. Execució, observació i reproducció experimental

### D069. Detectar maquinari, instal·lació i capability per separat

**Decisió:** detectar CPU/RAM i acceleradors; localitzar runtimes per
ruta explícita, registre, PATH i directoris acotats; sondejar què
exposa cada build i després decidir el backend i readiness.

**Motiu i conseqüència:** que hi hagi una GPU CUDA o Vulkan no prova
que el binari la suporti. `READY` vol dir que no s'ha trobat un bloqueig,
no que el model arrencarà segur. HIP/ROCm i Vulkan tenen camins implementats
i proves de codi, però no equivalen a campanyes reals validades en aquest TFG.

**Estat i fonts:** vigent. `24d6c26`, `ea26ffc`, `a53bc11`, `f14b3e9`;
[runtimes](../evidence.md), [limitacions](../limitations.md).

### D070. Execució local controlada i workers aïllats

**Decisió:** executar GGUF amb `llama-cli --single-turn`; executar
Transformers mitjançant un worker Python separat. Capturar stdout,
stderr, exit code, durada i observacions amb timeout i cancel·lació.

**Motiu i conseqüència:** un procés interactiu no ha de deixar penjada
una acció d'un sol prompt. Aïllar imports i inferència pesants evita
barrejar-los amb el loop de la TUI; no és un sandbox de seguretat per
executar codi generat. La CLI Run i la TUI no tenen exactament el mateix
suport d'artefactes, i això s'ha de declarar.

**Estat i fonts:** vigent. `309b74d`, `900ed05`, `01e80fd`;
[execució](../evidence.md), [llama.cpp](../llamacpp.md).

### D071. Congelar predicció abans de mesurar

**Decisió:** `MemoryEstimate`, `ExecutionObservation`, `BenchmarkObservation`
i `PredictionComparison` són objectes separats. Registrar inputs,
maquinari, pla, procedència i predicció abans de l'execució.

**Motiu i conseqüència:** una observació no reescriu retrospectivament
el que s'havia predit. Reavaluar offline amb el codi actual és una
vista derivada que conserva la predicció original; no detecta la
màquina actual, no consulta el Hub ni crea una nova execució.

**Estat i fonts:** vigent. `9d411ce`, `2ca2e70`, `a57229c`;
[protocol experimental](../experimental-protocol-biosfer.md),
[reevaluació](../../src/jaull/experiments/reevaluation.py).

### D072. JSON immutable per registre, sense SQLite innecessari

**Decisió:** persistir experiments, benchmarks i qualitat com a
registres individuals versionats al directori de dades de l'usuari.
No introduir SQLite només per aquesta fase.

**Motiu i conseqüència:** facilita inspecció, còpia i auditoria per
a una campanya petita. El store valida abans de reutilitzar, però
la persistència local no garanteix autenticitat ni que qualsevol
JSON aportat per tercers sigui una mesura fiable.

**Estat i fonts:** vigent. `db34670`, `da3e41e`;
[persistència](../evidence.md), [pla vFinal](../recommendation-vfinal-plan.md).

### D073. Errors, interrupcions i sortides incompletes també es conserven

**Decisió:** mantenir intents fallits amb diagnòstic, sortida i codi
real. Exit zero sense totes les files necessàries no és benchmark
complet. Un intent fallit no substitueix una mesura vàlida com a evidència
de throughput, però no s'esborra per fer més bonics els resultats.

**Motiu i conseqüència:** OOM, timeout, parseig fallit o backend incorrecte
informen la qualificació. Només els registres que passen el contracte
de la mètrica entren a comparació i ranking. Cancel·lar una cua conserva
els candidats ja completats i neteja processos que l'avaluador controla.

**Estat i fonts:** vigent. `868b80c`, `1387178`, `fc663bf`;
[validació experimental](../experimental-validation.md), [evidència](../evidence.md).

### D074. Casos i bundles enllacen evidència, no la reinventen

**Decisió:** manifests de cas referencien experiments, benchmarks i
fitxers originals amb identitats i hashes. Exportació portable amb
validació abans i després de transferir. Rebutjar rutes que surtin
de l'arrel d'evidència, també mitjançant symlinks.

**Motiu i conseqüència:** un cas pot ser vàlid, parcial o inconsistent;
un manifest sol no conté els logs. Els hashes són de bytes exactes,
inclosos finals de línia. No reparar històrics canviant digests després
d'una conversió de fitxers. Els nous exports JSON són LF.

**Estat i fonts:** vigent. `a57229c`, `71440c6`, `f98595d`;
[cases](../../src/jaull/cases), [validació](../experimental-validation.md),
[evidència original](../../validation/README.md).

### D075. Comparabilitat és una porta, no una etiqueta estètica

**Decisió:** rebutjar ratios, diferències atribuïdes i guanyadors
quan identitat, metodologia, configuració o mètrica no permeten
la comparació. Mostrar els valors com a referències amb bloquejos.

**Motiu i conseqüència:** no comparar l'`--ctx-size` heretat del pla
com si llama-bench l'hagués aplicat: el command no passa `-c`.
No comparar latència de Validate amb tg128 com una mateixa mètrica.
Una baseline de zero no té ratio relatiu definit. «No comparable»
no vol dir zero error ni que els dos models siguin equivalents.

**Estat i fonts:** vigent. `56a1864`, `491b6fc`;
[comparació experimental](../experimental-validation.md),
[comparador de benchmarks](../../src/jaull/evaluation/benchmark_comparison.py).

### D076. Sincronitzar temps GPU de Transformers i versionar la metodologia

**Decisió:** els benchmarks `transformers_isolated_inference_v3`
sincronitzen la GPU i separen prefill de decode estable. Els v2
anteriors continuen llegint-se sense reetiquetar-los.

**Motiu i conseqüència:** mesurar només l'enviament asíncron de kernels
no mesura el temps efectiu d'execució. Una correcció metodològica
requereix una nova mesura si es vol evidència del protocol actual.
Això no converteix Transformers en evidència Fastest de la cohort
llama-bench ni valida timing multi-GPU.

**Estat i fonts:** vigent. `868b80c`;
[limitacions de mesura](../limitations.md), [validació](../experimental-validation.md).

### D077. Dues fonts de VRAM, sense fingir equivalència

**Decisió:** separar memòria atribuïda al procés per NVML de buffers
de dispositiu reportats pel runtime. Registrar font, scope i
`driver_confirmed = false` per al segon cas. Excloure `CUDA_Host`
de la suma de memòria del dispositiu.

**Motiu i conseqüència:** la GeForce sota WDDM no proporciona la
VRAM per procés que necessita aquesta comparació, tant en Windows
com en WSL. Una lectura global inclou l'escriptori i altres processos.
Els buffers de llama.cpp donen evidència útil, però no inclouen
necessàriament context CUDA i tot l'allocator.

**Estat i fonts:** vigent. `1a19b3c`, `3a6dc26`;
[contracte d'observació](../qwen2.5-tests/b001-r7-observation-contract.md),
[parser](../../src/jaull/runtime/llama_cpp_memory_report.py).

### D078. Bloquejar comparacions de memòria que mesuren una altra cosa

**Decisió:** no comparar RSS sota offload com si fos la part host
de la col·locació. No comparar una predicció física de GPU quan
el placement és ambigu o només hi ha límits d'extrems.

**Motiu i conseqüència:** mmap pot deixar residents les pàgines
del fitxer complet després de pujar pesos a GPU. El problema no
és que falti un split host/dispositiu al model, sinó que RSS no
mesura aquest split. Overhead contra compute buffer és un proxy,
no una base suficient per calibrar totes les assignacions.

**Estat i fonts:** vigent. `1a19b3c`, `3a6dc26`;
[B001-R4](../qwen2.5-tests/b001-r4-rtx2060-baseline.md), [limitacions](../limitations.md).

### D079. No calibrar prediccions amb throughput o una sola GPU

**Decisió:** els predictors deriven de metadades i polítiques;
les mesures els jutgen. No utilitzar tok/s com a constant de
calibració de memòria ni reduir reserva/headroom per encaixar
una campanya local favorable.

**Motiu i conseqüència:** evita overfitting experimental. La
millora tensor-exacta es basa en una regla estructural observada
del runtime, no en ajustar a mà un percentatge perquè surti
l'offload desitjat. Validar més maquinari continua sent necessari.

**Estat i fonts:** vigent. [Protocol](../experimental-protocol-biosfer.md),
[B001-R6](../qwen2.5-tests/b001-r6-tensor-policy.md), [validació](../experimental-validation.md).

## 11. Interfície, rendiment i verificació del producte

### D080. Progress real, cancel·lació i lifecycle sense bloquejos

**Decisió:** mostrar tasques realment completades, mantenir la
cerca cancel·lable i comunicar resultats de workers amb patrons
que no deixin un fil esperant una app ja tancada. Esperar el muntatge
dels widgets abans de consultar-los.

**Motiu i conseqüència:** «no avança» pot ser xarxa, una feina llarga
o un bug de lifecycle. Una animació no simula percentatge de progrés.
Les correccions de `call_from_thread`, `post_message`, muntatge i
IDs duplicats són decisions de fiabilitat, no canvis al ranker.

**Estat i fonts:** vigent amb regressions successives. `900ed05`,
`4e54b21`, `7025bb3`, `5ef3635`;
[TUI](../../src/jaull/tui), [galeria](../ui.md).

### D081. Espai d'avaluació amb jerarquia visual i accions visibles

**Decisió:** separar Candidates/Artifact, Settings i Results;
mostrar dataset i permisos fora d'Advanced; plegar rutes, SHA
i diagnòstics llargs. Mostrar readiness amb estats inequívocs,
errors concrets i candidats locals o pendents de descàrrega.

**Motiu i conseqüència:** inspiració en eines de terminal com llmfit,
però sense copiar-ne el model de memòria o presentar el resultat
com a equivalent. A 80 columnes Start, Cancel i Back han de seguir
accessibles; detalls complets no han de convertir la primera vista
en un mur de text.

**Estat i fonts:** vigent. `7025bb3`, `491b6fc`;
[pantalles](../ui.md), [avaluació](../../src/jaull/tui/screens/quality_evaluation.py).

### D082. Consentiment separat i guanyadors només quan es poden comparar

**Decisió:** no iniciar feina per obrir una pantalla. Separar permís
de descarregar models del permís del dataset. Prepare torna a
comprovar readiness i Start; canviar de suite neteja resultats i
propostes anteriors sense esborrar registres persistits.

**Motiu i conseqüència:** un canvi visual no ha de llançar Docker
ni consumir GPU. Una taula sempre pot mostrar dades, però no
marca guanyador quan difereixen protocols. En velocitat
(`330bf0f`), un guanyador requereix també rang separat i prou
crides; una nota de qualitat reutilitzable no bloqueja mesurar Speed.

**Estat i fonts:** vigent a `330bf0f`;
[readiness](../../tests/test_quality_readiness.py),
[avaluació de velocitat](../../tests/test_tui_speed_evaluation.py).

### D083. Evaluate candidates segons la prioritat, sobre plans exactes

**Decisió:** Quality proposa IFEval, Fastest un benchmark sense
Docker ni dataset, i Balanced els dos eixos sobre els mateixos
GGUF. La cua és inicialment seqüencial. No canviar la quantització
del pla per «la millor del repositori» mentre s'avalua.

**Motiu i conseqüència:** una mesura d'altres bytes no podria
ordenar la cerca. Es reutilitzen resultats aplicables i s'executa
només l'eix que falta; un conflicte de qualitat no es resol amb
un rerun automàtic. Des de `330bf0f`, prémer Start autoritza les
crides que falten fins a tres: 0→3, 1→2 i 2→1. S'intercalen entre
els plans seleccionats, amb 60 segons de refredament entre crides.
Un pla amb tres o més, si es marca manualment, rep una crida extra.
No és un scheduler periòdic: és la cua de l'acció que l'usuari ha iniciat.

**Estat i fonts:** vigent a `491b6fc`, amb refrescos locals;
[candidats](../../src/jaull/advisor/quality_candidates.py),
[pantalla](../../src/jaull/tui/screens/quality_evaluation.py).

### D084. Recompare és una nova proposta del pool guardat

**Decisió:** conservar l'ordre base i les recomanacions mostrades;
rellegir evidència local i calcular la política sobre els mateixos
plans, sense nova cerca al Hub. No modificar silenciosament
l'original ni anunciar com a nous moviments ja aplicats.

**Motiu i conseqüència:** separa prior recommendation, experiment,
actualització d'evidència i revised recommendation. «No moves»
necessita motius per pla: SHA absent, altra quantització, falta
un eix, cohorts incompatibles, estrat diferent o dades insuficients.
Els refrescos locals de final de run no han de replanificar al Hub.

**Estat i fonts:** vigent, amb correccions locals de refresc;
`7025bb3`, `491b6fc`, `5ef3635`;
[shadow i Recompare](../../src/jaull/recommendation/shadow.py),
[tests TUI](../../tests/test_tui_ifeval.py).

### D085. Informes que expliquen l'ordre real i no inventen posicions

**Decisió:** JSON i Markdown separen criteris base de `quality_policy`,
`speed_policy` i `balanced_policy`, amb SHA, cohort, IDs, mètrica
i posicions abans/després quan es poden reconstruir. La revisió
local afegeix nombre de crides i rang.

**Motiu i conseqüència:** un valor mesurat no és un nou eix de la
clau base. Si una cerca antiga no coincideix amb la política
actual, les posicions no es fabriquen. Cal preservar l'atribució
concreta en serialitzar subclasses de referències publicades;
un export ric no prova que tots els parsers legacy el recarreguin
amb els mateixos camps.

**Estat i fonts:** vigent amb ampliació local. `982d408`,
`7025bb3`, `491b6fc`;
[reporting](../../src/jaull/reporting/recommendation.py),
[regressió de fidelitat](../../tests/test_report_fidelity_regression.py).

### D086. Caché acotada i observabilitat abans d'optimitzar a cegues

**Decisió:** memòria cau persistent d'anàlisis i capçaleres GGUF,
memoització dins del run, concurrència limitada i telemetria
de latència per fase. No persistir `tokenizer.*` quan no es
reutilitza i ocupa gran part de la memòria cau.

**Motiu i conseqüència:** una cerca freda pot costar minuts per
metadades i shards. Una prova registrada va passar de 269 s
a 8,9 s en una repetició amb memòria cau; no és una garantia de latència
per a qualsevol cerca ni justifica augmentar sense límit la concurrència.

**Estat i fonts:** vigent. `048e197`, `6cc7c82`, `36494ad`;
[arquitectura](../../ARCHITECTURE.md), [diari](../JAULL_DEVLOG.md).

### D087. Alliberar disc sense destruir evidència

**Decisió:** oferir una vista de models locals i output folders,
amb confirmació abans d'eliminar. Els registres persistits es
conserven; l'usuari pot recuperar espai de pesos o sortides.

**Motiu i conseqüència:** la campanya pot ocupar gigabytes i
convertir la neteja en part del flux normal. Eliminar fitxers
que un manifest referencia pot fer que aquell bundle deixi
de ser complet: conservar el registre no conserva màgicament
tots els bytes de l'experiment.

**Estat i fonts:** vigent. `7025bb3`;
[regressions de Storage](../../tests/test_storage_cleanup.py), [UI](../ui.md).

### D088. Proves offline, casos complets i esperes per condicions

**Decisió:** fakes per a xarxa/maquinari, carpetes de dades
aïllades, snapshots i captures headless en diverses mides,
tests de pipeline complet i CI Linux/Windows. Esperar condicions
observables amb timeout, no un nombre fix de `pause` o un sleep
que codifica la velocitat d'una màquina.

**Motiu i conseqüència:** els tests llegien stores personals
grans i es feien lents; els falsos errors Windows exposaven
scheduling diferent de Textual. Proves parell a parell no van
detectar un Top 5 dolent; afegir un cas complet i comprovar
que falla en reintroduir el bug aporta una garantia més útil.

**Estat i fonts:** vigent. `2ca2e70`, `4e54b21`, `7c68cb0`,
`7025bb3`, `5ef3635`;
[tests](../../tests), [CI](../../.github/workflows).

### D089. Finals de línia, procedència i logs compartibles

**Decisió:** fixar LF per al codi i exports nous, protegir els
bytes originals sota `validation/`, capturar el commit només
si pertany al checkout de Jaull i registrar dirty state a les
campanyes. Logs raw opt-in perquè poden contenir prompts
i respostes; no guardar secrets en informes.

**Motiu i conseqüència:** Git no ha d'alterar evidència en
transportar-la a Windows; una instal·lació no ha d'atribuir-se
el commit d'un repositori pare. Compartir un bundle requereix
revisar contingut i privacitat, no només executar hash validation.
La publicació anonimitzada de la campanya 4060 es distingeix
dels bytes originals conservats privadament; anonimitzar no
autoritza a presentar fitxers modificats com els originals.

**Estat i fonts:** vigent. `0c402b1`, `7fc8ae7`, `310ce63`;
[validació experimental](../experimental-validation.md), [.gitattributes](../../.gitattributes).

### D090. Elements visuals decoratius fora del contracte científic

**Decisió:** animacions marines i del tauró a pantalles d'espera,
i xifres de memòria destacades, sense que les animacions reportin
dades o condicionin el ranking. Les superfícies de treball
prioritzen lectura, navegació i informació plegable.

**Motiu i conseqüència:** comunicar activitat sense inventar
progrés. Fonts, glifs, color del terminal i cost d'animació
són limitacions de presentació, no errors de la mesura ni
evidència de rendiment del model.

**Estat i fonts:** vigent. `a98167d`, `7025bb3`;
[UI](../ui.md), [limitacions visuals](../limitations.md).

## 12. Resultats que van justificar decisions

Aquesta secció resumeix proves que es poden utilitzar a la memòria. **No són
benchmarks universals ni substitueixen els manifests i registres originals.**

### 12.1. Col·locació, memòria i observació

| Observació registrada | Decisió que sosté | Límit de la conclusió |
|---|---|---|
| Descriptors dels darrers blocs i output de dense qwen2 reprodueixen el model buffer CUDA0 amb diferència aproximada de 0,01 MiB en els punts provats | D030-D031: separar HFA i launch policy; refinament tensor-exacte | Un model/layout, build i backend verificats; no tota la memòria del procés |
| B001-R4: RSS aproximat de 4723 MiB amb offload parcial i 4724 MiB amb offload complet | D078: no comparar RSS amb el split host sota mmap | No identifica per si sol les pàgines residents de qualsevol runtime |
| WDDM dona memòria NVML per procés no disponible en Windows i WSL | D077: conservar font runtime-reported separada | Cal altra plataforma per obtenir atribució de driver usable |
| Casos documentats: total físic predit aproximat 3535,6 MiB, buffers CUDA aproximats 2892,63 MiB; diferència -18,2% | D025, D077-D079: separar pressupost, observació i proxy d'overhead | No és error contra tota la VRAM del procés; compute buffer no equival a tot l'overhead |
| Qwen 7B Q4_K_M: throughput varia molt segons el nivell d'offload i no és estrictament monòton als punts baixos | D060 i D079: mesurar la configuració i no calibrar amb una regla simple de mida | Una GPU, model i build; no una corba universal d'offload |

Fonts: [B001-R4](../qwen2.5-tests/b001-r4-rtx2060-baseline.md),
[B001-R6](../qwen2.5-tests/b001-r6-tensor-policy.md),
[B001-R7](../qwen2.5-tests/b001-r7-observation-contract.md),
[B001-R9](../qwen2.5-tests/b001-r9-launch-policy-throughput.md),
[evidència original](../../validation/README.md).

### 12.2. Qualitat controlada i abast de la comparació

| Cas | Resultat registrat | Què es pot afirmar |
|---|---|---|
| HellaSwag, 100 exemples fixats: TinyLlama Q4_K_M i Qwen2.5-1.5B Q5_K_M | 36%/38% acc/acc_norm i 46%/63% | Comparació limitada d'aquest subset i protocol, no capacitat general |
| IFEval complet: Qwen2.5-1.5B Q5_K_M i Qwen2.5-0.5B Q4_K_M | 222/541 = 41,0% i 138/541 = 25,5%; truncament aproximat del 12% i 37% | Encert en aquest split; canvien mida i quantització alhora |
| IFEval posterior del Qwen2.5-0.5B Q5_K_M | 25,9%; dues execucions amb les mateixes respostes | Deduplicació del resultat determinista, no dues mostres independents |
| IFEval LFM2.5-1.2B Q5_K_M | 434/541 = 80,2% | Bon resultat en seguiment d'instruccions sota aquest contracte; no domini universal |
| Recompare controlat amb els dos Qwen de la mateixa cohort i estrat | El 1.5B passa del #2 al #1 amb Quality | Demostra aplicació funcional de la política sense alterar l'original |

No barrejar el **25,5% del Q4_K_M** amb el **25,9% del Q5_K_M** com si
fossin una correcció del mateix experiment. Els truncaments i possibles
degeneracions formen part del resultat sota el límit de generació fixat.

Fonts: [pilot](../quality-evaluation-pilot.md),
[diari](../JAULL_DEVLOG.md), [auditoria i campanya IFEval](../recommendation-vfinal-plan.md),
[pla de la parella qualitat/velocitat](../fastest-balanced-plan.md).

### 12.3. Soroll de velocitat i canvi d'agregació

Campanya de l'11 d'octubre: RTX 2060, tres rondes intercalades, 60 segons
de refredament abans de cada crida, cinc repeticions internes, Q5_K_M.

| Model | R1 tg128 | R2 tg128 | R3 tg128 | Mitjana de les tres rondes | Rang observat |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | 227,5 | 256,1 | 247,1 | 243,6 | 227,5-256,1 |
| Qwen2.5-1.5B | 131,4 | 150,9 | 150,4 | 144,3 | 131,4-150,9 |
| LFM2.5-1.2B | 229,4 | 231,6 | 222,2 | 227,7 | 222,2-231,6 |

Unitats: tok/s. La mitjana d'aquesta taula **no és la mediana de tota
la cohort que calcula D066**: aquesta pot incloure crides anteriors
vàlides i la crida de 218,6 de l'intent interromput.

Qwen petit i LFM tenen rangs solapats malgrat una diferència aproximada
del 7% a la mitjana. Els rangs observats dels dos se separen del Qwen
1.5B en aquesta campanya. Això no és un test de significació. A més,
la diferència d'estrat de llicència impedeix atribuir un intercanvi
Qwen/LFM a la política real d'aquell cas.

El soroll entre crides supera la desviació interna en el cas del
Qwen petit; no es pot generalitzar aquesta desigualtat als tres
models ni a sessions futures. La dispersió percentuada `(max-min)/mitjana`
no és desviació estàndard ni interval de confiança.

Font: [campanya de soroll](../qwen2.5-tests/rtx2060-tg128-noise-20261011.md).

### 12.4. Comparació 2060/4060 i controls no equivalents

La campanya 4060 inclou Validate reeixit i benchmarks complets. En els
controls documentats de Qwen 7B amb offload complet, la 2060 dona
aproximadament 60,85 tok/s i la 4060 52,73 tok/s. Canvien build,
entorn i condicions: **això no demostra que una 2060 sigui més ràpida
que una 4060 en igualtat de condicions**.

Aquest resultat reforça D063, D071 i D075: enregistrar màquina i
protocol, distingir comparació descriptiva de comparació controlada
i no explicar un stall o una diferència de throughput canviant
diverses variables alhora. El control full versus 28 unitats dins
de la 4060 tampoc no és una campanya aleatoritzada de només placement.

Fonts: [investigació 4060](../qwen2.5-tests/rtx4060-b001-r10-investigation.md),
[comparació de campanyes](../qwen2.5-tests/rtx4060-campaign-comparison.md).

### 12.5. Confirmació de l'últim experiment del dia

**Actualització comunicada per Ton, 11 d'octubre de 2026:** l'últim
experiment de velocitat està acabat i s'ha seguit el criteri Wilson
en la revisió de comparació. No queda registrat aquí com una execució
pendent. La metodologia executable combina la mediana i el rang
observat per a velocitat amb Wilson sobre l'encert IFEval per a
Balanced, tal com recull D068.

La confirmació no aporta noves xifres ni IDs de benchmark en el missatge;
per això no s'afegeix una taula numèrica nova ni s'assumeix que sigui
exactament el mateix conjunt de registres de la secció 12.3. Per a
la memòria, la confirmació de finalització i els resultats numèrics
verificables han de continuar distingint-se. L'acceptació del criteri
tampoc no és, per si sola, una prova de significació estadística.

Fonts: confirmació de Ton durant aquesta redacció,
[política documentada](../fastest-balanced-plan.md) i
[implementació de comparació](../../src/jaull/recommendation/shadow.py).

## 13. Decisions de tancament i treball pendent

### D091. Tancar quan el procés és coherent, no quan s'han trobat «els millors del món»

**Decisió de projecte:** l'última fase funcional ha de permetre explicar
per què entra cada recomanació, per què ocupa aquella posició, quina
evidència té, què s'ha estimat i què falta. Validar escenaris representatius
2060, 4060 i CPU-only, chat/coding i Quality/Fastest/Balanced.

**Motiu i conseqüència:** limita el desenvolupament i deixa temps per
la memòria i Biosfer. La matriu històrica de 27 casos sense moviments
demostrava fallback i manca de cobertura, no que les polítiques
fossin inútils ni que totes les cerques futures donarien zero moviments.

**Estat i fonts:** criteri vigent; no certifica release o feature freeze
formal. [Pla vFinal](../recommendation-vfinal-plan.md),
[matriu](../../scripts/shadow_review_matrix.py).

### D092. Ampliar cobertura sense heretar notes entre quantitzacions

**Decisió:** la manca de mesures aplicables és un problema de cobertura
del pool, no un motiu per relaxar el SHA exacte. Avaluar els artefactes
que Search realment ha seleccionat és preferible a escollir models
a mà i esperar que després coincideixin.

**Motiu i conseqüència:** l'1.5B podia no arribar a inspecció i el
0.5B podia sortir Q6_K mentre s'havia mesurat Q4/Q5. Una futura
ampliació de variants ha d'explicar identitat, pressupost i viabilitat;
no afegir models de manera amagada només perquè estan mesurats.
Comparar BF16/Q8/Q5/Q4 és una campanya pendent, no una conclusió actual.

**Estat i fonts:** problema identificat i línia de treball pendent;
[pla vFinal](../recommendation-vfinal-plan.md),
[cobertura](../fastest-balanced-plan.md).

### D093. Campanya final petita i casos de Biosfer separats

**Decisió de projecte:** preparar aproximadament 3-5 casos ben triats
amb inputs, maquinari, shortlist, plans, HFA i evidència; executar
Validate/Benchmark/Evaluate reals d'alguns candidats i conservar
els registres per a taules i gràfiques de la memòria.

**Motiu i conseqüència:** corregir bugs reals, però documentar millores
de cobertura o precisió fora d'abast. El protocol Biosfer proposa
workloads i variables de cost/maquinari; no acredita que la integració
amb Biosfer, GPU llogada, càlcul de cost o totes les classes s'hagin
executat. Les campanyes 2060/4060 no validen automàticament aquests casos.

**Estat i fonts:** protocol definit; integració i campanya final pendents.
[Protocol Biosfer](../experimental-protocol-biosfer.md),
[comparació 4060](../qwen2.5-tests/rtx4060-campaign-comparison.md).

### D094. Ampliacions explícitament ajornades

**Decisió:** no convertir aquesta fase en un Open LLM Leaderboard
local. HumanEval/MBPP requereix sandbox aïllat per executar codi
generat. Writing, translation i summarization no tenen aquí una
suite local full auditada pròpia; el RAG correspon al treball Biosfer.

**Conseqüència:** tampoc no es dona per fet suport qualitat multiruntime,
multi-GPU, MoE general, portabilitat de notes entre placements, TTFT
de servei, load generator concurrent, autotuning, calibració de
reserva amb n=1 o actualització automàtica del catàleg publicat.
Els experiments puntuals 1/2/4 usuaris no són un servidor de producció.

**Estat i fonts:** pendent o fora d'abast segons el cas;
[pla vFinal](../recommendation-vfinal-plan.md),
[limitacions](../limitations.md), [protocol Biosfer](../experimental-protocol-biosfer.md).

### D095. Release i feature freeze només després de verificar el tancament

**Decisió de projecte:** revisar instal·lació neta, smoke manual,
docs i CI Linux/Windows abans de congelar una versió. Després,
prioritzar bugs, memòria, metodologia, resultats i demo.

**Motiu i conseqüència:** «tots els tests locals passen» no substitueix
la CI Windows ni valida nous canvis sense commit. S'ha parlat de
`v0.2.0-alpha` o `v0.2.0`, però el `pyproject.toml` llegit continua
a `0.1.0`; aquest document no anuncia aquella release ni la crea.

**Estat i fonts:** intenció pendent de tancament formal;
[pla vFinal](../recommendation-vfinal-plan.md), [STATUS](../../STATUS.md),
[versió del paquet](../../pyproject.toml).

### D097. Selecció automàtica d'una configuració tècnicament viable

**Data i problema observat:** 2026-10-19. En un Ryzen 5 5500U amb WSL2 i
aproximadament 7,4 GiB de RAM, Results podia presentar un Transformers INT4/INT8
que cabia segons l'estimació però que la ruta actual de Jaull bloquejava en CPU.
Una conversió GGUF confirmada podia aparèixer només després d'obrir Paths.

**Causa tècnica:** `select_configuration` aturava l'escala en trobar memòria
suficient; `generate_execution_plans` produïa una única precisió Transformers;
`build_execution_plan` comprovava només el format/runtime; i el workflow no
passava l'estimador amb memòria cau al ranking per estimar altres variants.
La diversificació també podia agrupar repositoris només pel sufix del nom.

**Alternatives considerades:** ocultar `Blocked`, rebutjar runtimes no instal·lats,
preferir sempre GGUF o expandir totes les combinacions. Les quatre opcions
perden informació o barregen compatibilitat, readiness i preferència de política.

**Decisió implementada:** compartir regles pures de suport de backend; distingir
el fit de memòria de la compatibilitat tècnica i de la instal·lació. Estimar
cada configuració amb les APIs i les memòries cau del workflow, amb màxim quatre
precisions Transformers i vuit variants GGUF, retenint la variant ja seleccionada.
Un artefacte Transformers empaquetat continua sent una representació concreta.
Quan no existeix una ruta viable coneguda, reutilitzar conversions inspeccionades
i hits de cerca amb linatge declarat abans de fer, com a màxim global, tres
consultes dirigides i sis inspeccions addicionals. No es descarreguen pesos ni
s'executen models durant Search.

**Motiu i conseqüència:** la política existent escull el millor pla elegible de
cada identitat confirmada. Les altres configuracions conserven la seva estimació
i motiu a Paths. Un runtime absent continua sent una condició de readiness,
segons D040/D069. Results, reconstrucció de plans i exportació utilitzen la
configuració seleccionada. Un SHA conegut contradictori impedeix també el matching
de referències locals; no es modifica IFEval, Wilson, tg128, cohorts ni mediana.

**Proves i evidència:** `tests/test_compatible_plan_selection.py` cobreix CPU,
CUDA, Vulkan/HIP observats, backends absents, linatge incert, múltiples quants,
pressupostos, reutilització, llicència, SHA/backend i coherència Results/Paths/export.
La regressió de diversitat declara ara `base_model` explícitament quan exigeix
agrupar un model i una conversió. La consulta de metadades del 2026-10-19 va trobar
linatge explícit i fitxers Q4_K_M/Q5_K_M/Q6_K amb SHA publicat a
`TheBloke/Mistral-7B-Instruct-v0.2-GGUF` i
`unsloth/Qwen3-4B-Instruct-2507-GGUF`; no és una validació d'inferència.

**Validació local (Python 3.12.3):** 2838 proves passen, incloses arquitectura,
regressions de ranking, exportació i captures TUI. Ruff, mypy (256 fitxers),
compileall, `uv lock --check --offline` i `git diff --check` passen. `uv build`
i la verificació dels artefactes nous de Jaull també passen; el verificador
executat sense aïllar-los seleccionava un paquet antic `local_ai_check` ja
present a `dist/`, que s'ha conservat. No s'ha executat la matriu Windows/Python
3.13 ni inferència real en aquest canvi.

**Límits pendents:** cerca acotada, no exhaustiva; dependència de metadades de
linatge declarades; suport CPU de quantització bitsandbytes no validat a Jaull;
cap nova predicció de memòria compartida o prova empírica d'offload en aquest
portàtil. Els registres experimentals històrics no es reescriuen.

**Fonts:** [configuració](../../src/jaull/estimator/configuration.py),
[expansió acotada](../../src/jaull/application/recommendation/alternatives.py),
[motor de plans](../../src/jaull/recommendation/engine_v2.py),
[recomanació](../recommendation.md).

### D098. Vulkan sobre Intel Arc a Windows i WSL2, sense donar res per validat

**Actualització 2026-10-26:** la primera prova física a la B60 (WSL2 amb Mesa Dozen) ha
refinat dues regles d'aquesta decisió: l'emparellament de noms (veure D100) i la memòria
compartida d'una iGPU al costat d'una targeta dedicada (veure D099). La resta del text
descriu l'abast original i no s'ha esborrat.

**Actualització 2026-10-20:** l'ampliació posterior del pilot HellaSwag permet
CPU/CUDA/Vulkan confirmat (schema 3); IFEval conserva el contracte CUDA auditat.
La identitat efectiva es contrasta amb les capacitats, el comandament i les
assignacions observades del servidor; inicialitzar un backend no prova que s'hagi
utilitzat. La validació física de la B60 continua pendent. L'abast original de D098
que es descriu a continuació no incloïa aquesta ampliació del pilot.

**Data i necessitat:** 2026-10-19. El PC de Biosfer té una Intel Arc Pro B60 (24 GB de
VRAM dedicada) amb Windows i WSL2, i tota la ruta validada fins ara és CUDA. Calia
preparar el suport Vulkan sense accés a aquell equip, de manera que cap resultat
simulat es pogués confondre amb evidència real.

**Auditoria prèvia:** la major part ja era independent del sistema operatiu (detecció
Vulkan, selecció de backend, localitzador de binaris amb `.exe`, regles de D097). Els
punts que assumien NVIDIA/CUDA eren: memòria de dispositius Vulkan no AMD, el `devices[0]`
del planificador, de la matriu de benchmarks, de la TUI i de `benchmark_plan_speed`, la
preparació (`READY` si el backend llistava *algun* dispositiu), l'entorn d'evidència
(només NVML) i la prova de quin dispositiu s'ha mesurat (només `CUDA<i>`). La condició
`device == "cuda"` de `execution_plans/service.py` resulta ser el flag `device_map` de
Transformers, no una restricció de llama.cpp: no es toca.

**Alternatives considerades:** (a) fer servir la memòria lliure que informa
`llama-cli --list-devices` com a disponibilitat de maquinari; (b) DXGI
`QueryVideoMemoryInfo` o `AdapterRAM` a Windows; (c) un pont WSL2→Windows; (d) afegir
SYCL; (e) generalitzar `_cuda_blocker` perquè la qualitat també corri a Vulkan. (a)
faria dependre el rànquing del binari instal·lat, cosa que el README i un test prohibeixen;
(b) o bé necessita crides natives que no es poden comprovar sense la màquina, o bé és un
camp de 32 bits que satura prop de 4 GiB; (c), (d) i (e) són fora de l'abast demanat.

**Decisió implementada:** cinc preguntes separades (maquinari present, backend exposat
pel sistema, backend suportat pel binari concret, dispositiu triat, pla executable).
Capacitat (mida de la heap local del dispositiu) i pressupost (`VK_EXT_memory_budget`,
una assignació del procés i no memòria lliure) es llegeixen de `vulkaninfo` només per
a un dispositiu discret que cap font millor cobreixi; la capacitat mai substitueix la
disponibilitat. Les targetes idèntiques no s'adivinen. Un nou emparellador
(`runtime/device_matching.py`) només accepta un dispositiu de llama.cpp si és
demostrablement l'accelerador seleccionat; tots els consumidors l'usen i, per a Vulkan,
la preparació passa a `NOT_READY`/`UNKNOWN` si no es pot demostrar. CUDA i HIP conserven
el comportament validat (un sol dispositiu s'accepta pel nom que sigui). El llistat
d'adaptadors de Windows només identifica maquinari: backend Vulkan `UNKNOWN`, cap xifra
de memòria. `classify_execution_environment` distingeix Windows, WSL1, WSL2 i Linux, i
en WSL2 sense cap dispositiu Vulkan útil s'explica sense culpar el host ni usar cap
runtime de Windows. L'entorn d'evidència Vulkan inclou dispositiu i versió de driver
(`ComputeBackendInfo.driver_version`, camp opcional) i la cohort de velocitat només
canvia per a registres Vulkan: els digests CUDA i CPU són idèntics.

**Emparellament estricte (decisió de l'autor, 2026-10-19):** fins a tenir sortides reals
de la B60, l'emparellament Vulkan és d'igualtat de nom després de normalitzacions
innòcues (marques `(R)`/`(TM)` i una anotació final entre parèntesis); no s'admet cap
coincidència parcial ni heurística tolerant (`RTX 2060` ≠ `RTX 2060 SUPER`). Sense
coincidència o amb diversos dispositius indistingibles no se'n tria cap: no es fixa
`--device` i la preparació és `UNKNOWN` o `NOT_READY`, mai `READY`. El motiu es mostra
sempre i diu què s'havia seleccionat i què ha llistat el binari, per exemple *«llama-cli
lists several vulkan devices (Vulkan0: X, Vulkan1: X) and none can be tied to 'X'; no
device is selected automatically»*; arriba a l'etiqueta del pla i als avisos del pla. Si
els noms de `vulkaninfo` i de llama.cpp difereixen a la B60, es registren tal com són i
es corregeix en un canvi propi, no afluixant la regla.

**Semàntica del pressupost de memòria Vulkan:** `heapBudget` és una estimació de l'SO o
del driver de quanta memòria *aquest procés* pot assignar a la heap (pot dependre de la
càrrega del sistema) i `heapUsage` és el que aquest procés ja hi té. No són VRAM lliure
global: el pressupost es guarda com a disponibilitat només amb la reserva i el marge
propis de Jaull a sobre, no se li resta l'ús del procés que només fa la consulta, mai es
crea una disponibilitat a partir de la mida o de l'ús sols, la font queda etiquetada
(`vulkan-memory-budget`) i `scan` mostra un avís que ho diu. Un pressupost nul o més gran
que la heap es descarta.

**Integració al flux normal (`uv run jaull`, 2026-10-19):** (1) l'avís de WSL2 («Running
inside WSL2: no usable Vulkan accelerator is exposed here… does not show that the Windows
host cannot accelerate the GPU») s'afegeix als avisos del perfil de maquinari, de manera
que surt a `scan`, a la pantalla de maquinari del flux guiat i a `doctor`, no només a
aquest últim; mai afirma que Windows tampoc pugui. (2) La pantalla de maquinari mostra,
per a un accelerador que no és NVML, la capacitat com a VRAM de capçalera, la fila
«Compute backend» (Vulkan, o «CPU only — … Vulkan not confirmed» si només Windows ha
llistat l'adaptador) i l'avís del pressupost; el camí NVIDIA no canvia. (3) `jaull doctor`
combina NVML, `vulkaninfo` i el llistat de Windows en un sol dispositiu i separa
*detectat* («Accelerators», amb «vulkan: not confirmed (adapter listed by Windows)») de
*executable* («Execution ready», que ara porta el motiu del dispositiu). (4) Un pla amb
`UNKNOWN` per `SELECTED_DEVICE_NOT_IDENTIFIED` es bloqueja igual que `NOT_READY` a Results,
Paths, Run, Validate i Benchmark; els altres `UNKNOWN` mantenen el comportament previ.
(5) **Cap acció es pot saltar la comprovació:** Run i Benchmark tornen a preparar el pla i
ara comproven la preparació *fresca* abans de llançar (abans només es mirava el pla amb què
s'havia obert la pantalla), el benchmark es nega a mesurar un dispositiu Vulkan no
identificat, i `benchmark_plan_speed` refusa un pla bloquejat. `jaull run` (CLI) continua
sense concepte de preparació, també per a CUDA: és una ordre explícita i queda fora
d'aquesta decisió.

**Coherència de l'offloading:** en una prova inicial es van veure 28/28 blocs de Hardware
Fit amb `--n-gpu-layers 23`; era un artefacte del fixture (una avaluació enregistrada per a
una RTX 2060 de 6 GB combinada amb maquinari Intel). Amb l'estimador i la política de
llançament reals sobre un perfil B60 simulat: si el model hi cap, Hardware Fit dona tots
els blocs i el llançament `-1` («tot»); amb offloading parcial són dues magnituds
diferents (blocs viables segons els pressupostos de bytes, i unitats de llama.cpp segons
l'heurística conservadora del llançament) que mai es converteixen l'una en l'altra; sense
cabuda no hi ha offloading; i amb capacitat coneguda però sense pressupost no es reclama
mai VRAM (`effective_device` = CPU). Cap d'aquests nombres s'ha mesurat en una B60. La
correspondència unitats↔blocs només existeix per a una compilació de llama.cpp en CUDA;
per a Vulkan la comparació física amb la predicció continua `NOT_COMPARABLE`, i el
refinament per tensors locals només s'aplica a una CUDA discreta. **Conseqüència de
producte:** si el driver de la B60 no imprimeix pressupost, Jaull no reclamarà la GPU i
recomanarà CPU fins que es validi físicament.

**Motiu i conseqüència:** un binari que llista un dispositiu no demostra que sigui el
bo, i una GPU Intel no implica Vulkan. La memòria observada continua sent la que
informa llama.cpp (`RuntimeReportedAllocation`), sense pic de VRAM per a Vulkan, i la
comparació física amb la predicció continua `NOT_COMPARABLE`. WSL2 no hereta Vulkan de
Windows i no hi ha cap pont; si la B60 només funcionés des de Windows, caldria una
decisió nova amb identitat de l'executable, traducció de rutes i gestió de processos.

**Proves i evidència:** mòduls de proves nous i extensions als existents (tots sense
GPU, xarxa ni TTY): `tests/test_vulkan_device_memory.py`, `test_windows_adapters.py`,
`test_execution_environment.py`, `test_runtime_device_matching.py`,
`test_vulkan_evidence_identity.py`, `test_vulkan_observation_and_paths.py` i, per a la
integració, `test_intel_execution_gates.py`, `test_tui_intel_hardware.py`,
`test_doctor_intel.py`, `test_intel_offload_coherence.py` i `test_tui_intel_flow.py` (que
recorre les pantalles reals amb l'estimador, el rànquing, el planificador i l'assessor
reals i només simula el Hub, les sondes de la màquina i els executables de llama.cpp,
amb scaffolding a `tests/_intel_fixtures.py` i `tests/_intel_flow.py`). Els fixtures
són sintètics: no hi ha cap captura real de la B60. **Estat: `IMPLEMENTED`; `VALIDATED`
cap; `UNVERIFIED` el format de `vulkaninfo` amb pressupost, els noms de dispositiu, les
línies `ggml_vulkan:`, la disponibilitat de `vulkaninfo.exe` a Windows i l'exposició de
la GPU a WSL2; `BLOCKED_BY_ENVIRONMENT` totes les proves físiques.**

**Validació local (Python 3.12.3):** 2938 proves passen; Ruff, mypy (259 fitxers),
compileall i `git diff --check` passen. No s'ha executat cap prova en una GPU Intel, ni a
Windows nadiu ni a WSL2, i no s'ha executat la matriu Windows/Python 3.13.

**Límits pendents:** protocol físic a
[biosfer-vulkan-validation-protocol.md](../biosfer-vulkan-validation-protocol.md);
si `vulkaninfo` i
llama.cpp anomenen la mateixa targeta diferent, Jaull no fixa `--device` i ho diu
(`UNKNOWN`/`NOT_READY` amb motiu), sense error; no hi ha font de memòria per a Intel a
Linux (`xe`) ni per DXGI. No s'implementen el pont WSL2→Windows, SYCL ni la qualitat sobre
Vulkan. **Tancament:** amb això s'acaba el que es pot fer sense maquinari; no s'amplia cap
funcionalitat més sobre fixtures sintètics. El pas següent és capturar les sortides reals
(`vulkaninfo`, `vulkaninfo --summary`, `llama-cli --list-devices`, consulta d'adaptadors)
a Windows i a WSL2 segons el protocol i corregir només el que mostrin. Pensat per a un
commit independent de D097.

**Fonts:** [disseny i estat](../vulkan-windows-wsl2.md),
[emparellador](../../src/jaull/runtime/device_matching.py),
[memòria Vulkan](../../src/jaull/hardware/vulkan.py),
[adaptadors de Windows](../../src/jaull/hardware/windows_adapters.py),
[entorn d'execució](../../src/jaull/domain/execution_environment.py), D097.

### D099. Una iGPU de memòria compartida no converteix la màquina en un únic pool

**Data i necessitat:** 2026-10-26, primera prova física a la B60 (WSL2, Mesa Dozen; l'equip
té també una iGPU Radeon del Ryzen 7 5700G). La recomanació sortia com a
`llama.cpp · CPU · Tight` amb 11,90 GiB i cap model gran, tot i la targeta de 23,8 GiB.

**Causa (amb evidència):** dues funcions independents deien «si *algun* accelerador
comparteix memòria, tota la màquina és de memòria unificada»:
`estimator/hardware_fit.py::_memory_topology` (l'anàlisi d'ajust) i
`discovery/candidate_filter.py::_has_unified_memory` (la preselecció). Amb la iGPU
present, tot es jutjava contra els 13,6 GiB de RAM i la B60 s'ignorava. La regla és
d'agost (commits `8fea9e9` i `d904e6e`); no s'havia vist perquè fins ara cap equip havia
combinat un dispositiu dedicat i una iGPU amb la detecció d'acceleradors genèrica.
També afectaria un portàtil Ryzen amb una NVIDIA dedicada. **No era l'elecció del backend
d'execució**, que ja triava bé la B60.

**Decisió implementada:** la màquina només és un únic pool de memòria si no hi ha cap
dispositiu dedicat amb memòria coneguda (pressupost observat a l'analitzador, capacitat
a la preselecció, com ja feien `available_accelerator_memory_bytes` i
`planning_accelerator_memory_bytes`). Si n'hi ha, es planifica contra la targeta dedicada
i la iGPU compartida no hi compta. Una màquina només amb la iGPU continua sent unificada.

**Proves:** un test per a l'analitzador (`test_hardware_fit_analyzer.py`) i un per a la
preselecció (`test_accelerator_memory.py`), tots dos amb el cas «dedicada + iGPU», que van
fallar abans del canvi. Cerca real amb un perfil equivalent: el primer model passa de
`cpu_ram`/CPU/tight a `gpu_resident`/Vulkan.

**Estat:** `IMPLEMENTED`; observat a la B60 amb la llista de recomanacions. `UNVERIFIED`
que l'execució a la targeta sigui correcta per a aquests models (pendent de Run/Validate
sobre la B60).

### D100. Els noms de dispositiu de Dozen s'emboliquen en un parèntesi

**Data i necessitat:** 2026-10-26, mateixa prova. Amb dues GPU, `jaull doctor` donava
`Execution ready: unknown` i deia que cap dispositiu de llama.cpp es podia lligar a la B60,
tot i que els dos noms eren literalment idèntics.

**Causa (amb evidència):** la regla de D098 («una anotació final entre parèntesis es
descarta») estava pensada per a `… Graphics (BMG G21)`, però Mesa Dozen emboliqua el nom
real dins del parèntesi: `Microsoft Direct3D12 (Intel(R) Arc(TM) Pro B60 Graphics)` i
`Microsoft Direct3D12 (AMD Radeon(TM) Graphics)`. En treure l'últim parèntesi tots dos es
reduïen a `Microsoft Direct3D12`, es consideraven el mateix dispositiu i la preparació
era ambigua. És exactament el cas que D098 preveia: l'emparellament estricte es mantenia
fins a tenir sortides reals.

**Decisió implementada:** el nom complet ha de coincidir (sense majúscules, puntuació ni
`(R)`/`(TM)`), i el parèntesi final només es deixa de banda quan **un sol** dels dos noms
el porta. Si tots dos en porten, forma part de la identitat. Més estricte, no més
tolerant; sense noms, fabricants ni drivers fixats al codi (`names_agree`).

**Proves:** `test_runtime_device_matching.py` amb els dos noms reals; abans fallava amb el
mateix «ambiguous». Amb el `--list-devices` real, passat pel parser de Jaull: `Vulkan0`,
`matched_by_name`, `ready`; a la B60, `doctor` passa a `Execution ready: yes`.

**Estat:** `VALIDATED` a la B60 per a la detecció i l'emparellament; `UNVERIFIED` el
mateix emparellament en Windows natiu, on els noms no porten embolcall.

### D101. Tres defectes de la cadena de recomanació, trobats amb la cerca real a la B60

**Data i necessitat:** 2026-10-26. Un cop la B60 comptava, la llista saltava de 7B a 32B,
posava un 1,5B per sobre d'un 7B i, a Balanced, recomanava `Q2_K`. Es va guardar l'estat
de cada etapa d'una sola cerca real i es va analitzar etapa per etapa, sense canviar res
fins a tenir l'evidència.

**1. Preselecció.** Només 12 candidats passen a la inspecció. Dins de cada grup es
preferia un candidat d'una família no vista, i la família és el primer mot del nom (tots
els `Qwen2.5-…` són `qwen2`): amb `qwen2` ja triat, els Qwen oficials (puntuació ~22) es
saltaven per `secmlr/SWE-BENCH-…-qwen_code_14b` (39 descàrregues, ~7). Els 14B oficials
no arribaven mai a la inspecció. **Decisió:** dins del grup mana la puntuació; la família
només desempata entre puntuacions idèntiques (amb una tolerància de coma flotant, no de
puntuació); els candidats de mida desconeguda en tenen una per ronda i la resta es guanya
per puntuació al farciment. No s'ha introduït cap tolerància de puntuació: caldria
justificar-la amb dades.

**2. Pes dels GGUF doblat.** Els repos de Qwen publiquen cada quantització sencera i també
en parts; se sumaven tots els fitxers (Q8_0 de 7B: 15,08 GiB en lloc de 7,54), el 14B Q8_0
s'estimava en uns 29 GiB quan en cap 14,6, i es baixaven els dos conjunts. **Decisió:** si
un mateix conjunt de pesos és publicat sencer i en parts, es compta el fitxer sencer
(`file_count=1`); les parts que són tot el que hi ha d'aquell nom es mantenen. S'invalida
la memòria cau d'anàlisis (esquema 3 → 4), que desava els totals doblats.

**3. Desempat de Balanced.** Amb els vuit plans del 7B idèntics a tots els eixos, els
desempatava el nom de la quantització en ordre alfabètic (`Q2_K` primer; al 32B, `IQ2_M`).
El projecte ja tenia la política explícita (`QUANTIZATION_LADDERS`: Balanced
`Q5_K_M, Q4_K_M, Q4_K_S, Q3_K_M`), que s'usa per generar els plans, però es perdia en
reordenar-los. **Decisió:** el desempat dins d'un mateix model usa la posició a l'escala de
la prioritat. Tres tests existents afirmaven `Q4_K_M` per a Balanced (un amb el comentari
«la política d'empat de Balanced va triar Q4_K_M», mentre el selector triava `Q5_K_M`):
eren l'efecte de l'ordre alfabètic documentat com a política, i s'han actualitzat a
`Q5_K_M`. Si es prefereix `Q4_K_M`, el canvi correcte és l'escala, no l'ordre alfabètic.

**4. Preselecció adaptada al maquinari.** El pas 1 no n'hi havia prou: les places es
repartien per classe de mida sense mirar què cap a la màquina, de manera que a una B60 de
24 GiB models minúsculs ocupaven places de la inspecció. L'indici de col·locació
(`coarse_placement_hint`) també menteix si ignora que AWQ/GPTQ necessiten CUDA o que un
repositori no GGUF no té camí de Vulkan; ara ho tenen en compte. La preselecció fa dues
passades: cobertura (la millor classe de col·locació i un candidat de mida desconeguda) i
després farciment per rellevància, ordenat per (mode de col·locació, puntuació) amb el
mateix grup d'empat. Es va trobar que aplicar només la meitat de la cobertura empitjorava
Balanced (es perdien llavors GGUF que després l'expansió havia de recuperar), per això
els dos canvis van junts.

**Revisió de l'auditoria:** la capacitat de VRAM de la preselecció es calcula només
entre dispositius dedicats que exposen un backend compatible amb l'artifact. No es
pot combinar «CUDA disponible» d'una targeta petita amb els 24 GiB d'una B60 que
només exposa Vulkan. GGUF considera CUDA/HIP/Vulkan; Transformers, CUDA/HIP, i
AWQ/GPTQ només CUDA segons el loader actual. Les disponibilitats `unknown` i
`unavailable` no habiliten el pressupost; els perfils antics només NVML conserven
el fallback CUDA. Es tria una capacitat, mai la suma de targetes. Això continua sent
un indici de preselecció, no un pressupost observat per executar ara.

**5. Sense farciment artificial a Balanced.** Amb ús comercial demanat, la confirmació de
llicència és la primera clau, de manera que un 135M Apache passava per davant d'un Llama
3.1 8B de llicència `unknown` (política vigent, no canviada) i omplia places que cap model
capaç ocupava. Provar de fer-ho al rànquing (un terra de capacitat a la clau) trencava la
política «executable abans que capaç» i la de llicència primer, i es va revertir.
**Decisió:** a la selecció final de la llista (`weak_plans_last`), només per a Balanced, un
pla de capacitat `WEAK` ocupa una plaça només després dels que no ho són; l'ordre dins de
cada grup no es toca i els febles no s'eliminen si queden places. Speed, Memory i Quality
no canvien.

**Revisió de l'auditoria:** la partició global podia avançar un 7B amb ajust `unknown`
davant d'un 135M `comfortable`. Ara la preferència opera només dins de grups contigus
amb igual estat de rebuig, confirmació d'ajust i classe de viabilitat d'execució.
`UNKNOWN` de capacitat no es considera millor que `WEAK`. La selecció precedeix les
mesures i no en desfà l'ordre. `apply_active_policy` comparteix la seqüència entre
recomanació i exportació, i Recompare usa la mateixa selecció de base. JSON/Markdown
separen `selection_policy` de la política mesurada i conserven les posicions abans
i després de cada etapa quan es poden reconstruir des del pool desat.

**Validació d'aquest refinament:** tests offline amb perfils mixtos CUDA/HIP/Vulkan,
viabilitat confirmada/desconeguda/offload, exportació immutable i evidència mesurada
que preval sobre la preferència de metadades. No és una validació física addicional
de la B60 ni una nova campanya de rendiment o qualitat.

**6. Selector de configuració de Balanced.** El selector GGUF s'aturava al primer graó
de l'escala amb ajust acceptable, i `TIGHT` ho era; així un `Q5_K_M` just guanyava a un
`Q4_K_M` còmode. **Decisió:** només per a Balanced, un graó `TIGHT` es recorda però no
s'agafa mentre n'hi hagi un de posterior, no agressiu, amb ajust `COMFORTABLE` o
`COMPATIBLE`; si no n'hi ha cap, es manté el `TIGHT` amb un avís. Quality conserva la
millor quantització que càpiga, fins i tot ajustada; Speed i Memory no canvien (Speed no
suposa que Q4 sigui més ràpid sense mesures). Els graons agressius (`AGGRESSIVE_QUANTIZATIONS`)
no es consideren «raonables» per fugir d'un ajust just. Només s'ha canviat el camí GGUF;
el de Transformers segueix igual. El marge es mesura amb l'estat de l'estimador, no
amb una comprovació pròpia sobre la memòria disponible estimada: això queda pendent.
Tests: quatre casos parametritzats a `tests/test_recommendation_configuration.py`.

**7. Els models petits només omplen el que els grans deixen.** Amb una B60 de 24 GiB,
cinc models de menys de 4B molt populars ocupaven places d'inspecció que cap model de
7B-32B de la mateixa classe de col·locació tenia. **Decisió:** dins de la millor classe de
col·locació, els models dels cubs `tiny` i `small` (els que ja defineix `_SCALE_BUCKETS`)
només entren a la cobertura i al farciment quan no en queda cap de més gran; en una
màquina on només caben models petits, la classe millor només en conté i entren igualment.
No s'han afegit constants. Un test que abans afirmava que un 14B amb 39 descàrregues no
desplaça els oficials s'ha ajustat a 2 places: ara un 14B poc conegut pot passar davant
d'un 1,5B molt popular a la tercera plaça, que és la política decidida.

**8. Consulta de cerca adaptada a la mida.** Totes les consultes ordenen per descàrregues,
de manera que els models petits populars omplen cada pàgina i cap 14B de xat arribava als
40 candidats (verificat amb la cerca real: l'únic 14B era un Coder de Transformers).
**Decisió:** si caben mides per sobre de `small`, s'afegeix una consulta més amb el filtre
`num_parameters` del Hub, entre el límit superior de `small` i la mida màxima que cap
resident en aquesta màquina (`largest_fitting_billions`, amb les mateixes estimacions
gruixudes de la preselecció). Si no hi caben, no s'afegeix res. Canvi visible a l'informe:
l'etiqueta `<ús>:scale` a `search_queries` (snapshot `report.json` regenerat, una línia).
La mida és només un filtre de cerca; la identitat del model no es dedueix del nom.

**9. `weak_plans_last` ja no depèn de trams contigus.** La primera versió només reordenava
dins de trams contigus de la mateixa viabilitat; un pla amb llicència confirmada que
necessita offload (runnability 1) tallava el tram i el SmolLM2-135M conservava la plaça
(reproduït amb l'estat real). **Decisió:** només es mou el pla sense capacitat coneguda i
només cap enrere, darrere de cada pla capaç que sigui almenys tan viable com ell; els
capaços mantenen l'ordre entre ells (la llicència primer i «executable abans que capaç» hi
segueixen manant). Conseqüència acceptada: un pla feble que s'executa pot quedar darrere
d'un capaç que necessita offload, perquè és l'únic que el feia sortir de la llista.

**Resultat observat (cerca real, mateix perfil):** de `32B Q2_K, 1,5B, deepseek 7B, 3B,
gpt2` a `7B Q8_0, 14B Q8_0 (19,6 GiB, GPU resident), 32B Q6_K, 1,5B, 3B`; a Balanced, de
`Q2_K` a `Q5_K_M`.

**Límits pendents, sense decidir:** (a) resolt a Balanced pel punt 6; el marge sobre la memòria disponible estimada
encara no es comprova per separat; (b) resolt pels punts 4 i 7; (c) amb ús comercial demanat, la confirmació de llicència va abans de la
capacitat (política vigent, no s'ha canviat; el punt 5 només evita que els febles
omplin la llista); (d) `QUALITY` no distingeix la mida dins del
mateix nivell de capacitat.

**Estat:** `IMPLEMENTED` amb tests que van fallar abans; cerca real repetida amb el canvi.
`UNVERIFIED` fora d'aquest perfil sintètic (B60 + iGPU AMD + 13,6 GiB de RAM) i de la
cerca en directe d'aquell dia, que pot variar.

**Fonts:** [preselecció](../../src/jaull/discovery/candidate_filter.py),
[classificador GGUF](../../src/jaull/huggingface/classifiers.py),
[rànquing v2](../../src/jaull/recommendation/engine_v2.py),
[analitzador d'ajust](../../src/jaull/estimator/hardware_fit.py),
[emparellador](../../src/jaull/runtime/device_matching.py), D097, D098.

### D102. KV cache d'un MoE amb la fórmula d'atenció i confiança mitjana

**Data i necessitat:** 2026-10-26. A la cerca real de la B60, tres models de 12B-30B
(un 27B multimodal, un MoE de 12B amb 2,5B actius i un MoE de 30B amb 3B actius) es van
avaluar i es van rebutjar amb «memory estimate is incomplete». Eren justament els de la
franja que l'usuari esperava veure. Es va mirar l'estimació de cada un abans de canviar res.

**Causes (dues, no una):** (a) els dos MoE queden amb la KV `unknown` perquè
`_detect_exotic_architecture` tractava qualsevol `num_experts` com a arquitectura no
modelada (D023); (b) el 27B és un compost multimodal (`text_config`) i, a més, la lectura
de la capçalera GGUF s'ha rendit pel límit de 8 MiB. **El punt (b) no s'ha tocat.**

**Decisió:** D023 es manté per a MLA, compostos multimodals i `auto_map`, però ja no per a
MoE en la KV cache. Els experts són a les capes FFN i no canvien la KV, que depèn només de
capes, caps KV i dimensió de cap; la capçalera GGUF aporta aquestes tres xifres igual que a
un model dens. Un MoE rep la fórmula de sempre amb confiança `MEDIUM` i una nota explícita
que diu que no s'ha validat contra cap execució MoE (la validació del projecte és amb
Qwen2.5 dens). Un MoE amb MLA (`kv_lora_rank`) continua desconegut. Els pesos d'un MoE
continuen amb el repartiment uniforme i la col·locació d'experts no es modela.

**Límits:** una arquitectura híbrida (atenció lineal o finestra alternant) sobreestimaria la
KV, que és el sentit segur; no s'ha mesurat cap cas MoE. La velocitat d'un MoE amb offload
no es prediu.

**Estat:** `IMPLEMENTED` amb tests (la KV d'un MoE iguala la d'un dens i baixa la
confiança; un MoE amb MLA continua desconegut). `UNVERIFIED` contra una execució real d'un
MoE a la B60.

**Fonts:** [KV cache](../../src/jaull/estimator/kv_cache.py),
[tests](../../tests/test_kv_cache_estimator.py), D023, D101.

### D103. Desempat per descàrregues en lloc del nom del repositori

**Data i necessitat:** 2026-10-26. A la B60, Llama 2 7B (62 k descàrregues) quedava per
sobre de Llama 3.1 8B (511 k) i no hi havia cap eix que els separés: empataven en
idoneïtat, runnability, capacitat, llicència (`unknown` als dos), confiança i escala de
quantització. Els decidia el nom del repositori en ordre alfabètic, i `TheBloke` va abans
que `bartowski` perquè les majúscules s'ordenen primer. No és cap senyal de qualitat.

**Decisió:** quan tots els eixos empaten, passa davant el repositori amb més descàrregues;
el nom només desempata dues quantitats iguals. La clau queda després de l'escala de
quantització (`_ladder_rank`), de manera que la popularitat d'un repositori mai canvia
quina quantització es prefereix. Val per a les quatre prioritats. `ranking_criteria`
publica un criteri nou, `popularity`, amb el mateix rang que la clau, i els snapshots de
l'informe s'han regenerat (una línia cadascun). No s'ha pujat `REPORT_SCHEMA_VERSION`:
només s'afegeix un element a una llista de criteris. El baseline de política
(`recommendation-policy-baseline.json`) i l'ordre daurat de Balanced canvien només en els
empats entre models petits (Tiny, Mini, Small passen a l'ordre de descàrregues); Medium
continua segon per capacitat. Speed i Memory no canvien. La nota publicada de la política
de Balanced passa a `balanced-capability-within-viability-v2` (D101, punt 9).

**Límits:** les descàrregues afavoreixen els models antics i majoritaris; aquí només
resolen empats, mai passen per davant d'un eix. No mesuren qualitat. **No canvia la
llista final de l'exemple,** perquè el diversificador (`diversify_ranked_plans`) tria,
entre plans amb la mateixa signatura d'avaluació, el de família no vista abans que el
millor classificat; així, un MoE de codi de 140 k descàrregues del lloc 29 del rànquing
base arriba al segon lloc d'un xat general. Això és un efecte del diversificador, no del
desempat, i queda obert.

**Estat:** `IMPLEMENTED` amb un test (quatre prioritats) que fallava abans. Cerca real:
Llama 3.1 passa del lloc 52 al 60 i Llama 2 del 51 al 61, o sigui que 3.1 ja precedeix 2.

**Fonts:** [rànquing v2](../../src/jaull/recommendation/engine_v2.py),
[diversitat](../../src/jaull/recommendation/diversity.py), D101, D102.

### D104. La diversitat només desempata entre plans que el rànquing no distingeix

**Data i necessitat:** 2026-10-26. Un MoE de codi de 12B (lloc 29 del rànquing base) arribava
al segon lloc d'un xat general perquè `diversify_ranked_plans` triava, entre plans amb la
mateixa signatura d'avaluació, el de família no vista abans que el millor classificat. És el
mateix defecte que la preselecció (D101, punt 1: família nova per sobre de millor
puntuació), en un altre lloc. L'usuari ho va formular així: pot haver-hi tres Qwen seguits si
són els millors, però no té sentit que un Qwen petit passi davant d'un model d'una altra
família amb més paràmetres, i el criteri val per a totes les cerques.

**Decisió:** el millor classificat va sempre primer; la varietat només tria entre plans que
el rànquing no pot distingir. «Empat» es defineix amb la clau del propi rànquing
(`ranking_tie_key`: tot el que compara abans del desempat final per nom), de manera que les
descàrregues i l'escala de quantització també separen plans. Cada cridador (`recommend`, la
instantània `shadow` i l'informe) passa `tie_key_for(requirements)`; sense clau, el
diversificador usa la signatura d'avaluació més les descàrregues. `_ranking_key` es
reescriu com a `ranking_tie_key` més el nom, la quantització i el runtime.

**Límits:** «més paràmetres primer» no és una regla de diversitat i no s'ha tocat aquí. A
Quality el rànquing ordena la quantització abans que la mida dins del mateix nivell de
capacitat, de manera que un 7B `Q8_0` passa per davant d'un 14B `Q6_K`; és una política
pendent (D101, límit d), i no té sentit a Speed i Memory, on el model petit és l'objectiu.

**Estat:** `IMPLEMENTED` amb un test que fallava abans. Cerca real: Mellum2 passa del
segon al quart lloc de General chat.

**Fonts:** [diversitat](../../src/jaull/recommendation/diversity.py),
[rànquing v2](../../src/jaull/recommendation/engine_v2.py), D101, D103.

### D105. Una quantització agressiva no passa per davant d'una que càpiga

**Data i necessitat:** 2026-10-26. A la cerca real de la B60, un 30B a 1 bit (`IQ1_M`) i un
32B a 2 bits (`IQ2_M`) ocupaven places de Balanced. A la clau de Balanced l'eix de marge de
memòria (`_comfortable_memory_rank`) va abans que l'escala de quantització, de manera que un
`IQ1_M` de 12 GiB «còmode» guanyava a un `Q4_K_M` de 19 GiB només «compatible». La política
escrita (`AGGRESSIVE_QUANTIZATIONS`: «només val la pena oferir-ho quan res més càpiga») no
es complia al rànquing, i la llista era incompleta.

**Decisió:** (1) la llista d'agressives arriba fins a `Q3_K_S` en bits per pes (uns 3,4),
criteri que ja seguia, i inclou la família `IQ1`/`IQ2`, `IQ3_XXS`, `IQ3_XS`, `Q1_0` i les
variants `Q2_K_L`/`Q2_K_XL`; (2) a Balanced hi ha un eix nou, `aggressive_quantization`,
després de la idoneïtat i abans de la runnability, que posa una quantització agressiva
darrere d'una que càpiga (un pla amb offload, que cap, passa abans que un d'1 bit resident);
(3) el selector prova les quantitzacions fora de l'escala de menys a més agressiva en lloc
d'alfabèticament, que posava `IQ1_M` primera. **Quality** ja s'hi protegia amb
`_quantization_quality_rank`; **Speed i Memory no canvien** perquè les seves escales
inclouen a propòsit `Q3_K_S` i `Q2_K` (Memory busca el més petit). `ranking_criteria` publica
el criteri nou a Balanced, que arriba als snapshots de l'informe.

**Límits:** la llista és per etiqueta de quantització, no per mesura; una etiqueta nova no
llistada no es detecta. No es toca la mida del model: un 30B a `Q4_K_M` ajustat (`tight`)
encara pot entrar si és l'únic que càpiga.

**Estat:** `IMPLEMENTED` amb tests (rànquing de Quality i Balanced; ordre del selector) que
fallaven abans. Cerca real: General chat passa del 32B `IQ2_M` al `Q3_K_M`, i Coding del 30B
`IQ1_M` al `Q4_K_M`.

**Fonts:** [polítiques](../../src/jaull/estimator/policies.py),
[selector](../../src/jaull/estimator/configuration.py),
[rànquing v2](../../src/jaull/recommendation/engine_v2.py), D101, D104.

### D106. Els plans sense capacitat coneguda també van al final a Quality

**Data i necessitat:** 2026-10-27. La prova a la B60 va mostrar el SmolLM2-135M al quart lloc
de Quality sobre xat general, per davant d'un Coder 7B. `weak_plans_last` (D101, punt 9) només
s'aplicava a Balanced. L'usuari va demanar que el criteri valgui per a totes les cerques.

**Decisió:** la regla s'aplica a Balanced i a Quality (`CAPABILITY_FIRST_PRIORITIES`), les
dues prioritats que busquen un model capaç. **Speed i Memory no l'apliquen:** un model petit
és el seu objectiu. L'informe publica `selection_policy` per a les dues prioritats, amb
versió `<prioritat>-capability-within-viability-v2`, i la nota ja no parla de «grups de
viabilitat contigus». El snapshot `report.json` canvia una línia (la nota).

**Límits:** això no ordena la mida dins d'un mateix nivell de capacitat; un 7B `Q8_0`
continua per davant d'un 14B `Q6_K` a Quality (D101, límit d). A Quality també poden entrar
plans amb offload (un 32B `Q6_K` de 32,5 GiB), que són lents; el rànquing els ordena pel que
cap, no per la velocitat.

**Estat:** `IMPLEMENTED` amb un test que fallava abans. `VALIDATED` parcialment: la prova
real a la B60 va mostrar el defecte; la correcció s'ha comprovat amb la cerca real a WSL i
queda pendent de repetir-la a la B60.

**Fonts:** [selecció](../../src/jaull/recommendation/selection.py),
[informe](../../src/jaull/reporting/recommendation.py), D101, D104, D105.

### D107. Quatre barreres que deixaven fora el Qwen3.8-27B de la B60

**Data i necessitat:** 2026-10-27. L'usuari va assenyalar que a una B60 de 24 GiB hi cap un
model de 27B i que Quality saltava del 30B al 14B. Es va comprovar a Hugging Face: no hi ha un
Qwen2.5 de 27B oficial (només fine-tunes i LoRA de tercers), però el **Qwen3.8-27B** existeix i
el seu GGUF d'`unsloth` té 5,9 M de descàrregues. Provant-lo sol amb el servei real, Jaull
donava **zero recomanacions**. Hi havia quatre causes independents, resoltes en aquest ordre
perquè desbloquejar l'estimació abans de netejar els fitxers hauria recomanat el projector de
visió com si fos el model.

**1. Fitxers auxiliars comptats com a quantitzacions.** El repositori publica
`mmproj-F16.gguf` (0,86 GiB), `mmproj-BF16.gguf`, `imatrix_unsloth.gguf` i un `MTP/mtp-…gguf`
al costat dels pesos; un altre repositori publica un model esborrany de decodificació
especulativa (`…-draft-Q4_0.gguf`, 1,56 GiB) que es va arribar a recomanar com a «27B a
`Q4_0`, 3,3 GiB». **Decisió:** el classificador deixa fora els `.gguf` amb els mots `mmproj`,
`imatrix`, `mtp` o `draft` al nom, o dins una carpeta `MTP/`. Són convencions de noms de fitxer
de l'ecosistema llama.cpp, usades només per excloure fitxers de la llista, mai per identificar
un model. La memòria cau d'anàlisis passa a l'esquema 7.

**2. Capçalera GGUF més gran que 8 MiB.** El vocabulari de 248 k tokens del Qwen3.5 supera el
pressupost de descàrrega i la KV quedava desconeguda. Totes les claus que necessita
l'estimació (arquitectura, `block_count`, `embedding_length`, `attention.head_count`) van
**abans** del tokenitzador. **Decisió:** si la lectura s'acaba a mitja d'un valor però aquestes
claus ja hi són, s'accepta la capçalera parcial; si no, es continua demanant bytes com abans.
Això també estalvia rondes de descàrrega a tots els models.

**3. Configuració «composta» multimodal.** La configuració base del repositori imbrica
`text_config` i `vision_config`, i `_detect_exotic_architecture` marcava el GGUF com a
compost (D023) encara que la capçalera ja dona les dimensions. **Decisió:** quan la capçalera
GGUF aporta capes, caps i mida d'embedding, el fusionador de configuracions descarta
`text_config`, `vision_config` i `audio_config`: un GGUF només conté el decoder de text. Sense
capçalera es conserva l'avís.

**4. Descobriment.** Cap consulta podia retornar-lo: el servidor filtra per
`pipeline_tag=text-generation` (el repositori no en declara) i la cerca és per subcadena del
nom (no conté «instruct» ni «coder»). **Decisió:** es duu una segona consulta de mida
(`<ús>:scale-open`) sense frase ni `pipeline_tag`, només amb l'etiqueta del format i la
finestra de paràmetres. El filtre del client continua descartant el que és multimodal o no de
text. L'etiqueta `scale-open` arriba al snapshot de l'informe.

**Resultat observat (cerca real, mateix perfil):** `unsloth/Qwen3.8-27B-GGUF` `Q4_K_M`,
19,9 GiB, `compatible`, entra a General chat de Balanced. Fins llavors, zero recomanacions.

**Límits pendents, sense decidir:** (a) Balanced posa qualsevol pla `comfortable` per davant
d'un `compatible` (el 27B ocupa el 83 % de la memòria, per sobre del 75 %); l'usuari ha dit
que Balanced ha de preferir els dos abans que `tight`; (b) a igualtat d'eixos, decideixen les
descàrregues: dos models Ornith de 9B (5,1 M i 2,4 M de descàrregues, de procedència no
verificada) passen per davant d'un 14B de Qwen; el que falta és la mida abans de les
descàrregues; (c) el repositori de `JonathanColetti` publica cada quantització dues vegades
(`…-noMTP-…`) i queda descartat per ambigu.

**Estat:** `IMPLEMENTED` amb tests que fallaven abans (classificador, capçalera parcial,
fusionador, consulta de mida). `UNVERIFIED` a la B60: cap execució real d'un 27B.

**Fonts:** [classificador](../../src/jaull/huggingface/classifiers.py),
[lector GGUF](../../src/jaull/metadata/gguf_reader.py),
[fusionador](../../src/jaull/metadata/config_merger.py),
[consultes](../../src/jaull/discovery/query_builder.py), D023, D101, D102.

### D108. A Quality, un pla que càpiga a la GPU abans que una quantització millor amb offload

**Data i necessitat:** 2026-10-27. A Quality un 32B es mostrava a `Q6_K` amb 32,5 GiB, uns 8,5 GiB
volcats a una RAM de 13,6 GiB, perquè la clau ordenava la qualitat de la quantització
(`_quantization_quality_rank`) abans de l'ajust a la GPU, mentre que un `Q3_K_M` residente de
20,2 GiB cabia. L'usuari ho va formular així: val més un `Q4` sense offload que un `Q6` amb
offload, que és molt més lent.

**Decisió:** la clau de Quality passa a (idoneïtat, capacitat, **runnability**, qualitat de la
quantització, ajust). La runnability agrupa «normalment executable» (ajust fort o adequat) i
deixa l'offload en una classe degradada, igual que a Balanced. `ranking_criteria` publica
l'eix nou a Quality i el test de fidelitat de l'informe s'actualitza.

**Límits:** no ordena la mida del model dins d'un mateix nivell de capacitat; un 7B `Q8_0`
continua per davant d'un 14B `Q6_K` (D101, límit d), i un 32B `Q3_K_M` residente queda per
darrere d'un 14B `Q8_0`. Si això és millor per programar queda `UNVERIFIED`; es pot mesurar
amb el pipeline d'avaluació de qualitat de Jaull.

**Estat:** `IMPLEMENTED` amb un test que fallava abans.

**Fonts:** [rànquing v2](../../src/jaull/recommendation/engine_v2.py), D101, D105, D106.

### D109. A Balanced, `comfortable` i `compatible` tenen el mateix marge; `tight` va darrere

**Data i necessitat:** 2026-10-27. Un 27B a `Q4_K_M` ocupava el 83 % de la B60 (`compatible`) i
quedava al lloc 24 darrere de tots els plans per sota del 75 % (`comfortable`), encara que
tots dos càpiguen amb marge. L'usuari va indicar (i ja ho feia el selector, D098) que Balanced
ha de preferir `comfortable` o `compatible` abans que `tight`.

**Causes (dues):** (a) `_comfortable_memory_rank` posava `compatible` darrere de `comfortable`;
(b) l'eix `execution_fitness` de Balanced, que ordena `strong` abans que `adequate`, repetia la
mateixa distinció després de la runnability. A més, la valoració posa `compatible` i `tight` al
mateix nivell (`ADEQUATE`), de manera que Balanced no els havia distingit mai.
**Decisió:** (1) a Balanced `comfortable` i `compatible` valen 0 i `tight` val 1, mirant l'estat
de memòria del pla; (2) s'elimina l'eix `execution_fitness` de la clau i del criteri publicat de
Balanced (la runnability ja és aquest nivell amb `strong` i `adequate` agrupats); (3) el criteri
`memory_headroom` publica l'estat real (`comfortable`, `compatible`, `tight`) en lloc del nivell,
perquè `tight` i `compatible` no semblin un ordre trencat. Quality conserva `execution_fitness`.

**Resultat:** a la cerca real, l'efecte és petit perquè, amb tots els eixos empatats,
decideixen les descàrregues (límit b de D107). El 30B-A3B de Coding passa del `Q4_K_M` `tight` al
`Q3_K_M` `compatible`, que és el que demana la política.

**Estat:** `IMPLEMENTED` amb un test que fallava abans; snapshots de l'informe regenerats.

**Fonts:** [rànquing v2](../../src/jaull/recommendation/engine_v2.py), D098, D105, D107.

### D110. Cerca més ampla i més ràpida: capçaleres en paral·lel, agrupació per model base i mida

**Data i necessitat:** 2026-10-27. L'usuari va preguntar si la cerca era massa estreta i si es
podia accelerar. Es va mesurar abans de canviar res. **Velocitat:** una cerca en fred (sense cap
memòria cau) tardava 64 s; la cerca a Hugging Face són 1,4 s, la inspecció 2 s i la resta, uns
60 s, és l'estimació: per a cada quantització de cada repositori GGUF es llegeix la capçalera
(una petició de rang de 256 KiB o més), en sèrie, durant el rànquing. **Amplada:** a la B60 hi
caben fins a 32B i dels 100 repositoris GGUF més descarregats entre 20B i 32B hi ha uns 46 noms
diferents (Gemma 4 26B-A4B, gpt-oss-20b, Qwen3.6-27B…), però cada consulta torna només 20
resultats per descàrregues i uns 12 eren recàrregues del mateix Qwen3.8-27B.

**Decisions:**
1. **Capçaleres en paral·lel** (`prefetch_gguf_headers`, 8 fils, `MAX_CONCURRENT_HEADER_READS`)
   abans del rànquing, a la mateixa memòria cau de capçaleres que ja consulta l'estimador. No
   canvia cap càlcul: el top 5 de la mateixa cerca va ser idèntic. **Mesurat: 61,4 s → 29,7 s**;
   el rànquing passa de 44,9 s a 0,1 s i la precàrrega (147 capçaleres) en triga 15.
2. **Agrupació per model base abans d'inspeccionar** (`collapse_quantized_reuploads`): els
   repositoris GGUF amb la mateixa etiqueta `base_model:quantized:X` es redueixen al més
   descarregat. Només s'agrupen còpies GGUF entre elles: un primer intent també fusionava les
   quantitzacions AWQ, FP8 o MLX i un AWQ d'1 M de descàrregues va amagar el repositori GGUF
   oficial del Qwen2.5-Coder-32B. Un fine-tune (que quantitza un altre repositori) no es fusiona.
3. **Una consulta de mida per cubell** (`size_bands`, amb els cubells que ja existeixen: 4-14B,
   14-32B i més enllà si la màquina ho permet), cadascuna amb frase i oberta. A la B60 passa de 8
   a 12 consultes: +0,6 s. Per això `MAX_UNIQUE_CANDIDATES` passa de 40 a 100: els resultats ja
   estan baixats, no costa xarxa, i amb 12 consultes el tall de 40 treia els repositoris oficials.
4. **Mida del model abans de les descàrregues en un empat**, només a Balanced i Quality. La mida
   ve de `general.size_label` de la capçalera GGUF (`14B`, `12B-A2.5B`, `30B-A3B`), que declara
   l'artefacte i no el nom del repositori, i entra a `resolve_parameter_count` entre la
   configuració i el nom (`GGUF_METADATA`, confiança mitjana). Per a un MoE es compta el total.
   El nou criteri `model_size` va abans de l'escala de quantització i de les descàrregues.
   Esquema de la memòria cau de capçaleres a 2.

**Correccions per una auditoria posterior (tancades abans de tocar més el rànquing):**
(1) *Agrupar sense perdre alternatives viables.* Compartir model base no vol dir compartir
quantitzacions: la còpia més descarregada pot tenir només fitxers que no càpiguen. Ara
`collapse_quantized_reuploads` retorna `Reuploads(kept, alternatives)` i
`rescue_with_reupload_alternatives` inspecciona les còpies germanes, de més a menys descarregades,
quan la còpia mantinguda no té cap col·locació confirmada, dins del pressupost de variants
(`MAX_VARIANT_DEEP_INSPECTION`). (2) *Una capçalera parcial no és una completa.* Un tall dins del
tokenitzador només s'accepta si ja hi són les cinc claus que llegeix l'estimació (context,
embedding, capes, caps i caps KV); la capçalera es marca `is_complete=False`, no es dedueix cap
clau absent i l'esquema de la memòria cau passa a 3. Si una clau pot encara venir després, es
continua demanant bytes. (3) *Cancel·lar no espera peticions bloquejades.* La precàrrega consulta
la cancel·lació abans d'enviar feina i cada 0,1 s, i en cancel·lar fa `shutdown(wait=False,
cancel_futures=True)`; una petició ja a la xarxa acaba pel seu temps màxim HTTP (10 s). (4) *L'estat
exportat té les mateixes dades que el rànquing:* la precàrrega i `declare_sizes` s'executen abans
d'enriquir i publicar els candidats.

**Límits pendents:** (a) la confiança del pla (metadades de la targeta més estimació) va abans
de la mida: `unsloth/Qwen3.8-27B-GGUF` té dues penalitzacions (sense idiomes, sense
`pipeline_tag`) i queda per darrere de models més petits; (b) un MoE es compta pel total, de
manera que un 30B-A3B passa per davant d'un 14B dens, cosa discutible; (c) els derivats de
tercers (abliterated, uncensored) segueixen ocupant places; (d) es llegeix una capçalera per
quantització. Reutilitzar la informació d'arquitectura d'un repositori només seria acceptable
revisant-ne la consistència, i mai les dades pròpies de cada artefacte o quantització.

**Estat:** `IMPLEMENTED` amb tests que fallaven abans. Temps: `VALIDATED` al WSL (61,4 → 29,7 s,
mateix top 5); `UNVERIFIED` a la B60.

**Fonts:** [precàrrega](../../src/jaull/workflow/header_prefetch.py),
[consultes](../../src/jaull/discovery/query_builder.py),
[filtre de candidats](../../src/jaull/discovery/candidate_filter.py),
[famílies](../../src/jaull/domain/families.py), D101, D107, D109.

### D111. Quality: la mida del model abans de la quantització (preferència heurística) i diagnòstic de la preselecció

**Data i necessitat:** 2026-10-27. A la B60, Quality · General chat posava dos 9B a `Q8_0` per
davant d'un Qwen3.8-27B a `Q4_K_M`. La clau de Quality comparava la qualitat de la quantització
abans que la mida del model, i la capacitat s'agrupa en nivells (un 7B i un 27B són tots dos
`strong`), de manera que `Q8_0` guanyava a `Q4_K_M` abans de mirar la mida. Això barreja la
fidelitat de la quantització amb la capacitat del model. Una revisió externa va confirmar el
mecanisme reproduint-lo amb candidats controlats.

**Decisió:** la clau de Quality passa a (idoneïtat, capacitat, **quantització agressiva**,
runnability, **mida del model**, qualitat de la quantització, ajust). Dins d'un mateix model la
quantització continua decidint (`Q6_K` abans de `Q4_K_M`); entre models, el més gran primer.
**És una preferència heurística explícita, no una demostració de qualitat:** la mida declarada
(`general.size_label`) no mesura què fa millor un model, i un 9B pot ser millor que un 27B per a
una tasca concreta. L'informe ho diu a la nota del rànquing de Balanced i Quality, i `ranking.criteria`
publica `aggressive_quantization` i `model_size` com a eixos de Quality. **No es canvia la regla
d'agressives de D105:** un 32B a `Q2_K` no passa automàticament per davant d'un 9B a `Q8_0`, ni
tampoc s'afirma que el 9B sigui millor; falta evidència per resoldre aquest intercanvi.

**Diagnòstic de la preselecció (fet offline sobre l'estat guardat de Balanced · General chat,
sense noves cerques):**
1. *El 14B de xat no arriba a la inspecció.* `bartowski/Qwen2.5-14B-Instruct-GGUF` és al conjunt de
   100 candidats, però queda amb 20,90 punts i l'última plaça (12a) la té `tencent/Hy-MT2-7B-GGUF`
   amb 21,11, per 0,21. Els 12 triats inclouen tres especialitzats en codi en una cerca de xat
   (Qwen3-Coder-30B-A3B, Qwen2.5-Coder-14B i -7B, que sumen +1,5 per coincidir amb 3 consultes), dos
   Ornith i dues variants del Qwen3.8-27B. El punt de tall és una puntuació de popularitat
   (`log` de descàrregues i de m'agrada), no un criteri d'idoneïtat.
2. *Per què el 27B queda darrere del Qwen2.5-7B a Balanced:* el primer criteri que els separa és
   `plan_confidence` (`low` abans de `unknown`), perquè el 27B té dues penalitzacions de metadades
   (sense idiomes i sense `pipeline_tag`). Aquí la confiança sí intervé, a diferència del 14B,
   que no hi arriba. No s'ha canviat.
3. *Una diferència de mida petita decideix:* Ornith 1.0 (9B) passa per davant del Qwen2.5-7B (7,6B)
   per `model_size`, però `9B` és una etiqueta arrodonida i `7.6B` una de precisa; comparar-les
   com a exactes és soroll.
4. *Les 26 candidates de format no GGUF* queden fora (`too_large`), amb un màxim de 2 de recurs.

**Pendent:** (a) i (d) resolts a D112 (idoneïtat abans que popularitat; versions repetides com a
reserva, sense fusionar-les ni ampliar la inspecció); (b) la confiança de metadades abans de la mida
i (c) un marge a `model_size` continuen oberts.

**Estat:** Quality `IMPLEMENTED` amb un test que fallava abans; el diagnòstic és una execució
offline, no una prova. `UNVERIFIED` a la B60.

**Fonts:** [rànquing v2](../../src/jaull/recommendation/engine_v2.py),
[preselecció](../../src/jaull/discovery/candidate_filter.py),
[informe](../../src/jaull/reporting/recommendation.py), D105, D110.

### D112. Preselecció: una altra tasca no desplaça els adequats, i les versions repetides esperen

**Data i necessitat:** 2026-10-27. El diagnòstic de D111 va mostrar que el Qwen2.5-14B-Instruct de
xat no arribava a la inspecció (20,90 punts contra 21,11 de l'última plaça) mentre tres models de
codi hi ocupaven lloc en una cerca de xat, perquè tres consultes els havien retornat (+1,5). Una
revisió externa va demanar corregir la selecció de candidats sense barrejar-la amb la confiança, i
sense ampliar les inspeccions per quedar «a pocs dècims».

**Decisions:**
1. *Idoneïtat abans que popularitat, amb la senyal que ja existia.* `targets_another_task` reutilitza
   les paraules clau negatives de `task_match` (un `coder` en una cerca de xat; vegeu
   `_NEGATIVE_KEYWORDS`). Un repositori que anuncia una altra tasca omple plaça només després dels
   altres, dins del seu grup de col·locació i mida; les descàrregues, els m'agrada i el nombre de
   consultes ordenen el que queda. És una afirmació sobre el que diu el repositori, no sobre la
   seva qualitat. **No s'ha usat `task_match` com a número:** el Qwen3.8-27B oficial puntua 0,78
   només per no declarar `pipeline_tag` i quedaria darrere de molts models petits. A Coding no hi ha
   paraules negatives, de manera que la preselecció no hi canvia.
2. *Diversitat sense fusionar.* Una segona versió del mateix model (mateixa família i mateixa mida
   declarada pel nom, per exemple Ornith 1.0 i 1.5 de 9B) espera darrere dels altres candidats del
   seu grup; es manté com a reserva i s'inspecciona si el pressupost no s'esgota. No s'eliminen com a duplicats:
   una altra mida, una altra família, o un fine-tune d'una altra família no s'endarrereix. Això
   és diferent del defecte de D101 (una família nova passant per sobre d'una millor puntuació): aquí
   només s'endarrereix una còpia de la mateixa família i mida, i només dins del grup.
3. *Els 12 llocs d'inspecció no s'amplien.*

**Verificat (offline, sobre el conjunt guardat de 100 candidats, sense cerques noves):** el 14B de xat
entra; Qwen3-Coder-30B-A3B i Qwen2.5-Coder-14B surten de la preselecció. Regressió amb els números
reals del cas (`_SAVED_CHAT_POOL`) a `tests/test_discovery_search.py`.

**Pendent, sense canviar:** (a) SmolLM2-135M continua ocupant la plaça de «mida desconeguda» (una
per ronda, de D101); (b) la confiança de metadades abans de la mida, que fa baixar el 27B:
primer cal revisar quina incertesa continua vigent després d'inspeccionar (un `pipeline_tag` absent
no hauria de continuar significant format desconegut si ja s'ha confirmat; l'idioma desconegut
sí que continua sent informació); (c) un marge a `model_size` només amb justificació explícita i
coneixent la precisió de l'etiqueta.

**Estat:** `IMPLEMENTED` amb tests que fallaven abans. `UNVERIFIED` a la B60.

**Fonts:** [preselecció](../../src/jaull/discovery/candidate_filter.py),
[senyal de tasca](../../src/jaull/recommendation/scoring.py), D101, D111.

### D113. Menys conservadorisme amb la evidència dels metadades (idioma i pipeline_tag)

**Data i necessitat:** 2026-10-27. L'usuari va assenyalar que de vegades som massa conservadors i
que en alguns casos podem ser més permissius. S'ha separat en dos tipus, perquè no costen el mateix
si s'equivoquen: el conservadorisme sobre la **evidència dels metadades** (idioma, `pipeline_tag`,
confiança), on un excés només fa que una bona opció quedi pitjor col·locada, i el sobre la
**memòria**, on l'error és asimètric (sobreestimar la capacitat acaba en un error de memòria). Els
llindars de memòria (75 %, 90 %, marge del 10 %) no es toquen sense una mesura.

**Decisions:**
1. *Un idioma sense declarar és desconegut, sense penalització global.* La majoria de les targetes
   ometen el camp, de manera que no diu res del model. Ja es reflecteix a `language_fit`
   (desconegut), al requisit («suport no confirmat als metadades») i a l'explicació. Abans, a més,
   baixava un nivell la confiança dels metadades i restava 0,5 a la preselecció. Ara no.
2. *Un `pipeline_tag` absent és un avís, no una pèrdua de confiança.* Diu quina és la tasca, que la
   idoneïtat ja pondera (el `task_match` perd la bonificació del pipeline); no diu res del format ni
   del runtime, que confirma la inspecció. L'avís continua visible. Una llicència absent continua
   baixant la confiança: és una altra cosa.
3. *«Només altres idiomes declarats» ja no rebutja el pla.* Era una porta dura
   (`LANGUAGE_INCOMPATIBLE`). Les targetes declaren els idiomes de l'enfocament d'entrenament i moltes
   ometen l'anglès, de manera que no demostra que el model no pugui amb l'idioma demanat. Ara baixa
   la idoneïtat un nivell i `language_fit` passa de `blocked` a `weak`. El codi
   `LANGUAGE_INCOMPATIBLE` es manté al model per als informes desats, però ja no es genera.
4. *La confiança abans que la mida (Balanced) no es toca.* Amb 1 i 2, el Qwen3.8-27B passa de
   confiança `unknown` a `low`, igual que els rivals, i la clau no ha de canviar.

**Verificat (offline, sense cerques noves):** el rànquing guardat de Balanced · General chat, refet
amb les regles velles, es reprodueix exactament; amb les noves, el Qwen3.8-27B puja del lloc 5 al 2.
La reconstrucció només coneix la confiança base `high` o `medium`; no sap si la targeta no tenia
dades (`low`), de manera que és una aproximació per a aquest conjunt. La puntuació composta
diagnòstica d'un cas fix del test d'informe passa de 46 a 47 perquè el seu terme de metadades
compta penalitzacions; l'ordre no canvia.

**Pendent:** la política de llicència amb ús comercial es manté fixa; un marge a `model_size` només
amb justificació explícita i coneixent la precisió de l'etiqueta; un `pipeline_tag` absent a
repositoris no GGUF, on la inspecció no confirma tant com a un GGUF.

**Estat:** `IMPLEMENTED` amb tests que fallaven abans. `UNVERIFIED` a la B60.

**Fonts:** [filtre de candidats](../../src/jaull/discovery/candidate_filter.py),
[rànquing v2](../../src/jaull/recommendation/engine_v2.py),
[recomanació](../recommendation.md), D111, D112.

### D114. Idoneïtat per inferència d'un perfil generalista (recurs prudent, sense evidència)

**Data i necessitat:** 2026-10-27. A Coding, el Qwen3.8-27B (un bon model tant per programar com per
xatejar) quedava al lloc 49 amb idoneïtat `weak`: la idoneïtat només mesura si el repositori
**anuncia** la tasca pel nom o les etiquetes (`coder`, `code`), no si hi serveix. Una revisió va
demanar combinar l'evidència publicada, quan sigui aplicable, amb un recurs prudent quan falti,
sense tractar l'absència d'evidència com a incapacitat.

**Decisió (opció A; l'opció B queda com a política a part):** un model instruct o de xat amb
capacitat almenys `adequate` (puntuació de capacitat ≥ 0,55, el llindar que ja usa el motor) és
`adequate` per a una tasca que el seu nom no anuncia. La regla és `infers_generalist_suitability`
(a `scoring.py`) i només s'aplica si **tot** això és cert: la `task_match` és sota `adequate`; hi
ha una senyal d'ajust a instruccions (`instruct`, `instruction`, `chat`, `assistant`,
`conversational`); no hi ha senyal de model base; no hi ha cap paraula forta d'una **altra** tasca
(traducció, resum, extracció, matemàtiques, RAG, codi quan no s'ha demanat) i la capacitat és
`adequate` o millor. La paraula `instruct` no n'hi ha prou: un especialista d'una altra tasca no
puja, ni un model base, ni un model massa petit. L'etiqueta de **llinatge** `base_model:org/Model`
no compta com a senyal de model base (partida en paraules donava `base` i el Qwen3.8-27B
semblava un model preentrenat); un nom amb `base` sí.
L'explicació diu: «La idoneïtat per a [tasca] s'infereix del seu perfil generalista i instruct; no
està confirmada per cap avaluació aplicable.»

**Efecte observat (offline, sobre el conjunt guardat de Coding, sense cerques noves):** el
Qwen3.8-27B passa de `weak` (lloc 49) a `adequate` (lloc 7), per darrere dels models amb `coder`
al nom, que continuen `strong`. Els Ornith també passen a `adequate`; que el Qwen els avanci no
està forçat: decideixen els criteris següents (la mida, en aquest cas).

**Límits acceptats, sense corregir amb regles de nom:**
- Els models orientats explícitament a codi continuen per davant: la idoneïtat va abans que la
  mida. No es promociona cap model concret.
- Un especialista d'una altra tasca que no ho diu a les etiquetes es cola. `tencent/Hy-MT2-7B-GGUF`
  era el cas: només té `gguf` i `conversational`. Ho resol D116 quan el repositori base ho declara
  (aquí ho fa: `translation`); si no ho declara enlloc, continua passant per generalista.
- No s'ha tocat el catàleg d'avaluacions ni s'ha introduït cap llindar.

**Pendent (opció B, política a part):** que una avaluació publicada **aplicable** pugi la
idoneïtat. Cal una regla explícita: identitat exacta o llinatge confirmat, font, tasca, metodologia
i resultat conservats, la referència publicada separada d'una avaluació local del GGUF, i una
comparació amb protocol comparable. Les referències no es poden triar només perquè portin `coder`
al nom, i el catàleg actual (11 avaluacions, cap d'un model d'aquest tipus) no en té. Una avaluació
publicada que dona mal resultat no es pot tractar com un punt a favor.

**Estat:** `IMPLEMENTED` amb tests (generalista, especialista, base, capacitat baixa, llinatge,
coder sense canvis i explicació). `UNVERIFIED` a la B60.

**Fonts:** [senyal](../../src/jaull/recommendation/scoring.py),
[motor](../../src/jaull/recommendation/engine_v2.py),
[explicacions](../../src/jaull/recommendation/explanations.py), D111, D113.

### D115. Cobertura de models petits en equips limitats, sense canviar la preselecció de la B60

**Data i necessitat:** 2026-10-27. En un portàtil amb uns 7,5 GiB de RAM, l'usuari trobava
només una recomanació de 7B a Quality i trobava a faltar alternatives petites. Es va reproduir
offline un mecanisme que ho podia explicar: el càlcul preliminar considerava plausible un
7B GGUF (uns 5,55 GiB) i dotze candidats d'aquesta mida ocupaven totes les inspeccions, deixant
fora models de 0,5B, 1,5B i 3B. La regla de cobertura que evitava omplir la B60 de models petits
s'aplicava també al portàtil. Les consultes acotades per mida començaven a 4B i no compensaven
aquesta manca de cobertura. No s'ha reproduït la cerca real del portàtil.

**Decisió:** adaptar la cobertura al sostre preliminar de paràmetres que ja calcula Jaull,
sense classificar equips pel fabricant ni modificar les fórmules de memòria.
- Si el sostre arriba com a màxim al cubell mitjà existent (14B), les consultes inclouen els
  intervals 0-1B, 1-2B i 2-4B, retallats pel sostre, a més dels intervals més grans que hi càpiguen.
  Cada interval manté una consulta amb frase i una d'oberta, amb el límit habitual per pàgina.
- La preselecció conserva primer un candidat més gran plausible i després cobreix cadascun
  dels tres cubells petits, quan hi ha candidats en la millor classe de col·locació. En equips
  limitats, les places restants no donen prioritat automàtica als models de més de 4B;
  es mantenen els criteris de col·locació, tasca, popularitat i diversitat. Amb una sola plaça
  no es desplaça el candidat més gran per reservar cobertura petita.
- Si el sostre supera 14B, es mantenen les consultes i la prioritat anterior pels models més
  grans. La regla depèn del context i la concurrència sol·licitats, no només del dispositiu.
- No es modifica el rànquing final, la idoneïtat, la confiança, la quantització ni els límits
  globals de candidats i inspeccions. Un 7B pot continuar sent la primera recomanació.

**Verificat:** tests offline amb 7,5 GiB de RAM, CPU i integrada Vulkan amb memòria compartida,
per a Coding i General chat. Les alternatives petites arriben a la inspecció encara que
hi hagi dotze 7B més populars; un test del workflow també les conserva a Results quan la
inspecció revela que la majoria dels 7B no caben. El conjunt guardat de General chat de 24 GiB
manté exactament els dotze candidats i el seu ordre, també amb una B60 Vulkan sintètica al
costat d'una integrada i afegint candidats petits molt populars al conjunt. Es comproven
els límits dels intervals, el pressupost i el determinisme.

**Comprovacions executades:** 399 tests seleccionats aprovats, incloent discovery, workflow,
rànquing, informes i arquitectura; Ruff, mypy, compileall i `git diff --check` correctes.
No s'ha executat la suite completa per aquest canvi ni s'han fet cerques de xarxa,
descàrregues de pesos o inferències per validar-lo.

**Revisió posterior:** la primera correcció només reservava una plaça per a tot l'interval
1-4B. L'usuari encara trobava un 1,5B però cap 3B; es va reproduir que el segon quedava fora de
la inspecció. Ara es divideix aquell interval en 1-2B i 2-4B, tant a les consultes com a la
preselecció. El test exigeix simultàniament 0,5B, 1,5B i 3B i també comprova que un 3B viable
arribi a Results. Les cinc recomanacions continuen sent un màxim, no una quota a omplir.
El recompte anterior de 399 tests correspon a la primera correcció.
La revisió passa 406 tests seleccionats, incloent arquitectura i regressió de la B60;
Ruff, mypy, compileall i `git diff --check` també passen. La suite completa continua pendent.

**Límits:** el sostre és una heurística de cobertura, no una garantia de viabilitat. En equips
limitats s'afegeixen com a màxim sis consultes de metadades respecte de la cerca original sense
cobertura petita; aquesta revisió n'afegeix dues respecte de la primera correcció. No s'ha mesurat
la seva latència real. La preservació de la B60 s'ha comprovat amb dades guardades i perfils sintètics,
no amb una nova execució física. Cal comprovar els resultats reals al portàtil.

**Estat:** `IMPLEMENTED`, amb regressions offline; `UNVERIFIED` a les màquines físiques.

**Fonts:** [preselecció](../../src/jaull/discovery/candidate_filter.py),
[consultes](../../src/jaull/discovery/query_builder.py),
[regressions](../../tests/test_small_machine_discovery.py),
[recomanació](../recommendation.md), D110, D112.

### D116. Una re-quantització hereta la tasca que declara el seu repositori base

**Data i necessitat:** 2026-10-27. Amb el recurs generalista de D114, `tencent/Hy-MT2-7B-GGUF`, que
és un model fet només per traduir, passava a `adequate` per programar. Les seves etiquetes són
`gguf`, un identificador d'article i `conversational`: res en un repositori GGUF dona la tasca. El
repositori **base** que declara (`base_model:quantized:tencent/Hy-MT2-7B`) sí que ho diu:
`pipeline_tag = translation` i l'etiqueta `translation`. Un repositori que re-quantitza un altre
rarament copia els seus camps de tasca.

**Decisió:** abans de la preselecció, les re-quantitzacions **sense `pipeline_tag` propi** consulten
el seu repositori base declarat (una consulta per base, 8 fils, tolerant a errors) i hereten
**només les etiquetes que anomenen una tasca** que el motor entén (`task_vocabulary`: les paraules
clau de cada cas d'ús). És la mateixa evidència estructurada (linatge declarat) que ja confirma la
identitat d'un model, no una regla de nom. **No s'hereta el `pipeline_tag`:** el del Qwen3.8-27B
base és `image-text-to-text` (multimodal) i heretar-lo rebutjaria un GGUF que carrega com a model de
text. Si la base no es pot llegir, el candidat queda com estava.

`targets_another_task` (D112) també considera una altra tasca la traducció, el resum i l'extracció
(i continua comptant `coder` en una cerca de xat), llevat que sigui la tasca demanada. Així Hy-MT2,
amb `translation` heretada, no passa a `adequate` per programar (D114) i es relega a la preselecció
d'una cerca de xat.

**Cost mesurat (conjunt guardat de 100 candidats, consultes reals només a les bases):** 3
re-quantitzacions sense pipeline, 3 bases, **0,2 s amb 8 fils** (0,5 s en sèrie). Només Hy-MT2 hereta
una etiqueta (`translation`); el Qwen3.8-27B no canvia ni es rebutja.

**Límits:** només cobreix les re-quantitzacions que declaren la base i no tenen pipeline; un
especialista que no ho declara enlloc continua passant per generalista; el vocabulari és el dels
casos d'ús, de manera que una tasca no llistada (embeddings, OCR) no es detecta aquí.

**Estat:** `IMPLEMENTED` amb tests que fallaven abans (herència, pipeline multimodal, una consulta per
base, errors, paral·lelisme i cancel·lació, i `targets_another_task`). `UNVERIFIED` a la B60.

**Fonts:** [linatge](../../src/jaull/discovery/lineage.py),
[senyals de tasca](../../src/jaull/recommendation/scoring.py),
[preselecció](../../src/jaull/discovery/candidate_filter.py), D112, D114.

### D117. Un idioma sense declarar no degrada l'etiqueta de la recomanació

**Data i necessitat:** 2026-10-27. A la B60, tres de les quatre llistes (Coding · Quality, xat ·
Quality i xat · Balanced) deien «Best-Effort Suggestion» i només una deia «Recommended». L'usuari va
demanar no ser tan conservadors. Amb les dades guardades, els quatre resultats que baixaven a
`best_effort` tenien **la mateixa i única causa**: la targeta no declarava idiomes, el filtre de
requisits ho comptava com a requisit no complert («Support for EN not confirmed in metadata»,
penalització del 15 %, multiplicador 0,85) i `choose_tier` baixa a `best_effort` amb qualsevol
multiplicador inferior a 1. Els models que declaraven `en` sortien com a `recommended`. És el mateix
conservadorisme que D113 va treure de la preselecció i de la confiança, en un camí que D113 no tocava.

**Decisió:** al filtre de requisits (`requirements_gate`), un idioma demanat es classifica en tres
estats. *Confirmat* (el declara o la targeta és `multilingual`): sense cost. *Sense declarar*: és
desconegut; es manté l'avís («no confirmat als metadades») però **no costa res** i no degrada
l'etiqueta. *Només declara altres idiomes*: continua sent una penalització suau del 15 %, amb l'avís
«la targeta només declara altres idiomes». L'etiqueta de la recomanació no canvia de regla
(`choose_tier`): només deixa de rebre un multiplicador artificial.

**Efecte (offline, sobre els resultats guardats de les captures):** els quatre `best_effort` (Ornith
1.5, Ornith 1.0, Qwen3.8-27B i Qwen3-Coder-30B-A3B) passen a `recommended`; la resta no canvia. La
puntuació composta diagnòstica del test d'informe passa de 47 a 55 i el seu `hard_penalty` de 0,85 a
1,0; l'ordre no canvia.

**Límits, sense canviar:** `BEST MATCH` continua sent pràcticament inabastable amb una estimació de
GGUF perquè la confiança del pla és `low` (suposicions de política documentades, com el marge de
sobrecàrrega) i `choose_tier` la limita a `recommended`; només l'evidència mesurada la puja. I un
pla `tight` pot quedar primer a Quality (el 30B-A3B a 22,42 GiB de la captura), perquè la
runnability agrupa `tight` amb `compatible` i Quality prioritza la mida; si cal ser més estrictes
amb `tight` a Quality és una decisió pendent.

**Estat:** `IMPLEMENTED` amb tests que fallaven abans. `UNVERIFIED` a la B60.

**Fonts:** [filtre de requisits](../../src/jaull/recommendation/requirements_gate.py),
[etiqueta](../../src/jaull/recommendation/tier.py), D113, D114.

### D118. La paraula Coder no demostra superioritat davant d'un generalista adequat

**Data i necessitat:** 2026-10-27. L'informe Coding · Quality de la B60 inclou un generalista
de 27B inspeccionat, amb Q4_K_M compatible, però recomana especialistes de 8-9B. La idoneïtat
inferida de D114 continua sent `adequate`, mentre que anunciar codi dona `strong`; aquest
eix es comparava abans de la capacitat i de la mida. És una preferència pels metadades de
tasca, no una comparació de qualitat de programació.

**Decisió:** només a Coding · Quality i Balanced, `strong` i `adequate` comparteixen el rang
d'idoneïtat elegible. No es modifica la valoració original ni s'atribueix evidència mesurada
al generalista. Es conserva la degradació quan els idiomes declarats no coincideixen, i no
s'eleven perfils `weak`, `unknown`, traductors o models base. La resta de criteris, restriccions,
estimacions i polítiques mesurades continuen vigents; no hi ha excepcions per noms de models.
L'informe JSON i Markdown explica l'agrupació i manté visible el nivell original.

**Límits:** no garanteix que un 27B sigui millor que un 8B ni que entrin cinc models grans.
No canvia la descoberta, la confiança derivada de benchmarks històrics ni el càlcul de KV.
La B60 real i el seu conjunt complet de candidats encara s'han de tornar a provar.

**Evidència:** regressions offline amb generalista de 27B Q4 compatible i Coder de 8B Q8
comfortable, en ambdós ordres d'entrada i prioritats; proteccions per tasca, idioma i Speed;
coherència entre els criteris del motor i els informes. D114 i D116 continuen descrivint
d'on prové la inferència, no una mesura de qualitat.

**Verificació:** 438 tests de recomanació, workflow, informes, cobertura de màquines petites
i arquitectura passen. Les quatre variants de la regressió principal fallen en restaurar
el criteri anterior en memòria, sense modificar fitxers. Ruff i mypy també passen.

### D119. Tenir una execució guardada no eleva la confiança global del pla

**Data i necessitat:** 2026-10-27. A Coding · Balanced, el Coder de 7B amb un benchmark guardat
sortia davant dels models grans. `_confidence` retornava `high` quan trobava qualsevol
experiment o benchmark coincident. Balanced compara aquesta confiança abans de la mida:
una referència històrica `llama_bench_v1`/tg64 podia alterar l'ordre encara que l'avaluació de
velocitat no la considerés comparable. El comentari que deia que aquestes referències no
ordenaven els plans no cobria aquest camí indirecte.

**Decisió:** la confiança del pla continua derivant dels metadades i de la predicció de memòria,
també quan hi ha registres locals. L'existència d'una execució o benchmark no demostra qualitat
ni valida totes les hipòtesis de memòria. Es mantenen els identificadors, les observacions,
la memòria mesurada i l'historial. Les polítiques explícites de mesures comparables no canvien.
També s'elimina l'ascens per tenir només una validació guardada, pel mateix motiu.

**Límit:** no es prohibeix que una mesura pertinent afecti el seu eix. Per exemple, el criteri
de petjada de memòria de Speed/Memory continua podent consultar memòria mesurada. El canvi
elimina el bonus global de confiança, no totes les formes d'evidència local.

**Evidència:** vuit regressions cobreixen Balanced/Quality, benchmark històric/validació i
els dos ordres d'entrada. Comproven l'ordre base i les recomanacions finals, que la confiança
no puja, i que els IDs i mesures es conserven. Les vuit fallen en reinstaurar en memòria
l'ascens antic a `high` i passen amb la correcció. No s'han modificat els registres guardats.

**Auditoria de cobertura, separada del fix:** s'han revisat dos estats antics locals de
depuració, amb 85 candidats de Quality i 86 de Balanced. No són el JSON de 74 candidats
que l'usuari ha tornat a enviar (`2026-10-27T11:36:48Z`) ni la captura posterior amb el 27B
segon. En el pool Quality antic hi ha onze repositoris amb mida inferida superior a 9B
fora de Qwen, tots EXAONE de 32B; no equivalen a onze models base diferents ni a onze
alternatives confirmades. Cap no figura entre els avaluats d'aquell estat. En reproduir
la preselecció actual sobre aquell pool entren `LGAI-EXAONE/EXAONE-4.0-32B-GGUF` i el
Coder oficial de 32B: és una reproducció de la selecció, no una inspecció ni prova de fit.
DeepSeek-Coder-V2-Lite sí que s'havia inspeccionat, però quedava `unknown` per MLA i amb
llicència comercial sense confirmar. No és una demostració que no càpiga.

**Què no es pot concloure:** l'exportació rebuda mostra els 14 avaluats, no els altres 60
candidats ni les respostes originals de cada consulta. No permet atribuir l'absència d'una
altra família a una frontera concreta. Les consultes amb `coder instruct` depenen del nom
del repositori, i les obertes continuen sent pàgines limitades per descàrregues; no garanteixen
cobertura exhaustiva. No s'han afegit marques, quotes ni excepcions al selector per maquillar
aquesta incertesa. Falta una captura nova amb el pool complet per auditar el cas actual.

**Traça local dels estats auditats (no versionats):**
`b60_state_quality.pkl`, SHA-256
`9cc839daa829380c024e69d861e330c3e99119a1b4050cacf84a71621c7ec832`;
`b60_state_balanced.pkl`, SHA-256
`edb8049cfa2a2e988a6c0d133d7f31959ce48774b324a4f2e0ff36f4a4bfab77`.

**Fonts:** [motor](../../src/jaull/recommendation/engine_v2.py),
[evidència](../evidence.md), D114-D118. Substitueix la implicació de D117 que una mesura
guardada eleva per si sola la confiança global; no canvia el tier ni l'estimador.

### D120. L'overhead d'un GGUF no creix amb els pesos (pas 1 de 2)

**Data i necessitat:** 2026-10-30. La pantalla de resultats donava a la B60 xifres de memòria
molt per damunt del que cal: el Qwen3.8-27B Q4_K_M sortia amb 20,77 GiB i etiqueta `Tight`
quan, de fet, s'hi executava amb 23,6 GiB lliures. Es va comparar l'estimació amb el que llama.cpp
calcula ell mateix en carregar (`common_memory_breakdown_print` i `projected to use`, amb
`llama-server.exe -lv 4`, build b11889, Vulkan natiu a Windows, context 4096).

**Mesures** (un únic dispositiu, B60):

| Model | Pesos a la GPU (MiB) | Context (MiB) | Còmput (MiB) | **Total real** | **Jaull abans** | De més |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Coder 30B-A3B Q4_K_M | 17524 | ~384 (derivat) | ~88 (derivat) | **17,57 GiB** (17996 MiB) | 22,42 GiB | +4,85 GiB (+28 %) |
| Qwen3.8-27B Q4_K_M | 14674 | 854 | 156 | **15,32 GiB** (15684 MiB) | 20,77 GiB | +5,45 GiB (+36 %) |
| Ornith 1.5 9B Q8_0 | 8045 | 329 | 116 | **8,29 GiB** (8490 MiB) | 12,70 GiB | +4,41 GiB (+53 %) |

Per al Coder, el total ve de `projected to use`; context i còmput són derivats (fórmula de KV i
resta), no mesures directes. Els altres dos surten de la taula de llama.cpp.

**Què fa l'estimador** (constants de `estimator/policies.py`; per al Coder: pesos 17,28 + KV 0,375 +
overhead 2,23 + reserva 0,50, i marge del 10 % → 22,42 GiB): `overhead = 512 MiB + 10 % dels pesos`,
i després el marge del 10 % s'aplica a tot, overhead inclòs. El buffer de còmput real és de
90-160 MiB en els tres models, entre 13 i 20 vegades menys que l'overhead estimat. Els dos
percentatges s'apilen i expliquen uns 4,3 GiB dels ~5 GiB de més.

**Decisió (pas 1):** per a repositoris GGUF, que a Jaull s'executen amb llama.cpp, l'overhead
passa a ser els 512 MiB base **sense** el 10 % dels pesos (`GGUF_OVERHEAD_WEIGHT_FRACTION = 0.0`).
Les 512 MiB són unes quatre vegades el mesurat. Per a la resta de tipus de repositori es manté
el 10 %, perquè no s'hi ha mesurat res. La reserva de 512 MiB i el marge del 10 % **no es toquen**.

**Efecte previst** (càlcul sobre les constants, **no una mesura**): Coder ≈ 20,5 GiB, 27B ≈ 19,0 GiB,
Ornith ≈ 11,7 GiB. Conseqüència de política: el 27B passaria de `Tight` a `Compatible`. A
Balanced, `comfortable` i `compatible` pesen igual i `tight` penalitza, de manera que **el 27B
deixa de penalitzar-se en memòria**. Pot canviar el rànquing, i per això va com a decisió pròpia.
També fa que, en una descàrrega parcial, hi càpiguen més blocs a la GPU (vegeu l'evidència).

**Pas 2 (pendent, no implementat):** substituir la reserva fixa i el marge percentual per
**1 GiB fix**, que és el llindar que llama.cpp usa en el seu propi ajust
(`will leave 5617 >= 1024 MiB of free device memory`). Donaria aproximadament Coder 19,2, 27B 17,8
i Ornith 11,1 GiB. És una heurística, i abans cal comprovar si Jaull compara contra la memòria
**lliure** mesurada o contra la total: la B60 ja tenia 770 MiB ocupats per Windows (23613 lliures
de 24382), i si compara contra la lliure la reserva de 512 MiB es comptaria dues vegades. No
s'ha mesurat cap cas que fallés per memòria, i el cost d'un OOM al PC de Biosfer és molt superior
al de marcar un model com `Tight`; per això el pas 2 és separat.

**Límits:** tres models, un sol context (4096), llama.cpp amb Vulkan. Els `llama-server` usaven
4 slots, i amb un únic `llama-cli` el context seria una mica menor (si hi ha biaix, va cap al costat
conservador). No es valida CUDA ni CPU. La preselecció aproximada de `candidate_filter` (que també
tracta repositoris que no són GGUF) no canvia. Els pesos que llama.cpp deixa a la CPU (embeddings,
de 167 a 1030 MiB) i el KV de les capes recurrents dels models híbrids `qwen35` tampoc no es
corregeixen aquí; en són millores a part, cadascuna amb la seva evidència.

**Fonts:** [comparativa de rendiment](../b60-comparativa-wsl2-windows.md),
[estimació](../estimation.md), [política](../../src/jaull/estimator/policies.py). Relacionada amb
D102 (KV de MoE) i D098 (B60 sobre WSL2).

### D121. Les mesures de la B60 es fan amb el binari de Windows i no amb WSL2

**Data i necessitat:** 2026-10-30. Jaull executa llama.cpp dins de WSL2, i a la B60 de Biosfer això
passa per Mesa Dozen (Vulkan sobre Direct3D 12). Calia decidir amb quin entorn es fan les mesures
de velocitat i la prova d'IFEval que s'inclouen a la memòria.

**Evidència que ho motiva** (detall a [la comparativa](../b60-comparativa-wsl2-windows.md)): mateix
driver d'Intel (`32.0.101.8976`), mateix llama.cpp (b11889), mateixos GGUF.

| Model | Generació a WSL2 + Dozen | Generació a Windows natiu |
|---|---:|---:|
| Ornith 9B Q8_0 | 4,1 t/s | 36,2 t/s |
| Qwen3-Coder 30B-A3B Q4_K_M | 2,5 t/s | 99,2 t/s |
| Qwen3.8-27B Q4_K_M | cau (`Segmentation fault`) | 15,55 t/s |

A WSL2 la B60 va entre 8 i 40 vegades més lenta, i el Qwen3.8-27B cau sempre dins de `libLLVM-14.so`
(driver d'Intel exposat a WSL, mateix offset en quatre caigudes). **VALIDATED:** la diferència
i la caiguda. **UNVERIFIED:** quin shader provoca la caiguda i per què el 9B de la mateixa
família no cau.

**Decisió (de l'autor):** les mesures de rendiment de la B60 que van a la memòria es fan amb el
**binari de Vulkan de Windows** (`llama-b11889-bin-win-vulkan-x64`). La mesura en WSL2 + Dozen es
conserva només com a comparació i com a limitació de l'entorn, no com a xifra representativa de la
targeta.

**Com es fa:** `llama-server.exe` s'executa a Windows lligat a `127.0.0.1`. El cortafuegs de Windows
no deixa que WSL hi arribi directament, de manera que s'obre un túnel SSH invers amb el client de
Windows (`ssh.exe -R`), que el deixa visible dins de WSL. L'avaluador (`lm-evaluation-harness`,
tasca `ifeval`) corre a WSL i parla amb aquell port. No s'ha modificat el cortafuegs.

**Què no es decideix aquí:**
- Jaull no passa a ser natiu de Windows: el seu pilot de qualitat exigeix Linux o WSL i
  bloqueja IFEval amb Vulkan a propòsit.
- El pont dins de Jaull (perquè el seu propi IFEval faci servir un servidor de Windows) **no és una
  decisió presa**: s'explora per separat, sense ser part d'aquesta decisió. Els resultats de la
  prova externa no s'han de confondre amb l'IFEval de Jaull.
- SYCL i OpenVINO queden fora d'abast. Hi ha xifres a articles externs que no s'han verificat.

**Efecte sobre Jaull:** el Run, Speed i Benchmark de Jaull a WSL2 no són representatius de la
velocitat real de la B60. És una limitació de l'entorn, no un defecte de l'estimador ni del
llançador: la mesura de Jaull (4,8 t/s) coincideix amb `llama-bench` a mà (4,10 t/s) al mateix entorn.

**Límits:** una sola màquina i una sola execució per model. A WSL2 es va fer servir flash attention
desactivada i a Windows el valor per defecte; la mesura estricta a Windows amb `-fa off` és
**PENDENT**. La prova d'IFEval és externa i té una sola execució per model.

**Fonts:** [comparativa](../b60-comparativa-wsl2-windows.md), [taula](../b60-taula-comparativa.md),
[registre de la prova d'IFEval](../ifeval-b60-prova.md), D098 i D120.

### D122. IFEval amb servidor Vulkan natiu de Windows i Jaull a WSL2

**Data i necessitat:** 2026-10-30. La comparativa de la B60 ja mesura la penalització de
WSL2 + Dozen (aproximadament 8x i 40x en generació) i la caiguda del 27B que no es produeix
a Windows, amb el mateix build i controlador. No es repeteixen ni es reinterpreten aquelles
mesures. Cal permetre que la inferència d'IFEval passi pel binari Windows, sense traslladar
Jaull ni l'avaluador a Windows.

**Decisió:** contracte `ifeval-external-vulkan-v1`, registre schema 4. Jaull continua a
WSL2; l'operador inicia `llama-server.exe` natiu mitjançant interoperabilitat WSL. El
contenidor només envia HTTP a l'endpoint loopback explícit que proporciona l'operador,
normalment un túnel SSH invers. No s'obre el tallafoc ni s'assumeixen ports o índexs Vulkan.
La primera topologia admesa exigeix Docker Engine Linux local al mateix WSL, amb xarxa host.
Docker Desktop, daemons remots/rootless i Jaull natiu a Windows queden fora d'abast.

**Identitat i garanties:** el protocol CUDA existent, la seva comanda i identitat schema 2
no canvien; una regressió fixa el digest sintètic i els arguments de llançament anteriors.
HellaSwag conserva `observable-runtime-v2`/schema 3. La ruta nova declara expressament
servidor `externally_supplied`, OS Windows, endpoint i digest del manifest. Dispositiu
confirmat amb l'emparellador D098, sense fallback a CPU, pressupost lliure/total observat
pel binari natiu abans de carregar, comprovació de cabuda, offload complet i assignació
model/compute al dispositiu en el log, SHA256/build del binari abans i després, model
rehashat abans/després i plantilla pròpia del GGUF. El camí Windows del model es comprova
contra el mateix fitxer accessible a WSL. Un pressupost desconegut o evidència contradictòria
impedeix importar el resultat. El servidor i el túnel són de l'operador: no es tanquen des
de Jaull, ni en cancel·lar. Només es neteja el contenidor propi.

**Límits:** evidència subministrada per un operador de confiança, no atestació remota del
procés. Cal una sessió dedicada, el manifest recent, la comanda generada i logs sense editar;
no es pot compartir el slot ni substituir el model. El pressupost de memòria és una
observació prèvia, no una reserva de VRAM. No s'ha canviat el rànquing. Es mantenen els
541 prompts, generació greedy, context 4096, límit 1280, reasoning i caché desactivats, i el
detector de llengua amb llavor: **és l'IFEval de Jaull, no comparable amb xifres publicades**.

**Estat:** IMPLEMENTED (proves offline); ruta física B60 UNVERIFIED, cap execució IFEval
d'aquest camí VALIDATED. Biosfer té Docker Desktop 4.95.0 (comprovat el 2026-10-30 amb
`docker version` al seu WSL2), i aquest contracte el refusa: el preparador exigeix un Engine
Linux local amb xarxa host (`docker_host_network()` retorna fals amb Docker Desktop). Amb la
topologia actual la prova física és, doncs, BLOCKED_BY_ENVIRONMENT: cal decidir amb l'autor si s'instal·la a WSL o es dissenya
una altra topologia, sense ampliar silenciosament l'abast a Jaull natiu de Windows.

**Fonts:** [comparativa B60](../b60-comparativa-wsl2-windows.md), D098,
[protocol i comandes de validació](../quality-evaluation-pilot.md),
[servidor extern](../../pilot/quality_eval/external_server.py),
[contracte](../../src/jaull/evaluation/quality_external.py).

## 14. Què s'ha substituït i no s'ha de redactar com a actual

| Decisió o simplificació anterior | Substitució o límit actual | Decisions |
|---|---|---|
| Jaull només inspecciona i mai executa | Execució i mesura explícites després de Search | D002, D070, D083 |
| Score ponderat com a ordre principal | Claus base per plans i etapa de política mesurada | D037, D057-D059, D064-D065 |
| Capability barreja mida, metadades i popularitat | Prior separat dels altres eixos i de qualitat mesurada | D038, D046 |
| Runtime no instal·lat elimina el model | Readiness governa accions, no ranking | D040, D069 |
| Sumar RAM i VRAM resol la col·locació | HFA amb pools i límits separats | D026-D028 |
| Tot pes és un bloc uniforme | Descomposició arquitectònica i launch tensor-exacte quan està auditat | D028, D030-D031 |
| Blocs HFA són `-ngl` | Unitats de planificació i unitats de llançament separades | D030 |
| `sliding_window` sempre limita KV | Comprovar activació i arquitectura | D022 |
| Batch de seqüències es tradueix a batch de tokens | No hi ha traducció automàtica | D034 |
| Reserva/marge s'assignen al procés | Comparació física els exclou | D025 |
| VRAM global o RSS offload equivalen a footprint del model | Fonts i portes metodològiques explícites | D077-D078 |
| Qualsevol evidència local supera una publicada | Pertinència, identitat i comparabilitat abans de procedència | D044-D046 |
| HellaSwag 100 dona qualitat general | Diagnòstic; IFEval full té un abast controlat diferent | D048-D050 |
| Dues execucions iguals de qualitat sempre són conflicte | Deduplicar contingut equivalent, abstenir-se si discrepa | D054 |
| «IFEval full» basta per aplicar | Pins auditats, mostres i subjecte exacte | D055 |
| Speed aplicable pot entrar a la clau base | Només política mesurada sobre grups permesos | D064 |
| Guanya l'última crida de velocitat | Agregació local per cohort amb mínim de crides | D066 |
| Bandes connexes de soroll són la revisió definitiva | Substituïdes al codi local per precedències per parelles, pendent de validar | D067-D068 |
| Recompare torna a buscar o canvia l'original | Mateix pool, nova evidència local i proposta separada | D084 |

## 15. Guia per redactar la memòria

| Capítol possible | Decisions i material que el fonamenten |
|---|---|
| Problema, objectius i abast | D001-D006, D091-D095; què qualifica Jaull i què correspon a Biosfer |
| Anàlisi i requisits | D013, D035-D043; usuaris, task, constraints, maquinari i diversitat |
| Disseny i arquitectura | D007-D012, D017-D019, D069-D074; contractes, fronteres i flux d'execució |
| Model d'estimació | D020-D034; fórmules, fonts, col·locació, conservadorisme i incertesa |
| Recomanació amb evidència | D044-D068; protocols, cohorts, shadow, activació i fallback |
| Implementació i UX | D080-D090; accessibilitat del flux, estat, permisos, memòria cau i proves |
| Metodologia experimental | D071-D079 i secció 12; predicció congelada, repeticions, portes i registres |
| Resultats i discussió | Secció 12 i D092-D094; resultats per artefacte i límits de generalització |
| Conclusions i treball futur | D091-D095; què s'ha completat i què queda explícitament pendent |

Tres afirmacions centrals que el treball pot defensar, si s'acompanyen dels casos:

1. Jaull **separa estimació, decisió i observació**, i fa visible la procedència
   i la incertesa en lloc de substituir-les per una puntuació global.
2. La recomanació és **reproduïble sobre inputs i evidència congelats**, i
   només reordena amb mesures quan identitat, pertinència i protocol ho permeten.
3. Els experiments serveixen per **contrastar i limitar afirmacions**, no
   per reescriure prediccions, triar només runs favorables o demostrar superioritat
   universal dels models seleccionats.

Evitar afirmacions més fortes que l'evidència: «prediu velocitat», «sempre evita
OOM», «suporta qualsevol GPU», «qualitat general mesurada», «equivalència
estadística», «tria els cinc millors models de Hugging Face» o «Biosfer ja integrat».

## 16. Índex de fonts i versions

### 16.1. Documents i codi

- [README](../../README.md), [arquitectura canònica](../../ARCHITECTURE.md)
  i [estat](../../STATUS.md): descripció i límits del producte.
- [Diari de desenvolupament](../JAULL_DEVLOG.md): origen, bugs que van
  motivar decisions i resultats; les darreres línies poden descriure una fase anterior.
- [Estimació](../estimation.md), [HFA/llmfit](../hardware-fit-vs-llmfit.md),
  [recomanació](../recommendation.md), [evidència](../evidence.md),
  [limitacions](../limitations.md) i [UI](../ui.md): contractes actuals i context.
- [Pla vFinal](../recommendation-vfinal-plan.md),
  [pla Fastest/Balanced](../fastest-balanced-plan.md) i
  [esborrany anterior de política](../politica-calidad-velocidad-equilibrio.md):
  decisions successives, no tres especificacions simultàniament vigents.
- [Catàleg publicat](../model-capability-evidence.md),
  [pilot inicial](../quality-evaluation-pilot.md),
  [diari històric del pilot](../history/quality-pilot-journal-2026-10.md):
  atribució, comparació i evolució del protocol de qualitat.
- [Protocol Biosfer](../experimental-protocol-biosfer.md) i
  [validació entre màquines](../experimental-validation.md): metodologia
  i propostes de campanya; no prova de totes les execucions proposades.
- [Directori de proves experimentals](../../validation),
  [informes Qwen](../qwen2.5-tests) i
  [campanya de soroll](../qwen2.5-tests/rtx2060-tg128-noise-20261011.md):
  suport empíric; conservar originals i revisar el manifest de cada cas.
- [Motor base](../../src/jaull/recommendation/engine_v2.py),
  [qualitat](../../src/jaull/recommendation/quality.py),
  [velocitat](../../src/jaull/recommendation/local_evidence.py) i
  [polítiques compartides](../../src/jaull/recommendation/shadow.py):
  autoritat per contrastar l'estat executable, amb atenció al dirty worktree.

### 16.2. Commits de referència

Els identificadors curts de les decisions es resolen a l'historial local.
Aquests punts d'entrada faciliten la consulta a GitHub:

| Commit | Per què consultar-lo |
|---|---|
| [326068f](https://github.com/Ton-Llop/Jaull/commit/326068f0c5c7329638e6faaa3030f662177e98ee) | Abast i arquitectura inicials abans de l'execució real |
| [b347466](https://github.com/Ton-Llop/Jaull/commit/b347466) | Separació inicial de workflow, discovery i recommendation |
| [452df33](https://github.com/Ton-Llop/Jaull/commit/452df33) | Correcció del ranking i de la interpretació d'evidència |
| [67b85b0](https://github.com/Ton-Llop/Jaull/commit/67b85b0) | Desacoblament de ranking i readiness |
| [8fea9e9](https://github.com/Ton-Llop/Jaull/commit/8fea9e9) | Hardware Fit com a anàlisi de col·locació |
| [9918cea](https://github.com/Ton-Llop/Jaull/commit/9918cea) | Llançament local GGUF amb descriptors de tensors |
| [3a6dc26](https://github.com/Ton-Llop/Jaull/commit/3a6dc26) | Observació de buffers reportats pel runtime |
| [89a30c7](https://github.com/Ton-Llop/Jaull/commit/89a30c7) | Capability deixa de duplicar altres dimensions |
| [1d12f63](https://github.com/Ton-Llop/Jaull/commit/1d12f63) | Correcció de batch de seqüències versus tokens |
| [89db042](https://github.com/Ton-Llop/Jaull/commit/89db042) | Catàleg d'avaluacions publicades com a referències |
| [8a69267](https://github.com/Ton-Llop/Jaull/commit/8a69267) | Pilot Docker de qualitat GGUF |
| [7025bb3](https://github.com/Ton-Llop/Jaull/commit/7025bb3) | IFEval, flux d'avaluació i activació de Quality |
| [491b6fc](https://github.com/Ton-Llop/Jaull/commit/491b6fc) | v2, Fastest/Balanced i avaluació per prioritat |
| [5ef3635](https://github.com/Ton-Llop/Jaull/commit/5ef36354794492798efad713a7feb7ff353d5616) | Últim commit revisat en aquesta redacció |

### 16.3. Com mantenir aquest registre

Conservar els IDs D001-D122 perquè es puguin citar a la memòria. Quan canviï
una decisió, no esborrar la motivació antiga: marcar-la com a substituïda,
enllaçar la nova i afegir data, commit, efecte i evidència. Afegir nous IDs
sense renumerar els existents.

Abans d'utilitzar aquest text com a estat final del TFG, fixar el commit de la
campanya final (D068 ja és a `330bf0f`) i comprovar quins pendents de D091-D095 s'han completat. **Aquest document
és una reconstrucció traçable fins a la data de tall, no un certificat de
validació ni un substitut dels registres originals.**
