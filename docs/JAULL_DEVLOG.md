# Jaull — DevLog

> Resum del que he anat fent. Les dates del principi són aproximades perquè
> encara no portava cap seguiment diari.

---

## 29/07 — La idea

Se m'acut una eina que et miri el PC i et digui quins models d'IA local hi pots fer anar de debò. La idea surt del TFG i de la pregunta pràctica de quin hardware necessitaria una empresa per tenir models locals. De moment es diu `local-ai-checker`.

## 30/07 — Detecció de hardware

Detecció automàtica de CPU, RAM, disc i GPU. Per les NVIDIA tiro de NVML, que és qui et diu la VRAM total i la lliure.

## 31/07 — Hugging Face

Connecto amb el Hub per buscar models i llegir-ne la metadata sense descarregar res. Aquí ja es veu que un mateix model «lògic» existeix en un munt de repos i formats diferents, cosa que després donarà bastanta feina.

## 01/08 — Estimació de memòria

La part central: calcular si un model cap. Tinc en compte pesos, precisió/quantització, KV cache, mida de context, overhead del runtime i marge de seguretat, i classifico el resultat com a *comfortable*, *compatible*, *tight*, *offloading required* o *insufficient*. La norma que em poso és que cada número ha de dir d'on surt, res de constants màgiques.

## 02/08 — Recomanació

Primer commit de l'estructura i primera versió del ranking: Jaull ja no només diu si un model cap, sinó quins són els millors. Pondero hardware, memòria, ús que li vols donar, idioma, llicència, format i si realment es podrà executar.

## 03/08 — Workflow guiat i canvi de nom

Munto el flux de preguntes (chat general, programació, documents; prioritat entre qualitat, velocitat i memòria; context; concurrència) i et busca els millors matchs. Canvio el nom a **Jaull**. El projecte comença a créixer i faig el primer refactor seriós per separar `workflow`, `discovery`, `recommendation` i `presentation`, amb `AdvisorService` com a façana perquè la CLI i la TUI no hagin de saber què passa per dins.

## 04/08 — TUI

Em poso de debò amb la interfície en Textual. L'objectiu és que l'usuari no hagi de saber què és GGUF, una quantització o la VRAM per obtenir una recomanació. També faig l'`ArtifactService`: resoldre el fitxer concret, descarregar-lo i verificar-lo.

## 05–07/08 — Que no recomani coses que no pots executar

Provant amb hardware real veig que models AWQ sortien com a bona opció encara que després no els poguessis executar amb el runtime que tens. Afegeixo informació d'executabilitat real, formats, quantitzacions, nombre de paràmetres i confiança de l'estimació. La descàrrega passa a ser real, amb verificació de mida i SHA-256.

## 09/08 — Primer intent amb llama.cpp

Intento compilar `llama.cpp` al portàtil i peta: WSL es queda sense memòria i l'OOM killer es carrega el `cc1plus`. Decideixo no insistir i preparar la prova al PC amb NVIDIA. Irònicament és un bon recordatori del problema que Jaull intenta resoldre.

## 10/08 — Primera inferència real

Compilo `llama-cli` amb CUDA al PC bo, descarrego TinyLlama des del mateix Jaull i l'executo. La primera resposta és per emmarcar, perquè el TinyLlama decideix que GGUF és una mena de competició d'MMA, però tècnicament funciona. També descobreixo que `llama-cli` es queda en mode interactiu després de generar i li he d'afegir `--single-turn`.

El mateix dia tanco el primer end-to-end real:

```text
Hugging Face → ArtifactService → GGUF verificat → AdvisorService
    → LlamaCppRunner → ExecutionBackend → llama-cli → CUDA → text
```

Aquest és dels milestones importants: Jaull deixa de només predir i passa a executar. De passada unifico l'execució, perquè la CLI s'estava muntant el seu propi runner pel seu compte en comptes de passar per l'`AdvisorService`.

## 11/08 — La TUI ja fa tot el recorregut

Integro el flux sencer dins la TUI: hardware → necessitats → recomanacions → triar model → descarregar i verificar → escriure prompt → executar → veure la resposta. Afegeixo prompts consecutius amb historial, i em barallo amb uns quants errors de Textual (`DuplicateIds` i el lifecycle dels workers) que bloquejaven els tests.

## 12/08 — Predicció contra realitat

Començo a mesurar què passa de veritat quan s'executa (durada, RAM i VRAM de pic) i a comparar-ho amb el que Jaull havia predit. Aquesta és la meitat del TFG que més m'interessa. També amplio la detecció a Vulkan, no només CUDA, que al final són les «APIs» que connecten llama.cpp amb la teva gràfica.

## 13/08 — Experiments reproduïbles

Munto el pipeline d'experiments: cada validació guarda un registre immutable amb el hardware, l'artefacte, la predicció congelada i el que va passar realment. I unes quantes hores barallant-me amb el CI de Windows per tres xorrades.

## 14–15/08 — Benchmarks

Benchmarks reals des de la TUI amb `llama-bench`, i ara també detecto Transformers, així que puc promptejar, validar i benchmarkejar per les dues vies. Els registres es guarden en JSON; ja m'està bé, res de SQLite de moment.

## 17/08 — Rendiment i Top 5

Cache d'anàlisi de models i discovery concurrent, perquè cada cerca repetia massa feina. Ara la recomanació dona 5 opcions i les diferencia millor entre elles en comptes de treure cinc variants del mateix.

## 20–22/08 — Endreçar la casa

Documento tot perquè el codi es pugui obrir. Arreglo problemes de correctitud del ranking (identitat del model, evidència local) i poso fronteres d'arquitectura explícites amb un test que les vigila llegint els imports. També deslligo el ranking de si tens el runtime instal·lat: que no tinguis `llama-cli` ha de canviar si pots executar-ho ara, no si el model és bona idea per la teva màquina.

## 24–25/08 — HardwareFit Analyzer

La part gran d'aquest tram. En comptes de sumar RAM i VRAM com si fossin la mateixa bossa, decideixo una col·locació: `GPU_RESIDENT`, `GPU_OFFLOAD`, `CPU_RAM` o `TOO_LARGE`. Arreglo la semàntica del sliding window al KV cache (que no és el mateix tenir el camp que tenir-lo actiu) i munto un harness per validar les prediccions contra llama.cpp de veritat, no contra la meva intuïció.

## 27–28/08 — Shortlist conscient del hardware

La shortlist ja filtra tenint en compte el HardwareFit, així que deixa d'omplir el Top 5 amb coses que no caben. I taurons a la TUI 🦈

## 30–31/08 — El HFA era massa prudent

Descobreixo que el HardwareFit és bastant conservador. No ho «arreglo» a base d'ajustar constants fins que quadri: el que faig és que et diagnostiqui **per què** surten 18 blocs i no 19, i quant falta per arribar-hi. També millora on col·loca el KV cache quan hi ha offload.

## 02/09 — Codecov

Cobertura al CI perquè es vegi què està realment provat.

## 03/09 — Un sol camí d'execució

Unifico la planificació d'execució, que estava escampada entre la CLI, la TUI i el motor de recomanació. Ara tot el que s'executa passa per un únic lloc que garanteix que els flags estan complets.

## 09–10/09 — Estimador més estricte

Endureixo la correctitud de l'estimador i el matching de l'evidència local (que un benchmark d'una altra màquina o d'una altra quantització no compti com a prova). Afegeixo límits al placement dels pesos que no són blocs, és a dir els embeddings i el cap de sortida, que mai es mouen igual que les capes.

## 10–11/09 — Manifest de casos experimentals

Lligo un experiment amb els seus benchmarks i els logs originals com una sola cosa, sense duplicar cap mesura, i el manifest diu si el cas és `valid`, `partial` o `inconsistent` amb el motiu concret. De passada tanco tres forats que arrossegava: ara es guarden el commit de git, el comando executat i el backend observat.

## 13/09 — Bundles i primer baseline

Els casos s'exporten a bundles portables, i faig el primer baseline de debò amb Qwen2.5-7B a la 2060. Surt incòmode i per això val la pena: la política automàtica dona 21,5 tok/s i demanar-ho tot a mà en dona 59,4, per estalviar 133 MiB.

## 13/09 — La política regalava rendiment

La fórmula tenia tres biaixos cap al mateix costat: dividia tots els pesos entre els blocs, restava el KV sencer, i no podia arribar mai a «totes». Corregit i tornat a mesurar, de 20,6 a 44,9 tok/s.

## 13/09 — Per què no puc comparar la VRAM

Ho donava per pendent i resulta que no és cosa meva: amb la GPU en mode WDDM, que és tota GeForce amb pantalla, NVML no atribueix memòria per procés — comprovat des de Windows i des de WSL2. I el motiu que el codi donava per la RAM era fals: l'obstacle és el `mmap`, que fa que el RSS no es mogui encara que canviï la col·locació.

## 14/09 — Llegir el GGUF en comptes d'endevinar-lo

Estimava els bytes dels embeddings i del cap de sortida projectant el config, que no prova res sobre l'artefacte real; ara llegeixo el tensor table. La comprovació que més m'agrada: els descriptors reprodueixen els buffers que llama.cpp va registrar fa un mes amb 0,01 MiB d'error.

## 16/09 — El tier sempre deia el mateix

Miro una captura de resultats a la 4060 i totes les recomanacions porten «BEST-EFFORT SUGGESTION». Resulta que de quatre capçaleres només n'hi havia una d'assolible: qualsevol confiança LOW hi anava directa, i com que l'overhead és sempre una heurística assumida, tota estimació és LOW. Ara LOW limita a RECOMMENDED, que és el que vol dir de debò: hi ha una suposició documentada a dins, no que no sàpiga on va el model.

## 16/09 — Capability mesurava popularitat

A la mateixa captura hi ha tres models de menys de 1.5B per davant d'un 4B. El culpable: el 55 % de `capability_score` eren metadata i descàrregues, que ja tenien pes propi al ranking. A Qwen3-4B li falta la línia `language:` al model card i això li costava més (-0,123) del que li donaven 3400 milions de paràmetres de més (+0,106). Trec els dos termes duplicats i el 4B passa del cinquè lloc al segon.

## 16/09 — Un test que miri el Top 5 sencer

El bug anterior va passar per davant de 1600 tests perquè cap comparava més de dos candidats a la vegada. Cada regla era correcta per parelles; el que estava malament era el resultat conjunt. Afegeixo un test amb sis candidats que fixa l'ordre, i el comprovo restaurant la fórmula vella per veure que falla de debò.

## 17/09 — int4 s'estimava d'una manera i s'executava d'una altra

El pitjor dels trobats. Per int4 i int8 Jaull emetia `torch_dtype=torch.int8`, que no quantitza res, i els dos workers el passaven al carregador tal qual: carregaven pesos sense quantitzar i els reportaven com si fossin la configuració predita. Un experiment hauria comparat una predicció de 0,5 bytes per paràmetre contra una execució que no ho era. Ara la precisió viatja en un sol flag i el worker construeix el `BitsAndBytesConfig`; sense `bitsandbytes` es nega amb un motiu explícit en comptes de degradar en silenci. Cap mesura anterior estava contaminada: tot el que hi ha mesurat és llama.cpp.

## 17/09 — El report es contradeia a si mateix

El mateix candidat sortia dues vegades al mateix fitxer amb penalitzacions diferents, i el número que s'imprimia al costat del rank no era el que ordenava — el model amb més puntuació sortia quart. Ara es publiquen els eixos que decideixen de veritat, i el compost queda etiquetat com a diagnòstic. De passada arreglo tres frases que deien coses falses, com culpar el model card de la confiança baixa quan la causa és l'overhead heurístic.

## 17/09 — El contracte d'observació no s'arribava a disparar

Torno a executar el B001 per veure el primer error per component i em surt `runtime_allocation: None`. El runner demanava el log de llama.cpp només si un flag ho deia, i ningú el posa: sense ell aquest build no escriu ni una línia a stderr. Tot el contracte era inabastable des del camí que executa models, i cap test ho veia perquè tots donaven al parser un log ja gravat. Corregit, i a la segona: 4124,91 MiB de buffers reals a dispositiu. La comparació encara no dona número, però ara per la raó bona.

## 20/09 — Matriu de contextos a la 2060

Executo el mateix model a 512, 2048 i 4096 de context per veure si l'estimador es comporta de manera coherent quan creix el KV. Ho fa: com més context, menys blocs a GPU. Les quatre execucions arrenquen i els percentatges d'error surten buits amb una nota explicant per què no es calculen, que és el que toca.

## 24/09 — Quant costa ser prudent

La matriu anterior mesurava que tot arrencava però no a quina velocitat, i llegida així semblava que Jaull regalava 3–4x de rendiment. No és cert: el recompte de blocs de l'HFA no és el que Jaull llança. Mesuro el nivell que emet la política de debò i el cost real és **1,4x–2,1x**. El que compra a canvi és marge — amb tot a GPU la targeta es queda amb 196 MiB lliures de 6144, per sota del que ja hi ha ocupat de base. No toco cap constant: tres mesures d'una sola repetició en una màquina no són base per moure un marge de seguretat.

## 30/09 — Endurir l'evidència

Tanco errors que podien perdre un intent fallit o acceptar evidència incorrecta: els runners conserven les sortides malformades, els stores rebutgen IDs contradictoris i els SHA coneguts no es poden substituir per un sidecar. Els bundles tornen a verificar els fitxers després de copiar-los i les referències no poden escapar de l'arrel per un symlink.

Corregeixo el timing GPU de Transformers sincronitzant el treball i separant prefill de decode: metodologia v3, sense reescriure els records v2. Els benchmarks fallits ja no desplacen els exitosos i Jaull no atribueix el commit d'un repositori pare. 1906 tests, Ruff i mypy verds; cap canvi a l'HFA, al ranking ni a les constants de memòria.

## 01/10 — Pilot de qualitat amb Docker

Executo `lm-evaluation-harness` en un contenidor contra `llama-server`, amb versions fixades i els GGUF exactes de TinyLlama i Qwen. Guardo les peticions HTTP i comprovo que els scores coincideixin amb els logprobs reals. El replay amb TinyLlama mostra que canviar el repartiment CPU/GPU pot canviar els scores: el placement també ha de formar part del protocol. Encara és un pilot separat de la TUI i del ranking.

## 02/10 — De tres mostres a cent

Repeteixo HellaSwag amb 100 exemples fixos i els dos models en seqüència a la 2060. TinyLlama encerta 36/100, o 38 amb normalització; Qwen, 46/100 i 63. La comparació passa els gates, però continua sent un subset d'un sol benchmark, no una mesura de qualitat general. El reinici del PC deixa un intent corrupte: el rebutjo i el conservo, sense reconstruir resultats. 2060 tests, Ruff i mypy verds.

## 03/10 — Qualitat i velocitat no són el mateix

Mesuro `pp512` i `tg128` amb `llama-bench`, cinc repeticions, quatre threads i tot a CUDA0. TinyLlama genera 190,73 tok/s i Qwen 123,94: en aquestes proves, un dona més velocitat i l'altre més encerts. Guardo les mostres, els flags efectius i els records immutables. És throughput amb la GPU compartida amb l'escriptori, no latència fins al primer token. 28 tests de benchmark/arquitectura, Ruff i mypy verds; cap canvi al ranking ni a l'HFA.

## 04/10 — Avaluar des de la interfície

Porto el pilot dins la TUI: una acció explícita que prepara i llança l'avaluació d'un GGUF concret, amb preflight de Docker, runtime i artefacte. Res s'executa sol en obrir Jaull — una avaluació es dispara perquè algú la demana, no perquè s'hagi carregat una pantalla.

## 05/10 — Separar el que diu l'editor del que hem mesurat

La pestanya d'avaluació passa a ensenyar dues coses que fins ara es confonien: el que un publisher ha publicat sobre el model, i el que hem mesurat nosaltres sobre l'artefacte exacte. Cada bloc amb la seva etiqueta, i el publicat avisa que és «no mesurat en aquest artefacte» i en quin mode es va avaluar.

També reordeno Hardware i Your needs. La RAM i la VRAM passen a ser xifres grans de set segments, perquè tot el que diu Jaull és una afirmació sobre aquests dos números. El wizard passa de sis preguntes en una columna —amb el botó d'enviar sempre fora de pantalla— a dues columnes alineades: el bloc baixa de 37 files a 24 i el botó puja de la fila 41 a la 28. Per sota de 100 columnes es plega a una.

## 07/10 — El catàleg l'heretava tothom

Enganxo el catàleg de resultats publicats al ranking i descobreixo que `engine_v2` copiava la llista sencera a tots els plans sense filtrar. Tal com estava, l'MMLU de Qwen hauria aparegut sota un candidat de Mistral. Ara cada pla només rep l'evidència que anomena el seu model.

Tres errors més pel camí, tots trobats revisant-ho:

L'exportació perdia l'atribució. `PlanAssessment.external_evaluations` està tipat amb la classe base, i pydantic serialitzava **19 camps com a 10**: desapareixien `repo_id`, `variant`, `source_kind`, `protocol` i la precisió. Una puntuació publicada sense dir de qui és, és pitjor que no dir-ne res.

El linatge es confirmava pel tipus d'evidència, no pel repositori. Metadata que apuntava a `org/Other` donava per bo un match contra `org/Target`, i canviar l'ordre de dues entrades canviava el resultat.

I quan ho vaig arreglar comparant amb `logical_model_repo_key`, vaig tornar a colar la heurística per la porta del darrere: aquesta clau retalla sufixos, així que metadata que declara `org/Thing-7B-GGUF` confirmava `org/Thing-7B`. Ara la comparació exigeix el nom exacte, només plegant majúscules.

## 07/10 — Dotze candidats reals

Em pensava que els repacks típics no portarien `base_model` a la metadata i que la regla estricta ens deixaria sense cobertura. Ho mesuro amb una cerca real: 6 consultes, 40 candidats, auditoria dels 12 primers.

De linatge: **7 amb base declarada, 2 repositori directe, 3 sense linatge i cap per heurística de nom**. O sea que la regla estricta no costa cobertura. El coll d'ampolla és un altre: **8 dels 12 no tenen entrada al catàleg** i només 1 passa sencer. `bartowski/Qwen2.5-32B-Instruct-GGUF` hereta l'MMLU-Redux de Qwen2.5 per metadata declarada, que és exactament el cas que volia permetre.

Escric la política de Quality, Fastest i Balanced com a esborrany per revisar, sense implementar-ne res. Balanced queda com a porta sobre `min_generation_tps` i no com a pesos: sumar qualitat i velocitat en un número tornaria a muntar el score global que vaig treure a propòsit.

## 07/10 — Un test que només fallava a Windows

CI en vermell amb un sol test, i només a 80×24 sense mesura. El botó sortia a la fila 33 amb un viewport de set files començant a la 13. Traço la geometria pas a pas i la causa no era el contingut: **`scroll_visible` és diferit**, no mou l'offset a la crida sinó en un refresc posterior. La fila que vaig mesurar just després de la crida és exactament la que reportava CI.

El test comptava refrescos —un `pause` i a córrer— i això codifica el scheduling d'una màquina. Ara espera la condició que li importa de debò: que el botó estigui dins del viewport. 32 execucions seguides sense fallar. 2289 tests, Ruff i mypy verds.

## 08/10 — Un IFEval que es pot repetir

HellaSwag mesura si un model tria la continuació bona; per a un xat el que importa és si segueix instruccions. Monto IFEval amb un contracte tancat: `lm-eval` fixat, `llama-server` amb la plantilla del mateix GGUF i el raonament apagat, context 4096, fins a 1280 tokens i llavor fixa. Cada registre guarda el SHA exacte de l'artefacte i cada resposta.

Faig les 541 preguntes amb Qwen2.5-1.5B: 222/541 en estricte (41,0 %). TinyLlama queda fora pel camí: amb context 2048 no hi cap el protocol, i ara la selecció ho rebutja abans de descarregar res. Les polítiques Quality/Fastest/Balanced passen d'esborrany a codi, però en **shadow**: calculen una proposta al costat del ranking i no el toquen.

## 09/10 — Mateix contracte, mateixa resposta

Repeteixo sense voler el Qwen2.5-0.5B Q5_K_M i surt el millor resultat del dia: **541 de 541 respostes idèntiques**. El contracte és determinista de debò. A partir d'aquí, dues execucions amb les mateixes respostes compten com una, i si mai discrepen no se n'aplica cap.

LFM2.5-1.2B fa 434/541 (80,2 %), però el primer intent no es va poder importar: llama.cpp talla per bytes una línia del seu tokenitzador i deixa un caràcter UTF-8 a mitges al log. Els logs ara toleren aquests bytes; el que es valida segueix sent estricte.

La pantalla d'avaluació es configura sola: troba la imatge Docker que toca, comprova servidor i dataset, i ho pinta amb ✓ ! ✗. Results posa els models costat a costat i només marca guanyador si els protocols són comparables. Afegeixo una pantalla per alliberar espai, perquè els models i les execucions ja ocupaven gairebé 5 GB.

Recompare amb un parell controlat: Qwen2.5-1.5B (41,0 %) puja del #2 al #1 per sobre del 0.5B (25,9 %), mateixa cohort i mateix estrat. LFM2.5 no s'hi ordena perquè la seva llicència el posa en un altre estrat, que és el que toca. Però la matriu de 27 casos amb cerques normals dona **zero moviments**: amb prioritat Quality el pool no conté cap dels GGUF mesurats, i els models visibles solen ser safetensors sense SHA. El camí funciona; la cobertura és gairebé nul·la.

De passada, descobreixo que els tests llegien les meves dades reals de `~/.local/share/jaull`: amb 300 MB de registres, les captures passaven de 17 s a 55 s. Ara cada test té les seves carpetes.

## 10/10 — Quality activat

Activo Quality amb el fallback a la vista. Només intercanvia plans amb qualitat mesurada, comparables i del mateix estrat; sense evidència, l'ordre és exactament el d'abans. La cerca continua desant l'ordre base, i l'informe separa els criteris base del pas de qualitat, amb la mesura, el SHA i les posicions abans i després.

Speed i Balanced es queden en shadow: Balanced necessita qualitat **i** velocitat, i encara no hi ha cap mesura de velocitat aplicable. Els bloquejos de la suite dins del sandbox de Codex eren del sandbox: un exemple mínim sense Jaull es queda igual, i fora passa tot.

---

## Ara mateix

Quality ja ordena amb qualitat mesurada, però en una cerca normal gairebé mai té res a aplicar: la cobertura de GGUF mesurats dins del pool és el coll d'ampolla, no la regla. El parell controlat demostra que el camí funciona amb dades reals.

Speed i Balanced segueixen en shadow fins que la velocitat sigui aplicable. El que ho bloqueja és el batching efectiu de `llama-bench`, i és el següent pas: Balanced és la prioritat per defecte i avui no pot fer servir res de tot això.

Queda pendent confirmar la CI de Linux i Windows. Per a la campanya final: ampliar la cobertura GGUF del pool i comparar configuracions del mateix model (BF16/Q8/Q5/Q4) en velocitat, memòria i qualitat.
