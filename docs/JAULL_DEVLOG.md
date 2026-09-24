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

---

## Ara mateix

La política de llançament ja no és el problema: mesurada, costa 1,4x–2,1x i el que compra és marge, i moure'l demanaria una campanya de calibratge amb repeticions i més d'una màquina.

El que bloqueja de debò és la traducció entre els blocs de transformer que compta l'HFA i les unitats de `--n-gpu-layers` que fa servir llama.cpp. Sense això la comparació per component no es pot publicar, encara que ja tingui les dues meitats: la predicció i, des del 17/09, la mesura.

I una cosa pendent que és meva: la fontaneria del contracte d'observació està provada i no l'ha fet servir cap campanya. Les mesures del 20 i el 24 es munten els seus propis JSON en comptes de passar per `ExperimentRequest`, així que encara no hi ha ni un sol record amb la mesura a dins. Convertir això en evidència és el següent pas.
