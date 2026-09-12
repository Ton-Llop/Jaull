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

---

## Ara mateix

Estic muntant el **manifest de casos experimentals**: lligar un experiment amb els seus benchmarks i els logs originals com una sola cosa, sense duplicar cap mesura. La idea és poder comparar la 2060 amb la 4060 sense barrejar evidència de runs diferents i sabent en tot moment què falta de cada cas.
