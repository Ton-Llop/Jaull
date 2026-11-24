# Predicció de Jaull contra realitat: els casos que ja existeixen

Revisió del 2026-11-23, feta només amb els documents del projecte. No s'ha executat cap prova nova ni
s'ha tocat cap fórmula. És el primer pas de l'avaluació d'eficàcia de Jaull: abans de decidir cap prova
nova, mirar què diuen les que ja es van fer.

Cada xifra porta la font i una etiqueta. **MESURAT** és una lectura de llama.cpp o d'una execució.
**CALCULAT** és la fórmula de Jaull aplicada a mà, sense passar per l'estimador complet (com a D120 i D123).
**UNVERIFIED** és el que no s'ha comprovat.

## Què vol dir «viable» aquí

No hi ha un únic criteri, i la diferència és gran. Hi ha tres nivells, de menys a més exigent:

1. **La regla de llama.cpp.** Accepta una càrrega si, després de col·locar pesos, KV i còmput, queden almenys
   1.024 MiB lliures a la targeta (`will leave N >= 1024 MiB of free device memory`).
2. **Carrega.** El model s'arrenca amb totes les capes a la GPU, encara que la regla no es compleixi (amb `-ngl` fix).
3. **Genera.** A més, acaba una resposta sense errors de memòria.

La major part de les mesures d'aquest document són del nivell 1 o 2. **No s'ha observat cap cas en què un model
que Jaull donava per viable fallés per memòria durant la inferència** (D120, D123: «no s'ha mesurat cap cas que
fallés per memòria»). Això no vol dir que no pugui passar: vol dir que no s'ha provat prou a prop del límit.

## 1. B60 (24 GiB, 23.613 MiB lliures): s'hi cap o no?

llama.cpp b11889, Vulkan natiu a Windows. «Jaull aleshores» és la predicció abans del canvi indicat. «Jaull ara»
és la fórmula actual (pesos + KV + 256 MiB + 1 GiB).

| # | Model i context | Jaull aleshores | Jaull ara (CALCULAT) | llama.cpp (MESURAT) | Resultat |
|---|---|---|---|---|---|
| B1 | Qwen3-Coder 30B Q4_K_M, 4096 | 22,42 GiB, 97 % de la lliure (abans de D120) | 18,91 GiB, `compatible` | 17,57 GiB, 5,6 GiB lliures | Encert, sobreestimat |
| B2 | Qwen3-Coder 30B Q4_K_M, 32768 | **23,41 GiB, `insufficient`** (després del pas 1 de D120) | 21,53 GiB, `tight` | 20,23 GiB, **2.901 MiB lliures** | **Fals negatiu** (corregit a D123) |
| B3 | Qwen3.8-27B Q4_K_M, 4096 | 20,77 GiB, `tight` (abans de D120) | 17,59 GiB | 15,32 GiB | Encert, sobreestimat |
| B4 | Ornith 1.5 9B Q8_0, 4096 | 12,70 GiB (abans de D120) | 10,87 GiB | 8,29 GiB | Encert, sobreestimat |
| B5 | Qwen2.5 32B Q4_K_M, 4096 | 22,54 GiB, `tight` (pas 1 de D120) | 20,73 GiB, `compatible` | 19,28 GiB, 3.875 MiB lliures | Encert, sobreestimat |
| B6 | Qwen2.5 32B Q4_K_M, 16384 | 25,83 GiB, `offload` (abans de D123) | 23,73 GiB, `offload` | 22,29 GiB, **791 MiB lliures** | Coincideix amb llama.cpp (vegeu sota) |
| B7 | Qwen3.6-35B-A3B IQ4_XS, 32768 | 20,3 GiB (abans de D125) | 18,45 GiB | 16,79 GiB (17.194 MiB) | Encert, sobreestimat |
| B8 | Qwen3.6-35B-A3B IQ4_XS, 131072 | **27,8 GiB, no cap** (abans de D125) | 20,32 GiB | 18,76 GiB (19.210 MiB) | **Fals negatiu** (corregit a D125) |
| B9 | Qwen3.6-35B-A3B IQ4_XS, 262144 | **37,8 GiB, no cap** (abans de D125) | 22,82 GiB, `tight` | 21,38 GiB (21.898 MiB), 1.715 MiB lliures | **Fals negatiu** (corregit a D125) |

Fonts: B1, B3, B4 a D120; B2 a D120 (mesura a 32K) i D123; B5, B6 a D123; B7 a B9 a D125. Els valors «ara» de B3 i B4
són els de la fórmula actual aplicada al fitxer (`measured.ts` de la web). Els de B1, B2, B5, B6 i B7 a B9 són els
que citen D123 i D125.

**B6 és un cas a part.** llama.cpp diu `cannot meet free memory target of 1024 MiB, need to reduce device memory by
232 MiB`. Com que `-ngl 99` el fixa l'usuari, carrega les 65 capes i en queden 791 MiB, per sota del seu llindar.
Jaull diu `offload`, que és el que llama.cpp faria sol (treure una capa o dues). Per la regla de llama.cpp,
coincideixen. Que la inferència acabi amb aquest marge és **UNVERIFIED**.

### Què en surt

- **Falsos positius: cap** en aquests nou casos. Cap model que Jaull donés per viable ha deixat de carregar.
- **Falsos negatius: tres**, tots amb causa coneguda i ja corregits: B2 (overhead massa gran, D120/D123) i B8 i B9
  (KV d'un model híbrid calculat quatre vegades massa gran, D125).
- **El biaix que queda és conservador**: la fórmula actual queda entre 1,3 i 2,6 GiB per sobre de la mesura. Hi ha
  dues causes documentades: l'1 GiB de marge que Jaull deixa a propòsit (la regla de llama.cpp) i que els pesos
  són la mida del fitxer, quan llama.cpp en deixa de 167 a 1.030 MiB al host (embeddings).
- **Compte amb la lectura de les xifres «ara».** Es van ajustar després de veure aquestes mateixes mesures (D120 i D123
  són calibratges, amb «mesura de base en quatre models i una sola GPU»). Per això no són una validació: sobre
  aquests casos no poden fallar. La part que sí és una predicció de veritat és la de **B7 a B9**: les prediccions del
  KV i de l'estat recurrent es van desar abans de mesurar (D125) i van coincidir exactament (640, 2.560 i 5.120 MiB).

## 2. Components: l'error és al KV o als pesos?

| Component | Cas | Predit | Mesurat | Etiqueta |
|---|---|---|---|---|
| KV (atenció híbrida) | Qwen3.6-35B-A3B a 32K, 128K i 256K | 640, 2.560, 5.120 MiB | 640, 2.560, 5.120 MiB | Exacte (D125, desat abans de mesurar) |
| Estat recurrent | Qwen3.6-35B-A3B | 62,81 MiB | 62,81 MiB | Exacte (D125) |
| KV + estat, 4 slots | Qwen3.8-27B | 867 MiB | 854 MiB | +1,5 % (D125) |
| KV per persona, 32K | Qwen3.6, 1, 2 i 4 persones | 702,8 / 1.405,6 / 2.811,2 MiB | 702,81 / 1.405,62 / 2.811,25 MiB | Exacte (`concurrencia-b60.md`) |
| KV | Qwen2.5 32B, 4096 | 1,0 GiB | 1,0 GiB | Exacte (D123) |
| KV | Coder 30B, 32768 | 3.072 MiB | 3.072 MiB | Exacte (D120) |
| Pesos a la GPU | Qwen3.6 | 16.899 MiB (el fitxer, 16,50 GiB) | 16.383 MiB | Sobreestimat 515 MiB (embeddings al host) |
| Còmput | Qwen3.6 a 32K, 128K i 256K | 256 MiB plans | 108, 204 i 332 MiB | **Queda curt a 256K, per 76 MiB** |

L'últim punt és l'únic on el supòsit de Jaull queda per sota de la realitat: el còmput creix amb el context i el 256 MiB pla ho
subestima a 262.144 tokens. L'1 GiB de marge ho absorbeix, i són només tres punts d'un model (**UNVERIFIED** que
sigui lineal en general).

## 3. Targetes petites (6 i 8 GiB): on el biaix conservador costa

Qwen2.5-7B Q4_K_M, la mateixa mesura en dues GPU. Aquí és on el comentari de Reddit sobre les targetes de 6 a 14 GB
tenia raó.

| GPU | Què va triar Jaull | Què va passar amb full offload | Cost |
|---|---|---|---|
| RTX 2060 6 GiB, WSL2, b10357 (abans de D120 i D123) | `-ngl` 24 a 26 (blocs 15 a 19 de 28) | **Arrenca amb 29/29** en els tres contextos, deixant **196 a 366 MiB lliures** | Segons el protocol: **1,41 a 2,10 vegades** més lent (tirades curtes, B001-R9, `ngl` 25 o 26 contra 29) o **uns 3 vegades** (`llama-bench` tg128, 3 repeticions: 19,4 a 21,5 contra 59,4 a 60,9 tok/s, `ngl` 24 o 25 contra tot) |
| RTX 4060 8 GiB, Windows, b11258 | 28 de 29 unitats | Full offload arrenca i acaba | **44,2 contra 52,7 tok/s** (+19 %) |

Fonts: `b001-r4-rtx2060-baseline.md`, `b001-r6` (via `rtx4060-campaign-comparison.md`), `b001-r8-2060-context-matrix.md`.
Una sola tirada per cel·la a la 2060 (**UNVERIFIED** la variància), i els entorns de la 2060 i la 4060 no són
comparables entre ells (canvien runtime, sistema i CPU).

Dues lectures, i cal dir-les totes dues:

- **Per la regla de llama.cpp, Jaull té raó**: a la 2060, 4.466 + 224 + 256 + 1.024 = 5.970 MiB contra 4.439 a 4.993
  lliures (**CALCULAT**, amb els números de `b001-r8`). Tant Jaull com llama.cpp diuen «no hi cap amb marge».
- **Per «arrenca i va molt més de pressa», Jaull és massa prudent**: el full offload funciona, entre 1,4 i 3
  vegades més ràpid segons el protocol (vegeu la taula; els dos protocols no donen la mateixa xifra i no s'han
  reconciliat). És un cas de «carrega» (nivell 2), no de «genera sense problemes amb marge» (nivell 3): no s'ha provat
  quant de temps aguanta amb 200 MiB lliures ni amb un navegador obert. Ho descriu el mateix document: «the typical
  performance gap and failure threshold remain unmeasured».

### Fórmula actual sobre el cas de la 2060 (2026-11-23, sense GPU)

Re-execució **offline** amb les entrades congelades de `validation/b001-r8-2060-context-matrix/ctx4096/prediction.json`
(pesos 4.466 MiB, 28 blocs, descomposició de blocs) i les constants actuals (marge fix de 1.024 MiB en lloc del
percentatge, sobrecàrrega de 256 MiB, reserva de 1.024 MiB). VRAM lliure 4.439, 4.601 i 4.993 MiB, els valors vistos
a B001-R8 i R9. Script: `rtx2060_now.py`, fora del repo.

| Context | Cal per a residència completa | Blocs a GPU segons HFA, abans (R8/R9) | Blocs a GPU segons HFA, ara |
|---:|---:|---:|---:|
| 512 | 5.774 MiB | 19 a 20 de 28 | 20 a 23 de 28 |
| 2048 | 5.858 MiB | 18 a 20 de 28 | 19 a 22 de 28 |
| 4096 | 5.970 MiB | 15 a 19 de 28 | 19 a 22 de 28 |

- **Cap xifra de lliure arriba a la residència completa**: ni 4.993 MiB. La fórmula nova segueix demanant
  l'offload parcial a la 2060. El full offload mesurat va fer servir 4.328 a 4.528 MiB.
- **Què ha canviat**: amb la mateixa VRAM lliure (4.601 MiB, la de R9) la fórmula deixa 0 a 1 bloc més a la GPU que
  abans (20, 20, 19 contra 21, 20, 20). El salt a full offload no es fa. Les comparacions amb R8 (15 a 19 blocs) no
  són netes: la VRAM lliure d'aquelles tirades era diferent.
- **Per què no cap**: la fórmula exigeix 1.024 MiB lliures (la regla de llama.cpp). El full offload va deixar 196 a 366.
  És el mateix cas «arrenca però sense marge» de dalt, nivell 2.
- **Límit d'aquesta execució**: el nombre d'unitats `-ngl` que s'enviaria a llama.cpp (el que va mesurar R9) surt del
  camí amb el GGUF local, que aquí no s'ha pogut executar. Amb el càlcul agregat de reserva les unitats són 17 a
  22; la política viva va emetre 25 a 26 a R9. No són comparables i no es pot dir quina sortiria ara.
- **No és una validació**: només diu què calcula el codi actual amb entrades antigues.

## 4. Errors que no són de memòria de la GPU

Aquí Jaull va dir «cap» i la realitat va ser una altra. No són errors de la fórmula, sinó coses que la fórmula no mira.

| Cas | Què va passar | Font | Estat |
|---|---|---|---|
| El model cap a la VRAM, però la RAM no | Càrrega amb mmap al PC de Biosfer: 12,2 GB residents, **0,5 GB de RAM lliure**, avís del monitor | X004, `concurrencia-b60.md` | Mitigat amb `-lm dio` per defecte a Windows (0,7 GB residents) |
| La VRAM passa a la RAM en repòs | Uns 70 s sense feina, la VRAM de la B60 es copia a RAM i el PC es queda sense | X004, `docs/cli.md` | Mitigat amb el reposo de 45 s a Intel + Windows; no ho evita a WSL2 |
| La memòria cau de prompts creix | De 1,3 a 5,6 GB residents, RAM lliure fins a 3,7 GB | `concurrencia-b60.md` | **PENDENT**: el sostre es calcula un sol cop |
| La velocitat depèn de l'entorn | La mateixa B60 i el mateix llama.cpp van de **4,6 a 18 vegades més lent** a WSL2 que a Windows; el Qwen3.8-27B peta a WSL2 | `b60-comparativa-wsl2-windows.md` | Jaull no prediu velocitat |
| Un benchmark que no acaba | Les campanyes de la 4060 amb b10357 no completaven la generació; la fallada amb `n_batch=1` no era d'OOM | `rtx4060-campaign-comparison.md` | Resolt amb un altre build; la causa de l'aturada inicial **no s'ha identificat** |

La segona fila del quadre anterior és la mateixa que va justificar la política del reposo. Cap d'aquests casos
falsifica la fórmula de memòria, però tots són maneres de ser «inviable» que Jaull no veu. Si l'avaluació parla de
«configuracions viables», aquesta distinció ha de ser explícita.

## Què es pot dir i què no

Es pot dir:

- En aquests nou casos de la B60, Jaull no ha donat cap fals positiu de memòria de GPU, i les tres vegades que va dir
  «no hi cap» quan sí hi cabia tenien causa coneguda i s'han corregit.
- Les prediccions de KV i d'estat recurrent, fetes abans de mesurar, han sortit exactes en el Qwen3.6, a tres contextos i a
  tres nombres de persones.
- A les targetes petites el biaix conservador té un cost mesurat en velocitat.
- Les maneres de ser inviable que Jaull no prediu (RAM del host, reposo de la VRAM, entorn d'execució) són reals i
  han passat en aquest projecte.

No es pot dir:

- Que el percentatge d'encerts sigui X %: són nou casos d'una sola GPU, només un (B6) a prop del límit, i els valors
  «ara» es van ajustar amb les mateixes dades.
- Que Jaull sigui millor que un mètode senzill. Això demana una comparació amb un mètode A escrit abans de mesurar
  (vegeu el pla de l'experiment) i, per al nivell 3, proves noves.
- Que les configuracions que «carreguen» sigui viables per a l'ús diari: no s'ha mirat què passa amb marge baix durant una hora.

## Per fer

- **FET (offline, 2026-11-23):** el cas de la 2060 amb la fórmula actual (vegeu l'apartat 3). **PENDENT** la part
  física: repetir-ho a la 2060 amb la versió actual i un fitxer GGUF local, per veure les unitats `-ngl` reals.
- **PENDENT:** buscar més casos a prop del límit, a la 2060 i a la 4060, que són els que més diferència poden fer.
- **PENDENT:** una definició de «viable» per escrit abans de qualsevol prova nova (nivell 1, 2 o 3).
