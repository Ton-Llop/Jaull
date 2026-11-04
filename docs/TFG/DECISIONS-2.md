# Decisions de l'entorn d'experimentació (fora de Jaull)

**Data d'inici:** 3 de novembre de 2026.

## 1. Per a què serveix aquest document

[`DECISIONS.md`](DECISIONS.md) explica per què Jaull té la forma que té (D001-D122). Aquest
document recull les decisions que no són del codi de Jaull: quines eines i quins models
s'usen per fer les proves de la memòria, i com es fan. Són material de treball per redactar el TFG,
no una especificació.

Els IDs són `X001`, `X002`... (X de "extern"), i no es renumeren. Quan una decisió canviï, no
s'esborra l'antiga: es marca com a substituïda i s'enllaça la nova. Els estats són els mateixos que a
`DECISIONS.md`: **VALIDATED** (comprovat amb una execució real), **UNVERIFIED** (hipòtesi o
impressió) i **PENDENT** (no fet).

### X001. L'arnès agentic és OpenCode

**Data i necessitat:** 2026-11-03. Després de l'IFEval cal provar els models com a agent de
programació sobre Lipopótamo, el codi de Biosfer. Calia triar l'arnès: el programa que dona al
model les eines per llegir, editar i executar ordres.

**Alternatives vistes:** Aider, Pi i OpenCode. L'Aider no s'ha provat. Pi (versió 1.5.0) i OpenCode s'han
instal·lat i configurat contra el mateix `llama-server`.

**Decisió (de l'autor):** OpenCode.

**Per què, amb el que sabem:**
- Els permisos són de sèrie. Es configuren a `opencode.json` amb `allow`, `ask` o `deny`, patrons per
  a les ordres de `bash`, `external_directory` i `doom_loop` (talla el model si repeteix la mateixa
  crida). Pi no en porta: calia una extensió de tercers, `pi-permission-system` 0.8.0, d'un sol autor,
  de la qual no s'ha comprovat la compatibilitat amb Pi 1.5.0.
- La documentació d'OpenCode inclou una configuració per a `llama-server` amb un proveïdor
  `@ai-sdk/openai-compatible`, i una compactació automàtica de la sessió quan el context s'acosta al límit.
- A la prova de fum amb Pi, l'autor va veure molts bucles. **UNVERIFIED** que la causa sigui Pi: no s'ha
  mirat la sessió, i hi havia un context de només 16384 tokens.
- L'autor considera OpenCode una plataforma més estable. **UNVERIFIED**: no s'ha fet cap prova
  d'estabilitat ni s'ha comparat de manera sistemàtica amb Pi.

**Prova de fum amb OpenCode** (Qwen3-Coder 30B, tasca: escriure tests de `pytest` per a `calc.py`).
Primera execució, fallida per l'entorn (el venv no era actiu, no hi havia `pytest`): el model va escriure els
tests, no va arribar a executar `pytest` i va donar la tasca per bona. Segona execució: 29,9 s, `pytest`
amb 3 tests aprovats, i `calc.py` al 100 % de cobertura en la verificació externa. **VALIDATED** com a
prova que el muntatge funciona; no és cap mesura de qualitat.

**Costos i riscos que s'accepten:**
- OpenCode usa la pantalla alternativa del terminal: no hi ha historial de desplaçament a `tmux`. Per
  treure la sessió es fa servir `/export` (Markdown).
- A `/models` apareixen també els models gratuïts del núvol d'OpenCode. Amb el codi de Biosfer **només** s'usen
  els del proveïdor local `b60`, perquè el codi no pot sortir de la màquina.
- No s'ha anotat la versió exacta d'OpenCode instal·lada.

**Límits:** una sola prova de fum, una sola tasca. No es pot dir que OpenCode sigui millor que Pi o
l'Aider: és l'arnès triat, per les raons de dalt.

**Fonts:** [OpenCode, proveïdors](https://opencode.ai/docs/providers/),
[permisos](https://opencode.ai/docs/permissions/), [configuració](https://opencode.ai/docs/config/).

### X002. El model del benchmark agentic és el Qwen3-Coder 30B-A3B (Q4_K_M)

**Data i necessitat:** 2026-11-03. Es disposava de tres models amb IFEval: el Coder, el Qwen2.5 32B i el
Qwen3.8 27B. Calia triar quin s'usa per al benchmark llarg sobre Lipopótamo.

**Decisió (de l'autor):** el Qwen3-Coder 30B-A3B Q4_K_M, per velocitat i per context.

**Evidència:**

| | Coder 30B-A3B | Qwen3.8 27B | Qwen2.5 32B |
|---|---:|---:|---:|
| Generació a Windows natiu (t/s) | 99,2 | 15,55 | no mesurat |
| IFEval, prompt strict | 0,789 ± 0,018 | 0,821 ± 0,017 (sense raonament) | 0,797 ± 0,017 |
| Context que cap a la B60 | 32768 mesurat | no mesurat | no mesurat |

- **Context, VALIDATED:** a `-c 32768` llama.cpp projecta 20712 MiB (pesos 17524, KV 3072, còmput 116) de
  23613 MiB lliures, i en queden 2901. El KV és de 96 MiB per cada 1000 tokens. Vegeu D120.
- **Velocitat, VALIDATED** per al Coder i el 27B (D121). El Coder és uns 6,4 cops més ràpid.
- **IFEval no discrimina:** les diferències entre els tres són de l'ordre de l'error estàndard de la
  diferència (uns 0,024). A més, IFEval mesura el seguiment d'instruccions de format, no la capacitat de
  programar com a agent.
- **Context del Qwen2.5 32B, VALIDATED:** a `-c 16384` carrega les 65 capes però en queden només 791 MiB
  lliures, per sota de l'1 GiB que demana llama.cpp (KV 4096 MiB, 256 MiB per cada 1000 tokens). El límit pràctic és
  d'uns 15.000 tokens. A `-c 4096` en queden 3875 lliures. Vegeu D123.
- **Context del Qwen3.8 27B, UNVERIFIED:** els 16384 de la configuració vénen d'una estimació de KV per
  dimensions, no d'una mesura.
- **Compactació d'OpenCode, UNVERIFIED:** amb `reserved` 10000 per defecte, un context d'uns 15.000 tokens faria
  compactar cap als 5.000, que és gairebé la mida de tota una sessió de la prova de fum (5.843 tokens). Per això,
  amb aquests valors, OpenCode no és viable amb un model dens a la B60. No s'ha provat.

**Què no es pot afirmar:** que el Coder programi millor, o igual, que els models densos com a agent.
No s'ha mesurat. La decisió és pràctica (rapidesa i context), i els resultats del benchmark valdran per
a aquest model.

**PENDENT:** una prova de fum curta amb el 27B i el 32B per poder dir què passa amb els altres dos.

**Fonts:** [registre d'IFEval](../ifeval-b60-prova.md), D120, D121.

### X003. Protocol d'execució del benchmark agentic

**Data i necessitat:** 2026-11-03. Les dues proves de fum van mostrar què cal controlar abans de comparar
res. **Estat: PENDENT**, el protocol es tanca abans de veure resultats sobre Lipopótamo.

**Decisions preses:**
1. **Còpia d'un sol ús.** L'agent treballa en una còpia de Lipopótamo a `~/` de WSL (no a `/mnt/c`), amb el
   commit fixat. No toca mai l'original. Requereix permís per escrit de Biosfer, **PENDENT**.
2. **Permisos tancats.** `"*": "deny"` i només es permet llegir, editar, `glob` i `grep`; a `bash`, `python`,
   `python3`, `pytest`, `coverage`, `mutmut`, `git status`, `git diff`, `git log` i `ls`. `external_directory` i
   `doom_loop` en `deny`. Això no és un aïllament: `ls` i `python` poden llegir fora de la carpeta. La
   protecció real és la còpia d'un sol ús i que no hi hagi credencials a la màquina.
3. **Servidor i context.** `llama-server.exe` de Windows natiu amb `-c 32768 -np 1 --jinja --cache-ram 0 -lm dio
   --sleep-idle-seconds 45` (els tres últims, per X004: la càrrega no omple la RAM i el servidor s'adorm abans que el
   controlador buidi la VRAM; provisional, amb un cost d'uns 33 s a la primera petició després d'una pausa de més de
   45 s, que s'ha de comptar a part a les mètriques de temps), i el mateix `limit.context` a `opencode.json`. Compactació amb els valors per defecte (`auto` actiu, `prune` desactivat,
   `reserved` 10000, és a dir, es compacta cap als 22.700 tokens). Es declara com a paràmetre fix.
4. **Verificació externa, no la paraula de l'agent.** A la primera prova de fum el model va afirmar que els
   tests passaven sense haver-los executat. L'èxit es mesura amb `python -m pytest` i amb `coverage` sobre el
   codi original (`coverage run --include=<fitxer>`), que l'autor executa després, i es comprova amb un `diff`
   contra el motlle que el codi sota prova no ha canviat. Executar-ho amb `pytest` a seques és una dada a part:
   a la primera execució del `calc.py` falla amb `ModuleNotFoundError: No module named 'calc'` (els tests
   importen el mòdul sense `conftest.py` ni configuració de `pythonpath`) i amb `python -m pytest` passen els 3.
   L'agent va fer servir `python -m pytest`; no es compta com a fallada, però es registra.
5. **Carpeta neta a cada execució.** A la primera prova, l'agent va deixar `test_runner.py` i
   `test_simple.py` que ningú havia demanat, i van entrar a la cobertura total (97 % contra el 100 % del
   codi sota prova). Es parteix d'un motlle amb només el codi inicial i un `venv` compartit fora de la carpeta.
6. **Mètriques per execució:** temps total de la sessió (el que mostra OpenCode és el de l'última resposta),
   torns, tokens de context, compactacions, ordres denegades, fitxers extra creats, si el codi sota prova
   ha quedat intacte, tests que passen amb `python -m pytest`, si també passen amb `pytest` a seques, cobertura
   del codi original i, si cal, puntuació de mutacions.

7. **Els tests de l'agent s'avaluen també contra el codi original**, en una carpeta temporal amb el motlle i
   només el seu directori `tests/`. Si l'agent canvia el codi sota prova, la cobertura de la seva carpeta és de
   codi modificat i no es pot comparar (vegeu la prova de variància).

**Prova de variància (2026-11-03, Qwen3-Coder 30B, tasca: tests de `calc.py`, 3 execucions, mateixa ordre):**

| | Execució 1 | Execució 2 | Execució 3 |
|---|---|---|---|
| Codi sota prova | intacte | intacte | **modificat** (docstrings i un `if b == 0`) |
| Fitxers extra | cap | cap | **`setup.py`** |
| `python -m pytest` | 3 passen | 3 passen | 3 passen |
| `pytest` a seques | falla (`No module named 'calc'`) | igual | igual |
| Cobertura de `calc.py` | 100 % (4 sentències) | 100 % (4) | 100 % **de codi modificat** (6) |
| Context al final | 4.792 tokens (15 %) | 4.019 (12 %) | **PENDENT** |
| Compactació | no | no | **PENDENT** |
| Skill `pytest` | no | sí, **UNVERIFIED** si es va executar | **PENDENT** |
| Temps de la sessió | no comparable | no comparable | no comparable |

El temps no es pot comparar: la primera execució va anar amb una conversa ja oberta, i la caché de prompt del
servidor la pot haver avantatjat (**UNVERIFIED**); el que mostra OpenCode és, a més, el de l'última resposta.

**Conclusió, amb tres repeticions:** el resultat de la tasca és estable (els tres cops, 3 tests que passen), però el
**procés no ho és**: a la tercera l'agent va canviar el codi que havia de respectar i va crear un fitxer no demanat,
i a la segona va carregar una skill. Una execució per tasca no basta per descartar aquestes desviacions; amb només tres
repeticions no s'estima una taxa. **VALIDATED** com a observació, no com a mesura de freqüència.

**Pendent abans de començar:** el `diff` dels tests entre execucions, de on surt la skill `pytest`, decidir la
llista de tasques, el permís de Biosfer i resoldre X004: mentre el servidor pugui buidar la VRAM cap a la RAM
del PC, el benchmark no es pot deixar desatès.

**Límits previstos:** una sola base de codi, tres repeticions per tasca com a mínim, i un tipus de tasca
(escriure tests). Els resultats descriuen aquest model en aquest cas, no en general.

**Fonts:** X001, X002, D120.

### X004. El PC de Biosfer es queda sense RAM amb el servidor carregat

**Data i necessitat:** 2026-11-04. En obrir el servidor i enviar-li la primera petició llarga, Biosfer va dir que el
PC anava molt malament. Abans, amb el servidor per defecte, el disc C: i la RAM havien arribat al 100 % mentre
l'agent «pensava», i un `llama-server.exe` penjat va arribar a 28 GB de memòria privada. Calia saber què passava
perquè l'agent s'ha de poder fer servir mentre el PC té altres feines.

**Màquina:** 27,9 GB de RAM; el monitor penja de la iGPU AMD (1920×1080) i la B60 no en té cap; la VM de WSL2
té un topall de 13 GB. Amb el PC en repòs, la RAM lliure era de 13 a 18 GB.

**Mesures.** Un vigilant de PowerShell (RAM lliure, memòria privada i residente del servidor, CPU, disc) més el
contador de memòria de GPU de Windows per adaptador, mostrejats cada 2 o 3 s. El vigilant mata el servidor si la
RAM lliure baixa de 8 GB.

**1. Càrrega del model, amb el mode per defecte (mmap).** El servidor porta el GGUF de 17,5 GB a la RAM encara que
les capes acabin a la B60. En 10 s el residente va pujar a 13,1 GB i la RAM lliure va passar de 13,0 a 0,1 GB, amb
604.527 pàgines per segon. **VALIDATED** (una execució; el vigilant la va tallar).

**2. Amb `-lm dio` (DirectIO) la càrrega ja no consumeix RAM.** Residente de 0,3 a 0,4 GB i RAM lliure estable
(de 13,5 a 12,6 GB en la primera mesura). La «memòria privada» de 20,7 GB no és RAM física: puja a 20,7 mentre la
lliure gairebé no es mou, i és la VRAM de la B60 comptada com a memòria compromesa. **VALIDATED**, 8 execucions.
Que `-lm dio` no canvia la velocitat de generació **no s'ha comprovat**.

**3. Uns 70 s sense feina a la GPU, el contingut de la VRAM passa a la RAM.** Amb el servidor ja carregat, la VRAM
de la B60 baixa de 20,3 GB a 0 en uns 12 s i la RAM lliure cau de ~16 GB a menys de 5 GB al mateix ritme (uns
5,5 GB cada 3 s). El procés no canvia: residente i privada constants.

| # | Situació | Fi de l'última activitat | Inici de la caiguda | Diferència |
|---|---|---|---|---|
| 1 | Servidor sol després de la càrrega | 09:56:14 | 09:57:24 | 70 s |
| 2 | Servidor sol | 10:02:01 | 10:03:11 | 70 s |
| 4 | Servidor sol, `-c 8192` | 10:13:54 | 10:15:05 | 71 s |
| 5 | Servidor sol | 10:18:36 | 10:19:44 | 68 s |
| 6 | Servidor sol | 10:22:41 | 10:23:51 | 70 s |
| 7 | 10 generacions llargues amb `curl.exe` (HTTP 200, ~16 s cadascuna) | 10:31:01 | 10:32:12 | 71 s |
| 8 | OpenCode amb una tasca real (de 61 a 64 t/s amb ~13.000 tokens de context) | ~10:41:53 | ~10:43:03 | ~70 s |

L'execució 3 (peticions de 8 tokens des de `curl` de WSL, sense mirar el codi HTTP) no encaixa i no es compta: pot
ser que les peticions no arribessin al servidor. **UNVERIFIED**. A les execucions 7 i 8 la hora de fi surt de
`srv.log`, alineada amb el vigilant, amb un error d'uns ±3 s. Mentre OpenCode treballava, la RAM lliure es va
quedar entre 16,0 i 16,5 GB, el disc a ~0 % i la CPU per sota del 25 %.

**El que no hi ha:** `srv.log` no mostra res a aquesta hora, el registre d'esdeveniments del sistema no té cap
avís de vídeo ni reinici del controlador, i el contingut de la memòria compartida de GPU no es mou.

**Interpretació, UNVERIFIED en el mecanisme.** Un termini tan constant fa pensar en un comportament del
controlador de Windows (WDDM) que treu de la VRAM els recursos d'una GPU sense pantalla i sense feina. A llama.cpp
hi ha una incidència oberta amb el mateix patró: la
[#25646](https://github.com/ggml-org/llama.cpp/issues/25646) descriu una Intel Arc Pro B70 secundària, sense monitor,
a Windows 11 (controlador 32.0.101.8804) que perd els pesos de la VRAM després d'uns 70 s d'inactivitat; la GPU
principal no es veu afectada i amb una NVIDIA com a secundària no passa. El nostre cas coincideix en el termini, en
la B60 sense monitor i en el sistema operatiu. El que aporten les mesures d'aquí, i que no he vist a la incidència,
és que la RAM lliure del PC baixa al mateix ritme que la VRAM. El primer cuelgue de Biosfer podria tenir aquesta
causa (servidor carregat i esperant la petició), però no s'ha demostrat. Segons la incidència, la petició següent
tarda molt més (un prompt eval de 174 a 1,9 t/s en el seu cas): no s'ha mesurat aquí.

La incidència [#27097](https://github.com/ggml-org/llama.cpp/issues/27097) (generació lenta per memòria
Vulkan `DEVICE_LOCAL|HOST_VISIBLE`, variable `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`) és un altre problema: aquí la
generació és ràpida (99 t/s a Windows) i no encaixa amb el que s'ha mesurat.

**Conseqüència per al benchmark:** qualsevol pausa de més de 70 s de l'agent (una sessió desatesa, una tasca
llarga de l'usuari entre torns) pot deixar sense RAM el PC de Biosfer.

**Què passa si no es talla (9a execució, 11:22 aprox., observació de l'autor, sense el registre a mà).** Es va
aturar el vigilant que mata el servidor i es va deixar córrer la caiguda. El PC va quedar amb un lag extrem, però no
penjat del tot: el `taskkill` va trigar de 15 a 20 s a executar-se, i un prompt enviat al voltant dels 70 s no va
acabar de funcionar (sense detall del resultat). En acabar el procés el PC es va recuperar. Queda a mig camí entre
«es recupera amb lag» i «es pengen les màquines»: no es va comprovar quant triga el sistema a tornar a la normalitat
si el servidor no es mata.

**Mitigació provada: `--sleep-idle-seconds 45` (10a execució, 11:54 a 11:58).** Amb aquest flag el servidor
descarrega el model després de 45 s sense peticions, i la VRAM queda lliure abans que el controlador l'expulsi als
~70 s. El text d'ajuda no deia que descarregués el model; s'ha vist a la prova.

**Què fa el flag.** `--sleep-idle-seconds N` fa que el servidor, després de N segons sense cap petició, passi a un
estat de repòs. Segons les mesures, en repòs el procés segueix viu i escoltant al port, però ha descarregat el
model: la VRAM de la B60 queda a 0 i la memòria privada del procés baixa a 0,1 GB. Quan arriba una petició, el
servidor torna a carregar el model des del disc i respon. Amb el valor per defecte, -1, està desactivat. El binari
només descriu el flag amb «number of seconds of idleness after which the server will sleep»; la resta ve del que
s'ha observat, no del codi, que no s'ha llegit.

- Càrrega acabada a les 11:55:17 (VRAM 20,3 GB). A les 11:56:03, 46 s després, la VRAM de la B60 passa a 0 i la
  memòria privada del servidor de 20,7 a 0,3 GB; el procés segueix viu.
- Durant els ~76 s següents, amb el servidor adormit, la RAM lliure no es mou (de 10,0 a 10,4 GB). No hi ha la caiguda
  de ~16 GB.
- Una petició a les 11:57:19 va retornar HTTP 200 en 18,96 s. La VRAM va tornar a pujar (12,0 GB a les 11:57:21, 17,2 a
  les 11:57:24 i 20,3 a les 11:57:39) i la RAM lliure només va baixar de 10,1 a 9,6 GB.
- **VALIDATED** en aquest primer cicle. Que en dormir es perdi la memòria cau de prompts i s'hagin de reprocessar
  s'ha vist després, a l'11a execució (vegeu sota).
- Cost: la primera petició després d'una pausa de més de 45 s paga la recàrrega (uns 19 s aquesta vegada, amb una
  petició de 16 tokens). Una sola mesura.

**Sessió d'OpenCode amb el flag (11a execució, 12:03 a 12:09).** Dues tasques de programació amb una pausa de més de
45 s entre elles i una altra pausa al final.
- El servidor s'adormia 45 s després de l'última petició. Segons `srv.log`, l'última acaba a 4:54,9 des de
  l'arrencada i «entering sleeping state» és a 5:40,2 (45,3 s). Les dues vegades la VRAM va passar a 0 (12:05:32 i
  12:08:55) i la RAM lliure no es va moure (de ~10,0 a ~10,4 GB).
- Mentre treballava, la VRAM es va quedar a 20,3 GB i la RAM lliure entre 9,4 i 10,1 GB. No hi va haver cap
  caiguda. **VALIDATED**, dos cicles de son i dos despertars.
- Despertar abans de la tasca 2, segons `srv.log`: surt de l'estat de son a 3:05,5; la primera tasca es llança a
  3:24,6 (19,0 s de recàrrega del model); i aquesta primera petició reprocessa tot el context, 7.680 tokens en
  13,8 s (555 t/s). En total, uns 33 s des que arriba la petició fins que el model comença a generar. **VALIDATED**,
  una sola mesura. Queda confirmat que en dormir es perd la memòria cau de prompts. Amb el context de ~13.000 tokens
  d'altres execucions el reprocessament seria més llarg: és una estimació, no una mesura.
- Velocitat de generació amb ~12.000 tokens de context: de 63 a 67 t/s.
- Segons l'autor, Biosfer i la resta de gent al PC no van notar res durant la sessió. Observació de l'autor, sense
  mesura pròpia (UNVERIFIED en el sentit que no hi ha cap registre del seu costat).

**Decisions:**
0. Provisional, a l'espera de més proves: `--sleep-idle-seconds 45` als arrencs del servidor (vegeu la mitigació
   de dalt). El keepalive de la PR #29267 queda com a alternativa si es disposa d'un build que el porti; el
   `b11889` no el té.
1. Als arrencs següents del servidor, `-lm dio`. El punt 3 de X003 encara no ho recull i s'ha d'actualitzar quan
   se'n comprovi l'efecte en la velocitat.
2. Cap prova ni benchmark sense el permís de Biosfer a cada franja, el vigilant en marxa i el tall de RAM lliure.
3. El benchmark no es deixa desatès fins que X004 estigui resolt.

**PENDENT, una variable cada vegada:**
- Comprovar si el `llama-server.exe` b11889 té `--gpu-heartbeat`: `./llama-server.exe --help | grep -i -E 'heartbeat|sleep'`.
  El PR [#25214](https://github.com/ggml-org/llama.cpp/pull/25214) l'afegeix (Vulkan: `vkCmdFillBuffer` periòdic
  sobre un buffer de proves) i, segons la pàgina, està en esborrany i sense fusionar. Si el build no el porta, caldria
  un build més nou o el de la branca del PR; no s'ha provat.
- Una petició mínima cada ~30 s per mantenir la GPU desperta. Pot invalidar la memòria cau de ~13.000 tokens de
  l'slot únic (`-np 1`) i obligar a reprocessar-los (**UNVERIFIED**): cal mesurar-ho. A la incidència #25646 uns
  pings periòdics van resultar poc fiables amb finestres d'expulsió curtes, en un cas d'APU d'AMD.
- Provar si la B60 deixa d'afectar-se quan té un monitor o un endoll fals (HDMI dummy) connectat: la incidència diu
  que la GPU principal no es veu afectada. És maquinari del PC de Biosfer i **UNVERIFIED**.
- Treure l'estalvi d'energia del bus PCIe al PC de Biosfer (`powercfg`). És un canvi de sistema i només amb el seu
  permís; tampoc se sap si és aquest el mecanisme.
- Comparar la velocitat de generació amb `-lm dio` i amb el mode per defecte.

**Límits:** una sola màquina, un sol model, un sol build de llama.cpp (b11889) i un sol controlador d'Intel. De
vuit execucions amb `-lm dio`, set entren a la taula i una s'exclou (la 3). El vigilant consulta contadors de
Windows cada 2 o 3 s a totes, i no s'ha provat una execució sense ell.

**Explicació senzilla, per rellegir en el futur.**

*Què passa normalment*
1. El model és un fitxer de 17,5 GB al disc.
2. Per fer-lo servir, el servidor el copia a la memòria de la targeta gràfica (la VRAM). Allà n'ocupa uns 20 GB.
3. Si la targeta passa uns 70 segons sense fer res, Windows o el controlador d'Intel copien tot aquest contingut de
   la VRAM a la RAM del PC. No sabem del cert per què ho fan.
4. La RAM del PC de Biosfer no té lloc per a tant, així que el PC va lentíssim.

*Què fa `--sleep-idle-seconds 45`*
1. Quan passen 45 segons sense peticions, el servidor esborra el model de la VRAM. El fitxer del disc no es toca.
2. Als 70 segons Windows mira i la VRAM és buida. No hi ha res a copiar a la RAM, així que la RAM no s'omple.
3. El servidor continua encès i escoltant, però sense el model carregat.

*Quan arriba una petició nova*
1. El servidor torna a copiar el model del disc a la VRAM (uns 19 segons).
2. Torna a llegir la conversa sencera per tenir-la en memòria (uns 14 segons amb 7.700 tokens).
3. Llavors respon.

Per això, després d'una pausa de més de 45 segons, la primera resposta triga uns 33 segons. Mentre hi hagi peticions
seguides, res d'això passa i va a la velocitat normal. L'únic que no està confirmat és el punt 3 de «Què passa
normalment» (per què ho fa Windows). El que sí s'ha mesurat és que, amb el flag, la RAM no s'omple.

**Estat (2026-11-04):** tancat provisionalment. Es manté `--sleep-idle-seconds 45` com a mitigació (X003, punt 3).
Les proves de la llista PENDENT (valor de 60 s, velocitat amb i sense `-lm dio`, keepalive de la PR #29267, endoll
HDMI, `powercfg`) no es fan ara. Segons l'autor, després de l'última execució s'han esborrat del PC de Biosfer els
scripts de mesura i no hi queda cap servidor de prova en marxa. El comentari preparat per a la incidència #25646
queda a decisió de l'autor.

**Fonts:** X002, X003, [taula B60](../b60-taula-comparativa.md),
[llama.cpp #25646](https://github.com/ggml-org/llama.cpp/issues/25646),
[llama.cpp #25214](https://github.com/ggml-org/llama.cpp/pull/25214),
[llama.cpp #27097](https://github.com/ggml-org/llama.cpp/issues/27097) (consultades el 2026-11-04).
