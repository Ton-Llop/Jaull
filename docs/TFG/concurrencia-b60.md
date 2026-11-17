# Concurrència a la B60: quants usuaris alhora

Pregunta per a Biosfer: si el Qwen3.6-35B-A3B corre a la B60 com a servei compartit a la xarxa, quantes persones el
poden fer servir alhora i amb quina velocitat cadascuna. També valida una part de Jaull: l'estimador multiplica la
memòria cau KV pel nombre d'usuaris simultanis (`concurrent_users`), i això no s'havia comparat mai amb una mesura.

Estats: **VALIDATED**, **UNVERIFIED**, **PENDENT**.

## Muntatge

- `llama-server.exe` b11889, Vulkan natiu a Windows, Qwen3.6-35B-A3B UD-IQ4_XS, `--reasoning off` (resposta directa,
  com en un xat), sense `--sleep-idle-seconds` perquè el model no es descarregui entre rondes.
- `-np N` slots paral·lels amb `-c 32768 × N`, de manera que cada usuari té 32.768 tokens de context. **PENDENT**
  comprovar al registre del servidor que cada slot rep 32.768 i no el total.
- Client: `scripts/concurrency-bench/load.py`, des de WSL a monk-e pel pont `127.0.0.1:8081`. N peticions alhora,
  prompt d'unes 650 paraules amb un prefix únic per petició (sense memòria cau de prompt), 256 tokens de resposta,
  temperatura 0, 3 repeticions.
- Mesures per petició: temps fins al primer token (TTFT), tokens generats i tok/s de generació. Per ronda: tok/s
  agregats (tokens de totes les peticions entre el temps total).

| Servidor | Nivells de càrrega | Què mostra |
|---|---|---|
| `-np 1` | 1 | Referència: un sol usuari |
| `-np 2` | 1, 2 | Dos usuaris alhora |
| `-np 4` | 1, 2, 4, 8 | Quatre alhora; amb 8, la meitat espera a la cua del servidor |
| `-np 8 --kv-unified -c 196608 --kv-unified-per-slot 131072` | 1, 4, 8, 16 | La configuració proposada per a Biosfer: KV compartit |

## Predicció de Jaull (feta abans de mesurar)

KV + estat recurrent del Qwen3.6-35B-A3B a 32.768 tokens per usuari, calculat amb `estimate_kv_cache` el 2026-11-13
(confiança mitjana):

| Usuaris | Predicció (MiB) | Mesura de llama.cpp: KV + estat recurrent (MiB) |
|---|---:|---:|
| 1 | 702,8 | 640,00 + 62,81 = 702,81 |
| 2 | 1.405,6 | 1.280,00 + 125,62 = 1.405,62 |
| 4 | 2.811,2 | 2.560,00 + 251,25 = 2.811,25 |

**VALIDATED (2026-11-13): la predicció coincideix amb la mesura en els tres casos**, al centèsim de MiB. Mesurat
arrencant el servidor amb `-lv 4` i sense càrrega (`server-mem-np{1,2,4}.log`, línies `KV buffer size` i `RS buffer
size`). Confirma que l'estimador multiplica bé el KV i l'estat recurrent pel nombre d'usuaris en aquest model híbrid.

La resta de la memòria de la GPU, en els mateixos registres: pesos 16.383,34 MiB a la GPU (més 515,31 MiB a
l'amfitrió) i un buffer de càlcul de 108,02 MiB que no canvia amb el nombre de slots. Amb `-np 4` la GPU té, per tant,
unes 19.303 MiB ocupades per llama.cpp. Amb `-np 8` i 32K fixos serien unes 22.114 MiB (càlcul a partir d'aquestes
xifres, no mesurat), que deixa poc més d'1,5 GiB de la B60 lliure: és la configuració que va penjar el PC.

## Resultats (2026-11-13)

Medianes de totes les peticions i rondes (3 repeticions per nivell); cap petició fallida; tokens comptats pel servidor.
TTFT és el temps fins al primer token; «per usuari» és la velocitat de generació que veu cada persona; «agregat» són
els tokens de tots dividits pel temps de la ronda.

| Servidor | Usuaris alhora | TTFT (s) | tok/s per usuari | tok/s agregats |
|---|---:|---:|---:|---:|
| `-np 1`, 32K | 1 | 0,67 | 61,2 | 52,6 |
| `-np 2`, 32K per slot | 1 | 0,68 | 61,5 | 52,9 |
| | 2 | 1,15 | 46,1 | 76,1 |
| `-np 4`, 32K per slot | 1 | 0,68 | 61,3 | 52,7 |
| | 2 | 1,31 | 24,2 | 43,0 |
| | 4 | 2,97 | 31,4 | 92,1 |
| | 8 (4 a la cua) | 7,24 | 31,4 | 98,8 |
| `-np 8`, KV compartit de 192K, màxim 128K per slot | 1 | 0,61 | 62,5 | 53,5 |
| | 4 | 2,12 | 33,2 | 104,1 |
| | 8 | 4,45 | 19,4 | 116,9 |
| | 16 (8 a la cua) | 11,97 | 19,4 | 118,5 |

**Què se'n treu, VALIDATED per a aquesta càrrega:**
- Un usuari sol: primer token en uns 0,6–0,7 s i 61–62 tok/s, igual amb qualsevol configuració del servidor.
- Més usuaris alhora reparteixen la GPU: cadascú va més lent, però el total puja. El sostre és d'uns 118 tok/s
  agregats amb 8 slots (uns 100 amb 4).
- Amb 8 usuaris alhora, cadascun veu 19 tok/s i el primer token als 4,5 s. Per sobre dels slots configurats ningú va
  més lent, però els que sobren esperen a la cua: amb 16, el primer token puja a 12 s de mediana.
- El KV compartit (`--kv-unified`) deixa que un sol usuari arribi a 128K de context mentre el servidor accepta 8
  peticions alhora; amb el repartiment fix, 8 slots en un pressupost així donen només 32K per usuari.

**Incidències:**
- `-np 8` amb 32K fixos per slot (`-c 262144`) va arrencar, però amb la càrrega el PC de Biosfer es va penjar
  (l'autor ho va veure; no es va mesurar). **UNVERIFIED** la causa: la hipòtesi és que aquesta reserva deixa la GPU
  sense marge i Windows passa a fer servir la RAM del sistema, que comparteixen altres usuaris i WSL. Amb la bossa
  compartida de 192K, uns 1,3 GiB menys de KV, no va passar.
- Amb `-np 4` i 2 usuaris, dues de les tres rondes van donar la meitat de velocitat (24 tok/s cadascun, 43 agregats)
  i la tercera, 46 i 76 com amb `-np 2`. Repetit el mateix dia amb un servidor nou i 5 rondes
  (`np4-n2-rep.csv`): totes normals, 49–51 tok/s per usuari i 80–84 agregats (mediana 50,2 i 82,8). **No es
  reprodueix.** La causa més probable és una altra feina a la GPU durant aquelles dues rondes (el PC és compartit),
  **UNVERIFIED**; no hi ha indicis que sigui un comportament de llama.cpp. La fila de 2 usuaris amb `-np 4` de la taula
  és la de la primera tanda; la de la repetició és més representativa.

**Recomanació per a Biosfer (proposta, no mesurada amb ús real):** `-np 8 --kv-unified -c 196608
--kv-unified-per-slot 131072`. Fins a 8 persones alhora llegeixen còmodament (19 tok/s és més ràpid que una persona
llegint), i qui necessiti un context llarg, com l'agent de codi, el té. No omplir la GPU fins al límit en un PC
compartit.

## Servidor arrencat per Jaull (2026-11-16)

La mateixa prova, però amb el servidor arrencat per `jaull serve start` en lloc dels paràmetres escrits a mà. Comprova
que el dimensionament automàtic dona el mateix servei que la configuració proposada.

- Ordre: `jaull serve start unsloth/Qwen3.6-35B-A3B-GGUF -q IQ4_XS --users 8 --context 131072 --shared-context
  --backend vulkan --llama-server C:\llama-win\llama-server.exe`, amb Jaull executat a Windows natiu.
- Pla que va imprimir Jaull: 8 slots amb una bossa compartida de 131.072 tokens; context màxim per persona en aquesta
  màquina, 131.072 (el model en suporta 262.144); memòria de GPU prevista 19,8 GiB amb un límit de servei de 20,2 GiB.
  La memòria cau de prompts a la RAM va quedar **desactivada**, perquè Windows només tenia 5,6 GiB lliures en
  arrencar. **UNVERIFIED** qui ocupava la resta (probablement la màquina virtual de WSL).
- Diferències amb el 2026-11-13: bossa de 128K en lloc de 192K, i client executat a Windows contra `127.0.0.1:8090`,
  sense el pont SSH.
- Mateix client i mateixa càrrega: `load.py`, 3 repeticions per nivell. Dues tandes seguides amb el mateix servidor
  (`serve-np8-jaull.csv` és la segona). Cap petició fallida.

| Usuaris alhora | TTFT (s) | tok/s per usuari | tok/s agregats | 2026-11-13, a mà (per usuari / agregats) |
|---:|---:|---:|---:|---:|
| 1 | 0,57 / 0,60 | 67,3 / 64,8 | 58,5 / 56,5 | 62,5 / 53,5 |
| 4 | 2,23 / 2,17 | 33,9 / 34,1 | 103,6 / 105,9 | 33,2 / 104,1 |
| 8 | 4,28 / 4,22 | 20,7 / 20,5 | 121,8 / 123,2 | 19,4 / 116,9 |
| 16 (8 a la cua) | 11,48 / 11,72 | 20,2 / 20,2 | 123,3 / 122,0 | 19,4 / 118,5 |

Cada cel·la amb dues xifres és primera tanda / segona tanda.

**VALIDATED per a aquesta càrrega:** el servidor arrencat per Jaull dona el mateix rendiment que la configuració
escrita a mà, o una mica més (+3–5 %). Part de la diferència pot venir de no passar pel pont SSH; no s'ha separat.
Les dues tandes coincideixen en un 4 %. Amb 16 usuaris torna a sortir el límit de 8 slots: la velocitat no baixa,
però el primer token arriba als 11–12 s perquè la meitat espera a la cua. La primera petició de cada tanda (TTFT
10–12 s) és l'escalfament del model i queda fora de les medianes.

### Context gairebé ple

`load.py --levels 8 --repeats 1 --prompt-paragraphs 300 --max-tokens 256` (`serve-pool-full.csv`): 8 prompts
d'uns 13.275 tokens alhora, unes 108.000 posicions en total amb les respostes, el 82 % de la bossa. **No la va
omplir.**

| Peticions | Correctes | TTFT mediana (s) | tok/s per usuari (mediana) | Temps total (s) |
|---:|---:|---:|---:|---:|
| 8 | 8 | 82,0 | 3,5 | 157,7 |

- Hi caben 8 converses llargues alhora, sense errors, però la primera resposta triga més d'un minut i la generació baixa
  a uns 3,5 tok/s per persona.
- Les peticions comparteixen la GPU amb la lectura dels prompts de les altres. La que va començar abans (TTFT 22 s) va
  generar a 1,9 tok/s, i la que va començar l'última (TTFT 108 s), a 5,2 tok/s. Per tant, la lentitud no ve només de la
  mida del context. **UNVERIFIED** quina part pesa més: només hi ha una ronda.
- «tok/s agregats» (13,0) no és comparable amb la taula anterior: divideix els 2.048 tokens generats pel temps total,
  que inclou la lectura dels prompts.

### Bossa desbordada

`--prompt-paragraphs 400` (`serve-pool-overflow.csv`): 8 prompts d'uns 17.700 tokens, uns 142.000 en total, per
sobre dels 131.072 de la bossa. Servidor tornat a arrencar amb la mateixa ordre. Aquesta vegada Windows tenia 19,5 GiB
lliures i Jaull va activar la memòria cau de prompts amb 8 GiB, que en la prova de 300 paràgrafs estava desactivada.
Un primer intent (0 de 8 en 2 s) es va llançar amb el servidor ja aturat i no compta.

**VALIDATED (registre del servidor, `server.log`):** quan la bossa s'omple, llama.cpp no posa cap petició a la cua.
Prova lots cada vegada més petits (`failed to find free space in the KV cache, retrying with smaller batch size`, de
128 fins a 1) i, quan ni un sol token hi cap, respon `Context size has been exceeded` a **totes** les peticions
actives alhora i allibera els 8 slots. En aquell moment, set slots tenien unes 17.700 posicions cadascun i el vuitè
9.190 (unes 133.000 en total).

- Les 8 respostes es tallen al mateix instant (179,7 s). Les que ja generaven en portaven entre 2 i 54 de 256; la
  vuitena no va rebre cap token.
- Abans del tall, la generació va anar a uns 0,35 tok/s per persona. Mentre la bossa estava gairebé plena, el servidor
  feia els reintents amb lots petits.
- `load.py` va comptar com a correctes les 7 que havien rebut algun token, perquè no reconeixia un error enviat a mig
  flux. Les dades vàlides d'aquesta prova són les del registre, no la columna `ok` del CSV. Corregit després de la
  prova: ara un error a mig flux marca la petició com a fallida i en guarda el missatge.

Conseqüència per a la bossa compartida (`--kv-unified`, opció `--shared-context` de Jaull): una persona pot fer servir
tot el context si les altres no el necessiten, però si la suma de les converses supera la bossa, **fallen totes les
que estan en marxa**, no només la que s'ha passat. Amb slots fixos, una conversa massa llarga hauria de fallar
només ella, però cada persona té menys context. **UNVERIFIED**: no s'ha provat.

## Crides a eines i son d'inactivitat (2026-11-16)

Servidor arrencat per Jaull amb `unsloth/Qwen3.6-35B-A3B-GGUF -q IQ4_XS --users 4 --context 65536 --shared-context
--backend vulkan`, sense `--jinja` (Jaull no el passa). GPU prevista 18,3 GiB, límit de servei 20,2 GiB.

- **VALIDATED**: una petició amb una eina (`scripts/concurrency-bench/tools_probe.py`) retorna una crida real:
  `get_weather({"city":"Paris"})`, `finish_reason=tool_calls`, 2,6 s. El model de Biosfer pot, per tant, editar
  fitxers des d'opencode amb aquest servidor. No s'ha provat opencode mateix.
- **VALIDATED**: Jaull detecta la B60 com a Intel dedicada i passa `--sleep-idle-seconds 45` sense cap opció.
  Després de més de 60 s sense peticions, l'estat continuava `ready` (procés viu, mateix PID) i la mateixa petició va
  tardar 26,1 s, contra 2,6 s en calent: el cost de recarregar el model. Una sola mesura.
- **UNVERIFIED**: la VRAM i la RAM lliure durant l'adormiment no es van registrar en aquesta prova; el que es va
  veure és que el servidor responia bé. Que això eviti el problema de la RAM de la nota `DECISIONS-2.md` es va
  mesurar el 2026-11-04, no ara.

## opencode contra el servidor de Jaull (2026-11-17)

Portàtil de l'autor, Jaull dins de WSL2, `LFM2.5-1.2B-Instruct-GGUF` amb 16384 tokens de context, i opencode amb el
`opencode.json` que Jaull va afegir a la configuració que l'autor ja tenia.

- **VALIDATED**: la fusió va deixar intacte l'altre proveïdor, el tema i el model per defecte de l'autor, va afegir el
  proveïdor `jaull`, va crear la còpia `opencode.json.jaull-backup` i el fitxer va quedar amb mode 0600.
- **VALIDATED**: opencode compacta sol abans d'arribar al límit que Jaull li dóna. Amb dos usuaris i una bossa
  compartida de 32.768 tokens, Jaull escriu `limit.context = 16384` (la meitat de la bossa, `SERVE_CLIENT_POOL_SHARE`),
  de manera que cap client pot omplir-la sol. En un prompt llarg que feia escriure 24 capítols amb eines, opencode va
  mostrar la línia «Compaction» amb 12.834 tokens (78 % de 16.384), i el monitor de Jaull va mostrar aquella
  conversa amb 1.875 tokens just després i 10.162 poc més tard, amb la bossa al 31 % i l'altra conversa en repòs. El
  client no va passar dels 16.384, així que mai no es va veure el límit de 32.768 del servidor.
- Limitat: dos usuaris, un sol model petit, i no es va guardar el log del servidor d'aquesta prova. No s'ha provat
  què passa si un client ignora el límit que li dóna la configuració.
- Una sola prova, amb un model de 1,2B. No s'ha repetit amb el Qwen3.6 de Biosfer ni amb diversos usuaris alhora.
- **UNVERIFIED**: que això protegeixi els altres usuaris d'una bossa compartida (`--shared-context`). Aquesta prova va
  ser amb un sol client.
- El model de 1,2B va respondre que no podia llistar carpetes tot i tenir les eines: tria malament les eines. No és un
  problema del servidor, però vol dir que aquesta prova valida la compactació i no la utilitat de l'agent.
- Només opencode mateix gasta uns 8.700 tokens (sistema i eines) abans de la primera pregunta: amb 16384 de context
  queda poc marge. Es recomana 32768 o més.
- Amb HTTPS i el certificat propi de Jaull, opencode no es refia del certificat i cal llançar-lo amb
  `NODE_EXTRA_CA_CERTS=<certificat> opencode`. Amb aquesta variable va funcionar.

## El son d'inactivitat i el monitor (2026-11-17)

Prova al portàtil, CPU, llama.cpp b11889, Qwen2.5-1.5B amb `--sleep-idle-seconds 10`, amb el servidor adormit abans de
cada consulta (`/props` informa de `is_sleeping` sense despertar-lo).

- **VALIDATED**: `/health`, `/models`, `/props` i `/metrics` no el desperten; `/slots` sí. El monitor de Jaull demanava
  `/slots` a cada refresc, de manera que reactivava el model sempre i, com que la recàrrega triga, vencia el temps
  d'espera de 2 s i mostrava «Not answering». A la B60, un servidor sa es va aturar per aquest motiu (l'autor va prémer
  Stop a les 10:58:05, sense cap fallada: no hi ha rastre a `dmesg`).
- Corregit: el monitor pregunta abans a `/props` i, si dorm, no demana `/slots`.
- A WSL2 amb Dozen, recarregar un Qwen3-Coder-30B va trigar uns 2 minuts (la primera càrrega, 2 min 3 s). A Windows,
  el Qwen3.6 de 17 GiB va trigar uns 31 s més 48 s. El cost del son és molt més alt a WSL2.
- **A WSL2 amb Dozen el son no va protegir** (Qwen3-Coder 30B, 2026-11-17, segons l'autor): el servidor va entrar en
  repòs 93 s després d'acabar la càrrega (1:28 a 3:02), no als 45 s, i el PC va quedar amb la RAM plena i molt de lag; es va
  haver de matar el procés. El 9B de la mateixa màquina havia dormit als 40 s sense lag. A Windows natiu va dormir als
  40-46 s. Per això Jaull ja no el posa per defecte dins de WSL; `--sleep-idle N` continua disponible.
- **UNVERIFIED**: quan va començar el lag respecte a l'entrada en repòs (abans o després) i per què el repòs va trigar
  93 s; no es va registrar la RAM amb marques de temps. L'aturada a Windows nativa és la que es va mesurar.
- **UNVERIFIED**: que el mateix passi amb altres builds de llama.cpp.

## La RAM en carregar amb Jaull a Windows natiu (2026-11-17)

Qwen3.6-35B-A3B `IQ4_XS` (17,7 GB), B60, Vulkan, PC de 27,9 GB, 2 usuaris de 32768, servidor engegat des de la TUI de
Jaull a Windows, sense `-lm dio` (Jaull encara no el passava). Mesurat amb `memwatch.ps1` cada 6 s.

- **VALIDATED** (una execució): abans de carregar hi havia 8,5 GB de RAM lliure. En carregar, el procés va passar a
  12,2 GB residents i 17,7 GB de memòria privada, i la RAM lliure va baixar a **0,5 GB**, amb 16,9 GB a la VRAM de la
  B60. El monitor de Jaull va avisar «System RAM is running low». L'autor va aturar el servidor per això.
- Coincideix amb el que ja es va mesurar a X004: el mode per defecte (mmap) porta el GGUF a la RAM encara que les capes
  acabin a la targeta. Amb `-lm dio` no ho feia (0,3-0,4 GB residents, 8 execucions).
- Conseqüència: Jaull passa ara `-lm dio` per defecte a Windows natiu quan el model cap sencer a la GPU. **No s'ha
  tornat a mesurar amb Jaull ni la velocitat de generació**; queda per a la propera prova.
- La prova del son (45 s) amb la RAM al límit no es va poder completar.

## Límits

- Prompts sintètics i iguals de llargs; un xat real té preguntes i respostes de mida diferent.
- El 2026-11-13, el client passava pel pont SSH entre WSL i Windows, que afegeix una mica de latència a cada token. El
  2026-11-16 es va executar a Windows, al mateix PC que el servidor.
- Una sola màquina i una sola configuració del servidor.
