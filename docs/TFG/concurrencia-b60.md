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

## Límits

- Prompts sintètics i iguals de llargs; un xat real té preguntes i respostes de mida diferent.
- El client passa pel pont SSH entre WSL i Windows, que afegeix una mica de latència a cada token.
- Una sola màquina i una sola configuració del servidor.
