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

## Predicció de Jaull (feta abans de mesurar)

KV + estat recurrent del Qwen3.6-35B-A3B a 32.768 tokens per usuari, calculat amb `estimate_kv_cache` el 2026-11-13
(confiança mitjana):

| Usuaris | Predicció (MiB) | Mesura de llama.cpp (MiB) |
|---|---:|---:|
| 1 | 702,8 | **PENDENT** |
| 2 | 1.405,6 | **PENDENT** |
| 4 | 2.811,2 | **PENDENT** |

A un usuari, 640 MiB són KV (ja mesurat a D125) i uns 63 MiB, estat recurrent. La mesura es llegeix al registre del
servidor (`KV buffer size` i `RS buffer size`).

## Resultats

**PENDENT.** Taula per nivell: TTFT mediana, tok/s per usuari (mediana), tok/s agregats (mediana de les rondes) i
peticions fallides.

## Límits

- Prompts sintètics i iguals de llargs; un xat real té preguntes i respostes de mida diferent.
- El client passa pel pont SSH entre WSL i Windows, que afegeix una mica de latència a cada token.
- Una sola màquina i una sola configuració del servidor.
