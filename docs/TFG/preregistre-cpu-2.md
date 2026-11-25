# Preregistre de la campanya CPU 2: `-lm none` per defecte, models repetits i models nous

Escrit el 2026-11-25 **abans de fer cap mesura d'aquesta campanya**. Les prediccions són a
`validation/cpu-v2-20261125/predicciones/` i es van generar amb Jaull al commit `d62f924`, amb `src/` sense canvis locals.
**La fórmula de l'estimador no ha canviat** respecte a la campanya CPU 1 (`preregistre-cpu.md`): els 24 casos repetits tenen exactament les mateixes
prediccions. Només ha canviat el mode de càrrega que Jaull recomana a CPU sobre Windows (D129).

## Què es vol comprovar

1. Que, amb `-lm none`, el consum de memòria **coincideix amb l'estimació** també en una repetició i en models que no s'havien vist.
2. Que la diferència entre mmap i sense mmap **es repeteix** i, si es pot, **amb quin model passa**.
3. Que la primera petició amb mmap és lenta perquè **paga la lectura del model** (i no perquè mmap vagi més lent). Per això aquesta campanya fa una petició de calentament abans de mesurar.

## Què es mesura

La mateixa màquina (Ryzen 5 5500U, 15,7 GiB de RAM, Windows natiu, WSL apagat) i el mateix llama.cpp b11889 només CPU, amb els mateixos scripts de
`scripts/cpu/windows/`, amb tres canvis que no afecten les definicions anteriors: carpeta de sortida pròpia (`resultats-v2`), la **petició de calentament**
(`-Warmup`: 32 tokens amb un altre prompt, abans de la petició que es mesura) i el temps de càrrega i el detall del prompt al resum.
Els 128 tokens, els 6 fils, el context i els llindars de seguretat (400 MiB de RAM disponible, 3.000 MiB de fitxer de paginació) són els mateixos.
No es mesura la velocitat amb `llama-bench`.

**RAM disponible objectiu:** entre 7.000 i 9.000 MiB just abans de cada cas, com a la passada B, amb aplicacions obertes i sense WSL.
Es mesura abans de cada cas i el veredicte de Jaull es calcula amb aquest valor. Si no es compleix, es presenta igualment amb la RAM real.

## Els casos (45 execucions)

Els 24 casos C01 a C24 són els de la campanya 1, amb les mateixes prediccions. A C01, C05 i C12 s'hi afegeix mmap, que no s'havia mesurat en aquests models.
Els casos N01 a N05 són **models nous** (no s'han mesurat mai): un Llama de 1,1B, un híbrid LFM2.5 de 1,2B i el Qwen2.5-Coder-7B en Q2_K.

| Cas | Model | Context | Modes | Pesos predits (MiB) | KV (MiB) | Overhead (MiB) | Total (MiB) |
|---|---|---:|---|---:|---:|---:|---:|
| N01 | TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 2.048 | sense mmap + amb mmap | 637,8 | 44,0 | 575,8 | 1.769,6 |
| N02 | LFM2.5-1.2B Q5_K_M | 8.192 | sense mmap + amb mmap | 804,3 | 256,0 | 592,4 | 2.164,7 |
| N03 | LFM2.5-1.2B Q5_K_M | 32.768 | sense mmap | 804,3 | 1.024,0 | 592,4 | 2.932,7 |
| C14 | Qwen2.5-0.5B Q4_K_M | 32.768 | sense mmap | 379,4 | 384,0 | 549,9 | 1.825,3 |
| C13 | Qwen2.5-1.5B Q5_K_M | 32.768 | sense mmap | 1.225,9 | 896,0 | 634,6 | 3.268,5 |
| C12 | Qwen2.5-3B Q4_K_M | 32.768 | sense mmap + amb mmap | 2.007,4 | 1.152,0 | 712,7 | 4.384,2 |
| N04 | Qwen2.5-Coder-7B Q2_K | 8.192 | sense mmap + amb mmap | 2.876,2 | 448,0 | 799,6 | 4.635,8 |
| N05 | Qwen2.5-Coder-7B Q2_K | 32.768 | sense mmap | 2.876,2 | 1.792,0 | 799,6 | 5.979,8 |
| C09 | Qwen2.5-7B IQ2_M | 8.192 | sense mmap | 2.651,5 | 448,0 | 777,2 | 4.388,7 |
| C05 | Qwen2.5-7B Q3_K_M | 2.048 | sense mmap + amb mmap | 3.632,0 | 112,0 | 875,2 | 5.131,2 |
| C06 | Qwen2.5-7B Q3_K_M | 8.192 | sense mmap | 3.632,0 | 448,0 | 875,2 | 5.467,2 |
| C10 | Qwen2.5-7B IQ2_M | 32.768 | sense mmap | 2.651,5 | 1.792,0 | 777,2 | 5.732,7 |
| C07 | Qwen2.5-7B Q3_K_M | 16.384 | sense mmap | 3.632,0 | 896,0 | 875,2 | 5.915,2 |
| C11 | Qwen2.5-Coder-7B Q3_K_M | 16.384 | sense mmap | 3.632,0 | 896,0 | 875,2 | 5.915,2 |
| C01 | Qwen2.5-7B Q4_K_M | 2.048 | sense mmap + amb mmap | 4.466,1 | 112,0 | 958,6 | 6.048,7 |
| C02 | Qwen2.5-7B Q4_K_M | 8.192 | sense mmap | 4.466,1 | 448,0 | 958,6 | 6.384,7 |
| C03 | Qwen2.5-7B Q4_K_M | 16.384 | sense mmap | 4.466,1 | 896,0 | 958,6 | 6.832,7 |
| C08 | Qwen2.5-7B Q3_K_M | 32.768 | sense mmap | 3.632,0 | 1.792,0 | 875,2 | 6.811,2 |
| C04 | Qwen2.5-7B Q4_K_M | 32.768 | sense mmap | 4.466,1 | 1.792,0 | 958,6 | 7.728,7 |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | sense mmap + amb mmap | 6.703,9 | 56,0 | 1.182,4 | 8.454,3 |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | sense mmap + amb mmap | 6.703,9 | 56,0 | 1.182,4 | 8.454,3 |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | sense mmap + amb mmap | 6.999,2 | 384,0 | 1.211,9 | 9.107,1 |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | sense mmap + amb mmap | 7.208,7 | 180,3 | 1.232,9 | 9.133,9 |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | sense mmap + amb mmap | 7.697,4 | 56,0 | 1.281,7 | 9.547,1 |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | sense mmap + amb mmap | 7.697,4 | 56,0 | 1.281,7 | 9.547,1 |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | sense mmap + amb mmap | 7.697,4 | 56,0 | 1.281,7 | 9.547,1 |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | sense mmap + amb mmap | 7.208,7 | 1.076,3 | 1.232,9 | 10.029,9 |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | sense mmap + amb mmap | 6.999,2 | 1.536,0 | 1.211,9 | 10.259,1 |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | sense mmap + amb mmap | 6.999,2 | 3.072,0 | 1.211,9 | 11.795,1 |

El total és pesos, KV, overhead i la reserva de 512 MiB.

## Regles de veredicte (fixades abans de mesurar)

- **Jaull diu «cap»** si el total predit és com a molt la RAM disponible mesurada abans d'engegar el servidor.
- **El cas funciona** si carrega, genera 128 tokens, la RAM disponible mínima és d'almenys 1.024 MiB i el fitxer de paginació creix com a molt 512 MiB.
- **Un cas parat pel límit de seguretat compta com a marge no assolit** (lectura adoptada després de la campanya 1, fixada aquí de bon principi). El límit de 400 MiB el posem nosaltres: no vol dir que el model no hagi pogut generar. La regla és simètrica: una aprovació de Jaull que es para és un fals positiu.
- La **línia base** és la regla «mida del GGUF × 1,2 és com a molt la RAM disponible».
- Es presenten **per separat** sense mmap i amb mmap, i no es barregen amb la campanya 1.

## Hipòtesis (escrites abans de mesurar)

| # | Hipòtesi | Es considera complerta si... |
|---|---|---|
| H1 | Sense mmap, Jaull no aprova res que no funcioni | Cap aprovació de Jaull en mode `-lm none` té marge no assolit. **Si n'hi ha una, D129 queda en dubte i s'investiga abans de res.** |
| H2 | Els buffers dels 24 casos repetits són els mateixos | Pesos, KV i còmput observats coincideixen amb la campanya 1 a ±1 MiB. |
| H3 | Als models nous, l'error del total és negatiu (Jaull és conservador) | L'error observat menys predit és negatiu a tots els casos nous sense mmap, i entre −700 i −300 MiB als tres models amb arquitectura coneguda (N01, N04, N05). |
| H4 | Als models nous els pesos s'acosten a la mida del fitxer | L'error dels pesos és com a molt l'1 % del fitxer. |
| H5 | L'LFM2.5, un híbrid, **no** té el KV ben predit | Obert: no es preveu un nombre. S'anotarà l'error del KV i si Jaull el reconeix. |
| H6 | El consum amb mmap és més alt on hi ha buffer `CPU_REPACK` | A N01, C12, C01 i C05, amb mmap la RAM disponible mínima és més baixa que sense mmap en almenys la meitat del buffer `CPU_REPACK` del registre. A N02 i N04 (Q5_K_M i Q2_K) no es fa cap predicció: es mira si hi ha `CPU_REPACK`. |
| H7 | Amb calentament, la primera petició amb mmap deixa de ser molt més lenta | Al Mellum (C15 a C19), el temps del prompt mesurat amb mmap és com a molt el doble que sense (a la campanya 1 va ser 5,9 vegades). |
| H8 | Sense mmap carregar triga poc més | El temps de càrrega sense mmap és entre 0 i 10 s més que amb mmap. |

## Què no es farà

- No es modificarà l'estimador ni cap constant durant ni després d'aquesta campanya (la propera calibració de l'overhead de CPU es farà amb una altra campanya).
- No s'afegiran ni es trauran casos un cop començada. Si un cas falla per un error del script (i no del model), es repeteix i es diu.
- Els resultats es presenten tal com surtin, també si H1 falla.
