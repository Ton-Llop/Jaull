# Resultats de la campanya CPU 2

Mateix portàtil (Ryzen 5 5500U, 15,7 GiB de RAM, Windows natiu, WSL apagat) i mateix llama.cpp b11889 només CPU.
Preregistre: `preregistre-cpu-2.md`, amb les prediccions de Jaull comitejades abans de la primera mesura. La fórmula de l'estimador no ha canviat.
Hi ha 45 execucions (24 configuracions de la campanya 1 repetides, més 5 de models nous) amb una petició de calentament abans de mesurar.
RAM disponible abans de cada cas: de 6.881 a 10.581 MiB.

## En resum

- **Sense mmap, Jaull no ha aprovat mai res que no funcionés.** 28 aprovacions i 28 van funcionar amb marge, amb un mínim de 1.745 MiB lliures (C21).
  L'aprovació més justa (C21) tenia només 89 MiB de marge sobre l'estimació de Jaull i va quedar 1.745 MiB lliures. Cap fals negatiu.
- **Els buffers de les 24 configuracions repetides són idèntics als de la campanya 1** (diferència màxima: 0,0 MiB).
- **Amb el mode de càrrega que Jaull fa servir a CPU sobre Windows (`-lm none`), no hi ha hagut cap fals positiu en 66 aprovacions de dues campanyes** (38 i 28). L'error del total és sempre del costat prudent; el KV del Mellum es queda curt, i el tapa l'overhead sobrat.
- **Amb mmap hi ha una limitació coneguda:** Jaull aprova 15 configuracions i 8 acaben sense marge (C01, C15, C16, C17, C18, C19, C20, C21), amb les mateixes prediccions. Només afecta qui força mmap o treballa a Linux o WSL, on no s'ha mesurat.
- **Els models nous confirmen l'overhead conservador** (error del total entre −522 i −651 MiB als tres models amb arquitectura coneguda) **i un error nou**: Jaull sobreestima el KV de l'LFM2.5, un híbrid.
- **Sense mmap, la regla simple (GGUF × 1,2) té aquí un fals positiu i Jaull cap:** aprova la C22, que va quedar a 444 MiB lliures, perquè no mira el KV. A la campanya 1 va ser al revés (Jaull en tenia 4 de falsos negatius i la regla 1).
- **No s'ha pogut comprovar** la hipòtesi H7 al Mellum: amb mmap es para abans de generar.

## Ha encertat Jaull?

### Resum

Sense mmap, que és el mode amb què Jaull carrega el model a CPU sobre Windows (D129):

| Cas | Model | Context | Predit | Observat | Ha encertat? |
|---|---|---:|---:|---:|:-:|
| N01 | TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 2.048 | 1.258 | 735 | ✅ |
| N02 | LFM2.5-1.2B Q5_K_M | 8.192 | 1.653 | 974 | ✅ |
| N03 | LFM2.5-1.2B Q5_K_M | 32.768 | 2.421 | 1.286 | ✅ |
| C14 | Qwen2.5-0.5B Q4_K_M | 32.768 | 1.313 | 827 | ✅ |
| C13 | Qwen2.5-1.5B Q5_K_M | 32.768 | 2.756 | 2.216 | ✅ |
| C12 | Qwen2.5-3B Q4_K_M | 32.768 | 3.872 | 3.270 | ✅ |
| N04 | Qwen2.5-Coder-7B Q2_K | 8.192 | 4.124 | 3.473 | ✅ |
| N05 | Qwen2.5-Coder-7B Q2_K | 32.768 | 5.468 | 4.841 | ✅ |
| C09 | Qwen2.5-7B IQ2_M | 8.192 | 3.877 | 3.248 | ✅ |
| C05 | Qwen2.5-7B Q3_K_M | 2.048 | 4.619 | 3.886 | ✅ |
| C06 | Qwen2.5-7B Q3_K_M | 8.192 | 4.955 | 4.228 | ✅ |
| C10 | Qwen2.5-7B IQ2_M | 32.768 | 5.221 | 4.616 | ✅ |
| C07 | Qwen2.5-7B Q3_K_M | 16.384 | 5.403 | 4.684 | ✅ |
| C11 | Qwen2.5-Coder-7B Q3_K_M | 16.384 | 5.403 | 4.684 | ✅ |
| C01 | Qwen2.5-7B Q4_K_M | 2.048 | 5.537 | 4.720 | ✅ |
| C02 | Qwen2.5-7B Q4_K_M | 8.192 | 5.873 | 5.062 | ✅ |
| C03 | Qwen2.5-7B Q4_K_M | 16.384 | 6.321 | 5.518 | ✅ |
| C08 | Qwen2.5-7B Q3_K_M | 32.768 | 6.299 | 5.596 | ✅ |
| C04 | Qwen2.5-7B Q4_K_M | 32.768 | 7.217 | 6.430 | ✅ |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | 7.942 | 6.986 | ✅ |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | 7.942 | 7.346 | ✅ |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | 8.595 | 7.510 | ✅ |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | 8.622 | 7.302 | ✅ |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | 9.035 | 7.889 | ✅ |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | 9.035 | 7.979 | ✅ |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | 9.035 | 8.339 | ✅ |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | 9.518 | 8.226 | ✅ |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | 9.747 | 8.668 | ✅ |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | 11.283 | 10.212 | ✅ |
| **Encerts** | | | | | **29 de 29** |

✅ és un encert amb el criteri de marge (almenys 1.024 MiB lliures): aprovar una configuració que funciona o rebutjar-ne una que no. ❌ FP és un fals positiu: Jaull va dir «cap» i el cas es va quedar sense marge.
**Predit** és el que Jaull calcula per a pesos, KV i overhead, en MiB, i **Observat** és el que va reservar llama.cpp. El veredicte «cap» fa servir, a més, la reserva de 512 MiB i la RAM lliure mesurada a cada cas.
El 29 de 29 són 28 aprovacions i 1 rebuig correctes. **Amb mmap hi ha una limitació coneguda, a la secció «Limitació coneguda: mmap».**

### Detall

Una fila per configuració. **Marge** és la RAM lliure menys el total predit de Jaull (positiu vol dir «cap»). **Mínim** és la RAM lliure més baixa durant l'execució, i **parat** vol dir que l'script va aturar el servidor al límit de 400 MiB que vam posar nosaltres.
«Encert» segueix el criteri de marge (almenys 1.024 MiB lliures): aprovar una configuració que funciona o rebutjar-ne una que no. Els errors van en negreta.

| Cas | Model | Context | Total predit | Error de memòria | Sense mmap: RAM lliure | Marge de Jaull | Mínim | Resultat | Amb mmap: RAM lliure | Marge de Jaull | Mínim | Resultat |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---|---:|---|
| N01 | TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 2.048 | 1.770 | −522 | 7.821 | +6.051, cap | 7.110 | Encert | 7.816 | +6.046, cap | 6.704 | Encert |
| N02 | LFM2.5-1.2B Q5_K_M | 8.192 | 2.165 | −678 | 7.509 | +5.344, cap | 6.569 | Encert | 7.650 | +5.485, cap | 6.710 | Encert |
| N03 | LFM2.5-1.2B Q5_K_M | 32.768 | 2.933 | −1.134 | 7.724 | +4.791, cap | 6.495 | Encert |  | | | | |
| C14 | Qwen2.5-0.5B Q4_K_M | 32.768 | 1.825 | −486 | 7.585 | +5.760, cap | 6.748 | Encert |  | | | | |
| C13 | Qwen2.5-1.5B Q5_K_M | 32.768 | 3.268 | −541 | 7.645 | +4.376, cap | 5.441 | Encert |  | | | | |
| C12 | Qwen2.5-3B Q4_K_M | 32.768 | 4.384 | −602 | 7.633 | +3.249, cap | 4.314 | Encert | 7.608 | +3.224, cap | 3.245 | Encert |
| N04 | Qwen2.5-Coder-7B Q2_K | 8.192 | 4.636 | −651 | 6.881 | +2.245, cap | 3.683 | Encert | 9.229 | +4.593, cap | 4.167 | Encert |
| N05 | Qwen2.5-Coder-7B Q2_K | 32.768 | 5.980 | −627 | 7.560 | +1.580, cap | 2.631 | Encert |  | | | | |
| C09 | Qwen2.5-7B IQ2_M | 8.192 | 4.389 | −629 | 7.767 | +3.378, cap | 4.461 | Encert |  | | | | |
| C05 | Qwen2.5-7B Q3_K_M | 2.048 | 5.131 | −733 | 7.621 | +2.490, cap | 3.882 | Encert | 7.925 | +2.794, cap | 3.133 | Encert |
| C06 | Qwen2.5-7B Q3_K_M | 8.192 | 5.467 | −727 | 7.968 | +2.501, cap | 3.798 | Encert |  | | | | |
| C10 | Qwen2.5-7B IQ2_M | 32.768 | 5.733 | −605 | 7.974 | +2.241, cap | 2.752 | Encert |  | | | | |
| C07 | Qwen2.5-7B Q3_K_M | 16.384 | 5.915 | −719 | 8.369 | +2.454, cap | 3.119 | Encert |  | | | | |
| C11 | Qwen2.5-Coder-7B Q3_K_M | 16.384 | 5.915 | −719 | 8.005 | +2.090, cap | 3.324 | Encert |  | | | | |
| C01 | Qwen2.5-7B Q4_K_M | 2.048 | 6.049 | −816 | 7.981 | +1.932, cap | 3.394 | Encert | 8.102 | +2.053, cap | 785 | **Fals positiu** |
| C02 | Qwen2.5-7B Q4_K_M | 8.192 | 6.385 | −810 | 8.135 | +1.750, cap | 3.141 | Encert |  | | | | |
| C03 | Qwen2.5-7B Q4_K_M | 16.384 | 6.833 | −802 | 7.991 | +1.158, cap | 2.536 | Encert |  | | | | |
| C08 | Qwen2.5-7B Q3_K_M | 32.768 | 6.811 | −703 | 8.393 | +1.582, cap | 2.766 | Encert |  | | | | |
| C04 | Qwen2.5-7B Q4_K_M | 32.768 | 7.729 | −786 | 8.592 | +863, cap | 2.338 | Encert |  | | | | |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | 8.454 | −957 | 8.690 | +236, cap | 1.849 | Encert | 8.750 | +296, cap | **parat** (8) | **Fals positiu** |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | 8.454 | −597 | 9.263 | +809, cap | 2.017 | Encert | 9.304 | +850, cap | **parat** (279) | **Fals positiu** |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | 9.107 | −1.085 | 9.468 | +361, cap | 1.987 | Encert | 9.666 | +559, cap | 445 | **Fals positiu** |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | 9.134 | −1.320 | 9.594 | +460, cap | 2.282 | Encert | 9.776 | +642, cap | 3.152 | Encert |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | 9.547 | −1.146 | 9.837 | +290, cap | 1.963 | Encert | 9.891 | +344, cap | **parat** (301) | **Fals positiu** |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | 9.547 | −1.056 | 10.100 | +553, cap | 2.210 | Encert | 10.097 | +550, cap | **parat** (312) | **Fals positiu** |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | 9.547 | −696 | 10.127 | +580, cap | 1.945 | Encert | 10.173 | +626, cap | **parat** (304) | **Fals positiu** |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | 10.030 | −1.292 | 10.315 | +285, cap | 1.938 | Encert | 10.419 | +389, cap | 2.781 | Encert |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | 10.259 | −1.079 | 10.348 | +89, cap | 1.745 | Encert | 10.409 | +150, cap | 510 | **Fals positiu** |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | 11.795 | −1.071 | 10.535 | −1.260, no | 444 | Encert | 10.581 | −1.214, no | 488 | Encert |

L'**error de memòria** és el que llama.cpp ha reservat menys el que Jaull havia predit (pesos, KV i còmput), sense mmap. Els casos amb mmap que es van parar abans de carregar no en tenen.

### Matriu de confusió

| Mode | Mètode | Aprova i funciona | Fals positiu | Fals negatiu | Rebutja i no cap |
|---|---|---:|---:|---:|---:|
| sense mmap | Jaull | 28 | 0 | 0 | 1 |
| sense mmap | GGUF × 1,2 | 28 | 1 | 0 | 0 |
| amb mmap | Jaull | 7 | 8 | 0 | 1 |
| amb mmap | GGUF × 1,2 | 7 | 9 | 0 | 0 |

Sense mmap l'única diferència és la C22 (el 14B a 16.384 tokens): Jaull la rebutja amb raó (11.795 MiB predits contra 10.535 lliures) i el servidor va generar però amb només 444 MiB lliures; la regla simple l'aprova, perquè no té en compte el KV.
Amb mmap la regla simple té un fals positiu més (la C22, també rebutjada per Jaull).

## Models nous

| Cas | Model | Context | Error dels pesos | KV observat contra predit (MiB) | Còmput (MiB) | Error del total |
|---|---|---:|---|---|---:|---:|
| N01 | TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 2.048 | −1,6 (−0,26 %) | 44,0 contra 44,0 | 55 | −522 |
| N02 | LFM2.5-1.2B Q5_K_M | 8.192 | −2,3 (−0,28 %) | 96,2 contra 256,0 | 76 | −678 |
| N03 | LFM2.5-1.2B Q5_K_M | 32.768 | −2,3 (−0,28 %) | 384,2 contra 1.024,0 | 100 | −1.134 |
| N04 | Qwen2.5-Coder-7B Q2_K | 8.192 | −5,7 (−0,20 %) | 448,0 contra 448,0 | 154 | −651 |
| N05 | Qwen2.5-Coder-7B Q2_K | 32.768 | −5,7 (−0,20 %) | 1.792,0 contra 1.792,0 | 178 | −627 |

- **Pesos:** dins de ±0,3 % de la predicció als cinc (H4 complerta).
- **Overhead:** l'error del total és negatiu a tots, i entre −522 i −651 MiB a TinyLlama i al Coder-7B Q2_K (H3 complerta). Al LFM2.5 és de −678 i −1.134 MiB per l'error del KV.
- **LFM2.5 (H5):** Jaull n'estima el KV com si totes les capes tinguessin atenció i el sobreestima unes 2,7 vegades (96 MiB observats contra 256 predits a 8.192 tokens, 384 contra 1.024 a 32.768).
  La proporció és d'aproximadament 0,375, que és el que sortiria si només 6 de les 16 capes en tinguessin; **no s'ha comprovat a l'arquitectura del model**. És un error conservador, de la mateixa família que el del Mellum (amb signe contrari).

## Limitació coneguda: mmap

**Jaull no carrega amb mmap per defecte, però només a CPU sobre Windows natiu.** A Linux i a WSL el mode per defecte de llama.cpp és mmap i no s'ha mesurat allà. Qui demani `-lm mmap` o executi llama.cpp pel seu compte també hi cau.
Amb mmap, Jaull aprova 15 configuracions i 8 acaben sense marge, amb les mateixes prediccions que funcionen sense mmap:

| Cas | Model | Context | Predit | Observat | Ha encertat? |
|---|---|---:|---:|---:|:-:|
| N01 | TinyLlama-1.1B-Chat-v1.0 Q4_K_M | 2.048 | 1.258 | 735 | ✅ |
| N02 | LFM2.5-1.2B Q5_K_M | 8.192 | 1.653 | 974 | ✅ |
| C12 | Qwen2.5-3B Q4_K_M | 32.768 | 3.872 | 3.270 | ✅ |
| N04 | Qwen2.5-Coder-7B Q2_K | 8.192 | 4.124 | 3.473 | ✅ |
| C05 | Qwen2.5-7B Q3_K_M | 2.048 | 4.619 | 3.886 | ✅ |
| C01 | Qwen2.5-7B Q4_K_M | 2.048 | 5.537 | 4.720 | ❌ FP |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | 7.942 | 6.986 | ❌ FP |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | 7.942 | 7.346 | ❌ FP |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | 8.595 | 7.510 | ❌ FP |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | 8.622 | 7.302 | ✅ |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | 9.035 | 7.889 | ❌ FP |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | 9.035 | 7.979 | ❌ FP |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | 9.035 | 8.339 | ❌ FP |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | 9.518 | 8.226 | ✅ |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | 9.747 | 8.668 | ❌ FP |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | 11.283 | 10.212 | ✅ |
| **Encerts** | | | | | **8 de 16** |

Els tres fals positius que van executar (C01, C20, C21) van generar, amb 785, 445 i 510 MiB lliures. Els cinc del Mellum es van parar pel límit de 400 MiB que vam posar nosaltres: no se sap si haurien generat.
Amb el criteri «executa» els falsos positius confirmats serien 0 i els 5 del Mellum quedarien com a desconeguts; amb el criteri de marge són 8. Aquesta taula fa servir el segon.

### Diferència entre modes

Diferència de RAM lliure mínima (sense mmap menys amb mmap, en MiB), amb el buffer `CPU_REPACK` que imprimeix el registre:

| Cas | Quantització | Buffer REPACK | Mínim sense mmap | Mínim amb mmap | Amb mmap gasta de més |
|---|---|---:|---:|---:|---:|
| C01 | Q4_K_M | 2.977 | 3.394 | 785 | +2.609 |
| C05 | Q3_K_M | 1.202 | 3.882 | 3.133 | +749 |
| C12 | Q4_K_M | 1.266 | 4.314 | 3.245 | +1.069 |
| C15 | Q4_K_M | 4.279 | 1.963 | 301 (parat) | +1.662 |
| C16 | Q4_K_M | 4.279 | 2.210 | 312 (parat) | +1.898 |
| C17 | Q4_K_M | 4.279 | 1.945 | 304 (parat) | +1.641 |
| C18 | MXFP4 | 5.623 | 1.849 | 8 (parat) | +1.841 |
| C19 | MXFP4 | 5.623 | 2.017 | 279 (parat) | +1.738 |
| C20 | Q3_K_M | 2.513 | 1.987 | 445 | +1.542 |
| C21 | Q3_K_M | 2.513 | 1.745 | 510 | +1.235 |
| C22 | Q3_K_M | 2.513 | 444 | 488 | −44 |
| C23 | Q6_K | no | 2.282 | 3.152 | −870 |
| C24 | Q6_K | no | 1.938 | 2.781 | −843 |
| N01 | Q4_K_M | 455 | 7.110 | 6.704 | +406 |
| N02 | Q5_K_M | no | 6.569 | 6.710 | −141 |
| N04 | Q2_K | no | 3.683 | 4.167 | −484 |

- **H6 complerta:** a N01, C12, C01 i C05 (Q4_K_M i Q3_K_M amb `CPU_REPACK`), amb mmap la RAM lliure baixa entre el 62 % i el 89 % del buffer REPACK, més que la meitat que s'havia previst.
- **Sense `CPU_REPACK` passa al revés,** com a la campanya 1: el Q6_K, el Q5_K_M de l'LFM2.5 i el Q2_K del Coder-7B (que només té un buffer REPACK de 28 MiB) gasten menys amb mmap.
- **Al Mellum la diferència queda tallada** perquè amb mmap el servidor es para (de 5 casos, 5 s'aturen abans d'acabar de generar): els valors de la taula són un mínim.
- **Hi ha una limitació amb Q2_K:** no se sap si el Q2_K reordena poc o el model és massa petit per notar-ho; només s'ha provat amb un model.

## Velocitat, càrrega i primera petició

Casos amb mmap que van arribar a generar. Generació en tok/s; prompt en ms (petició de 16 a 20 tokens, mesurada **després** del calentament); càrrega en segons.

| Cas | Generació sense mmap | Generació amb mmap | Dif. | Prompt sense mmap (ms) | Prompt amb mmap (ms) | Càrrega sense (s) | Càrrega amb (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C01 | 7,9 | 7,9 | −0 % | 492 | 495 | 9 | 6 |
| C05 | 9,3 | 9,3 | +0 % | 551 | 566 | 6 | 3 |
| C12 | 15,2 | 11,8 | −22 % | 271 | 280 | 5 | 3 |
| C20 | 4,8 | 4,8 | −0 % | 1.050 | 1.077 | 11 | 10 |
| C21 | 4,8 | 4,8 | −1 % | 1.125 | 1.086 | 11 | 10 |
| C22 | 4,8 | 4,8 | +0 % | 1.074 | 1.113 | 11 | 15 |
| C23 | 4,7 | 4,8 | +2 % | 1.079 | 1.062 | 10 | 6 |
| C24 | 4,8 | 4,8 | 0 % | 1.061 | 1.077 | 10 | 7 |
| N01 | 47,7 | 47,4 | −1 % | 94 | 103 | 2 | 2 |
| N02 | 32,9 | 36,8 | +12 % | 180 | 152 | 2 | 2 |
| N04 | 10,6 | 9,6 | −10 % | 609 | 1.055 | 5 | 2 |

- **El prompt amb calentament és igual en els dos modes a 8 de 11 casos** (diferències de ±5 %); a la resta, +9 % (N01), −16 % (N02) i +73 % (N04, 609 contra 1.055 ms). A la campanya 1 la diferència era de 6 vegades al Mellum i del 17 % al 14B: el calentament la fa desaparèixer als models on es pot comprovar, que és coherent amb que la primera petició pagués la lectura del model.
  **H7 no es pot avaluar tal com es va escriure**, perquè estava pensada per al Mellum i amb mmap el Mellum no arriba a generar. Només s'ha comprovat en altres models, i la N04 és una excepció sense explicació.
- **La generació no té una diferència sistemàtica:** de −22 % a +12 % segons el model (una mesura per cas, sense repeticions). La frase de la campanya 1, «igual o millor sense mmap», no es pot mantenir tal com estava escrita.
- **Càrrega:** sense mmap triga entre 0 i 4 s més en 12 dels 13 casos que van arribar a carregar amb els dos modes (H8); dos d'ells, C15 i C17, no són a la taula perquè amb mmap es van parar abans de generar. La C22 va al revés: 15 s amb mmap contra 11 sense.

## Hipòtesis del preregistre

| # | Hipòtesi | Resultat |
|---|---|---|
| H1 | Sense mmap, cap aprovació falla el marge | **Complerta.** 28 aprovacions, 0 falsos positius. |
| H2 | Els buffers repetits són els mateixos (±1 MiB) | **Complerta.** Diferència màxima de 0,0 MiB a 24 casos. |
| H3 | Error del total negatiu als models nous, entre −700 i −300 als tres coneguts | **Complerta.** −522 (N01), −651 (N04), −627 (N05). |
| H4 | Pesos dins de l'1 % del fitxer | **Complerta** (−0,20 % a −0,28 %). |
| H5 | L'LFM2.5 no té el KV ben predit (obert) | **Observat:** el sobreestima unes 2,7 vegades. |
| H6 | Més consum amb mmap on hi ha `CPU_REPACK` (≥ meitat del buffer) | **Complerta** als 4 casos previstos (62 % a 89 %). |
| H7 | Amb calentament, el Mellum amb mmap té un prompt ≤ el doble | **No avaluable:** el Mellum es para amb mmap. Als altres models el prompt és igual en els dos modes. |
| H8 | Sense mmap carregar triga entre 0 i 10 s més | **Complerta a 12 de 13 casos** (la C22 és −4 s). |

## Què canvia respecte a la campanya 1

- **D129 queda validada a una màquina:** la predicció de H1 era el criteri, i s'ha complert sense cap excepció, també als models que no s'havien vist mai.
- **Correcció:** a D129 i a `resultados-cpu.md` es deia que la generació és «igual o millor sense mmap». Amb aquesta campanya hi ha un cas on mmap és un 12 % més ràpid i un altre on és un 22 % més lent. La raó de D129 és la memòria, no la velocitat.
- **Nou error de Jaull, conservador:** el KV dels models híbrids que no tenen atenció a totes les capes (LFM2.5).

## Limitacions

- Una sola màquina, un llama.cpp i una execució per cas: la diferència de velocitat de ±20 % pot ser soroll.
- La RAM disponible va pujar durant la campanya (fins a 10.581 MiB); el rang objectiu era de 7.000 a 9.000 MiB, i **no es va complir als darrers casos** (C15 a C22, amb 9.800 a 10.600). No s'ha investigat per què va pujar. Cada veredicte es calcula amb la RAM del seu cas.
- El Mellum amb mmap no arriba a generar, de manera que H7 queda oberta.
- Els casos parats es van aturar pel límit de seguretat nostre: se sap que no van arribar al marge, no si haurien pogut generar.
- Tres dels cinc casos nous (N01 a N03) són models petits i queden lluny del límit: validen pesos, KV i overhead, no el comportament al límit.
- La causa de l'extra de mmap segueix sent una hipòtesi sustentada en les mesures; no s'ha llegit el codi de llama.cpp.
