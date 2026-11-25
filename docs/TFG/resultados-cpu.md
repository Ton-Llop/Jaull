# Resultats de la campanya CPU

Portàtil amb AMD Ryzen 5 5500U (6 nuclis, 15,7 GiB de RAM, DDR4 a 3.200 MT/s), Windows natiu i llama.cpp b11889 només CPU.
Les prediccions de Jaull es van comitejar abans de la primera mesura (`preregistre-cpu.md`, commit `1ad8027`, a les 10:56; la primera mesura és de les 11:29).
La passada B es va fer amb un addendum escrit abans (`preregistre-cpu-addendum.md`). Els scripts són a `scripts/cpu/` i la configuració dels models a `scripts/cpu/cases.csv`.

## En resum

- **Sense mmap, Jaull no ha aprovat mai res que no funcionés.** A la passada A en va aprovar 23 configuracions i totes van funcionar amb marge.
  A la B, amb la RAM més ajustada, en va aprovar 15 i també van funcionar, dues amb només 90 i 149 MiB de marge sobre la seva estimació.
- **Els rebutjos són prudents.** En 6 casos Jaull deia que no hi cabien i la RAM lliure va baixar de 400 MiB abans d'acabar la càrrega.
  Només en 3 casos va rebutjar una configuració que hauria funcionat amb marge, i sempre de poc (C18 (536 MiB per sota), C19 (26 MiB per sota), C23 (289 MiB per sota)).
- **Als Qwen, els pesos i el KV són exactes i l'error és l'overhead.** Jaull sobreestima la memòria de CPU entre 486 i 1.320 MiB,
  sempre en el mateix sentit. Ve de l'overhead general (512 MiB més el 10 % dels pesos), que a CPU no s'havia calibrat mai.
- **Hi ha un error real al KV del Mellum.** Jaull el limita a la finestra lliscant de 1.024 tokens i en prediu 56 MiB. llama.cpp en reserva 91, 175 i 511 MiB a 2.048, 8.192 i 32.768 tokens.
- **Amb mmap la memòria lliure s'esgota molt abans.** Aquí Jaull sí que dona falsos positius del marge (7 en total). La causa probable és el reordenament dels pesos (`CPU_REPACK`): amb mmap, l'original continua a la memòria.
- **Una regla d'una línia (GGUF × 1,2) no queda pitjor que Jaull** sense mmap: el seu valor és explicar els errors, no encertar més.

## Ha encertat Jaull?

Una fila per configuració, amb la passada A (molta RAM) i la B (RAM ajustada) una al costat de l'altra.
**Marge de Jaull** és la RAM lliure menys el seu total predit: si és positiu, Jaull diu «cap». **Error de memòria** és el que llama.cpp ha reservat menys el que Jaull havia predit (pesos, KV i còmput).
**Mínim lliure** és la RAM lliure més baixa durant l'execució. «Ha encertat?» segueix el criteri de marge (almenys 1.024 MiB lliures):
**Encert** és aprovar una configuració que funciona o rebutjar-ne una que no, i els errors van en negreta.
**Parat** vol dir que l'script va aturar el servidor al límit de seguretat de 400 MiB, que el vam posar nosaltres, i es compta com a marge no assolit.

### Sense mmap

| Cas | Model | Context | Total predit | Error de memòria | A: RAM lliure | A: marge de Jaull | A: mínim lliure | A: ha encertat? | B: RAM lliure | B: marge de Jaull | B: mínim lliure | B: ha encertat? |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| C01 | Qwen2.5-7B Q4_K_M | 2.048 | 6.049 | −816 | 10.036 | +3.987, cap | 5.408 | Encert | 8.008 | +1.959, cap | 3.405 | Encert |
| C02 | Qwen2.5-7B Q4_K_M | 8.192 | 6.385 | −810 | 10.055 | +3.670, cap | 5.068 | Encert | 8.034 | +1.649, cap | 3.005 | Encert |
| C03 | Qwen2.5-7B Q4_K_M | 16.384 | 6.833 | −802 | 10.046 | +3.213, cap | 4.623 | Encert | 8.046 | +1.213, cap | 2.730 | Encert |
| C04 | Qwen2.5-7B Q4_K_M | 32.768 | 7.729 | −786 | 10.438 | +2.709, cap | 4.150 | Encert | 7.819 | +90, cap | 1.561 | Encert |
| C05 | Qwen2.5-7B Q3_K_M | 2.048 | 5.131 | −733 | 10.015 | +4.884, cap | 6.159 | Encert | 7.840 | +2.709, cap | 4.029 | Encert |
| C06 | Qwen2.5-7B Q3_K_M | 8.192 | 5.467 | −727 | 9.991 | +4.524, cap | 5.824 | Encert | 7.929 | +2.462, cap | 3.754 | Encert |
| C07 | Qwen2.5-7B Q3_K_M | 16.384 | 5.915 | −719 | 10.030 | +4.115, cap | 5.431 | Encert | 7.923 | +2.008, cap | 3.312 | Encert |
| C08 | Qwen2.5-7B Q3_K_M | 32.768 | 6.811 | −703 | 10.448 | +3.637, cap | 4.949 | Encert | 8.704 | +1.893, cap | 2.266 | Encert |
| C09 | Qwen2.5-7B IQ2_M | 8.192 | 4.389 | −629 | 10.007 | +5.618, cap | 6.846 | Encert | 7.272 | +2.883, cap | 4.232 | Encert |
| C10 | Qwen2.5-7B IQ2_M | 32.768 | 5.733 | −605 | 10.000 | +4.267, cap | 5.471 | Encert | 7.903 | +2.170, cap | 3.399 | Encert |
| C11 | Qwen2.5-Coder-7B Q3_K_M | 16.384 | 5.915 | −719 | 10.037 | +4.122, cap | 5.403 | Encert | 7.917 | +2.002, cap | 3.392 | Encert |
| C12 | Qwen2.5-3B Q4_K_M | 32.768 | 4.384 | −602 | 9.981 | +5.597, cap | 6.758 | Encert | 7.825 | +3.441, cap | 4.544 | Encert |
| C13 | Qwen2.5-1.5B Q5_K_M | 32.768 | 3.268 | −541 | 9.968 | +6.700, cap | 7.789 | Encert | 7.809 | +4.540, cap | 5.290 | Encert |
| C14 | Qwen2.5-0.5B Q4_K_M | 32.768 | 1.825 | −486 | 9.975 | +8.150, cap | 9.132 | Encert | 7.710 | +5.885, cap | 6.817 | Encert |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | 9.547 | −1.146 | 11.423 | +1.876, cap | 3.536 | Encert | 7.945 | −1.602, no | **parat** (330) | Encert |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | 9.547 | −1.056 | 11.490 | +1.943, cap | 3.522 | Encert | 6.932 | −2.615, no | **parat** (86) | Encert |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | 9.547 | −696 | 11.519 | +1.972, cap | 3.266 | Encert | 8.240 | −1.307, no | **parat** (110) | Encert |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | 8.454 | −957 | 10.526 | +2.072, cap | 3.656 | Encert | 7.918 | −536, no | 1.027 | **Fals negatiu** |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | 8.454 | −597 | 11.235 | +2.781, cap | 3.942 | Encert | 8.428 | −26, no | 1.219 | **Fals negatiu** |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | 9.107 | −1.085 | 11.419 | +2.312, cap | 3.953 | Encert | 9.256 | +149, cap | 1.210 | Encert |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | 10.259 | −1.079 | 11.459 | +1.200, cap | 2.835 | Encert | 6.111 | −4.148, no | **parat** (46) | Encert |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | 11.795 | −1.071 | 11.474 | −321, no | 1.387 | **Fals negatiu** | 7.361 | −4.434, no | **parat** (370) | Encert |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | 9.134 | −1.320 | 11.387 | +2.253, cap | 4.073 | Encert | 8.845 | −289, no | 1.328 | **Fals negatiu** |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | 10.030 | −1.292 | 11.531 | +1.501, cap | 3.257 | Encert | 7.329 | −2.701, no | **parat** (4) | Encert |
| **Encerts** | | | | | | | | **23 de 24** | | | | **21 de 24** |

### Amb mmap

Aquí es veu el que explica la secció sobre mmap: a la passada A, Jaull aprova 9 de les 10 configuracions i la RAM lliure acaba per sota del marge en 6 de les 7 amb `CPU_REPACK`.
La passada B hi queda molt bé (9 de 10) perquè quasi tot són rebutjos, i els rebutjos amb poca RAM es confirmen sols: la lectura honesta de mmap és la passada A.

| Cas | Model | Context | Total predit | Error de memòria | A: RAM lliure | A: marge de Jaull | A: mínim lliure | A: ha encertat? | B: RAM lliure | B: marge de Jaull | B: mínim lliure | B: ha encertat? |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | 9.547 | −1.146 | 11.469 | +1.922, cap | 460 | **Fals positiu** | 7.227 | −2.320, no | **parat** (345) | Encert |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | 9.547 | −1.056 | 11.462 | +1.915, cap | **parat** (355) | **Fals positiu** | 7.713 | −1.834, no | **parat** (133) | Encert |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | 9.547 | −696 | 11.519 | +1.972, cap | 451 | **Fals positiu** | 8.273 | −1.274, no | **parat** (20) | Encert |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | 8.454 | — | 10.627 | +2.173, cap | **parat** (58) | **Fals positiu** | 8.007 | −447, no | **parat** (10) | Encert |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | 8.454 | — | 11.229 | +2.775, cap | **parat** (358) | **Fals positiu** | 8.557 | +103, cap | **parat** (39) | **Fals positiu** |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | 9.107 | −1.084 | 11.396 | +2.289, cap | 1.735 | Encert | 8.971 | −136, no | **parat** (275) | Encert |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | 10.259 | −1.078 | 11.453 | +1.194, cap | 655 | **Fals positiu** | 7.845 | −2.414, no | **parat** (326) | Encert |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | 11.795 | −1.070 | 11.493 | −302, no | 466 | Encert | 7.576 | −4.219, no | **parat** (47) | Encert |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | 9.134 | −1.320 | 11.435 | +2.301, cap | 4.844 | Encert | 7.533 | −1.601, no | 1.015 | Encert |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | 10.030 | −1.292 | 11.470 | +1.440, cap | 3.983 | Encert | 9.644 | −386, no | **parat** (319) | Encert |
| **Encerts** | | | | | | | | **4 de 10** | | | | **9 de 10** |

## Com s'ha mesurat

Cada cas carrega el model amb `llama-server`, genera 128 tokens i es guarden els buffers que llama.cpp imprimeix (model, KV i còmput) i la RAM lliure de Windows cada segon.
La «RAM lliure» és la memòria disponible que dona Windows, que inclou la memòria cau que es pot recuperar.
Es mesuren dos modes de càrrega: `-lm none` (el model es carrega a RAM, sense mmap) i `-lm mmap`.
Jaull diu «cap» si el seu total (pesos, KV, overhead i la reserva de 512 MiB) és com a molt la RAM lliure mesurada abans d'engegar el servidor.
La heurística diu «cap» si la mida del GGUF × 1,2 és com a molt aquesta mateixa RAM.

**Què vol dir «funciona».** El criteri de marge, fixat al preregistre, demana que la RAM lliure mínima durant l'execució sigui d'almenys 1.024 MiB
i que el fitxer de paginació no creixi més de 512 MiB.

**El límit de seguretat el vam posar nosaltres.** Per no deixar el portàtil sense memòria, l'script atura el servidor si la RAM lliure baixa de 400 MiB o si el fitxer de paginació creix més de 3.000 MiB.
Com que 400 MiB queda per sota del marge d'1.024 MiB, un cas parat té el criteri de marge incomplert per construcció.
Aquí comptem els casos parats com a **marge no assolit verificat**. Això no vol dir que el model no hauria pogut generar: el vam aturar abans de saber-ho.
El preregistre els deixava com a «no verificats»; les xifres amb aquesta lectura són a «Jaull contra una regla simple».
La mateixa regla val en els dos sentits: una aprovació de Jaull que es para compta com a fals positiu.

## Què és mmap i per què canvia els resultats

**Sense mmap (`-lm none`).** llama.cpp reserva un bloc de RAM, llegeix el fitxer i hi copia els pesos. Després el fitxer ja no fa falta i el consum és fàcil de veure: és el que imprimeix el registre.
Carregar costa més, però la memòria que s'ocupa és la dels buffers.

**Amb mmap (`-lm mmap`, el mode per defecte de llama.cpp).** El sistema operatiu projecta el fitxer a l'espai d'adreces del procés i en va carregant pàgines a mesura que es llegeixen.
El model comença a funcionar abans, es pot compartir entre processos i, en teoria, pot funcionar amb un model més gran que la RAM si el sistema va descartant pàgines (més lent, perquè les ha de tornar a llegir).
Les pàgines projectades són memòria cau del fitxer, i Windows no les compta igual que una reserva privada.

**Per què importa en aquesta campanya.** A CPU, llama.cpp reordena alguns tensors quantitzats (Q4_K_M, Q3_K_M i MXFP4 al registre) en un format més ràpid per als nuclis, i això es veu al registre com un buffer `CPU_REPACK`.
Sense mmap, els buffers `CPU` i `CPU_REPACK` sumen exactament els pesos (al 14B, 4.481 més 2.513 MiB són els 6.994 del fitxer): l'original no es conserva.
Amb mmap, la hipòtesi és que el fitxer projectat, que és de només lectura, no es pot reordenar al mateix lloc: la còpia reordenada es crea a part i les pàgines originals continuen a la memòria.
Els buffers que llama.cpp declara són **els mateixos** en els dos modes, però la RAM lliure de Windows no ho és:

| Cas | Quantització | Buffer REPACK | RAM lliure mínima sense mmap | Amb mmap | Amb mmap gasta de més |
|---|---|---:|---:|---:|---:|
| C15 | Q4_K_M | 4.279 | 3.536 | 460 | +3.076 |
| C16 | Q4_K_M | 4.279 | 3.522 | 355 | +3.167 |
| C17 | Q4_K_M | 4.279 | 3.266 | 451 | +2.815 |
| C20 | Q3_K_M | 2.513 | 3.953 | 1.735 | +2.218 |
| C21 | Q3_K_M | 2.513 | 2.835 | 655 | +2.180 |
| C23 | Q6_K | no | 4.073 | 4.844 | −771 |
| C24 | Q6_K | no | 3.257 | 3.983 | −726 |

*(Passada A, MiB. Un signe negatiu vol dir que amb mmap en gasta menys.)*

Als models que tenen buffer `CPU_REPACK` (Q4_K_M i Q3_K_M), amb mmap se n'ocupen entre 2,2 i 3,2 GiB més, de l'ordre de la mida d'aquest buffer.
A l'Ornith (Q6_K), que no té `CPU_REPACK`, passa el contrari: amb mmap en gasta uns 0,7 GiB menys.
Les dades encaixen amb aquesta explicació, però **no s'ha comprovat al codi de llama.cpp ni amb cap prova directa** (per exemple, mesurant la memòria privada i la projectada per separat): és una hipòtesi.

**La prova que ho sustenta.** Cada segon es va guardar la memòria privada i el working set del servidor (passada A, MiB):

| Cas | Quantització | Mode | Privada (màx.) | Working set (màx.) | Caiguda de RAM lliure | Buffers de llama.cpp |
|---|---|---|---:|---:|---:|---:|
| C20 | Q3_K_M | sense mmap | 7.685 | 7.457 | 7.466 | 7.510 |
| C20 | Q3_K_M | amb mmap | 3.209 | 9.652 | 9.661 | 7.511 |
| C15 | Q4_K_M | sense mmap | 8.005 | 7.847 | 7.887 | 7.889 |
| C15 | Q4_K_M | amb mmap | 4.599 | 11.017 | 11.009 | 7.889 |
| C23 | Q6_K | sense mmap | 7.899 | 7.370 | 7.314 | 7.302 |
| C23 | Q6_K | amb mmap | 891 | 6.575 | 6.591 | 7.302 |

Amb mmap, la caiguda de RAM lliure és igual al working set, i el working set és la privada més quasi tot el fitxer projectat.
Al 14B, 3.209 de privada (els 2.513 de `CPU_REPACK` més KV i còmput) i uns 6.440 de pàgines projectades, sobre un fitxer de 6.994.
Sense `CPU_REPACK` (Ornith) la privada amb mmap és de només 891, i el total és menor perquè no es carreguen les pàgines que no es toquen.
Al Mellum, que és un MoE, en falten uns 1.157; probablement són experts que no s'han activat (no s'ha comprovat).

**Una salvedad.** Les pàgines projectades són memòria de fitxer neta, i Windows les pot recuperar si fa falta lloc.
Amb mmap el model potser funcionaria amb menys RAM, però més lent. El que mesura la campanya és que la RAM lliure baixa, no que el servidor es trenqui.

**Què costa deixar mmap.** Velocitat i temps de càrrega, passada A (tok/s i segons):

| Cas | Generació mmap | Generació sense | 1a petició mmap | 1a petició sense | Càrrega mmap (s) | Càrrega sense (s) |
|---|---:|---:|---:|---:|---:|---:|
| C15 | 17,7 | 20,4 | 6,9 | 40,5 | 8 | 11 |
| C17 | 18,0 | 20,6 | 5,9 | 42,1 | 9 | 10 |
| C20 | 4,9 | 4,9 | 12,4 | 14,5 | 8 | 11 |
| C21 | 4,8 | 4,8 | 13,3 | 13,9 | 8 | 12 |
| C22 | 4,9 | 4,9 | 13,4 | 13,8 | 11 | 11 |
| C23 | 4,7 | 4,8 | 9,6 | 14,5 | 6 | 11 |
| C24 | 4,8 | 4,8 | 9,5 | 14,2 | 7 | 11 |

La generació (128 tokens) és igual o millor sense mmap en tots aquests casos (la campanya 2, `resultados-cpu-2.md`, ho matisa: de −22 % a +12 % segons el model).
**Les xifres de prompt no són una velocitat de prompt:** el prompt és de només 16 tokens i es mesura en la primera petició. Amb mmap, el servidor diu «a punt» abans de tenir el model llegit i aquesta petició paga la càrrega de les pàgines (al Mellum, 2.323 ms contra 395 ms); sense mmap la càrrega ja s'ha fet abans. Que el Mellum, un MoE, ho noti més és plausible (cada token toca experts diferents), però no s'ha comprovat.
Carregar triga uns 3 s més sense mmap i, a canvi, la primera petició és més ràpida: el temps fins a la primera resposta queda gairebé igual. El fitxer ja era a la memòria cau del disc, així que amb un fitxer fred la diferència pot ser una altra.
Per això Jaull, a Windows i amb backend CPU, carrega per defecte amb `-lm none` (D129). Un model més gran que la RAM disponible no s'ha provat.

**Què en resulta per a Jaull.** L'estimació de Jaull segueix els buffers de llama.cpp (més un marge), i amb mmap una part del consum real no surt als buffers.
Per això els falsos positius del marge són tots amb mmap, i per això els dos modes es presenten per separat.
llama.cpp fa servir mmap per defecte, de manera que un usuari que no toqui res es troba amb el comportament d'aquest mode.

## Memòria: prediccions i observacions

Error = observat menys predit, en MiB, a la passada A i sense mmap (els buffers són els mateixos a les dues passades i amb els dos modes).
Per al KV s'hi suma l'estat recurrent, que Jaull també comptava dins del KV als models híbrids.

| Cas | Model | Context | Pesos | KV | Còmput | Total |
|---|---|---:|---:|---:|---:|---:|
| C01 | Qwen2.5-7B Q4_K_M | 2.048 | −5,6 | 0,0 | −810,6 | −816 |
| C02 | Qwen2.5-7B Q4_K_M | 8.192 | −5,6 | 0,0 | −804,6 | −810 |
| C03 | Qwen2.5-7B Q4_K_M | 16.384 | −5,6 | 0,0 | −796,6 | −802 |
| C04 | Qwen2.5-7B Q4_K_M | 32.768 | −5,6 | 0,0 | −780,6 | −786 |
| C05 | Qwen2.5-7B Q3_K_M | 2.048 | −5,7 | 0,0 | −727,2 | −733 |
| C06 | Qwen2.5-7B Q3_K_M | 8.192 | −5,7 | 0,0 | −721,2 | −727 |
| C07 | Qwen2.5-7B Q3_K_M | 16.384 | −5,7 | 0,0 | −713,2 | −719 |
| C08 | Qwen2.5-7B Q3_K_M | 32.768 | −5,7 | 0,0 | −697,2 | −703 |
| C09 | Qwen2.5-7B IQ2_M | 8.192 | −5,6 | 0,0 | −623,2 | −629 |
| C10 | Qwen2.5-7B IQ2_M | 32.768 | −5,6 | 0,0 | −599,2 | −605 |
| C11 | Qwen2.5-Coder-7B Q3_K_M | 16.384 | −5,7 | 0,0 | −713,2 | −719 |
| C12 | Qwen2.5-3B Q4_K_M | 32.768 | −5,7 | 0,0 | −596,2 | −602 |
| C13 | Qwen2.5-1.5B Q5_K_M | 32.768 | −5,6 | 0,0 | −535,1 | −541 |
| C14 | Qwen2.5-0.5B Q4_K_M | 32.768 | −5,7 | 0,0 | −480,6 | −486 |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | −3,4 | +35,0 | −1.177,7 | −1.146 |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | −3,4 | +119,0 | −1.171,7 | −1.056 |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | −3,4 | +455,0 | −1.147,7 | −696 |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4_MOE | 8.192 | −3,4 | +119,0 | −1.072,4 | −957 |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4_MOE | 32.768 | −3,4 | +455,0 | −1.048,4 | −597 |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | −5,7 | 0,0 | −1.078,9 | −1.085 |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | −5,7 | 0,0 | −1.072,9 | −1.079 |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | −5,7 | 0,0 | −1.064,9 | −1.071 |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | −200,8 | −2,0 | −1.116,9 | −1.320 |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | −200,8 | −2,0 | −1.088,9 | −1.292 |

| Component | Error mínim | Error màxim | Lectura |
|---|---:|---:|---|
| Pesos | −200,8 | −3,4 | El buffer del model és la mida del fitxer menys la capçalera, i no hi ha embeddings al host. L'Ornith en té −200,8 MiB menys. |
| KV | −2,0 | +455,0 | Exacte als Qwen i al 14B, i a 2 MiB a l'Ornith híbrid. Només falla el Mellum. |
| Còmput | −1.177,7 | −480,6 | Jaull compta 512 MiB més el 10 % dels pesos, i llama.cpp reserva entre 70 i 200 MiB. |
| **Total** | **−1.320** | **−486** | Sempre negatiu: Jaull és conservador. |

### Les hipòtesis escrites abans de mesurar

| # | Hipòtesi | Resultat |
|---|---|---|
| 1 | El KV del Mellum és constant (56 MiB) perquè Jaull el limita a la finestra lliscant | **Refutada.** Observat: 91, 175 i 511 MiB a 2K, 8K i 32K. Creix amb el context, uns 14 KiB per token, perquè només algunes capes tenen finestra. |
| 2 | El buffer del model s'acosta a la mida del fitxer | **Confirmada** (−200,8 a −3,4 MiB), llevat de l'Ornith (−200,8 MiB). |
| 3 | L'error del total és negatiu a tots els casos | **Confirmada** (−1.320 a −486 MiB). |
| 4 | Jaull no reconeix l'etiqueta MXFP4_MOE | **Confirmada**, la tracta com a `unknown` amb confiança baixa. Els pesos coincideixen igualment (−3,4 MiB). |
| 5 | L'Ornith híbrid té el KV exacte i un bloc de predicció múltiple sense carregar | **KV exacte** (el KV més l'estat recurrent coincideixen). El buffer del model és 200,8 MiB menor que el fitxer, compatible amb un bloc que llama.cpp no carrega. No s'ha comprovat al seu codi. |
| 6 | El 14B és un transformer estàndard i només falla l'overhead | **Confirmada.** |

## Passada A: molta RAM (de 9.968 a 11.531 MiB lliures)

Sense mmap, Jaull aprova 23 de 24 configuracions i totes funcionen, amb un mínim de RAM lliure d'almenys 2.835 MiB.
Rebutja una, la C22 (el 14B a 16.384 tokens: 11.795 MiB contra 11.474 lliures), i va funcionar amb 1.387 MiB lliures: un fals negatiu prudent.

Amb mmap, la RAM lliure cau molt més pels mateixos buffers:

| Cas | Model | Context | Mode | RAM lliure | Total predit | Jaull | Mínim lliure | Jaull: resultat |
|---|---|---:|---|---:|---:|---|---|---|
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | mmap | 10.627 | 8.454 | cap | **parat** (58 MiB) | Fals positiu |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | mmap | 11.229 | 8.454 | cap | **parat** (358 MiB) | Fals positiu |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | mmap | 11.396 | 9.107 | cap | 1.735 MiB, 4,9 tok/s | Encert |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | mmap | 11.435 | 9.134 | cap | 4.844 MiB, 4,7 tok/s | Encert |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | mmap | 11.469 | 9.547 | cap | 460 MiB, 17,7 tok/s | Fals positiu |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | mmap | 11.462 | 9.547 | cap | **parat** (355 MiB) | Fals positiu |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | mmap | 11.519 | 9.547 | cap | 451 MiB, 18,0 tok/s | Fals positiu |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | mmap | 11.470 | 10.030 | cap | 3.983 MiB, 4,8 tok/s | Encert |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | mmap | 11.453 | 10.259 | cap | 655 MiB, 4,8 tok/s | Fals positiu |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | mmap | 11.493 | 11.795 | no | 466 MiB, 4,9 tok/s | Encert |

## Passada B: RAM ajustada (de 6.111 a 9.644 MiB lliures)

| Cas | Model | Context | Mode | RAM lliure | Total predit | Jaull | Mínim lliure | Jaull: resultat |
|---|---|---:|---|---:|---:|---|---|---|
| C14 | Qwen2.5-0.5B Q4_K_M | 32.768 | sense mmap | 7.710 | 1.825 | cap | 6.817 MiB, 63,2 tok/s | Encert |
| C13 | Qwen2.5-1.5B Q5_K_M | 32.768 | sense mmap | 7.809 | 3.268 | cap | 5.290 MiB, 22,1 tok/s | Encert |
| C12 | Qwen2.5-3B Q4_K_M | 32.768 | sense mmap | 7.825 | 4.384 | cap | 4.544 MiB, 15,1 tok/s | Encert |
| C09 | Qwen2.5-7B IQ2_M | 8.192 | sense mmap | 7.272 | 4.389 | cap | 4.232 MiB, 7,3 tok/s | Encert |
| C05 | Qwen2.5-7B Q3_K_M | 2.048 | sense mmap | 7.840 | 5.131 | cap | 4.029 MiB, 9,1 tok/s | Encert |
| C06 | Qwen2.5-7B Q3_K_M | 8.192 | sense mmap | 7.929 | 5.467 | cap | 3.754 MiB, 9,1 tok/s | Encert |
| C10 | Qwen2.5-7B IQ2_M | 32.768 | sense mmap | 7.903 | 5.733 | cap | 3.399 MiB, 7,5 tok/s | Encert |
| C07 | Qwen2.5-7B Q3_K_M | 16.384 | sense mmap | 7.923 | 5.915 | cap | 3.312 MiB, 8,6 tok/s | Encert |
| C11 | Qwen2.5-Coder-7B Q3_K_M | 16.384 | sense mmap | 7.917 | 5.915 | cap | 3.392 MiB, 8,8 tok/s | Encert |
| C01 | Qwen2.5-7B Q4_K_M | 2.048 | sense mmap | 8.008 | 6.049 | cap | 3.405 MiB, 7,3 tok/s | Encert |
| C02 | Qwen2.5-7B Q4_K_M | 8.192 | sense mmap | 8.034 | 6.385 | cap | 3.005 MiB, 7,3 tok/s | Encert |
| C03 | Qwen2.5-7B Q4_K_M | 16.384 | sense mmap | 8.046 | 6.833 | cap | 2.730 MiB, 7,2 tok/s | Encert |
| C08 | Qwen2.5-7B Q3_K_M | 32.768 | sense mmap | 8.704 | 6.811 | cap | 2.266 MiB, 7,2 tok/s | Encert |
| C04 | Qwen2.5-7B Q4_K_M | 32.768 | sense mmap | 7.819 | 7.729 | cap | 1.561 MiB, 7,6 tok/s | Encert |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | sense mmap | 7.918 | 8.454 | no | 1.027 MiB, 17,3 tok/s | Fals negatiu |
| C18 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 8.192 | mmap | 8.007 | 8.454 | no | **parat** (10 MiB) | Encert |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | sense mmap | 8.428 | 8.454 | no | 1.219 MiB, 18,1 tok/s | Fals negatiu |
| C19 | Mellum2.1-12B-A2.5B-Thinking MXFP4 | 32.768 | mmap | 8.557 | 8.454 | cap | **parat** (39 MiB) | Fals positiu |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | sense mmap | 9.256 | 9.107 | cap | 1.210 MiB, 3,7 tok/s | Encert |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 2.048 | mmap | 8.971 | 9.107 | no | **parat** (275 MiB) | Encert |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | sense mmap | 8.845 | 9.134 | no | 1.328 MiB, 4,1 tok/s | Fals negatiu |
| C23 | Ornith-1.5-9B Q6_K | 4.096 | mmap | 7.533 | 9.134 | no | 1.015 MiB, 3,9 tok/s | Encert |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | sense mmap | 7.945 | 9.547 | no | **parat** (330 MiB) | Encert |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 2.048 | mmap | 7.227 | 9.547 | no | **parat** (345 MiB) | Encert |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | sense mmap | 6.932 | 9.547 | no | **parat** (86 MiB) | Encert |
| C16 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 8.192 | mmap | 7.713 | 9.547 | no | **parat** (133 MiB) | Encert |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | sense mmap | 8.240 | 9.547 | no | **parat** (110 MiB) | Encert |
| C17 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 32.768 | mmap | 8.273 | 9.547 | no | **parat** (20 MiB) | Encert |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | sense mmap | 7.329 | 10.030 | no | **parat** (4 MiB) | Encert |
| C24 | Ornith-1.5-9B Q6_K | 32.768 | mmap | 9.644 | 10.030 | no | **parat** (319 MiB) | Encert |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | sense mmap | 6.111 | 10.259 | no | **parat** (46 MiB) | Encert |
| C21 | Qwen2.5-Coder-14B Q3_K_M | 8.192 | mmap | 7.845 | 10.259 | no | **parat** (326 MiB) | Encert |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | sense mmap | 7.361 | 11.795 | no | **parat** (370 MiB) | Encert |
| C22 | Qwen2.5-Coder-14B Q3_K_M | 16.384 | mmap | 7.576 | 11.795 | no | **parat** (47 MiB) | Encert |

**Les aprovacions més justes.** C04 (90 MiB de marge i 1.561 lliures), C20 (149 MiB de marge i 1.210 lliures). Jaull deia que hi havia només un marge d'uns pocs centenars de MiB, i va quedar més d'un GiB lliure.

**Els rebutjos.** A C15, C16, C17, C21, C22, C24 (sense mmap) la RAM lliure va baixar de 400 MiB i l'script va aturar el servidor.
Jaull havia dit que no en tots. Compten com a rebutjos verificats pel criteri de marge.

## Jaull contra una regla simple

El preregistre fixava una línia base: dir «cap» si la mida del GGUF × 1,2 és com a molt la RAM lliure. Sense ella, «zero falsos positius» no vol dir res, perquè qualsevol regla prudent ho aconsegueix.
Criteri de marge, comptant els casos parats com a marge no assolit. Les dues passades es presenten juntes, però les configuracions es repeteixen: no són mesures independents.

| Mode | Mètode | Aprova i funciona | Fals positiu | Fals negatiu | Rebutja i no cap |
|---|---|---:|---:|---:|---:|
| sense mmap | Jaull | 38 | 0 | 4 | 6 |
| sense mmap | GGUF × 1,2 | 41 | 0 | 1 | 6 |
| amb mmap | Jaull | 3 | 7 | 0 | 10 |
| amb mmap | GGUF × 1,2 | 3 | 10 | 0 | 7 |

Sense mmap la regla simple falla menys que Jaull (1 fals negatiu contra 4) i tampoc dona cap fals positiu. El valor de Jaull no és aquí encertar més: és explicar quin component falla (pesos, KV, overhead).
Amb la lectura estricta del preregistre, on els casos parats són «no verificats», els falsos positius sense mmap segueixen sent 0 per a Jaull i 0 per a la regla; amb mmap passen a 3 i 4.

## Velocitat

`llama-bench`, 6 fils, `-p 512 -n 128`, tres repeticions (mitjana ± desviació, tokens per segon). La darrera columna és la mida del fitxer per la velocitat de generació.
Als models densos a partir de 3B queda entre 36 i 38 GB/s, que és el 70 a 75 % del màxim teòric de la memòria (51,2 GB/s): la velocitat depèn de l'amplada de banda i no del càlcul.
El 0,5B queda per sota perquè els costos fixos hi pesen, i el Mellum no té valor perquè només n'activa una part a cada token.
Jaull no prediu la velocitat: això és només una mesura.

| Cas | Model | Prompt (pp512) | Generació (tg128) | GB/s |
|---|---|---:|---:|---:|
| C14 | Qwen2.5-0.5B Q4_K_M | 346,5 ± 3,6 | 73,1 ± 1,7 | 29 |
| C15 | Mellum2.1-12B-A2.5B-Thinking Q4_K_M | 64,4 ± 2,5 | 20,5 ± 0,6 |  |
| C12 | Qwen2.5-3B Q4_K_M | 90,9 ± 2,0 | 17,8 ± 0,0 | 37 |
| C05 | Qwen2.5-7B Q3_K_M | 39,6 ± 0,4 | 9,5 ± 0,3 | 36 |
| C01 | Qwen2.5-7B Q4_K_M | 40,6 ± 0,1 | 8,2 ± 0,0 | 38 |
| C20 | Qwen2.5-Coder-14B Q3_K_M | 18,4 ± 0,5 | 4,9 ± 0,1 | 36 |
| C23 | Ornith-1.5-9B Q6_K | 32,5 ± 0,2 | 4,8 ± 0,0 | 37 |

El Mellum, que és un MoE amb 2,5B de paràmetres actius, genera a uns 20 tok/s i els densos de 9B i 14B ho fan a uns 5.
Si el que es vol és un ús interactiu a CPU, el MoE és l'opció que va de pressa.

## Limitacions

- Una sola màquina, una sola execució per cas i dues passades amb les mateixes configuracions: no són mesures independents.
- La RAM lliure va canviar molt entre casos (de 6.111 a 11.531 MiB) perquè es mesura amb les aplicacions que hi havia obertes. Cada veredicte es calcula amb la RAM del seu cas, però els casos no tenen la mateixa dificultat.
- Els casos parats es van aturar pel límit de seguretat nostre: se sap que no van arribar al marge, no si haurien pogut generar.
- La causa de la diferència entre mmap i sense mmap és una hipòtesi sustentada en les mesures (`CPU_REPACK`), no una lectura del codi de llama.cpp. Només s'ha provat amb tres quantitzacions (Q4_K_M, Q3_K_M i Q6_K) i amb els models Q4_K_M i Q3_K_M de 7B només sense mmap.
- L'Ornith i el Mellum són models nous i els dos tenen una arquitectura que Jaull només coneix en part (finestra lliscant i bloc de predicció múltiple).
- La fórmula de Jaull no s'ha modificat durant la campanya. L'overhead de CPU i el KV de finestra lliscant queden anotats com a treball futur.
