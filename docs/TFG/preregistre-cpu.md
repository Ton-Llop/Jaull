# Preregistre de la campanya CPU

Escrit el 2026-11-25, **abans de fer cap mesura**. Les prediccions són de Jaull (commit `400b1a4`, `src/` sense canvis) i no es modificaran.
Aquest fitxer es comiteja abans d'executar res.

## Pregunta

La campanya de la B60 va validar la memòria a una GPU de 24 GiB. Aquí es mira la ruta de CPU i RAM, que no s'ha calibrat mai:
per a un GGUF a CPU, Jaull aplica l'overhead general de 512 MiB més el 10 % dels pesos, no els 256 MiB fixos que es van mesurar a la B60.

- **P1.** Jaull encerta quines configuracions es poden executar a CPU, sobretot quan les aprova?
- **P2.** Quin error té l'estimació de memòria respecte del que reserva llama.cpp?
- **P3.** Una configuració que cap va prou de pressa per a un ús interactiu? (Mesura descriptiva: Jaull no prediu la velocitat.)

## Condicions

Portàtil amb AMD Ryzen 5 5500U (6 nuclis, 12 fils), 15,7 GiB de RAM (2 × 8 GB DDR4 a 3.200 MT/s), Windows natiu, sense WSL2 en marxa.
llama.cpp b11889 només CPU, `llama-server` amb `-np 1`, 6 fils (els nuclis físics) i `-lv 4`. Prompt fix i 128 tokens generats, temperatura 0.
Es parteix d'un estat net (`wsl --shutdown` i aplicacions pesades tancades) i es mesura la RAM disponible abans de cada cas.

## Prediccions de Jaull (`jaull estimate --device cpu`, MiB)

| Cas | Model | Quant. | Context | Pesos | KV | Overhead | Físic | Total | Modes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| C01 | Qwen2.5-7B-Instruct | Q4_K_M | 2.048 | 4.466 | 112 | 959 | 5.537 | 6.049 | nommap |
| C02 | Qwen2.5-7B-Instruct | Q4_K_M | 8.192 | 4.466 | 448 | 959 | 5.873 | 6.385 | nommap |
| C03 | Qwen2.5-7B-Instruct | Q4_K_M | 16.384 | 4.466 | 896 | 959 | 6.321 | 6.833 | nommap |
| C04 | Qwen2.5-7B-Instruct | Q4_K_M | 32.768 | 4.466 | 1.792 | 959 | 7.217 | 7.729 | nommap |
| C05 | Qwen2.5-7B-Instruct | Q3_K_M | 2.048 | 3.632 | 112 | 875 | 4.619 | 5.131 | nommap |
| C06 | Qwen2.5-7B-Instruct | Q3_K_M | 8.192 | 3.632 | 448 | 875 | 4.955 | 5.467 | nommap |
| C07 | Qwen2.5-7B-Instruct | Q3_K_M | 16.384 | 3.632 | 896 | 875 | 5.403 | 5.915 | nommap |
| C08 | Qwen2.5-7B-Instruct | Q3_K_M | 32.768 | 3.632 | 1.792 | 875 | 6.299 | 6.811 | nommap |
| C09 | Qwen2.5-7B-Instruct | IQ2_M | 8.192 | 2.652 | 448 | 777 | 3.877 | 4.389 | nommap |
| C10 | Qwen2.5-7B-Instruct | IQ2_M | 32.768 | 2.652 | 1.792 | 777 | 5.221 | 5.733 | nommap |
| C11 | Qwen2.5-Coder-7B-Instruct | Q3_K_M | 16.384 | 3.632 | 896 | 875 | 5.403 | 5.915 | nommap |
| C12 | Qwen2.5-3B-Instruct | Q4_K_M | 32.768 | 2.007 | 1.152 | 713 | 3.872 | 4.384 | nommap |
| C13 | Qwen2.5-1.5B-Instruct | Q5_K_M | 32.768 | 1.226 | 896 | 635 | 2.756 | 3.268 | nommap |
| C14 | Qwen2.5-0.5B-Instruct | Q4_K_M | 32.768 | 379 | 384 | 550 | 1.313 | 1.825 | nommap |
| C15 | Mellum2.1-12B-A2.5B-Thinking | Q4_K_M | 2.048 | 7.697 | 56 | 1.282 | 9.035 | 9.547 | nommap i mmap |
| C16 | Mellum2.1-12B-A2.5B-Thinking | Q4_K_M | 8.192 | 7.697 | 56 | 1.282 | 9.035 | 9.547 | nommap i mmap |
| C17 | Mellum2.1-12B-A2.5B-Thinking | Q4_K_M | 32.768 | 7.697 | 56 | 1.282 | 9.035 | 9.547 | nommap i mmap |
| C18 | Mellum2.1-12B-A2.5B-Thinking | MXFP4_MOE | 8.192 | 6.704 | 56 | 1.182 | 7.942 | 8.454 | nommap i mmap |
| C19 | Mellum2.1-12B-A2.5B-Thinking | MXFP4_MOE | 32.768 | 6.704 | 56 | 1.182 | 7.942 | 8.454 | nommap i mmap |
| C20 | Qwen2.5-Coder-14B-Instruct | Q3_K_M | 2.048 | 6.999 | 384 | 1.212 | 8.595 | 9.107 | nommap i mmap |
| C21 | Qwen2.5-Coder-14B-Instruct | Q3_K_M | 8.192 | 6.999 | 1.536 | 1.212 | 9.747 | 10.259 | nommap i mmap |
| C22 | Qwen2.5-Coder-14B-Instruct | Q3_K_M | 16.384 | 6.999 | 3.072 | 1.212 | 11.283 | 11.795 | nommap i mmap |
| C23 | Ornith-1.5-9B | Q6_K | 4.096 | 7.209 | 180 | 1.233 | 8.622 | 9.134 | nommap i mmap |
| C24 | Ornith-1.5-9B | Q6_K | 32.768 | 7.209 | 1.076 | 1.233 | 9.518 | 10.030 | nommap i mmap |

**Físic** és pesos + KV + overhead. **Total** hi suma la reserva de RAM de 512 MiB, que no és memòria assignada.
Les prediccions completes (JSON) són a `validation/cpu-20261125/predicciones/`.

## Regla de veredicte, fixada abans

- Jaull diu **«cap»** si el total és com a molt la RAM disponible just abans d’engegar el servidor (mesurada a Windows). Altrament diu **«no»**.
- El mètode simple diu «cap» si la mida del GGUF × 1,2 és com a molt aquesta mateixa RAM disponible.

## Què es considera que «cap»

Es mesuren dos criteris i es compten per separat, sense barrejar-los en una sola xifra:

1. **Executa:** el model carrega i genera els 128 tokens sense error ni parada.
2. **Marge:** a més, la RAM disponible mínima durant l’execució és d’almenys 1.024 MiB i el fitxer de paginació no creix més de 512 MiB.

Un cas que es para per seguretat (RAM disponible per sota de 400 MiB, o creixement del fitxer de paginació de més de 3.000 MiB) es registra com a **no verificat**, no com a fallit.
Cada cas es mesura amb `--no-mmap` (la càrrega és clara). Els casos C15 a C24 també es mesuren amb mmap, on la càrrega sempre «funciona» però pot anar a rastres.
Els Qwen2.5 de 7B o menys queden lluny del límit amb la RAM d’aquest portàtil i només es mesuren sense mmap.

## Observació comparable

Els buffers que imprimeix llama.cpp en carregar (model, KV i còmput), sumats. La RSS i el fitxer de paginació es registren a part:
amb mmap, la RSS inclou la memòria cau de pàgines i no mesura el consum del model.
L’error és **observat − predit**, en MiB (en valor absolut, perquè als models petits el percentatge enganya).

## Hipòtesis, escrites abans de mesurar

1. **KV del Mellum.** Jaull limita el KV a la finestra lliscant de 1.024 tokens i prediu 56 MiB a qualsevol context. Si llama.cpp reserva un KV que creix
   amb el context (perquè només algunes capes tenen finestra), la predicció és massa baixa i el total del Mellum quedaria subestimat: seria el cas on Jaull pot donar un fals positiu.
   Jaull avisa a més que no ha validat cap Mixture-of-Experts.
2. **Pesos.** A CPU no hi ha una part al host i una a la GPU: s’espera que el buffer del model sigui molt a prop de la mida del fitxer (a la B60 la diferència eren els embeddings al host).
3. **Overhead.** El de Jaull (512 MiB més el 10 % dels pesos) és més gran que el buffer de còmput de llama.cpp. S’espera un error negatiu a tots els casos, de l’ordre de 0,5 a 1 GiB.
4. **MXFP4_MOE.** Jaull no reconeix l’etiqueta i només estima la variant com a `unknown`, amb confiança baixa.
5. **Ornith 1.5 9B (arquitectura Qwen3.5 híbrida).** Jaull compta 33 capes, de les quals 8 tenen KV (180 MiB a 4.096 tokens i 1.076 MiB a 32.768) i la resta un estat recurrent de mida constant.
   La capa 33 podria ser un bloc de predicció múltiple que llama.cpp no carrega, com al Qwen3.8 27B de la B60. Si és així, el buffer del model quedarà per sota de la mida del fitxer.
6. **Qwen2.5 Coder 14B Q3_K_M.** És un transformer dens estàndard, sense cap particularitat d’arquitectura. S’espera que el KV coincideixi (192 KiB per token) i que l’error sigui només l’overhead.

## Com s’han triat els models

Els Qwen2.5 de 0,5B a 7B i el Mellum eren a l’abast (ja descarregats). El 14B Coder i l’Ornith 9B Q6_K surten de la llista de recomanacions de la TUI de Jaull per a aquest portàtil
(CPU, 9,45 GiB de RAM disponibles), on apareixien amb l’estat `Tight` (8,77 i 8,42 GiB, sense comptar la reserva). Es van triar perquè són aprovacions amb poc marge, que és on un fals positiu seria possible,
i no perquè Jaull els hagi qualificat bé. Els casos amb context alt (el 14B a 16.384 tokens, per exemple) es van afegir perquè Jaull els rebutgi.

## Velocitat

Set models (C14, C12, C05, C01, C15, C23 i C20) amb `llama-bench`, `-p 512 -n 128`, 6 fils i tres repeticions. Es dona la mediana. No és una validació de Jaull.

## Limitacions

Una màquina, models de tres famílies (Qwen2.5, Mellum i Ornith) i una sola execució per cas. La RAM disponible varia entre arrencades, i per això es mesura abans de cada cas.
No s’estableix cap precisió general de Jaull. Cap fórmula es modifica durant la campanya, i les configuracions que no s’executen per seguretat no compten com a errors.
