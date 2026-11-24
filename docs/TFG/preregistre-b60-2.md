# Preregistre 2: contextos grans a la B60

**Esborrany del 2026-11-24. Escrit abans de mesurar res d'aquest bloc.** Cal fer-ne commit abans d'arrencar cap prova.
Després de la primera mesura no es modifica; les correccions van en una secció nova al final, amb data. Complementa
`preregistre-b60.md` i `resultados-b60.md`: aquells sis casos eren a prop de la frontera de Jaull, i aquests cobreixen
contextos que ja no caben i el límit d'un model híbrid.

Codi de Jaull: el mateix de `preregistre-b60.md` (`314f56e`). No es toca cap fórmula.

## Pregunta

Fins on arriba Jaull en context, i en què s'equivoca? La comparació és amb **llama.cpp**, en MiB i per component. El sí o no
(la regla de llama.cpp: queden 1.024 MiB lliures o més) és només una lectura derivada.

## Maquinari i comanda

Els mateixos que a `resultados-b60.md`: B60 amb 23.613 MiB lliures (es torna a llegir; si canvia en més de 200 MiB es recalcula
i s'anota), Windows natiu, llama.cpp b11889 i
`-dev Vulkan1 -ngl 99 -c <context> -np 1 --jinja -lv 4 --host 127.0.0.1 --port 8099 -lm dio`.
Models de `C:\llama-win`, amb el sha256 ja comprovat.

## Protocol de seguretat: només ajust si no cap

llama.cpp imprimeix `projected to use X MiB of device memory vs. Y MiB of free device memory` uns 0,9 s després d'arrencar, abans de
reservar memòria. Els pesos es carreguen als 2,6 s i el KV (l'allocació gran) cap als 18 s. L'script llegeix aquesta línia:

- **Si el projectat és igual o superior a la lliure** (no hi cap a la VRAM), para el servidor i no carrega el KV. El resultat és el
  projectat i el que queda lliure, que és tot el que cal per al nivell 1. No es prova si carregaria ni si generaria: aquests casos
  poden fer desbordar la VRAM a memòria compartida, i un cop va penjar el PC de Biosfer.
- **Si hi cap**, encara que no compleixi la regla d'1 GiB (com P3), continua i genera 512 tokens com als sis casos anteriors.
- Es manté la guarda de RAM de Windows: avorta per sota de 3.072 MiB disponibles.

## Prediccions

Fórmula de Jaull actual, **CALCULAT** a mà: pesos (mida del fitxer) + KV + 256 MiB + 1.024 MiB. Les constants són les de
`preregistre-b60.md`: Coder 30B, pesos 17.695 i KV 0,09375 MiB per token; Qwen2.5 32B, 18.934 i 0,25. Qwen3.8-27B, híbrid
(D125): pesos 15.702, KV 0,0625 MiB per token (16 capes d'atenció de 65) i 153 MiB d'estat recurrent. Lliure: 23.613 MiB.

La columna de llama.cpp és una **extrapolació** de les mesures anteriors, no una predicció de Jaull: el còmput creix uns
0,001 MiB per token (de 116 MiB a 32K a 136 a 53K al Coder). Per al 27B no hi ha una mesura prèvia del buffer de pesos i s'estima en
15.150 MiB a partir de B3, amb una incertesa d'uns ±200 MiB.

| Cas | Model | Context | Jaull necessita (marge) | Jaull diu | llama.cpp derivat: projectat | Queden lliures |
|---|---|---:|---:|---|---:|---:|
| Q1 | Coder 30B | 8.192 | 19.743 (+3.870) | Cap | 18.387 | 5.226 |
| Q2 | Coder 30B | 98.304 | 28.191 (−4.578) | No | 26.921 | −3.308 |
| Q3 | Coder 30B | 131.072 | 31.263 (−7.650) | No | 30.026 | −6.413 |
| Q4 | Qwen2.5 32B | 32.768 | 28.406 (−4.793) | No | 26.935 | −3.322 |
| Q5 | Qwen2.5 32B | 65.536 | 36.598 (−12.985) | No | 35.159 | −11.546 |
| Q6 | Qwen3.8-27B | 98.304 | 23.279 (+334) | Cap | 21.675 | 1.938 |
| Q7 | Qwen3.8-27B | 114.688 | 24.303 (−690) | No | 22.713 | 900 |
| Q8 | Qwen3.8-27B | 131.072 | 25.327 (−1.714) | No | 23.756 | −143 |

Notes:

- Q1 i Q6 es carreguen i generen. Q2 a Q5 no caben amb molt de marge, i només es mira el projectat. Q7 i Q8 són els de la frontera
  del 27B: si el projectat de Q7 és menor que la lliure, es carrega i es genera com P3; si no, s'atura a l'ajust.
- La incertesa de ±200 MiB fa que Q7 (900) i Q8 (−143) puguin canviar de costat: Q7 podria complir la regla i Q8 podria cabre.
- Qwen2.5 32B va ser entrenat amb 32.768 tokens de context (YaRN arriba a 131.072). Q4 i Q5 mesuren memòria, no qualitat, i llama.cpp pot avisar
  que el context supera el d'entrenament.

## Com es compta

- **Error de predicció**: pesos + KV + còmput de Jaull menys els de llama.cpp, en MiB i per cas. És la mesura principal. Als sis
  casos anteriors, el KV era exacte i la resta, +291 a +468 MiB.
- **Encert** del sí o no: la predicció de Jaull (cap o no cap) coincideix amb el nivell 1 de llama.cpp. Es compta en casos, sense percentatges.
- Una execució per cas. Si en queden menys de 100 MiB d'un dels llindars (1.024 lliures, o el projectat igual a la lliure), es repeteix una vegada.
- **Seguretat**: si l'script no pot parar un servidor, ho diu a pantalla i l'autor el para amb `taskkill /PID`.

## Què no es pot dir amb això

- Res sobre si els casos que no caben carregarien, generarien o penjarien el PC: no es provarà a propòsit.
- Res sobre velocitat ni estabilitat amb el context ple (el prompt de prova té 16 tokens).
- Res sobre altres targetes, altres runtimes ni més d'un dia.
