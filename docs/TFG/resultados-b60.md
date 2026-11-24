# Resultats de la B60

Mesures del 2026-11-24 contra les prediccions de `preregistre-b60.md`. Les va executar l'autor per SSH a `monk-e` amb
un script que no és al repositori. Les xifres són del log de llama.cpp b11889 amb `-lv 4`. **MESURAT** vol dir una lectura
de llama.cpp, **CALCULAT** una operació feta sobre aquestes lectures i **UNVERIFIED** el que no s'ha comprovat.

## Condicions

- Windows natiu, Intel Arc Pro B60, 23.613 MiB lliures abans de cada càrrega.
- Comanda: `llama-server.exe -m <model> -dev Vulkan1 -ngl 99 -c <context> -np 1 --jinja -lv 4 --host 127.0.0.1 --port 8099 -lm dio`.
- Prova: prompt de 16 tokens i 512 tokens generats, sense memòria cau de prompt.
- Respecte del preregistre s'hi van afegir `-dev Vulkan1`, `--jinja` i `-lv 4`, que són els de la mesura de D123. `Vulkan0`
  és la gràfica integrada AMD i `-lv 4` fa imprimir el desglossament de memòria. `-lm dio` evita omplir la RAM del PC i no afecta la VRAM.
- Els GGUF de `C:\llama-win` tenen el mateix sha256 que els que va verificar Jaull (comprovat el 2026-11-24).
- El preregistre també definia un mètode senzill de referència (mida del fitxer per 1,2). No s'usa aquí: no mira el context i
  els sis casos són a prop de la frontera de Jaull, on no els distingeix.

## Resultats

Tot en MiB. **Projectat** és pesos + KV + còmput, tal com ho imprimeix llama.cpp. **Jaull (marge)** és el que Jaull necessita
(inclou l'1 GiB lliure) i el seu marge sobre els 23.613 MiB. **Dif.** és pesos + KV + 256 de Jaull menys el projectat.
La regla de llama.cpp es compleix quan en queden lliures 1.024 o més.

| Cas | Context | Pesos | KV | Còmput | Projectat | Lliure | Jaull (marge) | Dif. | tok/s | Resultat |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| P1 Coder 30B | 49.152 | 17.524 | 4.608 | 132 | 22.264 | 1.349 | 23.583 (+30), cap | +295 | 95,96 | Encert |
| P2 Coder 30B | 51.200 | 17.524 | 4.800 | 134 | 22.458 | 1.155 | 23.775 (−162), no | +293 | 96,00 | Fals negatiu |
| P3 Coder 30B | 53.248 | 17.524 | 4.992 | 136 | 22.652 | 961 | 23.967 (−354), no | +291 | 96,15 | Encert |
| P4 Qwen2.5 32B | 12.288 | 18.508 | 3.072 | 214 | 21.794 | 1.819 | 23.286 (+327), cap | +468 | 18,07 | Encert |
| P5 Qwen2.5 32B | 14.336 | 18.508 | 3.584 | 216 | 22.308 | 1.305 | 23.798 (−185), no | +466 | 18,05 | Fals negatiu |
| P6 Qwen2.5 32B | 15.360 | 18.508 | 3.840 | 217 | 22.565 | 1.048 | 24.054 (−441), no | +465 | 18,07 | Fals negatiu |

P3 és l'únic cas que no compleix la regla (961 lliures, `cannot meet free memory target of 1024 MiB`). Carrega les 49 capes igualment
perquè `-ngl 99` el fixa l'usuari, i genera.

## Què diu

- El KV de Jaull coincideix amb el de llama.cpp als sis casos.
- La diferència és de 291 a 295 MiB al Coder 30B i de 465 a 468 MiB al 32B, entre l'1,3 % i el 2,1 % del projectat. Sempre és positiva.
  Ve de dues coses: Jaull compta els pesos com la mida del fitxer, i llama.cpp en deixa 171 MiB (Coder) o 426 MiB (32B) al host,
  i Jaull fixa el còmput en 256 MiB, contra 132 a 217 mesurats.
- L'1 GiB és el llindar que llama.cpp aplica a la memòria que li queda lliure, no un excés de Jaull. La diferència real entre els
  dos és la de la taula, de 0,3 a 0,5 GiB.
- Jaull té 3 encerts (P1, P3, P4) i 3 falsos negatius (P2, P5, P6). No hi ha cap fals positiu. Aprova un cas quan el marge real
  supera la seva diferència: P1 té 325 MiB de marge per sobre del llindar i la diferència és de 295; P2, P5 i P6 en tenen 131, 281 i 24
  contra 293, 466 i 465.
- Cost, **CALCULAT** per interpolació lineal: el context màxim que Jaull aprova és d'uns 49.500 tokens al Coder 30B i la regla de
  llama.cpp arriba a uns 52.600 (3.100 tokens, el 6 %). Al 32B són uns 13.600 contra uns 15.450 (1.850 tokens, el 12 %).
- Cap model ha fallat en carregar ni en generar, tampoc P3. El Coder (MoE de 3B actius) genera a 96 tok/s i el 32B dens a 18 tok/s.
  La RAM de Windows disponible no va baixar de 11.681 MiB, i l'script hauria avortat per sota de 3.072.

## Segon bloc: contextos grans

Prediccions a `preregistre-b60-2.md`. Els casos que no caben no es carreguen: l'script llegeix la línia `projected to use … vs. … free`
que llama.cpp imprimeix abans de reservar memòria i para el servidor. Això és tot el que cal per al nivell 1. Tots els casos de la taula
tenien el mateix 23.613 MiB lliures de partida.

| Cas | Model | Context | Projectat: predit i mesurat | Lliure després | Jaull (marge) | Dif. | Resultat |
|---|---|---:|---|---:|---|---:|---|
| Q1 | Coder 30B | 8.192 | 18.387 → 18.384 | 5.229 | 19.743 (+3.870), cap | +335 | Encert |
| Q2 | Coder 30B | 98.304 | 26.921 → 26.920 | −3.307 | 28.191 (−4.578), no | +247 | Encert |
| Q3 | Coder 30B | 131.072 | 30.026 → 30.024 | −6.411 | 31.263 (−7.650), no | +215 | Encert |
| Q4 | Qwen2.5 32B | 32.768 | 26.935 → 26.934 | −3.321 | 28.406 (−4.793), no | +448 | Encert |
| Q5 | Qwen2.5 32B | 65.536 | 35.159 → 35.158 | −11.545 | 36.598 (−12.985), no | +416 | Encert |
| Q6 | Qwen3.8-27B | 98.304 | 21.675 → 21.216 | 2.397 | 23.279 (+334), cap | +1.039 | Encert |
| Q7 | Qwen3.8-27B | 114.688 | 22.713 → 22.256 | 1.357 | 24.303 (−690), no | +1.023 | **Fals negatiu** |
| Q8 | Qwen3.8-27B | 131.072 | 23.756 → 23.296 | 317 | 25.327 (−1.714), no | +1.007 | Encert |

Q1 carrega i genera 512 tokens a 93,36 tok/s. Q2 a Q5 es van parar a l'ajust, en un segon, sense reservar el KV; no es sap si carregarien.

- Jaull encerta set dels vuit casos i falla Q7, un fals negatiu. Els cinc primers són lluny de la frontera (el més proper és a més de 4 GiB del
  llindar d'1 GiB lliure) i només proven que Jaull no aprova el que no cap. La precisió fina es veu a Q6, Q7 i Q8.
- L'extrapolació del còmput que vaig fer abans de mesurar (uns 0,001 MiB per token) va encertar el projectat de llama.cpp amb 1 a 3 MiB de diferència.
- **Dif.** baixa amb el context: de +335 (8K) a +215 (131K) al Coder i de +448 a +416 al 32B. El còmput de llama.cpp creix amb el context
  (de 92 a 212 MiB al Coder i de 234 a 266 al 32B, **CALCULAT** per diferència) i Jaull el fixa en 256. Al 32B a 65.536, el còmput real
  (266) ja supera els 256 de Jaull per 10 MiB. Amb contextos molt més grans la diferència podria canviar de signe (**UNVERIFIED**).

**Q6 (Qwen3.8-27B híbrid)** carrega i genera 512 tokens a 15,43 tok/s, amb la RAM de Windows disponible mínima a 9.632 MiB.
La diferència de Jaull és de +1.039 MiB, el triple que als altres models, i gairebé tota és als pesos:

| Component | Jaull | llama.cpp | Diferència |
|---|---:|---:|---:|
| Pesos a la GPU | 15.702 | 14.674 | +1.028 |
| KV (16 capes d'atenció de 65) | 6.144 | 6.144 | 0 |
| Estat recurrent | 153 | 149,6 | +3 |
| Còmput | 256 | 248 | +8 |

llama.cpp deixa 682 MiB (embeddings) en un buffer del host, i els dos buffers sumen 15.356 MiB, 346 menys que el fitxer. Aquests 346
MiB no s'han explicat (**UNVERIFIED**). El KV híbrid i l'estat recurrent de Jaull són exactes.

L'extrapolació que vaig fer del projectat del 27B (21.675 MiB) va quedar 459 MiB per sobre del real. Em vaig basar en un buffer de
pesos estimat a 15.150 MiB amb una incertesa de ±200 MiB, i era de 14.674: la incertesa estava subestimada. Aquesta columna no és de Jaull.

**Q7 i Q8.** Després de Q6 vaig estimar el projectat de Q7 en uns 22.250 MiB i el de Q8 en uns 23.300. Va sortir 22.256 i 23.296. Aquesta
estimació és posterior a Q6 i no és una predicció preregistrada; la de Jaull continua sent la de `preregistre-b60-2.md`.

- **Q7** compleix la regla de llama.cpp (queden 1.357 MiB, el llindar és 1.024). Carrega i genera 512 tokens a 15,42 tok/s, amb la RAM de Windows
  disponible mínima a 10.946 MiB. Jaull havia dit que no (−690), i és un **fals negatiu**.
- **Q8** no la compleix: queden 317 MiB i llama.cpp diu `cannot meet free memory target of 1024 MiB, need to reduce device memory by 706 MiB`.
  Jaull encerta. L'script va parar el servidor a l'ajust perquè havia fixat un marge de seguretat de 500 MiB (317 < 500), **no perquè
  desbordés la VRAM**: hi cabia amb 317 MiB lliures. No es sap si hauria carregat ni si hauria generat.
- La diferència de Jaull al 27B és de +1.039, +1.023 i +1.007 MiB (Q6, Q7, Q8): constant i gairebé tota als pesos (+1.028).
  El còmput de llama.cpp creix 16 MiB cada 16.384 tokens (248, 264 i 280), com al Coder i al 32B.
- **Regla que ho resumeix:** Jaull aprova un cas si el marge real per sobre de l'1 GiB és igual o superior a la seva diferència. Q6 té 1.373
  de marge contra 1.039 i s'aprova; Q7 en té 333 contra 1.023 i no. Les diferències per model són uns 0,3 GiB (Coder 30B), 0,45 (32B) i 1,0 (27B).
- **Cost al 27B, CALCULAT:** el context màxim que Jaull aprova és d'uns 103.600 tokens, i la regla de llama.cpp arriba a uns 120.000
  (16.300 tokens, el 14 %), interpolant entre Q7 i Q8.

**Recompte dels 14 casos (sis del primer bloc i vuit del segon):** Jaull 10 encerts i 4 falsos negatius (P2, P5, P6, Q7), cap fals positiu.
Tots els errors són cap al costat prudent.

## Límits

- Una execució per cas. P3, P4 i P6 es van repetir (P3 i P6 pel criteri de 100 MiB del preregistre) i van sortir idèntics: això només
  diu que, amb la mateixa VRAM lliure de partida, el desglossament es repeteix. La variància entre sessions o amb un altre escriptori
  és **UNVERIFIED**.
- El context es reserva però no s'omple: el prompt té 16 tokens i a la sessió n'hi ha uns 527. «Genera» vol dir amb el context gairebé buit.
  La velocitat amb el context ple i el comportament durant hores no s'han mesurat.
- Una GPU, dos models, un sol dia. Sis casos no donen percentatges.
- Els sis casos es van triar a prop de la frontera de Jaull. Els contextos grans i els casos que no caben queden per a un segon preregistre.
- L'execució 0 de P1, sense `-dev` ni `-lv 4`, va carregar i generar a 95,74 tok/s però sense xifres de memòria. Es va descartar.
- Als casos que no caben (Q2 a Q5) no es va carregar el model, i Q8 es va parar per un marge de seguretat de 500 MiB. No sabem si haurien carregat ni generat.
- El Qwen2.5 32B es va entrenar amb 32.768 tokens de context. Q4 i Q5 mesuren memòria, no qualitat.

## Conclusions

«Cap» vol dir que llama.cpp deixa 1.024 MiB lliures o més (nivell 1) a la B60, amb 23.613 MiB lliures de partida.

| Model | Context | Real (llama.cpp) | Lliures | Jaull |
|---|---:|---|---:|---|
| Coder 30B | 8.192 | Cap | 5.229 | Cap ✔ |
| Coder 30B | 49.152 | Cap | 1.349 | Cap ✔ |
| Coder 30B | 51.200 | Cap | 1.155 | No ✘ |
| Coder 30B | 53.248 | No cap | 961 | No ✔ |
| Coder 30B | 98.304 | No cap | −3.307 | No ✔ |
| Coder 30B | 131.072 | No cap | −6.411 | No ✔ |
| Qwen2.5 32B | 12.288 | Cap | 1.819 | Cap ✔ |
| Qwen2.5 32B | 14.336 | Cap | 1.305 | No ✘ |
| Qwen2.5 32B | 15.360 | Cap | 1.048 | No ✘ |
| Qwen2.5 32B | 32.768 | No cap | −3.321 | No ✔ |
| Qwen2.5 32B | 65.536 | No cap | −11.545 | No ✔ |
| Qwen3.8 27B | 98.304 | Cap | 2.397 | Cap ✔ |
| Qwen3.8 27B | 114.688 | Cap | 1.357 | No ✘ |
| Qwen3.8 27B | 131.072 | No cap | 317 | No ✔ |

Context màxim aproximat, **CALCULAT** per interpolació lineal entre els casos més propers a la frontera:

| Model | Real (regla de llama.cpp) | Jaull | Jaull es queda curt en |
|---|---:|---:|---:|
| Coder 30B | ~52.600 | ~49.500 | ~3.100 tokens (6 %) |
| Qwen2.5 32B | ~15.450 | ~13.600 | ~1.850 tokens (12 %) |
| Qwen3.8 27B | ~120.000 | ~103.600 | ~16.300 tokens (14 %) |

- En 14 casos, Jaull té 10 encerts i 4 falsos negatius. No ha aprovat cap cas que no cabia.
- Els quatre errors van cap al costat prudent: Jaull desaconsella entre un 6 % i un 14 % de context que sí funciona.
- El KV de Jaull és exacte als 14 casos, també el del 27B híbrid. La diferència ve dels pesos (Jaull usa la mida del fitxer i llama.cpp en deixa
  part al host) i d'un còmput fix de 256 MiB. Fa uns 0,3 GiB al Coder, 0,45 al 32B i 1,0 al 27B.
- Jaull aprova un cas quan el marge real sobre l'1 GiB és igual o superior a aquesta diferència.
- Una sola GPU, tres models i un sol dia, amb una execució per cas (P3, P4 i P6 repetits i idèntics) i el context gairebé buit a la prova de generació.
