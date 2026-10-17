# Qwen2.5-0.5B FP16: runner de Jaull versus model.generate

Diagnòstic local del 14 d'octubre de 2026. No és una campanya de ranking,
no s'ha importat al store de benchmarks i no modifica registres anteriors.

## Pregunta i resultat

L'usuari havia obtingut 18,355 tok/s amb Transformers FP16, mentre Results
estimava aproximadament 170 tok/s amb la heurística d'ample de banda.
Es comprova si el runner de Jaull mesura una execució incorrectament col·locada
o és particularment més lent que l'API habitual de Transformers.

| Mètode, mateix model carregat | Passada 1 | Passada 2 | Mediana, tok/s |
|---|---:|---:|---:|
| `_decode_once` de producció | 20,886 | 18,368 | 19,627 |
| `model.generate`, finestra de decode comparable | 17,139 | 18,598 | 17,868 |

Ordre intercalat: runner/generate, generate/runner. Una passada completa de
calentament per mètode abans de prendre dades, dos segons entre mesures.
Tots dos recorren els mateixos inputs de cada forward: 129 crides per mètode,
sense cap divergència en la comprovació no cronometrada de tokens.

Els 290 tensors de paràmetres són `cuda:0` i `torch.float16`; els dos buffers
també són a CUDA. `model.training` és fals. Pesos: 988.065.536 bytes.
No s'observa offload de paràmetres a CPU. Ambdós mètodes mantenen aproximadament
1.005 MB d'allocacions de PyTorch; això no és tota la VRAM atribuïda al procés.

La lectura de 18,36 tok/s de l'usuari és compatible amb aquest diagnòstic.
No hi ha una penalització de 9x exclusiva del bucle de Jaull: l'API estàndard
també queda en aquest ordre de magnitud. La heurística de 170 tok/s falla
àmpliament en aquest cas i no s'ha reajustat cap coeficient per amagar-ho.

## Entorn i identitat

- RTX 2060 6 GiB; driver 617.42; Ryzen 5 3600; WSL2 `6.18.54.1`.
- Python 3.12.3, PyTorch `2.13.0+cu130`, Transformers `5.15.0`, CUDA runtime 13.0.
- Sis fils intra-op, dotze inter-op; afinitat de CPU 0-11; atenció `sdpa`.
- Snapshot local: `7ae557604adf67be50417f59c2c2f167def9a775`.
- SHA256 de `model.safetensors`:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
- Checkout base `2dd0eca706d0d4567ecfd6a08f525fcc8d437d80`, amb canvis locals.
- SHA256 del worker de producció llegit:
  `0788386e01465f72a72287c125b5dd4732c9ec18cf853ac8692477bd500ef5de`.
- SHA256 del script del diagnòstic:
  `6928681ee4d7a90f32178a34cadf01b52e12871eb29c68dab2c7543ca35211dc`.

Els manifests fixen els bytes de la comparació entre mètodes. El registre
original de l'usuari conservava `revision=main` i cap SHA: no es reconstrueix
retrospectivament una identitat immutable que aquell registre no va guardar.

## Contracte temporal

El runner fa prefill i selecciona el primer token abans de cronometrar 128
forwards amb KV cache. `generate` produeix 129 tokens nous: el primer correspon
al prefill i els altres 128 als forwards de decode. Un pre-hook inicia el
cronòmetre comparable abans del segon forward; el final inclou el tancament
de `generate`. És una aproximació instrumentada de la mateixa finestra, no
una equivalència perfecta de tot el cost d'API. També es conserva el temps
end-to-end de `generate` a `summary.json`.

S'utilitza el prompt sintètic cru `Benchmark generation.`, sense plantilla,
amb argmax/greedy, `use_cache=True` i una sola seqüència. La penalització de
repetició s'ha fixat explícitament a 1,0: una primera comparació exploratòria
havia heretat el valor 1,1 del model i no seguia els mateixos tokens. No es
barregen les dades d'aquella exploració amb la taula final.

## Límits i observacions

- Només dues crides per mètode en un mateix procés i sessió. No és una prova
  que compleixi la porta de tres crides independents per ordenar Fastest.
- Hi havia ús de GPU abans de començar. La telemetria NVML és global del
  dispositiu; WDDM no permet atribuir-ne tota la memòria o ús a aquest procés.
  No es pot certificar absència d'altra càrrega.
- Durant les mesures s'observen aproximadament 25-40% d'ús global de GPU,
  44-45 graus i rellotges variables. El temps CPU és proper al temps de paret
  d'un nucli. És compatible amb cost de llançament/host i una GPU poc saturada,
  però **no identifica el coll d'ampolla** sense un perfil temporal.
- No s'ha perfilat CUDA/Python ni canviat el mode d'energia, els fils, les
  versions, l'atenció o el tipus de cache per buscar una configuració millor.
- No hi ha comparació amb GGUF/llama.cpp ni canvi de quantització.

## Fitxers

- `metadata.json`: entorn i manifest SHA256 dels fitxers locals.
- `events.jsonl`: col·locació, configuració efectiva, warmups, mesures i telemetria.
- `summary.json`: dades numèriques i comprovació del recorregut de tokens.
- `diagnostic-script.py.txt`: còpia exacta del script executat; només diagnòstic.

Reexecució offline, amb una carpeta de sortida nova:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
  uv run --offline --python 3.12 python \
  validation/qwen2.5-0.5b-transformers-decode-2060-20261014/diagnostic-script.py.txt \
  --output /tmp/jaull-transformers-new-check --rounds 2
```
