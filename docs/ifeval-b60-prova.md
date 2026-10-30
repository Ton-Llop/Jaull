# IFEval a la B60: prova externa

Registre d'una prova d'IFEval sobre tres models a la B60 de Biosfer. Es fa amb
`lm-evaluation-harness`, fora de Jaull, contra un `llama-server.exe` de Windows. No és l'IFEval
del pilot de qualitat de Jaull: aquell fa servir un detector d'idioma amb llavor i no és comparable
amb les xifres publicades. Aquí es fa servir la tasca `ifeval` de la llibreria tal com ve.

Estats: **VALIDATED** (execució real), **UNVERIFIED** (hipòtesi), **PENDENT** (no fet).

## Inici

Primera execució completa (Qwen3-Coder 30B-A3B): **2026-10-30**. L'avaluador la va donar per acabada
a les 13:19:53 CET després de 2528 s; l'inici surt, doncs, cap a les 12:38 (l'hora de 12:41 que es
va apuntar era la del moment d'obrir aquest document).

## Muntatge

| Peça | Valor |
|---|---|
| Targeta | Intel Arc Pro B60, 24 GiB, dispositiu `Vulkan1` a Windows |
| Driver d'Intel | `32.0.101.8976` |
| Servidor | `llama-server.exe`, `llama.cpp` b11889, binari de Windows amb Vulkan |
| Evaluador | `lm_eval`, tasca `ifeval`, dins d'un entorn virtual a `monk-e` (WSL) |
| Connexió | túnel SSH invers: el servidor de Windows (`127.0.0.1:8080`) es veu des de WSL com `127.0.0.1:8081` |
| API | `local-chat-completions`, una petició alhora (`num_concurrent=1`) |

El cortafuegs de Windows no deixa que WSL arribi directament al servidor, per això el túnel invers
amb `ssh.exe` de Windows. No s'ha tocat el cortafuegs.

## Condicions fixes per als tres models

- Temperatura 0, plantilla de xat del GGUF (`--apply_chat_template`).
- Servidor: `-dev Vulkan1 -ngl 99 -c 8192 -np 1`, flash attention amb el valor per defecte.
- La tasca fixa `max_gen_toks` a 1280 i no usa mostres d'exemple.
- Els 541 prompts complets, sense `--limit`.

Comanda del servidor (Coder):

```
./llama-server.exe -m 'C:\llama-win\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf' -dev Vulkan1 -ngl 99 -c 8192 -np 1 --host 127.0.0.1 --port 8080
```

Comanda de l'evaluador:

```
lm_eval --model local-chat-completions \
  --model_args model=qwen3-coder-30b,base_url=http://127.0.0.1:8081/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
  --tasks ifeval --apply_chat_template \
  --gen_kwargs temperature=0 \
  --output_path ~/ifeval-results/qwen3-coder-30b --log_samples
```

Els resultats i les mostres queden a `~/ifeval-results/<model>/`.

## Prova de muntatge

Cinc prompts amb `--limit 5` contra el Coder, a les 12:36: **VALIDATED** que el camí sencer
funciona. Les quatre mètriques van donar 1,0, però són cinc prompts i no diuen res del model.
Van tardar 1 min 57 s, uns 23,6 s per petició.

Aquests 23,6 s eren més del que costarien 1280 tokens a 99 t/s (uns 13 s). L'execució completa
va trigar 4,7 s per prompt de mitjana (2528 s entre 541), de manera que la xifra de la prova curta
estava inflada. **UNVERIFIED** per què: la hipòtesi més probable és l'escalfament de les primeres
peticions, però no s'ha mirat el registre del servidor.

## Resultats

Les quatre mètriques són les de la tasca: `prompt_level_strict_acc`, `prompt_level_loose_acc`,
`inst_level_strict_acc` i `inst_level_loose_acc`.

| Model | Quantificació | Estat | Durada | prompt strict | prompt loose | inst strict | inst loose |
|---|---|---|---|---:|---:|---:|---:|
| Qwen3-Coder 30B-A3B Instruct | Q4_K_M | **VALIDATED**, 541 prompts | 2528 s (42 min) | 0,789 ± 0,018 | 0,824 ± 0,016 | 0,853 | 0,878 |
| Qwen3.8-27B | Q4_K_M | PENDENT | | | | | |
| Qwen2.5 32B Instruct | Q3_K_M | PENDENT (no descarregat) | | | | | |

### Qwen3-Coder 30B-A3B, resultat

Fitxer: `~/ifeval-results/qwen3-coder-30b/qwen3-coder-30b/results_2026-10-30T13-19-53.014322.json`.
`lm_eval` 0.4.13, tasca `ifeval` versió 4, 541 mostres efectives, 0 shots, `max_gen_toks` 1280,
llavors 0 / 1234 / 1234. Valors exactes:

| Mètrica | Valor | Error estàndard |
|---|---:|---:|
| `prompt_level_strict_acc` | 0,78928 | 0,01755 |
| `prompt_level_loose_acc` | 0,82440 | 0,01637 |
| `inst_level_strict_acc` | 0,85252 | no calculat |
| `inst_level_loose_acc` | 0,87770 | no calculat |

Una execució, sense repeticions. No es compara amb cap xifra publicada: el servidor, la plantilla
de xat i la quantificació (Q4_K_M) no són els d'una publicació.

Per estimar la resta: el Coder ha trigat 42 minuts. Si el temps depengués només de la generació,
el 27B (15,55 t/s contra 99,18) tardaria unes 6,4 vegades més, d'ordre de 4 a 5 hores. És una
estimació, **UNVERIFIED**.

## Per anotar en acabar cada model

- Hora de final i durada.
- `sha256` del GGUF i comanda exacta del servidor.
- Si les respostes porten text de raonament (`<think>`). Cal mirar una mostra: el Qwen3.8 pot raonar i
  això canvia la longitud i el resultat. La política ha de ser la mateixa per als tres.
- Tokens per segon que dona el servidor durant la prova.
- Incidències (talls de túnel, reintents, errors).

## Incidències

Cap de moment.

## Límits

- Una sola màquina i una sola execució per model, sense repeticions: no hi ha variància mesurada.
- El Qwen2.5 32B està en Q3_K_M i els altres dos en Q4_K_M. Una part de la diferència pot ser de
  quantificació i no del model.
- No es compara amb xifres publicades d'IFEval: el servidor, la plantilla de xat i la quantificació
  no són els de cap publicació.
