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
| Qwen3.8-27B, amb raonament | Q4_K_M | **VALIDATED**, 541 prompts (en dues parts); 176 respostes gairebé buides | 4 h 19 min + 3 h 44 min | 0,634 ± 0,021 | 0,641 ± 0,021 | 0,615 | 0,621 |
| Qwen3.8-27B, sense raonament (`--reasoning off`) | Q4_K_M | **VALIDATED**, 541 prompts | 3 h 05 min 46 s | 0,821 ± 0,017 | 0,848 ± 0,015 | 0,872 | 0,897 |
| Qwen2.5 32B Instruct | Q4_K_M | **VALIDATED**, 541 prompts | 2 h 24 min 35 s | 0,797 ± 0,017 | 0,812 ± 0,017 | 0,859 | 0,871 |

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

Recompte de respostes gairebé buides, amb el mateix criteri que als altres models (menys de 20 caràcters), fet després de
tenir els altres dos resultats:

| Respostes | Nombre | Encerts estrictes |
|---|---:|---:|
| Amb resposta | 535 | 422 (78,9 %) |
| Gairebé buides | 6 | 5 |

Els 422 + 5 = 427 encerts sobre 541 donen 0,7893, igual que el resum de `lm_eval`. **VALIDATED**. Les 6 que són buides no
s'han mirat (**UNVERIFIED** per què).

Per estimar la resta: el Coder ha trigat 42 minuts. Si el temps depengués només de la generació,
el 27B (15,55 t/s contra 99,18) tardaria unes 6,4 vegades més, d'ordre de 4 a 5 hores. És una
estimació, **UNVERIFIED**.

### Qwen3.8-27B, resultat

Resultat final a `~/ifeval-results/qwen3.8-27b/` (2026-11-02, 12:43). `lm_eval` 0.4.13, mateixes condicions que el Coder:
`-c 8192 -np 1`, temperatura 0, 1.280 tokens màxims. Valors del resum de `lm_eval`:

| Mètrica | Valor | Error estàndard |
|---|---:|---:|
| `prompt_level_strict_acc` | 0,6340 | 0,0207 |
| `prompt_level_loose_acc` | 0,6414 | 0,0206 |
| `inst_level_strict_acc` | 0,6151 | no calculat |
| `inst_level_loose_acc` | 0,6211 | no calculat |

L'execució va anar en dues parts. La primera va començar el 30 d'octubre a les 15:03 i es va aturar a les 19:22, quan WSL
es va apagar, amb 301 respostes desades a la memòria cau. La segona, del 2 de novembre des de les 08:58:41, va fer les 240
restants en 3 h 44 min (56,18 s per prompt). El `total_evaluation_time_seconds` del resultat només recull la segona part.

El resultat és força inferior al del Coder (0,789), i l'estona per prompt és d'unes 12 vegades més. Revisant les mostres,
cap resposta conté etiquetes `<think>` al text, però **176 de les 541 (32,5 %) són gairebé buides** (menys de 20 caràcters).
**VALIDATED**: el recompte. **UNVERIFIED**: la causa. És coherent amb un model que raona abans de respondre, amb el raonament
separat del contingut pel servidor, i que en un terç dels prompts esgota els 1.280 tokens sense arribar a la resposta. Una
resposta buida compta com a error, de manera que aquest resultat no és comparable amb el del Coder, que no raona. No s'ha
comprovat al registre del servidor que el raonament sigui el motiu de les respostes buides.

Separant les respostes segons si arriben o no, a nivell de prompt estricte: de les 365 amb resposta, n'encerten 332 (91,0 %);
de les 176 gairebé buides, n'encerten 11 (6,3 %), probablement restriccions que una resposta buida compleix per casualitat.
Aquesta taxa del 91 % no es pot llegir com una nota corregida ni comparar amb la del Coder: els prompts que acaben dins del
límit de tokens poden ser els més curts o més fàcils. Només mostra que el 0,634 reflecteix el límit de tokens més que la
capacitat del model.

Per tenir una comparació en les mateixes condicions calia repetir el Qwen3.8 27B amb el raonament desactivat
(`--reasoning off` al servidor), amb una memòria cau i una carpeta de sortida noves: la memòria cau de `lm_eval` s'indexa per
la petició i no pels paràmetres del servidor, i en reaprofitaria les respostes amb raonament. S'ha fet; vegeu la secció següent.

### Qwen3.8-27B sense raonament, resultat

Resultat final a `~/ifeval-results/qwen3.8-27b-sinrazonar/` (2026-11-02, 18:32). Mateixes condicions que les altres execucions
(`-c 8192 -np 1`, temperatura 0, 1.280 tokens màxims, 541 prompts), amb dues diferències: el servidor porta `--reasoning off`, i la
memòria cau és nova (`~/ifeval-cache/qwen3.8-27b-sinrazonar`), perquè no reaprofités les respostes de l'execució anterior.
Valors del resum de `lm_eval`:

| Mètrica | Valor | Error estàndard |
|---|---:|---:|
| `prompt_level_strict_acc` | 0,8207 | 0,0165 |
| `prompt_level_loose_acc` | 0,8484 | 0,0154 |
| `inst_level_strict_acc` | 0,8717 | no calculat |
| `inst_level_loose_acc` | 0,8969 | no calculat |

Una sola tirada, de les 15:26 a les 18:32: 3 h 05 min 46 s, 20,60 s per prompt, contra els 56 s per prompt de l'execució amb
raonament. És unes 4,4 vegades el temps del Coder i 1,3 vegades el del Qwen2.5 32B.

La nota puja de 0,634 a 0,821 a nivell de prompt estricte. Recompte sobre les mostres, amb el mateix criteri que a
l'execució anterior (menys de 20 caràcters és una resposta gairebé buida):

| Respostes | Nombre | Encerts estrictes |
|---|---:|---:|
| Amb resposta | 531 | 436 (82,1 %) |
| Gairebé buides | 10 | 8 |

Els 436 + 8 = 444 encerts sobre 541 donen 0,8207, igual que el resum de `lm_eval`. **VALIDATED**: les respostes gairebé buides
passen de 176 a 10, així que l'execució sense raonament ja no té el problema que feia el 0,634 poc fiable. No s'ha separat
quanta part del salt de la nota ve de les respostes recuperades i quanta d'un canvi en la qualitat de les que ja hi eren. Les
10 que continuen buides no s'han mirat (**UNVERIFIED** per què); 8 d'elles compten com a encert, probablement perquè una
resposta buida compleix per casualitat algunes restriccions.

Comparació dels tres models, a nivell de prompt estricte: 0,821 (Qwen3.8 27B), 0,797 (Qwen2.5 32B), 0,789 (Coder). Les
diferències són de l'ordre de l'error estàndard de la diferència (uns 0,024). Amb una execució per model no es pot ordenar els
tres amb seguretat.

### Qwen2.5 32B Instruct, resultat

Resultat final a `~/ifeval-results/qwen2.5-32b/` (2026-11-02, 15:19). `lm_eval` 0.4.13, mateixes condicions que el Coder:
`-c 8192 -np 1`, temperatura 0, 1.280 tokens màxims, 541 prompts. Valors del resum de `lm_eval`:

| Mètrica | Valor | Error estàndard |
|---|---:|---:|
| `prompt_level_strict_acc` | 0,7967 | 0,0173 |
| `prompt_level_loose_acc` | 0,8115 | 0,0168 |
| `inst_level_strict_acc` | 0,8585 | no calculat |
| `inst_level_loose_acc` | 0,8705 | no calculat |

Una sola tirada, 2 h 24 min 35 s (16,04 s per prompt segons la barra de progrés de `lm_eval`), unes 3,4 vegades el
temps del Coder.

A nivell de prompt estricte, 0,797 contra 0,789 del Coder. La diferència, 0,008, és menor que l'error estàndard de cada
mesura (0,017 i 0,018). Amb una execució per model no es pot dir que un segueixi millor les instruccions que l'altre, i
tampoc que siguin iguals.

Quantificació: Q4_K_M. **VALIDATED** per la comanda del servidor, que porta el fitxer `Qwen2.5-32B-Instruct-Q4_K_M.gguf`
(hi ha també el seu `.sha256` al costat). La comparació de `sha256sum` entre l'origen i la còpia a `C:\llama-win` va
coincidir, segons Ton; el valor del hash no s'ha anotat. Comanda del servidor:

```
./llama-server.exe -m 'C:\llama-win\Qwen2.5-32B-Instruct-Q4_K_M.gguf' -dev Vulkan1 -ngl 99 -c 8192 -np 1 --host 127.0.0.1 --port 8080
```

Recompte de respostes gairebé buides, amb el mateix criteri que al 27B (menys de 20 caràcters):

| Respostes | Nombre | Encerts estrictes |
|---|---:|---:|
| Amb resposta | 534 | 424 (79,4 %) |
| Gairebé buides | 7 | 7 |

Els 424 + 7 = 431 encerts sobre 541 donen 0,7967, igual que el resum de `lm_eval`. **VALIDATED**: el Qwen2.5 32B no té el
problema de respostes buides del 27B amb raonament (7 de 541 contra 176). Les 7 que són buides no s'han mirat (**UNVERIFIED**
per què); totes compten com a encert, probablement perquè una resposta buida compleix per casualitat algunes restriccions.

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
- El Qwen3.8 27B s'ha avaluat de dues maneres, amb raonament i sense. La comparació amb el Coder i el Qwen2.5 32B, que no
  raonen, és la de l'execució sense raonament; l'altra es conserva perquè mostra l'efecte del límit de 1.280 tokens.
- Els tres models són en Q4_K_M, però el fitxer del Qwen3.8 27B és `Qwen3.8-27B-UD-Q4_K_M.gguf`. El nom porta `UD`, així que
  potser no és la mateixa mescla de bits que una Q4_K_M estàndard com la dels altres dos (**UNVERIFIED**, no s'ha comparat).
  Fins que es comprovi, la quantificació no es pot descartar del tot com a part de les diferències.
- No es compara amb xifres publicades d'IFEval: el servidor, la plantilla de xat i la quantificació
  no són els de cap publicació.
