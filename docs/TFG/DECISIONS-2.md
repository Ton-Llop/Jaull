# Decisions de l'entorn d'experimentació (fora de Jaull)

**Data d'inici:** 3 de novembre de 2026.

## 1. Per a què serveix aquest document

[`DECISIONS.md`](DECISIONS.md) explica per què Jaull té la forma que té (D001-D122). Aquest
document recull les decisions que no són del codi de Jaull: quines eines i quins models
s'usen per fer les proves de la memòria, i com es fan. Són material de treball per redactar el TFG,
no una especificació.

Els IDs són `X001`, `X002`... (X de "extern"), i no es renumeren. Quan una decisió canviï, no
s'esborra l'antiga: es marca com a substituïda i s'enllaça la nova. Els estats són els mateixos que a
`DECISIONS.md`: **VALIDATED** (comprovat amb una execució real), **UNVERIFIED** (hipòtesi o
impressió) i **PENDENT** (no fet).

### X001. L'arnès agentic és OpenCode

**Data i necessitat:** 2026-11-03. Després de l'IFEval cal provar els models com a agent de
programació sobre Lipopótamo, el codi de Biosfer. Calia triar l'arnès: el programa que dona al
model les eines per llegir, editar i executar ordres.

**Alternatives vistes:** Aider, Pi i OpenCode. L'Aider no s'ha provat. Pi (versió 1.5.0) i OpenCode s'han
instal·lat i configurat contra el mateix `llama-server`.

**Decisió (de l'autor):** OpenCode.

**Per què, amb el que sabem:**
- Els permisos són de sèrie. Es configuren a `opencode.json` amb `allow`, `ask` o `deny`, patrons per
  a les ordres de `bash`, `external_directory` i `doom_loop` (talla el model si repeteix la mateixa
  crida). Pi no en porta: calia una extensió de tercers, `pi-permission-system` 0.8.0, d'un sol autor,
  de la qual no s'ha comprovat la compatibilitat amb Pi 1.5.0.
- La documentació d'OpenCode inclou una configuració per a `llama-server` amb un proveïdor
  `@ai-sdk/openai-compatible`, i una compactació automàtica de la sessió quan el context s'acosta al límit.
- A la prova de fum amb Pi, l'autor va veure molts bucles. **UNVERIFIED** que la causa sigui Pi: no s'ha
  mirat la sessió, i hi havia un context de només 16384 tokens.
- L'autor considera OpenCode una plataforma més estable. **UNVERIFIED**: no s'ha fet cap prova
  d'estabilitat ni s'ha comparat de manera sistemàtica amb Pi.

**Prova de fum amb OpenCode** (Qwen3-Coder 30B, tasca: escriure tests de `pytest` per a `calc.py`).
Primera execució, fallida per l'entorn (el venv no era actiu, no hi havia `pytest`): el model va escriure els
tests, no va arribar a executar `pytest` i va donar la tasca per bona. Segona execució: 29,9 s, `pytest`
amb 3 tests aprovats, i `calc.py` al 100 % de cobertura en la verificació externa. **VALIDATED** com a
prova que el muntatge funciona; no és cap mesura de qualitat.

**Costos i riscos que s'accepten:**
- OpenCode usa la pantalla alternativa del terminal: no hi ha historial de desplaçament a `tmux`. Per
  treure la sessió es fa servir `/export` (Markdown).
- A `/models` apareixen també els models gratuïts del núvol d'OpenCode. Amb el codi de Biosfer **només** s'usen
  els del proveïdor local `b60`, perquè el codi no pot sortir de la màquina.
- No s'ha anotat la versió exacta d'OpenCode instal·lada.

**Límits:** una sola prova de fum, una sola tasca. No es pot dir que OpenCode sigui millor que Pi o
l'Aider: és l'arnès triat, per les raons de dalt.

**Fonts:** [OpenCode, proveïdors](https://opencode.ai/docs/providers/),
[permisos](https://opencode.ai/docs/permissions/), [configuració](https://opencode.ai/docs/config/).

### X002. El model del benchmark agentic és el Qwen3-Coder 30B-A3B (Q4_K_M)

**Data i necessitat:** 2026-11-03. Es disposava de tres models amb IFEval: el Coder, el Qwen2.5 32B i el
Qwen3.8 27B. Calia triar quin s'usa per al benchmark llarg sobre Lipopótamo.

**Decisió (de l'autor):** el Qwen3-Coder 30B-A3B Q4_K_M, per velocitat i per context.

**Evidència:**

| | Coder 30B-A3B | Qwen3.8 27B | Qwen2.5 32B |
|---|---:|---:|---:|
| Generació a Windows natiu (t/s) | 99,2 | 15,55 | no mesurat |
| IFEval, prompt strict | 0,789 ± 0,018 | 0,821 ± 0,017 (sense raonament) | 0,797 ± 0,017 |
| Context que cap a la B60 | 32768 mesurat | no mesurat | no mesurat |

- **Context, VALIDATED:** a `-c 32768` llama.cpp projecta 20712 MiB (pesos 17524, KV 3072, còmput 116) de
  23613 MiB lliures, i en queden 2901. El KV és de 96 MiB per cada 1000 tokens. Vegeu D120.
- **Velocitat, VALIDATED** per al Coder i el 27B (D121). El Coder és uns 6,4 cops més ràpid.
- **IFEval no discrimina:** les diferències entre els tres són de l'ordre de l'error estàndard de la
  diferència (uns 0,024). A més, IFEval mesura el seguiment d'instruccions de format, no la capacitat de
  programar com a agent.
- **Contextos dels densos, UNVERIFIED:** els 16384 tokens que figuren a la configuració dels altres dos
  vénen d'una estimació de KV per dimensions (uns 0,25 GiB per 1000 tokens), no d'una mesura.

**Què no es pot afirmar:** que el Coder programi millor, o igual, que els models densos com a agent.
No s'ha mesurat. La decisió és pràctica (rapidesa i context), i els resultats del benchmark valdran per
a aquest model.

**PENDENT:** una prova de fum curta amb el 27B i el 32B per poder dir què passa amb els altres dos.

**Fonts:** [registre d'IFEval](../ifeval-b60-prova.md), D120, D121.

### X003. Protocol d'execució del benchmark agentic

**Data i necessitat:** 2026-11-03. Les dues proves de fum van mostrar què cal controlar abans de comparar
res. **Estat: PENDENT**, el protocol es tanca abans de veure resultats sobre Lipopótamo.

**Decisions preses:**
1. **Còpia d'un sol ús.** L'agent treballa en una còpia de Lipopótamo a `~/` de WSL (no a `/mnt/c`), amb el
   commit fixat. No toca mai l'original. Requereix permís per escrit de Biosfer, **PENDENT**.
2. **Permisos tancats.** `"*": "deny"` i només es permet llegir, editar, `glob` i `grep`; a `bash`, `python`,
   `python3`, `pytest`, `coverage`, `mutmut`, `git status`, `git diff`, `git log` i `ls`. `external_directory` i
   `doom_loop` en `deny`. Això no és un aïllament: `ls` i `python` poden llegir fora de la carpeta. La
   protecció real és la còpia d'un sol ús i que no hi hagi credencials a la màquina.
3. **Servidor i context.** `llama-server.exe` de Windows natiu amb `-c 32768 -np 1 --jinja`, i el mateix
   `limit.context` a `opencode.json`. Compactació amb els valors per defecte (`auto` actiu, `prune` desactivat,
   `reserved` 10000, és a dir, es compacta cap als 22.700 tokens). Es declara com a paràmetre fix.
4. **Verificació externa, no la paraula de l'agent.** A la primera prova de fum el model va afirmar que els
   tests passaven sense haver-los executat. L'èxit es mesura amb `pytest` i amb `coverage` sobre el codi
   original (`coverage run --include=<fitxer>`), que l'autor executa després.
5. **Carpeta neta a cada execució.** A la primera prova, l'agent va deixar `test_runner.py` i
   `test_simple.py` que ningú havia demanat, i van entrar a la cobertura total (97 % contra el 100 % del
   codi sota prova). Es parteix d'un motlle amb només el codi inicial i un `venv` compartit fora de la carpeta.
6. **Mètriques per execució:** temps, torns, tokens, compactacions, ordres denegades, fitxers extra
   creats, tests que passen, cobertura del codi original i, si cal, puntuació de mutacions.

**Pendent abans de començar:** mesurar la variància de l'agent amb ell mateix (tres execucions de la mateixa
tasca des d'una carpeta neta), decidir la llista de tasques i el permís de Biosfer.

**Límits previstos:** una sola base de codi, una execució per model i tasca (si la variància ho permet), i un
tipus de tasca (escriure tests). Els resultats descriuen aquest model en aquest cas, no en general.

**Fonts:** X001, X002, D120.
