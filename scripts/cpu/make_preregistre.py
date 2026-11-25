"""Build docs/TFG/preregistre-cpu.md from the predictions already written by predict.py.

    uv run python scripts/cpu/make_preregistre.py
"""

from __future__ import annotations

import csv
from pathlib import Path

PRED = Path("validation/cpu-20261125/predicciones/predicciones.csv")
CASES = Path("scripts/cpu/cases.csv")
COMMIT = (Path("validation/cpu-20261125/predicciones/commit.txt").read_text()).strip().splitlines()
commit = COMMIT[0].split("=")[1]
cases = {c["id"]: c for c in csv.DictReader(CASES.open(encoding="utf-8"))}
rows = list(csv.DictReader(PRED.open(encoding="utf-8")))


def fmt(x: str) -> str:
    return f"{float(x):,.0f}".replace(",", ".")


table = ["| Cas | Model | Quant. | Context | Pesos | KV | Overhead | Físic | Total | Modes |", "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
for r in rows:
    c = cases[r["id"]]
    model = c["repo"].split("/")[1].replace("-GGUF", "")
    quant = "MXFP4_MOE" if c["quant"] == "unknown" else c["quant"]
    table.append(
        f"| {r['id']} | {model} | {quant} | {int(r['ctx']):,} | {fmt(r['weights_mib'])} | {fmt(r['kv_mib'])} | "
        f"{fmt(r['overhead_mib'])} | {fmt(r['physical_mib'])} | {fmt(r['total_mib'])} | {c['modes'].replace(';', ' i ')} |".replace(",", ".")
    )

mmap_ids_list = [c["id"] for c in cases.values() if "mmap" in c["modes"].split(";")]
mmap_ids = f"{mmap_ids_list[0]} a {mmap_ids_list[-1]}"
by_id = {r["id"]: r for r in rows}
ornith_kv4 = fmt(by_id["C23"]["kv_mib"])
ornith_kv32 = fmt(by_id["C24"]["kv_mib"])

text = f"""# Preregistre de la campanya CPU

Escrit el 2026-11-25, **abans de fer cap mesura**. Les prediccions són de Jaull (commit `{commit[:7]}`, `src/` sense canvis) i no es modificaran.
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

{chr(10).join(table)}

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
Cada cas es mesura amb `--no-mmap` (la càrrega és clara). Els casos {mmap_ids} també es mesuren amb mmap, on la càrrega sempre «funciona» però pot anar a rastres.
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
5. **Ornith 1.5 9B (arquitectura Qwen3.5 híbrida).** Jaull compta 33 capes, de les quals 8 tenen KV ({ornith_kv4} MiB a 4.096 tokens i {ornith_kv32} MiB a 32.768) i la resta un estat recurrent de mida constant.
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
"""
Path("docs/TFG/preregistre-cpu.md").write_text(text, encoding="utf-8")
print("docs/TFG/preregistre-cpu.md written")
