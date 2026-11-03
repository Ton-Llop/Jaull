# B60: WSL2 contra Windows, en una taula

Mateixa targeta (Intel Arc Pro B60, 24 GiB), mateix driver d'Intel (`32.0.101.8976`), mateix
`llama.cpp` (b11889) i els mateixos fitxers GGUF. Només canvia on s'executa.

- **WSL2 + Dozen:** Vulkan traduït a Direct3D 12 dins de WSL2.
- **Windows natiu:** el binari de Vulkan de Windows, que fa servir el driver d'Intel directament.

## Velocitat

Tokens per segon, mesurats amb `llama-bench` i `-ngl 99`. Més alt és millor.

| Model | | Prompt (t/s) | Generació (t/s) | Generació, a escala |
|---|---|---:|---:|---|
| **Ornith 9B** Q8_0 | WSL2 | 141,7 | 4,1 | █ |
| | Windows | 975,8 | **36,2** | ███████ |
| **Qwen3-Coder 30B-A3B** Q4_K_M | WSL2 | 34 ¹ | 2,5 | ▏ |
| | Windows | 788,8 | **99,2** | ████████████████████ |
| **Qwen3.8-27B** Q4_K_M | WSL2 | 49,9 | ✗ cau | ✗ |
| | Windows | 339,4 | **15,6** | ███ |
| **Qwen2.5-32B** Q4_K_M ² | WSL2 ³ | 86,6 | 3,9 | █ |
| | Windows | 261,1 | **18,1** | ████ |

Escala de les barres: cada `█` són 5 t/s.

¹ Mesura amb molta variància (± 27); no s'hi pot confiar gaire.

² Windows, el 2026-11-03, amb la mateixa comanda (`-ngl 99 -p 128 -n 64 -r 2`) i el mateix build
(b11889). Fitxer de 18,48 GiB, 32,76 B de paràmetres. Les desviacions d'`llama-bench` són de ± 1,56 t/s
al prompt i ± 0,01 t/s a la generació. El 32B (18,1 t/s) va una mica més de pressa que el 27B (15,6),
tot i tenir més paràmetres, i molt més a poc que el Coder, que és un MoE amb uns 3 B actius. Per què el
27B va més lent que el 32B no s'ha investigat (**UNVERIFIED**); el 27B és una arquitectura híbrida (`qwen35`).

³ WSL2, el 2026-11-03: mesurat amb el Benchmark de Jaull (`pp512` i `tg128`) i no amb `llama-bench`
directe, de manera que les mides de prompt i de generació no són les de Windows (128 i 64). A D121,
Jaull va donar 4,8 t/s i `llama-bench` 4,10 al mateix entorn, així que la comparació és aproximada. Una
sola execució. El 32B **no cau** a WSL2.

## Quantes vegades més ràpid va Windows

| Model | Prompt | Generació |
|---|---:|---:|
| Ornith 9B | 6,9 × | **8,8 ×** |
| Qwen3-Coder 30B-A3B | 23 × ¹ | **39,5 ×** |
| Qwen3.8-27B | 6,8 × | no es pot comparar (a WSL2 cau) |
| Qwen2.5-32B | 3,0 × ³ | **4,6 ×** ³ |

## Què vol dir

- A WSL2 + Dozen la targeta va entre **4,6 i 40 vegades més lenta** que a Windows amb el mateix
  binari i el mateix model. La targeta no és el problema. El 32B dens és el que perd menys.
- El Qwen3.8-27B **cau sempre a WSL2** (`Segmentation fault` dins de `libLLVM-14.so`, del driver
  d'Intel) i **funciona a Windows** amb el mateix fitxer.
- Un MoE com el Coder 30B va més lent que el 9B dens a WSL2, i a Windows va unes 2,7 vegades més
  ràpid, que és el que s'esperaria d'un entorn sa.

## Límits

Una sola màquina. A WSL2 es va fer servir flash attention desactivada, i a Windows el valor per
defecte. Les mides de prompt i les repeticions (`-p`, `-n`, `-r`) tampoc no són idèntiques. Una
diferència d'aquest ordre no s'explica només per això, però la mesura estricta a Windows amb
`-fa off` és **PENDENT**. Detall i fonts a
[b60-comparativa-wsl2-windows.md](b60-comparativa-wsl2-windows.md).
