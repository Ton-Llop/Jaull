# Captures per a la memòria

Captures reals fetes durant les proves. No són SVG generats (els de `docs/assets/` sí que ho són).

| Fitxer | Què mostra | Data |
|---|---|---|
| `jaull-tui-results-biosfer-b60-2026-11-10.png` | Pantalla de resultats de la TUI de Jaull al PC de Biosfer. Perfil «Coding · Quality · Context 4096 · 1 concurrent request». Maquinari detectat: AMD Ryzen 7 5700G, 13.6 GiB de RAM, Direct3D12 (Arc Pro). Cinc models recomanats; el 3 (Qwen3.6 35B A3B, Q3_K) surt «Compatible · 17.38 GiB». | 2026-11-10, 13:17 |

Coses que es veuen a la captura i que convé tenir clares si es fa servir:

- La RAM que Jaull detecta (13.6 GiB) és la que veu WSL2, no la del PC sencer. Això no s'ha investigat
  en aquesta sessió: **UNVERIFIED** que sigui el límit per defecte de WSL2.
- La variant que proposa per al Qwen3.6 35B A3B és Q3_K. La que s'ha mesurat de debò al B60 és IQ4_XS
  (16,50 GiB), de manera que la captura no mostra la mesura sinó la predicció d'aquell moment.
- La barra verda de tmux de baix és de la sessió d'`ifeval`, no forma part de Jaull.
- Conté el nom d'usuari del PC? No: només el maquinari. No porta dades de pacients ni credencials.
