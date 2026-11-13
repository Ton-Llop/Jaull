# Disseny: Setup (desplegar un servidor de model local amb Jaull)

Estat: **proposta**, pendent de revisió de l'autor (2026-11-13). No hi ha codi.

## Per què

Jaull recomana, prepara i mesura, però deixar un model funcionant com a servei encara és feina manual: compilar o
baixar llama.cpp amb el backend correcte, triar `-ngl`, `-np`, `-c`, obrir-lo a la xarxa sense exposar-lo de més. A
Biosfer això ho ha fet l'autor a mà, i quan marxi ningú no ho sabrà repetir. Setup ho converteix en un assistent per a
algú no expert, aprofitant el que Jaull ja sap del maquinari i de la memòria.

## Abast

Windows i Linux, x64, amb Vulkan, CUDA o CPU. Validació física prevista: B60 amb Vulkan a Windows (Biosfer), RTX 2060
amb CUDA i el portàtil amb CPU. La resta de combinacions del catàleg queden «provades amb simulacions».

Fora d'aquesta versió: arrencada automàtica amb el PC (tasca programada o servei de systemd); TLS si el build de
llama.cpp no el porta; el límit de la RAM experimental (incidència del 2026-11-13, vegeu «Pendents relacionats»).

## Flux a la TUI

Pantalla principal (`welcome.py`) → nova opció **Setup**:

1. **Maquinari i backend.** Detecció habitual; Jaull proposa el backend amb `backend_selection` i deixa triar-ne un
   altre si n'hi ha més d'un de possible.
2. **llama.cpp.** Si el localitzador ja en troba un de vàlid per a aquest backend, es fa servir. Si no, es baixa el
   binari oficial del build fixat, se'n comprova el SHA-256 i es registra al `RuntimeRegistry`.
3. **Servidor.** «Només aquest PC» o «compartir a la xarxa local», i quants usuaris alhora. Jaull proposa els slots i
   el context que hi caben amb marge.
4. **Model**, de tres maneres: (a) un dels ja baixats (pantalla Storage); (b) escriure un repositori i una
   quantització, que es baixa i es verifica; (c) «no sé quin»: obre la cerca guiada i el pla triat torna al Setup.
5. **Resum i arrencar.** URL, clau d'API, ruta de l'script d'arrencada i com aturar-lo.

La CLI fa el mateix sense pantalles: `jaull runtime install`, `jaull serve start`, `jaull serve stop`,
`jaull serve status`.

## Peces

| Peça | Mòdul (proposat) | Responsabilitat | Depèn de |
|---|---|---|---|
| Catàleg de binaris | `runtime/llama_cpp_release.py` + constants a `runtime/policies.py` | Build fixat (b11889) → nom del fitxer, SHA-256 i, a CUDA, el paquet `cudart`, per sistema, arquitectura i backend | Res |
| Instal·lador | `runtime/llama_cpp_install.py` | Baixar per HTTPS, verificar el hash abans de descomprimir, descomprimir a `user_data_dir("runtimes", "llama.cpp", "<build>-<backend>")`, registrar | Catàleg, `RuntimeRegistry` |
| Planificador del servidor | `application/serving/planner.py` | Triar `-np`, `-c`, `--kv-unified`, `--kv-unified-per-slot`, `--host`, `--port` a partir de l'estimació de memòria i d'un marge de política; reaprofita `-ngl` i `-dev` del planificador actual | Estimador, `application/execution/planner.py` |
| Gestor del procés | `execution/server_process.py` | Arrencar `llama-server` desacoblat, desar PID i configuració, esperar `/health`, aturar, estat | Res de Jaull |
| Script d'arrencada | dins del gestor | Escriure `start-server.bat` / `.sh` amb la mateixa ordre | Planificador |
| Pantalles | `tui/screens/setup.py` | L'assistent de cinc passos | `AdvisorService` |
| Façana | `AdvisorService` | Només compondre: cap política nova | Tot l'anterior |

Regles del projecte que s'apliquen: el `HardwareFitAnalyzer` no aprèn flags de llama.cpp (els flags del servidor viuen
al planificador del servidor, a la capa d'aplicació/runtime); cap número màgic (marges, port per defecte, build i hashes
a `policies.py`); models Pydantic congelats; cap `except Exception`.

## Com es dimensiona el servidor

Entrada: l'estimació de memòria del model triat i el nombre d'usuaris demanat.

- KV per usuari: el de l'estimador amb `concurrent_users = N`, validat a la B60 per a 1, 2 i 4 usuaris
  (`docs/TFG/concurrencia-b60.md`).
- Amb més d'un usuari, KV compartit (`--kv-unified`) amb un context total `-c` i un màxim per usuari
  (`--kv-unified-per-slot`), com a la configuració proposada per a Biosfer.
- Marge: el sostre de la GPU menys un marge de política. El cas de `-np 8` amb 32K fixos (uns 22,1 GiB de 24) va penjar
  el PC de Biosfer; la configuració amb 192K compartits va funcionar. **PENDENT** fixar el valor del marge a
  `policies.py` a partir d'aquestes dues dades, marcat com a hipòtesi.
- Si el que demana l'usuari no hi cap, el Setup ho diu i proposa menys usuaris o menys context; no arrenca una
  configuració que l'estimador dona per inviable.

## Versions més noves

Petició de l'autor (2026-11-13): el Setup ha d'avisar si hi ha una versió més nova que la que fa servir, de llama.cpp
o del que sigui (Jaull mateix, el model).

- llama.cpp: consultar l'última release de `ggml-org/llama.cpp` a GitHub i comparar-la amb el build fixat. Si n'hi ha
  una de més nova, **avisar**, no instal·lar-la sola: el build fixat és el validat a la B60, i un build nou s'hauria de
  provar abans. L'usuari pot triar instal·lar-la explícitament (el catàleg de hashes es llegeix llavors de la release
  nova, marcat com a no validat).
- Model: si el repositori té una revisió més nova que la baixada, avisar.
- Sense xarxa, l'avís simplement no surt; mai bloqueja el Setup.
- llama.cpp publica builds cada poques hores, així que un avís per qualsevol versió nova sortiria sempre. Acordat
  (2026-11-13):
  - Avisar només si el build fixat té més d'un llindar d'antiguitat respecte a l'últim (proposta: 30 dies), constant
    a `policies.py` marcada com a decisió de producte, no mesurada. L'avís és informatiu (a `doctor` i al resum del
    Setup).
  - Consultar l'API de GitHub com a molt un cop al dia, amb la resposta en memòria cau; el límit sense autenticar és
    d'unes 60 consultes per hora.
  - «Instal·lar l'última» és una opció explícita: el SHA-256 surt de la mateixa API de GitHub (el publica per a cada
    fitxer), la baixada continua verificada, i la instal·lació queda marcada com a no validada.
  - Moure el build fixat és una decisió de l'autor després de provar-lo; un script regenera la taula del catàleg
    (fitxers i hashes) a partir d'una versió donada.
- Va al pla de la part 2 o posterior; la part 1 només instal·la el build fixat.

## Seguretat per defecte

1. El servidor escolta a `127.0.0.1`. Compartir a la xarxa és una opció explícita.
2. En compartir, Jaull **genera** la regla de tallafocs per a la subxarxa interna (Windows: `New-NetFirewallRule`;
   Linux: l'ordre equivalent) i la mostra per a l'administrador. No l'aplica sense confirmació explícita.
3. Clau `--api-key` aleatòria, desada al directori de dades de Jaull i mostrada una vegada.
4. Sense endpoints que no calen (`/slots`, interfície web pròpia) si el build ho permet. **PENDENT** comprovar els
   flags exactes del b11889 a Windows i Linux.
5. Nivell de registre normal: mai `-lv 4` en servei, perquè escriuria els prompts.
6. TLS opcional si el build el porta; si no, es documenta com a pendent.

## Errors

- Hash que no coincideix: s'esborra la baixada i no s'instal·la res.
- Port ocupat (el 8080 de Keycloak a Biosfer): el Setup ho detecta i proposa un altre port.
- `/health` que no respon en el temps de política: s'atura el procés i es mostren les últimes línies del registre.
- Processos orfes: `serve status` comprova que el PID desat és de veritat `llama-server` abans d'aturar-lo.

## Proves

- Unitàries: catàleg (cada combinació té fitxer i hash), instal·lador amb un arxiu fals (hash bo i dolent,
  descompressió, registre), planificador (usuaris i context que hi caben, cas que no hi cap, KV compartit), gestor del
  procés amb un executable fals que respon a `/health`, seguretat (localhost per defecte, clau generada, regla
  generada però no aplicada).
- Arquitectura: la prova d'importacions existent ha de seguir passant (res de `hardware_fit` a `runtime/`).
- Físiques: B60 + Vulkan + Windows, RTX 2060 + CUDA, portàtil + CPU; en cada una, instal·lar, arrencar, una petició
  amb `load.py` i aturar.

## Memòria del TFG

Quan existeixi la funcionalitat: un cas d'ús nou a la figura de casos d'ús d'execució i evidència, una fila a la taula
de casos d'ús amb el seu guió, un diagrama de seqüència del Setup, i el capítol 11 (proposta de configuració i
valoració amb l'empresa).

## Pendents relacionats

- **RAM experimental (2026-11-13):** a CPU, un pla passa a «Experimental RAM pressure» si el consum cap a la RAM total,
  sense deixar res per al sistema operatiu (`estimator/compatibility.py`, `hardware_fit.py`). Al portàtil (15,35 GiB)
  apareixia un model de 14,42 GiB. Cal una prova al portàtil abans de fixar un límit.
