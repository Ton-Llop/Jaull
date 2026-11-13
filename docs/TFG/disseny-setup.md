# Disseny: Setup (desplegar un servidor de model local amb Jaull)

Estat (2026-11-13): les tres parts implementades: la CLI (`jaull runtime install`, `jaull serve start/stop/status`)
i la pantalla Setup de la TUI. Vegeu «Estat de la implementació» al final.

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
  el PC de Biosfer; la configuració amb 192K compartits va funcionar. Fixat a 0,875 de la VRAM a
  `application/serving/policies.py`, marcat com a hipòtesi.
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
4. Sense endpoints que no calen (`/slots`, interfície web pròpia): el b11889 porta `--no-slots` i `--no-webui`
   (comprovat amb el build de Linux; a Windows, **PENDENT**).
5. Nivell de registre normal: mai `-lv 4` en servei, perquè escriuria els prompts.
6. TLS opcional: el b11889 el porta (`--ssl-cert-file`, `--ssl-key-file`); Jaull el passa amb `--tls-cert` i
   `--tls-key`. No s'ha provat amb un certificat real.

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

## Estat de la implementació (2026-11-13)

**Part 1, instal·lar llama.cpp:** `jaull runtime install --backend vulkan|cuda|cpu`. Comprovació física amb CPU al
portàtil; CUDA (RTX 2060) i Vulkan (B60) **PENDENT**.

**Part 2, servidor:** `jaull serve start/stop/status`, `jaull runtime install --latest` i
`scripts/llama_cpp_catalog.py`.

- Dimensionament (`application/serving/planner.py`): el KV compartit més gran que hi cap, amb cerca binària sobre
  l'estimador. Marge: `SERVE_GPU_MAX_USAGE_FRACTION = 0.875` de la VRAM que dona l'escaneig, marcat com a hipòtesi
  (22,1 GiB de 24 van penjar el PC de Biosfer; uns 20,8 GiB van funcionar). Si no hi cap, l'error diu el context
  més gran que hi cap. No se serveix un model que necessita descarregar capes a la RAM.
- Amb CPU, l'estimador no fa la col·locació (`hardware_fit` buit) i només dona l'avaluació; el servidor accepta
  `comfortable` i `compatible`. Es va trobar en la prova física: la primera versió refusava un model d'1,5B.
- Seguretat, els sis punts: escolta a `127.0.0.1`; amb `--lan`, regla de tallafocs generada i no aplicada; clau
  aleatòria en un fitxer 0600 passada amb `--api-key-file`, de manera que no surt a `ps` ni a l'script; `--no-slots`
  i `--no-webui`; cap `-lv`; HTTPS amb `--tls-cert` i `--tls-key` (el b11889 porta `--ssl-cert-file` i
  `--ssl-key-file`).
- Port per defecte 8090, perquè el 8080 és el de Keycloak a Biosfer. Un port ocupat es refusa i se'n proposa un de
  lliure.
- Versions més noves: des del novembre del 2026, `/releases/latest` de GitHub és una `v1.0.0` sense binaris i els
  builds `b<número>` surten com a prerelease. Per això es pren el `b<número>` més alt de `/releases`.

Prova física al portàtil (Linux, b11889 CPU, Qwen2.5-1.5B Q5_K_M, 2 usuaris de 8K):

| Comprovació | Resultat |
|---|---|
| Arrencada i `/health` | llest; el registre diu `n_slots = 2, n_ctx_slot = 8192, kv_unified = 'true'` |
| Petició sense clau | 401 |
| Petició amb clau | resposta correcta |
| `/slots`, interfície web | 501, 404 |
| On escolta | només `127.0.0.1:8090`; amb `--lan`, `0.0.0.0:8090` i la regla `ufw` amb la subxarxa |
| Clau a `ps` | no hi surt |
| Segona arrencada, port ocupat | refusades amb el missatge esperat |
| `stop`, `status` | aturat; `status` torna 1 |

**Pendents de la part 2:**
- Proves físiques a Windows: B60 + Vulkan (Biosfer) i RTX 2060 + CUDA. Amb Windows queden per veure l'arrencada
  desacoblada, l'script `.bat`, la regla `New-NetFirewallRule` i el permís del tallafocs que Windows demana la
  primera vegada.
- El context per usuari no es limita al context d'entrenament del model: amb SmolLM2 i `--context 2000000`, el
  missatge va proposar 131072.
- L'avís de revisió més nova del model i l'avís de llama.cpp a `doctor`: no s'han fet; l'avís de llama.cpp només surt
  a `serve start`.
- `plan_server` fa servir la sonda de dispositius del llama.cpp localitzat per triar `--device`; si hi ha builds de
  backends diferents instal·lats, la numeració podria no coincidir amb la del binari del servidor. No s'ha vist a la
  pràctica.

**Part 3, TUI:** opció «Set up a local server» a la pantalla principal, que obre `tui/screens/setup.py`: una sola
pantalla amb els cinc passos, en lloc de cinc pantalles. La feina lenta va en un fil i la pantalla només recull les
opcions; el text del pla i del resultat surt de `presentation/serving.py`, el mateix que escriu la CLI. Preparar
l'artefacte (resoldre, baixar si cal, verificar) és ara un sol mètode de la façana (`prepare_serve_artifact`) que fan
servir la CLI i la TUI.

- Model: (a) un dels baixats, amb la quantització llegida del nom del fitxer; (b) un repositori i una quantització.
  L'opció (c), obrir la cerca guiada i tornar al Setup amb el pla triat, **no s'ha fet**: la pantalla diu que es faci
  primer l'anàlisi guiada i que el model baixat apareixerà a la llista.
- Proves: sis proves amb un `AdvisorService` fals (arrencar i aturar a 80×24 i 160×48, repositori escrit i xarxa
  local, camps que falten, instal·lar llama.cpp, error de dimensionament, accés des del menú).
- Prova real al portàtil, sense pantalla: la pantalla amb l'`AdvisorService` real, backend CPU, SmolLM2 amb
  2 usuaris de 4K: arrenca, mostra el pla i l'adreça, i s'atura.
- Les proves de la TUI van trobar dos errors corregits: l'estat del servidor tapava la sortida de l'arrencada, on hi
  ha la clau que només es mostra una vegada; i la pantalla importava `jaull.huggingface`, que la regla d'arquitectura
  prohibeix a la TUI.
- Captura de pantalla de `docs/assets/tui-welcome.svg`: no s'ha regenerat; ara el menú té una opció més.
