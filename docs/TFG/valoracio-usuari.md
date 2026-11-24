# Valoració d'un usuari real de Biosfer

Notes del 2026-11-23 sobre una persona de Biosfer que fa servir un servidor engegat amb Jaull des
del 2026-11-18. Les dades vénen de l'autor, que va estar present al principi; res d'això s'ha mesurat
amb eines. No inclou noms ni credencials.

Cada punt diu què és una **observació de l'autor** i què és un **judici** seu. El que falta porta
l'etiqueta **PENDENT**.

## Què es va fer

- Una persona de Biosfer.
- Des del 2026-11-18. En el moment d'escriure això, cinc dies.
- B60 amb Windows natiu, llama.cpp b11889 amb Vulkan, Qwen3.6-35B-A3B (`UD-IQ4_XS`), servidor engegat
  amb Jaull (`Set up a local server`).
- El client és opencode, dins del WSL2 del mateix usuari.
- El servidor escolta a tota la LAN. No és per compartir-lo: WSL2 té una xarxa pròpia (NAT) i no arriba al
  `127.0.0.1` de Windows, així que l'adreça d'escolta ha de ser una altra (vegeu «Conseqüències»).

## Què va passar

- **Observació de l'autor:** l'autor va ajudar al principi. El pas que ho va necessitar va ser el certificat
  HTTPS: Jaull el fa servir per defecte quan es comparteix amb `--lan`, i fer que opencode se'n refiï va costar.
- **Decisió:** es va treure HTTPS. La clau d'API i el que escriu l'usuari viatgen ara sense xifrar per la LAN
  de l'edifici. Per ara l'usa una sola persona.
- **Incidències, segons l'usuari:** cap en cinc dies, ni de RAM ni de lentitud ni d'una altra mena. **No s'ha
  mesurat**: és el que ha explicat.
- **Judici de l'autor (UNVERIFIED):** l'usuari ho hauria pogut fer sol. No es va provar.

## Què es pot dir i què no

Es pot dir:

- Un model recomendat per Jaull, servit per Jaull, s'usa en un escenari real de l'empresa (OE11) amb el
  maquinari i el sistema que el projecte recomana (Windows natiu, no WSL2).
- Dona un primer suport a OE10 (aplicar Jaull a Biosfer).
- El pas difícil de la instal·lació va ser el certificat, no el model, la memòria ni la comanda.

No es pot dir:

- Que la interfície serveixi a un usuari no expert (OE8): hi va haver ajuda en un pas. Aquesta prova no ho
  tanca.
- Que el servidor aguanti més d'una persona, ni durant més de cinc dies. La concurrència és la de
  `concurrencia-b60.md`, mesurada a part.
- Que la configuració sigui segura per a una LAN compartida: sense HTTPS la clau d'API pot llegir-se a la xarxa.
  La clau segueix sent obligatòria, i la regla del tallafocs no s'ha comprovat en aquest PC (**PENDENT**).

## Conseqüències per a Jaull

- **Mode «aquest PC i WSL»:** el cas Windows amb servidor i WSL2 amb client és el d'aquesta prova. Obliga a
  escoltar a `0.0.0.0` i a obrir el port a més xarxa de la necessària. El disseny d'un mode que limiti la regla
  del tallafocs a la xarxa interna de WSL2 existeix i no està construït (vegeu `STATUS.md`).
- **El certificat:** quan es comparteix, HTTPS és per defecte i un certificat autosignat obliga cada client a
  fiar-se'n a mà. A WSL2 cal passar la ruta `/mnt/c/...`. Va ser el pas que va fer fallar la instal·lació sola.
- Les dues coses van al capítol 15 (discussió) i al 16 (treball futur).

## Per completar

- **PENDENT:** una conversa curta amb l'usuari, amb preguntes fixes i per escrit (què li va costar, què
  canviaria, si el recomanaria).
- **PENDENT:** mirar la regla del tallafocs del port 8090 en aquest PC.
- **PENDENT:** repetir-ho amb una segona persona que ho instal·li sense l'autor al costat (valoració d'11.9).
