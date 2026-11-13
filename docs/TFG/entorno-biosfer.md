# Entorn de Biosfer: com arrencar Lipopótamo i veure'l des del portàtil

Notes del 2026-11-10, preses mentre es provava a mà la funcionalitat que va fer l'agent
(Qwen3.6-35B-A3B, vegeu `agentic-pruebas-biosfer.md`). No inclouen codi de Biosfer ni credencials.

Cada punt diu què es va **veure** i què és una **hipòtesi**. El que no s'ha tornat a comprovar
després del canvi porta l'etiqueta **PENDENT**.

## Màquines

- **monk-e**: el PC de Biosfer (Windows amb WSL2 i Docker Desktop). Usuari `tonllop`, SSH pel port 2222.
  El repositori és a `/home/tonllop/projects/lipopotamo`.
- **Portàtil**: Ubuntu sobre WSL2 en un Windows. Usuari `antoni`. El navegador és el de Windows.

La confusió més repetida del dia: llançar una comanda al lloc equivocat. Les ordres `ssh -L` van **només
al portàtil**. El prompt `tonllop@monk-e` vol dir que ets dins del PC, encara que hi hagis entrat per SSH.

## Ports

| Port a monk-e | Què és | Notes |
|---|---|---|
| 5173 | frontend (nginx dins de Docker, port 80 del contenidor) | el navegador entra pel 5174, vegeu avall |
| 8000 | backend (FastAPI) | |
| 8080 | Keycloak de Lipopótamo | `KEYCLOAK_PORT=8080` al `.env` |
| 8180, 8280, 5433, 5454, 3310, 3900 | contenidors `tortuga_*` (altre projecte) | no es poden aturar, no s'han tocat |
| 8082 (Windows) i 8081 (WSL) | `llama-server` del model i el pont `ssh -R` fins a WSL | vegeu el punt 3 de les trampes |

## Túnel des del portàtil

```bash
ssh -p 2222 -L 5174:127.0.0.1:5173 -L 8080:127.0.0.1:8080 -L 8000:127.0.0.1:8000 tonllop@192.168.1.107
```

I s'obre `http://localhost:5174`. S'entra pel **5174 i no pel 5173** perquè el client Keycloak
`lipopotamo-frontend` accepta aquest origen. Amb el 5173 el navegador bloqueja la resposta del token:
`falta la cabecera CORS 'Access-Control-Allow-Origin'` (estat 200). El backend també admet el 5174 a
`API_CORS_ORIGINS`. L'alternativa és afegir `http://localhost:5173` a *Web origins* del client, a la consola
de Keycloak; no s'ha fet perquè canvia la configuració del Keycloak de Biosfer.

## Trampes trobades, en ordre

### 1. Un `ssh -L` llançat des de monk-e

Es va llançar el túnel dins del PC. Allà el 8180 ja és de `tortuga_keycloak` (`bind: Address already in use`),
però el 5173 es va quedar agafat per aquell `ssh` en segon pla. Docker Desktop no va poder publicar el 5173
a WSL i, quan el procés es va matar, el reenviament no es va reobrir sol. Solució: `kill` del `ssh` i
`docker restart lipopotamo_frontend`. Es comprova amb `ss -ltnp | grep :5173` a monk-e.

### 2. Una regla `portproxy` antiga al Windows del portàtil

El 5173 del Windows del portàtil el tenia un `svchost.exe` (servei `iphlpsvc`) amb dues regles
`netsh interface portproxy`, 3000 i 5173, cap a `172.17.6.33`, una IP de WSL caducada. El reenviament de
WSL no podia obrir `127.0.0.1:5173` i el navegador no arribava al túnel. Es va esborrar la regla del 5173 amb
`netsh interface portproxy delete v4tov4 listenport=5173 listenaddress=0.0.0.0` (PowerShell com a
administrador). La del 3000 es va deixar.

Comprovar des del Windows, no des de WSL: `curl.exe -sI http://127.0.0.1:5174` amb la ruta completa
(`/mnt/c/Windows/System32/curl.exe`, des de `cd /mnt/c`).

### 3. El 8080 estava ocupat pel servidor del model

El `llama-server.exe` del 35B (PID 8712, `--port 8080`) ocupava el port que Keycloak vol. `make up-dev`
fallava amb `ports are not available: exposing port TCP 127.0.0.1:8080`. Es va aturar el servidor amb
`taskkill`. **El servidor del model ha de viure en un altre port** (8082) mentre l'aplicació estigui aixecada.

El camí fins al model no és un sol túnel, són dos, i la meitat del dia es va perdre per això:

1. `llama-server.exe` escolta a `127.0.0.1:8082` de **Windows**.
2. Des de WSL **no es veu** el `127.0.0.1` de Windows (comprovat: `curl` des de WSL a `127.0.0.1:8082` dona
   `Connection refused` i, amb `curl.exe` de Windows, `{"status":"ok"}`).
3. Un `ssh.exe -N -R 8081:127.0.0.1:8082 -p 2222 tonllop@192.168.1.107` llançat des de Windows (a monk-e,
   dins d'un panell de tmux) obre el **8081 dins de WSL** i el porta al 8082 de Windows. Aquest era el pont
   que feia servir OpenCode (`baseURL` a `127.0.0.1:8081`), amb el destí escrit a mà (`...:8080`).
   Quan Keycloak va ocupar el 8080, el pont va passar a parlar amb Keycloak (`404` a `/v1/models`).
4. `lm_eval` (IFEval) i OpenCode corren dins de monk-e i fan servir directament el `127.0.0.1:8081` de WSL
   (vegeu `ifeval-b60-prova.md`): no necessiten cap túnel des del portàtil. Si algun dia es vol cridar el
   model des del portàtil, el túnel ha d'anar al 8081 **de WSL**, no al 8082: `-L 8081:127.0.0.1:8081`.
   Aquesta última línia no s'ha provat.

Canviar el port del servidor obliga a relançar el pont `-R` (`kill` del `ssh.exe` i tornar-lo a obrir).

### 4. Rutes i usuari del `.env` són els del portàtil

El `.env` porta `PROJECT_ROOT=/home/antoni/projects/lipopotamo`, i probablement `PROJECT_SOURCE_DIR` igual.
A monk-e no existeix. Efectes vistos:

- `juggler` i `acrobat` en bucle de reinici: `No module named 'juggler'` / `'acrobat'`. El muntatge
  tenia com a origen `/home/antoni/projects/lipopotamo` (una carpeta buida) i com a destí la ruta bona.
- El compose fa servir `user: "${UID:-1000}:${GID:-1000}"`. A monk-e `tonllop` és l'uid **1002**, no
  l'1000, i el backend no podia escriure a `shared_folders/drag-drop` (`PermissionError`), de manera que
  `POST /uploads/drop` donava **500**. El missatge CORS que veia el navegador era un efecte secundari: les
  respostes d'error del backend surten sense capçalera CORS.

Correcció, sense tocar el `.env` (les variables de la shell tenen prioritat):

```bash
cd ~/projects/lipopotamo
env UID="$(id -u)" GID="$(id -g)" PROJECT_ROOT="$(pwd)" PROJECT_SOURCE_DIR="$(pwd)" \
  docker compose -f docker-compose.yaml --env-file .env up -d --force-recreate backend juggler acrobat
```

S'usa `env` perquè `UID` és una variable només de lectura a bash. Amb això el backend ja podia escriure a
`shared_folders`, però quedava una tercera ruta (vegeu el punt següent).

### 5. El volum `acrobat_unzip_tmp` era de `root`

Després de corregir l'uid, la pujada seguia fallant, ara a `acrobat`: `PermissionError: [Errno 13]
Permission denied: '/tmp/lipopotamo/acrobat/unzip/acrobat_event_...'`. El volum amb nom
`lipopotamo_acrobat_unzip_tmp` es va crear el 5 de novembre com a `root` (`drwxr-xr-x root root`, buit) i
l'uid 1002 no hi pot escriure. Es va comprovar muntant el volum en un contenidor temporal amb la mateixa
imatge:

```bash
docker run --rm --user root --entrypoint sh -v lipopotamo_acrobat_unzip_tmp:/v lipopotamo-acrobat:latest \
  -c "chown -R $(id -u):$(id -g) /v"
docker restart lipopotamo_acrobat
```

El volum era buit, per això es va canviar el propietari en lloc d'esborrar-lo. Segons l'usuari, després d'això la
pujada de ZIPs va funcionar. No s'ha mesurat res més que això: no es va comprovar que el lot arribés a publicar-se
ni a executar-se.

### 6. MongoDB respon buit fins que es reinicia (2026-11-13)

Tres dies després, el frontend no mostrava cap lot: el backend responia `200` amb `outputs.state rows=0` i, consultant
MongoDB directament, totes les col·leccions de la base `lipopotamo` tenien 0 documents. Els fitxers de
`shared_folders/drag-drop/` i `shared_folders/outputs/` seguien al disc, i els logs d'`acrobat` del dia 10 deien que els
lots s'havien desat a MongoDB. Les dades són en un muntatge `bind` (`.docker-data/mongo/db`), no en un volum anònim.
Un `docker restart lipopotamo_mongodb` les va tornar a mostrar. **VALIDATED** que el reinici ho resol; **UNVERIFIED**
la causa. Segons l'usuari, ja li havia passat abans i també es resolia així.

Aquell dia `juggler` i `acrobat` no corrien (no sortien a `docker ps`) i es van haver de tornar a aixecar amb les
variables de la shell del punt 4.

Abans de reiniciar res, no esborrar mai `.docker-data/mongo` ni fer `docker compose down -v`: les dades són al disc
i un reinici del contenidor no les toca.

### 7. Altres coses curtes

- El backend va quedar en estat `Created` després d'un `--force-recreate` i va caldre `docker start
  lipopotamo_backend`. Docker no va avisar.
- El directori `keycloak/import` és buit, així que el realm `lipopotamo` no ve del repositori: viu a la
  base de dades de Keycloak, dins del volum de Docker.
- `juggler` i `acrobat` no calen per provar el panell de comparació. `acrobat` sí cal per pujar ZIPs a
  **Uploads**.

## Del que hi ha al repositori no es va tocar res

L'agent no va fer cap commit (`HEAD` continua a `86b98a3`, el mateix que `origin/master`). Tots els canvis
d'entorn d'aquest document són variables de la shell, una regla de Windows i un client de Keycloak
(aquest últim sense modificar). Cap fitxer del repositori de Biosfer s'ha editat per fer l'entorn
funcionar.
