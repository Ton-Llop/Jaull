# Esquelet: capítol 11 (aplicació a Biosfer) i conclusions

Guió per redactar, no text final. Cada punt diu què s'hi ha d'explicar, amb quina dada i on és. Les xifres són les
dels documents citats; si una no hi és, porta **PENDENT**. Estructura dels apartats: la de l'esborrany complet de la
memòria.

## Capítol 11. Aplicació de Jaull a casos d'ús de Biosfer

### 11.1 Definició dels escenaris
- Pregunta de partida de l'empresa: quin maquinari i quins models per tenir IA local per a les seves tasques.
- Quatre famílies de càrrega proposades (`docs/experimental-protocol-biosfer.md`, «Biosfer workloads»).
- Confirmada només la programació: un agent de codi sobre Lipopótamo. **PENDENT**: resposta de Biosfer sobre les
  altres tres.

### 11.2 Definició de requisits
- Requisits de Jaull per a l'escenari: xat/codi, prioritat, context, concurrència 1, ús comercial.
- Restricció dura: el codi i les dades no surten del PC (privacitat, cap. 13 i 14).

### 11.3 Caracterització de l'equip
- B60 de 24 GiB, Ryzen 7 5700G, 27,9 GB de RAM, WSL2 limitat a 13 GB, monitor a la GPU integrada
  (`docs/TFG/entorno-biosfer.md`, memòria 2.2).
- **Troballa principal:** el mateix llama.cpp b11889 genera entre 4,6 i 18 vegades més lent a WSL2 (Mesa Dozen) que a
  Windows natiu, i el Qwen3.8-27B peta a WSL2 (`docs/b60-comparativa-wsl2-windows.md`).
- Conseqüència per a l'empresa: el servidor ha de córrer a Windows natiu.

### 11.4 Generació de recomanacions
- Primera cerca real de Jaull a la B60 (captura `docs/TFG/capturas/jaull-tui-results-biosfer-b60-2026-11-10.png`).
- Errors trobats en aplicar-lo (cicle de la metodologia): el Qwen3.6 no apareixia (D124), el KV dels models híbrids es
  sobreestimava ×4 (D125), una etiqueta `gguf` confonia la preselecció (D127).
- Barrat de contextos: quin context màxim hi cap (D128).

### 11.5 Validació de les configuracions
- Memòria: KV del Qwen3.6-35B-A3B predit i mesurat iguals a 32K, 128K i 256K (640, 2.560 i 5.120 MiB); Qwen3.8-27B
  867 MiB predits contra 854 mesurats (D125). El Coder 30B, que abans Jaull donava per no apte, carrega amb 2,9 GiB
  lliures (D120, D123).
- Velocitat: **PENDENT** de triar quines xifres de `llama-bench` es posen (D121, `docs/b60-taula-comparativa.md`).

### 11.6 Avaluació sobre tasques representatives
- **Qualitat general (IFEval, `docs/ifeval-b60-prova.md`):** Qwen3.6-35B-A3B 0,824, Qwen3.8-27B sense raonament
  0,821, Qwen2.5 32B 0,797 i Coder 30B 0,789. Els dos primers empaten; el Qwen3.6 respon unes 4 vegades més ràpid
  (5,2 s per prompt contra 20,6 s).
- **Estudi de cas, Prova 3 (`docs/TFG/agentic-pruebas-biosfer.md`):** funcionalitat nova a Lipopótamo que compila i
  funciona en el cas bàsic, 2 o 3 correccions, cinc defectes al codi, més lent que un desenvolupador.
- **Mesura repetible, mini-benchmark (`docs/TFG/mini-benchmark-agentic.md`):** 6 de 6 pels tests, 3 de 6 amb el
  criteri complet del projecte; 79 minuts sense intervencions. **PENDENT**: la ronda del núvol.
- Missatge del capítol: els tests no bastaven per jutjar el codi de l'agent; la revisió humana segueix sent necessària.

### 11.7 Comparació d'alternatives
- Taula: els quatre models (qualitat IFEval, velocitat, memòria, context).
- Local contra núvol: **PENDENT** (mini-benchmark de demà). Si la ronda es fa amb Codex, dir que l'eina és diferent.

### 11.8 Proposta de configuració
- B60 + Windows natiu + llama.cpp b11889 Vulkan + Qwen3.6-35B-A3B UD-IQ4_XS.
- Context recomanat: **PENDENT** (barrat de contextos a la B60; a la Prova 3 es va fer servir 131.072).
- Ús recomanat: canvis de codi acotats amb revisió humana; no delegar-hi feina sense supervisió.
- Com instal·lar-ho: «Desplegar» i la guia (**PENDENT**, disseny després del benchmark).

### 11.9 Valoració amb l'empresa
- **PENDENT**: sessió de 15–20 minuts amb algú de Biosfer que instal·li o faci servir el desplegament sense l'autor, i
  la seva opinió. És la validació de l'OE10.

## Capítol 15. Discussió (punts a recollir)
- Prediccions: on encerta (KV, pesos GGUF) i on és conservador o desconegut (overhead del runtime més enllà de 4.096
  tokens, sense validar amb `-lv 4`).
- Recomanador: sense GPU, Balanced no veu la velocitat fins que l'usuari la mesura (portàtil sense GPU).
- Experimental: una màquina, una tirada per model, cap variància mesurada.
- Agents: els tests visibles fan la tasca més fàcil; possible contaminació dels models del núvol.
- Treball futur: encaminador local + núvol, desplegament per a més backends, corbes de concurrència.

## Capítol 16. Conclusions

### 16.1 Assoliment dels objectius

| Objectiu | Estat | Evidència principal |
|---|---|---|
| OE1 Detectar el maquinari | Assolit | RTX 2060, RTX 4060, portàtil sense GPU, B60 (Windows i WSL2) |
| OE2 Descobrir models sense baixar pesos | Assolit | Metadades i capçaleres GGUF per rang |
| OE3 Estimar memòria i viabilitat | Assolit, amb límits | KV exacte a la B60; overhead extrapolat més enllà de 4.096 tokens |
| OE4 Seleccionar segons requisits | Assolit, amb límits | D124 corregit; Balanced sense velocitat en CPU |
| OE5 Plans d'execució concrets | Assolit | Comandes de llama.cpp amb dispositiu Vulkan identificat |
| OE6 Executar i contrastar prediccions | Assolit | Registres immutables i comparacions (cap. 10) |
| OE7 Mesurar velocitat i qualitat | Assolit | Protocols de velocitat; IFEval de quatre models |
| OE8 CLI i TUI per a no experts | Parcial | Interfície guiada; sense prova amb usuaris (**PENDENT** 11.9) |
| OE9 Diferents configuracions de maquinari | Assolit | Quatre màquines, dos sistemes a la B60 |
| OE10 Aplicar Jaull a Biosfer | Parcial | Cap. 11; falta la valoració de l'empresa i el desplegament |
| OE11 Fer servir un model recomanat en un escenari real | Assolit | Prova 3 i mini-benchmark |

### 16.2 Principals aportacions (candidates)
- Separar predicció, observació i comparació, i que només una mesura comparable pugui reordenar recomanacions.
- Model de memòria per a atenció híbrida, validat a la B60.
- La diferència WSL2/Windows a la B60, amb mesures.
- Un banc de proves agèntic repetible construït amb l'historial del mateix projecte.

### 16.3 Aplicabilitat
- Per a Biosfer: la proposta de configuració (11.8) i el desplegament.
- Per a altres: qualsevol màquina amb Linux/Windows i llama.cpp.

### 16.4 Aportació a la formació personal
- Material ja escrit a Objectius Formatius (v9, impersonal): teoria contra màquina, no fiar-se d'una mesura,
  preparar sessions, registre de decisions.

### 16.5 Valoració final
- **PENDENT**: la valoració de l'autor.
