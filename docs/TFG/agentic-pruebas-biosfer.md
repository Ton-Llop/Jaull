# Proves agentic amb Lipopótamo (PC de Biosfer)

**Aquest fitxer descriu per dins una part de Lipopótamo (rutes de fitxers, funcions, comportament). No es publica
ni es puja a un repositori obert sense el permís de Biosfer.**

Registre de les execucions de l'agent sobre el codi de Biosfer. L'entorn i el protocol són a
[`DECISIONS-2.md`](DECISIONS-2.md): X001 (OpenCode), X002 (el Coder 30B), X003 (protocol) i X004 (la RAM del PC).
Els estats són els de sempre: **VALIDATED** (comprovat), **UNVERIFIED** (hipòtesi o relat de l'autor) i **PENDENT**.

## Prova 1: comparació de resultats a Multi Select (2026-11-05)

**Configuració.**
- Model: Qwen3-Coder 30B-A3B Q4_K_M, a través d'OpenCode.
- Servidor: `llama-server.exe` de Windows amb `-c 32768 -np 1 --jinja --cache-ram 0 -lm dio
  --sleep-idle-seconds 45`, al port 8090. OpenCode hi arriba pel port 8081 de WSL, amb un túnel invers.
- Codi: clon local de Lipopótamo a `~/projects/lipopotamo`, branca `Qwen3-30B-A3B`. L'agent no havia de fer commits.
- Sense el vigilant de RAM de X004. Només un monitor que llegeix.

**Incidència d'entorn abans de la prova.** El primer enviament va fallar en 218 ms amb «Unable to find matching target
resource method». El port 8080 de Windows i de WSL el tenia el Keycloak de la pila de Docker de Lipopótamo, i el túnel
portava OpenCode cap allà en lloc de cap al model. S'ha resolt movent el model al port 8090 i tornant a llançar el túnel
cap a aquest port. **VALIDATED** (la resposta del Keycloak es va veure amb `curl`).

**Tasca (prompt tal com es va enviar).** Només s'han tornat a ajuntar les línies que el terminal havia partit.

```
OBJETIVO

En Outputs Dashboard existe un modo Multi Select para seleccionar varios batches. Queremos que el usuario pueda comparar sus resultados comunes. Cuando haya varios batches seleccionados y el usuario abra la visualización de resultados, debe aparecer una vista comparativa en lugar de la vista de checks/muestras erróneas de un único batch.

La experiencia habitual con un solo batch debe seguir funcionando como antes.

CONTEXTO DEL ISSUE

El usuario selecciona varios batches y quiere comparar variables y distribuciones entre conjuntos. La imagen de referencia muestra, para una variable como VLDL-C:

- Una distribución de los valores de los distintos batches, distinguidos por color.
- Un gráfico de correlación entre dos conjuntos.
- Boxplots que permiten comparar sus distribuciones.

La imagen es una referencia funcional, no exige copiar exactamente su composición.

PUNTOS DE ENTRADA DEL CÓDIGO

Antes de implementar, revisa únicamente lo necesario a partir de estos archivos:

- frontend/src/components/dashboard/outputs/OutputsDashboard.tsx
  Gestiona Multi Select, selectedOutputIds, la selección de batches y la carga de resultados y distribuciones de un batch.

- frontend/src/components/dashboard/outputs/OutputsSidePanel.tsx
  Gestiona el panel lateral y actualmente muestra que los checks están deshabilitados durante Multi Select.

- frontend/src/components/dashboard/outputs/OutputsDetailPanel.tsx
  Contiene la vista de resultados y las visualizaciones de distribuciones existentes. Revisa si sus componentes y estilos se pueden reutilizar.

- frontend/src/services/outputs.ts
  Contiene fetchLiposcaleResults, fetchSoftwareDistributions y los tipos de datos usados por el frontend.

- backend/api/outputs_results.py
  Contiene los endpoints de resultados y distribuciones. Estos leen vistas materializadas de results_views; revisa sus respuestas y parámetros antes de decidir si necesitas cambios de backend.

- frontend/src/components/dashboard/outputs/types.ts y constants.ts
  Contienen tipos y constantes compartidos.

También puedes consultar frontend/package.json para identificar la librería de gráficos y los comandos de verificación.

No recorras repetidamente todo el repositorio: sigue las referencias concretas de estos archivos cuando necesites entender un contrato.

COMPORTAMIENTO REQUERIDO

1. En modo Multi Select, permite seleccionar al menos dos batches y acceder a una acción clara de comparación de resultados.

2. Conserva el comportamiento existente cuando se selecciona un solo batch o se desactiva Multi Select. La comparación no debe interferir con checks, validaciones, lineage ni la navegación habitual.

3. Carga los resultados reales de cada batch seleccionado usando las APIs existentes siempre que sea posible. Respeta el software y el modo de análisis aplicables. Evita modificar la generación de resultados, el checker o los documentos de MongoDB.

4. Identifica las variables numéricas comunes entre los batches que realmente tengan datos disponibles. Ofrece un selector para cambiar la variable comparada. No presentes variables incompatibles como si fueran equivalentes; respeta nombres y unidades cuando estén disponibles.

5. Para la variable seleccionada, muestra como mínimo:

   - Distribución de los valores por batch.
   - Un boxplot por batch, con colores y leyenda que permitan identificar cada conjunto.
   - Información básica por batch, como el número de valores válidos.

6. Correlación: solo dibuja un scatter entre dos batches cuando exista una forma fiable de emparejar sus muestras mediante identificadores compartidos. Nunca emparejes filas por posición ni inventes correspondencias.

   Si hay dos batches y no hay muestras emparejables, muestra un estado claro que lo explique. Si hay más de dos batches, permite elegir qué dos comparar para este gráfico. La distribución y los boxplots deben seguir funcionando aunque no haya pares.

7. Contempla estados de carga, error y ausencia de datos: menos de dos batches, un batch sin resultados, ninguna variable común, valores no numéricos y ausencia de muestras comunes. No incluyas valores nulos o no finitos en los cálculos.

8. Al cambiar los batches o la variable, actualiza correctamente los gráficos y evita mostrar datos obsoletos. Mantén un rendimiento razonable con batches grandes.

9. Sigue el sistema visual existente de Lipopotamo y comprueba modo claro y oscuro. La interfaz debe seguir siendo utilizable en pantallas pequeñas. Los colores deben ser distinguibles y acompañarse de nombres, no depender únicamente del color.

10. Si el contrato de las APIs existentes no permite obtener los datos necesarios, explica en el código de forma mínima el motivo y añade la solución más pequeña y compatible. No crees una segunda fuente de verdad ni alteres resultados científicos.

CRITERIOS DE ACEPTACIÓN

- Seleccionar dos batches con resultados comunes muestra una comparación basada en sus datos reales.
- Cambiar la variable actualiza distribución y boxplots.
- La correlación usa únicamente muestras emparejadas por identidad fiable y avisa cuando no es posible.
- La selección múltiple de más de dos batches funciona sin perder la identificación de cada conjunto.
- Los batches sin resultados y la falta de variables comunes no rompen la pantalla.
- Volver a selección individual recupera la vista normal de Outputs.
- No se modifican datos existentes ni se realizan commits.

VERIFICACIÓN

Ejecuta las comprobaciones que permita el entorno, como mínimo el build o la comprobación de TypeScript del frontend. Añade pruebas focalizadas para la selección de variables comunes y el emparejamiento de muestras si el proyecto tiene infraestructura de tests adecuada.

Al terminar, indica:

- Qué archivos has modificado y por qué.
- Cómo se activa y utiliza la comparación.
- Qué comprobaciones has ejecutado y su resultado.
- Qué casos no has podido verificar o qué decisiones necesitan confirmación.

He dejado la correlación condicionada a que existan muestras realmente emparejables: dos batches pueden tener la misma variable sin contener necesariamente las mismas muestras, y unirlas por orden produciría una gráfica engañosa.
```

**Resultat.**
- En una comprovació a mig camí, el servidor havia rebut 15 peticions (`launch_slot_` al log), OpenCode havia
  compactat la sessió 5 vegades, i `git status` i `git diff --stat` al clon no mostraven cap fitxer modificat.
  **VALIDATED**, en aquell moment.
- Més tard, l'autor va explicar que l'agent havia fet unes 20 execucions i només s'havia dedicat a autocompactar-se,
  sense escriure res. **UNVERIFIED**: relat de l'autor, sense el log ni el recompte final.
- La prova es dona per **no completada**. L'autor ha decidit passar a un altre prompt, molt més concret.

**Anàlisi de la sessió exportada de la prova 1 (`/export`, Markdown), VALIDATED.** Sense llegir-ne el contingut:
- 1,06 MB i 29.955 línies. 19 respostes de l'assistent i 4 missatges de l'autor. Dos d'aquests, els dos primers, són el
  mateix prompt de 119 línies (mateixa empremta), enviat dues vegades.
- Eines cridades: 17 `read`, 1 `shell` i 1 `write`. Els blocs grans de l'assistent (uns 1.740 línies cadascun, que
  seria prop de 14.000 tokens, xifra aproximada) són lectures de fitxers. Segons el `git status` del clon, no hi va
  haver cap fitxer modificat. El `write` (línia 22976 de l'exportació) no s'ha pogut inspeccionar: **PENDENT** què va escriure i on. El fitxer
  `OutputsComparePanel.tsx` va aparèixer al clon més tard, però durant la prova 2 (vegeu més avall), de manera que no
  es pot atribuir a aquesta sessió.
- Línies diferents: 3.610 d'unes 29.955. Hi ha molta repetició.
- Fitxers que apareixen més: `OutputsDashboard.tsx` (unes 40 vegades, amb rutes diferents) i `OutputsDetailPanel.tsx`
  (unes 16). Mida d'aquests fitxers: `OutputsDashboard.tsx` 3.263 línies, `OutputsDetailPanel.tsx` 8.006 línies i
  `OutputsSidePanel.tsx` 918.

**Lectura, UNVERIFIED.** Un sol `read` en retorna unes 1.740 línies, la meitat de la finestra útil abans de compactar.
`OutputsDashboard.tsx` en té 3.263, així que no cap sencer, i el model el torna a llegir després de cada compactació.
Que el contingut dels blocs sigui repetit encara no s'ha comprovat bloc a bloc.

**Hipòtesi, UNVERIFIED.** Amb un context de 32.768 tokens OpenCode compacta cap als ~22.700. Una tasca que obliga a
llegir diversos fitxers llargs omple el context amb una o dues lectures, es compacta, el model oblida el detall i torna
a llegir els mateixos fitxers. La hipòtesi no s'ha comprovat mirant quins fitxers llegia a cada cicle. Ja es va
anticipar a X002 com a risc sense provar.

**Mètriques de X003 que no es van recollir.** Hora d'inici i de fi, nombre de torns, tokens de context al final,
ordres denegades i fitxers llegits a cada cicle. Per a la propera prova, apuntar-ho des del principi.

**Què demostra i què no.** Amb aquest model, aquest context i aquest prompt, l'agent no va avançar. No es pot dir que
el Coder 30B no serveixi per a Lipopótamo: el prompt demanava una funcionalitat sencera de frontend, amb diversos
fitxers per llegir. El resultat descriu aquesta combinació.

## Prova 2: la mateixa funcionalitat en tres passos

**Configuració.** La mateixa que la prova 1 (model, servidor, túnel i context de 32.768 tokens), tret que l'autor
anoti algun canvi.

**Disseny dels prompts, segons l'autor.** Després de la prova 1, la tasca s'ha reescrit en tres prompts seqüencials.
Cadascun diu quins fitxers es poden tocar, els canvis exactes, els tipus de dades que ja existeixen i l'ordre de
verificació, i prohibeix explorar la resta del repositori. La intenció és que l'agent llegeixi pocs fitxers a cada pas
i el context no s'ompli. Aquesta intenció no està provada. El primer prompt acaba demanant que no es faci el pas 2
encara.

**Una precisió de l'autor.** Al primer pas es demana avisar si la selecció no es conserva entre pàgines, perquè
`selectedRows` es deriva de l'estat que ja existeix i caldria comprovar aquest cas amb dades reals. Així no es força
el model a inventar una solució sense veure com carrega la taula.

**Prompt 1: muntar la vista (tal com es va enviar).**

```
Implementa SOLO el paso 1 de una comparación de batches en Lipopotamo.

No hagas commits. No ejecutes Docker. No toques backend. No explores todo el repositorio.

Primero ejecuta:
git status --short

Archivos permitidos:
- frontend/src/components/dashboard/outputs/OutputsDashboard.tsx
- frontend/src/components/dashboard/outputs/OutputsSidePanel.tsx
- frontend/src/components/dashboard/outputs/OutputsComparePanel.tsx (nuevo)

CONTEXTO EXACTO

En OutputsDashboard.tsx ya existen:
- isMultiSelectEnabled: boolean
- selectedRows: OutputListRow[]
- selectedOutputIds: string[]
- OutputsSidePanel
- OutputsDetailPanel

Ahora mismo:
- handleSelectSoftware bloquea la selección cuando Multi Select está activo.
- OutputsSidePanel muestra el texto “Checks are disabled while multi-select mode is active.”
- El panel de detalle solo aparece si selectedRow y selectedSoftwareId existen.

OBJETIVO DE ESTE PASO

Permitir abrir un panel de comparación cuando hay al menos 2 batches seleccionados.

CAMBIOS EXACTOS

1. Crea OutputsComparePanel.tsx en la misma carpeta.

   Debe exportar un componente por defecto con esta firma:

   type Props = {
     selectedRows: OutputListRow[]
     onClose: () => void
   }

   Importa OutputListRow desde:
   ../../../types/outputs

   Por ahora muestra:
   - Título: “Compare batches”.
   - El nombre de cada batch seleccionado: row.batch o row.id.
   - Un botón “Close” que llama a onClose.
   - Usa componentes Chakra y los colores de tema existentes, por ejemplo
     workspaceHeading, workspaceText, nestedPanelBg y nestedPanelBorder.

2. En OutputsDashboard.tsx:
   - Importa OutputsComparePanel.
   - Añade un estado local:
     const [isCompareOpen, setIsCompareOpen] = useState(false)
   - Añade un botón “Compare results” que solo se vea cuando
     isMultiSelectEnabled sea true y selectedRows.length >= 2.
   - Al pulsarlo, pon isCompareOpen a true.
   - Al desactivar Multi Select o quedar menos de dos batches, cierra la comparación.
   - Cerca del render actual de ReportPreview y OutputsDetailPanel, renderiza
     OutputsComparePanel cuando isCompareOpen sea true.
   - No cambies la lógica de selección individual ni sus paneles.

3. Usa selectedRows, no solo las filas de la página visible. Si descubres que
   selectedRows no conserva selecciones entre páginas, indícalo en el resultado
   y evita inventar datos.

4. Ejecuta:
   cd frontend
   npm run build

   Si falla por la versión local de Node, ejecuta:
   npx tsc -b

No hagas el paso 2 todavía. Al terminar, explica qué has cambiado y el resultado
del build.
```

**Prompt 1, versió 2: lectura per rangs (enviat en una sessió nova d'OpenCode).** La prova 1 va fallar perquè
`OutputsDashboard.tsx` té 3.263 línies i un sol `read` en retorna unes 1.740. Aquesta versió diu quins rangs de línies
llegir (imports 55-90, `isMultiSelectEnabled` 295-315, `selectedRows` 580-615, `handleSelectSoftware` 1040-1100,
interruptor de Multi Select 2600-2660, `OutputsSidePanel` i `ReportPreview` 2740-2800, `OutputsDetailPanel`
2845-2875), treu `OutputsSidePanel.tsx` dels fitxers permesos, limita la sortida del build amb `tail -40` i prohibeix
tornar a llegir el fitxer sencer després d'una compactació. La resta del prompt 1 és igual. Els rangs surten de
`grep -n -o` sobre els símbols clau, i són orientatius.

*Resultat, segons l'informe de l'agent i el que es veu a la pantalla.* **VALIDATED** només el que es pot comprovar
al clon i a la pantalla:
- Duració: 12 min 14 s. Context al final: 9.258 tokens (28 %). Va llegir rangs petits i no es va encallar en el bucle
  de lectura de la prova 1.
- `git status` després: `OutputsComparePanel.tsx` nou i no seguit (`??`); `git diff --stat` buit, és a dir, cap fitxer
  existent modificat. `OutputsDashboard.tsx` queda intacte.
- **UNVERIFIED**, afirmacions de l'agent que no s'han comprovat o són falses: va dir que el pas 1 estava
  implementat i que la integració al dashboard «es farà al pas 2», quan el prompt posa els canvis del dashboard dins
  del pas 1; ell mateix llistava cinc canvis com a «no s'ha fet encara». També va proposar `usePersistentState` per
  a `isCompareOpen` quan el prompt demanava `useState`.
- Build: no es va executar, segons l'agent per «restriccions de permisos». Això és la configuració d'`opencode.json`,
  que no permet `npm`. Tot i això va afirmar que el codi «hauria de compilar». És el mateix patró de la primera prova de
  fum (X001): donar la tasca per bona sense executar-la.

**Missatge de seguiment, a la mateixa sessió.** Com que l'agent no havia fet les edicions del dashboard, l'autor li
va enviar un missatge que deia que el pas 1 no estava complet, que fes les cinc edicions d'`OutputsDashboard.tsx` una a
una amb `edit` (rellegint uns 15 línies després de cadascuna), amb `useState` i no `usePersistentState`, que
executés `npm --prefix frontend run build 2>&1 | tail -40` i que no digués que compilava si no havia vist el
resultat.

*Resultat, segons l'informe de l'agent.* Duració del missatge de seguiment: 42 min 42 s (el pas 1 inicial va ser de
12 min 14 s).
- El build va donar `Permission denied: shell`. La causa és el `opencode.json` global, que no permet `npm` (només
  `python`, `pytest`, `coverage`, `mutmut`, `git status` exacte, `git diff*`, `git log*` i `ls*`), i que el
  patró `git status` sense `*` tampoc deixa passar `git status --short`. **VALIDATED** la llista de permisos; la
  correcció proposada no s'havia aplicat quan es va enviar el missatge.
- L'agent va dir que havia fet les cinc edicions «exactament com es va demanar» i va recomanar a l'autor
  executar el build. A diferència de la prova anterior, aquest cop no va afirmar que compilava. Va dir que havia
  col·locat el render de `OutputsComparePanel` «just abans del `return`», no al JSX prop de `ReportPreview` i
  `OutputsDetailPanel` com demanava el prompt: **UNVERIFIED**, cal mirar el diff.

*Dades del servidor durant aquesta sessió, VALIDATED.* Servidor encès 93 minuts. Velocitat de generació de 48 a
80 t/s (la més baixa, amb ~24.500 tokens de context). Després d'una compactació, la memòria cau del servidor només
servia un 17 % del prompt (`f_sim_best = 0,174`), i va rellegir uns 10.000 tokens a 470-575 t/s. VRAM de la B60 a
20,4 GB. RAM lliure del PC: 2,3 GB, baixant des de 10-18 GB abans de la sessió (la VM de WSL, els contenidors i
l'ús de l'altra persona). El «2,5 tok/s» i l'«1,7 tok/s» que mostra OpenCode no són la velocitat de generació del
servidor: probablement són una mitjana del missatge sencer, amb lectures, esperes i compactacions (**UNVERIFIED**).

**PENDENT** del prompt 1: `git diff --stat`, errors de TypeScript (`npx tsc -b`), i revisar a mà les cinc edicions
(sobretot si el render és dins del JSX).

**Prompt 2: cargar y comparar datos (tal com es va guardar).**

```
Continúa con el paso 2 de la comparación. Usa los cambios del paso anterior.

No hagas commits. No ejecutes Docker. No toques backend.
No examines OutputsDetailPanel.tsx: este paso solo afecta al componente nuevo.

Archivo principal:
frontend/src/components/dashboard/outputs/OutputsComparePanel.tsx

Puedes consultar:
frontend/src/types/outputs.ts
frontend/src/services/outputs.ts

DATOS EXISTENTES

En frontend/src/services/outputs.ts existe:

fetchSoftwareDistributions(
  batchId,
  softwareId,
  variable?,
  analysisMode?,
  signal?,
  timeGroupIds?,
  options?
)

El resultado tiene este tipo, definido en frontend/src/types/outputs.ts:

SoftwareDistributionData = {
  available: boolean
  software: string
  batch: string
  variables: string[]
  items: {
    variable: string
    referenceVariable: string
    sampleValues: { label: string; value: number }[]
  }[]
}

Cada item representa una variable; sampleValues contiene sus muestras y valores.

OutputListRow incluye:
- id: identificador del batch.
- batch: nombre visible.
- availableSoftwares: string[].
- libraryFrozen: boolean opcional.
- libraryFrozenAt: string opcional.

OBJETIVO

Cargar las distribuciones reales de cada batch y mostrar las variables comunes.

CAMBIOS EXACTOS

1. En OutputsComparePanel, calcula la intersección de availableSoftwares de
   todos los batches seleccionados, limitada a:
   liposcale, glicoscale, nemo, lippol.

2. Muestra un selector de software. Selecciona el primero disponible por defecto.
   Si no hay ningún software común, muestra “No common software available”.

3. Cuando cambien los batches o el software:
   - Crea un AbortController.
   - Limpia los resultados anteriores.
   - Llama a fetchSoftwareDistributions para cada batch:
     fetchSoftwareDistributions(
       row.id,
       software,
       undefined,
       'global',
       controller.signal,
       [],
       row.libraryFrozen && row.libraryFrozenAt
         ? { libraryFrozenAt: row.libraryFrozenAt }
         : undefined
     )
   - Usa Promise.allSettled para que el fallo de un batch no rompa la pantalla.
   - Cancela las peticiones anteriores en el cleanup del useEffect.
   - Muestra loading mientras cargan y el nombre de cada batch que haya fallado.

4. Calcula las variables comunes mediante la intersección de item.variable de
   los batches cargados. No mezcles variables de distintos softwares.

   Si falla un batch o no devuelve datos, indícalo claramente. La comparación
   debe requerir al menos dos batches con datos disponibles.

5. Muestra un selector de variable común.

6. Para la variable seleccionada, muestra por cada batch:
   - Nombre.
   - Número de valores válidos.
   - Lista sencilla de sus valores para verificar la carga.

   Un valor válido debe ser typeof value === 'number' y Number.isFinite(value).
   No conviertas null a 0.

7. No hagas aún los gráficos. No inventes datos.

8. Ejecuta cd frontend && npm run build. Si el build no se puede ejecutar por
   Node, indica la versión y prueba npx tsc -b.

Al terminar, explica los archivos modificados y los resultados de verificación.
```

**Resultat del prompt 2 (enviat a la mateixa sessió, sense `AGENTS.md`).** La versió enviada porta tres ajustos sobre el
text de dalt: no s'intenta el build (denegat per permisos) i es demana declarar que no està verificat, regla de lectura
per rangs (cap lectura de més de 200 línies, i no cal llegir `types/outputs.ts` ni `services/outputs.ts`), i les
edicions amb `edit` o `write` rellegint només el rang editat.
- Duració: 15 min 5 s. Context al final: 14.912 tokens (46 %). **VALIDATED** (pantalla d'OpenCode).
- Informe de l'agent, **UNVERIFIED**: diu que ha implementat `getCommonVariables()`, el selector de software amb el
  primer disponible per defecte i el missatge «No common software available», un indicador de càrrega, el tractament
  d'errors, el selector de variable comuna, i per cada batch el nom, el nombre de valors vàlids i una llista limitada
  a 10 valors amb un missatge de «més valors». Diu també que cancel·la les peticions anteriors i que només compta
  nombres finits.
- L'agent acaba dient que el component està «completament funcional», tot i que el prompt demanava no afirmar que
  funciona ni que compila. És una afirmació que no ha pogut comprovar. El límit de 10 valors és una decisió pròpia que
  el prompt no demanava.
- **PENDENT:** nombre de línies d'`OutputsComparePanel.tsx`, errors de TypeScript, i revisar a mà si la càrrega usa
  `Promise.allSettled`, si hi ha `AbortController` amb neteja al `useEffect`, i si el valor vàlid es comprova amb
  `typeof` i `Number.isFinite` sense convertir `null` a 0.

**Prompt 3: gráficos (tal com es va guardar).**

```
Continúa con el paso 3. El componente OutputsComparePanel ya carga
SoftwareDistributionData y permite seleccionar software y variable.

No hagas commits. No ejecutes Docker. No toques backend.

Archivo a modificar:
frontend/src/components/dashboard/outputs/OutputsComparePanel.tsx

OBJETIVO

Sustituir la lista sencilla de valores por gráficos de comparación reales.

LIBRERÍA DISPONIBLE

El proyecto carga Plotly mediante:
import { loadPlotly } from '../../../lib/plotly'

No instales dependencias. Puedes usar useRef y useEffect con Plotly.react.
Al desmontar, llama a Plotly.purge para limpiar el gráfico.

Puedes consultar frontend/src/components/plotter/PlotterWorkspace.tsx, función
PlotlyPanel, únicamente como ejemplo de uso de loadPlotly.

CAMBIOS EXACTOS

1. Crea dentro de OutputsComparePanel un componente pequeño para representar
   gráficos Plotly, que reciba data y layout.

2. Para la variable seleccionada, dibuja:

   A. Distribución:
      - Una traza scatter por batch.
      - Eje X: posición de cada muestra dentro de su batch.
      - Eje Y: valor.
      - Nombre de la muestra en hover.
      - Colores distintos y leyenda con el nombre de cada batch.

   B. Boxplots:
      - Una traza tipo box por batch.
      - Datos en Y; boxpoints configurado para mostrar las muestras.
      - Nombre y color de batch visibles.

3. Correlación:
   - Si hay más de dos batches con datos, ofrece dos selectores para elegirlos.
   - Empareja solo sampleValues con el mismo label completo y no vacío.
   - Si hay labels duplicados en cualquiera de los dos batches, exclúyelos del
     emparejamiento para no relacionar muestras ambiguas.
   - Dibuja cada par como un punto scatter: valor del primer batch en X y
     del segundo en Y; muestra el label en hover.
   - Si no hay pares, muestra:
     “No matching sample identifiers between these batches”.
   - NO emparejes por posición ni elimines prefijos de los identificadores.
     Si los IDs completos son distintos por contener el nombre de batch, informa
     de que no se han podido emparejar; no supongas que los sufijos coinciden.

4. Mantén los estados de loading/error/sin datos ya implementados.
   Los gráficos deben adaptarse al ancho disponible y ser legibles en modo
   claro y oscuro. Reutiliza los colores de tema existentes cuando sea posible.

5. Comprueba:
   - Dos batches con la misma variable.
   - Dos batches sin muestras emparejables.
   - Más de dos batches.
   - Cambio de variable y software.
   - Retorno a la vista individual.

   Si no puedes probar un caso en el navegador, dilo explícitamente.

6. Ejecuta cd frontend && npm run build. No hagas commits.

Al terminar, resume qué has cambiado, qué has verificado y qué queda pendiente.
```

**Resultats, per prompt.** Tot **PENDENT**. Per a cada pas, apuntar:

| | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---|---|---|
| Hora d'enviament i de fi | | | |
| Compactacions | | | |
| Fitxers modificats | | | |
| `npm run build` o `npx tsc -b` | | | |
| Casos que l'agent diu que no ha pogut provar | | | |
| Verificació externa de l'autor | | | |

## Conclusions provisionals i següents passos (2026-11-05)

**Impressió de l'autor, UNVERIFIED.** Fer servir aquest model com a agent en un espai de treball tan ampli i
robust com Lipopótamo és molt poc pràctic amb aquesta configuració: és lent, el context ho consumeix tot i funciona
poc en un projecte ja començat. L'autor la dona per una conclusió d'aquestes dues proves.

**Què la recolza, VALIDATED.** A la prova 1, unes 20 peticions i 5 compactacions sense escriure cap fitxer; a la
prova 2, més de 55 minuts només per al pas 1, amb un component nou i les edicions del dashboard gràcies a un
missatge de seguiment, i amb un informe de l'agent que no coincidia amb el que havia fet. **Què no es pot separar
encara:** si el límit és el model (30B amb ~3B actius), el context de 32K, la mida dels fitxers d'aquest projecte
(3.263 i 8.006 línies) o l'arnès. Per això la conclusió descriu aquesta combinació, no els agents locals en general.

**Entrenar el model amb Lipopótamo, descartat per ara.** No memoritza el codi amb exactitud, i `edit` necessita el
text exacte d'ara, de manera que hauria de llegir el fitxer igualment; el codi canvia i caldria reentrenar; un
30B no sembla caber en una targeta de 24 GiB per entrenar-lo (no provat); i caldria el permís de Biosfer.

**Proves següents, per ordre.**
1. Acabar la prova 2 amb el Coder (passos 1, 2 i 3) amb lectura per rangs i permisos de `npm`.
2. Experiment de control amb el mateix model, servidor i arnès sobre codi de l'autor, amb fitxers petits, per separar
   «la mida dels fitxers de Lipopótamo» de «el muntatge».
3. Mesurar l'esforç de l'autor, no només el de l'agent: intervencions, prompts reescrits i resultats falsos.
4. Repetir els mateixos prompts amb el Qwen3.8 27B, al context màxim mesurat. Els 16.384 de la seva configuració són
   una estimació.
5. Reduir el context necessari: una nota curta del projecte (`AGENTS.md`, que l'autor està preparant), un índex o
   mapa del codi per llegir només rangs, i quantitzar la memòria de context (`-ctk q8_0 -ctv q8_0`), que no s'ha
   provat amb Vulkan a la B60.

## Model candidat: Qwen3.6-35B-A3B, UD-IQ3_S (proposat per un usuari de Reddit, 2026-11-05)

**Dades de la pàgina del model** (`unsloth/Qwen3.6-35B-A3B-GGUF` a Hugging Face), llegides amb una eina que resumeix la
pàgina, **UNVERIFIED**:
- Arquitectura híbrida amb MoE: 40 capes, repetint 3 capes de Gated DeltaNet (atenció lineal) i 1 de Gated Attention,
  totes amb MoE. 256 experts, 8 encaminats i 1 de compartit per token. 16 caps de consulta i 2 caps KV. 35.000 milions
  de paràmetres, 3.000 milions actius. Context natiu de 262.144 tokens. Publicat a l'abril de 2026.
- Mides: `UD-IQ3_S` 13,7 GB (uns 12,8 GiB) i `UD-Q4_K_M` 22,1 GB.

**Estimació de memòria, actualitzada amb la capçalera del fitxer (2026-11-05).** La capçalera del `UD-IQ3_S`, llegida
per rang sense baixar-lo, confirma 40 capes, un `full_attention_interval` de 4 (10 capes d'atenció completa), 2 caps
KV i una dimensió de cap de 256. El KV són 20 MiB per cada mil tokens en f16, davant dels 96 MiB del Coder: 1,25 GiB a
65536, 2,5 GiB a 131072 i **5,0 GiB a 262144**. Amb uns 12,8 GiB de pesos, 5,0 de KV, uns 63 MiB d'estat recurrent
(estimat) i el buffer de còmput (que a 256K no se sap, possiblement entre uns 0,2 i 1 GiB) més l'1 GiB de marge de
llama.cpp, surten uns **19 a 20 GiB dels ~23,6 de la B60**. Amb el `Q4_K_M` (uns 20,6 GiB de pesos) no hi cabria. És un
càlcul, no una mesura: el KV real de llama.cpp s'ha de mesurar a la B60.

**El que no se sap.** Si el `b11889` el carrega. La velocitat de generació i de lectura de prompt a Vulkan sobre la
B60: el Qwen3.8 27B, que també és híbrid, va donar 15,6 t/s, menys que el Qwen2.5 32B dens (18,1 t/s) amb més
paràmetres. La qualitat de l'IQ3_S per a codi i eines: un usuari diu que és «molt robust», sense dades. I si és millor
que el Coder com a agent: el Coder és de 2025 i especialitzat en codi, aquest és general.

**Pla de proves**, amb el permís de Biosfer per baixar 13,7 GB al seu disc:
1. Arrencar a `-c 65536` i llegir al log els MiB de KV; després 131072 i 262144. Treure els MiB per mil tokens reals.
2. `llama-bench` per a `pp` i `tg`, amb context buit i amb context llarg.
3. IFEval (prompt strict), contra el 0,789 ± 0,018 del Coder.
4. Si tot va bé, els tres prompts de l'agent.
Vigilar la RAM lliure del PC i fer servir `--sleep-idle-seconds`, com a X004.

**Jaull, pendent.** Segons el codi de l'estimador de KV (`kv_cache.py`) i la detecció d'arquitectures no estàndard,
no hi ha res per a atenció lineal ni per a models híbrids: ni `layer_types` ni un interval d'atenció completa. Per a
un model així, o bé retornaria «desconegut» (per exemple, si la configuració és composta multimodal), o bé
aplicaria la fórmula estàndard a les 40 capes i sobreestimaria el KV unes 4 vegades (**UNVERIFIED**: és una
lectura del codi, no s'ha executat Jaull amb aquest model). Abans de tocar res cal mesurar el KV real amb llama.cpp,
perquè la regla de Jaull és que un benchmark valida una hipòtesi i no canvia una fórmula per si sol. Que Jaull el
mostri abans que el Coder és un tema de política de rànquing: només es toca amb evidència.

**Actualització, 2026-11-10.** El model s'ha provat amb el fitxer `UD-IQ4_XS` (16,50 GiB) i no amb l'`IQ3_S`: el KV i la
velocitat estan a [D125](DECISIONS.md) i [X005](DECISIONS-2.md), i Jaull ja el tracta com a híbrid (D125, D126).

## Prova 3: Prompt 1 amb el Qwen3.6-35B-A3B a 128K (2026-11-10)

**Configuració.** Servidor `llama-server.exe` b11889 amb Vulkan natiu a Windows, `-c 131072 -np 1 -lm dio
--sleep-idle-seconds 45 --cache-ram 0 --jinja`, amb el raonament que el model porta per defecte i sense
`--no-reasoning-preserve`. Mateix túnel i mateix `opencode.json` global, amb `limit.context` a 131072. Mateix clon de
Lipopótamo, en una branca nova (`qwen36-p1`). Dues diferències amb la prova 2 del Coder, que no es poden separar:
**el model** i **el context** (32K contra 128K). **Prompt 1 original**, el mateix que es va enviar al Coder, no la versió 2
per rangs. Sense `AGENTS.md`. Als permisos de `bash` s'hi han afegit `git status*` i `npm --prefix frontend run build*`
abans de la prova (el Coder no els tenia). **Límit de sortida:** `limit.output` era de **4.096 tokens**, heretat sense
voler de l'entrada del Coder en crear la del 35B (un error de l'autor del muntatge, no del model), i es va pujar a 32.768
quan, a la segona tanda, una escriptura va quedar tallada («Response still truncated after two output token limit
continuations»). El pas 1 i els seus seguiments de build es van fer, doncs, amb 4.096.

**Fitxers.** Segons `git`, **VALIDATED**: `OutputsDashboard.tsx` modificat (39 insercions, cap esborrat) i
`OutputsComparePanel.tsx` nou.

| | Prompt 1, Qwen3.6-35B-A3B |
|---|---|
| Durada del pas 1 | 3 min 38 s |
| Context al final del pas 1 | 26.265 tokens (20 %) |
| Compactacions | 0 |
| Seguiments per al build | tres: 1 min 16 s, 2 min 24 s i 1 min 54 s |
| Context al final | 33.966 tokens (26 %) |
| Fitxers modificats | 1 existent (+39) i 1 de nou |
| `npm run build` | passa, segons l'agent («built in 1.31s», sense errors de TypeScript); **UNVERIFIED** fins que es comprovi a mà |
| Verificació externa de l'autor | **PENDENT** (build a mà i revisió del diff) |

**Què va passar amb el build.** Va ser un problema d'entorn, no del model. El primer informe deia que les «instruccions
del sistema» bloquejaven el shell; amb els permisos nous va poder executar l'ordre, i va fallar amb `sh: 1: tsc: not
found` perquè al clon no hi havia `node_modules`. Va intentar `npm install`, denegat pels permisos, i va repetir
l'ordre 16 vegades (12 amb `tsc: not found` i 4 amb `Permission denied`) abans de parar i explicar-ho. L'autor va fer
`npm ci` (476 paquets, 52 s; Node v26.13.0; `git status` no va canviar) i, al missatge següent, el build va passar.
Una de les hipòtesis que es van provar pel camí, que el permís fallés per la canonada `| tail -40`, **es va descartar**:
l'ordre amb la canonada sí que va passar el filtre.

**Què es pot dir, i què no.**
- **VALIDATED:** el 35B va editar un fitxer existent al primer intent, cosa que el Coder no va fer al pas 1 (el seu
  `git diff --stat` va quedar buit); el pas 1 no va compactar i va gastar una cinquena part del context; va detectar sense
  que el prompt el hi expliqués que `selectedRows` es deriva de les files de la pàgina visible (el prompt demanava
  indicar-ho) i ho va deixar per al pas 2; i mai no va dir que el codi compilés sense haver vist el resultat.
- **UNVERIFIED:** que el codi sigui correcte més enllà de compilar. Cal comprovar a mà que el render és prop de
  `ReportPreview` i `OutputsDetailPanel` (el Coder el va posar «just abans del `return`»), que l'estat és un `useState`
  (el Coder va proposar `usePersistentState`) i que el panell es tanca en desactivar Multi Select o quedar menys de dues
  files. Les afirmacions de l'agent sobre què ha fet no s'han contrastat amb el diff, més enllà del recompte de fitxers.
- **No es pot concloure** que el context sigui el que ha fet la diferència amb el Coder: ha canviat també el model, el
  mode de raonament i els permisos de `bash`.

**Límits.** Una sola execució, d'un sol pas, sobre un projecte. Les durades i els tok/s que mostra OpenCode són mitjanes
del missatge i no la velocitat del servidor (68-70 t/s amb el context buit).

**Següent.** Passar a una branca nova i enviar la tasca sencera de la prova 1, en un sol missatge, amb dues salvaguardes
afegides: llegir per rangs els fitxers de més de 1.500 línies i no repetir una ordre que falla dues vegades amb el mateix
error. Apuntar els tokens de context, les compactacions i les ordres repetides.
