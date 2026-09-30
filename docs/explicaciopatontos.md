# Jaull explicado sin complicarse

Jaull ayuda a responder una pregunta: **¿qué modelo de IA puedo ejecutar en este
ordenador y cómo compruebo que funciona?**

No es un chatbot ni entrena modelos. Mira tu equipo, busca candidatos, estima
la memoria que necesitan y prepara una forma concreta de ejecutarlos. Después
puede hacer pruebas reales y guardar lo que ocurrió.

El estado actual está en [STATUS.md](../STATUS.md). Esta guía explica las ideas;
no sustituye a los records ni a los informes experimentales.

## 1. El recorrido completo

```text
Tu ordenador
    ↓
Lo que necesitas
    ↓
Modelos candidatos
    ↓
Estimación y recomendaciones
    ↓
Plan de ejecución y comprobaciones
    ↓
Run / Validate / Benchmark
    ↓
Resultados guardados
```

Primero Jaull lee información, no descarga los pesos ni ejecuta el modelo.
La ejecución empieza cuando la pides. Si falta un GGUF compatible, el flujo
puede descargarlo antes de ejecutarlo.

## 2. Qué mira del ordenador

Principalmente CPU, RAM, GPU y VRAM disponible.

- **RAM:** memoria del sistema.
- **VRAM:** memoria de la tarjeta gráfica.
- **Offloading:** repartir trabajo y pesos entre GPU y CPU cuando hace falta.

Con una GPU dedicada, RAM y VRAM no son una sola bolsa de memoria. Que tengas
8 GiB de VRAM y 32 GiB de RAM no significa que un modelo de 40 GiB pueda cargarse
sin más. Hay que comprobar qué parte necesita cada dispositivo.

## 3. Qué le cuentas tú

Hay dos preguntas diferentes:

- **Tarea:** programar, conversar, escribir o responder sobre documentos.
- **Modo de uso:** interactivo o batch, por ejemplo procesar trabajos en lote.

Un modelo puede servir para programar tanto de forma interactiva como en batch.
La tarea ayuda a encontrar modelos relevantes; el modo no convierte un modelo
general en un buen modelo de programación.

También importan el contexto, los usuarios previstos, los idiomas y la licencia.
El contexto se mide en tokens: fragmentos de texto que procesa el modelo.
Más contexto suele necesitar más memoria para la **KV cache**, que conserva
información de la conversación mientras genera la respuesta.

Jaull puede guardar objetivos opcionales como una velocidad mínima o un TTFT
máximo. **Pedir un objetivo no demuestra que el equipo lo cumpla.**
TTFT significa tiempo hasta recibir el primer token.

## 4. Qué significa una recomendación

Jaull estima pesos, KV cache y otros costes de ejecución. Añade reserva y margen
de seguridad para decidir si el plan cabe. Esos márgenes son presupuesto:
no son bytes que el proceso necesariamente va a ocupar.

Puede indicar que un modelo cabe cómodamente, va justo, necesita offloading,
no cabe o tiene compatibilidad desconocida.

**Unknown no significa que no funcione.** Significa que Jaull no tiene datos
suficientes para confirmarlo. Debe explicar el motivo y mostrarlo como alternativa
no confirmada. Una validación puede ser útil si existe un plan ejecutable.

La posición se decide por los criterios del motor. El score compuesto que
aparece en el report es diagnóstico cuando se usa el motor de planes: no es
el número que ordena esa lista ni una nota medida de calidad del modelo.

## 5. Modelo, archivo y runtime no son lo mismo

Piensa en estos tres elementos:

- **Modelo:** por ejemplo Qwen2.5-7B-Instruct.
- **Artefacto:** los archivos concretos, como un GGUF Q4_K_M.
- **Runtime:** el programa que los carga, como llama.cpp o Transformers.

La cuantización reduce la precisión de los pesos para ocupar menos memoria.
Puede afectar a la calidad; un modelo cuantizado no ocupa lo mismo que su
versión FP16. En Transformers, int4/int8 requiere un mecanismo de cuantización,
no basta con cambiar un `torch_dtype`.

El **plan** reúne artefacto, runtime, backend, contexto y flags. Su readiness
explica si está listo, si Jaull puede prepararlo o si está bloqueado.
Un modelo puede ser buena recomendación y tener Run deshabilitado porque falta
el runtime, una dependencia como bitsandbytes o un artefacto compatible.
Eso no debería empeorar su posición como recomendación.

Antes de ejecutar un GGUF, Jaull verifica su tamaño y SHA-256. El SHA es una
huella del archivo: sirve para comprobar que los bytes son los esperados.
La verificación completa recalcula esa huella desde el archivo.

## 6. Run, Validate y Benchmark

| Acción | Qué hace |
|---|---|
| Run | Ejecuta el modelo con tu prompt. |
| Validate | Hace una prueba controlada y guarda predicción, observación y comparación. |
| Benchmark | Mide rendimiento con una metodología y configuración registradas. |

Validate puede guardar un fallo. Eso también aporta información: no hay que
convertirlo en éxito ni borrarlo porque otra ejecución posterior funcionó.

Un benchmark con `llama-bench` mide trabajo de prefill y generación. No prueba
automáticamente cuatro usuarios simultáneos ni el contexto del plan: solo los
parámetros que se aplicaron de verdad al benchmark.

La duración total de Validate tampoco es TTFT: puede incluir carga y preparación.

## 7. Predicción frente a observación

- **Predicción:** lo que Jaull esperaba antes de ejecutar.
- **Observación:** lo que se midió o reportó durante la ejecución.
- **Comparación:** si se pueden contrastar y cuál es la diferencia.

No todo número de memoria mide lo mismo. Los buffers reportados por llama.cpp
no incluyen necesariamente toda la VRAM del proceso. La medición por proceso
de NVML puede no estar disponible bajo WDDM. Y la RAM RSS no equivale a los
pesos que quedaron en CPU: llama.cpp usa archivos mapeados en memoria.

Por eso a veces verás `methodologically_unavailable`: existen datos, pero
compararlos como si midieran lo mismo sería engañoso. No significa necesariamente
que el modelo haya fallado.

Los records originales no se cambian. Reevaluar uno significa aplicar el código
actual a sus entradas guardadas, no volver a ejecutar el modelo ni inventar
información que faltaba.

## 8. Qué hemos probado y qué falta

Hay pruebas reales en RTX 2060 y RTX 4060. La campaña final de la 4060 tiene
Validate y benchmark completos. Puedes leer la
[comparación entre campañas](qwen2.5-tests/rtx4060-campaign-comparison.md).
No es una comparación controlada de GPU: cambian también runtime y entorno.

Todavía falta medir servicio concurrente: un único modelo cargado atendiendo
1, 2 y 4 usuarios. Lanzar cuatro copias del modelo sería otro experimento.
Qualification y `jaull.lock` son pasos posteriores, no funciones terminadas.

## 9. Cómo empezar y dónde mirar

```bash
uv run jaull doctor
uv run jaull ui
```

Doctor revisa el entorno. La TUI permite seguir el recorrido guiado.
llama.cpp necesita sus ejecutables correspondientes; no viene instalado
simplemente por clonar Jaull.

- [CLI](cli.md): comandos y opciones.
- [Evidencia](evidence.md): qué se guarda y qué se puede comparar.
- [Limitaciones](limitations.md): qué no podemos afirmar todavía.
- [Arquitectura](../ARCHITECTURE.md): cómo se organiza el código.
- [Índice experimental](../validation/README.md): dónde están las pruebas.

**En una frase:** Jaull propone una configuración, explica sus límites y permite
comprobarla con datos reales; no promete resultados que todavía no ha medido.
