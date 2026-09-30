> Publication note (2026-09-30): this is an anonymized derivative. Statements below about original bytes/hashes describe collection-time evidence, not this public copy. See validation/public-anonymization.json and the campaign PUBLICATION.md.

﻿# Intento con runtime recuperado: fallo de persistencia

El runtime oficial b11258 se recuperó localmente; los 54 hashes originales EXE/DLL coinciden. El SHA del modelo también coincide. Se invocó ExperimentRunner, pero build_experiment_record falló porque el helper recalculó MemoryEstimate sin vincular su runtime_recommendation al runtime elegido. No se guardó un ExperimentRecord de este intento. La traza está en failure.txt.

Los raw outputs de este intento no se guardaron: la excepción ocurrió antes de la persistencia opt-in de ExperimentRunner. No se reconstruyen ni se inventan. No se puede declarar el resultado de inferencia a partir de esta traza.

La corrección se hará únicamente en un nuevo helper temporal, usando AdvisorService.plan_execution con el launch seleccionado anterior y comprobando la coherencia antes de ejecutar. No hay cambios de código del proyecto ni de flags ni Benchmark adicional. Este directorio queda preservado.
