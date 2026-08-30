# Validación de Jaull contra llama.cpp — Qwen2.5-7B en RTX 2060

## Qué se probó

| | |
|---|---|
| Modelo | Qwen2.5-7B-Instruct, Q4_K_M, 28 transformer blocks |
| GPU | NVIDIA RTX 2060, 6 GiB (4754.4 MiB libres al medir) |
| llama.cpp | build `b10357`, commit `689e227db` |
| Contextos | 2048, 4096, 8192 |
| Offload | `-ngl 18`, `-ngl 23`, `-ngl 29` |

Objetivo: comparar lo que **estima** Jaull con lo que **hace** llama.cpp.

> Aviso de unidades: los `transformer_blocks` del HardwareFitAnalyzer y el
> `--n-gpu-layers` de llama.cpp **no son la misma cosa**. Cuando abajo se lee
> `-ngl 18`, es un valor de sonda elegido a partir del número del HFA, no una
> traducción oficial entre ambos.

---

## 1. Jaull da tres números distintos

Para `ctx = 4096`:

```
HardwareFitAnalyzer   18 / 28 blocks     lo que se muestra en la tarjeta
_pick_layers()        -ngl 23            lo que se ejecuta de verdad
llama.cpp             -ngl 29            el máximo que hemos validado
```

Son dos políticas internas separadas. El HFA planifica capacidad; `_pick_layers`
decide el flag. Hoy no comparten fórmula, así que dan resultados distintos.

Por contexto:

| ctx | HFA | `_pick_layers` | llama.cpp real |
|---:|---:|---:|---|
| 2048 | 19 blocks | `-ngl 24` | `-ngl 29` |
| 4096 | 18 blocks | `-ngl 23` | `-ngl 29` |
| 8192 | 17 blocks | `-ngl 22` | `-ngl 29`, al límite |

Jaull acierta la **dirección** (más contexto, menos offload) pero se queda muy
corto en la cantidad.

### Por qué el HFA elige 18 y no 19

| | 18 blocks | 19 blocks |
|---|---:|---:|
| VRAM estimada | 4663.7 MiB | 4875.0 MiB |
| VRAM disponible | 4754.4 MiB | 4754.4 MiB |
| Resultado | cabe, sobran 90.8 | no cabe, faltan 120.5 |

---

## 2. Datos medidos

| ctx | `-ngl` | offload | CUDA model | GPU KV | CPU KV | KV total | CUDA compute | tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0 | 0/29 | 0 | 0 | 224 | 224 | 183.44 | 5.55 |
| 2048 | 18 | 18/29 | 2706.70 | 68 | 44 | 112 | 182.33 | — |
| 4096 | 18 | 18/29 | 2706.70 | 136 | 88 | 224 | 183.44 | 12.82 |
| 8192 | 18 | 18/29 | 2706.70 | 272 | 176 | 448 | 198.31 | — |
| 4096 | 23 | 23/29 | 3349.15 | 176 | 48 | 224 | 183.44 | 19.85 |
| 2048 | 29 | 29/29 | 4168.09 | 112 | 0 | 112 | 134.01 | — |
| 4096 | 29 | 29/29 | 4168.09 | 224 | 0 | 224 | 136.01 | 55.96 |
| 8192 | 29 | 29/29 | 4168.09 | 448 | 0 | 448 | 140.01 | — |

Todo en MiB. El rendimiento solo se midió a `ctx = 4096`.

Dos cosas que se ven de inmediato:

- **Los pesos no dependen del contexto.** A `-ngl 18`, `CUDA model` es 2706.70
  en los tres contextos. El experimento está bien aislado.
- **Ni con full offload se vacía la RAM.** A `-ngl 29` quedan 292.36 MiB del
  modelo mapeados en CPU (la tabla de embeddings). `29/29` no significa "todo
  en la GPU".

---

## 3. El KV sigue una regla exacta

```
KV por bloque       = KV total / 28
bloques KV en GPU   = ngl - 1
```

Ejemplo, `ctx = 4096` (KV total 224 MiB, o sea 8 MiB por bloque):

| `-ngl` | bloques en GPU | GPU KV | CPU KV |
|---:|---:|---:|---:|
| 15 | 14 | 112 | 112 |
| 18 | 17 | 136 | 88 |
| 19 | 18 | 144 | 80 |
| 21 | 20 | 160 | 64 |
| 23 | 22 | 176 | 48 |
| 29 | 28 | 224 | 0 |

Se cumple **sin error en las 10 ejecuciones** con offload (6 a ctx 4096, 2 a
ctx 2048, 2 a ctx 8192).

El `-1` no es casualidad: llama.cpp expone `n_layer + 1` unidades offloadables,
porque la última es la capa de salida. Por eso hacen falta `-ngl 29` para los
28 bloques.

**Aquí estaba el error del HFA:** cargaba el KV entero contra la VRAM aunque el
offload fuese parcial, así que reservaba 224 MiB para los 18 bloques que planifica.

Para contrastarlo hay que comparar en la misma unidad. 18 bloques del HFA
equivalen a `-ngl 19` en este modelo y build, no a `-ngl 18`:

| | KV en GPU |
|---|---:|
| HFA antes (18 bloques) | 224 MiB |
| HFA ahora (18 bloques) | **144 MiB** |
| llama.cpp `-ngl 19` (18 bloques) | **144 MiB** |

Con el reparto proporcional el HFA acierta el valor exacto que reporta
llama.cpp para la colocación equivalente.

---

## 4. Arreglar el KV no mueve la frontera

Comprobado con el candidato 19, `ctx = 4096`:

| | VRAM estimada | ¿cabe en 4754.4? |
|---|---:|---|
| HFA antes | 4875.0 MiB | no, por 120.5 |
| HFA con KV proporcional | 4779.6 MiB | no, por 25.2 |

Cierra el 79 % del hueco, pero el bloque 19 **sigue sin entrar**. El HFA
seguiría eligiendo 18.

> El placement del KV hay que arreglarlo porque está mal, no porque desbloquee
> capas. No desbloquea ninguna.

El KV **sí** pasa a contar en el paso de un candidato al siguiente, cosa que
antes no hacía. De 18 a 19 bloques el coste sube 221.91 MiB:

| término | Δ 18 → 19 |
|---|---:|
| pesos estimados del bloque | +159.50 MiB |
| **su parte del KV** | **+8.00 MiB** |
| overhead reasignado | +34.24 MiB |
| margen reasignado | +20.17 MiB |
| reserve | +0.00 MiB |
| **total** | **+221.91 MiB** |

Y encaja con los dos extremos: sobraban 196.76 MiB en 18 y faltan 25.15 en 19,
que suman exactamente esos 221.91.

---

## 5. Cuánto cuesta ser conservador

A `ctx = 4096`:

| `-ngl` | bloques | qué es | tok/s | % de la mejora total |
|---:|---:|---|---:|---:|
| 0 | 0 | solo CPU | 5.55 | — |
| 18 | 17 | | 12.82 | 0 % |
| 19 | 18 | **equivale al plan del HFA** | 13.75 | 2 % |
| 21 | 20 | | 15.87 | 7 % |
| 23 | 22 | **lo que Jaull ejecuta** | 19.85 | 16 % |
| 29 | 28 | full offload | 55.96 | 100 % |

`-ngl 29` es **2.8× más rápido** que el `-ngl 23` que Jaull lanza hoy.

Y la ganancia **no es lineal**. De `-ngl 19` a `-ngl 23` recorres casi la mitad
de las capas que faltaban y te llevas solo el 14 % de la mejora. Los últimos 6
slots dan el 84 %.

Tiene sentido: mientras quede una capa en CPU, cada token paga un viaje
host↔GPU. Eso solo desaparece con residencia completa.

**Consecuencia para Jaull:** quedarse "casi" no sirve de casi nada.

---

## 6. El compute buffer no depende de los pesos

Este es el resultado más limpio, porque solo cambió una variable:

```
ctx 4096, -ngl 18  →  CUDA model 2706.70   compute 183.44
ctx 4096, -ngl 23  →  CUDA model 3349.15   compute 183.44
```

**642 MiB más de pesos en la GPU, y el compute buffer no se movió ni un byte.**
Con full offload incluso baja, a 136.01.

Con el contexto se mueve muy poco. Multiplicándolo por 4:

```
-ngl 18:  182.33 → 198.31   (+8.8 %)
-ngl 29:  134.01 → 140.01   (+4.5 %)
```

En las 8 ejecuciones el compute buffer nunca pasó de 198.31 MiB.

El HFA usa hoy `overhead = 512 MiB + 10 % de los pesos`. Para este modelo esa
fracción sola añade unos 446 MiB. Los datos no apoyan esa forma.

Cuidado: esto **no** demuestra que el overhead del HFA deba ser igual al compute
buffer. El HFA intenta cubrir más cosas. Lo que demuestra es que la variable
elegida (los pesos) no es la que manda.

---

## 7. Los dos caminos son conservadores por motivos distintos

Importante, porque se arreglan de forma diferente.

**El runtime (23) falla por una sola constante.** `_pick_layers` supone que cada
capa pesa `pesos / 28` = 159.50 MiB. Lo medido es 132.85 MiB.

```
presupuesto para pesos        3762.4 MiB
con 159.50 MiB/capa   →  n = 23
con 132.85 MiB/capa   →  n = 28
```

La causa es que reparte **todos** los pesos entre 28 bloques, incluidos los
292.36 MiB de embeddings que nunca salen del host.

*(Aun corrigiéndolo se quedaría en 28, porque `_pick_layers` topa en el número
de bloques y llama.cpp necesita 29. Es un segundo fallo, independiente.)*

**El HFA (18) falla por otra cosa:** overhead y safety margin. A 18 bloques
carga 632.6 MiB de overhead y 424.0 de margen sobre la GPU, más 512 de reserve.
Son 1568 MiB de los 4754 disponibles — un tercio de la tarjeta, sin
contrapartida física medible.

---

## 8. A ctx 8192 hay un techo real

```
CUDA model   4168.09
KV            448.00
compute       140.01
             --------
total        4756.10 MiB   vs   4754.43 MiB disponibles
```

La diferencia son ~1.7 MiB. El run arrancó, así que la suma no debe leerse como
una medición exacta del uso instantáneo: los buffers se reportan en momentos
distintos y hay redondeos.

Pero la conclusión práctica es clara: **a ctx 8192, el full offload está justo
en el límite de esta tarjeta.**

Eso importa porque demuestra que el techo existe. Jaull no es conservador
"porque sí" — solo lo es de más.

---

## 9. Qué no podemos concluir todavía

Todo esto es **una** GPU, **un** modelo y **un** build. Aún no podemos decir que:

- `bloques KV en GPU = ngl - 1` valga para cualquier modelo o runtime;
- el overhead del HFA deba ser el compute buffer de llama.cpp;
- el safety margin esté mal;
- el device reserve esté mal;
- Jaull deba recomendar siempre el máximo `-ngl` que arranque.

Falta repetirlo con otros modelos y, más adelante, con otro hardware.

---

## 10. Siguiente paso

1. **Corregir el placement del KV.** Es lo único de lo que tenemos una regla
   exacta y evidencia directa de que el modelo está mal. Sabiendo que no va a
   cambiar el 18.
2. **Revisar el `runtime_overhead`.** Es el término que más desplaza la frontera
   y ya sabemos que la variable que usa no es la correcta.
3. **Unificar los dos caminos**, para que el número que se muestra y el que se
   ejecuta salgan del mismo cálculo. Ojo: unificar hoy, sin arreglar antes lo
   anterior, haría a Jaull más lento (bajaría de 23 a 19), no más rápido.

---

## En una frase

Jaull calcula muy bien **cuánta** memoria hace falta, pero simplifica demasiado
**dónde** acaba, y eso le cuesta 2.8× de rendimiento en este caso.
