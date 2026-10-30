# B60: comparativa de rendiment, WSL2 + Dozen contra Vulkan natiu a Windows

Resultats mesurats a la màquina de Biosfer (Intel Arc Pro B60, 24 GiB) entre el 27 i el 30 d'octubre de 2026.
Aquest document recull només el que s'ha mesurat o observat. Cada afirmació du la seva paraula d'estat:

- **VALIDATED**: mesurat en una execució real, amb la comanda indicada.
- **UNVERIFIED**: hipòtesi coherent amb les dades, però no provada.
- **PENDENT**: prova encara no feta.

## 1. Entorns comparats

| | WSL2 + Dozen | Windows, Vulkan natiu |
|---|---|---|
| Binari | `llama.cpp` b11889 (`d854d98ba`), compilat a WSL2 | `llama-b11889-bin-win-vulkan-x64.zip`, el mateix build |
| Camí gràfic | Vulkan → Mesa Dozen → Direct3D 12 → driver de Windows | Vulkan → driver Vulkan d'Intel |
| Conformitat | Mesa avisa `dzn is not a conformant Vulkan implementation, testing use only` | sense avís |
| `matrix cores` | `none` | `KHR_coopmat` |
| `warp size` | 16 | 32 |
| Dispositiu de la B60 | `Vulkan0` (la iGPU AMD és `Vulkan1`) | `Vulkan1` (la iGPU AMD és `Vulkan0`) |
| Driver d'Intel | `32.0.101.8976` (`DriverVer=08/28/2026`), el mateix a tots dos | el mateix |

L'ordre dels dispositius s'inverteix entre els dos entorns. Qui compari mesures ha de comprovar quin índex és la B60 amb `--list-devices` i no assumir-lo.

## 2. Resultats

### 2.1. Ornith 1.5 9B Q8_0 (`qwen35`, 9,10 GiB)

Mateix fitxer, mateixa targeta, mateix build.

| Entorn | `pp` (prompt) | `tg` (generació) | Notes |
|---|---:|---:|---|
| WSL2 + Dozen, `llama-bench` a mà, `-fa 0`, `-p 512 -n 128 -r 5` | 141,73 ± 4,21 t/s | 4,10 ± 0,59 t/s | `-dev Vulkan0 -ngl 99` |
| WSL2 + Dozen, benchmark de Jaull | 153,3 ± 2,8 t/s | 4,8 ± 0,0 t/s | `--flash-attn=off`, `--ctx-size=4096`, 5 repeticions |
| **Windows nadiu**, `llama-bench.exe`, `-p 128 -n 64 -r 2` | **975,84 ± 23,83 t/s** | **36,17 ± 0,04 t/s** | `-dev Vulkan1 -ngl 99`, flash attention en automàtic |

**VALIDATED.** A Windows, la mateixa targeta amb el mateix model és unes **6,4–6,9 vegades més ràpida en prompt** i unes **7,5–8,8 vegades més ràpida en generació**, segons es comparin amb la mesura de Jaull o amb la manual.

Dues precisions que limiten la comparació:

1. WSL2 va amb flash attention desactivada i Windows amb el valor per defecte (automàtic). Una diferència de vuit vegades no es pot explicar per això, però la mesura estricta (`-fa off` a Windows) és **PENDENT**.
2. Els paràmetres `-p`/`-n`/`-r` no són idèntics (512/128/5 a WSL2, 128/64/2 a Windows).

La mesura de Jaull (4,8 t/s) i la manual (4,10 t/s) són del mateix ordre. **VALIDATED**: Jaull mesura correctament i el llançador no és la causa de la lentitud.

### 2.2. Qwen3-Coder-30B-A3B-Instruct Q4_K_M (`qwen3moe`, 17,28 GiB, MoE amb ~3B actius)

| Entorn | `pp` | `tg` | Notes |
|---|---:|---:|---|
| WSL2 + Dozen, `llama-bench`, `-fa off`, `-p 128 -n 64 -r 2` | 34,07 ± 26,83 t/s | 2,51 ± 0,13 t/s | la gran variància del `pp` indica escalfament de shaders |
| WSL2 + Dozen, benchmark de Jaull | no registrat | 2,2 t/s | |
| WSL2 + Dozen, interfície web de `llama-server` | 6,02 t/s (313 tokens, 51 s) | 1,65 t/s (36 tokens, 21 s) | primera petició, resposta curta; només orientatiu |
| **Windows natiu**, `llama-bench.exe`, `-dev Vulkan1 -ngl 99 -p 128 -n 64 -r 2` | **788,78 ± 6,20 t/s** | **99,18 ± 0,10 t/s** | flash attention en automàtic |
| **Windows natiu**, interfície web de `llama-server.exe` | no registrat | 75,08 t/s (159 tokens, 2,1 s) | resposta curta, una sola mesura; només orientatiu |

**VALIDATED** a WSL2: el model carrega i genera sense caure, però molt lentament. Un MoE amb ~3B de paràmetres actius va més lent que el 9B dens (2,5 contra 4,1 t/s). Si el límit fos l'ample de banda de memòria, aniria més ràpid.

**VALIDATED** a Windows: el mateix fitxer de 17,28 GiB fa **99,18 t/s** de generació, unes **40 vegades més** que a WSL2 (2,51 t/s) i unes 45 vegades més que la mesura de Jaull (2,2 t/s). El `pp` és unes 23 vegades més alt, però la mesura de WSL2 té una variància molt gran (34,07 ± 26,83) i la ràtio és poc fiable.

A Windows l'ordre s'inverteix: el MoE (99 t/s) és unes 2,7 vegades més ràpid que el 9B dens (36 t/s), com s'esperaria d'un entorn sa. **UNVERIFIED**: que a WSL2 el límit fos el cost per operació o per enviament al driver (un MoE fa moltes operacions per token). L'inversió d'ordre ho fa compatible, però no ho demostra.

### 2.3. Qwen3.8-27B (`qwen35`, 27,32 B)

| Entorn | Resultat |
|---|---|
| WSL2 + Dozen, `UD-Q4_K_M` (15,32 GiB), `-ngl 99 -fa 0` | `pp128` 49,87 t/s, després `Segmentation fault` en passar a generació |
| WSL2 + Dozen, `UD-IQ1_S` (5,8 GiB), Run de Jaull | `exit code -11` durant l'escalfament |
| WSL2 + Dozen, `UD-IQ1_S`, `llama-bench -ngl 0` | `pp8` 1,99 t/s, `tg4` 0,77 t/s (tot a CPU) |
| WSL2 + Dozen, `UD-IQ1_S`, `-ngl 1` | `pp8` 2,28 t/s, `tg4` 0,71 t/s (sembla que només la capa de sortida a GPU; no verificat) |
| WSL2 + Dozen, `UD-IQ1_S`, `-ngl 2` a `-ngl 99` | `Segmentation fault` sempre |
| **Windows natiu**, `UD-Q4_K_M`, `llama-bench.exe`, `-dev Vulkan1 -ngl 99 -p 128 -n 64 -r 2` | `pp128` **339,37 ± 0,24 t/s**, `tg64` **15,55 ± 0,01 t/s**, **sense cap crash** |

**VALIDATED**: el mateix fitxer `UD-Q4_K_M` que cau a WSL2 + Dozen corre sense problemes a Windows, amb el mateix build i el mateix driver d'Intel. El crash és, per tant, propi del camí WSL2 + Dozen i no del model ni de la targeta. A WSL2 no hi ha mesura de generació per comparar, perquè cau abans. El `pp128` va de 49,87 a 339,37 t/s (unes 6,8 vegades més, sobre una sola repetició a WSL2).

**VALIDATED** a WSL2: un sol bloc del cos del model a la GPU basta perquè caigui. No depèn de la quantificació (`Q4_K_M` i `IQ1_S` cauen igual), ni de la memòria (la B60 mostrava 23 613 MiB lliures), ni de la quantitat de capes.

## 3. El crash del 27B a WSL2

Els `Segmentation fault` s'han reproduït en quatre ocasions (`llama-bench` i `llama-cli`, quantificacions diferents). El registre del nucli (`dmesg | grep -i segfault`) mostra sempre el mateix punt:

```
llama-bench[...]: segfault at ...7473613b ... error 4 in libLLVM-14.so[a974d4,...]
llama-cli[...]:   segfault at ...7473613b ... error 4 in libLLVM-14.so[a974d4,...]
```

- Mòdul: `libLLVM-14.so`. L'ordre `LD_DEBUG=libs` mostra que es carrega des de `/usr/lib/wsl/drivers/iigd_dch_d.inf_amd64_eebe517ae5149ae4/`, és a dir, **del driver gràfic d'Intel per a Windows exposat a WSL**. No és de la instal·lació de Mesa ni del LLVM del sistema.
- Offset de la instrucció: `a974d4`, idèntic en totes les caigudes.
- L'adreça que falla acaba en `7473613b`, bytes que es llegeixen com a text ASCII (`;ast`), com si una cadena es fes servir com a punter.

**VALIDATED**: el crash és determinista i cau sempre a la mateixa instrucció del compilador del driver.
**UNVERIFIED**: quin shader o forma de tensor el provoca, i per què el 9B de la mateixa família `qwen35` no el dispara. Una causa hipotètica és que depengui de les dimensions del model.

Variables d'entorn de llama.cpp provades sense cap efecte sobre el crash (cap va evitar-lo): `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`, `GGML_VK_DISABLE_MMVQ`, `GGML_VK_DISABLE_FUSION`, `GGML_VK_DISABLE_MULTI_ADD`, `GGML_VK_DISABLE_DOT2`, `GGML_VK_DISABLE_F16`, `GGML_VK_DISABLE_GRAPH_OPTIMIZE`.

### Un segon crash, aquest evitable

Sense `-fa off`, `llama-bench` cau en passar a generació amb:

```
ggml_vulkan: compute pipeline creation failed for xe_fa_decode_ph1: vk::Device::createComputePipeline: ErrorUnknown
```

El shader `xe_fa_decode_ph1` és el de flash attention per a decodificació. Jaull ja llança sempre `--flash-attn=off` en aquest camí, i per això no hi cau. **VALIDATED** (reproduït amb Ornith 9B Q8_0).

## 4. Referències externes

Tres articles que no hem verificat. Els resums els ha fet un model petit i alguns números no quadren, com ara una taula amb mides que semblen intercanviades. S'han de llegir com a orientació.

| Font | Què diu | Estat |
|---|---|---|
| `srmiles.com/articles/local-llm-on-b60` | llama.cpp amb SYCL: Ornith 9B ~1.200 t/s de prompt i ~50 t/s de generació | UNVERIFIED |
| `localaifrontier.com/blog/openvino-beats-vulkan-on-arc-b60/` | Vulkan sobre Mesa ANV, Qwen3-Coder-30B-A3B Q4_K_M: 38,6 t/s de generació i 412,3 t/s de prompt; OpenVINO amb un altre model, 67,95 t/s (els autors diuen que la comparació és orientativa) | UNVERIFIED |
| `dirtysouthalpha.com/blog/intel-arc-30b` | Vulkan amb el driver Xe de Mesa: Qwen3-Coder-30B-A3B IQ4_XS ~40 t/s, Qwen3.8-27B IQ4_XS 4,4 t/s | UNVERIFIED |

La nostra mesura a Windows és del mateix ordre que les d'aquests articles per a Ornith 9B (36 t/s) i **supera** les que citen per al Qwen3-Coder-30B-A3B (99 t/s contra ~39–40 t/s). No sabem per què. Les quantificacions i els protocols no són els mateixos (alguns articles fan servir `IQ4_XS` i no `Q4_K_M`), i els resums poden contenir errors. No en traiem cap conclusió.

L'article de `dirtysouthalpha.com` cita el 27B dens a 4,4 t/s (amb `IQ4_XS`). La nostra mesura a Windows amb `Q4_K_M` és de 15,55 t/s, unes 3,5 vegades més. No sabem per què difereix: quantificació i protocol no coincideixen.

## 5. Conclusions

1. **VALIDATED**: a WSL2 + Dozen, la B60 rendeix molt menys que a Windows amb Vulkan natiu, amb el mateix build, el mateix model i el mateix driver d'Intel: unes 8 vegades menys en generació amb Ornith 9B (4,1–4,8 contra 36,2 t/s) i unes **40 vegades menys** amb Qwen3-Coder-30B-A3B (2,51 contra 99,18 t/s). La targeta no és el coll d'ampolla.
2. **VALIDATED**: Jaull mesura bé. Les seves xifres coincideixen amb les de `llama-bench` a mà.
3. **VALIDATED**: el Qwen3.8-27B cau a WSL2 + Dozen amb una instrucció concreta del compilador de shaders del driver d'Intel, i **corre sense crash a Windows** amb el mateix build, el mateix fitxer i el mateix driver (339,37 t/s de prompt, 15,55 t/s de generació).
4. **UNVERIFIED**: que aquesta diferència s'expliqui per l'absència de `matrix cores` a Dozen, per l'ús de Direct3D 12 com a capa de traducció, o per ambdues coses. Les dades són compatibles amb això, però no ho separen.
5. **UNVERIFIED**: per què exactament cau el 27B a WSL2 + Dozen i no ho fa el 9B de la mateixa família. Sabem que el camí és la causa (funciona a Windows), però no quin shader ni quina forma de tensor ho provoca.

## 6. Proves pendents

- [x] Qwen3-Coder-30B-A3B Q4_K_M a Windows, `-dev Vulkan1 -ngl 99 -p 128 -n 64 -r 2`: fet, 788,78 / 99,18 t/s.
- [x] Qwen3.8-27B Q4_K_M a Windows, mateixes opcions: fet, corre sense crash (339,37 / 15,55 t/s).
- [ ] Ornith 9B Q8_0 a Windows amb `-fa off`, per tenir una comparació estricta amb WSL2.
- [ ] Comanda `dmesg | grep -i segfault` de qualsevol nova caiguda, per comprovar que continua a `a974d4`.

## 7. Decisió d'abast que això reobre

La nota del projecte deia que el pont WSL2 → Windows quedava sense implementar fins que es demostrés que Vulkan natiu a WSL2 no funciona bé. Aquestes dades hi apunten: funciona, però és molt més lent i cau amb alguns models. Reconsiderar-ho és una **decisió de l'autor**, i no s'ha implementat res.

## 8. Com reproduir-ho

WSL2 (binari de Linux, variables de Mesa Dozen ja exportades):

```
~/llama.cpp/build-vulkan/bin/llama-bench -m <model.gguf> -dev Vulkan0 -ngl 99 -fa off -p 512 -n 128 -r 5
```

Windows (llançat des de WSL2 amb interoperabilitat, binari natiu):

```
./llama-bench.exe --list-devices          # comprovar quin índex és la B60
./llama-bench.exe -m 'C:\llama-win\<model>.gguf' -dev Vulkan1 -ngl 99 -p 128 -n 64 -r 2
```

Per veure la interfície web de `llama-server.exe` des d'un altre ordinador, quan WSL2 no pot arribar a Windows (el cortafuegs descarta la connexió de WSL cap al port de Windows), s'ha fet servir un túnel invers amb el client SSH de Windows. El servidor escolta només a `127.0.0.1`, i `ssh.exe -N -R 8081:127.0.0.1:8080 -p 2222 <usuari>@<ip-del-pc>` obre un port a WSL que apunta al servidor. Després, des de l'ordinador remot, `ssh -N -L 8080:127.0.0.1:8081 -p 2222 <usuari>@<ip-del-pc>` i `http://localhost:8080`. No s'ha tocat el cortafuegs.

El binari de Windows es descarrega de la versió `b11889` de `ggml-org/llama.cpp`. L'etiqueta `latest` no té binaris (només un `nightly-tag.txt`), per això cal fer servir l'etiqueta del build.
