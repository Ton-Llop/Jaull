# Preregistre: sis casos a prop del límit a la B60

**Esborrany del 2026-11-24. Escrit abans de mesurar res.** Cal fer-ne commit abans d'arrencar cap prova: el
commit és la prova que les prediccions són anteriors a la mesura. Després de la primera mesura, aquest fitxer no
es modifica; si cal corregir-lo, s'afegeix una secció nova al final amb data.

Codi de Jaull: `314f56e` (2026-11-21). Les fórmules no es toquen durant aquestes proves.

## Pregunta

A prop del límit de la targeta, Jaull encerta més que una regla senzilla? La taula de
`prediccio-contra-realitat.md` no ho pot dir: té nou casos, gairebé tots amb molt de marge, i els valors «ara» es van
ajustar sobre aquestes mateixes mesures. Aquests sis casos són en contextos que no s'han mesurat mai.

## Maquinari i programari

- Intel Arc Pro B60, 24 GiB, **23.613 MiB lliures** (valor de les mesures anteriors). Es torna a llegir abans de
  cada cas. Si difereix en més de 200 MiB, es recalculen les prediccions i es deixa escrit.
- Windows natiu, llama.cpp b11889, Vulkan.
- Mateixa comanda que a B6 (`-ngl 99 -c N`, una persona). No s'afegeix cap flag nou.

## Definició de «viable» (tres nivells)

1. **Regla de llama.cpp**: queden almenys 1.024 MiB lliures després de carregar (el missatge `will leave N >= 1024 MiB`).
2. **Carrega** amb `-ngl 99`, encara que no es compleixi el nivell 1.
3. **Genera** 512 tokens sense error de memòria.

La predicció de Jaull es compara amb el **nivell 1**. Els nivells 2 i 3 s'anoten, però no decideixen l'encert.

## Mètode A (el senzill)

Escrit abans de mesurar i sense ajustar-lo després:

> Cap si `mida del fitxer × 1,2 ≤ VRAM lliure`. No depèn del context.

## Prediccions

Fórmula de Jaull actual (**CALCULAT** a mà, com a D120 i D123, no amb l'estimador complet):
pesos (mida del fitxer) + KV + 256 MiB + 1.024 MiB. KV lineal amb el context: Coder 30B 0,09375 MiB per token
(3.072 MiB a 32K, mesurat a D120); Qwen2.5 32B 0,25 MiB per token (1,0 GiB a 4096, mesurat a D123). Pesos: 17.695 MiB
(Coder 30B, 17,28 GiB) i 18.934 MiB (32B, 18,49 GiB).

La columna «llama.cpp, derivat» és una extrapolació de les mesures B2 i B6 (lliures 2.901 MiB a 32K i 791 MiB a 16K),
suposant que el buffer de còmput és constant. **No és una predicció de Jaull.** Es posa perquè és el que el
resultat probablement diu. El còmput creix amb el context (a Qwen3.6, de 108 a 332 MiB), així que la xifra real pot
quedar una mica per sota.

| # | Model i context | Jaull: necessita | Jaull: marge sobre 23.613 | Jaull: cap? | Mètode A | llama.cpp, derivat: lliures |
|---|---|---:|---:|---|---|---:|
| P1 | Coder 30B, 49.152 | 23.583 MiB | +30 MiB | **Sí, a la vora** | Sí | 1.362 MiB (sí) |
| P2 | Coder 30B, 51.200 | 23.775 MiB | −162 MiB | **No** | Sí | 1.170 MiB (sí) |
| P3 | Coder 30B, 53.248 | 23.967 MiB | −354 MiB | **No** | Sí | 978 MiB (no, per poc) |
| P4 | Qwen2.5 32B, 12.288 | 23.286 MiB | +327 MiB | **Sí** | Sí | 1.812 MiB (sí) |
| P5 | Qwen2.5 32B, 14.336 | 23.798 MiB | −185 MiB | **No** | Sí | 1.300 MiB (sí) |
| P6 | Qwen2.5 32B, 15.360 | 24.054 MiB | −441 MiB | **No** | Sí | 1.044 MiB (sí, per poc) |

**Què s'espera que passi.** El mètode A diu «cap» a tots, i per tant no distingeix cap context. Jaull té el límit
una mica abans del real, perquè deixa 1 GiB de marge i hi suma uns 1,3 a 2,6 GiB de biaix conservador (vegeu
`prediccio-contra-realitat.md`). Els casos que decideixen són **P2, P5 i P6**: Jaull diu «no» i la regla de llama.cpp
probablement diu «sí». Si és així, el cost de Jaull són contextos que es podrien usar i es desaconsellen. **P3** és el
que ha de sortir «no» per als dos.

## Com es compta

- **Encert** = la predicció de Jaull (cap/no cap) coincideix amb el nivell 1 mesurat. S'expressa en **casos**, no en
  percentatge: sis casos no donen percentatges.
- **Fals positiu** = Jaull diu «cap» i el nivell 1 no es compleix. **Fals negatiu** = Jaull diu «no» i es compleix.
- Una execució per cas, **sense repeticions**. La variància queda com a **UNVERIFIED**; si un resultat cau a menys
  de 100 MiB del llindar, es repeteix una vegada i s'anoten les dues.
- Cada cas guarda el log de llama.cpp sencer, la VRAM lliure abans de carregar i els tok/s.
- Amb la VRAM lliure inicial diferent de 23.613 MiB, el cas es recalcula i s'anota, no es descarta.

## Què no es pot dir amb això

- Que Jaull sigui millor que els mètodes que no es comparen aquí (només s'hi compara el mètode A).
- Res sobre altres targetes, altres runtimes o durades llargues (això és el nivell 3 d'una hora, que és un altre experiment).
- Que els marges siguin estables: el valor de VRAM lliure depèn de l'escriptori de Windows en aquell moment.

## Mitjà per executar-ho

Vegeu les instruccions del dia de la mesura. Les còpies dels models a Windows es van verificar amb sha256 el 2026-11-24
(Coder 30B fet; 32B i 27B pendents en escriure això).
