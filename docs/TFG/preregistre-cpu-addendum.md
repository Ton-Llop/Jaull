# Addendum al preregistre de la campanya CPU: passada B

Escrit el 2026-11-25 **després de veure els resultats de la passada A** (de les 11:29 a les 11:53) i **abans de fer la passada B**.
Les prediccions de `preregistre-cpu.md` (commit `1ad8027`) no canvien.

## Errata de la passada A

El preregistre deia que els casos es mesuraven amb `--no-mmap`. llama.cpp b11889 no té aquest flag: el mode de càrrega és `-lm`.
S'han fet servir `-lm none` (a RAM, sense mmap) i `-lm mmap`. El canvi es va fer abans de mesurar cap cas de la campanya,
quan la prova de fum va fallar amb `invalid argument: --no-mmap`. Les proves de fum no són dades de la campanya.

## Per què una passada B

A la passada A hi havia entre 10,0 i 11,5 GiB de RAM disponibles. Jaull va aprovar 23 de les 24 configuracions sense mmap i només en va rebutjar una (C22),
de manera que gairebé no hi ha aprovacions properes al límit ni rebutjos per comprovar.

## Què canvia i què no

- **Només canvia la RAM disponible.** L'objectiu és entre 8.000 i 8.800 MiB just abans de cada cas, amb aplicacions obertes i WSL en marxa.
  Es mesura abans de cada cas, com a la passada A, i el veredicte de Jaull es calcula amb aquest valor.
- **No canvia res més:** els casos i les prediccions, la regla de veredicte (total predit com a molt la RAM disponible mesurada), els criteris (executa i marge),
  els llindars de seguretat (400 MiB de RAM disponible i 3.000 MiB de creixement del fitxer de paginació), els models, els fils i els scripts.
- **No es torna a mesurar la velocitat.**

## Com es presenta

Les dues passades es presenten per separat, cada cas amb la seva RAM disponible mesurada, i no es barregen en una sola taula de confusió.
Els casos que es parin per seguretat són **no verificats**, no errors ni encerts.

## Què es vol veure

Si, amb la RAM més ajustada, Jaull rebutja configuracions que funcionen amb marge (falsos negatius) i si n'aprova alguna que no funciona (falsos positius).
Amb aquesta RAM s'espera que els Qwen2.5 de 7B quedin aprovats i que el 14B, l'Ornith i el Mellum Q4 quedin rebutjats, però això s'escriu aquí i no es dona per sabut.
