# DeepSeek-V4-Flash-0731 — port en curso

**DeepSeek-V4-Flash** (284B / 13B activos, contexto de 1M) como familia hermana
de colibrì. El motor **corre y genera texto**:

```
$ ./deepseek_v4 /modelos/DeepSeek-V4-Flash-0731 "La capital de Francia es" 6
config: 43 capas, dim 4096, 64 cabezas x 512, 256 expertos top-6
expertos: streaming con cache de 384 slots de 11008 totales
cargado en 7.5 s

La capital de Francia es una de las ciudades más visitadas

12 tokens en 76.8 s (0.16 tok/s)
expertos: 1300 hits / 1796 miss, 24.01 GB leidos
```

En un Ryzen 5700U (Zen 2, **sin AVX-512**) con 17,4 GB de RAM libre y el
checkpoint en un NVMe. 0,16 tok/s es lento, y se explica: ~2 GB de expertos
leídos por token y la ruta SIMD de `quant.h` cayendo a AVX2.

> **Sigue siendo trabajo en curso.** El forward está entero y validado pieza a
> pieza contra la implementación de referencia de DeepSeek, pero falta la pila
> MTP/DSpark, el enganche con `coli`, y el streaming real de colibrì (pool de
> hilos, PILOT, batch-union, `O_DIRECT`, dual-SSD) en lugar de la caché LRU
> mínima que trae `deepseek_v4.c`. Lista completa al final.

## Por qué encaja bien

Medido sobre el checkpoint real, no estimado:

| | GLM-5.2 | DeepSeek-V4-Flash |
|---|---|---|
| Disco | 372 GB | **156 GiB** |
| Residente | 17B → 9,9 GB en int4 | **8,67 GiB en precisión nativa** |
| Tráfico de expertos por token | ~11,4 GB | **~3,45 GB** |
| Expertos rutados | 256, top-8 | 256, **top-6** |

Los expertos son el 97,5 % del modelo (277B de 284B) y vienen en **MXFP4
nativo**: e2m1 empaquetado 2 nibbles por byte con una escala ue8m0 por cada 32
valores. Es exactamente lo que consume `matmul_mxfp4` para Kimi K3, así que se
streamean de los shards de Hugging Face **sin conversión**.

## Los dos parches a colibrì

Sin ellos el checkpoint no se puede abrir. Son pequeños y de utilidad general.

### 1. `st.h` — dtypes que DeepSeek etiqueta explícitamente

`st_dtype_code` sólo conocía BF16/F16/F32/U8/I8. DeepSeek etiqueta los bytes
cuantizados con su dtype real en vez de U8:

| dtype | tensores | qué es |
|---|---|---|
| `F8_E8M0` | 35.718 | las escalas MX |
| `F8_E4M3` | 390 | el conjunto denso |
| `I64` | 3 | las tablas `tid2eid` del routing hash |

Los tres se añaden como bytes crudos —que es lo que son— más un tamaño de
elemento propio para los enteros. `st_read_raw` ya era agnóstico al dtype.

Detalle: `model.py` declara `tid2eid` como `int32`, pero el checkpoint la guarda
en **I64**.

### 2. `tok.h` — formato de merges y familia de pre-tokenizer

**Merges.** `tok.h` aceptaba sólo el formato nuevo de `tokenizers` (arrays
`["izq","der"]`); DeepSeek usa el antiguo (cadena `"izq der"`) y la carga
fallaba con `malformed merge entry 0`. Partir por el primer espacio literal es
seguro: en byte-level los espacios del texto van como U+0120, nunca como 0x20.

**Pre-tokenizer.** DeepSeek no resuelve el pre-tokenizer en una alternación como
cl100k/o200k/kimi: encadena un `Sequence` de tres `Split` con
`behavior=Isolated`, y la **primera aísla los dígitos**. El orden no es
equivalente a meterlo todo en un regex. Caso mínimo que lo destapa:

```
"\  0"    DeepSeek -> "\", "  ", "0"        (doble espacio, UN token)
          cl100k   -> "\", " ", " ", "0"
```

Al separar el dígito antes, la racha de espacios que lo precede se queda sin
nada detrás dentro de su trozo y se agrupa entera. `pretok_chunk_dsv4`
implementa las etapas 0 (dígitos en grupos de ≤3) y 1 (han + kana) como
pre-paso y delega los huecos en `pretok_chunk`.

Verificado contra `tokenizers`: **2.623 líneas, 0 diferencias** — 400 sintéticas
(latín, CJK, cirílico, árabe, emoji, código) y 2.223 de texto real.

## Lo nuevo de la arquitectura

Lo que no existe en el motor GLM. Cada pieza está validada contra la
implementación de referencia de DeepSeek sobre el checkpoint real.

| pieza | fichero | verificación |
|---|---|---|
| **mHC** — residual `[b,s,4,dim]` con Sinkhorn | `dsv4_math.h` | err rel ~1e-7 |
| **CSA** — Compressor con puerta aprendida | `dsv4_math.h` | error 0.00e+00 |
| **Indexer** sobre KV comprimida | `dsv4_math.h` | 0/1536 índices distintos |
| `sparse_attn` con sumidero por cabeza | `dsv4_math.h` | 0 fuera de 1 ULP bf16 |
| RoPE entrelazada, directa e inversa | `dsv4_math.h` | error 0.00e+00 |
| FP8-e4m3 con escalas **UE8M0** | `dsv4_fp8.h` | 4,82e-07 |
| Router `sqrtsoftplus` + routing hash | `dsv4_moe.h` | 0/48 índices distintos |
| Bloque de atención (3 tipos) | `dsv4_attn.h` | 3,6e-03 a 4,3e-03 |
| Camino de **decode** incremental | `dsv4_decode.h` | 3,8e-03 a 5,9e-03 |
| Bloque MoE, 256 expertos MXFP4 | `dsv4_moe.h` | 1,67e-03 |

Los errores del orden de 1e-3 son acumulación de bf16 (ε = 3,9e-3), la misma
banda en la que caen las capas del modelo de referencia.

### El denso: colibrì ya lo rechazaba a propósito

`colibri.c:1485` menciona este modelo por su nombre:

> *"DeepSeek-V4 ships the SAME weight layout (FP8 E4M3, 128x128 blocks) with
> UE8M0 [...] recognizing the signature and refusing is safer than misreading"*

`fmt=8` lee esa geometría pero espera la escala en f32, como la publica Z.ai;
DeepSeek la publica en UE8M0 de 1 byte. `dsv4_fp8.h` es el decodificador que
faltaba. Cuidado con la convención **OCP E4M3-FN**: `exp==0xF` no está reservado
para infinito —sólo `mant==0x7` lo es— así que el máximo finito es 448.

## Reparto de memoria

Es la decisión de diseño central, y la que hace que quepa:

```
residente en RAM    8,67 GiB   atención, normas, routers, expertos compartidos,
                               embeddings y lm_head — EN SU FORMATO NATIVO
por streaming     137,10 GiB   los 11.008 expertos rutados, del NVMe, con LRU
```

Los descriptores de `dsv4_weight.h` hacen que el matmul lea FP8/BF16/MXFP4
directamente y **nunca materialice la matriz dequantizada**. Dequantizar el
denso a f32 serían 26,8 GiB y no cabría. Es la misma decisión que toma colibrì
con su struct `QT`.

## Lo que falta

- **Integración con `coli`**: `deepseek_v4.c` es autónomo (carga, tokeniza,
  genera). Falta que el launcher lo reconozca por `model_type` y que hable el
  protocolo del gateway.
- **MTP / DSpark**: 4.705 tensores (6,5 %). Son 3 bloques con `markov_head` y
  `confidence_head` — la pila MTP *es* la implementación de DSpark. Es un
  acelerador: el modelo genera sin ella.
- **Streaming real**: enganchar el pool de hilos, `experts_apply_union`, PILOT,
  `.coli_usage`, `O_DIRECT` y dual-SSD. Es infraestructura ya existente y
  agnóstica al modelo. Demostrado sobre un modelo reducido que la semántica no
  cambia (logits bit a bit idénticos con caché de 3 de 8 expertos).
- **Contexto largo**: CSA/HCA existen para 1M de tokens; lo probado llega a 128.

## Aviso sobre el criterio de validación

El top-k del router es **discreto**, y el redondeo bf16 que la atención acumula
(~1e-3) basta para que un token cercano al empate cambie de experto: medido, 3
de 448. No es un defecto —cualquier implementación que difiera un ULP hace lo
mismo— pero significa que un teacher-forcing contra `transformers` **no dará
32/32 exacto**, y el criterio hay que fijarlo sabiéndolo. Es el mismo problema
que este repo ya documenta en su top-k del DSA (`colibri.c:3383-3387`).
