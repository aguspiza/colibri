# DeepSeek-V4-Flash-0731 — port en curso

**DeepSeek-V4-Flash** (284B / 13B activos, contexto de 1M) como familia hermana
de colibrì. El motor **corre y genera texto**:

```
$ DSV4_CACHE=512 ./deepseek_v4 /modelos/DeepSeek-V4-Flash-0731 "La capital de Francia es" 8
config: 43 capas, dim 4096, 64 cabezas x 512, 256 expertos top-6
expertos: streaming con cache de 512 slots de 11008 totales

La capital de Francia es una de las ciudades más visitadas del mundo

14 tokens en 33.5 s (0.42 tok/s)
perfil (total): atencion 8.0 s | MoE 24.7 s | head 0.3 s | resto 0.5 s
  del MoE, 19.5 s son I/O de expertos y 5.2 s computo
expertos: 1765 hits / 1847 miss (49% acierto), 24.69 GB leidos
```

En un Ryzen 5700U (Zen 2, **sin AVX-512**) con 17,9 GB de RAM libre y el
checkpoint en un NVMe.

## Dónde se iba el tiempo

La primera versión daba 0,16 tok/s. Medir antes de tocar nada dejó el reparto
claro, y no era el que yo suponía:

| | antes | ahora | qué pasaba |
|---|---|---|---|
| Atención | 35,5 s (55 %) | **5,8 s** | mi kernel FP8 era C escalar de un hilo |
| `lm_head` | 1,9 s | **0,3 s** | ídem, en BF16 |
| MoE cómputo | 11,1 s | **4,4 s** | el router BF16, también escalar |
| MoE I/O | 15,9 s | 15,9 s | sin cambio: el disco manda |

El MoE ya iba rápido porque usa `matmul_mxfp4` de colibrì, que sí trae AVX2. El
resto lo había escrito yo priorizando que fuese legible y verificable contra la
referencia, y se notaba: la atención hacía 4,6 GMAC/token en 3,5 s mientras el
MoE hacía 6,5 GMAC en 1,1 s — **4,5× más lento por MAC**.

Vectorizar (AVX2+FMA) y repartir las filas de salida entre los 8 núcleos da
**2,6× en total**. Los errores contra la referencia no se mueven ni un dígito
(FP8 4,82e-07, MoE 1,67e-03): el orden de acumulación cambia, la precisión no.

## Ahora manda el disco

Con el cómputo arreglado, el I/O es el 58 % del tiempo. Se leen 20 GB por cada
14 tokens a **1,28 GB/s**, y ese número ya es el techo: el prefetch lanza las
lecturas del `top-6` en paralelo sobre 16 hilos —`compat_pread` es
`ReadFile`+`OVERLAPPED` sobre el handle crudo, así que es seguro— y no sube. Es
un NVMe sin DRAM a su ritmo sostenido.

Si no se puede leer más rápido, hay que leer menos. La caché LRU tiene poco
recorrido: en esa generación se piden **1.529 expertos-capa distintos** frente a
1.847 fallos, así que una caché infinita sólo ahorraría el 17 % restante.

Lo que sí lo cambia es **procesar varios tokens a la vez**, porque comparten
expertos. Medido sobre la traza de ruteo real del checkpoint:

| lote | expertos-capa en la unión | lecturas/token | reducción |
|---|---|---|---|
| 1 | 258 | 258 | 1,00x |
| 2 | 404 | 202 | 1,28x |
| **5** | **724** | **145** | **1,78x** |
| 8 | 995 | 124 | 2,07x |

`dsv4_moe.h` ya lo aprovecha: recorre la **unión** de expertos del lote y aplica
cada uno a todas las filas que lo eligieron, en vez de iterar fila a fila
releyendo. En el test de streaming eso baja las lecturas de 132,51 MB a 22,02 MB
con los logits **bit a bit idénticos**. Con `rows==1` el orden de acumulación es
el de antes, así que el decode no cambia.

Y 5 es justo `dspark_block_size` del `config.json`, que es lo que hace que la
pila MTP sea la siguiente pieza y no un adorno — ver abajo.

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
  `confidence_head` — la pila MTP *es* la implementación de DSpark. El modelo
  genera sin ella, pero es **la vía con más recorrido para el rendimiento**, no
  un adorno. El `config.json` la describe entera:

  ```
  dspark_block_size: 5          num_nextn_predict_layers: 1
  dspark_target_layer_ids: [40, 41, 42]
  dspark_markov_rank: 256       dspark_noise_token_id: 128799
  ```

  Verifica 5 tokens por pasada, y por la tabla de arriba eso es 1,78x menos I/O
  por token — que es donde se va el 58 % del tiempo. **Cabe en memoria**: de sus
  10,12 GiB en disco sólo **0,55 GiB son densos** (atención, `shared_experts`,
  `gate`, las `hc_*` y las dos cabezas); los otros 9,56 GiB son expertos y van
  por streaming como los demás. El residente pasaría de 8,67 a 9,22 GiB.
  Estructuralmente cada bloque es un bloque normal —atención + MoE de 256
  expertos + compartido— así que reutiliza las primitivas ya validadas; lo nuevo
  son `markov_head`, `confidence_head` y el bucle de aceptación.
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
