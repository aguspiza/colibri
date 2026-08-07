# DeepSeek-V4-Flash: de 0,16 a 0,78 tok/s

Bitácora de optimización del motor (`c/deepseek_v4.c`) sobre el checkpoint real
de 284B. Se documentan tanto los cambios que funcionaron como **los que no**,
porque tres de las cuatro hipótesis que parecían obvias resultaron falsas y
descartarlas con una medida costó mucho menos que implementarlas.

Máquina: Ryzen 5700U (Zen 2, 8 núcleos / 16 hilos, **sin AVX-512**), 31,4 GB de
RAM, checkpoint en un NVMe sin DRAM. 43 capas, 256 expertos top-6, 156 GiB en
disco.

## Resumen

| | tok/s | qué cambió |
|---|---|---|
| punto de partida | 0,16 | — |
| kernels vectorizados | 0,42 | AVX2+FMA y OpenMP en FP8 y BF16 |
| un descriptor por hilo | 0,51 | Windows serializaba las lecturas |
| aritmética en vez de `gather` | 0,65 | el `gather` de Zen 2 es lento |
| prefetch asíncrono | 0,70 | pool persistente de lectores |
| generación larga (101 tokens) | **0,78** | se amortiza el arranque en frío |

Perfil actual, 14 tokens con caché de 512 slots:

```
20.0 s   atencion 4.2 s | MoE 14.9 s | head 0.3 s | resto 0.5 s
         del MoE, 9.7 s son espera de I/O y 5.2 s computo
```

En todos los casos el texto generado es **idéntico** y los 95 checks del oráculo
siguen pasando. Ninguna optimización toca la semántica.

## Método: medir antes de tocar

Cada cambio salió de un perfil, no de una intuición. Las intuiciones fallaron
casi siempre:

- «el cuello es el MoE, que mueve el 97,5 % de los parámetros» → era la
  **atención**, con el 55 % del tiempo.
- «1,28 GB/s es el techo de este NVMe» → era el techo de **un solo lector**.
- «agrupar tokens reduce el I/O 1,78x» → lo **empeora**.
- «con más RAM para la caché ganaríamos» → el tiempo es **plano**.

### La trampa de la caché de páginas

Tres benchmarks de disco de esta sesión dieron resultados falsos —7,6 / 4,9 /
4,0 GB/s— porque las pasadas releían los mismos bytes desde la caché de páginas
del SO: medían RAM, no disco. En Windows `posix_fadvise(POSIX_FADV_DONTNEED)` no
basta para descartarlas.

**La única medida fiable es que cada pasada toque bytes que nadie ha leído
antes.** Con grupos disjuntos y en frío, el disco da:

| | 1 hilo | 4 hilos | 8+ |
|---|---|---|---|
| lecturas dispersas de 4 MB | 1,06 GB/s | 2,73 GB/s | ~2,9 GB/s |

Ese 2,9 GB/s es la referencia contra la que hay que juzgar cualquier cifra de
I/O del motor.

## Lo que funcionó

### 1. Los kernels propios eran C escalar de un hilo

El perfil inicial repartía así los 76,8 s de una generación de 12 tokens:
atención 35,5 s (55 %), MoE 27,3 s, `lm_head` 1,9 s.

Que la atención dominase era la pista: hace ~4,6 GMAC por token y tardaba 3,5 s,
mientras el MoE hacía 6,5 GMAC en 1,1 s. **4,5x más lento por MAC.** La razón es
que el MoE usa `matmul_mxfp4` de colibrì, que ya trae rutas AVX2, y el resto lo
había escrito yo priorizando legibilidad y verificabilidad contra la referencia.

Vectorizar con AVX2+FMA y repartir las filas de salida entre los núcleos: **2,6x**.
La escala UE8M0 pasa a una tabla de 256 floats — era un `ldexpf` por bloque de
128 y por fila de salida, ocho millones de llamadas a libm por token.

### 2. El `gather` de e4m3 costaba otro 2x

Ya vectorizado, el kernel FP8 seguía en **14 GFLOP/s** mientras el MXFP4 sacaba
33 en el mismo hardware. La atención relee sus 5,40 GB de pesos residentes en
cada token, y eso daba 8,4 GB/s efectivos: poco para ser un límite de memoria.

La causa era decodificar e4m3 con `_mm256_i32gather_ps` sobre una LUT de 256
floats. Cabe entera en L1, pero **en Zen 2 el gather se ejecuta como 8 accesos
secuenciados y no se encadena con el resto del bucle**.

Sale mucho mejor con aritmética de bits:

- `exp != 0`: `mag << 20` deja la mantisa en los bits 22..20 y el exponente en
  26..23; sumar `120 << 23` corrige el sesgo (e4m3 usa 7, f32 usa 127).
- `exp == 0`: subnormal, vale `mant * 2^-9`.
- `mag == 0x7F`: NaN en OCP E4M3-FN. La fórmula daría 480, así que se respeta con
  un `blend`. El checkpoint no trae ninguno —comprobado, el máximo es 0x7E =
  448— pero si un día aparece es mejor que se propague a que se lea como un
  número válido.

Atención **9,0 → 4,4 s**, o sea 18 GB/s: ahora sí es el ancho de banda de la RAM.

### 3. Windows serializaba las lecturas «paralelas»

Con el cómputo arreglado, el I/O pasó a ser el 58 % del tiempo a 1,28 GB/s. Di
por hecho que era el techo del disco porque el prefetch ya lanzaba las lecturas
del top-k sobre 16 hilos. Pero 1,28 es justo el ritmo de **un** lector.

`compat_pread` hace `ReadFile` + `OVERLAPPED` sobre un handle abierto en modo
síncrono, y **en Windows el SO serializa la E/S sobre el objeto fichero** aunque
se le pase un `OVERLAPPED`. El paralelismo era aparente.

La solución es **un descriptor por hilo y shard** (48 x 16 handles, abiertos al
cargar). I/O de 20,6 a 12,3 s, ancho de banda 1,28 → 1,98 GB/s.

De paso salieron del camino caliente los offsets de los 66.048 tensores de
experto (se resuelven al cargar) y el `realloc` de los buffers, que estaba en la
parte **serie** de cada tanda de prefetch.

### 4. Prefetch asíncrono con pool persistente

El prefetch bloqueaba hasta tener los seis expertos y sólo entonces calculaba:
los 5,2 s de cómputo del MoE y los 12,3 de I/O iban en serie. Con un pool de
lectores persistente, `tier_prefetch` encola y vuelve, y el MoE calcula el
primer experto mientras los demás siguen llegando.

Espera de I/O 12,3 → 9,7 s. Quita además el montaje y desmontaje de una región
OpenMP por capa y token — 602 en una generación de 14.

Un detalle de corrección: el LRU **no puede expulsar un slot con lecturas en
vuelo**, porque los workers están escribiendo en sus buffers. `tier_reserve` los
salta; como mucho hay `topk` a la vez y la caché tiene cientos, así que nunca se
queda sin candidato.

## Lo que NO funcionó

### Agrupar tokens (y con ello MTP/DSpark)

Parecía la palanca obvia. Los tokens consecutivos comparten expertos, así que un
lote de 5 pide 724 expertos-capa distintos en vez de 5x258 = 1290: **1,78x menos
peticiones**. Y 5 es justo el `dspark_block_size` del `config.json`.

Pero ésa es la comparación equivocada. El baseline real no son 258 peticiones
por token sino **124,2 fallos de caché**, porque la LRU ya explotaba ese mismo
solapamiento — y mejor. Simulando la política de caché sobre la traza de ruteo
real (el simulador reproduce **exactamente** los 2.980 fallos que mide el motor):

| lote | fallos/token | vs secuencial |
|---|---|---|
| 1 | **124,2** | — |
| 2 | 138,5 | 0,90x |
| 5 | 162,2 | 0,77x |
| 8 | 138,1 | 0,90x |

Agrupar **empeora**. En orden secuencial cada capa se revisita cada 258 accesos
y con 512 slots eso cabe; en lotes de 5 se revisita cada 724 y no cabe, así que
la agrupación convierte aciertos de caché en deduplicación y encima pierde.
Y por encima de ~2.048 slots los fallos se saturan en los *obligatorios* —cada
experto distinto se lee una vez— donde no hay nada que agrupar pueda mejorar.

El único ahorro que quedaba era amortizar los 5,40 GB/token de pesos de
atención, y ése salió mucho más barato arreglando el kernel (hallazgo 2).

**Y con más RAM tampoco.** Después de arreglar los kernels, el motor está *en su
roofline* en casi la mitad del tiempo:

| | tiempo | GFLOP/s | límite |
|---|---|---|---|
| atención | 4,2 s | 31 | FLOPs **y** ancho de banda a la vez |
| MoE cómputo | 5,2 s | 35 | FLOPs |

Los 5,40 GB/token de pesos de atención a 18 GB/s dan exactamente los 0,30
s/token medidos, y esos mismos 0,30 s dan 31 GFLOP/s — justo lo que rinde el
mejor kernel del repo en este hardware. Los dos límites coinciden, así que
amortizar el tráfico de pesos por lotes no libera nada: enseguida topa con los
FLOPs.

Eso es lo que decide la cuestión. **La decodificación especulativa cambia FLOPs
por menos pasos secuenciales** —de hecho *aumenta* el total de FLOPs, porque
calcula tokens que luego se descartan— y sale a cuenta cuando los FLOPs sobran,
que es el caso de una GPU. Aquí no sobran: el 47 % del tiempo ya está en el
roofline de AVX2 sobre 8 núcleos Zen 2. No hay presupuesto que gastar.

`c/dsv4_moe.h` implementa igualmente el recorrido por **unión** de expertos
—cada uno se aplica a todas las filas que lo eligieron— porque es correcto y
hace falta para cualquier camino por lotes. Con `rows == 1` el orden de
acumulación es el de antes y el decode da los mismos bits.

### Agrandar la caché de expertos

Barrido sobre el motor real, 14 tokens:

| caché | GB | acierto | bytes leídos | tiempo |
|---|---|---|---|---|
| 128 | 1,7 | 0 % | 48,29 GB | 20,5 s |
| 256 | 3,4 | 19 % | 44,03 GB | 20,5 s |
| 512 | 6,9 | 49 % | 24,69 GB | **20,0 s** |
| 768 | 10,3 | 55 % | 21,69 GB | 21,1 s |
| 1024 | 13,7 | 57 % | 20,87 GB | 21,9 s |

El tiempo es **plano** mientras los bytes varían 2,3x. Y con 101 tokens, donde
el simulador predecía que la caché grande ganaría de calle (45 fallos/token con
2048 slots frente a 114 con 512), la medida real sigue plana: **130,8 s con 512
frente a 129,5 s con 1024**, leyendo un 32 % menos.

La explicación: **nuestra caché LRU es redundante con la caché de páginas del
SO**, que guarda exactamente los mismos bytes. Agrandar la nuestra no añade
capacidad, se la quita al SO para duplicar lo que ya tenía. La prueba está en el
ancho de banda efectivo: con caché de 128 slots el motor «lee» 48,29 GB en 9,8 s
= **4,9 GB/s**, muy por encima de los 2,9 que da el disco. Esos bytes venían de
RAM.

**Aviso sobre el simulador**: modela sólo la LRU del motor y no ve la caché de
páginas de debajo, así que sobrestima el beneficio de más slots. Es fiable para
contar *peticiones* —reproduce los fallos medidos al fallo— pero no para
predecir *tiempo*.

Consecuencia práctica: `DSV4_CACHE` existe para experimentar, pero subirlo no
compensa. Una caché pequeña deja más RAM al SO, que la administra mejor.

### Mapear los shards en memoria (para ganar velocidad)

Con `pread` los bytes hacen dos viajes: disco → caché de páginas → nuestro
buffer, y la copia se paga incluso cuando la página ya estaba en RAM. Mapear los
shards y apuntar el kernel MXFP4 directamente a la página parecía eliminar de un
tirón la copia *y* la duplicación de caché. Es seguro además: `quant.h` no usa
ni una carga alineada, sólo `loadu` (49 de 49).

Implementado con `CreateFileMapping`/`MapViewOfFile` y `PrefetchVirtualMemory`
para pedirle al SO los rangos por adelantado —sin tocarlos, que costaría el
mismo ancho de banda que leerlos— sale **peor**: 23,7 s frente a 19,4.

El perfil explica el porqué: la espera de I/O baja a 0,0 s pero el cómputo del
MoE se dispara de 5,1 a 18,4 s. Los fallos de página ocurren *dentro* del
matmul. Y el motivo es estructural:

**Los buffers de los slots se reutilizan, así que sus páginas se mapean una vez
y no vuelven a fallar nunca. El mapeo falla en cada región nueva** — 13,4 MB por
experto son ~3.400 fallos de 4 KB, cada uno una entrada al kernel. Cambiar un
`memcpy` (10-20 GB/s, sin entrar al kernel) por 3.400 trampas por experto no
sale a cuenta ni de lejos. `PrefetchVirtualMemory` los convierte en fallos
blandos, pero blandos siguen costando ~1 µs cada uno.

La lección general: **mmap gana cuando las páginas se reutilizan y pierde cuando
se recorren datos nuevos constantemente**, que es exactamente el patrón de un
tier de expertos por streaming.

## `DSV4_LOAD`: mapear no compra velocidad, compra memoria privada

Descartar mmap por lento habría sido un error, porque el eje que importa no era
ése. `--no-mmap` de llama.cpp hacía dos cosas a la vez y en las versiones
recientes está deprecado en favor de `--load-mode`
(`none | mmap | mlock | mmap+mlock | dio`), precisamente porque **no hay una
respuesta buena para todas las máquinas**. Aquí pasa lo mismo.

Medido sobre 14 tokens, promediando dos pasadas:

| `DSV4_LOAD` | privado | working set | libre del sistema | tiempo |
|---|---|---|---|---|
| `read` (por defecto) | 13,2 GB | 13,2 GB | 8,2 GB | **19,3 s** |
| `dense` | **5,4 GB** | 12,2 GB | 8,4 GB | 21,5 s |
| `all` | **0,6 GB** | 24,7 GB | 0,6 GB | 24,4 s |

- **`read`** — todo con `pread` a buffers propios. Lo más rápido.
- **`dense`** — mapea el conjunto denso (atención, `embed`, `lm_head`,
  compartidos) y deja los expertos con `pread`. Cada población con la política
  que le conviene. Baja la memoria privada 7,8 GB por un 11 % de tiempo.
- **`all`** — mapea también los expertos y quita la caché LRU entera. El proceso
  se queda en **0,6 GB privados**: cabe en máquinas donde `read` ni arranca, a
  cambio de un 26 %.

El matiz importante es **qué compra `dense` exactamente**. No baja el working
set —las páginas siguen residentes, sólo que compartidas con la caché del SO—
sino que convierte 8,67 GB de memoria **anónima** en memoria **respaldada por
fichero**. Bajo presión, el SO puede descartarlas y releerlas del disco en vez
de mandarlas al fichero de paginación. Es más robusto, no más libre.

Y cuidado con la métrica: el primer intento midió *working set* y salía que
mapear **aumentaba** el consumo (24,9 GB) porque las páginas de fichero tocadas
cuentan en el working set del proceso. Para «cuánta RAM necesita este proceso»
hay que mirar **bytes privados**.

Lo que no está implementado es el segundo eje de llama.cpp, el de residencia
(`mlock` / `VirtualLock`). Aquí importa poco: el conjunto denso se toca en cada
token, así que se queda residente por su cuenta, y fijarlo en Windows exige
tocar la cuota de working set del proceso.

Los tres modos generan **el mismo texto**.

## Qué queda

- **El I/O sigue siendo el 49 %** del tiempo (9,7 s de 20,0). El suelo son los
  expertos distintos que hay que leer sí o sí: 1.529 en 14 tokens, unos 20,5 GB,
  que a 2,9 GB/s son ~7 s. Estamos cerca.
- **Eliminar la copia**: hoy los bytes van del disco a la caché de páginas y de
  ahí a nuestros buffers. Mapear los shards en memoria (`mmap` /
  `MapViewOfFile`) y apuntar el kernel MXFP4 directamente a las páginas
  eliminaría la copia *y* la duplicación de caché de un tirón. Es el cambio con
  más recorrido que queda, y encaja con lo que ya hace colibrì con sus gemelos
  `O_DIRECT`.
- **Solapar mejor**: la atención de la capa L+1 depende del MoE de la capa L, así
  que la cadena `attn -> router -> lectura -> MoE` es intrínsecamente serie. El
  único solape posible es el que ya se hace dentro de la capa.
