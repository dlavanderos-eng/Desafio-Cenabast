# Documentación del Challenge — Cenabast

## Parte I — Modelo predictivo de consumo

### 1. Definición del problema y filtrado de datos

El objetivo de negocio es predecir el **consumo** (salidas de bodega) de un producto (`gtin`) en una fecha dada. El archivo `movimientos.csv` contiene tanto entradas (`E`, reposición) como salidas (`S`, consumo real), y ambas magnitudes son muy distintas:

| tipo_movimiento | media | mediana | máx |
|---|---|---|---|
| S (salida/consumo) | 11.97 | 7 | 211 |
| E (entrada/reposición) | 53.15 | 27 | 943 |

Las entradas son ~4-5x más grandes que las salidas. Mezclarlas en un mismo target habría introducido un sesgo severo. Por eso, `preprocess()` filtra explícitamente a `tipo_movimiento == 'S'` cuando esa columna está presente en los datos crudos (modo entrenamiento). En modo servicio (la API solo entrega `gtin` + `fecha`, sin `tipo_movimiento`), no se aplica ningún filtro porque la columna no existe en el payload.

### 2. Feature engineering

**Calendario (estacionalidad):** `dia_semana`, `dia_mes`, `mes`, `es_fin_de_mes`.

**Contexto del producto** (`productos.csv`, cargado internamente por `preprocess()`): `uso_principal`, `linea_terapeutica`, `canasta_vigente` como variables categóricas.

**Stock** (`stock.csv`): nivel de inventario del producto en la fecha exacta del movimiento — la feature individual más importante del modelo. No introduce fuga de información porque es una variable observada independientemente del target.

**Tendencia histórica** (aprendida en `fit()`, no en `preprocess()`, para evitar data leakage): `gtin_media_historica`, `gtin_std_historica`, `gtin_dow_media_historica`.

### 3. Selección del algoritmo

`HistGradientBoostingRegressor` de scikit-learn, con `loss="absolute_error"` (optimiza directamente MAE, la métrica de evaluación), soporte nativo de categóricas vía `OrdinalEncoder`, e hiperparámetros conservadores para evitar sobreajuste dado el tamaño del dataset (~10k transacciones de salida).

### 4. Resultado del test de rendimiento

El modelo supera consistentemente el baseline oficial (media de consumo por producto) por un margen de **32-36%** de mejora en MAE, validado en 6 semillas distintas de `train_test_split` — muy por encima del 25% exigido, y estable entre semillas (no depende de un split afortunado).

---

## Parte VI — Análisis Logístico: del consumo al próximo pedido de reabastecimiento

### El problema real es distinto al problema modelado

El modelo de la Parte I responde una pregunta puntual: *"¿cuánto se va a consumir de este producto en esta fecha?"*. Pero la pregunta que realmente le importa a Cenabast operativamente es otra: **¿cuándo debo emitir el próximo pedido de reabastecimiento de este producto, y por qué cantidad?**

Son preguntas relacionadas pero no equivalentes. La primera es una predicción puntual de demanda; la segunda es una **decisión de inventario** que combina esa predicción con el estado actual del stock, el tiempo que demora en llegar un pedido, y la tolerancia al riesgo de quiebre de stock que la institución esté dispuesta a asumir — especialmente sensible tratándose de insumos de salud pública, donde un quiebre de stock no es solo una pérdida comercial sino un riesgo para la continuidad de tratamientos.

### Marco propuesto: punto de reorden (Reorder Point) con stock de seguridad

La formulación clásica de gestión de inventario para este problema es el **punto de reorden (ROP)**:

```
ROP = (demanda_promedio_diaria × lead_time_dias) + stock_de_seguridad
```

Donde:

- **`demanda_promedio_diaria`**: no la calcularía como un promedio histórico estático, sino como el **promedio de las predicciones del modelo de la Parte I** sobre el horizonte del lead time. Es decir, en vez de usar `gtin_media_historica` (una foto del pasado), uso el modelo ya entrenado para generar `predict()` sobre cada uno de los próximos `lead_time_dias`, y promedio esas predicciones. Esto captura estacionalidad relevante (si el lead time cae en un período de mayor consumo estacional para ese producto, el ROP lo refleja, cosa que un promedio histórico plano no haría).

- **`lead_time_dias`**: tiempo entre que se emite una orden de reabastecimiento y el producto está físicamente disponible en bodega. **Este dato no está en los datasets provistos** (`movimientos.csv`, `productos.csv`, `stock.csv` no registran fechas de orden vs. fechas de recepción) — sería el primer dato adicional que solicitaría al negocio: por proveedor, o al menos por línea terapéutica, ya que el lead time varía significativamente entre insumos de bajo requerimiento regulatorio y medicamentos que requieren procesos de importación/registro más largos.

- **`stock_de_seguridad`**: el colchón adicional para absorber variabilidad de demanda durante el lead time, dado que ni la demanda ni el lead time son perfectamente predecibles. Fórmula estándar (asumiendo demanda con distribución aproximadamente normal):

```
stock_de_seguridad = Z × σ_demanda × √lead_time_dias
```

  - **`σ_demanda`**: la desviación estándar de la demanda diaria por producto — **ya la calculamos y la tenemos disponible**: es exactamente `gtin_std_historica`, la misma feature que ya usa el modelo de consumo. No hace falta trabajo adicional para obtenerla.
  - **`Z`**: el factor asociado al nivel de servicio deseado (probabilidad de no quebrar stock durante el lead time). Por ejemplo, `Z = 1.65` corresponde a ~95% de nivel de servicio, `Z = 2.33` a ~99%. Esta es una decisión de política, no un cálculo — y probablemente debería **variar por criticidad clínica del producto**: un antihipertensivo de uso crónico (`ENFERMEDADES_CRONICAS`, según vimos en `productos.csv`) probablemente amerita un `Z` más alto (mayor tolerancia a sobre-stockear) que un insumo de bajo riesgo si se agota temporalmente.

### Por qué esto no es solo "envolver el modelo en una fórmula"

Dos matices importantes que iría a validar con el equipo de abastecimiento antes de implementar esto en producción:

1. **La varianza histórica (`gtin_std_historica`) asume que la variabilidad futura se parece a la pasada.** Para productos con pocos meses de historial, o que cambiaron de patrón de uso recientemente (por ejemplo, un cambio en protocolo clínico que aumente/disminuya su uso), este supuesto es débil. Una mejora natural sería ponderar más los datos recientes (una media móvil con decaimiento exponencial) en vez de tratar todo el histórico por igual.

2. **El ROP asume reposición continua (revisar el stock todo el tiempo).** Si en la práctica Cenabast hace pedidos en ciclos fijos (ej. semanal/mensual, no "apenas se cruza el ROP"), el modelo correcto no es el ROP puro sino un modelo de **revisión periódica**, donde el nivel objetivo de stock (`S`) debe cubrir la demanda durante `lead_time + intervalo_de_revision`, no solo el lead time. Antes de implementar, confirmaría con el equipo cuál es el proceso real de emisión de órdenes.

### Cantidad a pedir

Una vez definido *cuándo* pedir (al cruzar el ROP), la cantidad a pedir dependería de la política de inventario que use Cenabast — por ejemplo, un **Economic Order Quantity (EOQ)** si buscan minimizar costos de orden + almacenamiento, o simplemente reponer hasta un nivel máximo objetivo (`S_max - stock_actual`) si el criterio es simplicidad operativa. Esto es una decisión de negocio adicional que excede lo que los datos por sí solos pueden determinar, y la dejaría como siguiente conversación con el equipo antes de tomar una decisión de diseño unilateral.