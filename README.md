# Challenge Analista de Datos

## Problema

Se proporciona un dataset con datos históricos de stock y movimientos de medicamentos de la bodega de un hospital. El objetivo es **operacionalizar** un modelo predictivo que permita anticipar el **consumo** de cada producto para planificar el reabastecimiento.

Dado un producto (`gtin`) y una fecha, el modelo debe predecir:

- **`cantidad`**: cantidad de unidades que se consumirán (salidas) en esa fecha.

## Dataset

Tres archivos CSV en la carpeta `dataset/`:

### stock.csv — Registro diario de stock por producto

| Columna | Descripción |
|---------|-------------|
| `gtin` | Código de barras del producto |
| `fecha` | Fecha del registro (YYYY-MM-DD) |
| `stock` | Cantidad de unidades en stock ese día |

### movimientos.csv — Registro de movimientos de inventario

| Columna | Descripción |
|---------|-------------|
| `gtin` | Código de barras del producto |
| `fecha` | Fecha del movimiento (YYYY-MM-DD) |
| `cantidad` | Cantidad de unidades movidas |
| `tipo_movimiento` | `E` (entrada/pedido) o `S` (salida/consumo) |

### productos.csv — Catálogo de productos farmacéuticos

| Columna | Descripción |
|---------|-------------|
| `gtin` | Código de barras del producto |
| `material` | Nombre del medicamento |
| `uso_principal` | Indicación terapéutica principal |
| `linea_terapeutica` | Línea terapéutica a la que pertenece |
| `canasta_vigente` | Estado del producto en la canasta farmacéutica |

## Instrucciones

1. Crear un repositorio **público** en GitHub con el contenido de este challenge. El entregable debe ser **completamente anónimo**: usa un nombre de usuario que no revele tu identidad y asegúrate de que ninguna información personal (nombre, correo, foto, etc.) sea visible en el repositorio.

2. Usar la rama **main** para releases oficiales. Se recomienda [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). No elimines tus ramas de desarrollo.

3. No cambiar la estructura del challenge (nombres de carpetas y archivos).

4. Toda la documentación y explicaciones deben ir en `docs/challenge.md`.

5. Enviar tu challenge con un `POST` a:
   ```
   https://check-api-kdl2ntrxjq-tl.a.run.app/data-analyst
   ```
   Body de ejemplo:
   ```json
   {
     "id_postulacion": 12345678,
     "github_url": "https://github.com/tu-usuario/cenabast-challenge.git",
     "api_url": "https://tu-api-url"
   }
   ```
   > `id_postulacion` corresponde al N° Postulación en empleos públicos. **Envía el request solo una vez.**

## Evaluación

El challenge se evalúa sobre **20 puntos** en total. El **puntaje de corte** para aprobar es de **12 puntos**.

## Parte I — Modelo (2 pts)

Se proporciona un **template** en `challenge/model.py` con la interfaz del modelo predictivo. Los métodos están vacíos y deben implementarse.

- Implementar el modelo en `challenge/model.py` que supere el test de rendimiento (al menos 25% mejor que predecir la media por producto). Se puede usar el algoritmo que se estime conveniente.
- Hacer feature engineering en el método `preprocess()` para capturar los patrones del dataset (estacionalidad, tendencia, etc.).
- El modelo debe pasar los tests ejecutando `make model-test`.

> **No se puede** eliminar ni cambiar el nombre o argumentos de los métodos provistos. **Se puede** cambiar la implementación y crear clases/métodos adicionales.

## Parte II — API (4 pts)

Se proporciona un **template** en `challenge/api.py` usando FastAPI. El endpoint `/health` ya funciona, pero el endpoint `/predict` **no está implementado**.

- Implementar el endpoint `/predict` para retornar las predicciones del modelo.
- Implementar validación para **productos desconocidos** (retornar `400`).
- Implementar validación para **fechas inválidas** (retornar `400`).
- La API debe estar funcionando y los endpoints deben responder correctamente.

> **No se puede** eliminar ni cambiar el nombre o argumentos de los métodos provistos. **Se puede** cambiar la implementación y crear clases/métodos adicionales.

## Parte III — Tests (4 pts)

Los tests relacionados a la API deben pasar correctamente.

- Los tests de la API ubicados en `tests/api/test_api.py` deben pasar ejecutando `make api-test`.
- Se evalúa que los tests validen: el endpoint `/health`, el endpoint `/predict`, el manejo de **productos desconocidos** (400) y el manejo de **fechas inválidas** (400).
- Se evalúa que los tests del modelo ubicados en `tests/model/test_model.py` pasen ejecutando `make model-test`.

## Parte IV — Deploy en la Nube (2 pts)

Desplegar la API en GCP usando la Service Account entregada:

- Colocar la URL de la API en el `Makefile` (línea 26).
- La API debe pasar los tests ejecutando `make stress-test`.
- **La API debe estar desplegada hasta que se revisen los tests.**

Se entregará una Service Account de GCP con los permisos necesarios para desplegar en Cloud Run. Es responsabilidad del candidato configurar su entorno y realizar el despliegue correctamente.

## Parte V — CI/CD (5 pts)

Implementar un pipeline de **CI/CD** adecuado:

- Crear una carpeta `.github` y copiar la carpeta `workflows` provista dentro de ella.
- Completar ambos archivos `ci.yml` y `cd.yml`.
- El pipeline de CI debe ejecutar los tests (modelo y API) en cada push/PR.
- El pipeline de CD debe desplegar la API automáticamente al mergear a `main`.

## Parte VI — Análisis Logístico (3 pts)

Si bien el entregable mínimo exigido es la predicción del **consumo** (`cantidad` de salidas), el problema real es anticipar el **próximo pedido de reabastecimiento** por producto. Se evaluará el argumento en `docs/challenge.md` sobre cómo calcularía esta predicción.

> Esta parte **será evaluada** y aporta al puntaje total del challenge.