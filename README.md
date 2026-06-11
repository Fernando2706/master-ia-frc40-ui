# App grafica FRC40

Aplicacion de escritorio para operarios de planta:

1. Cargar `datos.xlsx` y `quimicos.xlsx`.
2. Preparar una referencia historica de funcionamiento.
3. Calcular una dosis orientativa diaria de quimicos.
4. Revisar el error medio esperado cuando haga falta.

## Instalacion en Windows

Abrir PowerShell en la carpeta principal:

```powershell
cd C:\Users\ferna\Master_IA\Practicas
```

Crear entorno virtual si no existe:

```powershell
python -m venv .venv
```

Activarlo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias de la app:

```powershell
python -m pip install -r app\requirements.txt
```

## Ejecutar

```powershell
python app\app.py
```

## Crear ejecutable `.exe`

Desde PowerShell, en la carpeta principal del proyecto:

```powershell
cd C:\Users\ferna\Master_IA\Practicas
.\app\build_exe.ps1
```

El ejecutable se genera en:

```text
dist\FRC40_Quimicos.exe
```

Si PowerShell bloquea el script, ejecutar una vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Despues volver a lanzar:

```powershell
.\app\build_exe.ps1
```

## Estructura del proyecto

```text
app/
  app.py                         Punto de entrada de la aplicacion
  requirements.txt               Dependencias de Python
  README.md                      Instrucciones de uso
  src/
    frc40_app/
      config.py                  Constantes: variables, objetivos y columnas
      paths.py                   Resuelve la carpeta de datos del usuario (AppData / XDG)
      preprocessing.py           Conversion de datos.xlsx + quimicos.xlsx a CSV limpio
      features.py                Creacion y normalizacion de variables para modelos
      modeling.py                Entrenamiento, seleccion de modelos y metricas
      ui.py                      Interfaz grafica Tkinter
      utils.py                   Utilidades compartidas
```

## Donde se guardan los datos

La aplicacion no escribe junto al ejecutable: cada usuario tiene su propia
carpeta de datos independiente de donde se coloque el `.exe`.

- **Windows:** `%LOCALAPPDATA%\FRC40\Quimicos`
- **Linux:** `$XDG_DATA_HOME/FRC40/Quimicos` (fallback `~/.local/share/FRC40/Quimicos`)
- **macOS:** `~/Library/Application Support/FRC40/Quimicos`

Dentro de esa carpeta se crean las subcarpetas `training_runs\YYYYMMDD_HHMMSS\`
y los modelos se guardan en `models\` dentro de cada run.

La pantalla *Actualizar datos* permite cambiar la carpeta de destino desde la
interfaz; si la cambias, esa nueva ruta sera la que se use a partir de ese
momento.

Las instalaciones antiguas que aun tengan la carpeta `app_outputs` junto al
ejecutable siguen siendo detectadas automaticamente en *Referencias
guardadas* para no perder referencias previas.

## Uso

La interfaz se organiza con una barra lateral:

- `Calcular dosis`
- `Actualizar datos`
- `Referencias guardadas`
- `Control de calidad`

### Pantalla `Calcular dosis`

Es la pantalla principal para uso diario.

1. Introducir fecha.
2. Introducir caudal previsto.
3. Introducir DQO entrada.
4. Introducir DQO salida deseada.
5. Pulsar `Calcular dosis`.

La app muestra la dosis orientativa en kg/dia para:

- Policloruro aluminio
- Coagulante organico
- Floculante cationico

### Pantalla `Actualizar datos`

Usar solo cuando haya nuevos Excel historicos.

Seleccionar:

- `datos.xlsx`
- `quimicos.xlsx`
- carpeta de salida

Pulsar:

```text
Actualizar referencia con estos Excel
```

Durante el entrenamiento la app muestra una barra de progreso y deshabilita el boton para evitar lanzamientos duplicados.

La app genera:

- `training_runs\YYYYMMDD_HHMMSS\frc40_full_app.csv`
- `training_runs\YYYYMMDD_HHMMSS\dataset_modelos_app.csv`
- `training_runs\YYYYMMDD_HHMMSS\models\policloruro_aluminio.joblib`
- `training_runs\YYYYMMDD_HHMMSS\models\coagulante_organico.joblib`
- `training_runs\YYYYMMDD_HHMMSS\models\floculante_cationico.joblib`
- `training_runs\YYYYMMDD_HHMMSS\models\metadata.json`

Cada actualizacion se guarda en una carpeta nueva para no pisar referencias anteriores.

### Pantalla `Referencias guardadas`

Al abrir la app, se carga automaticamente la referencia mas reciente si existe.

Desde esta pantalla se puede:

- ver referencias anteriores
- comparar fiabilidad y error medio por producto
- seleccionar una fila
- pulsar `Usar referencia seleccionada`

La referencia seleccionada queda activa para la pantalla `Calcular dosis`.

### Pantalla `Control de calidad`

Muestra:

- dias usados
- fiabilidad por producto
- error medio esperado en kg/dia
- detalle tecnico para revision interna

La fiabilidad resume como de bien encaja la referencia historica. El error medio indica cuantos kg/dia suele desviarse el calculo.
