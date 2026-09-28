# Simulación de un Sistema de Internación Hospitalaria

**UTN – FRBA · Ingeniería en Sistemas de Información · Simulación · TP N.º 5**

Simulación de eventos discretos con la metodología **Evento a Evento** para determinar la cantidad óptima de camas de internación de un hospital. El objetivo es equilibrar el tiempo de espera de los pacientes, las derivaciones a otros centros y el costo de las camas ociosas.

Dataset: [Healthcare Dataset (Kaggle)](https://www.kaggle.com/datasets/prasad22/healthcare-dataset), filtrado por `Medical Condition = Diabetes` (9.304 pacientes).

---

## Estructura del repositorio

```
├── simulacion_internacion.py   # Código de la simulación
├── healthcare_dataset.csv      # Dataset original
├── resultados/                 # Salidas generadas al correr el código
│   ├── resultados_simulacion.xlsx
│   ├── resultados_simulacion.csv
│   ├── sensibilidad_N.png
│   └── fdp_TI_IIA.png
├── notebook/                   # Colab del análisis de datos y ajuste de fdp (TP4)
└── diagrama/                   # Diagrama Evento a Evento (.drawio y .png)
```

---

## Cómo ejecutar la simulación

### Opción 1: Windows (VS Code / PowerShell)

1. Descargá el repositorio (**Code → Download ZIP**) y descomprimilo.
2. Abrí una terminal dentro de la carpeta.
3. Instalá las librerías:
   ```
   py -m pip install numpy pandas scipy matplotlib openpyxl
   ```
4. Ejecutá:
   ```
   py simulacion_internacion.py
   ```

> En Linux o macOS, usá `pip` y `python3` en lugar de `py`.

### Opción 2: Google Colab

1. Subí `simulacion_internacion.py` y `healthcare_dataset.csv` al panel de archivos de Colab.
2. En una celda, ejecutá:
   ```
   !python simulacion_internacion.py
   ```

La ejecución tarda aproximadamente 1 minuto. Al terminar, se crea o actualiza la carpeta `resultados/`. La semilla está fija (`SEMILLA_BASE = 42`), así que los resultados son reproducibles.

---

## Modelo

| Tipo | Variables |
|---|---|
| **Datos** | IA: intervalo entre arribos (Exponencial, λ = 5,09 pac/día) · TI: tiempo de internación (Uniforme discreta [1, 30] días) |
| **Control** | N: cantidad de camas · MAXC: máximo de pacientes esperando cama (10) |
| **Estado** | NS: pacientes en el sistema (internados + en espera) |
| **Resultado** | PTO(i): % de tiempo ocioso de cada cama · PEC: promedio de espera en cola (días) · PDER: % de pacientes derivados · CT: costo total (USD/mes) |

**TEF:** TPLL (tiempo de próxima llegada), TPS(i) (tiempo de próxima salida de la cama i).

**Configuración del experimento:** 730 días simulados, 90 días de calentamiento (warm-up), 30 corridas por escenario, N entre 70 y 100.

Los parámetros se modifican en la **Sección 1** del script.

---

## Integrantes

- Nombre Apellido
- Nombre Apellido
- Nombre Apellido
- Nombre Apellido
