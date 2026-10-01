# Clasificacion supervisada con WDBC

Trabajo de Carlos Mario Salvatore Ocampo Aguilar para la actividad R1-A2-S8.

## Contenido

- `Carlos_ML_Supervisado_Colab.ipynb`: cuaderno autocontenido para Google Colab.
- `analisis_ml_supervisado_carlos.py`: version ejecutable del experimento.
- `metricas_prueba.csv` y `metricas_validacion_cruzada.csv`: resultados numericos.
- `matrices_confusion.json` y `reportes_clasificacion.json`: detalle por clase.
- `comparacion_metricas.png`, `matrices_confusion.png` y `curvas_roc.png`: figuras del informe.

## Ejecucion

En Google Colab se puede abrir el cuaderno y ejecutar todas las celdas. En un
entorno local:

```bash
python -m pip install -r requirements.txt
python analisis_ml_supervisado_carlos.py
```

Los resultados se reproducen con semilla 42. La clase positiva es `maligno`.
El conjunto de prueba contiene el 20 % de los casos y no participa en la
validacion cruzada ni en el ajuste de los modelos.
