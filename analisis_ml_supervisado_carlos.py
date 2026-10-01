"""Comparacion reproducible de clasificadores supervisados.

Actividad R1-A2-S8 - Introduccion a Machine Learning.
Autor: Carlos Mario Salvatore Ocampo Aguilar.

La clase positiva es "maligno" porque, en este ejercicio, omitir un caso
maligno es el error que merece mayor atencion. El script usa una sola particion
estratificada para que todos los modelos se comparen sobre los mismos casos.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEMILLA = 42
NOMBRES_CLASE = ["Benigno", "Maligno"]


def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """Carga WDBC y codifica maligno=1, benigno=0."""
    datos = load_breast_cancer(as_frame=True)
    x = datos.data.copy()
    y = (datos.target == 0).astype(int)
    y.name = "diagnostico_maligno"
    return x, y


def construir_modelos() -> dict[str, object]:
    """Define los seis modelos y sus hiperparametros documentados."""
    escalado = lambda estimador: Pipeline(
        [("escalado", StandardScaler()), ("modelo", estimador)]
    )

    return {
        "Regresion logistica": escalado(
            LogisticRegression(
                C=1.0,
                max_iter=5000,
                class_weight="balanced",
                random_state=SEMILLA,
            )
        ),
        "Arbol de decision": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=SEMILLA,
        ),
        "Bosque aleatorio": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEMILLA,
            n_jobs=-1,
        ),
        "SVM RBF": escalado(
            SVC(
                C=2.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=SEMILLA,
            )
        ),
        "KNN": escalado(
            KNeighborsClassifier(n_neighbors=7, weights="distance", p=2)
        ),
        "Naive Bayes": escalado(GaussianNB()),
    }


def puntaje_continuo(modelo: object, x: pd.DataFrame) -> np.ndarray:
    """Devuelve probabilidad o margen para calcular AUC y curva ROC."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(x)[:, 1]
    return modelo.decision_function(x)


def ejecutar_experimento(salida: Path) -> pd.DataFrame:
    salida.mkdir(parents=True, exist_ok=True)
    x, y = cargar_datos()

    x_ent, x_pru, y_ent, y_pru = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEMILLA,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    metricas_cv = {
        "exactitud": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    filas_prueba: list[dict[str, float | str]] = []
    filas_cv: list[dict[str, float | str]] = []
    matrices: dict[str, list[list[int]]] = {}
    curvas: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    reportes: dict[str, dict] = {}

    modelos = construir_modelos()
    for nombre, modelo in modelos.items():
        resultado_cv = cross_validate(
            modelo,
            x_ent,
            y_ent,
            scoring=metricas_cv,
            cv=cv,
            n_jobs=-1,
        )
        fila_cv: dict[str, float | str] = {"modelo": nombre}
        for metrica in metricas_cv:
            valores = resultado_cv[f"test_{metrica}"]
            fila_cv[f"{metrica}_media"] = float(np.mean(valores))
            fila_cv[f"{metrica}_desv"] = float(np.std(valores, ddof=1))
        filas_cv.append(fila_cv)

        modelo.fit(x_ent, y_ent)
        pred = modelo.predict(x_pru)
        score = puntaje_continuo(modelo, x_pru)
        auc = roc_auc_score(y_pru, score)

        filas_prueba.append(
            {
                "modelo": nombre,
                "exactitud": accuracy_score(y_pru, pred),
                "precision": precision_score(y_pru, pred, zero_division=0),
                "recall": recall_score(y_pru, pred, zero_division=0),
                "f1": f1_score(y_pru, pred, zero_division=0),
                "roc_auc": auc,
            }
        )
        matriz = confusion_matrix(y_pru, pred, labels=[0, 1])
        matrices[nombre] = matriz.tolist()
        fpr, tpr, _ = roc_curve(y_pru, score)
        curvas[nombre] = (fpr, tpr, auc)
        reportes[nombre] = classification_report(
            y_pru,
            pred,
            labels=[0, 1],
            target_names=NOMBRES_CLASE,
            output_dict=True,
            zero_division=0,
        )

    prueba = pd.DataFrame(filas_prueba).sort_values(
        ["recall", "f1", "roc_auc"], ascending=False
    )
    resultados_cv = pd.DataFrame(filas_cv).sort_values(
        "recall_media", ascending=False
    )

    prueba.to_csv(salida / "metricas_prueba.csv", index=False, float_format="%.6f")
    resultados_cv.to_csv(salida / "metricas_validacion_cruzada.csv", index=False, float_format="%.6f")
    with (salida / "matrices_confusion.json").open("w", encoding="utf-8") as archivo:
        json.dump(matrices, archivo, ensure_ascii=False, indent=2)
    with (salida / "reportes_clasificacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(reportes, archivo, ensure_ascii=False, indent=2)

    resumen = {
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "observaciones": int(len(x)),
        "predictores": int(x.shape[1]),
        "faltantes": int(x.isna().sum().sum()),
        "clases_completas": {
            "benigno": int((y == 0).sum()),
            "maligno": int((y == 1).sum()),
        },
        "entrenamiento": int(len(x_ent)),
        "prueba": int(len(x_pru)),
        "semilla": SEMILLA,
        "mejor_por_recall": str(prueba.iloc[0]["modelo"]),
    }
    with (salida / "resumen_experimento.json").open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)

    graficar_comparacion(prueba, salida)
    graficar_matrices(matrices, salida)
    graficar_roc(curvas, salida)
    return prueba


def graficar_comparacion(resultados: pd.DataFrame, salida: Path) -> None:
    columnas = ["exactitud", "precision", "recall", "f1", "roc_auc"]
    grafica = resultados.set_index("modelo")[columnas]
    ax = grafica.plot(kind="bar", figsize=(12, 6), width=0.80)
    ax.set_ylim(0.75, 1.01)
    ax.set_ylabel("Puntaje en el conjunto de prueba")
    ax.set_xlabel("")
    ax.set_title("Comparacion de modelos supervisados")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(salida / "comparacion_metricas.png", dpi=300, bbox_inches="tight")
    plt.close()


def graficar_matrices(matrices: dict[str, list[list[int]]], salida: Path) -> None:
    fig, ejes = plt.subplots(2, 3, figsize=(12, 7.5))
    for eje, (nombre, matriz) in zip(ejes.flat, matrices.items()):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(matriz), display_labels=NOMBRES_CLASE
        )
        disp.plot(ax=eje, colorbar=False, cmap="Blues", values_format="d")
        eje.set_title(nombre)
    fig.suptitle("Matrices de confusion sobre el mismo conjunto de prueba", y=1.01)
    plt.tight_layout()
    plt.savefig(salida / "matrices_confusion.png", dpi=300, bbox_inches="tight")
    plt.close()


def graficar_roc(
    curvas: dict[str, tuple[np.ndarray, np.ndarray, float]], salida: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for nombre, (fpr, tpr, auc) in curvas.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{nombre} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC para la clase maligna")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(salida / "curvas_roc.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> pd.DataFrame:
    salida = Path(__file__).resolve().parent
    resultados = ejecutar_experimento(salida)
    print("\nResultados de prueba (clase positiva: maligno)\n")
    print(resultados.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
    return resultados


if __name__ == "__main__":
    main()
