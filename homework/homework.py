# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import json
import gzip
import pickle
import zipfile
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class Config:
    """Configuración centralizada del proyecto."""
    
    # Rutas
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / "files" / "input"
    MODEL_DIR = BASE_DIR / "files" / "models"
    OUTPUT_DIR = BASE_DIR / "files" / "output"
    
    # Archivos de datos
    TRAIN_ZIP = INPUT_DIR / "train_data.csv.zip"
    TEST_ZIP = INPUT_DIR / "test_data.csv.zip"
    TRAIN_CSV_NAME = "train_default_of_credit_card_clients.csv"
    TEST_CSV_NAME = "test_default_of_credit_card_clients.csv"
    
    # Nombres de columnas
    TARGET_COL = "default"
    ID_COL = "ID"
    CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
    NUM_COLS = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    
    # Hyperparámetros para GridSearch
    GRID_PARAMS = {
        "pca__n_components": [20, 21],
        "kbest__k": [12],
        "svc__kernel": ["rbf"],
        "svc__gamma": [0.099],
    }


def read_zipped_csv(zip_path: Path, csv_name: str) -> pd.DataFrame:
    """Lee un archivo CSV desde dentro de un ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(csv_name) as f:
            return pd.read_csv(f)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset:
    - Elimina columna ID
    - Renombra columna target
    - Elimina valores faltantes y registros con EDUCATION/MARRIAGE = 0
    - Agrupa valores EDUCATION > 4 en la categoría 4 (others)
    """
    cleaned = (
        df.copy()
        .drop(Config.ID_COL, axis=1)
        .rename(columns={"default payment next month": Config.TARGET_COL})
        .dropna()
    )
    
    # Filtrar registros con EDUCATION o MARRIAGE = 0
    cleaned = cleaned[
        (cleaned["EDUCATION"] != 0) & (cleaned["MARRIAGE"] != 0)
    ]
    
    # Agrupar EDUCATION > 4 en categoría 4 (others)
    cleaned.loc[cleaned["EDUCATION"] > 4, "EDUCATION"] = 4
    
    return cleaned


def build_pipeline_search() -> GridSearchCV:
    """
    Construye el pipeline de ML con GridSearchCV.
    
    Pipeline:
    1. ColumnTransformer: OneHotEncoder para categóricas + StandardScaler para numéricas
    2. PCA: Reducción de dimensionalidad
    3. SelectKBest: Selección de features más relevantes
    4. SVC: Support Vector Machine con kernel RBF
    
    IMPORTANTE: remainder="passthrough" es crítico para el rendimiento del modelo.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), Config.CAT_COLS),
            ("std", StandardScaler(), Config.NUM_COLS),
        ],
        remainder="passthrough",  # CRÍTICO: permite que otras columnas pasen
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("pca", PCA()),  # Todas las componentes
            ("kbest", SelectKBest(score_func=f_classif)),
            ("svc", SVC(kernel="rbf", random_state=42)),
        ]
    )

    return GridSearchCV(
        estimator=pipe,
        param_grid=Config.GRID_PARAMS,
        cv=10,  # 10 folds para validación cruzada
        refit=True,
        verbose=1,
        return_train_score=False,
        scoring="balanced_accuracy",
    )


def calculate_metrics(dataset_name: str, y_true, y_pred) -> Dict[str, Any]:
    """Calcula las métricas de evaluación del modelo."""
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def calculate_confusion_matrix(dataset_name: str, y_true, y_pred) -> Dict[str, Any]:
    """Calcula y formatea la matriz de confusión."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


def save_model(model: Any, path: Path) -> None:
    """Guarda el modelo comprimido con gzip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        pickle.dump(model, fh)


def save_metrics_jsonl(metrics: List[Dict[str, Any]], path: Path) -> None:
    """Guarda las métricas en formato JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")


def main():
    """Función principal que ejecuta todo el pipeline."""
    print("Cargando y limpiando datasets...")
    df_train = clean_dataset(read_zipped_csv(Config.TRAIN_ZIP, Config.TRAIN_CSV_NAME))
    df_test = clean_dataset(read_zipped_csv(Config.TEST_ZIP, Config.TEST_CSV_NAME))
    
    # Separar features y target
    X_train = df_train.drop(Config.TARGET_COL, axis=1)
    y_train = df_train[Config.TARGET_COL]
    X_test = df_test.drop(Config.TARGET_COL, axis=1)
    y_test = df_test[Config.TARGET_COL]
    
    # Entrenar modelo con GridSearchCV
    print("Entrenando modelo con GridSearchCV...")
    search = build_pipeline_search()
    search.fit(X_train, y_train)
    
    # Guardar modelo
    model_path = Config.MODEL_DIR / "model.pkl.gz"
    save_model(search, model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Predicciones
    print("Generando predicciones y calculando métricas...")
    y_train_pred = search.predict(X_train)
    y_test_pred = search.predict(X_test)
    
    # Calcular métricas y matrices de confusión
    train_metrics = calculate_metrics("train", y_train, y_train_pred)
    test_metrics = calculate_metrics("test", y_test, y_test_pred)
    train_cm = calculate_confusion_matrix("train", y_train, y_train_pred)
    test_cm = calculate_confusion_matrix("test", y_test, y_test_pred)
    
    # Guardar resultados en el orden requerido por los tests
    all_results = [train_metrics, test_metrics, train_cm, test_cm]
    metrics_path = Config.OUTPUT_DIR / "metrics.json"
    save_metrics_jsonl(all_results, metrics_path)
    print(f"Métricas guardadas en: {metrics_path}")
    
    print("\n✓ Proceso completado exitosamente.")


if __name__ == "__main__":
    main()
