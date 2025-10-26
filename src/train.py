import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 📁 Crear carpeta si no existe
def ensure_dir(path):
    """Crea la carpeta si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # 🎯 Argumentos por defecto
    parser = argparse.ArgumentParser(description="Entrenamiento automático del modelo ML")
    parser.add_argument("--data", type=str, default="data/dataset.csv", help="Ruta del dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Carpeta de salida")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporción de datos de test")
    args = parser.parse_args()

    # 📂 Asegurar que la carpeta de salida exista
    ensure_dir(args.output_dir)

    # 🧠 Cargar datos
    df = pd.read_csv(args.data)

    # Supongamos que el dataset tiene columnas: feature1, feature2, ..., target
    if "target" not in df.columns:
        raise ValueError("El dataset debe contener una columna llamada 'target'")

    X = df.drop("target", axis=1)
    y = df["target"]

    # ✂️ Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # ⚙️ Escalamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🤖 Modelo
    model = LogisticRegression(max_iter=500)

    # 🎯 MLflow Tracking
    mlflow.set_experiment("ML_Pipeline_Automation")

    with mlflow.start_run():
        # Entrenamiento
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Registro de métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Registro de parámetros
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("model", "LogisticRegression")

        # Guardar modelo con MLflow
        mlflow.sklearn.log_model(model, "model")

        # Guardar resultados locales
        results_path = os.path.join(args.output_dir, "metrics.txt")
        with open(results_path, "w") as f:
            f.write(f"Accuracy: {acc}\nF1 Score: {f1}\n")

        print(f"✅ Modelo entrenado y registrado correctamente.")
        print(f"   Accuracy: {acc:.3f}")
        print(f"   F1 Score: {f1:.3f}")

if __name__ == "__main__":
    main()
