# ML Pipeline con GitHub Actions

Proyecto de pipeline automatizado de Machine Learning con MLflow y GitHub Actions.

## Pasos rápidos

1. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Ejecutar entrenamiento:
```bash
make train
```

3. Ejecutar pruebas:
```bash
make test
```

4. Ejecutar MLflow UI:
```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5000
```

En GitHub Actions el workflow automatiza instalación, pruebas y entrenamiento, subiendo los artifacts del modelo.
