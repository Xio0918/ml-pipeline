import mlflow
import mlflow.sklearn
import argparse
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocess import load_data, preprocess, train_test_split_data
from utils import ensure_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/winequality-red.csv")
    parser.add_argument("--mlruns", type=str, default="mlruns")
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="models")
    return parser.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.output-dir)

    mlflow.set_tracking_uri(f"file://{args.mlruns}")
    mlflow.set_experiment("wine_quality_experiment")

    df = load_data(args.data_path)
    X, y = preprocess(df, target='quality')
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=args.n_estimators,
                                        max_depth=args.max_depth,
                                        random_state=args.random_state))
    ])

    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("r2", float(r2))

        mlflow.sklearn.log_model(pipeline, "model")
        joblib.dump(pipeline, f"{args.output_dir}/model.joblib")

        print(f"Run metrics: mse={mse:.4f}, r2={r2:.4f}")

if __name__ == "__main__":
    main()
