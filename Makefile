.ONESHELL:
lauch_mlflow_server:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
