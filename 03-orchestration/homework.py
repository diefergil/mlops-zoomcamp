import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from prefect import flow, task, get_run_logger
from mlops_zoomcamp.settings import DATA_DIR, MODELS_DIR


@task
def read_data(path):
    logger = get_run_logger()
    logger.info(f"Reading data from: {path}")
    df = pd.read_parquet(path)
    return df


@task
def get_paths(date):

    if date == None:
        date = datetime.now()
    else:
        date = datetime.strptime(date, "%Y-%m-%d")

    date_train = (date - timedelta(weeks=8)).strftime("%Y-%m")
    date_val = (date - timedelta(weeks=4)).strftime("%Y-%m")

    train_path = DATA_DIR / f"fhv_tripdata_{date_train}.parquet"
    val_path = DATA_DIR / f"fhv_tripdata_{date_val}.parquet"

    return train_path, val_path


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mse = mean_squared_error(y_val, y_pred, squared=True)
    logger.info(f"The MSE of validation is: {mse}")
    logger.info(f"The RMSE of validation is: {rmse}")
    return


@flow
def main(date="2021-08-15"):
    categorical = ["PUlocationID", "DOlocationID"]

    train_path, val_path = get_paths(date).result()
    df_train = read_data(train_path)
    df_val = read_data(val_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    
    with open(MODELS_DIR / f"model-{date}.bin", "wb") as f_out:
            pickle.dump(lr, f_out)
    
    with open(MODELS_DIR / f"dict_vec-{date}.bin", "wb") as f_out:
            pickle.dump(dv, f_out)


# main()
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    name="cron-schedule-deployment",
    flow=main,
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    tags=["ml"],
    flow_runner=SubprocessFlowRunner()
)