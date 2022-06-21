import pickle
import pandas as pd
import argparse


def load_model(path_model):

    with open(path_model, "rb") as f_in:
        dv, lr = pickle.load(f_in)

    return dv, lr


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, help="Year of data")
    parser.add_argument("month", type=int, help="Month of data")
    args = parser.parse_args()

    dv, lr = load_model("./model.bin")
    categorical = ["PUlocationID", "DOlocationID"]
    filename = f"fhv_tripdata_{args.year:04d}-{args.month:02d}.parquet"
    print(filename)
    df = read_data(f"https://nyc-tlc.s3.amazonaws.com/trip+data/{filename}")
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f"y pred mean: {y_pred.mean()}")
    df["ride_id"] = f"{args.year:04d}/{args.month:02d}_" + df.index.astype("str")
    df["duration"] = y_pred
    df_results = df[["ride_id", "duration"]]
    df_results.to_parquet(
        "df_predictions.parquet",
        engine="pyarrow",
        compression=None,
        index=False,
    )
