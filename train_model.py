import os
import pickle
from functools import partial

import mlflow
import numpy as np
import optuna
import pandas as pd
from cornac.data import Dataset
from cornac.eval_methods import BaseMethod
from cornac.experiment import Experiment
from cornac.metrics import AUC, NDCG, RMSE, Precision, Recall
from cornac.models import BPR
from sklearn.preprocessing import OrdinalEncoder

TRACKING_SERVER_HOST = os.getenv(
    "TRACKING_SERVER_HOST", "http://ec2-3-83-223-248.compute-1.amazonaws.com"
)  # fill in with the public DNS of the EC2 instance
mlflow.set_tracking_uri(f"{TRACKING_SERVER_HOST}:5000")
print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
mlflow.set_experiment("shop-recommender")


def preprocess_data(train_df):
    """
    Encode ID columns, convert ids to str and date columns to dates
    Args:
        train_orig:
        encoder_saving_path:

    Returns: Preprocessed data frame

    """
    # Ordinal encode ID columns
    product_oe = OrdinalEncoder()
    cetli_oe = OrdinalEncoder()
    user_oe = OrdinalEncoder()
    train_df["product_id"] = product_oe.fit_transform(
        train_df["product"].values.reshape(-1, 1)
    )
    train_df["cetli_id"] = cetli_oe.fit_transform(
        train_df["cetliId.objectId"].values.reshape(-1, 1)
    )
    train_df["user_id"] = user_oe.fit_transform(
        train_df["owner.objectId"].values.reshape(-1, 1)
    )

    with open(
        "artifacts/product_cetli_user_ordinal_encoders.pkl", "wb"
    ) as file:
        pickle.dump((product_oe, cetli_oe, user_oe), file)

    # Transform columns to the appropriate date types
    train_df[["cetli_id", "product_id"]] = train_df[
        ["cetli_id", "product_id"]
    ].astype(str)
    train_df.updatedAt = pd.to_datetime(train_df.updatedAt)
    train_df.createdAt = pd.to_datetime(train_df.createdAt)

    return train_df


def create_train_and_test_data(df):
    """
    Crate test and train data. There is one product from
    each cetli in the test set that is randomly picked

    Args:
        train: train dataframe

    Returns: Train and test datasets

    """

    df = df.sort_values(["selected"], ascending=False).drop_duplicates(
        ["cetliId.objectId", "product"], keep="first"
    )

    cetli_train_df = df
    cetli_test_df = pd.DataFrame(
        columns=df.columns, index=range(df.cetli_id.nunique())
    )

    for idx, id in enumerate(cetli_train_df.cetli_id.unique()):
        temp_df = cetli_train_df[cetli_train_df.cetli_id == id]
        test_cetli = temp_df.sample(1)
        cetli_train_df.drop(test_cetli.index, inplace=True)
        cetli_test_df.iloc[idx, :] = test_cetli
    print(f"cetli_train_df shape: {cetli_train_df.shape}")
    print(f"cetli_test_df shape: {cetli_test_df.shape}")

    return cetli_train_df, cetli_test_df


def create_rating(df):
    """
    It creates ranking that is the share of the product on the user's cetli
    Args:
        df: Train dataframe

    Returns: Dataframe with rating column

    """

    df = df.sort_values(["selected"], ascending=False).drop_duplicates(
        ["cetliId.objectId", "product"], keep="first"
    )
    user_pre_product = (
        df.groupby(["user_id", "product_id"])
        .count()["cetli_id"]
        .reset_index()
        .rename(columns={"cetli_id": "n_product"})
    )
    user_per_cetli = (
        df[["user_id", "cetli_id"]]
        .drop_duplicates()
        .groupby("user_id")
        .count()
        .reset_index()
    )
    user_per_cetli.rename(columns={"cetli_id": "n_cetli"}, inplace=True)

    df = df.merge(user_per_cetli, how="left", on="user_id")
    df = df.merge(user_pre_product, how="left", on=["user_id", "product_id"])

    df["rating"] = df["n_product"] / df["n_cetli"]

    return df


def create_cetli_data(train_df, test_df):
    """
    Creates cetli level test and train set
    Args:
        train_df: train dataframe
        test_df: test dataframe

    Returns: Test and train triplets to cornac training

    """

    train_df = create_rating(train_df)
    user_cetli_number_df = train_df[["user_id", "n_cetli"]].drop_duplicates()
    rating_df = train_df[["user_id", "product_id", "rating"]].drop_duplicates(
        ["user_id", "product_id"]
    )

    test_df = test_df.merge(
        rating_df, how="left", on=["user_id", "product_id"]
    )
    test_df = test_df.merge(user_cetli_number_df, how="left", on=["user_id"])

    test_df["n_cetli"] = np.where(
        test_df["n_cetli"].isna(), 1, test_df["n_cetli"]
    )
    test_df["rating"] = np.where(
        test_df["rating"].isna(), 1 / test_df["n_cetli"], test_df["rating"]
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    cetli_test_triplets = list(
        test_df[["cetli_id", "product_id", "rating"]]
        .sample(frac=1)
        .itertuples(index=False, name=None)
    )
    cetli_train_triplets = list(
        train_df[["cetli_id", "product_id", "rating"]]
        .sample(frac=1)
        .itertuples(index=False, name=None)
    )

    return cetli_train_triplets, cetli_test_triplets


def optimize_bpr(trial, eval_method, cetli_train_triplets):
    print(trial.number)

    params = {
        "k": trial.suggest_int("k", 5, 30),
        "max_iter": 1000,
        "learning_rate": trial.suggest_float("lr", 1e-5, 0.1, log=True),
        "lambda_reg": trial.suggest_float("lambda_reg", 1e-5, 0.5, log=True),
        "seed": 123,
        "verbose": False,
    }
    with mlflow.start_run():
        metrics = [
            RMSE(),
            Precision(k=10),
            Precision(k=1),
            Precision(k=5),
            Precision(k=3),
            Recall(k=10),
            NDCG(k=10),
            AUC(),
        ]

        exp = Experiment(
            eval_method=eval_method,
            models=[BPR(**params)],
            metrics=metrics,
            user_based=True,
            save_dir="../models/",
            verbose=False,
            show_validation=False,
        )
        exp.run()
        rmse = exp.result[0].metric_avg_results["RMSE"]
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        metric_dict = {}
        for key in exp.result[0].metric_avg_results.keys():
            new_key = key.replace("@", "").replace("(", "").replace(")", "")
            metric_dict[new_key] = exp.result[0].metric_avg_results[key]
        mlflow.log_metrics(metric_dict)

        cornac_data = Dataset.build(cetli_train_triplets)
        trained_model = BPR(**params).fit(train_set=cornac_data)
        with open("artifacts/bpr_model.pkl", "wb") as file:
            pickle.dump(trained_model, file)
        mlflow.log_artifacts("artifacts", artifact_path="")

    return rmse


def main(train_path="../data/Train.csv", n_trials=2):
    df = pd.read_csv(train_path)

    train_df = preprocess_data(df)

    train_df, test_df = create_train_and_test_data(df)

    cetli_train_triplets, cetli_test_triplets = create_cetli_data(
        train_df, test_df
    )

    print(cetli_train_triplets[:2])
    print(cetli_test_triplets[:2])
    cetli_bm = BaseMethod().from_splits(
        train_data=cetli_train_triplets,
        test_data=cetli_test_triplets,
        rating_threshold=0,
    )

    hd_study = optuna.create_study(
        study_name="bpr_opt_cetli",
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(max_resource="auto"),
    )

    modeified_optimizer_cetli = partial(
        optimize_bpr,
        eval_method=cetli_bm,
        cetli_train_triplets=cetli_train_triplets,
    )

    hd_study.optimize(
        modeified_optimizer_cetli, n_trials=n_trials, gc_after_trial=True
    )


main(train_path="data/Train.csv")
