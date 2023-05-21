# pylint: disable=C0114
import pickle
from io import BytesIO
from typing import Any

# pylint: disable=E0401
import boto3
import numpy as np
import pandas as pd
from cornac.models.bpr.recom_bpr import BPR
from sklearn.preprocessing._encoders import OrdinalEncoder


# pylint: disable=C0330
def downlod_artifact_from_s3(
    bucket_name: str, file_name: str, mlflow_exp_id: str
) -> Any:
    """
        Download artifacts from AWS S3 storage. Artifacts came from mlflow
        e.g. dir structure follows a strict logic
    Args:
        bucket_name: name of the bucket
        file_name: name of the file
        mlflow_exp_id: mllfow experiment id

    Returns: it returns the downloaded object

    """
    s_3 = boto3.resource("s3")
    with BytesIO() as data:
        s_3.Bucket(bucket_name).download_fileobj(
            f"3/{mlflow_exp_id}/artifacts/{file_name}",
            data,
        )
        data.seek(0)
        artifact = pickle.load(data)
    return artifact


# pylint: disable=C0330
def predict_products_using_cetli_id(
    cetli_id: str,
    top_n: int,
    model: BPR,
    oe_product_id: OrdinalEncoder,
    oe_cetli_id: OrdinalEncoder,
) -> pd.DataFrame:
    """
    Gives back scores for all products
    Args:
        cetli_id: existing user id
        top_n: request will get the top n elements
        model: cornac model that will predict
        oe_product_id: product id ordinal encoder
        oe_cetli_id: cetli id ordinal encoder

    Returns: array with the user or cetli id (first element)
             and the 260 product scores

    """

    encoded_cetli_id = int(
        oe_cetli_id.transform(np.array(cetli_id).reshape(-1, 1))[0][0]
    )

    model_rank = model.rank(encoded_cetli_id)
    ratings = pd.DataFrame(
        {"product_id": model_rank[0], "score": model_rank[1]}
    )

    ratings["product_name"] = oe_product_id.inverse_transform(
        ratings["product_id"].values.reshape(-1, 1)
    )

    ratings = ratings.sort_values("score", ascending=False)

    return ratings[["product_id", "product_name", "score"]].head(top_n)
