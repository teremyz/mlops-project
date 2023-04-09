# pylint: disable=C0114
import logging
import os
import pickle
from io import BytesIO
from typing import Dict, Union

# pylint: disable=E0401
import boto3
import numpy as np
import pandas as pd
from cornac.models.bpr.recom_bpr import BPR
# pylint: disable=E0401
from fastapi import FastAPI
from sklearn.preprocessing._encoders import OrdinalEncoder

logging.basicConfig(filename="prediction_web_sevice.log", level=logging.INFO)

MLFOW_EXPERIMENT_ID = os.getenv(
    "MLFOW_EXPERIMENT_ID", "21545d2284b748c3aa462e3f6c903bcf"
)


S3 = boto3.resource("s3")
with BytesIO() as data:
    S3.Bucket("my-mlflow-artifacts-remote").download_fileobj(
        f"3/{MLFOW_EXPERIMENT_ID}/artifacts/bpr_model.pkl", data
    )
    data.seek(0)
    MODEL = pickle.load(data)

logging.info("Model downloaded")

with BytesIO() as data:
    S3.Bucket("my-mlflow-artifacts-remote").download_fileobj(
        f"3/{MLFOW_EXPERIMENT_ID}/artifacts/"
        + "product_cetli_user_ordinal_encoders.pkl",
        data,
    )
    data.seek(0)
    OE_PRODUCT_ID, OE_CETLI_ID, _ = pickle.load(data)

logging.info("Encoders downloaded")


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

    logging.info("Sorting products...")
    ratings = ratings.sort_values("score", ascending=False)

    return ratings[["product_id", "product_name", "score"]].head(top_n)


# pylint: disable=C0103
app = FastAPI()


@app.get("/predict")
def predict_endpoint(
    cetli_id: str, top_n: int = 5
) -> Dict[str, Union[float, int, str]]:
    """
    It recommends additional products to an existing receipt (cetli)

    Args:
    cetli_id: receipt id
    top_n: it gives back the top_n prodcts

    Returns: returns dictionari tahat contains top_n product id, name, score

    """
    logging.info("Endpoint is predictiong...")
    prediction = predict_products_using_cetli_id(
        cetli_id=cetli_id,
        top_n=top_n,
        model=MODEL,
        oe_product_id=OE_PRODUCT_ID,
        oe_cetli_id=OE_CETLI_ID,
    )

    return prediction.to_dict()
