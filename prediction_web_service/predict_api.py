import os
import pickle
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI

MLFOW_EXPERIMENT_ID = os.getenv(
    "MLFOW_EXPERIMENT_ID", "21545d2284b748c3aa462e3f6c903bcf"
)

s3 = boto3.resource("s3")
with BytesIO() as data:
    s3.Bucket("my-mlflow-artifacts-remote").download_fileobj(
        f"3/{MLFOW_EXPERIMENT_ID}/artifacts/bpr_model.pkl", data
    )
    data.seek(0)
    model = pickle.load(data)

with BytesIO() as data:
    s3.Bucket("my-mlflow-artifacts-remote").download_fileobj(
        f"3/{MLFOW_EXPERIMENT_ID}/artifacts/"
        + "product_cetli_user_ordinal_encoders.pkl",
        data,
    )
    data.seek(0)
    oe_product_id, oe_cetli_id, oe_user_id = pickle.load(data)


def predict_products_using_cetli_id(
    cetli_id, top_n, model, oe_product_id, oe_cetli_id
):
    """
    Gives back scores for all products
    Args:
        cetli_id: existing user id
        model: cornac model that will predict
        oe_user_id: user id ordinal encoder
        oe_product_id: product id ordinal encoder

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

    ratings = ratings.sort_values("product_name", ascending=True)

    return ratings[["product_id", "product_name", "score"]].head(top_n)


# prediction = predict_products_using_cetli_id(
#     cetli_id="hW93qGPJug",
#     top_n=5,
#     model=model,
#     oe_product_id=oe_product_id,
#     oe_cetli_id=oe_cetli_id,
# )

app = FastAPI()


@app.get("/predict")
def predict_endpoint(cetli_id: str, top_n: int = 5):
    prediction = predict_products_using_cetli_id(
        cetli_id=cetli_id,
        # cetli_id='hW93qGPJug',
        top_n=top_n,
        model=model,
        oe_product_id=oe_product_id,
        oe_cetli_id=oe_cetli_id,
    )

    return prediction.to_dict()
