import os
import pickle
from io import BytesIO

import boto3
import pandas as pd

import prediction_web_service.predict_api as pp

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


def test_predict_products_using_cetli_id() -> None:
    cetli_id: str = "hW93qGPJug"
    top_n: int = 3
    actual_prediction = pp.predict_products_using_cetli_id(
        cetli_id=cetli_id,
        top_n=top_n,
        model=model,
        oe_product_id=oe_product_id,
        oe_cetli_id=oe_cetli_id,
    )
    expected_prediction: pd.DataFrame = pd.DataFrame(
        {
            "product_id": [176, 5, 194],
            "product_name": ["sajt", "avocado", "száraz törlő"],
            "score": [9.423813, 9.260786, 9.098109],
        },
        index=[109, 167, 47],
    )

    assert (
        actual_prediction.product_id.tolist()
        == expected_prediction.product_id.tolist()
    )
    assert (
        actual_prediction.product_name.tolist()
        == expected_prediction.product_name.tolist()
    )
    assert [round(x, 2) for x in actual_prediction.score.tolist()] == [
        round(x, 2) for x in expected_prediction.score.tolist()
    ]
