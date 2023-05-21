import os

import pandas as pd

import prediction_web_service.utils as pp

MLFOW_EXPERIMENT_ID = os.getenv(
    "MLFOW_EXPERIMENT_ID", "21545d2284b748c3aa462e3f6c903bcf"
)

MODEL = pp.downlod_artifact_from_s3(
    bucket_name="my-mlflow-artifacts-remote",
    file_name="bpr_model.pkl",
    mlflow_exp_id=MLFOW_EXPERIMENT_ID,
)

OE_PRODUCT_ID, OE_CETLI_ID, _ = pp.downlod_artifact_from_s3(
    bucket_name="my-mlflow-artifacts-remote",
    file_name="product_cetli_user_ordinal_encoders.pkl",
    mlflow_exp_id=MLFOW_EXPERIMENT_ID,
)


def test_predict_products_using_cetli_id() -> None:
    cetli_id: str = "hW93qGPJug"
    top_n: int = 3
    actual_prediction = pp.predict_products_using_cetli_id(
        cetli_id=cetli_id,
        top_n=top_n,
        model=MODEL,
        oe_product_id=OE_PRODUCT_ID,
        oe_cetli_id=OE_CETLI_ID,
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
