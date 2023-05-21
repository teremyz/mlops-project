# pylint: disable=C0114
import logging
import os
from typing import Dict, Union

# pylint: disable=E0401
from fastapi import FastAPI
from utils import downlod_artifact_from_s3, predict_products_using_cetli_id

logging.basicConfig(filename="prediction_web_sevice.log", level=logging.INFO)

MLFOW_EXPERIMENT_ID = os.getenv(
    "MLFOW_EXPERIMENT_ID", "21545d2284b748c3aa462e3f6c903bcf"
)

MODEL = downlod_artifact_from_s3(
    bucket_name="my-mlflow-artifacts-remote",
    file_name="bpr_model.pkl",
    mlflow_exp_id=MLFOW_EXPERIMENT_ID,
)
logging.info("Model downloaded")

OE_PRODUCT_ID, OE_CETLI_ID, _ = downlod_artifact_from_s3(
    bucket_name="my-mlflow-artifacts-remote",
    file_name="product_cetli_user_ordinal_encoders.pkl",
    mlflow_exp_id=MLFOW_EXPERIMENT_ID,
)

logging.info("Encoders downloaded")

# pylint: disable=C0103
app = FastAPI()


@app.get("/predict")
# pylint: disable=C0330
def predict_endpoint(
    cetli_id: str, top_n: int = 5
) -> Dict[str, Dict[str, Union[float, str]]]:
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
