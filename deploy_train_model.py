import os

from prefect.deployments import Deployment
from prefect.filesystems import RemoteFileSystem
from prefect.orion.schemas.schedules import CronSchedule

from train_model import main

MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "user")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "password")
MINIO_ENDPOINT_URL = os.getenv("MINIO_ENDPOINT_URL", "http://minio:9000")

minio_block = RemoteFileSystem(
    basepath="s3://prefect-flows/model-train-deployment",
    key_type="hash",
    settings=dict(
        use_ssl=False,
        key=MINIO_ROOT_USER,
        secret=MINIO_ROOT_PASSWORD,
        client_kwargs=dict(endpoint_url=MINIO_ENDPOINT_URL),
    ),
)
minio_block.save("minio", overwrite=True)

deployment = Deployment.build_from_flow(
    flow=main,
    name="shop-recommender-model_training",
    schedule=CronSchedule(cron="18 11 15 * *"),
    work_queue_name="recommender-queue",
    storage=RemoteFileSystem.load("minio"),
)

deployment.apply()
