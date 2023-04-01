#!/usr/bin/env bash
sleep 3

 python deploy_train_model.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    exit ${ERROR_CODE}
fi

 prefect agent start --work-queue recommender-queue
