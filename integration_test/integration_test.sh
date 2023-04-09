#!/usr/bin/env bash

if ["${LOCAL_IMAGE_NAME}" == ""]
 then
    LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
    export LOCAL_IMAGE_NAME="prediction_web_service:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} ../prediction_web_service/.
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

docker-compose up -d prediction_web_service

echo "docker-compose up finished"

sleep 5

~/.local/share/virtualenvs/prediction_web_service-TsEAhtVs/bin/python integration_test/test_web_service.py

echo "python test file run"

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down
