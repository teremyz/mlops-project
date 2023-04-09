#!/usr/bin/env bash

echo "publishing image ${LOCAL_IMAGE_NAME} to Dockerhub..."

docker push ${LOCAL_IMAGE_NAME}
