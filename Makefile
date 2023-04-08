LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=teremyz21/prediction_web_service:${LOCAL_TAG}

test:
	pytest tests/

quality_checks:
	isort prediction_web_service/.
	black prediction_web_service/.
	pylint prediction_web_service/.

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} prediction_web_service/.

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration_test/integration_test.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration_test/publish.sh

setup:
	pipenv install --dev
	pre-commit install
