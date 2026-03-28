PYTHON ?= python

.PHONY: install-cv install-rl install train-cv train-rl test-cv test-rl test run

install-cv:
	$(PYTHON) -m pip install -r cv_pipeline/requirements.txt

install-rl:
	$(PYTHON) -m pip install -r rl_agent/requirements.txt

install: install-cv install-rl

train-cv:
	$(PYTHON) -m cv_pipeline.detection.train --config cv_pipeline/config.yaml

train-rl:
	$(PYTHON) -m rl_agent.training.train --config rl_agent/config.yaml

test-cv:
	$(PYTHON) -m pytest cv_pipeline/tests -q

test-rl:
	$(PYTHON) -m pytest rl_agent/tests -q

test: test-cv test-rl

run:
	@if [ -z "$(SOURCE)" ]; then echo "Usage: make run SOURCE=path/to/video.mp4"; exit 1; fi
	$(PYTHON) -m cv_pipeline.pipeline.session --source $(SOURCE) --config cv_pipeline/config.yaml
