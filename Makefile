.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = my-psg-challenge
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -path ./mysql -prune -o -type f -name "*.py[co]" -exec rm {} +
	find . -path ./mysql -prune -o -type d -name "__pycache__" -exec rm {} + 

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Write requirements.txt
pipreqs: 
	pipreqs --force $(PROJECT_DIR)
	sed -i 's/PyYAML==5.1/PyYAML==3.13/g' requirements.txt

## Install Python Dependencies
requirements: test_environment
	pip install -U pip setuptools wheel pipreqs
	pip install -r requirements.txt

## Write config template
config_template:
	$(PYTHON_INTERPRETER) -m src.tools.config_template

## Start docker compose for dev
docker-dev-up:
	docker-compose -f docker-compose-dev.yaml up -d

## Stop docker compose for dev
docker-dev-stop:
	docker-compose -f docker-compose-dev.yaml stop

## debug
debug:
	$(PYTHON_INTERPRETER) -m ptvsd --host 0.0.0.0 --port 3000 --wait -m ${m}

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

import_data:
	$(PYTHON_INTERPRETER) -m src.data.import_data
	touch import_data

features_data: import_data
	$(PYTHON_INTERPRETER) -m src.features.create_features
	touch features_data

script_training: features_data
	$(PYTHON_INTERPRETER) -m src.script_training
	touch script_training

script_training_2_5: features_data
	$(PYTHON_INTERPRETER) -m src.script_training_2_to_5
	touch script_training_2_5

script_training_bi: features_data
	$(PYTHON_INTERPRETER) -m src.script_training_bi

script_training_teams: features_data
	$(PYTHON_INTERPRETER) -m src.script_training_change_team
	touch script_training_teams

script_prediction: script_training script_training_2_5 script_training_teams
	$(PYTHON_INTERPRETER) -m src.script_prediction


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
