help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.ONESHELL:
jupytext: ## Convert JSON notebook to plain text
	# conda init bash
	# eval "$(conda shell.bash hook)"
	# conda activate rl-olfaction
	jupytext --to py:percent **/*.ipynb

black-format:
	black .

isort-format:
	isort --profile black .

format: ## Autoformat everything
	make jupytext
	make isort-format
	make black-format

flake8:
	flake8 --max-line-length 88 --extend-ignore E203 --exclude="**/.ipynb_checkpoints/" .

lint: ## Lint all files
	make flake8
