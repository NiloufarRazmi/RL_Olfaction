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

ruff-format:
	# ruff check --fix
	ruff format

format: ## Autoformat everything
	make jupytext
	make ruff-format
	make black-format

ruff:
	ruff check

lint: ## Lint all files
	make ruff

env-update: ## Update the Conda environment & install CLI
	mamba env update -f environment.yml
