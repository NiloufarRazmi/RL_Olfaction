help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.ONESHELL:
jupytext: ## Convert JSON notebook to plain text
	# conda init bash
	# eval "$(conda shell.bash hook)"
	# conda activate rl-olfaction
	jupytext --to py:percent backprop/backprop.ipynb
	jupytext --to py:percent FuncApprox/QLearning.ipynb
	jupytext --to py:percent FuncApprox/FuncApprox.ipynb

black-format:
	black .

format: ## Autoformat everything
	make jupytext
	make black-format
