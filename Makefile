# cite-faithfulness — dev + eval targets.
# Windows note: run under Git Bash / WSL, or copy the commands directly.

.PHONY: install install-nli test lint demo smoke eval

install:            ## dev install (metric + tests, no torch)
	pip install -e ".[dev]"

install-nli:        ## add the real NLI judge (torch + sentence-transformers)
	pip install -e ".[nli]"

test:               ## run the metric unit tests (no model download)
	pytest -q

lint:
	ruff check citeval tests

demo:               ## self-contained ALCE scoring demo (no server, no downloads)
	python -m citeval.demo

smoke:              ## drive a running PaperPal with the mock NLI (needs uvicorn up)
	python -m citeval.run_faithfulness --name smoke --nli mock

eval:               ## full Week-1 faithfulness run against a running PaperPal
	python -m citeval.run_faithfulness --name w1-default --nli cross-encoder/nli-deberta-v3-base
