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

eval:               ## single faithfulness run against a running PaperPal
	python -m citeval.run_faithfulness --name w1-default --nli cross-encoder/nli-deberta-v3-base

sweep:              ## controlled study: retrieval-config x model-size (Windows/PowerShell)
	powershell -File scripts/run_sweep.ps1

report:             ## aggregate all runs/ into REPORT.md with bootstrap CIs + significance
	python -m citeval.report --all --baseline dense-8b

rescore:            ## re-score all runs offline after an NLI/metric change (seconds, no server)
	python -m citeval.rescore --all --in-place --nli cross-encoder/nli-deberta-v3-base

verify:             ## evaluate the NLI citation verifier vs gold pages (offline)
	python -m citeval.verify_eval --all

labels:             ## generate a human-labeling sheet for a run (fill in human_supported)
	python -m citeval.label_sheet --run dense-8b

figures:            ## render error-bar PNGs (needs: pip install -e ".[viz]")
	python -m citeval.report --all --baseline dense-8b --figures
