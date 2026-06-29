# Reproduce the full benchmark. Override scale via environment variables, e.g.:
#   make all GNN_CV_REPEATS=10 GNN_N_BOOTSTRAP=2000
PY ?= python3

.PHONY: all data cohort internal tf external sensitivity tables figures clean

all: data cohort internal tf external sensitivity tables figures

data:
	$(PY) scripts/01_download_data.py
cohort:
	$(PY) scripts/02_build_cohort.py
internal:
	$(PY) scripts/03_train_baselines.py
tf:
	$(PY) scripts/04_train_tensorflow.py
external:
	$(PY) scripts/05_external_validation.py
sensitivity:
	$(PY) scripts/06_sensitivity_analyses.py
tables:
	$(PY) scripts/07_make_tables.py
figures:
	$(PY) scripts/08_make_figures.py

clean:
	rm -f results/predictions/*.csv results/metrics/*.json tables/table*.csv figures/figure*
