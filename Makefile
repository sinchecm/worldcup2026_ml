PYTHON ?= python3
PIP ?= pip
STREAMLIT ?= streamlit

.PHONY: check prepare-data train app

check:
	$(PYTHON) -m py_compile streamlit_app.py src/*.py

prepare-data:
	$(PYTHON) src/prepare_real_data.py

train:
	$(PYTHON) src/main.py

app:
	$(STREAMLIT) run streamlit_app.py
