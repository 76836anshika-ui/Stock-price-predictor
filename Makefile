.PHONY: install test lint demo app

install:
	pip install -e ".[dev,app]"

test:
	pytest -q

lint:
	ruff check src tests app.py

demo:  ## offline end-to-end run on simulated data -> reports/demo_report.html
	stockpred DEMO --source synthetic --models ridge,hgb,ensemble --fast --out reports/demo_report.html

app:
	streamlit run app.py
