.PHONY: install lint test run docker-build docker-run

install:
	pip install -r requirements.txt
	pip install ruff pytest pre-commit
	pre-commit install

lint:
	ruff check backend tests

format:
	ruff format backend tests

test:
	pytest -v

run:
	cd backend && uvicorn main:app --reload

docker-build:
	docker build -t doc-chatbot .

docker-run:
	docker run -p 8000:7860 -e GROQ_API_KEY=$(GROQ_API_KEY) doc-chatbot
