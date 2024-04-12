.PHONY: install run

install:
	@echo "Installing dependencies..."
	@poetry install

run:
	@echo "Running the app..."
	@chainlit run app.py -w

all: run
