.PHONY: build up down jupyter bash

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

jupyter:
	docker compose exec dev jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

bash:
	docker compose exec dev bash