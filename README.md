# SuperNova

### Neural Network for predicting data types from raw strings eg csv, json etc
#### Run locally

FastAPI CLI's production-mode command. Adjust --port to whatever port you want locally (e.g. --port 8000)
`fastapi run main.py --port 80`

If you want to run it directly with plain uv in this project (which uses uv.lock/pyproject.toml)

`uv run fastapi run main.py --port 80`

For local development with auto-reload instead of the production server, use:

`fastapi dev main.py --port 80` (or `uv run fastapi dev main.py --port 80`)    

#### Docker
Docker run command: `docker run --name supernova --restart always -p 8090:80 -d potapenkooleg/supernova` 

Or use docker-compose.yml file attached 

Web UI available at **http://localhost:8090/docs**

DockerHub: **https://hub.docker.com/r/potapenkooleg/supernova**

## Screenshots

![Training & Vaildation Accuracy](assets/training.png)

![Test Accuracy](assets/accuracy.png)

![Confusion Matrix](assets/confusion-matrix.png)

