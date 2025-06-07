FROM tensorflow/tensorflow:latest

EXPOSE 80

WORKDIR /app

COPY ./requirements.docker.txt /app/requirements.docker.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.docker.txt

COPY ./ /app

CMD ["fastapi", "run", "/app/main.py", "--port", "80"]
