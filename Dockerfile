FROM tensorflow/tensorflow:latest

EXPOSE 80

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen

COPY ./ /app

CMD ["/app/.venv/bin/fastapi", "run", "/app/main.py", "--port", "80"]
