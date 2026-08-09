FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:latest AS uv

FROM --platform=linux/amd64 tensorflow/tensorflow:2.19.0

EXPOSE 80

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen

COPY ./ /app

CMD ["/app/.venv/bin/fastapi", "run", "/app/main.py", "--port", "80"]
