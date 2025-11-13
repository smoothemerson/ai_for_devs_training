FROM python:3.13-slim

RUN apt-get update -y && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir uv \
  && uv pip install -r pyproject.toml --system

COPY ./document/sertoes_livro_euclides ./document/sertoes_livro_euclides

COPY ./src ./src

RUN chmod +x ./src/main.py

CMD ["python", "-m", "src.main"]
