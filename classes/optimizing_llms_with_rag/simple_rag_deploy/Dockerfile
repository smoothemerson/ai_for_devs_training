FROM python:3.13-slim

# Install build-essential compiler and tools
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pyproject.toml file
COPY pyproject.toml .

# Install the libs in pyproject.toml file
RUN pip install --no-cache-dir uv \
  && uv pip install -r pyproject.toml --system

# Copy PDF file
COPY ./rag/document.pdf ./rag/document.pdf

# Copy the function code
COPY ./src/main.py ./main.py

# Set permission to make the file executable
RUN chmod +x main.py

# Set CMD to your handler
CMD ["python", "-m", "main", "lambda_handler"]
