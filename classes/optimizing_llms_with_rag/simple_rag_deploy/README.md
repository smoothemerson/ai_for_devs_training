# Deploy Simples de RAG

Este projeto contém código de exemplo para deploy de uma aplicação usando o desenvolvimento de LLM utilizando RAG para documentos PDF.

## Pré-requisitos

- Docker e Docker Compose instalados e em execução
- Python e pip
- UV

Se ainda não instalou as dependências Python, a partir da raiz do repositório execute:

```fish
uv sync
```

## Inicialização rápida

1. Inicie o modelo Ollama com o Docker Compose. A partir da raiz do repositório execute:

```fish
# inicia os serviços definidos em docker-compose.yml
docker compose up -d

# verifique o status dos containers
docker compose ps
```
