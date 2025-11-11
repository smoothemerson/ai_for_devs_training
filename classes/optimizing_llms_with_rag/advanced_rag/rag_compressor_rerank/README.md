# RAG Avançado

Este projeto contém código de exemplo para o desenvolvimento de LLM utilizando RAG de um projeto do LangChain envolvendo códigos escritos em Python. O objetivo é realizar o RAG Rerank no documento PDF para responder ao usuário.

## Pré-requisitos

- Docker e Docker Compose instalados e em execução
- Python e pip
- UV

Se ainda não instalou as dependências Python, a partir da raiz do repositório execute:

```fish
uv sync
```

Crie um arquivo .env e preencha as variáveis conforme o arquivo .env.example:

```fish
cp .env.example .env
```

## Inicialização rápida

1. Inicie o modelo Ollama com o Docker Compose. A partir da raiz do repositório execute:

```fish
# inicia os serviços definidos em docker-compose.yml
docker compose up -d

# verifique o status dos containers
docker compose ps
```
