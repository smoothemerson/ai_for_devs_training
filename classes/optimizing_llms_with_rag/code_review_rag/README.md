# RAG Code Review

Este projeto contém código de exemplo para o desenvolvimento de LLM utilizando RAG de um projeto do LangChain envolvendo códigos escritos em Python. O objetivo é realizar a revisão de código recuperado pelo RAG e responder à pergunta do usuário com base nas informações obtidas.

## Pré-requisitos

- Docker e Docker Compose instalados e em execução
- Python e pip
- (Opcional) um ambiente virtual Python

Se ainda não instalou as dependências Python, a partir da raiz do repositório execute:

```fish
python -m venv .venv
```

```fish
source ./.venv/bin/activate.fish
```

```fish
pip install -r requirements.txt
```

## Inicialização rápida

1. Inicie o modelo Ollama com o Docker Compose. A partir da raiz do repositório execute:

```fish
# inicia os serviços definidos em docker-compose.yml
docker compose up -d

# verifique o status dos containers
docker compose ps
```
