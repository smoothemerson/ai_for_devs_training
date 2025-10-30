# Desenvolvendo chatbots com Ollama

Este projeto contém código de exemplo para construir chatbots usando um modelo Ollama rodando localmente via Docker e uma interface em Streamlit.

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

2. Inicie a aplicação Streamlit que fornece a interface do chatbot:

```fish
streamlit run src/main.py
```

3. Abra o navegador no endereço padrão do Streamlit:

- http://localhost:8501

A aplicação Streamlit irá se conectar ao modelo Ollama local iniciado via Docker Compose para executar inferência.

## Como funciona

- Entradas: mensagens de usuário submetidas pela interface Streamlit
- Saídas: respostas geradas pelo modelo e exibidas na interface
