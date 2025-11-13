# Desafio RAG - Os Sertões

## 📖 Descrição

Este projeto implementa três diferentes estruturas de **Recuperação e Geração de Respostas (RAG)** para responder questões sobre o livro **"Os Sertões"** de Euclides da Cunha. O sistema utiliza modelos de linguagem local (Ollama) e diferentes estratégias de recuperação de informações para fornecer respostas precisas e contextualizadas.

## 🏗️ Arquitetura

O projeto implementa três abordagens distintas de RAG:

### 1. Naive RAG
- **Estratégia**: Divisão simples do documento em chunks de tamanho fixo
- **Chunk Size**: 4000 caracteres com overlap de 20
- **Recuperação**: Top-3 documentos mais similares
- **Características**: Abordagem direta e rápida

### 2. Parent RAG
- **Estratégia**: Hierarquia de documentos (pais e filhos)
- **Child Chunks**: 500 caracteres (overlap 50) para busca
- **Parent Chunks**: 4000 caracteres (overlap 200) para contexto
- **Vantagem**: Busca granular com contexto amplo

### 3. Rerank RAG
- **Estratégia**: Re-ranqueamento com Cohere Rerank v3.5
- **Recuperação Inicial**: Top-10 documentos
- **Re-ranking**: Reduz para top-3 mais relevantes
- **Vantagem**: Maior precisão na seleção de contexto

## 🚀 Tecnologias Utilizadas

- **Python 3.13+**
- **FastAPI** - API REST para endpoints
- **LangChain** - Framework para LLM e RAG
- **Ollama** - Servidor de modelos local
- **ChromaDB** - Banco de dados vetorial
- **Cohere** - Serviço de re-ranking
- **Docker** - Containerização
- **UV** - Gerenciamento de dependências

### Modelos Utilizados

- **LLM**: `llama3.2:3b` (Ollama)
- **Embeddings**: `nomic-embed-text:v1.5` (Ollama)
- **Rerank**: `rerank-v3.5` (Cohere)

## 📋 Pré-requisitos

- Python 3.13+
- Docker e Docker Compose
- UV (gerenciador de pacotes)
- NVIDIA GPU (recomendado para Ollama)
- Chave API do Cohere

## ⚙️ Instalação e Configuração

### 1. Clone o repositório
```bash
git clone https://github.com/smoothemerson/ai_for_devs_training.git
cd challenges/rag_challenge
```

### 2. Configure as variáveis de ambiente
```bash
cp .env.example .env
# Edite o arquivo .env e adicione sua COHERE_API_KEY
```

### 3. Instale as dependências
```bash
uv sync
```

### 4. Adicione o documento
Coloque o PDF do livro "Os Sertões" na pasta `document/` com o nome `sertoes_livro_euclides`.

### 5. Execute com Docker
```bash
# Inicie os serviços
docker-compose up -d
```

### 6. Execução local (alternativa)
```bash
# Certifique-se que o Ollama esteja rodando
ollama serve

# Baixe os modelos necessários
ollama pull llama3.2:3b
ollama pull nomic-embed-text:v1.5

# Execute a aplicação
uv run python -m src.main
```

## 🔧 Uso da API

### Endpoints Disponíveis

- **GET** `/` - Redireciona para documentação
- **GET** `/healthcheck` - Verificação de saúde
- **POST** `/chat/naive_rag` - Chat com Naive RAG
- **POST** `/chat/parent_rag` - Chat com Parent RAG  
- **POST** `/chat/rerank_rag` - Chat com Rerank RAG

## 🧪 Testes e Avaliação

### Script de Teste Automático

Execute o script `questions.py` para testar todas as abordagens RAG:

```bash
uv run python questions.py
```

Este script:
- Faz requisições para todos os endpoints RAG
- Testa todas as 5 questões de avaliação
- Salva os resultados na pasta `results/`
- Mede o tempo total de execução

### Resultados

Os resultados são salvos em:
- `results/naive_rag.txt`
- `results/parent_rag.txt`
- `results/rerank_rag.txt`

## 👨‍💻 Autor

**Emerson Rocha**
- Email: emersonfaria019@gmail.com

## 📚 Referências

- [Os Sertões - Euclides da Cunha (PDF)](https://fundar.org.br/wp-content/uploads/2021/06/os-sertoes.pdf)
- [LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)
- [Ollama](https://ollama.com)
- [Cohere Rerank](https://cohere.com)
