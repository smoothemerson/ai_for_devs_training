# Fine-tuning Challenge - Classificação de Mensagens

Este projeto foi desenvolvido para satisfazer o **Desafio Fine-tuning** da Rocketseat, que consiste em treinar um modelo de linguagem para classificar automaticamente mensagens de clientes de uma rede varejista.

## 📋 Objetivo

Realizar o fine-tuning de um modelo BERT para classificar mensagens de clientes em duas categorias:

- **"venda"**: Mensagens relacionadas à intenção de compra de produtos
- **"suporte"**: Mensagens relacionadas a dúvidas ou problemas com produtos

## 🛠️ Tecnologias Utilizadas

- **Python 3.13+**
- **Transformers** (Hugging Face)
- **BERT** (bert-base-uncased)
- **Datasets** para carregamento dos dados
- **Evaluate** para métricas de avaliação
- **PyTorch** como backend

## 📂 Estrutura do Projeto

```
.
├── datasets/
│   ├── train.jsonl  # Dados de treinamento
│   └── test.jsonl   # Dados de teste
├── main.py          # Script principal de treinamento
└── README.md        # Este arquivo
```

## 🚀 Como Executar

1. Instale as dependências:
```bash
uv sync
```

2. Execute o script de treinamento:
```bash
python main.py
```

3. O modelo treinado será salvo no diretório `bert-hate-speech-test/`

## 📊 Datasets

Os datasets utilizados são arquivos JSONL com o formato:
```json
{"prompt": "Olá, gostaria de fazer a aquisição do novo produto", "completion": "venda"}
{"prompt": "tudo bom, queria verificar como funciona a TV Smart x0912", "completion": "suporte"}
```

**Nota:** Os dados foram gerados sinteticamente para fins didáticos.

## 🎯 Resultados

O modelo foi treinado por 3 épocas e avaliado usando a métrica de acurácia. Os melhores checkpoints foram salvos automaticamente durante o treinamento.

### Métricas de Treinamento
- **Training Loss**: 0.0565
- **Training Runtime**: 37.66s
- **Training Samples/Second**: 39.83
- **Épocas Completadas**: 3.0

### Métricas de Avaliação
- **Eval Loss**: 0.0006
- **Eval Accuracy**: 100.00% (1.0000)
- **Eval Runtime**: 0.12s
- **Eval Samples/Second**: 837.07

O modelo alcançou **100% de acurácia** no conjunto de teste, demonstrando excelente capacidade de classificar mensagens entre "venda" e "suporte".

---

**Desenvolvido como parte do desafio de Fine-tuning da Rocketseat**