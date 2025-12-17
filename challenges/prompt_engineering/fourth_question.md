- Prompt utilizado:

```
<CONTEXTO>
Aja como um expert em python no framework FastAPI. Seu objetivo é devolver uma resposta para um usuário que não entende de python no framework FastAPI.

<TAREFA>
Elabore uma explicação clara para um leigo.

<REQUISITOS>
1. Explicação clara e objetiva, a resposta não deve ser longa e deve ser clara sem complexidade.
2. Explicação organizada de forma que se tiver listagens, use bullet points.

<PERGUNTA>
Crie um endpoint com FastAPI que valide e processe a entrada de um objeto do tipo `Item`. Siga as especificações abaixo:
    1. Estrutura do `Item`:
      1. **nome**: `string` com tamanho máximo de 25 caracteres.
      2. **valor**: `float`
      3. **data**: valor do tipo *date*, que não pode ser superior à data atual.
    2. Requisitos
      1. O endpoint deve validar os valores recebidos.
      2. Após a validação, o corpo da requisição (`Item`) deve ser retornado com um novo campo adicional: `uuid`. Este campo deve conter um identificador único gerado dinamicamente.
```

- Resposta obtida:
```
Endpoint FastAPI para processar Item
Aqui está o código direto ao ponto:

python"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from datetime import date
from uuid import uuid4

app = FastAPI()

class Item(BaseModel):
    nome: str = Field(..., max_length=25)
    valor: float
    data: date
    
    @field_validator('data')
    def validar_data(cls, v):
        if v > date.today():
            raise ValueError('Data não pode ser superior à data atual')
        return v

class ItemResposta(Item):
    uuid: str

@app.post("/processar-item", response_model=ItemResposta)
def processar_item(item: Item):
    # Gera UUID único
    item_dict = item.model_dump()
    item_dict['uuid'] = str(uuid4())
    
    return ItemResposta(**item_dict)
"""

Como funciona:
Validações automáticas:

nome: FastAPI limita a 25 caracteres automaticamente
valor: Aceita apenas números decimais
data: Validador customizado bloqueia datas futuras

Fluxo:

Recebe o JSON com nome, valor e data
Valida os dados automaticamente
Gera um UUID único
Retorna tudo junto (dados originais + UUID)

Exemplo de uso:
POST /processar-item
{
  "nome": "Produto X",
  "valor": 99.90,
  "data": "2024-12-15"
}
Resposta:
{
  "nome": "Produto X",
  "valor": 99.90,
  "data": "2024-12-15",
  "uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}

Se algo estiver errado (nome muito longo, data futura), FastAPI retorna erro automático com detalhes.
```
