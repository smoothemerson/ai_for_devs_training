from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(
    model="gpt-oss:20b",
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente útil que responde em português.",
        },
        {
            "role": "user",
            "content": "Por que o céu é azul? limite o texto a 50 palavras.",
        },
    ],
)

print(response.message.content)
