import json
import os
import time

import requests

start_time = time.time()

questions = [
    "Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?",
    "Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?",
    "Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?",
    "Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?",
    "Quais são os principais aspectos da crítica social e política presentes em 'Os Sertões'? Como esses aspectos refletem a visão do autor sobre o Brasil da época?",
]

api_url_base = "http://localhost:8000"

rag_urls = [
    f"{api_url_base}/chat/naive_rag",
    f"{api_url_base}/chat/parent_rag",
    f"{api_url_base}/chat/rerank_rag",
]

folder_name = "results"
os.makedirs(folder_name, exist_ok=True)

for question in questions:
    payload = {"prompt": question}

    for api_url in rag_urls:
        try:
            response = requests.post(
                api_url, json=payload, headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            answer = json.dumps(response.json(), ensure_ascii=False, indent=2)

            file_name = f"{folder_name}/{api_url.split('/')[-1]}.txt"

            os.makedirs(folder_name, exist_ok=True)

            print(f"Resposta obtida da API {api_url} para a pergunta: {question}")
            with open(file_name, "a", encoding="utf-8") as file:
                file.write(f"Pergunta: {question}\n")
                file.write(f"Código de status: {response.status_code}\n")
                file.write(f"Resposta: {answer}\n")
                file.write("\n")

        except requests.exceptions.RequestException as e:
            file_name = f"{folder_name}/{api_url.split('/')[-1]}.txt"

            with open(file_name, "a", encoding="utf-8") as file:
                file.write(f"Question: {question}\n")
                file.write(f"Error: {e}\n")
                file.write("\n")

end_time = time.time()
print(f"Script execution time: {end_time - start_time:.2f} seconds")
