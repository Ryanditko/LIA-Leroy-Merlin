"""| Arquivo     | Função                                        |
| ----------- | --------------------------------------------- |
| `main.py`   | Interface de linha de comando (modo terminal) |
| `ingest.py` | Alimenta a LIA com documentos                 |
| `app.py`    | Interface visual em navegador (Streamlit)     |
| `chat.py`   | Processa perguntas e gera respostas           |
"""

from fastapi import FastAPI, Request
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# Carregar embeddings
embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Carregar vetor de documentos
db = FAISS.load_local('faiss_index', embeddings)
retriever = db.as_retriever()

# Inicializar modelo LLM
pipe = pipeline(
    model='deepseek-ai/deepseek-llm-7b-chat',
    task='text-generation',
    model_kwargs={'temperature': 0.7, 'max_new_tokens': 512}
)

llm = HuggingFacePipeline(pipeline=pipe)

# Construção da QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Inicializar API FastAPI
app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Lia está rodando localmente dentro do sistema'}

@app.post('/chat')
async def chat(request: Request):
    data = await request.json()
    question = data.get('question', '')
    if not question:
        return {'error': 'Pergunta inválida.'}
    
    response = qa_chain.run(question)
    return {'resposta': response}

# Treinar o modelo
pipe("Pergunta: Qual é o modelo de negócio da Leroy Merlin?")