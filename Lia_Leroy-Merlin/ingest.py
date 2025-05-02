"""| Etapa                                         | Objetivo                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------- |
| ✅ **1. Base local**                           | Configurar estrutura com FastAPI, DeepSeek e FAISS.                       |
| ✅ **2. Ingestão**                             | Script `ingest.py` para processar a base Knowloadge.                      |
| ⏳ **3. Frontend simples**                     | Criar uma interface local para interação (CLI, Web, ou Streamlit).        |
| ⏳ **4. Personalização por usuário**           | Salvar histórico de conversas/local memory para adaptação ao colaborador. |
| ⏳ **5. Logs de uso e métricas básicas**       | Salvar interações em JSON ou banco local para análise.                    |
| ⏳ **6. Modo colaborador vs cliente**          | Criar perfis de acesso com tipos diferentes de respostas.                 |
| ⏳ **7. Integração futura com API da empresa** | (Etapa extra) puxar conteúdos ou dados do sistema interno da Leroy.       |
"""
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# Caminho para a pasta de conhecimento da Leroy Merlin
KNOWLEDGE_PATH = 'knowledge'

def load_documents():
    docs = []
    for root, _, files in os.walk(KNOWLEDGE_PATH): 
        for file in files:
            path = os.path.join(root, file) 
            if file.endswith('pdf'):
                loader = PyPDFLoader(path)
            elif file.endswith('txt'):
                loader = TextLoader(path)
            elif file.endswith('docx'):
                loader = Docx2txtLoader(path) 
            else:
                continue
            docs.extend(loader.load())
    return docs

def ingest():
    print('[LIA] Carregando os documentos...')
    documents = load_documents()

    print('[LIA] Separando os documentos em partes (chunks)...') 
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
    chunks = splitter.split_documents(documents) 

    print('[LIA] Gerando a documentação de embeddings...') 
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 

    print('[LIA] Construindo a base vetorial com FAISS...')
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local('db_faiss')

    print('[LIA] Ingestão concluída com sucesso!')

if __name__ == '__main__':
    ingest()
