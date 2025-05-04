import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# Caminhos e parâmetros
KNOWLEDGE_PATH = 'knowledge'
VECTOR_DB_PATH = 'db_faiss'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def load_documents():
    docs = []
    for root, _, files in os.walk(KNOWLEDGE_PATH): 
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                elif file.endswith('.txt'):
                    loader = TextLoader(path)
                elif file.endswith('.docx'):
                    loader = Docx2txtLoader(path)
                else:
                    print(f"[LIA] Ignorando arquivo não suportado: {file}")
                    continue
                docs.extend(loader.load())
            except Exception as e:
                print(f"[LIA] Erro ao carregar {file}: {e}")
    print(f"[LIA] Total de documentos carregados: {len(docs)}")
    return docs

def ingest(save_path=VECTOR_DB_PATH):
    print('[LIA] Iniciando processo de ingestão...')

    documents = load_documents()
    if not documents:
        print("[LIA] Nenhum documento encontrado.")
        return

    print('[LIA] Fragmentando documentos em chunks...')
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"[LIA] Total de chunks gerados: {len(chunks)}")

    print('[LIA] Gerando embeddings com modelo:', EMBEDDING_MODEL)
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)

    print('[LIA] Construindo base vetorial com FAISS...')
    db = FAISS.from_documents(chunks, embeddings)

    print(f'[LIA] Salvando base vetorial em: {save_path}')
    db.save_local(save_path)

    print('[LIA] Ingestão concluída com sucesso!')
    return db  # útil para testes

if __name__ == '__main__':
    ingest()
