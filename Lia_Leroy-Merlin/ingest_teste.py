from ingest import load_documents

from langchain.document_loaders import PyPDFLoader

# Teste simples para verificar se o PDF pode ser carregado
pdf_path = 'knowledge/Modelo_teste_LIA.pdf'  
try:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[TESTE] Total de documentos carregados do PDF: {len(documents)}")
except Exception as e:
    print(f"[TESTE] Erro ao carregar o PDF: {e}")