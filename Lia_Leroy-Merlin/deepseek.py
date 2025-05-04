from transformers import pipeline
from langchain.prompts import PromptTemplate

# Prompt customizado (caso queira importar para outros scripts)
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template='''Você é a LIA, uma IA assistente da Leroy Merlin. Sua missão é ajudar colaboradores e clientes com respostas diretas e bem informadas, sempre com base no conteúdo da empresa.

Use a seguinte base de conhecimento como contexto:

{context}

Agora, responda a esta pergunta de forma clara e objetiva:

{question}
''')

# Inicializa e retorna o pipeline do modelo DeepSeek
def load_deepseek_pipeline():
    return pipeline(
        model='deepseek-ai/deepseek-llm-7b-chat',
        task='text-generation',
        model_kwargs={'temperature': 0.5}
    )

# Função de teste simples
def test_deepseek():
    try:
        print("Iniciando o carregamento do modelo...")
        pipe = load_deepseek_pipeline()
        
        # Adicione o contexto aqui
        context = "A Leroy Merlin é uma empresa de varejo de materiais de construção que oferece uma ampla gama de produtos e serviços, incluindo entrega e instalação."
        
        # Pergunta que você deseja fazer
        question = "O que ela oferece em termos de serviços e produtos?"
        
        # Chamada ao modelo com o contexto
        response = pipe(f"{context} {question}")
        
        print("Modelo carregado com sucesso.")
        print("Texto gerado:", response[0]['generated_text'])
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Teste unitário direto se rodar standalone
if __name__ == "__main__":
    test_deepseek()
