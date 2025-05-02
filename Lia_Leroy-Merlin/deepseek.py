from transformers import pipeline

# Inicializar o modelo
pipe = pipeline(
    model='deepseek-ai/deepseek-llm-7b-chat',
    task='text-generation',
    model_kwargs={'temperature': 0.7, 'max_new_tokens': 512}
)

# Treinar o modelo
pipe("Pergunta: Qual é a história da Leroy Merlin?")
pipe("Pergunta: Quais são os valores da Leroy Merlin?")
pipe("Pergunta: Quem são os fundadores da Leroy Merlin?")
pipe("Pergunta: Quais são os produtos mais populares da Leroy Merlin?")
pipe("Pergunta: Quais são as características dos produtos da Leroy Merlin?")
pipe("Pergunta: Quais são as marcas que a Leroy Merlin vende?")
pipe("Pergunta: Quais são os serviços oferecidos pela Leroy Merlin?")
pipe("Pergunta: Como posso usar o serviço de entrega da Leroy Merlin?")
pipe("Pergunta: Como posso usar o serviço de instalação da Leroy Merlin?")
pipe("Pergunta: Quantos funcionários a Leroy Merlin tem?")
pipe("Pergunta: Quais são os benefícios oferecidos aos funcionários da Leroy Merlin?")
pipe("Pergunta: Como a Leroy Merlin treina seus funcionários?")
pipe("Pergunta: Quem são os clientes da Leroy Merlin?")
pipe("Pergunta: Como a Leroy Merlin trata seus clientes?")
pipe("Pergunta: Quais são as políticas de devolução da Leroy Merlin?")
pipe("Pergunta: Quais são as métricas de atendimento da Leroy Merlin?")
pipe("Pergunta: Como a Leroy Merlin mede a satisfação do cliente?")
pipe("Pergunta: Como a Leroy Merlin lida com reclamações de clientes?")
pipe("Pergunta: Qual é o modelo de negócio da Leroy Merlin?")
pipe("Pergunta: Como a Leroy Merlin ganha dinheiro?")
pipe("Pergunta: Quais são os planos futuros da Leroy Merlin?") 