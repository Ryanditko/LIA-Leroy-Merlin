# LIA - Assistente Leroy Merlin

Este é um projeto para criar um assistente de chatbot para a Leroy Merlin. O assistente é baseado em um modelo de linguagem de última geração (LLM) e é capaz de responder a perguntas sobre a empresa, seus produtos, seus serviços, seus funcionários, seus clientes, suas métricas de atendimento, seu modelo de negócio, etc.

## Estrutura do Projeto

O projeto é dividido em vários arquivos:

- `main.py`: Este é o arquivo principal do projeto. Ele contém o código para inicializar o assistente e para criar uma API para interagir com o assistente.
- `deepseek.py`: Este é o arquivo que contém o código para inicializar o modelo de linguagem.
- `app.py`: Este é o arquivo que contém o código para criar uma interface visual para interagir com o assistente.
- `chat.py`: Este é o arquivo que contém o código para processar perguntas e gerar respostas.

## Como Usar o Projeto

Para usar o projeto, você precisa ter o Python instalado no seu computador. Você também precisa instalar as seguintes bibliotecas Python:

- `fastapi`
- `transformers`
- `dotenv`
- `streamlit`

Você pode instalar essas bibliotecas usando o seguinte comando:

```bash
pip install fastapi transformers dotenv streamlit
```

Depois de instalar as bibliotecas, você pode iniciar o servidor da API usando o seguinte comando:

```bash
uvicorn main:app --reload
```

Você pode então acessar a interface do assistente em um navegador da web, digitando `http://localhost:8000` na barra de endereços do seu navegador.
