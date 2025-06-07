# LIA - Assistente Virtual Leroy Merlin

A LIA é uma assistente virtual inteligente desenvolvida para auxiliar clientes da Leroy Merlin com informações sobre produtos, reformas, manutenção e muito mais.

## 🚀 Funcionalidades

- Chat interativo com interface moderna
- Respostas especializadas em produtos de construção e decoração
- Dicas de reforma e manutenção
- Suporte técnico especializado
- Interface responsiva e amigável

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd [NOME_DO_DIRETÓRIO]
```

2. Crie um ambiente virtual Python:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
```
MODEL_NAME=microsoft/phi-2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 🎮 Como Usar

1. Inicie a aplicação:
```bash
streamlit run app.py
```

2. Acesse a interface web no navegador (geralmente em http://localhost:8501)

3. Digite suas perguntas na caixa de texto e interaja com a LIA

## 📝 Exemplos de Uso

- "Quais são os melhores produtos para reforma de banheiro?"
- "Como posso fazer uma pequena reforma na cozinha?"
- "Quais ferramentas preciso para instalar um chuveiro?"
- "Me ajude a escolher tintas para a sala"

## 🛠️ Tecnologias Utilizadas

- Python 3.8+
- Streamlit
- Transformers (Hugging Face)
- LangChain
- FAISS
- Sentence Transformers

## 🤝 Contribuindo

1. Faça um Fork do projeto
2. Crie uma Branch para sua Feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a Branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Suporte

Para suporte, envie um email para [SEU_EMAIL] ou abra uma issue no repositório. 