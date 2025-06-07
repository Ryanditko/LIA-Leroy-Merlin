import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class KnowledgeManager:
    def __init__(self):
        self.knowledge_file = "knowledge/base_conhecimento.json"
        self.learning_file = "knowledge/aprendizado.json"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_file = "knowledge/faiss_index.bin"
        self.texts_file = "knowledge/texts.pkl"
        self.knowledge_base = self._load_knowledge()
        self.learning_base = self._load_learning()
        self.texts, self.index = self._load_or_create_index()

    def _load_knowledge(self) -> Dict:
        """Carrega a base de conhecimento inicial"""
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_learning(self) -> Dict:
        """Carrega a base de aprendizado"""
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"interacoes": [], "novo_conhecimento": {}}

    def _load_or_create_index(self):
        """Carrega ou cria o índice de vetores"""
        if os.path.exists(self.index_file) and os.path.exists(self.texts_file):
            index = faiss.read_index(self.index_file)
            with open(self.texts_file, 'rb') as f:
                texts = pickle.load(f)
        else:
            texts = []
            index = faiss.IndexFlatL2(384)
            # Indexar o conhecimento inicial
            for cat, subs in self.knowledge_base.get('categorias', {}).items():
                for sub, content in subs.items():
                    text = f"{cat} {sub} {json.dumps(content, ensure_ascii=False)}"
                    texts.append(text)
            if texts:
                vectors = self.model.encode(texts)
                index.add(np.array(vectors, dtype=np.float32))
            self._save_index(index, texts)
        return texts, index

    def _save_index(self, index, texts):
        """Salva o índice e os textos"""
        faiss.write_index(index, self.index_file)
        with open(self.texts_file, 'wb') as f:
            pickle.dump(texts, f)

    def add_interaction(self, user_input: str, response: str, feedback: float = None):
        """Adiciona uma nova interação ao histórico"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "feedback": feedback
        }
        self.learning_base["interacoes"].append(interaction)
        with open(self.learning_file, 'w', encoding='utf-8') as f:
            json.dump(self.learning_base, f, ensure_ascii=False, indent=4)

    def add_knowledge(self, category: str, subcategory: str, content: Dict):
        """Adiciona novo conhecimento à base"""
        if category not in self.learning_base["novo_conhecimento"]:
            self.learning_base["novo_conhecimento"][category] = {}
        if subcategory not in self.learning_base["novo_conhecimento"][category]:
            self.learning_base["novo_conhecimento"][category][subcategory] = {}
        
        self.learning_base["novo_conhecimento"][category][subcategory].update(content)
        with open(self.learning_file, 'w', encoding='utf-8') as f:
            json.dump(self.learning_base, f, ensure_ascii=False, indent=4)
        
        # Atualiza o índice
        self._update_index(category, subcategory, content)

    def _update_index(self, category: str, subcategory: str, content: Dict):
        """Atualiza o índice com novo conhecimento"""
        # Converte o conteúdo em texto
        text = f"{category} {subcategory} {json.dumps(content, ensure_ascii=False)}"
        
        # Adiciona ao índice
        self.texts.append(text)
        vector = self.model.encode([text])[0]
        self.index.add(np.array([vector], dtype=np.float32))
        
        # Salva as alterações
        self._save_index(self.index, self.texts)

    def search_knowledge(self, query: str, k: int = 3) -> List[str]:
        """Busca conhecimento relevante"""
        if not self.texts:
            return []
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec, dtype=np.float32), k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

    def get_relevant_knowledge(self, query: str) -> Dict:
        """Obtém conhecimento relevante para uma query"""
        results = self.search_knowledge(query)
        relevant_knowledge = {}
        
        for result in results:
            try:
                # Tenta extrair informações estruturadas do resultado
                parts = result.split(" ", 2)
                if len(parts) >= 3:
                    category, subcategory, content = parts
                    if category not in relevant_knowledge:
                        relevant_knowledge[category] = {}
                    if subcategory not in relevant_knowledge[category]:
                        relevant_knowledge[category][subcategory] = {}
                    
                    # Adiciona o conteúdo ao conhecimento relevante
                    content_dict = json.loads(content)
                    relevant_knowledge[category][subcategory].update(content_dict)
            except:
                continue
        
        return relevant_knowledge

    def get_learning_stats(self) -> Dict:
        """Retorna estatísticas de aprendizado"""
        return {
            "total_interacoes": len(self.learning_base["interacoes"]),
            "novo_conhecimento": len(self.learning_base["novo_conhecimento"]),
            "vector_store_size": len(self.texts)
        } 