"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           PROMPTBOX v2.0                                       ║
║                     Sovereign AI Legal Assistant                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Architecture: Clean Architecture + Brain Switcher                            ║
║  Author: Caike (TCC FURB 2024)                                                ║
║  License: MIT                                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CHANGELOG v2.0:
- [NEW] Brain Switcher: Troca dinâmica de modelos (Phi-3, Llama-3, GGUF)
- [NEW] Config via YAML: Configurações externalizadas
- [NEW] Provider abstraction: Suporte a Ollama + llama.cpp
- [NEW] Design System: UI profissional estilo Vercel/Linear
- [FIX] Modo offline verdadeiro (HF_HUB_OFFLINE)
- [FIX] Cache otimizado para evitar re-runs do Streamlit
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# OFFLINE MODE - MUST COME BEFORE ANY HUGGINGFACE IMPORTS
# ============================================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  
os.environ["HF_DATASETS_OFFLINE"] = "1"

import re
import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Any
from functools import lru_cache

import requests
import streamlit as st
import sqlite3
import datetime
import time
import unicodedata
import dashboard_ui  # Dashboard B2B Analytics

# ============================================================================
# DATABASE LOGGING
# ============================================================================
DB_PATH = "promptbox.db"

def init_db():
    """Cria tabela de logs se não existir."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  query TEXT,
                  response_time REAL,
                  model_used TEXT)''')
    conn.commit()
    conn.close()

def log_interaction(query: str, time_taken: float, model: str = "Unknown"):
    """Salva interação no banco para métricas do Dashboard."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        local_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("""
            INSERT INTO logs (timestamp, query, response_time, model_used) 
            VALUES (?, ?, ?, ?)
        """, (local_time, query[:500], time_taken, model))
        conn.commit()
        conn.close()
    except Exception:
        pass

# Initialize DB on startup
init_db()

# ============================================================================
# SECURITY MANAGER - Anti-Jailbreak
# ============================================================================
class SecurityManager:
    """Camada de segurança contra prompt injection e intenções ilegais."""
    
    INJECTION_PATTERNS = [
        r"ignore\s+(todas?\s+)?(as\s+)?instru[çc][õo]es",
        r"esque[çc]a\s+(tudo|o\s+contexto|as\s+regras)",
        r"voc[êe]\s+agora\s+[ée]",
        r"aja\s+(como|enquanto)",
        r"finja\s+(ser|que)",
    ]
    
    ILLEGAL_INTENT_PATTERNS = [
        r"como\s+(posso\s+)?(esconder|ocultar|sonegar)\s+.*(bens?|patrimonio|dinheiro)",
        r"como\s+(posso\s+)?(fraudar|falsificar|forjar)",
        r"como\s+(posso\s+)?(subornar|corromper)",
        r"como\s+(posso\s+)?lavar\s+dinheiro",
        r"como\s+(posso\s+)?(matar|assassinar)",
    ]
    
    ILLEGAL_BLOCK_MSG = """🚫 **SOLICITAÇÃO BLOQUEADA**

Sua pergunta sugere intenção de obter orientação para atividade potencialmente ilegal.

Reformule sua pergunta de forma neutra."""

    @classmethod
    def is_safe(cls, query: str) -> tuple[bool, str]:
        """Valida e sanitiza a query."""
        if not query or not query.strip():
            return False, "⚠️ Pergunta vazia."
        
        query = query.strip()
        if len(query) < 3:
            return False, "⚠️ Pergunta muito curta."
        if len(query) > 2000:
            return False, "⚠️ Pergunta muito longa."
        
        query_lower = query.lower()
        
        # Check illegal intent
        for pattern in cls.ILLEGAL_INTENT_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return False, cls.ILLEGAL_BLOCK_MSG
        
        # Check injection
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return False, "⚠️ Reformule sua pergunta de forma objetiva."
        
        # Sanitize
        sanitized = query.replace("{", "｛").replace("}", "｝")
        sanitized = sanitized.replace("<", "＜").replace(">", "＞")
        return True, sanitized


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class ModelConfig:
    """Configuração de um modelo LLM."""
    name: str
    display_name: str
    provider: str  # ollama, llamacpp
    description: str = ""
    context_length: int = 4096
    temperature: float = 0.1
    gguf_path: Optional[str] = None


@dataclass 
class AppConfig:
    """Configuração centralizada da aplicação."""
    
    # App
    app_name: str = "PromptBox"
    version: str = "2.0.0"
    debug: bool = False
    offline_mode: bool = True
    
    # Models
    models: dict[str, ModelConfig] = field(default_factory=dict)
    active_model: str = "phi3"
    
    # RAG
    chroma_dir: str = "./chroma_db"
    collection_name: str = "promptbox"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    distance_threshold: float = 1.5
    
    # Ollama
    ollama_api: str = "http://127.0.0.1:11434/api/generate"
    ollama_health: str = "http://127.0.0.1:11434"
    timeout: int = 360
    health_timeout: int = 2
    
    # UI
    theme: str = "dark"
    accent_color: str = "#7c5cff"
    show_debug: bool = False
    
    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "AppConfig":
        """Carrega configuração do arquivo YAML."""
        config = cls()
        
        path = Path(config_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # App settings
                if 'app' in data:
                    config.app_name = data['app'].get('name', config.app_name)
                    config.version = data['app'].get('version', config.version)
                    config.debug = data['app'].get('debug', config.debug)
                    config.offline_mode = data['app'].get('offline_mode', config.offline_mode)
                
                # Models
                if 'models' in data:
                    for key, model_data in data['models'].items():
                        config.models[key] = ModelConfig(
                            name=model_data.get('name', key),
                            display_name=model_data.get('display_name', key),
                            provider=model_data.get('provider', 'ollama'),
                            description=model_data.get('description', ''),
                            context_length=model_data.get('context_length', 4096),
                            temperature=model_data.get('temperature', 0.1),
                            gguf_path=model_data.get('gguf_path'),
                        )
                
                config.active_model = data.get('active_model', config.active_model)
                
                # RAG
                if 'rag' in data:
                    config.chroma_dir = data['rag'].get('chroma_dir', config.chroma_dir)
                    config.collection_name = data['rag'].get('collection_name', config.collection_name)
                    config.embedding_model = data['rag'].get('embedding_model', config.embedding_model)
                    config.top_k = data['rag'].get('top_k', config.top_k)
                    config.distance_threshold = data['rag'].get('distance_threshold', config.distance_threshold)
                
                # Ollama
                if 'ollama' in data:
                    config.ollama_api = data['ollama'].get('api_url', config.ollama_api)
                    config.ollama_health = data['ollama'].get('health_url', config.ollama_health)
                    config.timeout = data['ollama'].get('timeout', config.timeout)
                    config.health_timeout = data['ollama'].get('health_timeout', config.health_timeout)
                
                # UI
                if 'ui' in data:
                    config.theme = data['ui'].get('theme', config.theme)
                    config.accent_color = data['ui'].get('accent_color', config.accent_color)
                    config.show_debug = data['ui'].get('show_debug', config.show_debug)
                    
            except Exception as e:
                st.warning(f"Could not load config.yaml: {e}. Using defaults.")
        
        # Ensure default model exists
        if not config.models:
            config.models["phi3"] = ModelConfig(
                name="phi3",
                display_name="Phi-3 Mini",
                provider="ollama",
                description="Default model"
            )
        
        return config
    
    def get_active_model(self) -> ModelConfig:
        """Retorna configuração do modelo ativo."""
        return self.models.get(self.active_model, list(self.models.values())[0])


# Global config instance (cached)
@st.cache_resource
def get_config() -> AppConfig:
    return AppConfig.load()


# ============================================================================
# AGENT TYPES
# ============================================================================

class AgentType(Enum):
    """Tipos de agentes especializados."""
    JURIDICO = ("JURIDICO", "⚖️", "Jurídico", "#3B82F6", True)
    FINANCEIRO = ("FINANCEIRO", "📊", "Financeiro", "#10B981", False)
    TECH = ("TECH", "💻", "Tech", "#8B5CF6", False)
    CHAT = ("CHAT", "💬", "Chat", "#71717A", False)
    
    def __init__(self, key: str, icon: str, label: str, color: str, uses_rag: bool):
        self.key = key
        self.icon = icon
        self.label = label
        self.color = color
        self.uses_rag = uses_rag


# ============================================================================
# LLM PROVIDERS (Brain Switcher)
# ============================================================================

class LLMProvider(ABC):
    """Interface abstrata para providers de LLM."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Gera resposta do modelo."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Verifica se o provider está disponível."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Retorna informações do modelo."""
        pass


class OllamaProvider(LLMProvider):
    """Provider para modelos via Ollama."""
    
    def __init__(self, config: AppConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = requests.post(
                self.config.ollama_api,
                json={
                    "model": self.model_config.name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', self.model_config.temperature),
                        "num_predict": kwargs.get('max_tokens', 500),
                    }
                },
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Erro: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "⏱️ Timeout: O modelo demorou demais para responder."
        except requests.exceptions.ConnectionError:
            return "🔌 Erro de conexão: Verifique se o Ollama está rodando (ollama serve)"
        except Exception as e:
            return f"❌ Erro: {str(e)}"
    
    def health_check(self) -> bool:
        try:
            response = requests.get(
                self.config.ollama_health,
                timeout=self.config.health_timeout
            )
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self) -> dict:
        return {
            "name": self.model_config.name,
            "display_name": self.model_config.display_name,
            "provider": "Ollama",
            "context_length": self.model_config.context_length,
        }


class LlamaCppProvider(LLMProvider):
    """Provider para modelos GGUF via llama-cpp-python (futuro)."""
    
    def __init__(self, config: AppConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo GGUF."""
        if not self.model_config.gguf_path:
            return
            
        try:
            from llama_cpp import Llama
            
            gguf_path = Path(self.model_config.gguf_path)
            if gguf_path.exists():
                self.model = Llama(
                    model_path=str(gguf_path),
                    n_ctx=self.model_config.context_length,
                    n_threads=4,
                    verbose=False
                )
        except ImportError:
            pass  # llama-cpp-python not installed
        except Exception as e:
            st.warning(f"Could not load GGUF model: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.model:
            return "❌ Modelo GGUF não carregado. Verifique o caminho no config.yaml"
        
        try:
            output = self.model(
                prompt,
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', self.model_config.temperature),
                stop=["###", "\n\n\n"]
            )
            return output["choices"][0]["text"]
        except Exception as e:
            return f"❌ Erro na geração: {str(e)}"
    
    def health_check(self) -> bool:
        return self.model is not None
    
    def get_model_info(self) -> dict:
        return {
            "name": self.model_config.name,
            "display_name": self.model_config.display_name,
            "provider": "llama.cpp (GGUF)",
            "context_length": self.model_config.context_length,
            "gguf_path": self.model_config.gguf_path,
        }


class PTBRSLMProvider(LLMProvider):
    """Provider para modelo PTBR-SLM via subprocess."""
    
    def __init__(self, config: AppConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.binary_path = Path(model_config.gguf_path or "../ptbr-slm/target/release/ptbr-slm.exe")
        self.tokenizer_path = Path(getattr(model_config, 'tokenizer_path', "../ptbr-slm/data/tokenizer_v16_full/tokenizer.json"))
        self.model_path = Path(getattr(model_config, 'model_checkpoint', "../ptbr-slm/checkpoints/model_final.mpk"))
        self.model_size = getattr(model_config, 'model_size', "85m")
    
    def generate(self, prompt: str, **kwargs) -> str:
        import subprocess
        
        if not self.binary_path.exists():
            return f"❌ Binary PTBR-SLM não encontrado em: {self.binary_path}"
        
        if not self.model_path.exists():
            return f"❌ Checkpoint não encontrado em: {self.model_path}\n💡 O modelo ainda não foi treinado. Aguarde o treino finalizar."
        
        try:
            temperature = kwargs.get('temperature', self.model_config.temperature)
            max_tokens = kwargs.get('max_tokens', 200)
            
            cmd = [
                str(self.binary_path),
                "generate",
                "--model", str(self.model_path),
                "--tokenizer", str(self.tokenizer_path),
                "--prompt", prompt,
                "--max-tokens", str(max_tokens),
                "--model-size", self.model_size,
                "--temperature", str(temperature),
                "--json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode == 0:
                try:
                    response = json.loads(result.stdout.strip())
                    return response.get("full_text", response.get("generated_text", ""))
                except json.JSONDecodeError:
                    # Fallback to raw output
                    return result.stdout.strip()
            else:
                return f"❌ Erro no PTBR-SLM: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "⏱️ Timeout: O modelo demorou demais para responder."
        except FileNotFoundError:
            return f"❌ Binary não encontrado. Compile com: cargo build --release --features cpu"
        except Exception as e:
            return f"❌ Erro: {str(e)}"
    
    def health_check(self) -> bool:
        return self.binary_path.exists() and self.model_path.exists()
    
    def get_model_info(self) -> dict:
        return {
            "name": self.model_config.name,
            "display_name": self.model_config.display_name,
            "provider": "PTBR-SLM (Rust)",
            "context_length": self.model_config.context_length,
            "binary_path": str(self.binary_path),
            "model_path": str(self.model_path),
            "model_size": self.model_size,
        }


class BrainSwitcher:
    """
    Gerenciador de modelos LLM.
    Permite trocar dinamicamente entre diferentes providers/modelos.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._providers: dict[str, LLMProvider] = {}
        self._current_provider: Optional[LLMProvider] = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializa todos os providers configurados."""
        for key, model_config in self.config.models.items():
            if model_config.provider == "ollama":
                self._providers[key] = OllamaProvider(self.config, model_config)
            elif model_config.provider == "llamacpp":
                self._providers[key] = LlamaCppProvider(self.config, model_config)
            elif model_config.provider == "ptbr-slm":
                self._providers[key] = PTBRSLMProvider(self.config, model_config)
        
        # Set active provider
        if self.config.active_model in self._providers:
            self._current_provider = self._providers[self.config.active_model]
        elif self._providers:
            self._current_provider = list(self._providers.values())[0]
    
    def switch_model(self, model_key: str) -> bool:
        """Troca para outro modelo."""
        if model_key in self._providers:
            self._current_provider = self._providers[model_key]
            self.config.active_model = model_key
            return True
        return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Gera resposta usando o provider atual."""
        if self._current_provider:
            return self._current_provider.generate(prompt, **kwargs)
        return "❌ Nenhum modelo configurado."
    
    def health_check(self) -> bool:
        """Verifica saúde do provider atual."""
        if self._current_provider:
            return self._current_provider.health_check()
        return False
    
    def get_current_model_info(self) -> dict:
        """Retorna informações do modelo atual."""
        if self._current_provider:
            return self._current_provider.get_model_info()
        return {}
    
    def get_available_models(self) -> list[str]:
        """Lista modelos disponíveis."""
        return list(self._providers.keys())



# ============================================================================
# RAG ENGINE
# ============================================================================

@dataclass
class RAGEngine:
    """Motor de Retrieval-Augmented Generation."""
    
    client: Optional[Any] = None
    collection: Optional[Any] = None
    embedder: Optional[Any] = None
    config: Optional[AppConfig] = None
    
    @classmethod
    @st.cache_resource
    def initialize(_cls, _version: str = "v2.0") -> "RAGEngine":
        """Factory method com cache."""
        config = get_config()
        engine = RAGEngine(config=config)
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return engine
        
        try:
            engine.client = chromadb.PersistentClient(path=config.chroma_dir)
            
            try:
                engine.collection = engine.client.get_collection(config.collection_name)
            except:
                engine.collection = engine.client.create_collection(config.collection_name)
            
            engine.embedder = SentenceTransformer(
                config.embedding_model,
                local_files_only=config.offline_mode
            )
            
        except Exception as e:
            st.warning(f"RAG initialization failed: {e}")
        
        return engine
    
    def retrieve(self, query: str) -> tuple[list[str], list[dict], list[float]]:
        """Busca híbrida: metadado + semântica."""
        if not all([self.collection, self.embedder]):
            return [], [], []
        
        docs, metas, distances = [], [], []
        
        try:
            # 1. Busca por artigo específico (metadado)
            article_match = re.search(r'art(?:igo)?\.?\s*(\d+)', query.lower())
            law_mentioned = self._detect_law(query)
            
            if article_match:
                article_num = article_match.group(1)
                
                where_filter = {"article": article_num}
                if law_mentioned:
                    where_filter = {
                        "$and": [
                            {"article": article_num},
                            {"source": law_mentioned}
                        ]
                    }
                
                try:
                    results = self.collection.get(
                        where=where_filter,
                        include=["documents", "metadatas"]
                    )
                    
                    if results and results.get("documents"):
                        for doc, meta in zip(
                            results["documents"][:3],
                            results["metadatas"][:3]
                        ):
                            if doc not in docs:
                                docs.append(doc)
                                metas.append(meta)
                                distances.append(0.0)
                except:
                    pass
            
            # 2. Busca semântica (complementar)
            if len(docs) < self.config.top_k:
                remaining = min(self.config.top_k - len(docs), 3)
                embedding = self.embedder.encode([query]).tolist()
                
                results = self.collection.query(
                    query_embeddings=embedding,
                    n_results=remaining + 2,
                    include=["documents", "metadatas", "distances"]
                )
                
                if results.get("documents"):
                    for doc, meta, dist in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                    ):
                        if doc not in docs and len(docs) < self.config.top_k:
                            docs.append(doc)
                            metas.append(meta)
                            distances.append(dist)
            
        except Exception as e:
            if self.config.debug:
                st.error(f"RAG Error: {e}")
        
        return docs, metas, distances
    
    def _detect_law(self, query: str) -> Optional[str]:
        """Detecta lei mencionada na query."""
        query_lower = query.lower()
        
        patterns = {
            "CONSTITUICAO_FEDERAL": ["constituição", "constituicao", "cf"],
            "CLT": ["clt", "trabalhista", "trabalho"],
            "CODIGO_CIVIL": ["código civil", "codigo civil", "cc"],
            "CODIGO_PENAL": ["código penal", "codigo penal", "cp", "penal"],
            "CODIGO_DEFESA_CONSUMIDOR": ["cdc", "consumidor"],
            "LGPD": ["lgpd", "proteção de dados"],
            "MARCO_CIVIL_INTERNET": ["marco civil", "internet"],
        }
        
        for code, keywords in patterns.items():
            if any(kw in query_lower for kw in keywords):
                return code
        
        return None


# ============================================================================
# ROUTER
# ============================================================================

class Router:
    """Classifica intenções e roteia para agentes."""
    
    KEYWORDS = {
        AgentType.JURIDICO: [
            "lei", "artigo", "clt", "direito", "crime", "contrato",
            "demissão", "férias", "fgts", "trabalhista", "constituição",
            "código", "cdc", "lgpd", "justa causa", "rescisão"
        ],
        AgentType.FINANCEIRO: [
            "calcula", "valor", "quanto", "soma", "juros", "investimento",
            "porcentagem", "+", "-", "*", "/", "lucro"
        ],
        AgentType.TECH: [
            "código", "python", "java", "javascript", "bug", "api",
            "programar", "loop", "função", "script", "sql"
        ],
    }
    
    def classify(self, query: str) -> AgentType:
        """Classifica a intenção da query."""
        query_lower = query.lower()
        
        scores = {agent: 0 for agent in AgentType}
        
        for agent, keywords in self.KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[agent] += 1
        
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else AgentType.CHAT


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

PROMPTS = {
    AgentType.JURIDICO: """Você é um assistente jurídico brasileiro especializado.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS o que foi perguntado
2. Use SOMENTE as informações do CONTEXTO fornecido
3. Se não houver contexto relevante, diga: "Não encontrei essa informação na base."
4. NÃO invente informações legais
5. NÃO faça perguntas de volta
6. NÃO sugira próximas perguntas
7. Seja direto e cite os artigos quando possível""",

    AgentType.FINANCEIRO: """Você é um analista financeiro objetivo.

REGRAS:
1. Faça cálculos precisos
2. Seja direto e objetivo
3. NÃO faça perguntas de volta
4. NÃO sugira próximas perguntas""",

    AgentType.TECH: """Você é um engenheiro de software sênior.

REGRAS:
1. Forneça código funcional quando apropriado
2. Use boas práticas e padrões
3. NÃO faça perguntas de volta
4. NÃO sugira próximas perguntas""",

    AgentType.CHAT: """Você é um assistente útil e conciso.

REGRAS:
1. Seja direto
2. NÃO faça perguntas de volta
3. NÃO sugira próximas perguntas""",
}


# ============================================================================
# DESIGN SYSTEM (UI)
# ============================================================================

class DesignSystem:
    """Sistema de design profissional estilo Vercel/Linear."""
    
    CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        /* Colors - Dark Theme */
        --bg-primary: #09090b;
        --bg-secondary: #18181b;
        --bg-tertiary: #27272a;
        --bg-elevated: #1c1c1f;
        
        --text-primary: #fafafa;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        
        --border: #27272a;
        --border-hover: #3f3f46;
        
        --accent: #7c5cff;
        --accent-hover: #9b7fff;
        --accent-muted: rgba(124, 92, 255, 0.15);
        
        --success: #22c55e;
        --success-muted: rgba(34, 197, 94, 0.15);
        
        --error: #ef4444;
        --error-muted: rgba(239, 68, 68, 0.15);
        
        --warning: #f59e0b;
        
        /* Sizing */
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-full: 9999px;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.4);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
        --shadow-glow: 0 0 20px rgba(124, 92, 255, 0.15);
    }
    
    /* Global Reset */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Defaults */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    
    /* Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--border);
    }
    
    .app-logo {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .logo-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--accent), #22d3ee);
        border-radius: var(--radius-sm);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 700;
        color: white;
    }
    
    .app-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    
    .header-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: var(--radius-full);
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-online {
        background: var(--success-muted);
        color: var(--success);
    }
    
    .status-offline {
        background: var(--error-muted);
        color: var(--error);
    }
    
    /* Model Selector */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: var(--radius-full);
        background: var(--bg-tertiary);
        color: var(--text-secondary);
        font-size: 12px;
        font-weight: 500;
        border: 1px solid var(--border);
    }
    
    .model-badge:hover {
        border-color: var(--border-hover);
    }
    
    /* Agent Badges */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: var(--radius-full);
        font-size: 13px;
        font-weight: 600;
        margin: 8px 0;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
    }
    
    /* Response Cards */
    .response-card {
        background: linear-gradient(180deg, var(--bg-elevated), var(--bg-secondary));
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 20px;
        margin-top: 8px;
        box-shadow: var(--shadow-md);
        line-height: 1.7;
        color: var(--text-primary);
    }
    
    .response-card code {
        background: var(--bg-tertiary);
        padding: 2px 6px;
        border-radius: var(--radius-sm);
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 13px;
        color: var(--accent);
    }
    
    .response-card pre {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px;
        overflow-x: auto;
        margin: 12px 0;
    }
    
    .response-card pre code {
        background: none;
        padding: 0;
        color: var(--text-primary);
    }
    
    /* Source Citations */
    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        background: var(--accent-muted);
        color: var(--accent);
        border-radius: var(--radius-full);
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Chat Input */
    .stChatInput > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
    }
    
    .stChatInput input {
        color: var(--text-primary) !important;
    }
    
    .stChatInput input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Status Expander */
    .stExpander {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
    }
    
    /* Selectbox (Model Switcher) */
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { 
        background: var(--border); 
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }
    
    /* Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse { animation: pulse 2s infinite; }
    </style>
    """
    
    @classmethod
    def inject(cls):
        """Injeta CSS na página."""
        st.markdown(cls.CSS, unsafe_allow_html=True)
    
    @staticmethod
    def header(is_online: bool, model_name: str = ""):
        """Renderiza header da aplicação."""
        status_class = "status-online" if is_online else "status-offline"
        status_dot = "●" if is_online else "○"
        status_text = "Online" if is_online else "Offline"
        
        st.markdown(f"""
        <div class="app-header">
            <div class="app-logo">
                <div class="logo-icon">◆</div>
                <span class="app-title">PromptBox</span>
            </div>
            <div class="header-right">
                <div class="model-badge">🧠 {model_name}</div>
                <div class="status-pill {status_class}">
                    <span class="pulse">{status_dot}</span> {status_text}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def agent_badge(agent: AgentType) -> str:
        """Retorna badge HTML do agente."""
        return f"""
        <div class="agent-badge" style="
            background: {agent.color}15;
            color: {agent.color};
            border: 1px solid {agent.color}30;
        ">
            {agent.icon} {agent.label}
        </div>
        """
    
    @staticmethod
    def response_card(content: str, sources: list[str] = None) -> str:
        """Formata resposta em card estilizado."""
        # Escape HTML
        safe = content.replace("<", "&lt;").replace(">", "&gt;")
        
        # Restore code blocks
        safe = re.sub(
            r'```(\w+)?\n(.*?)```',
            r'<pre><code>\2</code></pre>',
            safe, flags=re.DOTALL
        )
        safe = re.sub(r'`([^`]+)`', r'<code>\1</code>', safe)
        safe = safe.replace('\n', '<br>')
        
        # Add sources if available
        sources_html = ""
        if sources:
            tags = "".join([f'<span class="source-tag">📜 {s}</span>' for s in sources[:3]])
            sources_html = f'<div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border);">{tags}</div>'
        
        return f'<div class="response-card">{safe}{sources_html}</div>'


# ============================================================================
# APPLICATION
# ============================================================================

class PromptBoxApp:
    """Aplicação principal."""
    
    def __init__(self):
        self.config = get_config()
        self.brain = BrainSwitcher(self.config)
        self.rag = RAGEngine.initialize()
        self.router = Router()
    
    def _build_prompt(self, query: str, agent: AgentType, context: str = "") -> str:
        """Constrói prompt para o modelo."""
        parts = [PROMPTS.get(agent, "")]
        
        if context:
            parts.append(f"\n\n### CONTEXTO:\n{context}")
        
        parts.append(f"\n\n### PERGUNTA:\n{query}")
        parts.append("\n\n### RESPOSTA:")
        
        return "".join(parts)
    
    def _clean_response(self, text: str) -> str:
        """Remove padrões de tagarelice."""
        cut_patterns = [
            "Pergunta de seguinte",
            "Próxima pergunta",
            "Você gostaria",
            "Quer que eu",
            "Posso ajudar",
            "Alguma dúvida",
            "Se precisar",
        ]
        
        lines = text.split('\n')
        clean = []
        
        for line in lines:
            if any(p.lower() in line.lower() for p in cut_patterns):
                break
            clean.append(line)
        
        return '\n'.join(clean).strip()
    
    def _process(self, query: str) -> tuple[str, list[str]]:
        """Processa query e retorna resposta + fontes."""
        sources = []
        
        with st.status("Processando...", expanded=False) as status:
            # 1. Classificação
            status.write("🔍 Classificando...")
            agent = self.router.classify(query)
            st.markdown(DesignSystem.agent_badge(agent), unsafe_allow_html=True)
            
            # 2. RAG (se aplicável)
            context = ""
            if agent.uses_rag:
                status.write("📚 Buscando na base...")
                docs, metas, dists = self.rag.retrieve(query)
                
                relevant = []
                for doc, meta, dist in zip(docs, metas, dists):
                    if dist <= self.config.distance_threshold:
                        relevant.append(doc)
                        src = meta.get('source', '?')
                        art = meta.get('article', '?')
                        sources.append(f"{src} Art.{art}")
                
                if relevant:
                    context = "\n\n---\n\n".join(relevant)
                    status.write(f"✓ {len(relevant)} documento(s)")
                else:
                    status.write("⚠ Nenhum documento relevante")
            
            # 3. Geração
            status.write("🧠 Gerando...")
            prompt = self._build_prompt(query, agent, context)
            response = self.brain.generate(prompt)
            response = self._clean_response(response)
            
            status.update(label="✓ Concluído", state="complete")
        
        return response, sources
    
    def run(self):
        """Executa aplicação."""
        st.set_page_config(
            page_title="PromptBox",
            page_icon="◆",
            layout="centered",
        )
        
        DesignSystem.inject()
        
        # Initialize state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Header
        model_info = self.brain.get_current_model_info()
        DesignSystem.header(
            is_online=self.brain.health_check(),
            model_name=model_info.get("display_name", "Unknown")
        )
        
        # Sidebar - Model Switcher
        with st.sidebar:
            st.markdown("### ⚙️ Configurações")
            
            available = self.brain.get_available_models()
            current = self.config.active_model
            
            if len(available) > 1:
                selected = st.selectbox(
                    "Modelo",
                    available,
                    index=available.index(current) if current in available else 0,
                    format_func=lambda x: self.config.models[x].display_name
                )
                
                if selected != current:
                    self.brain.switch_model(selected)
                    st.rerun()
            
            st.markdown("---")
            st.markdown(f"**Versão:** {self.config.version}")
            st.markdown(f"**RAG:** {self.rag.collection.count() if self.rag.collection else 0} docs")
            
            # Dashboard toggle
            st.markdown("---")
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.show_dashboard = not st.session_state.get('show_dashboard', False)
                st.rerun()
        
        # Show Dashboard or Chat
        if st.session_state.get('show_dashboard', False):
            try:
                conn = sqlite3.connect('promptbox.db')
                dashboard_ui.render_admin_view(conn)
                conn.close()
            except Exception as e:
                st.error(f"Erro ao carregar dashboard: {e}")
            return
        
        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    st.markdown(
                        DesignSystem.response_card(msg["content"], msg.get("sources")),
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(msg["content"])
        
        # Input
        if query := st.chat_input("Digite sua pergunta..."):
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                # Security check
                is_safe, sanitized_or_error = SecurityManager.is_safe(query)
                
                if not is_safe:
                    # Log blocked query
                    log_interaction(query, 0.0, "SECURITY")
                    st.markdown(
                        DesignSystem.response_card(sanitized_or_error, []),
                        unsafe_allow_html=True
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": sanitized_or_error,
                        "sources": []
                    })
                else:
                    # Process safe query with timing
                    start_time = time.time()
                    response, sources = self._process(sanitized_or_error)
                    elapsed = time.time() - start_time
                    
                    # Log interaction for Dashboard
                    model_name = self.config.active_model
                    log_interaction(sanitized_or_error, elapsed, model_name)
                    
                    st.markdown(
                        DesignSystem.response_card(response, sources),
                        unsafe_allow_html=True
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    app = PromptBoxApp()
    app.run()


if __name__ == "__main__":
    main()
