"""Embedding generation system supporting both cloud and local AI models."""

import asyncio
import time
from typing import List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np
import requests
import openai
import ollama
from sentence_transformers import SentenceTransformer
from loguru import logger

from ..models import CodeChunk, EmbeddingModel
from ..config import config


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """Initialize OpenAI provider."""
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536  # Ada-002 dimension
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            self.client.embeddings.create(input=["test"], model=self.model)
            return True
        except Exception:
            return False


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Hugging Face embedding provider."""
    
    def __init__(self, api_key: str, model: str = "microsoft/codebert-base"):
        """Initialize Hugging Face provider."""
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        self.dimension = 768  # CodeBERT dimension
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Hugging Face API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json={"inputs": text}
                )
                response.raise_for_status()
                embedding = response.json()
                
                # Handle different response formats
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        # Take mean of token embeddings
                        embedding = np.mean(embedding, axis=0).tolist()
                    else:
                        embedding = embedding
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"HuggingFace embedding error for text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.dimension)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    async def is_available(self) -> bool:
        """Check if Hugging Face API is available."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"inputs": "test"}
            )
            return response.status_code == 200
        except Exception:
            return False


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama local embedding provider."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "codegemma"):
        """Initialize Ollama provider."""
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        self.dimension = 768  # Default dimension, will be updated after first call
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        embeddings = []
        
        for text in texts:
            try:
                response = self.client.embeddings(
                    model=self.model,
                    prompt=text
                )
                embedding = response['embedding']
                embeddings.append(embedding)
                
                # Update dimension if needed
                if len(embedding) != self.dimension:
                    self.dimension = len(embedding)
                    
            except Exception as e:
                logger.error(f"Ollama embedding error: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.dimension)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            models = self.client.list()
            return any(model['name'].startswith(self.model) for model in models['models'])
        except Exception:
            return False


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence transformer provider."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """Initialize sentence transformer provider."""
        self.model_name = model_name
        self.model = None
        self.dimension = 768
    
    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformer."""
        self._load_model()
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding error: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    async def is_available(self) -> bool:
        """Check if model is available."""
        try:
            self._load_model()
            return True
        except Exception:
            return False


class EmbeddingGenerator:
    """Main embedding generator that manages different providers."""
    
    def __init__(self):
        """Initialize embedding generator."""
        self.providers: Dict[str, EmbeddingProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available embedding providers."""
        # OpenAI provider
        if config.ai_models.openai_api_key:
            try:
                self.providers['openai'] = OpenAIEmbeddingProvider(
                    api_key=config.ai_models.openai_api_key
                )
                logger.info("Initialized OpenAI embedding provider")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        # Hugging Face provider
        if config.ai_models.huggingface_api_key:
            try:
                self.providers['huggingface'] = HuggingFaceEmbeddingProvider(
                    api_key=config.ai_models.huggingface_api_key
                )
                logger.info("Initialized Hugging Face embedding provider")
            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face provider: {e}")
        
        # Ollama provider
        try:
            self.providers['ollama'] = OllamaEmbeddingProvider(
                host=config.ai_models.ollama_host,
                model=config.ai_models.ollama_model
            )
            logger.info("Initialized Ollama embedding provider")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
        
        # Sentence Transformer fallback
        try:
            self.providers['sentence_transformer'] = SentenceTransformerProvider()
            logger.info("Initialized SentenceTransformer embedding provider")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer provider: {e}")
    
    async def generate_chunk_embeddings(self, chunks: List[CodeChunk], provider_name: Optional[str] = None) -> Dict[str, List[float]]:
        """Generate embeddings for code chunks."""
        if not chunks:
            return {}
        
        # Select provider
        provider = self._get_provider(provider_name)
        if not provider:
            raise ValueError(f"No available embedding provider")
        
        # Prepare texts for embedding
        texts = []
        chunk_ids = []
        
        for chunk in chunks:
            # Combine content with metadata for better embeddings
            text = self._prepare_chunk_text(chunk)
            texts.append(text)
            chunk_ids.append(chunk.id)
        
        # Generate embeddings in batches
        batch_size = config.indexing.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await provider.generate_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Create mapping
        embeddings_map = dict(zip(chunk_ids, all_embeddings))
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks using {provider.__class__.__name__}")
        return embeddings_map

    async def generate_embeddings(self, texts: List[str], provider_name: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for text queries."""
        if not texts:
            return []

        # Select provider
        provider = self._get_provider(provider_name)
        if not provider:
            raise ValueError(f"No available embedding provider")

        # Generate embeddings
        embeddings = await provider.generate_embeddings(texts)

        logger.info(f"Generated embeddings for {len(texts)} texts using {provider.__class__.__name__}")
        return embeddings

    def _prepare_chunk_text(self, chunk: CodeChunk) -> str:
        """Prepare chunk text for embedding generation."""
        # Combine content with metadata
        parts = []
        
        # Add type and name information
        if chunk.name:
            parts.append(f"{chunk.node_type.value}: {chunk.name}")
        
        # Add file context
        file_name = chunk.file_path.split('/')[-1]
        parts.append(f"File: {file_name}")
        
        # Add the actual code content
        parts.append(chunk.content)
        
        return "\n".join(parts)
    
    def _get_provider(self, provider_name: Optional[str] = None) -> Optional[EmbeddingProvider]:
        """Get embedding provider by name or default."""
        if provider_name and provider_name in self.providers:
            return self.providers[provider_name]
        
        # Use default provider
        default_model = config.ai_models.default_embedding_model
        if default_model == "cloud":
            cloud_model = config.ai_models.default_cloud_model
            if cloud_model in self.providers:
                return self.providers[cloud_model]
        elif default_model == "local":
            if "ollama" in self.providers:
                return self.providers["ollama"]
            elif "sentence_transformer" in self.providers:
                return self.providers["sentence_transformer"]
        
        # Fallback to any available provider
        for provider in self.providers.values():
            return provider
        
        return None
    
    async def get_available_providers(self) -> Dict[str, bool]:
        """Get status of all providers."""
        status = {}
        for name, provider in self.providers.items():
            try:
                status[name] = await provider.is_available()
            except Exception:
                status[name] = False
        return status
    
    def get_embedding_dimension(self, provider_name: Optional[str] = None) -> int:
        """Get embedding dimension for a provider."""
        provider = self._get_provider(provider_name)
        if provider:
            return provider.get_embedding_dimension()
        return 768  # Default dimension
