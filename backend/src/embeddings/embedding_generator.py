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
        logger.debug(f"OpenAI: Generating embeddings for {len(texts)} texts using model {self.model}")

        # Log text statistics
        total_chars = sum(len(text) for text in texts)
        avg_chars = total_chars / len(texts) if texts else 0
        logger.debug(f"OpenAI: Text stats - total chars: {total_chars}, avg chars per text: {avg_chars:.1f}")

        try:
            import time
            start_time = time.time()

            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            api_time = time.time() - start_time
            embeddings = [embedding.embedding for embedding in response.data]

            logger.debug(f"OpenAI: API call completed in {api_time:.2f}s")
            logger.debug(f"OpenAI: Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
            # Handle usage information safely
            usage = getattr(response, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 'unknown')
            else:
                prompt_tokens = 'unknown'
            logger.debug(f"OpenAI: Usage - prompt tokens: {prompt_tokens}")

            return embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            logger.debug(f"OpenAI: Failed texts count: {len(texts)}, first text preview: {texts[0][:100] if texts else 'none'}...")
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
        logger.debug(f"HuggingFace: Generating embeddings for {len(texts)} texts using model {self.model}")

        # Log text statistics
        total_chars = sum(len(text) for text in texts)
        avg_chars = total_chars / len(texts) if texts else 0
        logger.debug(f"HuggingFace: Text stats - total chars: {total_chars}, avg chars per text: {avg_chars:.1f}")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        embeddings = []

        import time
        start_time = time.time()

        for i, text in enumerate(texts):
            text_start_time = time.time()

            try:
                logger.debug(f"HuggingFace: Processing text {i+1}/{len(texts)} (length: {len(text)} chars)")

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
                        logger.debug(f"HuggingFace: Text {i+1} returned token embeddings, taking mean")
                        embedding = np.mean(embedding, axis=0).tolist()
                    else:
                        embedding = embedding
                        logger.debug(f"HuggingFace: Text {i+1} returned sentence embedding")

                embeddings.append(embedding)

                text_time = time.time() - text_start_time
                logger.debug(f"HuggingFace: Text {i+1} processed in {text_time:.2f}s, embedding dimension: {len(embedding)}")

            except Exception as e:
                logger.error(f"HuggingFace embedding error for text {i+1}: {e}")
                logger.debug(f"HuggingFace: Failed text preview: {text[:100]}...")
                # Return zero vector as fallback
                fallback_embedding = [0.0] * self.dimension
                embeddings.append(fallback_embedding)
                logger.debug(f"HuggingFace: Using fallback zero vector with dimension {len(fallback_embedding)}")

        total_time = time.time() - start_time
        logger.debug(f"HuggingFace: Generated {len(embeddings)} embeddings in {total_time:.2f}s")
        logger.debug(f"HuggingFace: Average time per text: {total_time/len(texts):.3f}s")

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

    def __init__(self, host: str = "http://localhost:11434", model: str = "codegemma", api_key: Optional[str] = None):
        """Initialize Ollama provider."""
        self.host = host
        self.model = model
        self.api_key = api_key

        # Initialize client with optional API key authentication
        client_kwargs = {}
        if api_key:
            client_kwargs['headers'] = {'Authorization': f'Bearer {api_key}'}

        self.client = ollama.Client(host=host, **client_kwargs)
        self.dimension = 768  # Default dimension, will be updated after first call
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        logger.debug(f"Ollama: Generating embeddings for {len(texts)} texts using model {self.model}")

        # Log text statistics
        total_chars = sum(len(text) for text in texts)
        avg_chars = total_chars / len(texts) if texts else 0
        logger.debug(f"Ollama: Text stats - total chars: {total_chars}, avg chars per text: {avg_chars:.1f}")

        embeddings = []

        import time
        start_time = time.time()

        for i, text in enumerate(texts):
            text_start_time = time.time()

            try:
                logger.debug(f"Ollama: Processing text {i+1}/{len(texts)} (length: {len(text)} chars)")

                response = self.client.embeddings(
                    model=self.model,
                    prompt=text
                )
                embedding = response['embedding']
                embeddings.append(embedding)

                text_time = time.time() - text_start_time
                logger.debug(f"Ollama: Text {i+1} processed in {text_time:.2f}s, embedding dimension: {len(embedding)}")

                # Update dimension if needed
                if len(embedding) != self.dimension:
                    logger.debug(f"Ollama: Updating dimension from {self.dimension} to {len(embedding)}")
                    self.dimension = len(embedding)

            except Exception as e:
                logger.error(f"Ollama embedding error for text {i+1}: {e}")
                logger.debug(f"Ollama: Failed text preview: {text[:100]}...")
                # Return zero vector as fallback
                fallback_embedding = [0.0] * self.dimension
                embeddings.append(fallback_embedding)
                logger.debug(f"Ollama: Using fallback zero vector with dimension {len(fallback_embedding)}")

        total_time = time.time() - start_time
        logger.debug(f"Ollama: Generated {len(embeddings)} embeddings in {total_time:.2f}s")
        logger.debug(f"Ollama: Average time per text: {total_time/len(texts):.3f}s")

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
        logger.debug(f"SentenceTransformer: Generating embeddings for {len(texts)} texts using model {self.model_name}")

        # Log text statistics
        total_chars = sum(len(text) for text in texts)
        avg_chars = total_chars / len(texts) if texts else 0
        logger.debug(f"SentenceTransformer: Text stats - total chars: {total_chars}, avg chars per text: {avg_chars:.1f}")

        self._load_model()
        logger.debug(f"SentenceTransformer: Model loaded, expected dimension: {self.dimension}")

        try:
            import time
            start_time = time.time()

            embeddings = self.model.encode(texts)

            encode_time = time.time() - start_time
            embeddings_list = embeddings.tolist()

            logger.debug(f"SentenceTransformer: Encoding completed in {encode_time:.2f}s")
            logger.debug(f"SentenceTransformer: Generated {len(embeddings_list)} embeddings")
            logger.debug(f"SentenceTransformer: Actual embedding dimension: {len(embeddings_list[0]) if embeddings_list else 0}")
            logger.debug(f"SentenceTransformer: Average time per text: {encode_time/len(texts):.3f}s")

            return embeddings_list

        except Exception as e:
            logger.error(f"SentenceTransformer embedding error: {e}")
            logger.debug(f"SentenceTransformer: Failed texts count: {len(texts)}")
            logger.debug(f"SentenceTransformer: First text preview: {texts[0][:100] if texts else 'none'}...")
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
                host=config.ai_models.effective_ollama_embedding_host,
                model=config.ai_models.effective_ollama_embedding_model,
                api_key=config.ai_models.ollama_embedding_api_key
            )
            auth_info = " (with API key)" if config.ai_models.ollama_embedding_api_key else " (no API key)"
            logger.info(f"Initialized Ollama embedding provider with host: {config.ai_models.effective_ollama_embedding_host}, model: {config.ai_models.effective_ollama_embedding_model}{auth_info}")
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
            logger.debug("No chunks provided for embedding generation")
            return {}

        # Select provider
        provider = self._get_provider(provider_name)
        if not provider:
            raise ValueError(f"No available embedding provider")

        logger.info(f"Starting embedding generation for {len(chunks)} chunks using {provider.__class__.__name__}")
        logger.debug(f"Provider details: model={getattr(provider, 'model', 'unknown')}, dimension={provider.get_embedding_dimension()}")

        # Prepare texts for embedding
        texts = []
        chunk_ids = []

        logger.debug("Preparing chunk texts for embedding...")
        for i, chunk in enumerate(chunks):
            # Combine content with metadata for better embeddings
            text = self._prepare_chunk_text(chunk)
            texts.append(text)
            chunk_ids.append(chunk.id)

            # Log details for first few chunks and every 100th chunk
            if i < 3 or (i + 1) % 100 == 0:
                logger.debug(f"Chunk {i+1}/{len(chunks)}: {chunk.node_type.value} '{chunk.name}' from {chunk.file_path}")
                logger.debug(f"  - Content length: {len(chunk.content)} chars")
                logger.debug(f"  - Prepared text length: {len(text)} chars")
                if i < 3:  # Show full prepared text for first few chunks
                    logger.debug(f"  - Prepared text preview: {text[:200]}...")

        # Generate embeddings in batches
        batch_size = config.indexing.batch_size
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"Processing {len(texts)} texts in {total_batches} batches of size {batch_size}")

        import time
        total_start_time = time.time()

        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch_start_time = time.time()
            batch_texts = texts[i:i + batch_size]
            batch_chunk_ids = chunk_ids[i:i + batch_size]

            logger.debug(f"Processing batch {batch_idx + 1}/{total_batches} with {len(batch_texts)} texts")
            logger.debug(f"  - Batch chunk IDs: {batch_chunk_ids[:3]}{'...' if len(batch_chunk_ids) > 3 else ''}")

            try:
                batch_embeddings = await provider.generate_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)

                batch_time = time.time() - batch_start_time
                logger.debug(f"  - Batch {batch_idx + 1} completed in {batch_time:.2f}s")
                logger.debug(f"  - Generated {len(batch_embeddings)} embeddings, dimensions: {len(batch_embeddings[0]) if batch_embeddings else 0}")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                raise

        total_time = time.time() - total_start_time

        # Create mapping
        embeddings_map = dict(zip(chunk_ids, all_embeddings))

        logger.info(f"Generated embeddings for {len(chunks)} chunks using {provider.__class__.__name__} in {total_time:.2f}s")
        logger.debug(f"Embedding generation stats:")
        logger.debug(f"  - Total chunks processed: {len(chunks)}")
        logger.debug(f"  - Total batches: {total_batches}")
        logger.debug(f"  - Average batch time: {total_time/total_batches:.2f}s")
        logger.debug(f"  - Average time per chunk: {total_time/len(chunks):.3f}s")
        logger.debug(f"  - Embeddings map size: {len(embeddings_map)}")

        return embeddings_map

    async def generate_embeddings(self, texts: List[str], provider_name: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for text queries."""
        if not texts:
            logger.debug("No texts provided for query embedding generation")
            return []

        # Select provider
        provider = self._get_provider(provider_name)
        if not provider:
            raise ValueError(f"No available embedding provider")

        logger.info(f"Generating query embeddings for {len(texts)} texts using {provider.__class__.__name__}")
        logger.debug(f"Provider details: model={getattr(provider, 'model', 'unknown')}, dimension={provider.get_embedding_dimension()}")

        # Log query text details
        for i, text in enumerate(texts):
            logger.debug(f"Query text {i+1}: length={len(text)} chars")
            logger.debug(f"Query text {i+1} preview: {text[:200]}...")

        import time
        start_time = time.time()

        # Generate embeddings
        embeddings = await provider.generate_embeddings(texts)

        total_time = time.time() - start_time

        logger.info(f"Generated embeddings for {len(texts)} texts using {provider.__class__.__name__} in {total_time:.2f}s")
        logger.debug(f"Query embedding stats:")
        logger.debug(f"  - Generated {len(embeddings)} embeddings")
        logger.debug(f"  - Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        logger.debug(f"  - Average time per text: {total_time/len(texts):.3f}s")

        return embeddings

    def _prepare_chunk_text(self, chunk: CodeChunk) -> str:
        """Prepare chunk text for embedding generation."""
        logger.debug(f"Preparing text for chunk {chunk.id} ({chunk.node_type.value})")

        # Combine content with metadata
        parts = []

        # Add type and name information
        if chunk.name:
            type_name_part = f"{chunk.node_type.value}: {chunk.name}"
            parts.append(type_name_part)
            logger.debug(f"  - Added type/name: {type_name_part}")
        else:
            logger.debug(f"  - No name for chunk, type: {chunk.node_type.value}")

        # Add file context
        file_name = chunk.file_path.split('/')[-1]
        file_part = f"File: {file_name}"
        parts.append(file_part)
        logger.debug(f"  - Added file context: {file_part}")

        # Add the actual code content
        parts.append(chunk.content)
        logger.debug(f"  - Added content: {len(chunk.content)} characters")

        prepared_text = "\n".join(parts)
        logger.debug(f"  - Final prepared text length: {len(prepared_text)} characters")

        return prepared_text
    
    def _get_provider(self, provider_name: Optional[str] = None) -> Optional[EmbeddingProvider]:
        """Get embedding provider by name or default."""
        logger.debug(f"Selecting embedding provider: requested={provider_name}")
        logger.debug(f"Available providers: {list(self.providers.keys())}")

        if provider_name and provider_name in self.providers:
            selected_provider = self.providers[provider_name]
            logger.debug(f"Using requested provider: {provider_name} ({selected_provider.__class__.__name__})")
            return selected_provider

        # Use default provider
        default_model = config.ai_models.default_embedding_model
        logger.debug(f"Using default model selection: {default_model}")

        if default_model == "cloud":
            cloud_model = config.ai_models.default_cloud_model
            logger.debug(f"Looking for cloud model: {cloud_model}")
            if cloud_model in self.providers:
                selected_provider = self.providers[cloud_model]
                logger.debug(f"Using cloud provider: {cloud_model} ({selected_provider.__class__.__name__})")
                return selected_provider
        elif default_model == "local":
            logger.debug("Looking for local providers...")
            if "ollama" in self.providers:
                selected_provider = self.providers["ollama"]
                logger.debug(f"Using Ollama provider ({selected_provider.__class__.__name__})")
                return selected_provider
            elif "sentence_transformer" in self.providers:
                selected_provider = self.providers["sentence_transformer"]
                logger.debug(f"Using SentenceTransformer provider ({selected_provider.__class__.__name__})")
                return selected_provider

        # Fallback to any available provider
        logger.debug("Using fallback provider selection...")
        for name, provider in self.providers.items():
            logger.debug(f"Using fallback provider: {name} ({provider.__class__.__name__})")
            return provider

        logger.warning("No embedding providers available!")
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
