from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config


class ApiKeyManager:
    """
    Manages API keys from:

    1. ECS secret: apikey
    2. Individual environment variables

    Required keys are only required when their corresponding
    provider is actually being used.
    """

    def __init__(self):
        self.api_keys: dict[str, str] = {}

        # ---------------------------------------------------------
        # Load keys from ECS secret
        # ---------------------------------------------------------
        raw = os.getenv("apikey")

        if raw:
            try:
                parsed = json.loads(raw)

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "apikeyliveclass is not a valid JSON object"
                    )

                self.api_keys = parsed

                log.info(
                    "Loaded API keys from ECS secret"
                )

            except Exception as e:
                log.warning(
                    "Failed to parse API_KEYS as JSON",
                    error=str(e),
                )

        # ---------------------------------------------------------
        # Load missing keys from individual environment variables
        # ---------------------------------------------------------
        supported_keys = [
            "GROQ_API_KEY",
            "GOOGLE_API_KEY",
        ]

        for key in supported_keys:
            if not self.api_keys.get(key):

                env_value = os.getenv(key)

                if env_value:
                    self.api_keys[key] = env_value

                    log.info(
                        f"Loaded {key} from individual env var"
                    )

        # ---------------------------------------------------------
        # Log which keys are available
        # ---------------------------------------------------------
        available_keys = {
            key: f"{value[:6]}..."
            for key, value in self.api_keys.items()
            if value
        }

        log.info(
            "API key manager initialized",
            available_keys=available_keys,
        )

    def get(self, key: str) -> str:
        """
        Return an API key.

        Raises:
            KeyError: If the requested key is missing.
        """

        value = self.api_keys.get(key)

        if not value:
            raise KeyError(
                f"API key for {key} is missing"
            )

        return value


class ModelLoader:
    """
    Loads embedding models and LLMs from config.yaml.

    Embeddings:
        Sentence Transformers / HuggingFace

    LLM:
        Google Gemini
        OR
        Groq

    Example YAML:

        embedding_model:
          provider: "sentence_transformers"
          model_name: "sentence-transformers/all-MiniLM-L6-v2"

        retriever:
          top_k: 10
          search_type: "mmr"
          fetch_k: 20
          lambda_mult: 0.5

        llm:
          groq:
            provider: "groq"
            model_name: "openai/gpt-oss-20b"
            temperature: 0
            max_output_tokens: 2048

          google:
            provider: "google"
            model_name: "gemini-3.6-flash"
            temperature: 0
            max_output_tokens: 2048
    """

    def __init__(self):

        # ---------------------------------------------------------
        # Load .env in local environment
        # ---------------------------------------------------------
        if os.getenv(
            "ENV",
            "local",
        ).lower() != "production":

            load_dotenv()

            log.info(
                "Running in LOCAL mode: .env loaded"
            )

        else:

            log.info(
                "Running in PRODUCTION mode"
            )

        # ---------------------------------------------------------
        # API keys
        # ---------------------------------------------------------
        self.api_key_mgr = ApiKeyManager()

        # ---------------------------------------------------------
        # YAML configuration
        # ---------------------------------------------------------
        self.config = load_config()

        log.info(
            "YAML config loaded",
            config_keys=list(
                self.config.keys()
            ),
        )

    # =============================================================
    # EMBEDDINGS
    # =============================================================

    def load_embeddings(self):
        """
        Load embedding model from config.yaml.

        Example:

            embedding_model:
              provider: "sentence_transformers"
              model_name: "sentence-transformers/all-MiniLM-L6-v2"

        This does NOT call Google's embedding API.
        """

        try:

            embedding_config = self.config.get(
                "embedding_model"
            )

            if not embedding_config:
                raise ValueError(
                    "Missing 'embedding_model' configuration "
                    "in config.yaml"
                )

            # -----------------------------------------------------
            # Read provider and model from YAML
            # -----------------------------------------------------
            provider = embedding_config.get(
                "provider"
            )

            model_name = embedding_config.get(
                "model_name"
            )

            if not provider:
                raise ValueError(
                    "Embedding provider is missing "
                    "in config.yaml"
                )

            if not model_name:
                raise ValueError(
                    "Embedding model_name is missing "
                    "in config.yaml"
                )

            provider = str(
                provider
            ).lower().strip()

            log.info(
                "Loading embedding model",
                provider=provider,
                model=model_name,
            )

            # =====================================================
            # SENTENCE TRANSFORMERS
            # =====================================================

            if provider in {
                "sentence_transformers",
                "sentence-transformers",
                "huggingface",
            }:

                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        "device": "cpu",
                    },
                    encode_kwargs={
                        "normalize_embeddings": True,
                    },
                )

                log.info(
                    "Sentence Transformer embedding loaded",
                    provider=provider,
                    model=model_name,
                )

                return embeddings

            # =====================================================
            # GOOGLE EMBEDDINGS
            # =====================================================
            #
            # Kept as an optional provider in case you want to
            # switch back through YAML later.
            #
            # Current YAML should use sentence_transformers.
            # =====================================================

            elif provider == "google":

                from langchain_google_genai import (
                    GoogleGenerativeAIEmbeddings,
                )

                embeddings = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get(
                        "GOOGLE_API_KEY"
                    ),
                )

                log.info(
                    "Google embedding model loaded",
                    provider=provider,
                    model=model_name,
                )

                return embeddings

            # =====================================================
            # UNSUPPORTED EMBEDDING PROVIDER
            # =====================================================

            else:

                raise ValueError(
                    f"Unsupported embedding provider: {provider}. "
                    f"Supported providers: "
                    f"sentence_transformers, google"
                )

        except Exception as e:

            log.error(
                "Error loading embedding model",
                error=str(e),
            )

            raise DocumentPortalException(
                "Failed to load embedding model",
                e,
            ) from e

    # =============================================================
    # LLM
    # =============================================================

    def load_llm(self):
        """
        Load LLM according to config.yaml and LLM_PROVIDER.

        Environment:

            LLM_PROVIDER=google

        or:

            LLM_PROVIDER=groq
        """

        try:

            # -----------------------------------------------------
            # Read LLM block
            # -----------------------------------------------------
            llm_block = self.config.get(
                "llm"
            )

            if not llm_block:
                raise ValueError(
                    "Missing 'llm' configuration "
                    "in config.yaml"
                )

            # -----------------------------------------------------
            # Select provider
            # -----------------------------------------------------
            provider_key = os.getenv(
                "LLM_PROVIDER",
                "google",
            ).lower().strip()

            if provider_key not in llm_block:

                raise ValueError(
                    f"LLM provider '{provider_key}' "
                    f"not found in config.yaml"
                )

            # -----------------------------------------------------
            # Read provider configuration
            # -----------------------------------------------------
            llm_config = llm_block[
                provider_key
            ]

            provider = llm_config.get(
                "provider"
            )

            model_name = llm_config.get(
                "model_name"
            )

            temperature = llm_config.get(
                "temperature",
                0,
            )

            max_output_tokens = llm_config.get(
                "max_output_tokens",
                2048,
            )

            if not provider:
                raise ValueError(
                    f"LLM provider is missing for "
                    f"'{provider_key}'"
                )

            if not model_name:
                raise ValueError(
                    f"LLM model_name is missing for "
                    f"'{provider_key}'"
                )

            provider = str(
                provider
            ).lower().strip()

            log.info(
                "Loading LLM",
                provider=provider,
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            # =====================================================
            # GOOGLE GEMINI
            # =====================================================

            if provider == "google":

                google_api_key = (
                    self.api_key_mgr.get(
                        "GOOGLE_API_KEY"
                    )
                )

                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=google_api_key,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )

                log.info(
                    "Google LLM loaded successfully",
                    model=model_name,
                )

                return llm

            # =====================================================
            # GROQ
            # =====================================================

            elif provider == "groq":

                groq_api_key = (
                    self.api_key_mgr.get(
                        "GROQ_API_KEY"
                    )
                )

                llm = ChatGroq(
                    model=model_name,
                    api_key=groq_api_key,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )

                log.info(
                    "Groq LLM loaded successfully",
                    model=model_name,
                )

                return llm

            # =====================================================
            # UNSUPPORTED LLM
            # =====================================================

            else:

                raise ValueError(
                    f"Unsupported LLM provider: {provider}"
                )

        except Exception as e:

            log.error(
                "Error loading LLM",
                error=str(e),
            )

            raise DocumentPortalException(
                "Failed to load LLM",
                e,
            ) from e


# =================================================================
# TEST
# =================================================================

if __name__ == "__main__":

    loader = ModelLoader()

    # =============================================================
    # TEST EMBEDDING
    # =============================================================

    print(
        "\n--- Testing Embedding Model ---"
    )

    embeddings = loader.load_embeddings()

    print(
        f"Embedding Model Loaded: {embeddings}"
    )

    query = "Hello, how are you?"

    result = embeddings.embed_query(
        query
    )

    print(
        f"Embedding Dimension: {len(result)}"
    )

    print(
        f"First 10 values: {result[:10]}"
    )

    # =============================================================
    # TEST LLM
    # =============================================================

    print(
        "\n--- Testing LLM ---"
    )

    llm = loader.load_llm()

    print(
        f"LLM Loaded: {llm}"
    )

    result = llm.invoke(
        "Hello, how are you?"
    )

    print(
        f"LLM Result: {result.content}"
    )