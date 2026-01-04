# -*- coding: utf-8 -*-
"""
LangSmith Tracing Integration for EthicsEngine

Provides optional LangSmith tracing for all LLM calls in the system.
Enable via environment variables or settings.

Environment Variables:
    LANGSMITH_ENABLED: Set to "true" to enable tracing (default: false)
    LANGSMITH_API_KEY: Your LangSmith API key
    LANGSMITH_PROJECT: Project name in LangSmith (default: "ethicsengine")
    LANGSMITH_ENDPOINT: LangSmith API endpoint (default: https://api.smith.langchain.com)

Usage:
    # At application startup
    from utils.langsmith_tracing import init_langsmith, is_langsmith_enabled
    
    if is_langsmith_enabled():
        init_langsmith()
    
    # For tracing specific operations
    from utils.langsmith_tracing import trace_llm_call, trace_benchmark_run
    
    with trace_llm_call("ethics_evaluation", metadata={"scenario_id": "HE-001"}):
        response = await engine.generate_response(...)
"""

import os
import logging
import functools
from typing import Optional, Dict, Any, Callable, TypeVar
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])

# Global state
_langsmith_initialized = False
_langsmith_client = None
_tracer = None


def get_langsmith_config() -> Dict[str, Any]:
    """Get LangSmith configuration from environment."""
    return {
        "enabled": os.getenv("LANGSMITH_ENABLED", "false").lower() in ("true", "1", "yes"),
        "api_key": os.getenv("LANGSMITH_API_KEY", ""),
        "project": os.getenv("LANGSMITH_PROJECT", "ethicsengine"),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        "tracing_v2": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes"),
    }


def is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    config = get_langsmith_config()
    return config["enabled"] and bool(config["api_key"])


def init_langsmith() -> bool:
    """
    Initialize LangSmith tracing.
    
    Sets up the necessary environment variables and initializes the LangSmith client.
    Returns True if initialization was successful, False otherwise.
    """
    global _langsmith_initialized, _langsmith_client, _tracer
    
    config = get_langsmith_config()
    
    if not config["enabled"]:
        logger.debug("LangSmith tracing is disabled")
        return False
    
    if not config["api_key"]:
        logger.warning("LangSmith enabled but no API key provided. Tracing disabled.")
        return False
    
    try:
        # Set environment variables for LangChain/LangSmith auto-tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config["api_key"]
        os.environ["LANGCHAIN_PROJECT"] = config["project"]
        os.environ["LANGCHAIN_ENDPOINT"] = config["endpoint"]
        
        # Try to import and initialize LangSmith
        try:
            from langsmith import Client
            _langsmith_client = Client(
                api_key=config["api_key"],
                api_url=config["endpoint"],
            )
            logger.info(f"LangSmith client initialized for project: {config['project']}")
        except ImportError:
            logger.info("LangSmith SDK not installed. Using environment variable tracing only.")
            logger.info("Install with: pip install langsmith")
        
        # Try to set up LangChain callback handler if available
        try:
            from langchain_core.tracers import LangChainTracer
            _tracer = LangChainTracer(project_name=config["project"])
            logger.info("LangChain tracer initialized")
        except ImportError:
            logger.debug("LangChain not available for callback tracing")
        
        _langsmith_initialized = True
        logger.info(f"LangSmith tracing enabled for project: {config['project']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}")
        _langsmith_initialized = False
        return False


def get_langsmith_client():
    """Get the LangSmith client if available."""
    return _langsmith_client


def get_tracer():
    """Get the LangChain tracer if available."""
    return _tracer


@contextmanager
def trace_llm_call(
    name: str,
    run_type: str = "llm",
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
):
    """
    Context manager for tracing an LLM call.
    
    Example:
        with trace_llm_call("ethics_evaluation", metadata={"scenario_id": "HE-001"}):
            response = await engine.generate_response(prompt)
    """
    if not _langsmith_initialized or not _langsmith_client:
        yield
        return
    
    start_time = datetime.now(timezone.utc)
    run_id = None
    
    try:
        from langsmith.run_trees import RunTree
        
        run = RunTree(
            name=name,
            run_type=run_type,
            inputs=metadata or {},
            project_name=os.getenv("LANGCHAIN_PROJECT", "ethicsengine"),
            tags=tags or [],
        )
        run_id = run.id
        
        yield run
        
        run.end(outputs={"status": "success"})
        run.post()
        
    except ImportError:
        # LangSmith not installed, just yield
        yield
    except Exception as e:
        logger.debug(f"LangSmith tracing error: {e}")
        yield


@contextmanager  
def trace_benchmark_run(
    batch_id: str,
    model_name: str,
    total_scenarios: int,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for tracing a full benchmark run.
    
    Example:
        with trace_benchmark_run("batch-001", "llama3.2:3b", 50):
            results = await run_benchmark(...)
    """
    trace_metadata = {
        "batch_id": batch_id,
        "model_name": model_name,
        "total_scenarios": total_scenarios,
        **(metadata or {}),
    }
    
    with trace_llm_call(
        name=f"he300_benchmark_{batch_id}",
        run_type="chain",
        metadata=trace_metadata,
        tags=["benchmark", "he300", model_name],
    ) as run:
        yield run


def trace_function(
    name: Optional[str] = None,
    run_type: str = "chain",
    tags: Optional[list] = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing a function.
    
    Example:
        @trace_function("evaluate_scenario", tags=["he300"])
        async def evaluate_scenario(scenario):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = name or func.__name__
            with trace_llm_call(func_name, run_type=run_type, tags=tags):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = name or func.__name__
            with trace_llm_call(func_name, run_type=run_type, tags=tags):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


class OllamaLangSmithWrapper:
    """
    Wrapper for Ollama calls that adds LangSmith tracing.
    
    This can be used to wrap the Ollama client or the OpenAI-compatible
    endpoint to automatically trace all LLM calls.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self._client = None
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a response from Ollama with tracing."""
        import httpx
        
        metadata = {
            "model": model,
            "prompt_length": len(prompt),
            "has_system": bool(system),
        }
        
        with trace_llm_call(f"ollama_{model}", metadata=metadata, tags=["ollama", model]):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "system": system,
                        "stream": False,
                        **kwargs,
                    },
                    timeout=300.0,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
    
    async def chat(
        self,
        model: str,
        messages: list,
        **kwargs,
    ) -> str:
        """Chat with Ollama with tracing."""
        import httpx
        
        metadata = {
            "model": model,
            "message_count": len(messages),
        }
        
        with trace_llm_call(f"ollama_chat_{model}", metadata=metadata, tags=["ollama", "chat", model]):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        **kwargs,
                    },
                    timeout=300.0,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")


def create_traced_openai_client():
    """
    Create an OpenAI client with LangSmith tracing enabled.
    
    Uses langchain-openai if available, falls back to standard openai client.
    """
    try:
        # Try LangChain's OpenAI wrapper which has built-in tracing
        from langchain_openai import ChatOpenAI
        
        config = get_langsmith_config()
        model = os.getenv("OPENAI_MODEL", "gpt-4")
        
        return ChatOpenAI(
            model=model,
            callbacks=[_tracer] if _tracer else None,
        )
    except ImportError:
        # Fall back to standard OpenAI
        try:
            from openai import AsyncOpenAI
            return AsyncOpenAI()
        except ImportError:
            logger.warning("Neither langchain-openai nor openai package available")
            return None


def get_langsmith_status() -> Dict[str, Any]:
    """Get current LangSmith status for health checks."""
    config = get_langsmith_config()
    
    return {
        "enabled": config["enabled"],
        "initialized": _langsmith_initialized,
        "project": config["project"] if _langsmith_initialized else None,
        "has_client": _langsmith_client is not None,
        "has_tracer": _tracer is not None,
        "endpoint": config["endpoint"] if _langsmith_initialized else None,
    }


# Auto-initialize on import if enabled
_auto_init_config = get_langsmith_config()
if _auto_init_config["enabled"] and _auto_init_config["api_key"]:
    init_langsmith()
