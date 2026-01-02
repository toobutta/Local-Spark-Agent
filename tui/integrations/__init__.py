"""SparkPlug TUI Integrations - External service integrations."""

from .dgx_spark import DGXSparkAPI, GPUMetrics, get_dgx_api
from .factory_cli import FactoryCLIAdapter, get_factory_cli
from .ollama import OllamaService, OllamaModel, get_ollama_service

__all__ = [
    'DGXSparkAPI',
    'GPUMetrics',
    'get_dgx_api',
    'FactoryCLIAdapter',
    'get_factory_cli',
    'OllamaService',
    'OllamaModel',
    'get_ollama_service',
]


