"""dhee - the world memory layer and context compiler for AI agents.

Dhee gives Codex, Claude Code, Cursor, Cline, Gemini CLI, Chotu, and MCP
clients durable memory, repo cognition, narrative scene intelligence, handoff,
proof, and prompt-safe context that survives sessions.

Quick Start:
    from dhee import Dhee

    d = Dhee()
    d.remember("User prefers dark mode")
    d.recall("what theme does the user like?")
    d.context("fixing auth bug")
    d.checkpoint("Fixed it", what_worked="git blame first")

Memory Classes:
    Engram       — batteries-included memory interface with sensible defaults
    CoreMemory   — model-free compatibility path: add/search/delete + decay
    SmartMemory  — + echo encoding, categories, knowledge graph (needs LLM)
    FullMemory   — + scenes, profiles, orchestration, cognition
    Memory       — alias for CoreMemory for zero-config imports
"""

from dhee.memory.core import CoreMemory
from dhee.memory.smart import SmartMemory
from dhee.memory.main import FullMemory
from dhee.simple import Dhee, Engram
from dhee.plugin import DheePlugin
from dhee.agent_runtime import Client, Run, Patch
from dhee.providers import (
    ElevenAgent,
    ElevenLabsAgent,
    GeminiAgent,
    GeminiAPIAgent,
    OpenAIAgent,
    OpenAIResponsesAgent,
)
from dhee.fs import ContextWorkspace
from dhee.context_kernel import DheeContextKernel, KernelScope
from dhee.core.category import CategoryProcessor, Category, CategoryType, CategoryMatch
from dhee.core.echo import EchoProcessor, EchoDepth, EchoResult
from dhee.configs.base import MemoryConfig, FadeMemConfig, EchoMemConfig, CategoryMemConfig, ScopeConfig
from dhee.memory.admission import (
    MemoryAdmissionDecision,
    evaluate_memory_candidate,
    forget_reason_for_memory,
    sanitize_admitted_content,
)

# Default import remains model-free for backwards compatibility.
Memory = CoreMemory

__version__ = "7.4.2"
__all__ = [
    # Memory classes
    "Engram",
    "CoreMemory",
    "SmartMemory",
    "FullMemory",
    "Memory",
    # Simplified interface (the 4-operation API)
    "Dhee",
    # Universal plugin
    "DheePlugin",
    # Universal agent runtime
    "Client",
    "Run",
    "Patch",
    "ElevenAgent",
    "ElevenLabsAgent",
    "GeminiAgent",
    "GeminiAPIAgent",
    "OpenAIAgent",
    "OpenAIResponsesAgent",
    # DheeFS
    "ContextWorkspace",
    "DheeContextKernel",
    "KernelScope",
    # CategoryMem
    "CategoryProcessor",
    "Category",
    "CategoryType",
    "CategoryMatch",
    # EchoMem
    "EchoProcessor",
    "EchoDepth",
    "EchoResult",
    # Config
    "MemoryConfig",
    "FadeMemConfig",
    "EchoMemConfig",
    "CategoryMemConfig",
    "ScopeConfig",
    # Memory admission
    "MemoryAdmissionDecision",
    "evaluate_memory_candidate",
    "forget_reason_for_memory",
    "sanitize_admitted_content",
]
