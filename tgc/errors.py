from __future__ import annotations


class CompilerError(Exception):
    """Base for everything this compiler raises."""


class ConfigError(CompilerError):
    """Raised when a caller asks for something that does not make sense."""


class GraphError(CompilerError):
    """Raised when a graph is malformed."""


class TypeInferenceError(CompilerError):
    """Raised when a node's output type cannot be derived from its inputs."""


class PassError(CompilerError):
    """Raised when a transformation cannot be applied."""


class ScheduleError(CompilerError):
    """Raised when a graph cannot be scheduled as asked."""


class AllocationError(CompilerError):
    """Raised when buffers cannot be placed.

    Deliberately not called MemoryError. Shadowing the builtin inside a compiler that also
    reports genuine allocation failures makes every except clause in the codebase ambiguous.
    """


class CodegenError(CompilerError):
    """Raised when a scheduled graph cannot be turned into code."""


class VerificationError(CompilerError):
    """Raised when compiled output disagrees with the reference."""
