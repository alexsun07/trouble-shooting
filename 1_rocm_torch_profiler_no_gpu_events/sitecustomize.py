"""Restore GPU visibility for the torch profiler in child processes.

HIP init sets ROCPROFILER_REGISTER_LIBRARY via C setenv(). Any child that
inherits it starts with rocprofiler already registered, so kineto never gets
dispatch callbacks and every GPU event is missing from the trace -- sglang's
scheduler ranks and aiter's tuner workers are both such children.

The variable is invisible to os.environ (Python snapshots the environment at
startup and never sees C-level setenv), so os.environ.pop() does nothing;
os.unsetenv() is required. A shell-level `unset` cannot work either: at shell
time the variable does not exist yet.

Usage:  PYTHONPATH=/path/to/this/dir <original command>
"""
import os

os.unsetenv("ROCPROFILER_REGISTER_LIBRARY")
