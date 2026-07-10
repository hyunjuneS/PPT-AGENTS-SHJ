"""Langfuse tracing wiring for the new LangGraph engine. Fully optional —
returns None (no-op) unless LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST are all set,
so local dev without a self-hosted Langfuse instance is unaffected.
"""

import os


def get_langfuse_handler(session_id: str | None = None):
    """Build a Langfuse LangChain callback handler, or None if unconfigured.

    Session correlation with the existing `.history/` workspace dump is done
    via `langfuse_session_id` in the LangGraph run's `metadata` (see
    deeppresenter/graph/design_graph.py's run_config) rather than here, since
    Langfuse SDK v3+'s CallbackHandler no longer takes a session_id at
    construction time — that changed between v2 and v3 along with the import
    path below. The self-hosted server's SDK-major-version compatibility
    wasn't known at implementation time (needs platform >= 3.125.0 for the
    v3+ path); this defensive try/except adapts to whichever `langfuse`
    version ends up installed rather than assuming one blindly. If your
    self-hosted instance predates that, pin `langfuse<3` in requirements.txt
    and the v2 branch below (which does accept session_id directly) is used
    automatically instead.
    """
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
        and os.environ.get("LANGFUSE_HOST")
    ):
        return None

    try:
        from langfuse.langchain import CallbackHandler  # SDK v3+ (also requires `langchain` installed)

        return CallbackHandler()
    except ImportError:
        from langfuse.callback import CallbackHandler  # SDK v2

        return CallbackHandler(session_id=session_id)
