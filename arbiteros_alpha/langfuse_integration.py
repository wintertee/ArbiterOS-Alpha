"""Langfuse integration for ArbiterOS policy result tracking.

This module provides a wrapper around Langfuse's CallbackHandler that automatically
syncs policy results from ArbiterOS history to Langfuse spans or events for observability.
"""

import logging
from typing import Any, Optional

from langfuse.langchain import CallbackHandler
from langfuse import Langfuse

from .core import ArbiterOSAlpha
from .history import HistoryItem


logger = logging.getLogger(__name__)


class ArbiterLangfuseHandler:
    """Wrapper around Langfuse CallbackHandler that syncs policy results to spans or events.

    This wrapper intercepts LangChain callback events and automatically adds
    policy check and route results from the most recent HistoryItem to Langfuse
    spans or events using the Langfuse client directly.

    Attributes:
        arbiter_os: The ArbiterOSAlpha instance to extract policy results from.
        handler: The underlying Langfuse CallbackHandler instance for LangChain compatibility.
        langfuse: The Langfuse client instance for direct span/event creation.
        use_event: Whether to use events instead of spans for policy results.
    """

    def __init__(
        self,
        arbiter_os: ArbiterOSAlpha,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        handler: Optional[CallbackHandler] = None,
        use_event: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the ArbiterLangfuseHandler.

        Args:
            arbiter_os: The ArbiterOSAlpha instance that tracks execution history.
            public_key: Optional Langfuse public key. If None, uses environment variable.
            secret_key: Optional Langfuse secret key. If None, uses environment variable.
            host: Optional Langfuse host URL. If None, uses default.
            handler: Optional pre-configured CallbackHandler instance. If None, creates a new one
                with the provided configuration.
            use_event: If True, use events instead of spans for policy results. Defaults to True.
            **kwargs: Additional arguments to pass to CallbackHandler and Langfuse client.
        """
        self.arbiter_os = arbiter_os
        self.use_event = use_event

        # Initialize Langfuse client directly
        langfuse_kwargs: dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host is not None:
            langfuse_kwargs["host"] = host

        # Add common Langfuse client parameters from kwargs
        langfuse_params = [
            "base_url",
            "timeout",
            "debug",
            "tracing_enabled",
            "flush_at",
            "flush_interval",
            "environment",
            "release",
        ]
        for param in langfuse_params:
            if param in kwargs:
                langfuse_kwargs[param] = kwargs[param]

        self.langfuse = Langfuse(**langfuse_kwargs)

        # Keep CallbackHandler for LangChain callback compatibility
        # Pass all kwargs to CallbackHandler (it will use what it needs)
        if handler is not None:
            self.handler = handler
        else:
            self.handler = CallbackHandler(public_key=public_key, **kwargs)

    def _get_most_recent_history_item(self) -> Optional[HistoryItem]:
        """Get the most recent HistoryItem from the execution history.

        Returns:
            The most recent HistoryItem if available, None otherwise.
        """
        if not self.arbiter_os.history.entries:
            return None

        # Get the last superstep
        last_superstep = self.arbiter_os.history.entries[-1]
        if not last_superstep:
            return None

        # Get the last entry in the superstep
        return last_superstep[-1]

    def _sync_policy_results(self) -> None:
        """Extract policy results from the most recent HistoryItem and add to span or event.

        This method accesses the most recent HistoryItem, extracts check_policy_results
        and route_policy_results, and adds them as metadata to a Langfuse span or event
        using the Langfuse client directly.
        """
        history_item = self._get_most_recent_history_item()
        if history_item is None:
            logger.debug("No history item available to sync policy results")
            return

        # Extract policy results
        check_results = history_item.check_policy_results
        route_results = history_item.route_policy_results

        # Only add metadata if there are policy results
        if not check_results and not route_results:
            logger.debug("No policy results to sync")
            return

        # Build metadata dictionary
        metadata: dict[str, Any] = {}
        if check_results:
            metadata["arbiteros_check_policy_results"] = check_results
        if route_results:
            metadata["arbiteros_route_policy_results"] = route_results

        # Create span or event using the Langfuse client directly
        try:
            if self.use_event:
                self.langfuse.create_event(
                    name="arbiter_policy_results",
                    metadata=metadata,
                )
            else:
                with self.langfuse.start_as_current_span(
                    name="arbiter_policy_results",
                    metadata=metadata,
                ):
                    pass
        except Exception as e:
            logger.warning(f"Failed to sync policy results to Langfuse: {e}")

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying handler.

        This allows the wrapper to transparently forward all CallbackHandler
        methods and attributes to the underlying handler while intercepting
        specific callback methods to inject policy metadata.

        Args:
            name: The attribute name to access.

        Returns:
            The attribute value from the underlying handler.
        """
        # Intercept callback methods that are called after node/chain execution
        # These are the right hooks to sync policy results since they run after
        # the instruction decorator has populated the HistoryItem with policy results
        callback_hooks = [
            "on_chain_end",
            "on_llm_end",
            "on_tool_end",
            "on_chain_start",
            "on_llm_start",
            "on_tool_start",
        ]

        if name in callback_hooks and hasattr(self.handler, name):
            original_method = getattr(self.handler, name)

            def wrapped_method(*args: Any, **kwargs: Any) -> Any:
                # Call the original method first
                result = original_method(*args, **kwargs)
                # Then sync policy results (after execution completes)
                # This ensures policy results are available in the history
                self._sync_policy_results()
                return result

            return wrapped_method

        # For all other attributes, delegate to the handler
        return getattr(self.handler, name)
