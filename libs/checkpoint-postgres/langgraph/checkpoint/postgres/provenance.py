"""Message checkpoint provenance for the Postgres checkpointer.

Two pieces work together:

1. `add_messages_with_provenance` — a reducer that marks the messages added
   or updated in the current super-step with a temporary in-memory tag.
2. `ProvenancePostgresSaver` / `ProvenanceAsyncPostgresSaver` — checkpointer
   subclasses that replace the tag with real provenance (the checkpoint ID
   being written and its parent checkpoint ID) right before serialization.

The tag only ever exists in memory and is never persisted. After a run, each
processed message carries `additional_kwargs["checkpoint_id"]` (the checkpoint
it was persisted in) and `additional_kwargs["parent_checkpoint_id"]` (the
checkpoint it descends from).

The reducer requires the `langgraph` core package (for `add_messages`), which
is not a hard dependency of `langgraph-checkpoint-postgres`; it is imported
lazily so the savers remain usable without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage

PENDING_PROVENANCE = "__provenance_pending"
"""Temporary marker added by the reducer and replaced by the saver."""


def add_messages_with_provenance(
    left: list[AnyMessage],
    right: list[AnyMessage] | list[dict] | AnyMessage | dict,
) -> list[AnyMessage]:
    """A provenance-aware variant of `add_messages`.

    The reducer runs within each super-step, where `right` is exactly the set
    of messages added or updated in that step, so it can mark the changed
    messages without comparing against the parent checkpoint.

    !!! note
        Requires the `langgraph` core package at call time.
    """
    from langgraph.graph.message import add_messages  # noqa: PLC0415

    result = add_messages(left, right)
    items = right if isinstance(right, list) else [right]
    changed_ids = {item["id"] if isinstance(item, dict) else item.id for item in items}
    for m in result:
        if m.id in changed_ids:
            m.additional_kwargs[PENDING_PROVENANCE] = True
    return result


def _apply_message_provenance(
    config: RunnableConfig,
    checkpoint: Checkpoint,
) -> None:
    """Replace pending markers with real provenance in place.

    Called by the savers from `put`/`aput`, where both `checkpoint["id"]`
    (the new checkpoint) and `config["configurable"]["checkpoint_id"]` (its
    parent) are already determined, and before any serialization happens.
    """
    messages = checkpoint["channel_values"].get("messages")
    if not isinstance(messages, list):
        return
    checkpoint_id = checkpoint["id"]
    parent_checkpoint_id = config["configurable"].get("checkpoint_id")
    for m in messages:
        kw = m.additional_kwargs
        if isinstance(m, BaseMessage) and kw.pop(PENDING_PROVENANCE, None):
            kw["checkpoint_id"] = checkpoint_id
            kw["parent_checkpoint_id"] = parent_checkpoint_id


class ProvenancePostgresSaver(PostgresSaver):
    """`PostgresSaver` that records message checkpoint provenance on write."""

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        _apply_message_provenance(config, checkpoint)
        return super().put(config, checkpoint, metadata, new_versions)


class ProvenanceAsyncPostgresSaver(AsyncPostgresSaver):
    """`AsyncPostgresSaver` that records message checkpoint provenance on write."""

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        _apply_message_provenance(config, checkpoint)
        return await super().aput(config, checkpoint, metadata, new_versions)


__all__ = [
    "PENDING_PROVENANCE",
    "ProvenanceAsyncPostgresSaver",
    "ProvenancePostgresSaver",
    "add_messages_with_provenance",
]
