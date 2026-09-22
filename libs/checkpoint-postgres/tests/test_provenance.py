# type: ignore

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    empty_checkpoint,
)
from psycopg import Connection
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres import ProvenancePostgresSaver
from langgraph.checkpoint.postgres.provenance import (
    PENDING_PROVENANCE,
    PROVENANCE_KEY,
    ProvenanceAsyncPostgresSaver,
)
from tests.conftest import DEFAULT_URI


@pytest.fixture
def saver() -> ProvenancePostgresSaver:
    with Connection.connect(
        DEFAULT_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        checkpointer = ProvenancePostgresSaver(conn)
        checkpointer.setup()
        yield checkpointer


def _config(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _tagged_checkpoint(message: str, version: str = "1") -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["messages"] = [
        HumanMessage(message, additional_kwargs={PENDING_PROVENANCE: True})
    ]
    # blobs are joined back on channel_versions at read time
    checkpoint["channel_versions"]["messages"] = version
    return checkpoint


def _metadata(step: int) -> CheckpointMetadata:
    return {"source": "input", "step": step, "writes": {}}


def test_put_attaches_provenance_and_strips_marker(saver) -> None:
    config = _config("t1")
    checkpoint = _tagged_checkpoint("hi")
    saved_config = saver.put(config, checkpoint, _metadata(1), {"messages": "1"})

    messages = saver.get_tuple(saved_config).checkpoint["channel_values"]["messages"]
    assert len(messages) == 1
    m = messages[0]
    assert PENDING_PROVENANCE not in m.additional_kwargs
    prov = m.additional_kwargs[PROVENANCE_KEY]
    assert prov["checkpoint_id"] == saved_config["configurable"]["checkpoint_id"]
    assert prov["parent_checkpoint_id"] is None


def test_put_records_parent_checkpoint_id(saver) -> None:
    first_config = saver.put(
        _config("t1"), _tagged_checkpoint("hi"), _metadata(1), {"messages": "1"}
    )
    first_id = first_config["configurable"]["checkpoint_id"]

    second_config = saver.put(
        _config("t1", first_id),
        _tagged_checkpoint("again", version="2"),
        _metadata(2),
        {"messages": "2"},
    )
    assert second_config["configurable"]["checkpoint_id"] != first_id

    messages = saver.get_tuple(second_config).checkpoint["channel_values"]["messages"]
    prov = messages[0].additional_kwargs[PROVENANCE_KEY]
    assert prov["checkpoint_id"] == second_config["configurable"]["checkpoint_id"]
    assert prov["parent_checkpoint_id"] == first_id

    # the first checkpoint's provenance is unchanged
    first_messages = saver.get_tuple(first_config).checkpoint["channel_values"][
        "messages"
    ]
    first_prov = first_messages[0].additional_kwargs[PROVENANCE_KEY]
    assert first_prov["parent_checkpoint_id"] is None


def test_put_without_messages_channel(saver) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["__root__"] = {"foo": "bar"}
    checkpoint["channel_versions"]["__root__"] = "1"
    saved_config = saver.put(_config("t1"), checkpoint, _metadata(1), {"__root__": 1})
    assert saver.get_tuple(saved_config).checkpoint["channel_values"]["__root__"] == {
        "foo": "bar"
    }


async def test_aput_records_provenance(conn) -> None:
    saver = ProvenanceAsyncPostgresSaver(conn)
    await saver.setup()
    saved_config = await saver.aput(
        _config("t1"), _tagged_checkpoint("hi"), _metadata(1), {"messages": "1"}
    )
    messages = (await saver.aget_tuple(saved_config)).checkpoint["channel_values"][
        "messages"
    ]
    prov = messages[0].additional_kwargs[PROVENANCE_KEY]
    assert prov["checkpoint_id"] == saved_config["configurable"]["checkpoint_id"]
    assert prov["parent_checkpoint_id"] is None
    assert PENDING_PROVENANCE not in messages[0].additional_kwargs


def test_reducer_and_graph_end_to_end() -> None:
    pytest.importorskip("langgraph.graph", reason="langgraph core not installed")

    # Deferred on purpose: langgraph core is not a test dependency of this
    # package, so these must stay behind the importorskip above.
    from typing import Annotated  # noqa: PLC0415

    from langchain_core.messages import AIMessage  # noqa: PLC0415
    from langgraph.graph import START, StateGraph  # noqa: PLC0415
    from typing_extensions import TypedDict  # noqa: PLC0415

    from langgraph.checkpoint.postgres.provenance import (  # noqa: PLC0415
        add_messages_with_provenance,
    )

    class State(TypedDict):
        messages: Annotated[list, add_messages_with_provenance]

    def respond(state: State) -> dict:
        n = len(state["messages"])
        return {"messages": [AIMessage(content=f"reply-{n}", id=f"ai-{n}")]}

    with Connection.connect(
        DEFAULT_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        saver = ProvenancePostgresSaver(conn)
        saver.setup()
        graph = (
            StateGraph(State)
            .add_node(respond)
            .add_edge(START, "respond")
            .compile(checkpointer=saver)
        )
        config = _config("t1")
        result = graph.invoke({"messages": [HumanMessage("hi")]}, config)

        assert len(result["messages"]) == 2
        for m in result["messages"]:
            prov = m.additional_kwargs[PROVENANCE_KEY]
            assert prov["checkpoint_id"]
            assert PENDING_PROVENANCE not in m.additional_kwargs

        human_prov = result["messages"][0].additional_kwargs[PROVENANCE_KEY]
        ai_prov = result["messages"][1].additional_kwargs[PROVENANCE_KEY]
        assert ai_prov["checkpoint_id"] != human_prov["checkpoint_id"]
        # the human message was written in the input checkpoint, whose id is
        # the parent of the checkpoint the AI reply was written in
        assert ai_prov["parent_checkpoint_id"] != ai_prov["checkpoint_id"]
