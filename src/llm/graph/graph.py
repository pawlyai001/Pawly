"""
Assemble and compile the Pawly LangGraph StateGraph.

build_graph() returns a compiled graph. Call graph.ainvoke(initial_state)
to run a full response turn.

Graph flow (with parallel branches for latency reduction):

    START ──┬──→ rule_triage ──────────────────┬──→ resolve_triage
            └──→ load_context ──→ generate ────┘         │
                                                   [override?]
                                                  ╱          ╲
                                   critical_override      finalize
                                          └──────────→ finalize ──→ END

rule_triage only needs user_message (instant keyword scan), so it runs
in parallel with the slower load_context + generate_response chain.
resolve_triage waits for both to finish before combining results.
"""

from langgraph.graph import END, START, StateGraph

from src.llm.graph.nodes import (
    critical_override_node,
    finalize_node,
    generate_response_node,
    load_context_node,
    resolve_triage_node,
    rule_triage_node,
    should_override,
)
from src.llm.graph.state import PawlyState


def build_graph():  # type: ignore[return]
    """
    Build and compile the response generation graph.

    Parallel branches:
        Branch A: rule_triage (instant, keyword-based)
        Branch B: load_context → generate_response (LLM call)

    Both converge at resolve_triage, which takes the stricter classification.
    """
    graph: StateGraph = StateGraph(PawlyState)

    graph.add_node("rule_triage", rule_triage_node)
    graph.add_node("load_context", load_context_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("resolve_triage", resolve_triage_node)
    graph.add_node("critical_override", critical_override_node)
    graph.add_node("finalize", finalize_node)

    # ── Parallel branches from START ─────────────────────────────────────────
    graph.add_edge(START, "rule_triage")
    graph.add_edge(START, "load_context")

    # ── Branch B: load_context feeds into generate_response ──────────────────
    graph.add_edge("load_context", "generate_response")

    # ── Both branches converge at resolve_triage ─────────────────────────────
    graph.add_edge("rule_triage", "resolve_triage")
    graph.add_edge("generate_response", "resolve_triage")

    # ── Conditional: override or finalize ────────────────────────────────────
    graph.add_conditional_edges(
        "resolve_triage",
        should_override,
        {"critical_override": "critical_override", "finalize": "finalize"},
    )
    graph.add_edge("critical_override", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
