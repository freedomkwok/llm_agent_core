<!-- SPDX-License-Identifier: Apache-2.0 -->

---
name: zep-graph-retrieval
description: Retrieve graph entities and relationship evidence from Zep for routing and grounding.
metadata:
  adk_additional_tools:
    - search_nodes
    - search_edges
    - get_node_by_id
    - get_edges_for_node
    - search_around_node
---

## Purpose

Use this skill to query Zep graph memory for entities, relationship facts, and node-local context.

## Workflow

1. Start with `search_nodes` when user intent names a topic/entity but no UUID is known.
2. Use `search_edges` when relation-level evidence is needed (facts connecting entities).
3. If a concrete node UUID is available, call `get_node_by_id`.
4. Expand context with `get_edges_for_node` or `search_around_node` when deeper neighborhood evidence is needed.
5. Prefer concise summaries in the final answer; do not dump raw graph payloads unless the user asks.

## Tool Selection Guidance

- Use `search_nodes` for discovery.
- Use `search_edges` for evidence and relationship lookup.
- Use `get_node_by_id` for exact retrieval by UUID.
- Use `get_edges_for_node` for direct adjacency.
- Use `search_around_node` to gather a combined neighborhood bundle in one call.

## Output Guidance

- Prioritize name/metadata/attributes/summary/score from node outputs.
- Cite relationship facts from edge outputs when explaining reasoning.
- If no relevant results are found, say so and suggest a narrower or alternate query phrase.
