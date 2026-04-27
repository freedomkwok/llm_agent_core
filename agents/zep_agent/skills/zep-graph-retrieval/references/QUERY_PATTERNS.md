## Query Patterns

Use these patterns to improve retrieval quality.

- Entity discovery:
  - "who/what is <entity>"
  - "<concept> on graph <graph_id>"
- Relationship evidence:
  - "<entity A> and <entity B> relation"
  - "<entity> capability / dependency / ownership"
- Neighborhood expansion:
  - Start with known UUID using `get_node_by_id`
  - Then call `get_edges_for_node` or `search_around_node`

## Practical Notes

- Keep `limit` small first (1-5), then expand only when needed.
- If no results, rephrase query using canonical entity names from previous hits.
- Prefer returning concise results rather than full raw edge/node structures.
