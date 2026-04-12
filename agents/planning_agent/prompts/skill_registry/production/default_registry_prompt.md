# Default Registry Prompt

You generate JSON for Zep retrieval. Return strict JSON only.

Schema:

```json
{
  "request": {
    "query": "...",
    "top_k": 8,
    "scope": null,
    "filters": {},
    "metadata": {}
  },
  "reason": "optional short reason"
}
```
