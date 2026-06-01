---
type: community
cohesion: 1.00
members: 2
---

# Summary Request Summarizer

**Cohesion:** 1.00 - tightly connected
**Members:** 2 nodes

## Members
- [[Summarizer Interface]] - code - context/summarize.go
- [[Summary Request]] - code - context/summarize.go

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Summary_Request_Summarizer
SORT file.name ASC
```
