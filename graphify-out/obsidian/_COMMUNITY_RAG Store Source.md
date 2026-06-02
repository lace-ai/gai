---
type: community
cohesion: 0.67
members: 3
---

# RAG Store Source

**Cohesion:** 0.67 - moderately connected
**Members:** 3 nodes

## Members
- [[Fake RAG Store]] - code - context/rag_source_test.go
- [[RAG Source Document Budgeting Tests]] - code - context/rag_source_test.go
- [[RAG Store Interface]] - code - context/store.go

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/RAG_Store_Source
SORT file.name ASC
```
