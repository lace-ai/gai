---
type: community
cohesion: 0.08
members: 38
---

# Content Builder Mistral

**Cohesion:** 0.08 - loosely connected
**Members:** 38 nodes

## Members
- [[Build Trace]] - code - context/prompt_builder.go
- [[Builder Budget Fitting]] - code - context/prompt_builder.go
- [[Builder Entry Admission]] - code - context/prompt_builder.go
- [[Builder Optional Summarization]] - code - context/prompt_builder.go
- [[Content Factory From Type]] - code - context/content.go
- [[Content Interface]] - code - context/content.go
- [[Content Tests]] - code - context/content_test.go
- [[Gemini Function Call Mapper]] - code - ai/gemini/model.go
- [[Gemini Generate]] - code - ai/gemini/model.go
- [[Gemini Generate Stream]] - code - ai/gemini/model.go
- [[Gemini Model Tests]] - code - ai/gemini/model_test.go
- [[Gemini Text Token Builder]] - code - ai/gemini/model.go
- [[History Source]] - code - context/history_source.go
- [[History Token Cache]] - code - context/history_source.go
- [[Message_8]] - code - context/message.go
- [[Mistral Generate]] - code - ai/mistral/model.go
- [[Mistral Generate Stream]] - code - ai/mistral/model.go
- [[Mistral Model Tests]] - code - ai/mistral/model_test.go
- [[Mistral No Choices Error]] - code - ai/mistral/errors.go
- [[Mistral Stream Text Extractor]] - code - ai/mistral/model.go
- [[Mistral Tool Call Accumulator]] - code - ai/mistral/model.go
- [[Prompt Budget]] - code - context/prompt_builder.go
- [[Prompt Builder]] - code - context/prompt_builder.go
- [[Prompt Builder Tests]] - code - context/prompt_builder_test.go
- [[Prompt File Loading Helpers]] - code - context/helper.go
- [[Prompt Session]] - code - context/prompt_builder.go
- [[Prompt View]] - code - context/prompt_builder.go
- [[RAG Document]] - code - context/rag_source.go
- [[RAG Overflow Summarization]] - code - context/rag_source.go
- [[RAG Source]] - code - context/rag_source.go
- [[Render Messages]] - code - context/message.go
- [[Source Budget]] - code - context/prompt_builder.go
- [[Source Interface]] - code - context/prompt_builder.go
- [[Text Content]] - code - context/content.go
- [[Tool Call Content]] - code - context/content.go
- [[Tool Result Content]] - code - context/content.go
- [[Tool Result Error Content]] - code - context/content.go
- [[errors.go_3]] - code - context/errors.go

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Content_Builder_Mistral
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Tokenizer chatMessageRequest mistralToolCallState]]
- 1 edge to [[_COMMUNITY_Tokenizer Model Provider]]

## Top bridge nodes
- [[Mistral Generate]] - degree 4, connects to 1 community
- [[Mistral Model Tests]] - degree 4, connects to 1 community
- [[Gemini Model Tests]] - degree 3, connects to 1 community