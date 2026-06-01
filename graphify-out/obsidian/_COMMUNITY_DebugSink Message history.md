---
type: community
cohesion: 0.14
members: 22
---

# DebugSink Message history

**Cohesion:** 0.14 - loosely connected
**Members:** 22 nodes

## Members
- [[.BuildParts()]] - code - context/history_source.go
- [[.DebugSink()]] - code - context/history_source.go
- [[.countRenderedMessages()]] - code - context/history_source.go
- [[Builder_2]] - code - context/message.go
- [[Content_1]] - code - context/message.go
- [[Context_7]] - code - context/history_source.go
- [[DebugSink_7]] - code - context/history_source.go
- [[HistorySource]] - code - context/history_source.go
- [[IsValidRole()]] - code - context/message.go
- [[Message]] - code - context/history_source.go
- [[Message_1]] - code - context/message.go
- [[Part_1]] - code - context/history_source.go
- [[PromptView]] - code - context/history_source.go
- [[RenderMessages()]] - code - context/message.go
- [[Role]] - code - context/message.go
- [[SessionStore]] - code - context/history_source.go
- [[SourceBudget]] - code - context/history_source.go
- [[Time]] - code - context/message.go
- [[Tokenizer_6]] - code - context/history_source.go
- [[countMessageContentTokens()]] - code - context/history_source.go
- [[history_source.go]] - code - context/history_source.go
- [[storedMessageTokens()]] - code - context/history_source.go

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/DebugSink_Message_history
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_testgo Part Context]]
- 2 edges to [[_COMMUNITY_Tokenize CountTokens History]]
- 2 edges to [[_COMMUNITY_Part promptLimit Source]]

## Top bridge nodes
- [[history_source.go]] - degree 5, connects to 2 communities
- [[.BuildParts()]] - degree 8, connects to 1 community
- [[countMessageContentTokens()]] - degree 6, connects to 1 community
- [[RenderMessages()]] - degree 4, connects to 1 community
- [[SessionStore]] - degree 3, connects to 1 community