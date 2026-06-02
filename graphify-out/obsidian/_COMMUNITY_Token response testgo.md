---
type: community
cohesion: 0.14
members: 22
---

# Token response testgo

**Cohesion:** 0.14 - loosely connected
**Members:** 22 nodes

## Members
- [[.String()_1]] - code - ai/tool_call.go
- [[.Validate()_3]] - code - ai/tool_call.go
- [[Context_6]] - code - ai/tool_call.go
- [[DebugSink_6]] - code - ai/tool_call.go
- [[DetectToolCallsInStream()]] - code - ai/tool_call.go
- [[RawMessage_2]] - code - ai/tool_call.go
- [[T_8]] - code - ai/response_test.go
- [[TestDetectToolCallsInStream()]] - code - ai/response_test.go
- [[Token_4]] - code - ai/response_test.go
- [[Token_5]] - code - ai/tool_call.go
- [[TokenType_1]] - code - ai/response_test.go
- [[ToolCall_4]] - code - ai/tool_call.go
- [[collectTokens()]] - code - ai/response_test.go
- [[expectedWrapToken]] - code - ai/response_test.go
- [[isWS()]] - code - ai/tool_call.go
- [[joinTokenData()]] - code - ai/tool_call.go
- [[normalizeExpectedTokens()]] - code - ai/response_test.go
- [[normalizeJSON()]] - code - ai/response_test.go
- [[normalizeTokens()]] - code - ai/response_test.go
- [[parseToolCall()]] - code - ai/tool_call.go
- [[response_test.go]] - code - ai/response_test.go
- [[tool_call.go]] - code - ai/tool_call.go

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Token_response_testgo
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Tokenizer Model Provider]]
- 1 edge to [[_COMMUNITY_Tokenizer chatMessageRequest mistralToolCallState]]
- 1 edge to [[_COMMUNITY_Name GenerateStream Tokenizer]]

## Top bridge nodes
- [[DetectToolCallsInStream()]] - degree 10, connects to 2 communities
- [[tool_call.go]] - degree 6, connects to 1 community
- [[parseToolCall()]] - degree 5, connects to 1 community