---
type: community
cohesion: 0.09
members: 66
---

# testgo Part Context

**Cohesion:** 0.09 - loosely connected
**Members:** 66 nodes

## Members
- [[.BuildParts()_2]] - code - context/rag_source.go
- [[.Messages()]] - code - context/prompt_builder_test.go
- [[.Render()]] - code - context/prompt_builder_test.go
- [[.Summarize()_1]] - code - context/prompt_builder_test.go
- [[.SystemPrompt()]] - code - context/helper.go
- [[.ToolSysPrompt()]] - code - context/helper.go
- [[.UserPrompt()]] - code - context/helper.go
- [[BuildTrace_1]] - code - context/prompt_builder_test.go
- [[BuildTraceEntry_2]] - code - context/prompt_builder_test.go
- [[Builder_1]] - code - context/helper.go
- [[Context_9]] - code - context/prompt_builder_test.go
- [[Context_10]] - code - context/rag_source.go
- [[Definition()]] - code - agent/summary/summary.go
- [[EntryOption]] - code - context/prompt_builder.go
- [[Message_3]] - code - context/prompt_builder_test.go
- [[Meta()]] - code - context/prompt_builder.go
- [[NewPart()]] - code - context/prompt_builder.go
- [[NewPromptBuilder()]] - code - context/prompt_builder.go
- [[Optional()]] - code - context/prompt_builder.go
- [[Part_3]] - code - context/prompt_builder_test.go
- [[Part_4]] - code - context/rag_source.go
- [[PromptView_3]] - code - context/rag_source.go
- [[Required()]] - code - context/prompt_builder.go
- [[Section_1]] - code - context/prompt_builder_test.go
- [[SourceBudget_2]] - code - context/rag_source.go
- [[SourceFunc]] - code - context/prompt_builder.go
- [[SourceTokenCap()]] - code - context/prompt_builder.go
- [[SummaryRequest_1]] - code - context/prompt_builder_test.go
- [[T]] - code - agent/agent_test.go
- [[T_10]] - code - context/prompt_builder_test.go
- [[TestLoopAppendsIterationMessagesToIncrementalPrompt()]] - code - loop/loop_test.go
- [[TestNewLoopCreatesReusableAgentLoop()]] - code - agent/agent_test.go
- [[TestPromptBuilderBudgetsRequiredSourceBeforeEarlierOptionalContext()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderBuildsStructuredPrompt()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderCountsSourcePartsWithoutExplicitTokens()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderDropsEarlierOptionalContextForLaterUserPrompt()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderDropsOptionalSourceOverBudget()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderDropsOptionalStaticSystemPartOverBudget()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderEmitsDebugEvents()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderEscapesPartIDs()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderFailsRequiredOverBudget()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderPassesSourceCap()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderRejectsDuplicateEmittedPartIDs()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderRejectsDuplicateIDs()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderRendersRequiredPartsBeforeOptionalParts()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderReusesTokenCountForSourceBudget()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderSourceCanInspectWholePlan()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderSourceFailurePolicy()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderSummarizesOptionalStaticUserPartBeforeDropping()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderTraceSplitsEntryAndPromptTokens()]] - code - context/prompt_builder_test.go
- [[TestPromptBuilderUsesCustomRenderer()]] - code - context/prompt_builder_test.go
- [[TestPromptSessionRebuildsSourcesWhenConversationReserveIsExceeded()]] - code - context/prompt_builder_test.go
- [[Tokens()]] - code - context/prompt_builder.go
- [[agent_test.go]] - code - agent/agent_test.go
- [[assertContainsAll()]] - code - context/prompt_builder_test.go
- [[assertContainsNone()]] - code - context/prompt_builder_test.go
- [[assertOrdered()]] - code - context/prompt_builder_test.go
- [[emptyConversation]] - code - context/prompt_builder_test.go
- [[fakeSummarizer]] - code - context/prompt_builder_test.go
- [[helper.go]] - code - context/helper.go
- [[loadPromptFromFile()]] - code - context/helper.go
- [[newHistoryPart()]] - code - context/history_source.go
- [[prompt_builder_test.go]] - code - context/prompt_builder_test.go
- [[sectionNameRenderer]] - code - context/prompt_builder_test.go
- [[traceEntry()]] - code - context/prompt_builder_test.go
- [[traceEntryStatus()]] - code - context/prompt_builder_test.go

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/testgo_Part_Context
SORT file.name ASC
```

## Connections to other communities
- 18 edges to [[_COMMUNITY_Part promptLimit Source]]
- 6 edges to [[_COMMUNITY_Name GenerateStream Tokenizer]]
- 4 edges to [[_COMMUNITY_Definition Model Tool]]
- 3 edges to [[_COMMUNITY_DebugSink Message history]]
- 3 edges to [[_COMMUNITY_rag DebugSink Document]]
- 1 edge to [[_COMMUNITY_Emit IncludeSensitiveData debuggo]]
- 1 edge to [[_COMMUNITY_Tokenize CountTokens History]]

## Top bridge nodes
- [[Required()]] - degree 26, connects to 2 communities
- [[NewPromptBuilder()]] - degree 26, connects to 2 communities
- [[NewPart()]] - degree 18, connects to 2 communities
- [[.BuildParts()_2]] - degree 10, connects to 2 communities
- [[SourceFunc]] - degree 14, connects to 1 community