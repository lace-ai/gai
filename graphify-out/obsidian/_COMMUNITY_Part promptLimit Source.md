---
type: community
cohesion: 0.06
members: 100
---

# Part promptLimit Source

**Cohesion:** 0.06 - loosely connected
**Members:** 100 nodes

## Members
- [[.AppendMessages()]] - code - context/prompt_builder.go
- [[.Budget()]] - code - context/prompt_builder.go
- [[.BuildParts()_1]] - code - context/prompt_builder.go
- [[.BuildPrompt()]] - code - context/prompt_builder.go
- [[.ContentLimit()]] - code - context/prompt_builder.go
- [[.Context()]] - code - context/prompt_builder.go
- [[.Conversation()]] - code - context/prompt_builder.go
- [[.Debug()]] - code - context/prompt_builder.go
- [[.Entries()]] - code - context/prompt_builder.go
- [[.Entry()]] - code - context/prompt_builder.go
- [[.LastTrace()]] - code - context/prompt_builder.go
- [[.Part()]] - code - context/prompt_builder.go
- [[.Prompt()]] - code - context/prompt_builder.go
- [[.Renderer()]] - code - context/prompt_builder.go
- [[.SectionEntries()]] - code - context/prompt_builder.go
- [[.Source()]] - code - context/prompt_builder.go
- [[.StartPrompt()]] - code - context/prompt_builder.go
- [[.System()]] - code - context/prompt_builder.go
- [[.User()]] - code - context/prompt_builder.go
- [[.admitEntryParts()]] - code - context/prompt_builder.go
- [[.appendParts()]] - code - context/prompt_builder.go
- [[.clone()]] - code - context/prompt_builder.go
- [[.countPrompt()]] - code - context/prompt_builder.go
- [[.dropOptionalContextFor()]] - code - context/prompt_builder.go
- [[.emit()]] - code - context/prompt_builder.go
- [[.emitEntry()]] - code - context/prompt_builder.go
- [[.failEntry()]] - code - context/prompt_builder.go
- [[.finalTrace()]] - code - context/prompt_builder.go
- [[.nextParts()]] - code - context/prompt_builder.go
- [[.normalizeMissingPartTokens()]] - code - context/prompt_builder.go
- [[.normalizePartTokens()]] - code - context/prompt_builder.go
- [[.part()]] - code - context/prompt_builder.go
- [[.partsFit()]] - code - context/prompt_builder.go
- [[.partsFitAfterDroppingOptionalContext()]] - code - context/prompt_builder.go
- [[.promptLimit()]] - code - context/prompt_builder.go
- [[.promptLimit()_1]] - code - context/prompt_builder.go
- [[.rebuildBase()]] - code - context/prompt_builder.go
- [[.record()]] - code - context/prompt_builder.go
- [[.sourceBudget()]] - code - context/prompt_builder.go
- [[.summarizeOptionalPart()]] - code - context/prompt_builder.go
- [[.tokenCount()]] - code - context/prompt_builder.go
- [[.validate()]] - code - context/prompt_builder.go
- [[.view()]] - code - context/prompt_builder.go
- [[BuildTrace]] - code - context/prompt_builder.go
- [[BuildTraceEntry]] - code - context/prompt_builder.go
- [[BuildTraceEntry_1]] - code - context/prompt_builder.go
- [[Builder_3]] - code - context/prompt_builder.go
- [[Context_8]] - code - context/prompt_builder.go
- [[Conversation_1]] - code - context/prompt_builder.go
- [[DebugSink_8]] - code - context/prompt_builder.go
- [[EntryKind]] - code - context/prompt_builder.go
- [[EntryView]] - code - context/prompt_builder.go
- [[IncrementalPromptBuilder]] - code - context/prompt_builder.go
- [[Message_2]] - code - context/prompt_builder.go
- [[NewPartGroup()]] - code - context/prompt_builder.go
- [[Part_2]] - code - context/prompt_builder.go
- [[Prompt_2]] - code - context/prompt_builder.go
- [[PromptBudget_1]] - code - context/prompt_builder.go
- [[PromptBuilder]] - code - context/prompt_builder.go
- [[PromptSession]] - code - context/prompt_builder.go
- [[PromptView_1]] - code - context/prompt_builder.go
- [[PromptView_2]] - code - context/prompt_builder.go
- [[Renderer]] - code - context/prompt_builder.go
- [[Section]] - code - context/prompt_builder.go
- [[Source_1]] - code - context/prompt_builder.go
- [[SourceBudget_1]] - code - context/prompt_builder.go
- [[Summarizer_2]] - code - context/prompt_builder.go
- [[Tokenizer_7]] - code - context/prompt_builder.go
- [[addPartIDs()]] - code - context/prompt_builder.go
- [[appendPromptText()]] - code - context/prompt_builder.go
- [[applyOptions()]] - code - context/prompt_builder.go
- [[builderEntry]] - code - context/prompt_builder.go
- [[builderEntry_1]] - code - context/prompt_builder.go
- [[builderPromptSession]] - code - context/prompt_builder.go
- [[builderView]] - code - context/prompt_builder.go
- [[cloneEntryViews()]] - code - context/prompt_builder.go
- [[cloneMessages()]] - code - context/prompt_builder.go
- [[cloneMeta()]] - code - context/prompt_builder.go
- [[clonePartIDMap()]] - code - context/prompt_builder.go
- [[cloneParts()]] - code - context/prompt_builder.go
- [[clonePartsMap()]] - code - context/prompt_builder.go
- [[cloneTrace()]] - code - context/prompt_builder.go
- [[entryAdmissionOptions]] - code - context/prompt_builder.go
- [[finalizeTrace()]] - code - context/prompt_builder.go
- [[keepRequiredParts()]] - code - context/prompt_builder.go
- [[markDroppedOptionalContextEntries()]] - code - context/prompt_builder.go
- [[markRequired()]] - code - context/prompt_builder.go
- [[newBuilderView()]] - code - context/prompt_builder.go
- [[newPromptBuildState()]] - code - context/prompt_builder.go
- [[orderedEntries()]] - code - context/prompt_builder.go
- [[partsTokenCount()]] - code - context/prompt_builder.go
- [[promptBudgetError()]] - code - context/prompt_builder.go
- [[promptBuildState]] - code - context/prompt_builder.go
- [[promptBuildState_1]] - code - context/prompt_builder.go
- [[prompt_builder.go]] - code - context/prompt_builder.go
- [[rebuildPartIDs()]] - code - context/prompt_builder.go
- [[renderOverheadTokens()]] - code - context/prompt_builder.go
- [[setTraceTokens()]] - code - context/prompt_builder.go
- [[validSection()]] - code - context/prompt_builder.go
- [[validatePartIDs()]] - code - context/prompt_builder.go

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Part_promptLimit_Source
SORT file.name ASC
```

## Connections to other communities
- 18 edges to [[_COMMUNITY_testgo Part Context]]
- 2 edges to [[_COMMUNITY_DebugSink Message history]]
- 1 edge to [[_COMMUNITY_rag DebugSink Document]]
- 1 edge to [[_COMMUNITY_Renderer XMLRenderer Render]]

## Top bridge nodes
- [[NewPartGroup()]] - degree 9, connects to 2 communities
- [[prompt_builder.go]] - degree 53, connects to 1 community
- [[Builder_3]] - degree 32, connects to 1 community
- [[Part_2]] - degree 28, connects to 1 community
- [[.BuildPrompt()]] - degree 22, connects to 1 community