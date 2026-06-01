---
source_file: "context/prompt_builder.go"
type: "code"
community: "testgo Part Context"
location: "L171"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/testgo_Part_Context
---

# Required()

## Connections
- [[.SystemPrompt()]] - `calls` [INFERRED]
- [[.ToolSysPrompt()]] - `calls` [INFERRED]
- [[.UserPrompt()]] - `calls` [INFERRED]
- [[Definition()]] - `calls` [INFERRED]
- [[EntryOption]] - `references` [EXTRACTED]
- [[TestLoopAppendsIterationMessagesToIncrementalPrompt()]] - `calls` [INFERRED]
- [[TestNewLoopCreatesReusableAgentLoop()]] - `calls` [INFERRED]
- [[TestPromptBuilderBudgetsRequiredSourceBeforeEarlierOptionalContext()]] - `calls` [INFERRED]
- [[TestPromptBuilderBuildsStructuredPrompt()]] - `calls` [INFERRED]
- [[TestPromptBuilderCountsSourcePartsWithoutExplicitTokens()]] - `calls` [INFERRED]
- [[TestPromptBuilderDropsEarlierOptionalContextForLaterUserPrompt()]] - `calls` [INFERRED]
- [[TestPromptBuilderDropsOptionalSourceOverBudget()]] - `calls` [INFERRED]
- [[TestPromptBuilderDropsOptionalStaticSystemPartOverBudget()]] - `calls` [INFERRED]
- [[TestPromptBuilderFailsRequiredOverBudget()]] - `calls` [INFERRED]
- [[TestPromptBuilderPassesSourceCap()]] - `calls` [INFERRED]
- [[TestPromptBuilderRejectsDuplicateEmittedPartIDs()]] - `calls` [INFERRED]
- [[TestPromptBuilderRendersRequiredPartsBeforeOptionalParts()]] - `calls` [INFERRED]
- [[TestPromptBuilderReusesTokenCountForSourceBudget()]] - `calls` [INFERRED]
- [[TestPromptBuilderSourceCanInspectWholePlan()]] - `calls` [INFERRED]
- [[TestPromptBuilderSourceFailurePolicy()]] - `calls` [INFERRED]
- [[TestPromptBuilderSummarizesOptionalStaticUserPartBeforeDropping()]] - `calls` [INFERRED]
- [[TestPromptBuilderTraceSplitsEntryAndPromptTokens()]] - `calls` [INFERRED]
- [[TestPromptSessionRebuildsSourcesWhenConversationReserveIsExceeded()]] - `calls` [INFERRED]
- [[newHistoryPart()]] - `calls` [INFERRED]
- [[prompt_builder.go]] - `contains` [EXTRACTED]
- [[testPromptBuilder()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/testgo_Part_Context