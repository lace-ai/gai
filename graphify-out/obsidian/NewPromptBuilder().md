---
source_file: "context/prompt_builder.go"
type: "code"
community: "testgo Part Context"
location: "L206"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/testgo_Part_Context
---

# NewPromptBuilder()

## Connections
- [[Builder_3]] - `references` [EXTRACTED]
- [[Definition()]] - `calls` [INFERRED]
- [[TestLoopAppendsIterationMessagesToIncrementalPrompt()]] - `calls` [INFERRED]
- [[TestNewLoopCreatesReusableAgentLoop()]] - `calls` [INFERRED]
- [[TestPromptBuilderBudgetsRequiredSourceBeforeEarlierOptionalContext()]] - `calls` [INFERRED]
- [[TestPromptBuilderBuildsStructuredPrompt()]] - `calls` [INFERRED]
- [[TestPromptBuilderCountsSourcePartsWithoutExplicitTokens()]] - `calls` [INFERRED]
- [[TestPromptBuilderDropsEarlierOptionalContextForLaterUserPrompt()]] - `calls` [INFERRED]
- [[TestPromptBuilderDropsOptionalSourceOverBudget()]] - `calls` [INFERRED]
- [[TestPromptBuilderDropsOptionalStaticSystemPartOverBudget()]] - `calls` [INFERRED]
- [[TestPromptBuilderEmitsDebugEvents()]] - `calls` [INFERRED]
- [[TestPromptBuilderEscapesPartIDs()]] - `calls` [INFERRED]
- [[TestPromptBuilderFailsRequiredOverBudget()]] - `calls` [INFERRED]
- [[TestPromptBuilderPassesSourceCap()]] - `calls` [INFERRED]
- [[TestPromptBuilderRejectsDuplicateEmittedPartIDs()]] - `calls` [INFERRED]
- [[TestPromptBuilderRejectsDuplicateIDs()]] - `calls` [INFERRED]
- [[TestPromptBuilderRendersRequiredPartsBeforeOptionalParts()]] - `calls` [INFERRED]
- [[TestPromptBuilderReusesTokenCountForSourceBudget()]] - `calls` [INFERRED]
- [[TestPromptBuilderSourceCanInspectWholePlan()]] - `calls` [INFERRED]
- [[TestPromptBuilderSourceFailurePolicy()]] - `calls` [INFERRED]
- [[TestPromptBuilderSummarizesOptionalStaticUserPartBeforeDropping()]] - `calls` [INFERRED]
- [[TestPromptBuilderTraceSplitsEntryAndPromptTokens()]] - `calls` [INFERRED]
- [[TestPromptBuilderUsesCustomRenderer()]] - `calls` [INFERRED]
- [[TestPromptSessionRebuildsSourcesWhenConversationReserveIsExceeded()]] - `calls` [INFERRED]
- [[prompt_builder.go]] - `contains` [EXTRACTED]
- [[testPromptBuilder()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/testgo_Part_Context