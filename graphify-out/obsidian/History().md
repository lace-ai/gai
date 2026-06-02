---
source_file: "context/history_source.go"
type: "code"
community: "Tokenize CountTokens History"
location: "L18"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Tokenize_CountTokens_History
---

# History()

## Connections
- [[SessionStore]] - `references` [EXTRACTED]
- [[Source]] - `references` [EXTRACTED]
- [[TestHistorySourceBuildsPartsWithinTokenBudget()]] - `calls` [INFERRED]
- [[TestHistorySourceIgnoresCurrentLoopBudget()]] - `calls` [INFERRED]
- [[TestHistorySourcePropagatesStoreErrors()]] - `calls` [INFERRED]
- [[TestHistorySourcePropagatesTokenizerErrors()]] - `calls` [INFERRED]
- [[TestHistorySourceRequiresStore()]] - `calls` [INFERRED]
- [[TestHistorySourceRequiresTokenizer()]] - `calls` [INFERRED]
- [[TestHistorySourceSkipsEmptyCurrentLoop()]] - `calls` [INFERRED]
- [[TestHistorySourceUsesEntryRequiredness()]] - `calls` [INFERRED]
- [[TestHistorySourceUsesStoredMessageTokenCounts()]] - `calls` [INFERRED]
- [[history_source.go]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/Tokenize_CountTokens_History