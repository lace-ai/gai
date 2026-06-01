# Graph Report - .  (2026-06-01)

## Corpus Check
- 63 files · ~112,554 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 835 nodes · 1532 edges · 54 communities (42 shown, 12 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 176 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Prompt Builder Internals|Prompt Builder Internals]]
- [[_COMMUNITY_Agent Tests And Summaries|Agent Tests And Summaries]]
- [[_COMMUNITY_GAI Package Architecture|GAI Package Architecture]]
- [[_COMMUNITY_Content Serialization|Content Serialization]]
- [[_COMMUNITY_Agent Loop Definition|Agent Loop Definition]]
- [[_COMMUNITY_Echo Tool Tests|Echo Tool Tests]]
- [[_COMMUNITY_History Session Tests|History Session Tests]]
- [[_COMMUNITY_AI Core Types|AI Core Types]]
- [[_COMMUNITY_Prompt Build Concepts|Prompt Build Concepts]]
- [[_COMMUNITY_Provider Request Types|Provider Request Types]]
- [[_COMMUNITY_Loop Tool Pipeline|Loop Tool Pipeline]]
- [[_COMMUNITY_Loop Execution|Loop Execution]]
- [[_COMMUNITY_Streaming Tool Detection|Streaming Tool Detection]]
- [[_COMMUNITY_History Source|History Source]]
- [[_COMMUNITY_RAG Source|RAG Source]]
- [[_COMMUNITY_Mock Model|Mock Model]]
- [[_COMMUNITY_Recording Model Tests|Recording Model Tests]]
- [[_COMMUNITY_Mock Session Store|Mock Session Store]]
- [[_COMMUNITY_Model Repository|Model Repository]]
- [[_COMMUNITY_Mistral Provider|Mistral Provider]]
- [[_COMMUNITY_Mistral Model Tests|Mistral Model Tests]]
- [[_COMMUNITY_Gemini Provider|Gemini Provider]]
- [[_COMMUNITY_XML Renderer|XML Renderer]]
- [[_COMMUNITY_Debug Sink|Debug Sink]]
- [[_COMMUNITY_Conversation Views|Conversation Views]]
- [[_COMMUNITY_AI Response Tokens|AI Response Tokens]]
- [[_COMMUNITY_Mock Provider|Mock Provider]]
- [[_COMMUNITY_Thumbnail Interface|Thumbnail Interface]]
- [[_COMMUNITY_Prompt Composition|Prompt Composition]]
- [[_COMMUNITY_Gemini Provider Tests|Gemini Provider Tests]]
- [[_COMMUNITY_Mistral Provider Tests|Mistral Provider Tests]]
- [[_COMMUNITY_Thumbnail Visual Style|Thumbnail Visual Style]]
- [[_COMMUNITY_Request Tests|Request Tests]]
- [[_COMMUNITY_AI Request|AI Request]]
- [[_COMMUNITY_Store Interfaces|Store Interfaces]]
- [[_COMMUNITY_Summarizer|Summarizer]]
- [[_COMMUNITY_RAG Budgeting Tests|RAG Budgeting Tests]]
- [[_COMMUNITY_Project Workflow|Project Workflow]]
- [[_COMMUNITY_Summary Request|Summary Request]]
- [[_COMMUNITY_Model Interface|Model Interface]]
- [[_COMMUNITY_Provider Interface|Provider Interface]]
- [[_COMMUNITY_Tokenizer Interface|Tokenizer Interface]]
- [[_COMMUNITY_Conversation Interface|Conversation Interface]]
- [[_COMMUNITY_Go CI Workflow|Go CI Workflow]]
- [[_COMMUNITY_Context Conversation Interface|Context Conversation Interface]]

## God Nodes (most connected - your core abstractions)
1. `Builder` - 32 edges
2. `Part` - 28 edges
3. `Required()` - 26 edges
4. `NewPromptBuilder()` - 26 edges
5. `T` - 25 edges
6. `Section` - 23 edges
7. `Model` - 21 edges
8. `NewPart()` - 18 edges
9. `Model` - 17 edges
10. `MockSessionStore` - 15 edges

## Surprising Connections (you probably didn't know these)
- `ModelRepository` --semantically_similar_to--> `ModelRepository`  [INFERRED] [semantically similar]
  ai/model_repository.go → README.md
- `DetectToolCallsInStream` --conceptually_related_to--> `loop Package`  [INFERRED]
  ai/tool_call.go → README.md
- `Prompt` --conceptually_related_to--> `PromptBuilder`  [INFERRED]
  ai/prompt.go → README.md
- `Summarizer` --conceptually_related_to--> `PromptBuilder`  [INFERRED]
  agent/summary/summary.go → README.md
- `Gemini Function Call Mapper` --semantically_similar_to--> `Tool Call Content`  [INFERRED] [semantically similar]
  ai/gemini/model.go → context/content.go

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Core AI Contracts** — ai_Provider, ai_Model, ai_Tokenizer, ai_AIRequest, ai_AIResponse, ai_Prompt [EXTRACTED 1.00]
- **Summary Agent Pipeline** — summary_system_prompt, summary_Definition, summary_Summarizer, summary_Summarize, agent_NewLoop, ai_Token [EXTRACTED 1.00]
- **Tool Call Stream Detection** — ai_DetectToolCallsInStream, ai_parseToolCall, ai_GenerateToolCallID, ai_ToolCall, ai_Token [EXTRACTED 1.00]
- **AI Provider Model Pattern** — gemini_provider, gemini_model, gemini_known_models, mistral_provider, mistral_model, mistral_known_models [INFERRED 0.85]
- **Streaming Tool Call Mapping** — gemini_generate_stream, gemini_map_function_call, mistral_generate_stream, mistral_tool_call_accumulator, content_tool_call_content [INFERRED 0.82]
- **Token Budgeted Prompt Sources** — prompt_builder, prompt_budget, source_budget, history_source, rag_source, builder_budget_fitting, builder_optional_summarization [INFERRED 0.86]
- **Prompt Rendering And Loop Execution** — renderer_renderPrompt, loop_Loop, loop_IncrementalPromptSession [INFERRED 0.75]
- **Tool Call Lifecycle** — iteration_AppendToken, loop_ToolExecutionPipeline, tool_CallTool, echo_tool_EchoTool [EXTRACTED 0.95]
- **Test Mocks Support Loop And History** — model_MockModel, model_MockTokenizer, session_store_MockSessionStore, loop_test_LoopScenarios, session_manager_test_HistorySource [EXTRACTED 0.90]
- **Thumbnail Branding Composition** — GAI_thumbnail_gai_title_typography, GAI_thumbnail_agent_framework_banner, GAI_thumbnail_go_mascot_agent [EXTRACTED 1.00]
- **Automation Agent Visual Theme** — GAI_thumbnail_go_mascot_agent, GAI_thumbnail_dashboard_tablet, GAI_thumbnail_robotic_arms, GAI_thumbnail_neon_sci_fi_workspace [INFERRED 0.85]
- **Neon Symmetric Layout System** — GAI_thumbnail_symmetric_layout, GAI_thumbnail_blue_purple_glow_palette, GAI_thumbnail_robotic_arms [INFERRED 0.80]

## Communities (54 total, 12 thin omitted)

### Community 0 - "Prompt Builder Internals"
Cohesion: 0.06
Nodes (56): builderEntry, builderEntry, builderPromptSession, builderView, BuildTrace, BuildTraceEntry, entryAdmissionOptions, EntryKind (+48 more)

### Community 1 - "Agent Tests And Summaries"
Cohesion: 0.09
Nodes (56): T, TestNewLoopCreatesReusableAgentLoop(), BuildTrace, emptyConversation, EntryOption, fakeSummarizer, Builder, loadPromptFromFile() (+48 more)

### Community 2 - "GAI Package Architecture"
Cohesion: 0.05
Nodes (54): GAI Framework, agent Package, ai Package, context Package, DebugSink, loop Package, ModelRepository, PromptBuilder (+46 more)

### Community 3 - "Content Serialization"
Cohesion: 0.06
Nodes (29): Content, NewContentFromType(), NewTextContent(), NewToolCallContent(), NewToolResultContent(), NewToolResultErrContent(), T, TestContentMarshalRoundTrip() (+21 more)

### Community 4 - "Agent Loop Definition"
Cohesion: 0.06
Nodes (36): Model, Tool, ToolResPreProcessor, NewLoop(), Definition, PromptBuilderFactory, RunInput, Context (+28 more)

### Community 5 - "Echo Tool Tests"
Cohesion: 0.08
Nodes (25): Int32, countingPromptBuilder, ToolCall, ToolResponse, NewEchoTool(), echoArgs, EchoTool, AIRequest (+17 more)

### Community 6 - "History Session Tests"
Cohesion: 0.13
Nodes (30): fakeConversation, Source, History(), rejectingEmptyTokenizer, assertHistoryContainsAll(), assertHistoryContainsNone(), assertHistoryStoreQueries(), Context (+22 more)

### Community 7 - "AI Core Types"
Cohesion: 0.09
Nodes (27): AIRequest, AIResponse, Client, Context, DebugSink, Mutex, Part, Provider (+19 more)

### Community 8 - "Prompt Build Concepts"
Cohesion: 0.08
Nodes (37): Build Trace, Builder Entry Admission, Builder Budget Fitting, Builder Optional Summarization, Content Interface, Content Factory From Type, Content Tests, Text Content (+29 more)

### Community 9 - "Provider Request Types"
Cohesion: 0.09
Nodes (24): AIRequest, AIResponse, Builder, Context, DebugSink, Provider, RawMessage, Token (+16 more)

### Community 10 - "Loop Tool Pipeline"
Cohesion: 0.07
Nodes (34): Agentic Loop Package, Echo Tool, Loop Error Sentinels, Append Token To Iteration, Iteration, Iteration Message Conversion, Iteration Part, Completed Tool Call Deduplication (+26 more)

### Community 11 - "Loop Execution"
Cohesion: 0.09
Nodes (24): IterationInformation, iterationToolCall, Loop, Context, Iteration, Message, Model, PromptBuilder (+16 more)

### Community 12 - "Streaming Tool Detection"
Cohesion: 0.14
Nodes (18): expectedWrapToken, collectTokens(), T, Token, normalizeExpectedTokens(), normalizeJSON(), normalizeTokens(), TestDetectToolCallsInStream() (+10 more)

### Community 13 - "History Source"
Cohesion: 0.14
Nodes (18): Content, countMessageContentTokens(), Context, DebugSink, Message, Part, PromptView, SourceBudget (+10 more)

### Community 14 - "RAG Source"
Cohesion: 0.15
Nodes (15): Document, fakeRAGStore, DebugSink, Source, RAG(), Context, T, TestRAGSourceBudgetsDocumentsInsideGroup() (+7 more)

### Community 15 - "Mock Model"
Cohesion: 0.18
Nodes (8): MockModel, MockModelResponse, MockTokenizer, AIRequest, AIResponse, Context, Token, Tokenizer

### Community 16 - "Recording Model Tests"
Cohesion: 0.19
Nodes (7): AIRequest, AIResponse, Context, Token, Tokenizer, recordingModel, toolCallModel

### Community 17 - "Mock Session Store"
Cohesion: 0.28
Nodes (10): AddMessageCall, AddMessagesCall, CreateSessionCall, GetMessagesCall, GetSessionCall, MockSessionStore, UpdateMessageTokensCall, Context (+2 more)

### Community 18 - "Model Repository"
Cohesion: 0.23
Nodes (8): Context, DebugSink, Model, Provider, NewModelRepository(), T, TestModelRepository(), ModelRepository

### Community 19 - "Mistral Provider"
Cohesion: 0.22
Nodes (9): Client, DebugSink, Model, Mistral Invalid API Key Error, Mistral Known Models, Provider, isKnownModel(), New() (+1 more)

### Community 20 - "Mistral Model Tests"
Cohesion: 0.30
Nodes (11): T, TestModelGenerate(), TestModelGenerateNoChoices(), TestModelGenerateStream(), TestModelGenerateStreamDetectsTextEncodedToolCall(), TestModelGenerateStreamToolCall(), TestModelGenerateStreamToolCallDeltas(), TestModelTokenizerCountTokens() (+3 more)

### Community 21 - "Gemini Provider"
Cohesion: 0.26
Nodes (8): DebugSink, Model, Gemini Invalid API Key Error, Gemini Known Models, Provider, isKnownModel(), New(), Gemini Provider Tests

### Community 22 - "XML Renderer"
Cohesion: 0.38
Nodes (9): Renderer, Builder, Part, Prompt, Section, renderPrompt(), writeEscaped(), writeXMLPart() (+1 more)

### Community 23 - "Debug Sink"
Cohesion: 0.27
Nodes (5): Context, DebugEvent, DebugSink, DebugSinkFunc, SensitiveDebugSinkFunc

### Community 24 - "Conversation Views"
Cohesion: 0.36
Nodes (4): Conversation, Section, testPromptView, EntryView

### Community 25 - "AI Response Tokens"
Cohesion: 0.38
Nodes (4): AIResponse, ToolCall, Token, TokenType

### Community 27 - "Thumbnail Interface"
Cohesion: 0.60
Nodes (5): Agent Framework Banner, Dashboard Tablet Interface, GAI Agent Framework Thumbnail, GAI Title Typography, Go Mascot Agent

### Community 28 - "Prompt Composition"
Cohesion: 0.60
Nodes (3): Prompt, Context, DebugSink

### Community 29 - "Gemini Provider Tests"
Cohesion: 0.60
Nodes (4): T, TestProviderModelAndListModels(), TestProviderModelValidation(), TestProviderValidate()

### Community 30 - "Mistral Provider Tests"
Cohesion: 0.60
Nodes (4): T, TestProviderModelAndListModels(), TestProviderModelValidation(), TestProviderValidate()

### Community 31 - "Thumbnail Visual Style"
Cohesion: 0.50
Nodes (4): Blue Purple Glow Palette, Neon Sci Fi Workspace, Mirrored Robotic Arms, Symmetric Centered Layout

### Community 32 - "Request Tests"
Cohesion: 0.67
Nodes (3): T, TestAIRequestCombinedPrompt(), TestAIRequestCombinedPromptSkipsEmptySections()

### Community 36 - "RAG Budgeting Tests"
Cohesion: 0.67
Nodes (3): RAG Source Document Budgeting Tests, Fake RAG Store, RAG Store Interface

## Knowledge Gaps
- **153 isolated node(s):** `Model`, `Tool`, `ToolResPreProcessor`, `Loop`, `T` (+148 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestLoopAppendsIterationMessagesToIncrementalPrompt()` connect `Agent Tests And Summaries` to `Echo Tool Tests`?**
  _High betweenness centrality (0.199) - this node is a cross-community bridge._
- **Why does `NewPromptBuilder()` connect `Agent Tests And Summaries` to `Prompt Builder Internals`, `Echo Tool Tests`?**
  _High betweenness centrality (0.146) - this node is a cross-community bridge._
- **Why does `DetectToolCallsInStream()` connect `Streaming Tool Detection` to `Provider Request Types`, `Echo Tool Tests`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Are the 24 inferred relationships involving `Required()` (e.g. with `TestNewLoopCreatesReusableAgentLoop()` and `.SystemPrompt()`) actually correct?**
  _`Required()` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `NewPromptBuilder()` (e.g. with `TestNewLoopCreatesReusableAgentLoop()` and `TestPromptBuilderBudgetsRequiredSourceBeforeEarlierOptionalContext()`) actually correct?**
  _`NewPromptBuilder()` has 24 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Model`, `Tool`, `ToolResPreProcessor` to the rest of the system?**
  _153 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Prompt Builder Internals` be split into smaller, more focused modules?**
  _Cohesion score 0.06060606060606061 - nodes in this community are weakly interconnected._