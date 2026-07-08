# Graph Report - .  (2026-07-06)

## Corpus Check
- 95 files · ~256,069 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1522 nodes · 2946 edges · 95 communities (82 shown, 13 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 312 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_textRunInput()|textRunInput()]]
- [[_COMMUNITY_workflow go|workflow go]]
- [[_COMMUNITY_loop test go|loop test go]]
- [[_COMMUNITY_StartOperationSpan()|StartOperationSpan()]]
- [[_COMMUNITY_historyObserver|historyObserver]]
- [[_COMMUNITY_model go|model go]]
- [[_COMMUNITY_search go|search go]]
- [[_COMMUNITY_Prompt Builder|Prompt Builder]]
- [[_COMMUNITY_session manager test go|session manager test go]]
- [[_COMMUNITY_searchObserver|searchObserver]]
- [[_COMMUNITY_summary go|summary go]]
- [[_COMMUNITY_renderer go|renderer go]]
- [[_COMMUNITY_Context|Context]]
- [[_COMMUNITY_HistorySource|HistorySource]]
- [[_COMMUNITY_source go|source go]]
- [[_COMMUNITY_Builder|Builder]]
- [[_COMMUNITY_Run()|Run()]]
- [[_COMMUNITY_loopRunState|loopRunState]]
- [[_COMMUNITY_capability go|capability go]]
- [[_COMMUNITY_RAG()|RAG()]]
- [[_COMMUNITY_agent go|agent go]]
- [[_COMMUNITY_content go|content go]]
- [[_COMMUNITY_source test go|source test go]]
- [[_COMMUNITY_model go|model go]]
- [[_COMMUNITY_prompt part go|prompt part go]]
- [[_COMMUNITY_NewNamedPart()|NewNamedPart()]]
- [[_COMMUNITY_part go|part go]]
- [[_COMMUNITY_NewTextPart()|NewTextPart()]]
- [[_COMMUNITY_renderObserver|renderObserver]]
- [[_COMMUNITY_message go|message go]]
- [[_COMMUNITY_MockModel|MockModel]]
- [[_COMMUNITY_MockSessionStore|MockSessionStore]]
- [[_COMMUNITY_ModelRepository|ModelRepository]]
- [[_COMMUNITY_NewTextContent()|NewTextContent()]]
- [[_COMMUNITY_iteration go|iteration go]]
- [[_COMMUNITY_DetectToolCallsInStream()|DetectToolCallsInStream()]]
- [[_COMMUNITY_Context|Context]]
- [[_COMMUNITY_message test go|message test go]]
- [[_COMMUNITY_Context|Context]]
- [[_COMMUNITY_T|T]]
- [[_COMMUNITY_renderer test go|renderer test go]]
- [[_COMMUNITY_scriptedWorkflowModel|scriptedWorkflowModel]]
- [[_COMMUNITY_buildChatCompletionRequest()|buildChatCompletionRequest()]]
- [[_COMMUNITY_response test go|response test go]]
- [[_COMMUNITY_Provider|Provider]]
- [[_COMMUNITY_GenerateStream()|GenerateStream()]]
- [[_COMMUNITY_Provider|Provider]]
- [[_COMMUNITY_CountTokens()|CountTokens()]]
- [[_COMMUNITY_Token|Token]]
- [[_COMMUNITY_NewContentFromType()|NewContentFromType()]]
- [[_COMMUNITY_MockProvider|MockProvider]]
- [[_COMMUNITY_AIRequest|AIRequest]]
- [[_COMMUNITY_LoadPromptFromFile()|LoadPromptFromFile()]]
- [[_COMMUNITY_historyStore|historyStore]]
- [[_COMMUNITY_Agent Framework Thumbnail|Agent Framework Thumbnail]]
- [[_COMMUNITY_Prompt|Prompt]]
- [[_COMMUNITY_toolSignatureTestTool|toolSignatureTestTool]]
- [[_COMMUNITY_provider test go|provider test go]]
- [[_COMMUNITY_provider test go|provider test go]]
- [[_COMMUNITY_Neon Sci Fi Workspace|Neon Sci Fi Workspace]]
- [[_COMMUNITY_rendererDebugSink|rendererDebugSink]]
- [[_COMMUNITY_History Source Tests|History Source Tests]]
- [[_COMMUNITY_CombinedPrompt|CombinedPrompt]]
- [[_COMMUNITY_TestAIRequestStoresPromptString()|TestAIRequestStoresPromptString()]]
- [[_COMMUNITY_TestToolCallStringNilReceiver()|TestToolCallStringNilReceiver()]]
- [[_COMMUNITY_TestPartTokensRecountsNegativeCachedValue()|TestPartTokensRecountsNegativeCachedValue()]]
- [[_COMMUNITY_store go|store go]]
- [[_COMMUNITY_summarize go|summarize go]]
- [[_COMMUNITY_RAG Source|RAG Source]]
- [[_COMMUNITY_RAG Store Interface|RAG Store Interface]]
- [[_COMMUNITY_Go 1 26 x Project|Go 1 26 x Project]]
- [[_COMMUNITY_Summarizer Interface|Summarizer Interface]]
- [[_COMMUNITY_XML Renderer Group Rendering Test|XML Renderer Group Rendering Test]]
- [[_COMMUNITY_Summary System Prompt Text|Summary System Prompt Text]]
- [[_COMMUNITY_LGPL v2 1 License|LGPL v2 1 License]]

## God Nodes (most connected - your core abstractions)
1. `NewTextContent()` - 38 edges
2. `historyObserver` - 31 edges
3. `Builder` - 25 edges
4. `StartOperationSpan()` - 24 edges
5. `EndSpan()` - 23 edges
6. `promptBuilderObserver` - 21 edges
7. `Context` - 20 edges
8. `textRunInput()` - 18 edges
9. `consumeWorkflow()` - 18 edges
10. `NewEchoTool()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `TestAgentNewRunCreatesLoop()` --calls--> `NewEchoTool()`  [INFERRED]
  agent/agent_test.go → loop/echo_tool.go
- `TestAgentToolsAutomaticallyAddPromptContract()` --calls--> `NewEchoTool()`  [INFERRED]
  agent/agent_test.go → loop/echo_tool.go
- `TestAgentToolDefinitionOptionsCustomizeAutomaticPromptContract()` --calls--> `NewEchoTool()`  [INFERRED]
  agent/agent_test.go → loop/echo_tool.go
- `TestAgentToolDefinitionOptionsCustomizeAutomaticPromptContract()` --calls--> `WithUsageProtocol()`  [INFERRED]
  agent/agent_test.go → context/tooldefinitions/source.go
- `TestAgentDoesNotDuplicateExistingToolDefinitions()` --calls--> `NewEchoTool()`  [INFERRED]
  agent/agent_test.go → loop/echo_tool.go

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thumbnail Branding Composition** — GAI_thumbnail_gai_title_typography, GAI_thumbnail_agent_framework_banner, GAI_thumbnail_go_mascot_agent [EXTRACTED 1.00]
- **Automation Agent Visual Theme** — GAI_thumbnail_go_mascot_agent, GAI_thumbnail_dashboard_tablet, GAI_thumbnail_robotic_arms, GAI_thumbnail_neon_sci_fi_workspace [INFERRED 0.85]
- **Neon Symmetric Layout System** — GAI_thumbnail_symmetric_layout, GAI_thumbnail_blue_purple_glow_palette, GAI_thumbnail_robotic_arms [INFERRED 0.80]
- **GAI Core Architecture** — README_ai_package, README_context_package, README_loop_package, README_agent_package [EXTRACTED 1.00]
- **Agent Workflow Pipeline** — README_agent_definition, README_workflow, README_loop_execution, README_stream_middleware, README_workflow_result [EXTRACTED 1.00]
- **Repository Automation And Quality** — coderabbit_code_review_configuration, coderabbit_gitleaks, coderabbit_semgrep, dependabot_version_updates, dependabot_gomod_ecosystem [INFERRED 0.78]

## Communities (95 total, 13 thin omitted)

### Community 0 - "textRunInput()"
Cohesion: 0.06
Nodes (58): Context, ContextSource, Conversation, Part, PromptInput, RunInput, T, Tokenizer (+50 more)

### Community 1 - "workflow go"
Cohesion: 0.07
Nodes (61): AgentMiddleware, AgentMiddlewareConfig, AgentResult, capturedStream, ErrorPolicy, forwardErrors(), forwardTokens(), Agent (+53 more)

### Community 2 - "loop test go"
Cohesion: 0.06
Nodes (45): EventType, countingPromptBuilder, ToolParameters, NewEchoTool(), echoArgs, EchoTool, failingToolResponseProcessor, collectLoopEvents() (+37 more)

### Community 3 - "StartOperationSpan()"
Cohesion: 0.07
Nodes (37): appendRenderNodeText(), clippedPrompt(), Builder, Context, DebugSink, Part, RenderNode, Span (+29 more)

### Community 4 - "historyObserver"
Cohesion: 0.07
Nodes (23): Context, DebugSink, Part, Span, Summary, Turn, Builder, Context (+15 more)

### Community 5 - "model go"
Cohesion: 0.07
Nodes (44): AIRequest, AIResponse, Client, Context, DebugSink, Mutex, Part, Provider (+36 more)

### Community 6 - "search go"
Cohesion: 0.08
Nodes (37): APIError, captureDebugSink, Option, decodeAPIError(), NewSearchTool(), NewSearchToolFromEnv(), attributeMap(), TestNewSearchToolValidatesConfiguration() (+29 more)

### Community 7 - "Prompt Builder"
Cohesion: 0.06
Nodes (46): Agent Definition, AgentMiddleware, agent Package, Agent Style Applications, ai Package, AIRequest, AIResponse, context Package (+38 more)

### Community 8 - "session manager test go"
Cohesion: 0.10
Nodes (34): fakeConversation, Source, History(), rejectingEmptyTokenizer, assertHistoryContainsAll(), assertHistoryContainsNone(), assertHistoryStoreQueries(), Context (+26 more)

### Community 9 - "searchObserver"
Cohesion: 0.08
Nodes (28): T, TestCloneWorkflowResultOwnsMutableExecutionData(), newSearchObserver(), searchObserver, Context, ToolCall, ToolResponse, CallTool() (+20 more)

### Community 10 - "summary go"
Cohesion: 0.08
Nodes (29): Context, Model, Summarizer, Tokenizer, Tool, AIRequest, AIResponse, Context (+21 more)

### Community 11 - "renderer go"
Cohesion: 0.14
Nodes (32): formatSimpleInstructionLabel(), formatSimpleLine(), Builder, Context, DebugSink, Part, indent(), isSimpleTextInstruction() (+24 more)

### Community 12 - "Context"
Cohesion: 0.11
Nodes (25): middlewareObserver, agentResultFields(), errorPolicyName(), Agent, AgentResult, capturedStream, Context, DebugSink (+17 more)

### Community 13 - "HistorySource"
Cohesion: 0.09
Nodes (29): countMessageContentTokens(), Context, Conversation, DebugSink, HistorySource, HistoryStore, Message, Model (+21 more)

### Community 14 - "source go"
Cohesion: 0.10
Nodes (23): Context, DebugSink, Span, Context, DebugSink, Part, Renderer, RenderNode (+15 more)

### Community 15 - "Builder"
Cohesion: 0.13
Nodes (16): ContextSource, Definition, Context, Builder, Conversation, DebugSink, Part, PromptInput (+8 more)

### Community 16 - "Run()"
Cohesion: 0.13
Nodes (27): AttemptErrorEvent(), AttemptStartEvent(), DoneEvent(), ErrorEvent(), Context, Iteration, Token, IterationDoneEvent() (+19 more)

### Community 17 - "loopRunState"
Cohesion: 0.08
Nodes (16): iterationObserver, iterationObserver, Context, Iteration, Loop, Span, Token, newLoopRunState() (+8 more)

### Community 18 - "capability go"
Cohesion: 0.12
Nodes (23): additionalPropertiesValue(), RawMessage, isSupportedToolParameterType(), NewToolDefinition(), T, TestNewToolDefinitionRejectsInvalidSchema(), TestNewToolDefinitionValidatesSchema(), TestResponseFormatValidation() (+15 more)

### Community 19 - "RAG()"
Cohesion: 0.11
Nodes (19): Document, fakeRAGStore, Context, DebugSink, Part, PromptView, Source, SourceBudget (+11 more)

### Community 20 - "agent go"
Cohesion: 0.14
Nodes (19): Context, DebugSink, Loop, Model, PromptBuilder, PromptInput, Tokenizer, Tool (+11 more)

### Community 21 - "content go"
Cohesion: 0.12
Nodes (8): Context, RenderNode, NewToolCallContent(), NewToolResultErrContent(), TextContent, ToolCallContent, ToolResultContent, ToolResultErrContent

### Community 22 - "source test go"
Cohesion: 0.17
Nodes (16): Context, DebugEvent, Mutex, T, ToolCall, ToolParameters, ToolResponse, captureSink (+8 more)

### Community 23 - "model go"
Cohesion: 0.14
Nodes (18): RawMessage, ResponseFormat, Tokenizer, ToolDefinition, chatResponseJSONSchema, chatToolFunctionRequest, chatCompletionResponse, chatCompletionStreamResponse (+10 more)

### Community 24 - "prompt part go"
Cohesion: 0.17
Nodes (11): MessagePart, Part, Content, Context, RenderNode, Role, Tokenizer, NewSystemPart() (+3 more)

### Community 25 - "NewNamedPart()"
Cohesion: 0.15
Nodes (14): NamedPart, Content, Context, Part, RenderNode, TextPart, Tokenizer, NewJSONPart() (+6 more)

### Community 26 - "part go"
Cohesion: 0.19
Nodes (10): Context, Message, RenderNode, Role, Tokenizer, Content, MapMessageToContent(), roleRenderType() (+2 more)

### Community 27 - "NewTextPart()"
Cohesion: 0.19
Nodes (13): emptyConversation, messageConversation, Message, Part, T, TestBuildPromptOrdersInputContextBeforeUserAndConversation(), TestBuildPromptRendersStructuredConversationContent(), TestNewPromptBuilderFromDefinition() (+5 more)

### Community 28 - "renderObserver"
Cohesion: 0.29
Nodes (10): addRendererPreview(), Context, DebugSink, Part, RenderNode, newRenderObserver(), rendererNodeStructure(), rendererSourceName() (+2 more)

### Community 29 - "message go"
Cohesion: 0.21
Nodes (11): combinedMessageContent(), Content, Context, TurnTokenStore, DebugSink, Message, Tokenizer, IsValidRole() (+3 more)

### Community 30 - "MockModel"
Cohesion: 0.18
Nodes (8): MockModel, MockModelResponse, MockTokenizer, AIRequest, AIResponse, Context, Token, Tokenizer

### Community 31 - "MockSessionStore"
Cohesion: 0.28
Nodes (10): AddMessageCall, AddMessagesCall, CreateSessionCall, GetMessagesCall, GetSessionCall, MockSessionStore, UpdateMessageTokensCall, Context (+2 more)

### Community 32 - "ModelRepository"
Cohesion: 0.23
Nodes (9): Context, DebugSink, Model, Provider, NewModelRepository(), T, TestModelRepository(), TestModelRepositoryRejectsNilProvider() (+1 more)

### Community 33 - "NewTextContent()"
Cohesion: 0.29
Nodes (15): NewTextContent(), NewToolResultContent(), T, TestHistoryContentImplementsContent(), TestHistoryPartRenderEmpty(), TestHistoryPartRendersSimpleContent(), TestHistoryPartRendersStructuredContent(), TestHistoryPartRendersSummary() (+7 more)

### Community 34 - "iteration go"
Cohesion: 0.19
Nodes (10): IterationPart, AIResponse, Iteration, Message, Token, ToolCall, ToolResponse, IterationInformation (+2 more)

### Community 35 - "DetectToolCallsInStream()"
Cohesion: 0.20
Nodes (12): DetectToolCallsInStream(), GenerateToolCallID(), Context, DebugSink, RawMessage, Token, isWS(), joinTokenData() (+4 more)

### Community 36 - "Context"
Cohesion: 0.16
Nodes (7): debugEventSink, debugTestTokenizer, failingPart, Context, DebugEvent, RenderNode, Tokenizer

### Community 37 - "message test go"
Cohesion: 0.22
Nodes (13): Context, turnTokenStore, T, TestMessageTokensHandlesNilContent(), TestMessageTokensRecountsNegativeCachedValue(), TestTurnTokenizeCountsCombinedMessagesWithoutUpdatingMessages(), TestTurnTokenizeEmitsDebugEventWhenSavingTokensFails(), TestTurnTokenizeHandlesNilMessageContent() (+5 more)

### Community 38 - "Context"
Cohesion: 0.19
Nodes (7): failingRenderPart, historyPartAdapter, Context, Message, RenderNode, Tokenizer, renderTestPart

### Community 39 - "T"
Cohesion: 0.26
Nodes (13): T, TestModelGenerate(), TestModelGenerateMapsMultipleResponseToolCalls(), TestModelGenerateMapsRequestCapabilities(), TestModelGenerateNoChoices(), TestModelGenerateStream(), TestModelGenerateStreamDetectsTextEncodedToolCall(), TestModelGenerateStreamToolCall() (+5 more)

### Community 40 - "renderer test go"
Cohesion: 0.37
Nodes (12): NewMessagePart(), T, TestRendererDebugStructureOmitsContentForNonSensitiveSink(), TestRenderersEmitDetailedTruncatedDebugEvents(), TestRenderersNotifyRenderResultCallback(), TestRenderersNotifyRenderResultCallbackForEmptyPrompt(), TestRenderToolSignatures(), TestSimpleRendererPreservesGenericNodeStructure() (+4 more)

### Community 41 - "scriptedWorkflowModel"
Cohesion: 0.22
Nodes (7): AIRequest, AIResponse, Context, Mutex, Token, Tokenizer, scriptedWorkflowModel

### Community 42 - "buildChatCompletionRequest()"
Cohesion: 0.20
Nodes (10): AIRequest, AIResponse, ToolChoice, chatMessageRequest, chatResponseFormat, chatToolRequest, chatCompletionRequest, buildChatCompletionRequest() (+2 more)

### Community 43 - "response test go"
Cohesion: 0.31
Nodes (10): expectedWrapToken, collectTokens(), T, Token, normalizeExpectedTokens(), normalizeJSON(), normalizeTokens(), TestAIResponseAppendTokenSeparatesThoughtsAndToolCalls() (+2 more)

### Community 44 - "Provider"
Cohesion: 0.27
Nodes (6): Client, DebugSink, Model, Provider, isKnownModel(), New()

### Community 45 - "GenerateStream()"
Cohesion: 0.29
Nodes (6): Builder, Token, ToolCall, mistralToolCallAccumulator, mistralToolCallState, mistralToolCallState

### Community 46 - "Provider"
Cohesion: 0.31
Nodes (5): Provider, isKnownModel(), New(), DebugSink, Model

### Community 47 - "CountTokens()"
Cohesion: 0.31
Nodes (5): Context, DebugSink, Provider, intPtr(), Tokenizer

### Community 48 - "Token"
Cohesion: 0.36
Nodes (5): AIResponse, RawMessage, ToolCall, Token, TokenType

### Community 49 - "NewContentFromType()"
Cohesion: 0.57
Nodes (6): NewContentFromType(), T, TestContentMarshalRoundTrip(), TestNewContentFromType(), TestNewContentFromTypeRejectsInvalidJSON(), TestNewContentFromTypeRejectsUnknownType()

### Community 51 - "AIRequest"
Cohesion: 0.33
Nodes (5): AIRequest, ReasoningConfig, ResponseFormat, ToolChoice, ToolDefinition

### Community 52 - "LoadPromptFromFile()"
Cohesion: 0.33
Nodes (4): Context, Builder, Part, LoadPromptFromFile()

### Community 53 - "historyStore"
Cohesion: 0.60
Nodes (3): Context, historyStore, HistoryState

### Community 54 - "Agent Framework Thumbnail"
Cohesion: 0.60
Nodes (5): Agent Framework Banner, Dashboard Tablet Interface, GAI Agent Framework Thumbnail, GAI Title Typography, Go Mascot Agent

### Community 55 - "Prompt"
Cohesion: 0.60
Nodes (3): Prompt, Context, DebugSink

### Community 57 - "provider test go"
Cohesion: 0.60
Nodes (4): TestProviderModelAndListModels(), TestProviderModelValidation(), TestProviderValidate(), T

### Community 58 - "provider test go"
Cohesion: 0.60
Nodes (4): T, TestProviderModelAndListModels(), TestProviderModelValidation(), TestProviderValidate()

### Community 59 - "Neon Sci Fi Workspace"
Cohesion: 0.50
Nodes (4): Blue Purple Glow Palette, Neon Sci Fi Workspace, Mirrored Robotic Arms, Symmetric Centered Layout

### Community 61 - "History Source Tests"
Cohesion: 0.67
Nodes (4): History Budget Helper, History Source Tests, Mock Session Store, Session Store Interface

### Community 62 - "CombinedPrompt"
Cohesion: 0.67
Nodes (3): CombinedPrompt, CombinedPromptWithDebug, Prompt

### Community 68 - "RAG Source"
Cohesion: 0.67
Nodes (3): RAG Document, RAG Overflow Summarization, RAG Source

### Community 69 - "RAG Store Interface"
Cohesion: 0.67
Nodes (3): RAG Source Document Budgeting Tests, Fake RAG Store, RAG Store Interface

## Knowledge Gaps
- **234 isolated node(s):** `Context`, `DebugSink`, `Source`, `PromptView`, `SourceBudget` (+229 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StartOperationSpan()` connect `StartOperationSpan()` to `DetectToolCallsInStream()`, `historyObserver`, `model go`, `searchObserver`, `buildChatCompletionRequest()`, `Context`, `GenerateStream()`, `source go`, `CountTokens()`, `loopRunState`?**
  _High betweenness centrality (0.286) - this node is a cross-community bridge._
- **Why does `NewTextContent()` connect `NewTextContent()` to `textRunInput()`, `TestPartTokensRecountsNegativeCachedValue()`, `loop test go`, `iteration go`, `historyObserver`, `message test go`, `renderer test go`, `searchObserver`, `summary go`, `content go`, `NewNamedPart()`, `NewTextPart()`?**
  _High betweenness centrality (0.267) - this node is a cross-community bridge._
- **Why does `newHistoryBuildObserver()` connect `historyObserver` to `StartOperationSpan()`, `HistorySource`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Are the 36 inferred relationships involving `NewTextContent()` (e.g. with `textRunInput()` and `TestAgentWorkflowEmitsLifecycleEventsAndSpans()`) actually correct?**
  _`NewTextContent()` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `StartOperationSpan()` (e.g. with `newRunCreationObserver()` and `newWorkflowObserver()`) actually correct?**
  _`StartOperationSpan()` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `EndSpan()` (e.g. with `.Finish()` and `.Finished()`) actually correct?**
  _`EndSpan()` has 20 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Context`, `DebugSink`, `Source` to the rest of the system?**
  _234 weakly-connected nodes found - possible documentation gaps or missing edges._