//
//  LLMChatOpenAI+StreamConversation.swift
//  LLMChatOpenAI
//
//  Created by Ruslan on 17/03/2026.
//

import Foundation

// MARK: - Public API
public extension LLMChatOpenAI {
    /// The result of a streamed conversation turn that may include tool calls.
    struct StreamConversationResult: Sendable {
        /// The accumulated assistant text from the final response.
        public let text: String

        /// The complete message history including the assistant's response,
        /// ready to be used as input for the next turn.
        public let messages: [ChatMessage]

        /// Token usage from the final streaming phase.
        public let usage: ChatCompletionChunk.Usage?
    }

    /// Streams a full conversation turn, handling tool calls transparently.
    ///
    /// Phase 1: get initial response (may include tool calls and/or text).
    /// Phase 2: if tools were called, execute them and stream the final answer.
    ///
    /// - Parameters:
    ///   - model: The model to use for completion.
    ///   - messages: An array of ``ChatMessage`` objects that represent the conversation history.
    ///   - options: Optional ``ChatOptions`` that customize the completion request.
    ///   - toolHandler: A closure that executes a tool call given its name and JSON arguments, returning a JSON result string. If `nil`, tool calls are ignored.
    ///   - onText: A closure called with the accumulated assistant text as it streams in.
    ///
    /// - Returns: A ``StreamConversationResult`` containing the final text, updated messages, and usage.
    func streamConversation(
        model: String,
        messages: [ChatMessage],
        options: ChatOptions? = nil,
        toolHandler: (@Sendable (String, String) -> String)? = nil,
        onText: @escaping @Sendable (String) async -> Void
    ) async throws -> StreamConversationResult {
        let phase1 = try await _firstPhase(model: model, messages: messages, options: options, onText: onText)

        // No tool calls or no handler → done
        guard let toolCalls = phase1.toolCalls, let toolHandler else {
            return StreamConversationResult(
                text: phase1.text,
                messages: phase1.updatedMessages,
                usage: phase1.usage
            )
        }

        // Execute tools, stream final answer
        let toolMessages = Self._executeTools(toolCalls, handler: toolHandler)
        let phase2Messages = phase1.updatedMessages + toolMessages

        let phase2 = try await _secondPhase(model: model, messages: phase2Messages, options: options, onText: onText)

        var finalMessages = phase2Messages
        finalMessages.append(ChatMessage(role: .assistant, content: phase2.text))

        return StreamConversationResult(
            text: phase2.text,
            messages: finalMessages,
            usage: phase2.usage
        )
    }
}

// MARK: - Private Implementation
private extension LLMChatOpenAI {
    struct _FirstPhaseResult {
        let text: String
        let toolCalls: [ChatMessage.ToolCall]?
        let updatedMessages: [ChatMessage]
        let usage: ChatCompletionChunk.Usage?
    }

    struct _SecondPhaseResult {
        let text: String
        let usage: ChatCompletionChunk.Usage?
    }

    /// Streams Phase 1. Accumulates both text and tool call chunks.
    /// Tool call arguments arrive in fragments and are merged by call id.
    func _firstPhase(
        model: String,
        messages: [ChatMessage],
        options: ChatOptions?,
        onText: @escaping @Sendable (String) async -> Void
    ) async throws -> _FirstPhaseResult {
        var textBuffer = ""
        var toolParts: [String: (name: String, args: String)] = [:]
        var lastSeenToolId: String?
        var usage: ChatCompletionChunk.Usage?

        for try await chunk in stream(model: model, messages: messages, options: options) {
            let choice = chunk.choices.first

            // Accumulate text and forward to caller
            if let delta = choice?.delta.content, !delta.isEmpty {
                textBuffer += delta
                await onText(textBuffer)
            }

            // Merge streaming tool call fragments
            if let toolCalls = choice?.delta.toolCalls {
                for tc in toolCalls {
                    if let id = tc.id { lastSeenToolId = id }
                    guard let id = tc.id ?? lastSeenToolId else { continue }
                    let name = tc.function?.name ?? ""
                    let args = tc.function?.arguments ?? ""

                    if var existing = toolParts[id] {
                        if !name.isEmpty { existing.name = name }
                        if !args.isEmpty { existing.args += args }
                        toolParts[id] = existing
                    } else {
                        toolParts[id] = (name, args)
                    }
                }
            }

            if let u = chunk.usage { usage = u }
        }

        // Drop incomplete tool entries (no name = malformed chunk)
        let finalized = toolParts
            .filter { !$0.value.name.isEmpty }
            .sorted { $0.key < $1.key }

        guard !finalized.isEmpty else {
            return _FirstPhaseResult(text: textBuffer, toolCalls: nil, updatedMessages: messages, usage: usage)
        }

        let toolCalls = finalized.map { (id, payload) in
            ChatMessage.ToolCall(id: id, function: .init(name: payload.name, arguments: payload.args))
        }

        var updated = messages
        updated.append(ChatMessage(role: .assistant, content: textBuffer, toolCalls: toolCalls))

        return _FirstPhaseResult(text: textBuffer, toolCalls: toolCalls, updatedMessages: updated, usage: usage)
    }

    /// Streams Phase 2: model sees tool results and produces the final user-facing answer.
    func _secondPhase(
        model: String,
        messages: [ChatMessage],
        options: ChatOptions?,
        onText: @escaping @Sendable (String) async -> Void
    ) async throws -> _SecondPhaseResult {
        var buffer = ""
        var usage: ChatCompletionChunk.Usage?

        for try await chunk in stream(model: model, messages: messages, options: options) {
            let choice = chunk.choices.first

            if let delta = choice?.delta.content, !delta.isEmpty {
                buffer += delta
                await onText(buffer)
            }

            if let u = chunk.usage { usage = u }
        }

        return _SecondPhaseResult(text: buffer, usage: usage)
    }

    /// Executes all tool calls synchronously and returns tool result messages.
    static func _executeTools(_ calls: [ChatMessage.ToolCall], handler: @Sendable (String, String) -> String) -> [ChatMessage] {
        calls.map { call in
            let result = handler(call.function.name, call.function.arguments)
            return ChatMessage(
                role: .tool,
                content: result,
                name: call.function.name,
                toolCallId: call.id
            )
        }
    }
}
