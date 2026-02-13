import XCTest
@testable import LLMChatOpenAI

final class ChatMessageEncodingTests: XCTestCase {
    func testEncodeAssistantMessageWithToolCalls() throws {
        let message = ChatMessage(
            role: .assistant,
            content: "",
            toolCalls: [
                .init(
                    id: "call_123",
                    function: .init(name: "get_weather", arguments: "{\"city\":\"Berlin\"}")
                )
            ]
        )

        let json = try encodedJSONObject(from: message)

        XCTAssertEqual(json["role"] as? String, "assistant")
        XCTAssertNotNil(json["tool_calls"] as? [[String: Any]])
    }

    func testEncodeToolMessageWithToolCallId() throws {
        let message = ChatMessage(
            role: .tool,
            content: "{\"temperature\":\"18C\"}",
            toolCallId: "call_123"
        )

        let json = try encodedJSONObject(from: message)

        XCTAssertEqual(json["role"] as? String, "tool")
        XCTAssertEqual(json["tool_call_id"] as? String, "call_123")
    }

    private func encodedJSONObject(from message: ChatMessage) throws -> [String: Any] {
        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return object
    }
}
