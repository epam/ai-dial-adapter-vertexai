import json

from aidial_sdk.chat_completion import Message
from google.genai.types import Content

from aidial_adapter_vertexai.chat.errors import UserError
from aidial_adapter_vertexai.chat.gemini.prompt.gemini_2 import Gemini_2_Prompt
from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig
from aidial_adapter_vertexai.chat.tools import ToolsConfig, ToolsMode

_NOTE = "<system-reminder>The task list changed.</system-reminder>"


def _user(text: str) -> Message:
    return Message.model_validate({"role": "user", "content": text})


def _tool_calls(*call_ids: str) -> Message:
    return Message.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": json.dumps({"cmd": "ls"}),
                    },
                }
                for call_id in call_ids
            ],
        }
    )


def _tool_result(call_id: str, content: str) -> Message:
    return Message.model_validate(
        {"role": "tool", "tool_call_id": call_id, "content": content}
    )


async def _parse(messages: list[Message]) -> Gemini_2_Prompt:
    call_ids = [
        call.id for message in messages for call in message.tool_calls or []
    ]
    prompt = await Gemini_2_Prompt.parse(
        file_storage=None,
        tools=ToolsConfig(
            tools=[],
            tool_choice="auto",
            tool_ids=dict.fromkeys(call_ids, "shell"),
            tools_mode=ToolsMode.TOOLS,
        ),
        static_tools=StaticToolsConfig.noop(),
        messages=messages,
    )
    assert not isinstance(prompt, UserError)
    return prompt


def _responses(content: Content) -> list[dict | None]:
    return [
        part.function_response.response
        for part in content.parts or []
        if part.function_response
    ]


def _texts(content: Content) -> list[str]:
    return [part.text for part in content.parts or [] if part.text is not None]


async def test_text_after_tool_result_is_folded_into_the_response():
    prompt = await _parse(
        [
            _user("List the files"),
            _tool_calls("call_1"),
            _tool_result("call_1", "a.txt"),
            _user(_NOTE),
        ]
    )

    contents = prompt.messages.raw_list
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert _responses(contents[-1]) == [{"output": f"a.txt\n\n{_NOTE}"}]
    assert _texts(contents[-1]) == []


async def test_folded_turn_keeps_the_indices_of_both_messages():
    prompt = await _parse(
        [
            _user("List the files"),
            _tool_calls("call_1"),
            _tool_result("call_1", "a.txt"),
            _user(_NOTE),
        ]
    )

    assert [indices for _, indices in prompt.messages.lst][-1] == {2, 3}


async def test_json_tool_result_is_folded_as_text():
    prompt = await _parse(
        [
            _user("List the files"),
            _tool_calls("call_1"),
            _tool_result("call_1", json.dumps({"files": ["a.txt"]})),
            _user(_NOTE),
        ]
    )

    assert _responses(prompt.messages.raw_list[-1]) == [
        {"output": f'{{"files": ["a.txt"]}}\n\n{_NOTE}'}
    ]


async def test_text_goes_to_the_last_of_parallel_tool_results():
    prompt = await _parse(
        [
            _user("List the files"),
            _tool_calls("call_1", "call_2"),
            _tool_result("call_1", "a.txt"),
            _tool_result("call_2", "b.txt"),
            _user(_NOTE),
        ]
    )

    assert _responses(prompt.messages.raw_list[-1]) == [
        {"output": "a.txt"},
        {"output": f"b.txt\n\n{_NOTE}"},
    ]


async def test_tool_result_without_text_is_unchanged():
    prompt = await _parse(
        [
            _user("List the files"),
            _tool_calls("call_1"),
            _tool_result("call_1", "a.txt"),
        ]
    )

    assert _responses(prompt.messages.raw_list[-1]) == [{"output": "a.txt"}]


async def test_user_turn_without_tool_results_is_unchanged():
    prompt = await _parse([_user("first"), _user("second")])

    assert _texts(prompt.messages.raw_list[-1]) == ["first", "second"]
