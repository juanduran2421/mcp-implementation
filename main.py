import argparse
import asyncio
import os
import sys
from pathlib import Path

import anthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

from mcp.shared.context import RequestContext
from mcp.types import (
    INTERNAL_ERROR,
    CreateMessageRequestParams,
    CreateMessageResult,
    ErrorData,
    TextContent,
)

from mcp_client import MCPClient

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a small document store. "
    "Use the available tools to read or edit documents when the user asks about them. "
    "Documents the user mentions with @ are included in their message."
)


async def run_mcp_tool(client: MCPClient, name: str, args: dict) -> tuple[str, bool]:
    """Run one tool on the MCP server and return (text, is_error)."""
    print(f"  [tool] {name}({args})")
    try:
        result = await client.call_tool(name, args)
        text = "\n".join(c.text for c in result.content if c.type == "text")
        return text, bool(result.isError)
    except Exception as e:
        return f"Error: {e}", True


class ClaudeChat:
    name = "Claude"

    @staticmethod
    def has_credentials() -> bool:
        return any(
            os.environ.get(var)
            for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
        ) or Path("~/.config/anthropic").expanduser().exists()

    def __init__(self, mcp: MCPClient, mcp_tools):
        self.mcp = mcp
        self.llm = anthropic.AsyncAnthropic()
        self.model = CLAUDE_MODEL
        self.messages: list = []
        self.tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in mcp_tools
        ]

    async def send(self, user_message: str) -> str:
        """Send a message, running MCP tools until Claude produces a final answer."""
        self.messages.append({"role": "user", "content": user_message})
        while True:
            response = await self.llm.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.messages,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            # Append the full content so thinking and tool_use blocks are preserved.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                reason = response.stop_details.category if response.stop_details else None
                return f"(Claude declined this request, category: {reason})"

            if response.stop_reason != "tool_use":
                return "\n".join(b.text for b in response.content if b.type == "text")

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    text, is_error = await run_mcp_tool(self.mcp, block.name, block.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": text,
                            "is_error": is_error,
                        }
                    )
            self.messages.append({"role": "user", "content": results})

    async def complete(self, messages: list[tuple[str, str]], system: str | None, max_tokens: int) -> str:
        """One-off completion for MCP sampling: no tools, no chat history."""
        response = await self.llm.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or anthropic.omit,
            messages=[{"role": role, "content": text} for role, text in messages],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined the sampling request")
        return "\n".join(b.text for b in response.content if b.type == "text")

    def rollback(self, length: int):
        del self.messages[length:]

    def history_length(self) -> int:
        return len(self.messages)


class GeminiChat:
    name = "Gemini"

    @staticmethod
    def has_credentials() -> bool:
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def __init__(self, mcp: MCPClient, mcp_tools):
        self.mcp = mcp
        self.llm = genai.Client()
        self.model = GEMINI_MODEL
        self.contents: list[types.Content] = []
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description or "",
                            parameters_json_schema=tool.inputSchema,
                        )
                        for tool in mcp_tools
                    ]
                )
            ],
            # We run the MCP tools ourselves so each call can be shown.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def send(self, user_message: str) -> str:
        """Send a message, running MCP tools until Gemini produces a final answer."""
        self.contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )
        while True:
            response = await self.llm.aio.models.generate_content(
                model=self.model, contents=self.contents, config=self.config
            )
            if not response.candidates or response.candidates[0].content is None:
                feedback = response.prompt_feedback
                reason = feedback.block_reason if feedback else "no response"
                return f"(Gemini returned no answer: {reason})"

            # Append the model's content as-is so thought signatures are preserved.
            self.contents.append(response.candidates[0].content)

            if not response.function_calls:
                return response.text or ""

            parts = []
            for call in response.function_calls:
                text, is_error = await run_mcp_tool(self.mcp, call.name, call.args or {})
                parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"error": text} if is_error else {"result": text},
                    )
                )
            self.contents.append(types.Content(role="user", parts=parts))

    async def complete(self, messages: list[tuple[str, str]], system: str | None, max_tokens: int) -> str:
        """One-off completion for MCP sampling: no tools, no chat history."""
        response = await self.llm.aio.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[types.Part.from_text(text=text)],
                )
                for role, text in messages
            ],
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=max_tokens
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no text for the sampling request")
        return response.text

    def rollback(self, length: int):
        del self.contents[length:]

    def history_length(self) -> int:
        return len(self.contents)


PROVIDERS = {"claude": ClaudeChat, "gemini": GeminiChat}


class SamplingHandler:
    """Answers the server's sampling requests with the chat's LLM.

    The MCP client needs this callback when it connects, but the chat is created
    afterwards (it needs the server's tools), so `chat` is set later.
    """

    def __init__(self):
        self.chat = None

    async def __call__(
        self, context: RequestContext, params: CreateMessageRequestParams
    ) -> CreateMessageResult | ErrorData:
        if self.chat is None:
            return ErrorData(code=INTERNAL_ERROR, message="No LLM is ready for sampling yet")

        messages = []
        for message in params.messages:
            blocks = message.content if isinstance(message.content, list) else [message.content]
            text = "\n".join(b.text for b in blocks if isinstance(b, TextContent))
            messages.append((message.role, text))

        print(f"  [sampling] server asked {self.chat.name} for a completion")
        try:
            text = await self.chat.complete(messages, params.systemPrompt, params.maxTokens)
        except Exception as e:
            return ErrorData(code=INTERNAL_ERROR, message=f"Sampling failed: {e}")

        print(f"  [sampling] {self.chat.name} returned: {text}")
        return CreateMessageResult(
            role="assistant",
            model=self.chat.model,
            content=TextContent(type="text", text=text),
        )


def choose_provider(requested: str | None):
    if requested:
        provider = PROVIDERS[requested]
        if not provider.has_credentials():
            key = "ANTHROPIC_API_KEY" if provider is ClaudeChat else "GEMINI_API_KEY"
            sys.exit(f"No {provider.name} credentials found. Set {key} first.")
        return provider

    for provider in PROVIDERS.values():
        if provider.has_credentials():
            return provider

    sys.exit(
        "No credentials found. Set one of these first:\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...   # Claude\n"
        "  export GEMINI_API_KEY=...             # Gemini"
    )


class ChatCompleter(Completer):
    """Suggest /prompt commands and their arguments, and document ids after @."""

    def __init__(self, doc_ids: list[str], prompts: dict):
        self.doc_ids = doc_ids
        self.prompts = prompts

    def _complete_doc_ids(self, prefix: str):
        for doc_id in self.doc_ids:
            if doc_id.startswith(prefix):
                yield Completion(doc_id, start_position=-len(prefix))

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)

        if text.startswith("/"):
            name, *values = text[1:].split(" ")
            if not values:
                # Still typing the command name: suggest the server's prompts.
                for prompt in self.prompts.values():
                    if prompt.name.startswith(name):
                        yield Completion(
                            prompt.name,
                            start_position=-len(name),
                            display_meta=prompt.description or "",
                        )
            elif name in self.prompts and len(values) <= len(
                self.prompts[name].arguments or []
            ):
                # Typing an argument: this server's prompt arguments are doc ids.
                yield from self._complete_doc_ids(word)
            return

        if word.startswith("@"):
            yield from self._complete_doc_ids(word[1:])


def prompt_usage(prompt) -> str:
    args = " ".join(f"<{arg.name}>" for arg in prompt.arguments or [])
    return f"/{prompt.name} {args}".strip()


async def build_prompt_message(client: MCPClient, prompts: dict, query: str) -> str:
    """Turn '/name arg1 arg2' into the text of the server's prompt."""
    name, *values = query[1:].split()
    prompt = prompts.get(name)
    if prompt is None:
        available = ", ".join(f"/{p}" for p in prompts) or "none"
        raise ValueError(f"Unknown command /{name}. Available: {available}")

    arg_defs = prompt.arguments or []
    required = [arg for arg in arg_defs if arg.required]
    if len(values) < len(required) or len(values) > len(arg_defs):
        raise ValueError(f"Usage: {prompt_usage(prompt)}")

    args = {arg.name: value for arg, value in zip(arg_defs, values)}
    messages = await client.get_prompt(name, args)
    return "\n\n".join(
        m.content.text for m in messages if m.role == "user" and m.content.type == "text"
    )


async def build_user_message(client: MCPClient, query: str, doc_ids: list[str]) -> str:
    """Attach the contents of any @mentioned documents to the user's query."""
    mentioned = {
        word.lstrip("@").rstrip(".,?!")
        for word in query.split()
        if word.startswith("@")
    }
    context = ""
    for doc_id in sorted(mentioned & set(doc_ids)):
        content = await client.read_resource(f"docs://documents/{doc_id}")
        context += f'<document id="{doc_id}">\n{content}\n</document>\n'
    return f"{context}\n{query}" if context else query


async def main():
    parser = argparse.ArgumentParser(description="Chat with your documents over MCP.")
    parser.add_argument(
        "provider",
        nargs="?",
        choices=PROVIDERS,
        default=os.environ.get("LLM_PROVIDER"),
        help="LLM to use (default: LLM_PROVIDER, or whichever API key is set)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MCP_SERVER_URL"),
        help="URL of a running MCP server, e.g. http://127.0.0.1:8000/mcp "
        "(default: MCP_SERVER_URL, or start mcp_server.py over stdio)",
    )
    args = parser.parse_args()
    provider = choose_provider(args.provider)

    sampler = SamplingHandler()
    if args.url:
        mcp_client = MCPClient(url=args.url, sampling_callback=sampler)
    else:
        mcp_client = MCPClient(
            command=sys.executable, args=["mcp_server.py"], sampling_callback=sampler
        )

    async with mcp_client as client:
        doc_ids = await client.read_resource("docs://documents")
        prompts = {p.name: p for p in await client.list_prompts()}
        chat = provider(client, await client.list_tools())
        sampler.chat = chat

        print(f"Connected to DocumentMCP using {chat.name} ({chat.model}).")
        print(f"Documents: {', '.join(doc_ids)}")
        if prompts:
            print(f"Commands: {', '.join(prompt_usage(p) for p in prompts.values())}")
        print("Mention a document with @name (e.g. @plan.md). Type 'exit' to quit.\n")

        prompt = PromptSession(
            completer=ChatCompleter(doc_ids, prompts), complete_while_typing=True
        )
        while True:
            try:
                query = (await prompt.prompt_async("> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue

            if query.startswith("/"):
                try:
                    message = await build_prompt_message(client, prompts, query)
                except ValueError as e:
                    print(e)
                    continue
            else:
                message = await build_user_message(client, query, doc_ids)

            checkpoint = chat.history_length()
            try:
                answer = await chat.send(message)
                print(f"\n{chat.name}: {answer}\n")
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                print(f"Authentication failed: {e.message}")
                break
            except anthropic.APIStatusError as e:
                print(f"API error ({e.status_code}): {e.message}")
                chat.rollback(checkpoint)
            except anthropic.APIConnectionError:
                print("Network error: could not reach the Claude API.")
                chat.rollback(checkpoint)
            except genai_errors.APIError as e:
                print(f"API error ({e.code}): {e.message}")
                if e.code in (401, 403) or "API key" in (e.message or ""):
                    break
                chat.rollback(checkpoint)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ConnectionError as e:
        sys.exit(str(e))
