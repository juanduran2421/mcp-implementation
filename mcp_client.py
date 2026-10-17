import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.session import SamplingFnT
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool
from pydantic import AnyUrl


class MCPClient:
    """Connects to an MCP server, either by URL or by starting it over stdio."""

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict | None = None,
        url: str | None = None,
        sampling_callback: SamplingFnT | None = None,
    ):
        if not url and not command:
            raise ValueError("Pass either a server url or a command to start.")
        self._command = command
        self._args = args or []
        self._env = env
        self._url = url
        self._sampling_callback = sampling_callback
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def _check_reachable(self):
        """Fail fast with a clear error if nothing is listening at the URL."""
        url = urlparse(self._url)
        port = url.port or (443 if url.scheme == "https" else 80)
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(url.hostname, port), timeout=5
            )
            writer.close()
            await writer.wait_closed()
        except (OSError, asyncio.TimeoutError):
            raise ConnectionError(
                f"Could not reach the MCP server at {self._url}. "
                "Is it running? Start it with: uv run mcp_server.py --http"
            ) from None

    async def connect(self):
        if self._url:
            await self._check_reachable()
            read, write, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(self._url)
            )
        else:
            server_params = StdioServerParameters(
                command=self._command, args=self._args, env=self._env
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write, sampling_callback=self._sampling_callback)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError(
                "Client session not initialized. Call connect() first."
            )
        return self._session

    async def list_prompts(self) -> list[types.Prompt]:
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name, args: dict[str, str]):
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages
    async def list_tools(self) -> list[Tool]:
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(
        self, tool_name: str, tool_input: dict
    ) -> CallToolResult | None:
        return await self.session().call_tool(tool_name, tool_input)

   
    async def read_resource(self, uri: str) -> Any:
        result = await self.session().read_resource(AnyUrl(uri))
        resource = result.contents[0]

        if isinstance(resource, types.TextResourceContents):
            if resource.mimeType == "application/json":
                return json.loads(resource.text)
            return resource.text

        # BlobResourceContents has no .text, only base64-encoded .blob
        return resource.blob

    async def cleanup(self):
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


async def main():
    async with MCPClient(
        command=sys.executable, args=["mcp_server.py"]
    ) as client:
        print("tools:", [t.name for t in await client.list_tools()])

        result = await client.call_tool("read_doc_contents", {"doc_id": "plan.md"})
        print("read_doc_contents:", result.content[0].text)

        print("documents:", await client.read_resource("docs://documents"))
        print(
            "deposition.md:",
            await client.read_resource("docs://documents/deposition.md"),
        )


if __name__ == "__main__":
    asyncio.run(main())
