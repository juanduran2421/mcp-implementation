# Document MCP

A small [Model Context Protocol](https://modelcontextprotocol.io) server that exposes an in-memory document store, plus a terminal chat client that talks to it using Claude or Gemini.

| File | What it is |
|---|---|
| `mcp_server.py` | The MCP server (`DocumentMCP`): tools, resources and prompts over the documents |
| `mcp_client.py` | `MCPClient`, a reusable client that connects over stdio or HTTP |
| `main.py` | Terminal chat app: an LLM (Claude or Gemini) that uses the server's tools |

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- Node.js 22.19+ (only for the MCP Inspector)
- An Anthropic or Gemini API key (only for the chat app)

Install the dependencies:

```bash
uv sync
```

## What the server offers

**Tools** (called by the model)

| Tool | Description |
|---|---|
| `read_doc_contents(doc_id)` | Returns a document's text |
| `edit_document(doc_id, old_str, new_str)` | Replaces text in a document |
| `summarize(text_to_summarize)` | Summarizes text using the **client's** LLM via [sampling](https://modelcontextprotocol.io/docs/concepts/sampling); the server has no model or API key of its own |

**Resources** (read by the client)

| URI | Description |
|---|---|
| `docs://documents` | List of document ids (JSON) |
| `docs://documents/{doc_id}` | One document's text |

**Prompts** (become slash commands in clients)

| Prompt | Description |
|---|---|
| `format(doc_id)` | Rewrites a document in Markdown |
| `summarize(doc_id)` | Reads a document and summarizes it |

Documents live in memory: edits are lost when the server process stops.

## Running the server

The server supports two transports. The mode is chosen by the `--http` flag.

### stdio (default)

```bash
uv run mcp_server.py
```

The server reads JSON-RPC from stdin and writes to stdout, so it is meant to be **started by a client** (the chat app, Claude Code, the Inspector), not run by hand. Each client gets its own server process and its own copy of the documents.

### HTTP

```bash
uv run mcp_server.py --http
```

Starts a standalone server at **`http://127.0.0.1:8000/mcp`** that keeps running until you stop it (Ctrl+C). Any number of clients can connect, and they all share the same documents. It only listens on this machine and has no authentication.

## Testing with the MCP Inspector

```bash
uv run mcp dev mcp_server.py
```

Opens the Inspector in your browser. Click **Connect**, then try the tools, resources and prompts. To inspect a running HTTP server instead, choose the **Streamable HTTP** transport and enter `http://127.0.0.1:8000/mcp`.

If it fails with `Cannot find native binding`, your Node.js is too old: switch to 22.19+ (`nvm use 24`) and clear the npx cache (`rm -rf ~/.npm/_npx`).

## Using the chat app

Set an API key for at least one provider:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Claude: https://platform.claude.com/settings/keys
export GEMINI_API_KEY=...             # Gemini: https://aistudio.google.com/apikey
```

Run it:

```bash
uv run main.py            # uses whichever key is set (Claude first)
uv run main.py claude     # force Claude
uv run main.py gemini     # force Gemini
```

By default it starts its own server over stdio. To use a running HTTP server instead:

```bash
uv run main.py --url http://127.0.0.1:8000/mcp
```

In the chat:

- `@plan.md` attaches a document to your message (Tab completes document ids)
- `/format plan.md`, `/summarize report.pdf` run a server prompt (type `/` for suggestions)
- `Summarize this: ...` makes the model call the `summarize` tool, which triggers sampling
- `exit` or Ctrl+D quits

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | auto | `claude` or `gemini` |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Claude model |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model |
| `MCP_SERVER_URL` | none (stdio) | URL of a running HTTP server |

## Using it from Claude Code

This repo includes a `.mcp.json`, so opening it in Claude Code offers the `documents` server automatically (approve it when asked). To add it by hand instead:

```bash
# stdio: Claude Code starts the server
claude mcp add documents -- uv --directory /path/to/document-mcp run mcp_server.py

# HTTP: connect to a server you started with --http
claude mcp add --transport http documents http://127.0.0.1:8000/mcp
```

The prompts appear as `/documents:format` and `/documents:summarize`.

## Notes

- The project pins `mcp[cli]<2`. In MCP SDK 2.x, `FastMCP` was renamed to `MCPServer` and the code would need migrating.
- If your editor reports `Import "mcp" could not be resolved`, select `.venv/bin/python` as the Python interpreter.
