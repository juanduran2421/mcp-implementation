import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.resources import FunctionResource
from mcp.server.fastmcp.prompts import base
from mcp.types import SamplingMessage, TextContent
from pydantic import Field

mcp = FastMCP("DocumentMCP", log_level="ERROR")

docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures",
    "outlook.pdf": "This document presents the projected future performance of the system",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment",
}

@mcp.resource(
    "docs://documents",
    mime_type="application/json"
)
def list_docs() -> list[str]:
    return list(docs.keys())

@mcp.resource(
    "docs://documents/{doc_id}",
    mime_type="text/plain"
)
def fetch_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    return docs[doc_id]
    
@mcp.tool(
    name="read_doc_contents",
    description="Read the contents of a document and return it as a string.",
)
def read_document(
    doc_id: str = Field(description="Id of the document to read"),
) -> str:
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    return docs[doc_id]


@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing a string in the document's content with a new string.",
)
def edit_document(
    doc_id: str = Field(description="Id of the document that will be edited"),
    old_str: str = Field(description="The text to replace. Must match exactly, including whitespace"),
    new_str: str = Field(description="The new text to insert in place of the old text"),
) -> str:
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    docs[doc_id] = docs[doc_id].replace(old_str, new_str)
    return docs[doc_id]

@mcp.tool(
    name="summarize",
    description="Summarize a piece of text using the client's language model.",
)
async def summarize(
    text_to_summarize: str = Field(description="The text to summarize"),
    ctx: Context = None,
) -> str:
    prompt = f"""
    Please summarize the following text:
    {text_to_summarize}
    """

    # Sampling: the server has no model of its own, so it asks the client to
    # run this prompt through whatever LLM the client is using.
    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt),
            )
        ],
        max_tokens=4000,
        system_prompt="You are a helpful research assistant",
    )

    if isinstance(result.content, TextContent):
        return result.content.text
    raise ValueError("Sampling failed")


@mcp.prompt(
    name="summarize",
    description="Summarizes the contents of a document.",
)
def summarize_document(
    doc_id: str = Field(description="Id of the document to summarize"),
) -> list[base.Message]:
    prompt = f"""
Read the document with id <document_id>{doc_id}</document_id> using the
'read_doc_contents' tool, then write a short summary of it.
"""
    return [base.UserMessage(prompt)]


@mcp.prompt(
    name="format",
    description="Rewrites the contents of the document in Markdown format."
)
def format_document(
    doc_id: str = Field(description="Id of the document to format")
) -> list[base.Message]:
    prompt = f"""
Your goal is to reformat a document to be written with markdown syntax.

The id of the document you need to reformat is:
<document_id>
{doc_id}
</document_id>

Add in headers, bullet points, tables, etc as necessary. Feel free to add in structure.
Use the 'edit_document' tool to edit the document. After the document has been reformatted...
"""

    return [
        base.UserMessage(prompt)
    ]

# Templates aren't returned by resources/list, so clients like Claude Code can't
# offer them in their @ menu. Register each doc as a concrete resource too.
for doc_id in docs:
    mcp.add_resource(
        FunctionResource(
            uri=f"docs://documents/{doc_id}",
            name=doc_id,
            mime_type="text/plain",
            fn=lambda d=doc_id: docs[d],
        )
    )


if __name__ == "__main__":
    # --http runs a standalone server at http://127.0.0.1:8000/mcp that clients
    # connect to; otherwise the client starts this script and talks over stdio.
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
