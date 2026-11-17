# drive-data-analyst

A conversational data-analysis agent built with the [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/getting-started/quickstart).
Give it a link to a CSV file in your Google Drive, then ask questions in natural language.

The agent uses your Google Drive connection (Studio connector `google_drive_mcp`) to find and
download the file, then answers only from four fixed analysis tools — no arbitrary code execution:

| Tool | Purpose |
| --- | --- |
| `load_csv` | Parse the downloaded CSV content into an in-memory dataset |
| `describe_columns` | Dtype, nulls, uniques, examples / min–max–mean per column |
| `filter_rows` | Fixed operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contains`, `in`, `is_null`, `not_null` |
| `groupby_aggregate` | Fixed funcs: `count`, `nunique`, `sum`, `mean`, `median`, `min`, `max`, `std` |

Every number in an answer comes from a tool result. Follow-up questions reuse the loaded dataset
and the conversation context; the agent reloads the file from Drive if the dataset is no longer
in memory.

## Run locally

```bash
uv run --env-file .env agents chat
```

`MISTRAL_API_KEY` must be set in `.env`.

## Deploy (Mistral-managed)

Requires a GitHub repository with this project pushed and the GitHub connector connected in
Mistral Studio, plus an API key whose user has Workflows access:

```bash
git push
uv run --env-file .env agents deploy
```

The agent acts on behalf of the session user through the Drive connector, so the deployment is
hardened by default — keep it that way.

## Chat with the deployed agent

```bash
uv run --env-file .env agents chat drive-data-analyst
```

Or in Vibe Code Web via the link printed by `agents deploy`.
