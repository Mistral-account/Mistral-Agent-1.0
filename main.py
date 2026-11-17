import base64
import io
import math
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import Field

from mistralai import agents

DATASETS: dict[str, pd.DataFrame] = {}

MAX_ROWS_IN_RESULT = 50


def _get(dataset_name: str) -> pd.DataFrame:
    df = DATASETS.get(dataset_name)
    if df is None:
        raise agents.ToolError(
            f"No dataset named '{dataset_name}' is loaded. "
            "Download the file with the Google Drive connector and call load_csv first."
        )
    return df


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _rows(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    return [
        {col: _json_safe(val) for col, val in row.items()}
        for row in df.head(limit).to_dict(orient="records")
    ]


@agents.tool
def load_csv(
    dataset_name: Annotated[str, Field(description="Short name to refer to this dataset in later questions, e.g. 'sales'.")],
    content_b64: Annotated[str, Field(description="The file content as base64, exactly as returned by the Google Drive download_file_content tool.")],
) -> dict[str, Any]:
    """Load a CSV file's content into memory as the active dataset for this conversation."""
    try:
        raw = base64.b64decode(content_b64, validate=False)
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as error:
        raise agents.ToolError(
            f"Could not parse the content as a CSV file: {error}. "
            "Make sure you downloaded the raw CSV file content."
        ) from error
    DATASETS[dataset_name] = df
    return {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "columns": [str(col) for col in df.columns],
    }


@agents.tool
def describe_columns(dataset_name: str) -> dict[str, Any]:
    """Describe every column of a loaded dataset: dtype, non-null count, unique count, and example or min/max values."""
    df = _get(dataset_name)
    columns: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "nulls": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            info["min"] = _json_safe(series.min())
            info["max"] = _json_safe(series.max())
            info["mean"] = _json_safe(round(float(series.mean()), 6))
        else:
            values = series.dropna().astype(str).unique()[:5]
            info["examples"] = values.tolist()
        columns[str(col)] = info
    return {"dataset": dataset_name, "rows": int(len(df)), "columns": columns}


@agents.tool
def filter_rows(
    dataset_name: str,
    column: str,
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "is_null", "not_null"],
    value: Annotated[Any, Field(description="Comparison value. For 'in', a list of values. Ignored for is_null/not_null.")] = None,
    limit: Annotated[int, Field(description="Maximum number of matching rows to return.", ge=1, le=200)] = 20,
) -> dict[str, Any]:
    """Filter rows of a loaded dataset with a fixed comparison operator and return the matches plus the total match count."""
    df = _get(dataset_name)
    if column not in df.columns:
        raise agents.ToolError(
            f"Column '{column}' does not exist. Available columns: {[str(c) for c in df.columns]}"
        )
    series = df[column]
    match operator:
        case "eq":
            mask = series == value
        case "ne":
            mask = series != value
        case "gt":
            mask = series > value
        case "gte":
            mask = series >= value
        case "lt":
            mask = series < value
        case "lte":
            mask = series <= value
        case "contains":
            mask = series.astype(str).str.contains(str(value), case=False, na=False)
        case "in":
            mask = series.isin(list(value))
        case "is_null":
            mask = series.isna()
        case "not_null":
            mask = series.notna()
    matched = df[mask]
    return {
        "dataset": dataset_name,
        "filter": {"column": column, "operator": operator, "value": value},
        "total_matches": int(len(matched)),
        "returned_rows": min(len(matched), limit),
        "rows": _rows(matched, limit),
    }


@agents.tool
def groupby_aggregate(
    dataset_name: str,
    group_by: Annotated[list[str], Field(description="Column names to group by.")],
    aggregations: Annotated[list[dict[str, str]], Field(description="One object per aggregation, e.g. {\"column\": \"price\", \"func\": \"mean\"}.")],
    sort_by: Annotated[str | None, Field(description="Aggregation result column to sort by, descending.")] = None,
    limit: Annotated[int, Field(description="Maximum number of groups to return.", ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Group a loaded dataset by one or more columns and compute fixed aggregates (count, nunique, sum, mean, median, min, max, std)."""
    df = _get(dataset_name)
    allowed = {"count", "nunique", "sum", "mean", "median", "min", "max", "std"}
    for agg in aggregations:
        if agg.get("func") not in allowed:
            raise agents.ToolError(
                f"Unknown func '{agg.get('func')}'. Allowed funcs: {sorted(allowed)}"
            )
        if agg.get("column") not in df.columns:
            raise agents.ToolError(
                f"Column '{agg.get('column')}' does not exist. Available columns: {[str(c) for c in df.columns]}"
            )
    missing = [col for col in group_by if col not in df.columns]
    if missing:
        raise agents.ToolError(
            f"Columns {missing} do not exist. Available columns: {[str(c) for c in df.columns]}"
        )
    agg_map: dict[str, list[str]] = {}
    for agg in aggregations:
        agg_map.setdefault(agg["column"], []).append(agg["func"])
    result = df.groupby(group_by, dropna=False).agg(agg_map)
    result.columns = [f"{col}_{func}" for col, func in result.columns]
    result = result.reset_index()
    if sort_by:
        if sort_by not in result.columns:
            raise agents.ToolError(
                f"Cannot sort by '{sort_by}'. Result columns: {[str(c) for c in result.columns]}"
            )
        result = result.sort_values(sort_by, ascending=False)
    return {
        "dataset": dataset_name,
        "group_by": group_by,
        "total_groups": int(len(result)),
        "returned_groups": min(len(result), limit),
        "groups": _rows(result, limit),
    }


agent = agents.Agent(
    model="mistral-large-4",
    name="drive-data-analyst",
    instructions=(
        "You are a conversational data analyst. The user gives you a link to a CSV file in their "
        "Google Drive and asks questions about it in natural language.\n\n"
        "Workflow:\n"
        "1. To get a file, use the Google Drive connector tools. If the user gives a Drive link, "
        "extract the file id from it and call get_file_metadata; otherwise use search_files or "
        "list_recent_files to find it. Confirm with the user when several files match.\n"
        "2. Download the file with download_file_content and pass its base64 content unchanged to "
        "the load_csv tool under a short dataset name. Do not try to read the CSV yourself.\n"
        "3. Answer questions only through the fixed analysis tools: describe_columns for structure, "
        "filter_rows for subsets, groupby_aggregate for aggregates. Never invent numbers: every "
        "figure you state must come from a tool result in this conversation.\n"
        "4. Start a new analysis with describe_columns so you know the real column names, then pick "
        "the tool that fits the question. Combine tools when needed.\n"
        "5. The user asks follow-up questions without repeating context. The dataset stays loaded "
        "under its dataset name: reuse it and everything discussed earlier. If a tool reports the "
        "dataset is no longer loaded, reload it from the same Drive file before answering.\n\n"
        "Answers: state the numbers and the operation that produced them (filters, group-by, "
        "aggregation functions). If a question cannot be answered with the fixed tools, say so "
        "plainly and offer the closest supported analysis. Do not execute arbitrary code and do not "
        "modify the user's Drive files."
    ),
    connectors=[agents.connector("google_drive_mcp")],
    tools=[load_csv, describe_columns, filter_rows, groupby_aggregate],
)


async def main() -> None:
    result = await agent.run(
        "Hello! Which CSV file in my Google Drive would you like to analyze? "
        "Send me a link or a file name."
    )
    print(result.text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
