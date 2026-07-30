import io
import json
import os
import re
import subprocess
import threading
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request
from openai import OpenAI

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
LOG_URL = os.environ["LOG_URL"]
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "pratik-iitm/q5")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")

MODEL = "gpt-5.4-mini"
MAX_HISTORY = 12
MAX_TOOL_ITERATIONS = 9
REQUEST_TIMEOUT = 20

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(REPO_DIR, "run.jsonl")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN, timeout=90)

conversation_history: dict[int, list] = {}
state_lock = threading.Lock()
log_lock = threading.Lock()
dataframe_cache: dict = {}

SYSTEM_PROMPT = (
    "You are a meticulous data analyst answering questions for an automated grading system. "
    "The LAST user message asks a data-analysis question and specifies the EXACT JSON shape to "
    "reply with. Earlier messages (if any) are context only for multi-turn conversations -- "
    "always answer the LAST message.\n\n"
    "You have three tools available. web_search(query) searches Wikipedia only (it is not a "
    "general web search engine, so search-engine operators like site: will not work -- use "
    "plain natural-language queries). fetch_url(url) downloads and reads a specific URL (HTML, "
    "CSV, XLS/XLSX, or PDF), including any dataset the question points at -- for tabular files "
    "it only shows a small preview (shape, columns, first 30 rows, numeric summary), which is "
    "NOT enough to answer a question about a max/min/ranking/filter over the full data. "
    "query_dataset(url, ...) runs a real filter/groupby/aggregate/sort over the FULL CSV or "
    "Excel file and returns just the rows you need -- always use this (not the fetch_url "
    "preview) before answering any question that needs a computed answer (highest/lowest/sum/"
    "average/count/rank/top-N/filtered-by-year-or-category) from a tabular dataset. When you "
    "fetch an HTML page, its outgoing links are listed at the end -- if the page itself lacks "
    "the exact figure, look for a link to the primary source (e.g. mospi.gov.in, a PDF report, "
    "or a 'List of Indian states by ...' Wikipedia article) and fetch_url that next. Use these "
    "tools whenever the question needs real data you are not certain of, or names/links a "
    "public dataset. Do not guess at numbers you could look up or compute, but if after a few "
    "tool calls you cannot find a better source, answer with your best knowledge rather than "
    "looping forever.\n\n"
    "For questions asking which state/entity ranks highest or lowest on some measure, prefer "
    "a single page that already compares all of them (e.g. search '<topic> in India' or 'List "
    "of Indian states by <topic>') over checking states one at a time -- reading several "
    "unrelated single-state pages and guessing from fragments is a common way to get this "
    "wrong.\n\n"
    "Once you have worked out the real answer, reply with ONLY the exact JSON object requested "
    "-- no explanation, no markdown, no code fences, just the raw JSON on its own line. Include "
    "every key the message asked for, in the exact shape requested (including the \"log_url\" "
    "key if one was requested -- put any placeholder string in it, it will be replaced "
    "automatically). Never add keys that were not requested."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search Wikipedia for facts, statistics, or references to public datasets "
                "(e.g. articles on Indian states often cite the underlying MOSPI/SRS report "
                "and its figures). Returns result titles, snippets, and article URLs -- follow "
                "up with fetch_url on a promising URL to read the full article or the primary "
                "source it cites."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Download and read a public URL. Handles HTML pages (returns visible text), "
                "CSV/XLS/XLSX (returns shape, columns, head rows, and a numeric summary), and "
                "PDF (returns extracted text). Use this to read the actual dataset a question "
                "points at."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "sheet": {
                        "type": "string",
                        "description": "Optional sheet name, for multi-sheet Excel files",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_dataset",
            "description": (
                "Load a public CSV or Excel URL as a table and run a real filter / groupby+"
                "aggregate / sort over the FULL data (not just a preview), returning the "
                "resulting rows. Use this for any question needing a computed answer "
                "(max/min/top-N/sum/mean/count/rank, optionally filtered by column values) "
                "from a tabular dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The CSV or Excel URL"},
                    "sheet": {"type": "string", "description": "Optional Excel sheet name"},
                    "filters": {
                        "type": "array",
                        "description": "Rows are kept only if ALL filters match",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "op": {
                                    "type": "string",
                                    "enum": ["==", "!=", ">", "<", ">=", "<=", "contains"],
                                },
                                "value": {},
                            },
                            "required": ["column", "op", "value"],
                        },
                    },
                    "groupby": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional columns to group by before aggregating",
                    },
                    "agg": {
                        "type": "object",
                        "description": (
                            "Maps a column name to an aggregation function when groupby is "
                            "used, e.g. {\"Value\": \"sum\"}. Allowed: sum, mean, max, min, "
                            "count, median, nunique."
                        ),
                    },
                    "sort_by": {"type": "string", "description": "Column to sort the result by"},
                    "ascending": {"type": "boolean", "default": True},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of columns to return",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 20, max 200)",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


def log_event(event: dict):
    event["timestamp"] = time.time()
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")


def _git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_DIR, check=check, capture_output=True, text=True
    )


def push_log_to_github():
    if not GITHUB_TOKEN:
        return
    try:
        _git("add", "run.jsonl")
        diff = _git("diff", "--cached", "--quiet", check=False)
        if diff.returncode == 0:
            return
        _git("commit", "-m", "chore: update run log")
        push_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
        _git("push", push_url, f"HEAD:{GIT_BRANCH}")
    except subprocess.CalledProcessError as e:
        log_event({"type": "error", "context": "push_log_to_github", "error": e.stderr[:1000]})


def web_search(query: str) -> str:
    # DuckDuckGo's HTML/lite endpoints serve an anomaly-detection page to most
    # datacenter IPs (returns 202 with no real results), so we use Wikipedia's
    # public search API instead -- it's unauthenticated, unblocked, and its
    # articles on Indian states/statistics usually cite the underlying MOSPI
    # report, which the model can then read directly with fetch_url. Wikipedia's
    # search doesn't understand search-engine operators (site:, quotes as
    # exact-phrase, etc); strip them so the leftover keywords still match.
    query = re.sub(r"\bsite:\S+", "", query).strip()
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 8,
            },
            headers={"User-Agent": "DataAnalystBot/1.0"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
    except (requests.RequestException, ValueError) as e:
        return f"ERROR searching for '{query}': {e}"

    if not hits:
        return "No results found."

    results = []
    for hit in hits:
        title = hit["title"]
        snippet = re.sub("<[^>]+>", "", hit.get("snippet", ""))
        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        results.append(f"- {title}\n  {snippet}\n  {url}")
    return "\n".join(results)


def _is_tabular(url: str, content_type: str) -> bool:
    lower_url = url.lower()
    return (
        lower_url.endswith((".xlsx", ".xls", ".csv"))
        or "spreadsheet" in content_type
        or "excel" in content_type
        or "csv" in content_type
    )


def _load_dataframe(url: str, sheet: str = None):
    """Fetch and parse a CSV/Excel URL into a pandas DataFrame, with caching."""
    import pandas as pd

    cache_key = (url, sheet)
    if cache_key in dataframe_cache:
        return dataframe_cache[cache_key]

    resp = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DataAnalystBot/1.0)"},
    )
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()
    lower_url = url.lower()

    if lower_url.endswith((".xlsx", ".xls")) or "spreadsheet" in content_type or "excel" in content_type:
        xls = pd.ExcelFile(io.BytesIO(resp.content))
        sheet_name = sheet if sheet in xls.sheet_names else xls.sheet_names[0]
        df = xls.parse(sheet_name)
    else:
        df = pd.read_csv(io.BytesIO(resp.content))

    dataframe_cache[cache_key] = df
    return df


def query_dataset(
    url: str,
    sheet: str = None,
    filters: list = None,
    groupby: list = None,
    agg: dict = None,
    sort_by: str = None,
    ascending: bool = True,
    columns: list = None,
    limit: int = 20,
) -> str:
    try:
        df = _load_dataframe(url, sheet)
    except Exception as e:
        return f"ERROR loading dataset from {url}: {e}"

    try:
        for f in filters or []:
            col, op, value = f["column"], f["op"], f["value"]
            if col not in df.columns:
                return f"ERROR: unknown column '{col}'. Available columns: {list(df.columns)}"
            series = df[col]
            if op == "==":
                df = df[series == value]
            elif op == "!=":
                df = df[series != value]
            elif op == ">":
                df = df[series > value]
            elif op == "<":
                df = df[series < value]
            elif op == ">=":
                df = df[series >= value]
            elif op == "<=":
                df = df[series <= value]
            elif op == "contains":
                df = df[series.astype(str).str.contains(str(value), case=False, na=False)]
            else:
                return f"ERROR: unknown op '{op}'"

        if groupby:
            missing = [c for c in groupby if c not in df.columns]
            if missing:
                return f"ERROR: unknown groupby column(s) {missing}. Available: {list(df.columns)}"
            if agg:
                bad = [c for c in agg if c not in df.columns]
                if bad:
                    return f"ERROR: unknown agg column(s) {bad}. Available: {list(df.columns)}"
                df = df.groupby(groupby).agg(agg).reset_index()
            else:
                df = df.groupby(groupby).size().reset_index(name="count")

        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"ERROR: unknown column(s) {missing}. Available: {list(df.columns)}"
            df = df[columns]

        if sort_by:
            if sort_by not in df.columns:
                return f"ERROR: unknown sort_by column '{sort_by}'. Available: {list(df.columns)}"
            df = df.sort_values(sort_by, ascending=ascending)

        limit = max(1, min(int(limit or 20), 200))
        total_rows = len(df)
        df = df.head(limit)
        result = f"Matched {total_rows} row(s), showing up to {limit}:\n\n{df.to_string(index=False)}"
        return result[:14000]
    except Exception as e:
        return f"ERROR querying dataset: {e}"


def fetch_url(url: str, sheet: str = None) -> str:
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DataAnalystBot/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"ERROR fetching {url}: {e}"

    content_type = resp.headers.get("Content-Type", "").lower()
    lower_url = url.lower()

    try:
        if _is_tabular(url, content_type):
            df = _load_dataframe(url, sheet)
            summary = f"Shape: {df.shape}\nColumns: {list(df.columns)}\n\n{df.head(30).to_string()}"
            numeric = df.select_dtypes(include="number")
            if numeric.shape[1] > 0:
                summary += f"\n\nNumeric summary:\n{numeric.describe().to_string()}"
            summary += (
                "\n\n(This is only a preview of the full data -- use query_dataset to filter/"
                "sort/aggregate over ALL rows before computing an answer.)"
            )
            return summary[:12000]

        if lower_url.endswith(".pdf") or "pdf" in content_type:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(resp.content))
            text = ""
            for page in reader.pages[:15]:
                text += page.extract_text() or ""
                if len(text) > 12000:
                    break
            return text[:12000] or "(no extractable text found in PDF)"

        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and href not in seen:
                link_text = a.get_text(strip=True)
                if link_text:
                    seen.add(href)
                    links.append(f"{link_text} -> {href}")
                    if len(links) >= 60:
                        break

        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        result = text[:10000]
        if links:
            result += "\n\n--- Links found on this page (fetch_url one if it looks like the primary source) ---\n"
            result += "\n".join(links)
        return result[:14000]
    except Exception as e:
        return f"ERROR parsing content from {url}: {e}"


def run_agent(history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "web_search":
                result = web_search(args.get("query", ""))
            elif name == "fetch_url":
                result = fetch_url(args.get("url", ""), args.get("sheet"))
            elif name == "query_dataset":
                result = query_dataset(
                    url=args.get("url", ""),
                    sheet=args.get("sheet"),
                    filters=args.get("filters"),
                    groupby=args.get("groupby"),
                    agg=args.get("agg"),
                    sort_by=args.get("sort_by"),
                    ascending=args.get("ascending", True),
                    columns=args.get("columns"),
                    limit=args.get("limit", 20),
                )
            else:
                result = f"Unknown tool {name}"

            log_event({"type": "tool_call", "tool": name, "args": args, "result_preview": result[:500]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content or ""


def send_telegram_message(chat_id, text):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=REQUEST_TIMEOUT,
    )


def process_update(update: dict):
    try:
        message = update.get("message") or update.get("edited_message")
        if not message or "text" not in message:
            return
        chat_id = message["chat"]["id"]
        user_text = message["text"]
        log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

        with state_lock:
            history = conversation_history.setdefault(chat_id, [])
            history.append({"role": "user", "content": user_text})
            history_snapshot = list(history[-MAX_HISTORY:])

        try:
            reply_text = run_agent(history_snapshot)
        except Exception as e:
            log_event({"type": "error", "context": "run_agent", "error": str(e)})
            reply_text = json.dumps({"error": "agent_failed", "log_url": LOG_URL})

        with state_lock:
            history.append({"role": "assistant", "content": reply_text})

        try:
            parsed = json.loads(reply_text)
        except json.JSONDecodeError:
            start, end = reply_text.find("{"), reply_text.rfind("}")
            parsed = json.loads(reply_text[start : end + 1])

        if isinstance(parsed, dict) and "log_url" in parsed:
            parsed["log_url"] = LOG_URL
        final_reply = json.dumps(parsed)

        log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
        send_telegram_message(chat_id, final_reply)
    except Exception as e:
        log_event({"type": "error", "context": "process_update", "error": str(e)})
    finally:
        push_log_to_github()


app = Flask(__name__)


@app.route("/telegram-webhook", methods=["POST"])
def telegram_webhook():
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        return "forbidden", 403
    update = request.get_json(force=True, silent=True) or {}
    threading.Thread(target=process_update, args=(update,), daemon=True).start()
    return "ok", 200


@app.route("/")
def health():
    return "ok", 200


def _ensure_git_identity():
    for key, value in (("user.email", "bot@render.local"), ("user.name", "Data Analyst Bot")):
        current = _git("config", key, check=False)
        if not current.stdout.strip():
            _git("config", key, value)


def register_webhook():
    if not RENDER_EXTERNAL_URL:
        print("RENDER_EXTERNAL_URL not set; skipping webhook registration (local/dev mode).")
        return
    url = f"{RENDER_EXTERNAL_URL}/telegram-webhook"
    resp = requests.post(
        f"{TELEGRAM_API}/setWebhook",
        json={"url": url, "secret_token": WEBHOOK_SECRET},
        timeout=REQUEST_TIMEOUT,
    )
    log_event({"type": "webhook_registration", "url": url, "response": resp.json()})


_ensure_git_identity()
register_webhook()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
