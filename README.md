# Data Analyst Telegram Bot

An LLM agent (via [aipipe.org](https://aipipe.org)) that answers data-analysis questions over
Telegram and replies with exactly one JSON object, as specified by each incoming message. It can
call two tools while reasoning: `web_search` and `fetch_url` (reads HTML/CSV/XLS/XLSX/PDF), so it
can look up real public data (e.g. MOSPI) instead of guessing.

Runs as a Flask webhook app (not long-polling) so it works on Render's free, scale-to-zero web
service tier without needing an always-on process.

## Local setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your own tokens
```

Env vars (see `.env.example`):
- `TELEGRAM_BOT_TOKEN` — from @BotFather
- `AIPIPE_TOKEN` — from aipipe.org/login
- `WEBHOOK_SECRET` — any random string, must match what you register with Telegram
- `GITHUB_TOKEN` — a GitHub PAT (Contents: read/write) scoped to this repo, used to auto-push
  `run.jsonl` after every exchange so `LOG_URL` stays up to date. Leave blank to disable.
- `GITHUB_REPO`, `GIT_BRANCH`, `LOG_URL` — where the log gets pushed / served from.

`RENDER_EXTERNAL_URL` is set automatically by Render; when present, the bot registers itself as
the Telegram webhook at startup. Locally (no public URL) this step is skipped, so local runs are
for testing the agent logic directly, not a live Telegram round-trip.

## Deploy (Render, free tier)

1. Push this repo to GitHub (public).
2. On [Render](https://render.com), New → Blueprint → connect this repo (`render.yaml` is
   already set up as a free Web Service).
3. Fill in the secret env vars in the dashboard (`TELEGRAM_BOT_TOKEN`, `AIPIPE_TOKEN`,
   `WEBHOOK_SECRET`, `GITHUB_TOKEN`).
4. Deploy. On boot the app calls Telegram's `setWebhook` pointing at its own Render URL.
5. Message the bot from Telegram (from a fresh chat, not just your test chat) to confirm it
   replies live.

Render's free web services spin down after ~15 min idle; an inbound Telegram message itself wakes
the instance (cold start ~30-60s), which is why the bot uses a webhook instead of long-polling —
polling would stay silent forever once the instance is asleep.

## Logging

Every incoming message, tool call, outgoing reply, and error is appended as one JSON line to
`run.jsonl`. After each exchange the bot commits and pushes that file back to this repo (if
`GITHUB_TOKEN` is set), so the raw GitHub URL below always serves the latest log:

```
https://raw.githubusercontent.com/pratik-iitm/q5/main/run.jsonl
```
