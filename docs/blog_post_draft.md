# I Built a Multi-Agent AI Research System — Here's Every Problem I Hit (And How I Fixed Them)



---

**The pitch sounds simple:** enter a topic, and three AI agents automatically search the web, extract facts, and write a publication-ready report. No hallucination. No copy-paste. Just grounded research.

The reality? Four distinct engineering problems that took me longer to fix than the entire core build.

This is that story.

---

## What I Built

MARS (Multi-Agent Research System) is a LangGraph pipeline with three specialised agents:

1. 🔍 **Web Searcher** — uses Tavily to find sources, then a Groq LLM to refine the query
2. 📄 **Fact Extractor** — fetches each URL, strips boilerplate, uses an LLM to pull out key facts
3. ✍️ **Report Writer** — synthesises all facts into a structured markdown report, citing sources

The whole thing is orchestrated by LangGraph's `StateGraph` and served through a Streamlit UI with a dark glassmorphic design.

Tech stack: Python, LangGraph, Groq (`llama-3.3-70b-versatile`), Tavily, httpx, BeautifulSoup4, Streamlit, LangSmith.

**All free. All open.** No OpenAI bill.

---

## Problem 1: Groq Rate Limits Killed My Pipeline Mid-Run

The Groq free tier gives you 30 requests per minute. My pipeline makes three LLM calls per research run (query refinement → fact extraction → report writing). During early testing, the pipeline would get through the first two agents and then crash with a `429 Too Many Requests` error right before generating the final report.

```
groq.RateLimitError: Rate limit reached for model `llama-3.3-70b-versatile`
```

**What I tried first (wrong):** wrapping everything in a single `try/except` and just retrying immediately. This made it worse — rapid retries burned through the quota even faster.

**What actually worked:** Two mechanisms together.

First, an explicit sleep between agent transitions in the orchestrator:
```python
time.sleep(2)   # Between search → extract
time.sleep(2)   # Between extract → write
```

Second, exponential backoff inside each agent on 429:
```python
BACKOFF_BASE = 2
MAX_RETRIES = 4

for attempt in range(MAX_RETRIES):
    try:
        return llm.invoke(messages)
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            wait = BACKOFF_BASE ** attempt  # 1s, 2s, 4s, 8s
            time.sleep(wait)
        else:
            raise
```

The 2-second sleep costs ~6 extra seconds per run. Worth it for zero rate-limit crashes.

---

## Problem 2: 30% of URLs Returned Empty Content

`httpx.get(url)` works great for static HTML pages. But a lot of modern sites (Next.js, React SPAs, anything that renders content client-side) return a near-empty HTML shell. The actual content is loaded by JavaScript after the initial response.

My extractor was silently returning empty strings for these pages — and I didn't notice for two days because the LLM would just... make something up to fill the void.

**Detection:** I added a length check after each fetch. Anything under 500 characters is almost certainly a JS-rendered shell.

**Fix:** Two-stage fetching with Playwright as fallback:
```python
async def _fetch_with_fallback(url: str) -> str:
    # Stage 1: Fast static fetch
    text = await _fetch_httpx(url)
    
    # Stage 2: If response is suspiciously short, try Playwright
    if len(text) < 500:
        text = await _fetch_playwright(url)
    
    return text
```

Playwright spins up a real Chromium browser, waits for JavaScript to execute, then returns the fully rendered DOM. It's slower (~3-5s per page vs ~0.5s for httpx), which is why it's only a fallback.

---

## Problem 3: The LLM Kept Hallucinating in Reports

This was the most subtle bug. The report *looked* great — well-structured, confident, well-cited. But when I cross-checked the facts against the actual source pages, roughly 20% of the "facts" were fabricated. The LLM was filling gaps from its training data.

This is a fundamental problem with LLMs — they're trained to sound confident and complete, even when they should say "I don't have enough information."

**Fix:** Grounding via strict prompt engineering. The writer agent only receives the extracted facts as input — never the raw URLs or original web pages. And the system prompt is explicit:

```
You are a research writer. You will receive a FACTS BLOCK containing only verified facts extracted from web sources.

STRICT RULES:
- Only use facts from the FACTS BLOCK below
- Do NOT add anything from your training data
- If a fact is not in the block, do not write it
- If you are unsure, write "Based on available sources..." not a confident assertion
- Every claim must be traceable to a specific URL in the facts block

FACTS BLOCK:
{facts}
```

The key insight: **constrained context beats unconstrained generation.** Giving the LLM *less* freedom produced *more* accurate output.

---

## Problem 4: Windows Dependency Hell

This one surprised me. `lxml` — the fastest HTML parser for BeautifulSoup4 — requires Microsoft C++ Build Tools to compile on Windows. That's a 4GB download from Microsoft's website, hidden behind a confusing Visual Studio installer.

Most AI engineering tutorials assume you're on Linux or macOS. They aren't written for Windows developers.

**Fix:** Switched to Python's built-in `html.parser`. It's slower than lxml (maybe 2-3x), but:
- Zero dependencies
- Ships with Python
- Works on every platform, no compiler needed

```python
# Before (failed on Windows without C++ Build Tools):
soup = BeautifulSoup(html, "lxml")

# After (works everywhere):
soup = BeautifulSoup(html, "html.parser")
```

For a project at this scale, the speed difference is ~50ms per page. Completely irrelevant.

---

## What I'd Do Differently

1. **Design for rate limits from day one.** I added the sleep/backoff as a patch. It should have been in the original architecture.
2. **Validate LLM outputs with a schema.** Every fact extraction should be a structured Pydantic object, not free-text. Would have caught the hallucination problem earlier.
3. **Add an integration test that checks fact grounding.** A test that verifies every sentence in the report maps to an extracted fact — not just that the pipeline runs.

---

## What's Next (v2 Roadmap)

- **ChromaDB vector store** — store extracted facts in a vector DB so follow-up questions don't re-fetch URLs
- **Parallel extraction** — run all URL fetches concurrently with `asyncio.gather()` instead of sequentially
- **React + Framer Motion UI** — replace Streamlit with a proper frontend for smoother agent status animations

---

## Try It Yourself

**Live demo:** [your-app.streamlit.app](#)
**GitHub:** [github.com/Ajitabh-12-7/MK-1](https://github.com/Ajitabh-12-7/MK-1)
**All keys are free** — Groq, Tavily, LangSmith all have free tiers.

---

*If you're a student building AI projects for your portfolio — focus on the problems you solved, not the tools you used. The tools are replaceable. Your engineering judgment isn't.*
