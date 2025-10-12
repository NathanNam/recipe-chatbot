#!/usr/bin/env python3
"""
Generate recipe queries from a CSV of tuples using an LLM.

This script reads tuples from a CSV (e.g., 100_unique_tuples.csv) and uses an LLM
to synthesize natural-language user queries for each tuple.

Input CSV must have columns (case-insensitive match supported):
  - "Cuisine Type"
  - "Dietary Restriction"
  - "Meal Type"

Outputs (by default):
  - generated_queries.csv
  - generated_queries.jsonl

Providers:
  - OpenAI (requires `openai` package and OPENAI_API_KEY in env)
  - LiteLLM (requires `litellm` package and provider-specific env)

Usage examples:
  python3 generate_synthetic_queries_modified.py \
    --input_csv 100_unique_tuples.csv \
    --out_csv generated_queries.csv \
    --out_jsonl generated_queries.jsonl \
    --provider openai \
    --model gpt-4o-mini

  python3 generate_synthetic_queries_modified.py \
    --input_csv 100_unique_tuples.csv \
    --provider litellm \
    --model gpt-4o-mini \
    --batch_size 25 --temperature 0.4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


# Optional imports (loaded lazily)
_openai = None
_litellm = None


@dataclass
class DimensionTuple:
    cuisine: str
    dietary: str
    meal: str


@dataclass
class QueryWithDimensions:
    id: str
    query: str
    CuisinePreference: str
    DietaryNeedsOrRestrictions: str
    MealType: str
    is_realistic_and_kept: int = 1
    notes_for_filtering: str = ""


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers to a canonical set and support common synonyms/typos."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    def find_col(*candidates):
        for cand in candidates:
            key = cand.lower().strip()
            if key in lower_map:
                return lower_map[key]
        return None

    cuisine_col = find_col("Cuisine Type", "Cuisine", "CuisineType")
    diet_col = find_col("Dietary Restriction", "Dietary", "Diet", "DietaryRestriction")
    meal_col = find_col("Meal Type", "Meal", "MealType")

    missing = [name for name, col in [
        ("Cuisine Type", cuisine_col),
        ("Dietary Restriction", diet_col),
        ("Meal Type", meal_col),
    ] if col is None]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}. Found columns: {list(df.columns)}")

    # Rename to canonical
    out = df.rename(columns={
        cuisine_col: "Cuisine Type",
        diet_col: "Dietary Restriction",
        meal_col: "Meal Type",
    })
    return out[["Cuisine Type", "Dietary Restriction", "Meal Type"]]


def load_tuples(csv_path: str | os.PathLike) -> List[DimensionTuple]:
    df = pd.read_csv(csv_path)
    df = _normalize_headers(df)

    tuples: List[DimensionTuple] = []
    for _, row in df.iterrows():
        tuples.append(DimensionTuple(
            cuisine=str(row["Cuisine Type"]).strip(),
            dietary=str(row["Dietary Restriction"]).strip(),
            meal=str(row["Meal Type"]).strip(),
        ))
    return tuples


def _system_prompt() -> str:
    return ("""You are a helpful assistant that writes concise, natural recipe-lookup queries for a Recipe Bot.
You will be given multiple (cuisine, dietary restriction, meal type) tuples and must output exactly one query per tuple, in JSON Lines format (one JSON object per line).
Each JSON object MUST contain keys: 'index' (int, the tuple index as provided), and 'query' (string).
Do not include any extra text before or after the JSON Lines.
The queries should:
1. Sound like real users asking for recipe help
2. Naturally incorporate all the dimension values
3. Vary in style and detail level
4. Be realistic and practical
5. Include natural variations in typing style, such as:
   - Some queries in all lowercase
   - Some with random capitalization
   - Some with common typos
   - Some with missing punctuation
   - Some with extra spaces or missing spaces
   - Some with emojis or text speak

Here are examples of realistic query variations for a beginner, vegan, quick recipe:

Proper formatting:
- "Need a simple vegan dinner that's ready in 20 minutes"
- "What's an easy plant-based recipe I can make quickly?"

All lowercase:
- "need a quick vegan recipe for dinner"
- "looking for easy plant based meals"

Random caps:
- "NEED a Quick Vegan DINNER recipe"
- "what's an EASY plant based recipe i can make"

Common typos:
- "need a quik vegan recip for dinner"
- "wat's an easy plant based recipe i can make"

Missing punctuation:
- "need vegan dinner ideas quick"
- "easy plant based recipe 20 mins"

With emojis/text speak:
- "need vegan dinner ideas asap! 🥗"
- "pls help with quick plant based recipe thx"

General rules:
- Keep each query short (<= 20 words).
- Respect the dietary restriction; don't add disallowed ingredients.
- Ensure the cuisine and meal type are explicit.
- Output ONLY valid JSON Lines; no prose, headers, or code fences."""
)



def _user_prompt_for_batch(batch: List[DimensionTuple], start_index: int) -> str:
    # Provide the tuples in a clear list with indices
    lines = ["Tuples:"]
    for i, t in enumerate(batch, start=start_index):
        lines.append(f"{i}: (Cuisine='{t.cuisine}', Dietary='{t.dietary}', Meal='{t.meal}')")
    lines.append("Output: JSON Lines with objects of the form: {\"index\": <int>, \"query\": \"...\"}")
    return "\n".join(lines)


def _lazy_import_openai():
    global _openai
    if _openai is None:
        try:
            import openai  # type: ignore
            _openai = openai
        except Exception as e:
            raise RuntimeError("OpenAI provider selected but `openai` package not installed.") from e
    return _openai


def _lazy_import_litellm():
    global _litellm
    if _litellm is None:
        try:
            import litellm  # type: ignore
            _litellm = litellm
        except Exception as e:
            raise RuntimeError("LiteLLM provider selected but `litellm` package not installed.") from e
    return _litellm


def call_llm_openai(model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    openai = _lazy_import_openai()
    # Newer OpenAI client style
    try:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        return resp.choices[0].message.content or ""
    except AttributeError:
        # Fallback to legacy API if needed
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        return resp["choices"][0]["message"]["content"]


def call_llm_litellm(model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    litellm = _lazy_import_litellm()
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    # LiteLLM returns an object similar to OpenAI's
    try:
        return resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
    except Exception:
        return resp["choices"][0]["message"]["content"]


def call_llm(provider: str, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    if provider.lower() == "openai":
        return call_llm_openai(model, system, user, temperature, max_tokens)
    elif provider.lower() == "litellm":
        return call_llm_litellm(model, system, user, temperature, max_tokens)
    else:
        raise ValueError("Unsupported provider. Use 'openai' or 'litellm'.")


def parse_jsonl_output(text: str) -> Dict[int, str]:
    """
    Parse JSON Lines output into {index: query}. Accepts minor formatting mistakes gracefully.
    """
    results: Dict[int, str] = {}
    # Remove any accidental wrapping text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        # Try strict JSON first
        obj = None
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            # Try to salvage {index: x, query: y}
            m = re.search(r'\"index\"\s*:\s*(\d+).*?\"query\"\s*:\s*\"(.*?)\"', ln)
            if m:
                obj = {"index": int(m.group(1)), "query": m.group(2)}
        if isinstance(obj, dict) and "index" in obj and "query" in obj:
            try:
                idx = int(obj["index"])
                q = str(obj["query"]).strip()
                if q:
                    results[idx] = q
            except Exception:
                pass
    return results


def batch_generate_queries(
    tuples: List[DimensionTuple],
    provider: str,
    model: str,
    batch_size: int = 20,
    temperature: float = 0.3,
    max_tokens: int = 512,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
) -> List[str]:
    """
    Returns a list of queries aligned to the input tuples order (one per tuple).
    """
    queries: List[Optional[str]] = [None] * len(tuples)
    sys_prompt = _system_prompt()

    for start in range(0, len(tuples), batch_size):
        end = min(start + batch_size, len(tuples))
        batch = tuples[start:end]
        user_prompt = _user_prompt_for_batch(batch, start_index=start)

        attempt = 0
        while attempt < max_retries:
            try:
                raw = call_llm(provider, model, sys_prompt, user_prompt, temperature, max_tokens)
                parsed = parse_jsonl_output(raw)
                # Fill queries for indices in this slice
                for i in range(start, end):
                    if i in parsed and not queries[i]:
                        queries[i] = parsed[i]
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    raise
                time.sleep(retry_backoff ** attempt)

        # As a fallback, synthesize trivial queries for any missing ones in this batch
        for i in range(start, end):
            if not queries[i]:
                t = tuples[i]
                dietary = "" if t.dietary.lower() == "none" else t.dietary
                if dietary:
                    article = "an" if dietary[:1].lower() in "aeiou" else "a"
                    queries[i] = f"Suggest {article} {dietary} {t.cuisine} {t.meal.lower()} recipe."
                else:
                    queries[i] = f"Suggest a {t.cuisine} {t.meal.lower()} recipe."

    # type: ignore
    return [q or "" for q in queries]


def save_outputs(
    tuples: List[DimensionTuple],
    queries: List[str],
    out_csv: str | os.PathLike,
    out_jsonl: str | os.PathLike,
) -> None:
    rows: List[QueryWithDimensions] = []
    for t, q in zip(tuples, queries):
        rows.append(QueryWithDimensions(
            id=str(uuid.uuid4()),
            query=q,
            CuisinePreference=t.cuisine,
            DietaryNeedsOrRestrictions=t.dietary,
            MealType=t.meal,
        ))
    # CSV
    df = pd.DataFrame([asdict(r) for r in rows])
    pd.DataFrame(df).to_csv(out_csv, index=False, encoding="utf-8")
    # JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate recipe queries from CSV tuples using an LLM.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the tuples CSV (e.g., 100_unique_tuples.csv)")
    parser.add_argument("--out_csv", type=str, default="generated_queries.csv", help="Path to output CSV")
    parser.add_argument("--out_jsonl", type=str, default="generated_queries.jsonl", help="Path to output JSONL")
    parser.add_argument("--provider", type=str, choices=["openai", "litellm"], default="openai", help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name for the provider")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of tuples per LLM request")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens for LLM response")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for API errors")
    parser.add_argument("--retry_backoff", type=float, default=1.5, help="Exponential backoff base for retries")
    args = parser.parse_args()

    tuples = load_tuples(args.input_csv)
    queries = batch_generate_queries(
        tuples=tuples,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
    save_outputs(tuples, queries, args.out_csv, args.out_jsonl)

    print(f"Loaded {len(tuples)} tuples from: {args.input_csv}")
    print(f"Wrote CSV:  {args.out_csv}")
    print(f"Wrote JSONL:{args.out_jsonl}")


if __name__ == "__main__":
    main()
