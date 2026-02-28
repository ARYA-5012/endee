"""
scripts/ingest_repo.py
──────────────────────
Command-line tool for indexing a GitHub repository into Endee.

Usage
-----
    python scripts/ingest_repo.py https://github.com/owner/repo

Optional flags
--------------
    --token   YOUR_GITHUB_TOKEN    Raise API rate limit 60 → 5000 req/hr
    --base-url http://localhost:8080/api/v1
    --force    Re-index even if this repo is already indexed

Example
-------
    python scripts/ingest_repo.py https://github.com/tiangolo/fastapi \\
        --token ghp_xxxx --force
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from core.chunker       import chunk_file
from core.embedder      import embed_texts
from core.endee_client  import DevSearchDB
from core.repo_loader   import fetch_repo_files, parse_github_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index a GitHub repository into Endee for DevSearch."
    )
    parser.add_argument("url",        help="GitHub repository URL")
    parser.add_argument("--token",    default=os.getenv("GITHUB_TOKEN", ""),
                        help="GitHub personal access token (optional)")
    parser.add_argument("--base-url", default=os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1"),
                        help="Endee server base URL")
    parser.add_argument("--auth-token", default=os.getenv("ENDEE_AUTH_TOKEN", ""),
                        help="Endee auth token (if NDD_AUTH_TOKEN was set)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-index even if this repo is already indexed")
    args = parser.parse_args()

    owner, repo_name = parse_github_url(args.url)
    repo = f"{owner}/{repo_name}"

    print(f"\n🔍 DevSearch Ingestion Pipeline")
    print(f"   Repo     : {repo}")
    print(f"   Endee URL: {args.base_url}\n")

    # ── Connect to Endee ──────────────────────────────────────────────────────
    try:
        db = DevSearchDB(base_url=args.base_url, auth_token=args.auth_token)
    except Exception as e:
        print(f"❌ Cannot connect to Endee: {e}")
        print("   Make sure Endee is running. See README for setup instructions.")
        sys.exit(1)

    # ── Check for existing index ──────────────────────────────────────────────
    already_exists = db.index_exists(repo)  # single API call

    if already_exists and not args.force:
        print(f"✅ '{repo}' is already indexed in Endee.")
        print("   Use --force to re-index.")
        sys.exit(0)

    if already_exists and args.force:
        print(f"🗑  Deleting existing index for '{repo}'...")
        db.delete_index(repo)

    db.create_index(repo, skip_exists_check=already_exists and args.force)
    print(f"📦 Created Endee index: {db.index_name_for_repo(repo)}")

    # ── Fetch files ───────────────────────────────────────────────────────────
    print(f"\n📥 Fetching files from GitHub...")
    t0 = time.time()

    all_chunks = []
    file_count = 0

    for code_file, idx, total in fetch_repo_files(args.url, args.token or None):
        file_count += 1
        chunks = chunk_file(code_file)
        all_chunks.extend(chunks)

        bar = "█" * int(20 * idx / total) + "░" * (20 - int(20 * idx / total))
        print(
            f"\r   [{bar}] {idx}/{total} files  |  {len(all_chunks)} chunks",
            end="", flush=True,
        )

    print(f"\n\n📄 Fetched {file_count} files → {len(all_chunks)} code chunks")

    if not all_chunks:
        print("⚠️  No chunks produced. The repository may be empty or unsupported.")
        sys.exit(1)

    # ── Embed ─────────────────────────────────────────────────────────────────
    print(f"\n🧠 Embedding {len(all_chunks)} chunks with all-MiniLM-L6-v2...")
    texts = [c.code for c in all_chunks]
    embeddings = embed_texts(texts, batch_size=64)
    print(f"   Done embedding.")

    # ── Upsert into Endee ─────────────────────────────────────────────────────
    print(f"\n⚡ Upserting into Endee...")
    upserted = db.upsert_chunks(repo, all_chunks, embeddings)

    elapsed = time.time() - t0
    print(f"\n✅ Ingestion complete!")
    print(f"   Vectors stored : {upserted}")
    print(f"   Time elapsed   : {elapsed:.1f}s")
    print(f"\n   Start the Streamlit app and search '{repo}' 🚀\n")


if __name__ == "__main__":
    main()
