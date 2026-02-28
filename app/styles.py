"""
app/styles.py
─────────────
Shared CSS for all DevSearch pages. Import and call inject_css() once per page.
Centralised here to avoid duplicate style blocks and ensure visual consistency.
"""

SHARED_CSS = """
<style>
/* ── Gradient hero title ─────────────────────────────────────────────────── */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* ── Stat cards ──────────────────────────────────────────────────────────── */
.stat-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.stat-num  { font-size: 2rem; font-weight: 700; color: #6366f1; }
.stat-label{ font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── Feature pills ───────────────────────────────────────────────────────── */
.pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #6366f1;
    border-radius: 9999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    color: #a5b4fc;
    margin: 0.2rem;
}

/* ── Info box ─────────────────────────────────────────────────────────────── */
.info-box {
    background: #0f2744;
    border-left: 4px solid #6366f1;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #cbd5e1;
}

/* ── Result cards (search) ───────────────────────────────────────────────── */
.result-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem 0.5rem 1.25rem;
    margin-bottom: 1rem;
}
.result-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
}
.file-path   { font-family: monospace; font-size: 0.9rem; color: #a5b4fc; }
.func-name   { font-size: 0.85rem; color: #34d399; font-family: monospace; }
.sim-badge   {
    background: #0f4c81;
    border-radius: 9999px;
    padding: 0.15rem 0.65rem;
    font-size: 0.78rem;
    color: #7dd3fc;
    font-weight: 600;
}
.lang-badge  {
    background: #2d1f63;
    border-radius: 9999px;
    padding: 0.15rem 0.65rem;
    font-size: 0.78rem;
    color: #c4b5fd;
    margin-right: 0.4rem;
}
.line-info   { font-size: 0.78rem; color: #64748b; }

/* ── Evidence cards (ask) ────────────────────────────────────────────────── */
.evidence-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
}
.ev-path  { color: #a5b4fc; font-family: monospace; }
.ev-sim   { color: #34d399; }

/* ── Index cards (indexes) ───────────────────────────────────────────────── */
.index-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.index-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #a5b4fc;
    font-family: monospace;
}
.index-meta {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.25rem;
}
.badge-green {
    background: #064e3b;
    color: #6ee7b7;
    border-radius: 9999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    margin-left: 0.5rem;
}
</style>
"""


def inject_css(st_module):
    """Inject the shared CSS into a Streamlit page."""
    st_module.markdown(SHARED_CSS, unsafe_allow_html=True)
