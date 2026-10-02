"""
generate_report_2026.py — Convert backtests/daily_brief.txt to a PDF report.

Reads the pre-generated daily brief, parses it into sections, and renders a
professional PDF using reportlab.

Run with:
    python generate_report_2026.py

Requires:
    pip install reportlab
"""

import datetime
import os
import platform
import re
import sys

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError:
    print("ERROR: reportlab not installed.  Run:  pip install reportlab")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BRIEF_PATH = "backtests/daily_brief.txt"
TODAY      = datetime.date.today()
PDF_PATH   = f"backtests/russell_brief_{TODAY}.pdf"

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = letter          # 8.5" × 11"
MARGIN         = 0.75 * inch
BODY_W         = PAGE_W - 2 * MARGIN   # ≈ 7"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

C_DARK  = colors.HexColor("#0f1c2e")
C_AMBER = colors.HexColor("#f59e0b")
C_LGRAY = colors.HexColor("#f0f2f5")
C_MGRAY = colors.HexColor("#e2e8f0")
C_TEXT  = colors.HexColor("#1a2332")
C_MUTED = colors.HexColor("#6b7280")
C_HINT  = colors.HexColor("#94a3b8")

# ---------------------------------------------------------------------------
# Font setup — try to register a Unicode-capable monospace font so that
# box-drawing characters (═ ─ │ █ ░ ★ ✓ ⚠ ⛔) render correctly.
# Falls back to Courier + ASCII sanitisation if no TTF font is found.
# ---------------------------------------------------------------------------

MONO = "Courier"    # updated below if a Unicode font is found
SANS = "Helvetica"
BOLD = "Helvetica-Bold"

_FONT_CANDIDATES = [
    # macOS
    ("/System/Library/Fonts/Menlo.ttc",          {"subfontIndex": 0}),
    ("/Library/Fonts/CourierNew.ttf",            {}),
    # Linux
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", {}),
    ("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", {}),
    # Windows
    ("C:/Windows/Fonts/cour.ttf",                {}),
]

_UNICODE_MONO = False

for _path, _kw in _FONT_CANDIDATES:
    if os.path.exists(_path):
        try:
            pdfmetrics.registerFont(TTFont("_UniMono", _path, **_kw))
            MONO = "_UniMono"
            _UNICODE_MONO = True
            break
        except Exception:
            continue

# ASCII replacements used when no Unicode font is available
_SANITIZE = str.maketrans({
    ord("═"): "=", ord("─"): "-", ord("│"): "|",
    ord("├"): "+", ord("┤"): "+",
    ord("★"): "*", ord("⚠"): "!", ord("⛔"): "X", ord("✓"): ">",
    ord("█"): "#", ord("░"): ".", ord("»"): ">",
})


def _clean(text: str) -> str:
    """Sanitise box-drawing chars if no Unicode font was registered."""
    return text if _UNICODE_MONO else text.translate(_SANITIZE)


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------

def _s(name: str, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)


S_H_TITLE = _s("h_title", fontName=SANS,  fontSize=18, textColor=colors.white,
                leading=24, spaceAfter=4)
S_H_SUB   = _s("h_sub",   fontName=SANS,  fontSize=9,  textColor=C_HINT, leading=14)
S_SEC     = _s("sec",     fontName=BOLD,  fontSize=11, textColor=C_DARK,
                spaceBefore=14, spaceAfter=4, leading=15)
S_INTRO   = _s("intro",   fontName=SANS,  fontSize=9.5, textColor=C_TEXT, leading=15)
S_EXP     = _s("exp",     fontName=SANS,  fontSize=9,  textColor=C_TEXT, leading=14)
S_DATA    = _s("data",    fontName=MONO,  fontSize=7.5, leading=10)


# ---------------------------------------------------------------------------
# Reusable layout helpers
# ---------------------------------------------------------------------------

def _tbl(content, col_w, bg=colors.white, border=C_MGRAY,
         lpad=8, rpad=8, tpad=6, bpad=6) -> Table:
    tbl = Table(content, colWidths=[col_w])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 0.5, border),
        ("LEFTPADDING",   (0, 0), (-1, -1), lpad),
        ("RIGHTPADDING",  (0, 0), (-1, -1), rpad),
        ("TOPPADDING",    (0, 0), (-1, -1), tpad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), bpad),
    ]))
    return tbl


def header_block() -> Table:
    """Full-width dark-blue page header."""
    subtitle = (
        f"{TODAY.strftime('%A, %B %d, %Y')}"
        f"&nbsp;&nbsp;<font color='#f59e0b'>|</font>&nbsp;&nbsp;"
        "Confidential — Internal Use Only"
    )
    return _tbl(
        [[Paragraph("Russell 2026 Reconstitution — Daily Brief", S_H_TITLE)],
         [Paragraph(subtitle, S_H_SUB)]],
        BODY_W, bg=C_DARK, border=C_DARK, lpad=16, rpad=16, tpad=14, bpad=14,
    )


def sec_heading(title: str) -> list:
    """Section title + amber rule."""
    return [
        Paragraph(title, S_SEC),
        HRFlowable(width="100%", thickness=2, color=C_AMBER, spaceAfter=6),
    ]


def data_block(text: str) -> Table:
    """White monospace block for terminal-style data."""
    cleaned = _clean(text.strip("\n"))
    return _tbl([[Preformatted(cleaned, S_DATA)]], BODY_W)


def explain_box(text: str) -> Table:
    """Light-grey explanation box with Helvetica body text."""
    return _tbl([[Paragraph(text, S_EXP)]], BODY_W,
                bg=C_LGRAY, lpad=10, rpad=10, tpad=8, bpad=8)


def full_section(title: str, data_text: str, explanation: str) -> list:
    """Assembles heading + data block + explanation box for one section."""
    elems: list = []
    elems += sec_heading(title)
    if data_text.strip():
        elems.append(data_block(data_text))
        elems.append(Spacer(1, 6))
    elems.append(explain_box(explanation))
    elems.append(Spacer(1, 10))
    return elems


# ---------------------------------------------------------------------------
# Explanation texts (static; not parsed from the brief)
# ---------------------------------------------------------------------------

EXPLAIN: dict[str, str] = {
    "MARKET SNAPSHOT": (
        "<b>RUT</b> = Russell 2000 index price. "
        "<b>VIX</b> = the market's fear gauge — below 20 is calm, above 30 is stressed. "
        "<b>SPX</b> = S&amp;P 500. "
        "<b>10Y-2Y Spread</b> = the yield curve — positive means normal economy, "
        "negative means recession warning. "
        "<b>RSI-14</b> = momentum indicator, above 70 is overbought, below 30 is oversold. "
        "<b>Days to recon</b> = trading days until June 26 effective date."
    ),
    "MACRO REGIME": (
        "<b>REGIME</b> is what our machine learning model thinks the Russell 2000 "
        "will do over the next few weeks, based on 21 macro features trained on 2010–2024 data. "
        "<b>BULLISH</b> = conditions favor going long. "
        "<b>BEARISH</b> = conditions favor reducing risk. "
        "<b>NEUTRAL</b> = mixed signals, use base position sizes. "
        "The multiplier adjusts how much of your book to risk: "
        "BULLISH = 1.25×, NEUTRAL = 1.0×, BEARISH = 0.5×. "
        "<b>CONVICTION</b> is a 0–10 score combining regime strength, model signal, "
        "hard stop distance, and days to recon. "
        "Below 4 = stay small. Above 7 = high confidence to size up."
    ),
    "HARD STOP": (
        "The hard stop is a circuit breaker. If the Russell 2000 falls more than <b>8%</b> "
        "over any 30-day window, close ALL positions immediately — do not wait. "
        "This is based on the June 2022 event where the entire reconstitution trade inverted "
        "and funds lost 11% in 3 weeks. "
        "<b>Gap to trigger</b> = how much further the market can fall before you must exit everything."
    ),
    "POSITION SIZING": (
        "Each row is a trade category. "
        "<b>Base</b> = recommended size as a percentage of your total portfolio. "
        "<b>Adj</b> = base multiplied by the regime multiplier. "
        "The arrow shows Base → Adjusted. Never exceed these sizes. "
        "Total gross exposure should stay around 5% of book — "
        "this is a precision trade, not a concentrated bet."
    ),
    "RANKED TRADE IDEAS": (
        "These are the highest-scoring individual stocks right now. "
        "<b>Score</b> = composite ranking (0–10) based on momentum, market-cap pressure, "
        "and volume impact. "
        "Setup types: <b>FLOW</b> = pure index mechanics forcing buying or selling. "
        "<b>MOMENTUM</b> = price trend supporting the trade. "
        "<b>REVERSAL</b> = stock has fallen hard and is expected to bounce after recon. "
        "<b>PARADOX</b> = promotion from Russell 2000 to Russell 1000 creates net selling pressure."
    ),
    "RISK FLAGS": (
        "<b>DANGER</b> = a stock in your long bucket has risen over 200% YTD — "
        "extreme momentum names can crash suddenly; use defined-risk options structures only "
        "(iron condors, call spreads). "
        "<b>CONFLICT</b> = same ticker appears in both a long and short bucket — "
        "resolve before sizing any position. "
        "<b>CAUTION</b> = regime is bearish but you have long positions — reduce. "
        "<b>CLOSE SOON</b> = fewer than 5 days to June 26 — begin exiting pre-recon positions. "
        "<b>WARNING</b> = hard stop gap below 3% — tighten stops now."
    ),
    "WEEKLY CALENDAR": (
        "FTSE Russell publishes updated preliminary lists every Friday until June 18. "
        "Each update can add or remove tickers. "
        "Check <b>russell.com</b> every Friday and update "
        "<b>data/candidates_override.csv</b> if the list changes. "
        "<b>June 8</b> is lockdown — no more changes after that date. "
        "<b>June 26 close</b> = exit all pre-recon positions into the closing auction. "
        "<b>June 29 open</b> = enter the post-recon deletion bounce basket."
    ),
}

WHAT_IS_THIS = (
    "Every June, the Russell 2000 index rebalances — adding new companies and removing others. "
    "This forces billions of dollars of passive fund buying and selling over a predictable "
    "30-day window. This system tracks that window, predicts which direction the broader market "
    "is heading, and tells you exactly how much to risk on each trade."
)

# ---------------------------------------------------------------------------
# Brief parser
# ---------------------------------------------------------------------------

def parse_brief(text: str) -> dict[str, str]:
    """Split the brief into named sections on '── TITLE ─────' lines."""
    sections: dict[str, str] = {}
    cur_name  = "__header__"
    cur_lines: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^── (.+?) ─+\s*$", line)
        if m:
            sections[cur_name] = "\n".join(cur_lines)
            cur_name  = m.group(1).strip()
            cur_lines = []
        else:
            cur_lines.append(line)

    sections[cur_name] = "\n".join(cur_lines)
    return sections


def get_section(sections: dict[str, str], prefix: str) -> str:
    """Case-insensitive prefix lookup; returns empty string if not found."""
    prefix_up = prefix.upper()
    for key, val in sections.items():
        if key.upper().startswith(prefix_up):
            return val.strip()
    return ""


# ---------------------------------------------------------------------------
# Footer / page-number callback
# ---------------------------------------------------------------------------

_FOOTER_TEXT = (
    "Generated by Russell 2026 Recon System  |  "
    "python dashboard.py  |  "
    "Next run: tomorrow before market open"
)


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(C_MGRAY)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 0.62 * inch, PAGE_W - MARGIN, 0.62 * inch)
    canvas.setFont(SANS, 7.5)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(MARGIN, 0.42 * inch, _FOOTER_TEXT)
    canvas.drawRightString(PAGE_W - MARGIN, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def build_pdf(sections: dict[str, str], output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=0.9 * inch,
        title="Russell 2026 Recon — Daily Brief",
        author="Russell 2026 Recon System",
    )

    story: list = []

    # ── 1. HEADER ─────────────────────────────────────────────────────────
    story.append(header_block())
    story.append(Spacer(1, 18))

    # ── 2. WHAT IS THIS REPORT ────────────────────────────────────────────
    story += sec_heading("What Is This Report?")
    story.append(Paragraph(WHAT_IS_THIS, S_INTRO))
    story.append(Spacer(1, 10))

    # ── 3–9. Data sections (parsed from brief + static explanation) ────────
    ordered: list[tuple[str, str, str]] = [
        ("MARKET SNAPSHOT",    "Market Snapshot",      "MARKET SNAPSHOT"),
        ("MACRO REGIME",       "Macro Regime",          "MACRO REGIME"),
        ("HARD STOP",          "Hard Stop",             "HARD STOP"),
        ("POSITION SIZING",    "Position Sizing",       "POSITION SIZING"),
        ("RANKED TRADE IDEAS", "Ranked Trade Ideas",    "RANKED TRADE IDEAS"),
        ("RISK FLAGS",         "Risk Flags",            "RISK FLAGS"),
        ("WEEKLY CALENDAR",    "Weekly Calendar",       "WEEKLY CALENDAR"),
    ]

    for brief_prefix, pdf_title, explain_key in ordered:
        data_text   = get_section(sections, brief_prefix)
        explanation = EXPLAIN.get(explain_key, "")
        story += full_section(pdf_title, data_text, explanation)

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Pre-flight checks
    if not os.path.exists(BRIEF_PATH):
        print(f"ERROR: {BRIEF_PATH} not found.")
        print("       Run  python dashboard.py  first to generate the brief.")
        sys.exit(1)

    with open(BRIEF_PATH, encoding="utf-8") as fh:
        text = fh.read()

    if not text.strip():
        print(f"ERROR: {BRIEF_PATH} is empty.")
        print("       Run  python dashboard.py  first.")
        sys.exit(1)

    os.makedirs("backtests", exist_ok=True)

    font_info = f"Unicode font: {MONO}" if _UNICODE_MONO else "Font: Courier (ASCII sanitisation active)"
    print(f"Font        : {font_info}")
    print(f"Parsing     : {BRIEF_PATH}")

    sections = parse_brief(text)
    found    = [k for k in sections if not k.startswith("__")]
    print(f"Sections    : {len(found)} found — {found}")

    print(f"Building PDF: {PDF_PATH}")
    build_pdf(sections, PDF_PATH)
    print(f"Saved       : {PDF_PATH}")


if __name__ == "__main__":
    main()
