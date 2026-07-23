"""
Export the full project scope document to PDF with clean metadata and no headers/footers.
Requirements: pip install playwright pikepdf && playwright install chromium
Usage:        python scripts/export_scope_full_pdf.py
Output:       scope_gait_pattern_analysis_full.pdf in the project root.
"""
import os, pikepdf
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent
HTML = (ROOT / "scripts" / "scope_full_project.html").resolve()
OUT  = str(ROOT / "scope_gait_pattern_analysis_full.pdf")

# 1. Render to PDF — no header/footer, backgrounds on
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(f"file://{HTML}", wait_until="networkidle")
    page.pdf(
        path=OUT,
        format="A4",
        print_background=True,
        display_header_footer=False,
        margin={"top": "16mm", "bottom": "16mm", "left": "16mm", "right": "16mm"},
    )
    browser.close()
    print(f"rendered: {OUT}")

# 2. Rewrite PDF metadata
with pikepdf.open(OUT, allow_overwriting_input=True) as pdf:
    with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
        meta["dc:title"]         = "Gait Pattern Analysis using Multiview Computer Vision — Project Scope"
        meta["dc:creator"]       = ["Asian Institute of Technology"]
        meta["dc:subject"]       = "Multiview Computer Vision · Gait Analysis · Movement Research"
        meta["pdf:Producer"]     = ""
        meta["xmp:CreatorTool"]  = ""
    pdf.docinfo["/Title"]    = "Gait Pattern Analysis using Multiview Computer Vision — Project Scope"
    pdf.docinfo["/Author"]   = "Asian Institute of Technology"
    pdf.docinfo["/Subject"]  = "Multiview Computer Vision · Gait Analysis · Movement Research"
    pdf.docinfo["/Creator"]  = ""
    pdf.docinfo["/Producer"] = ""
    pdf.save(OUT)

print(f"saved:    {OUT}")
