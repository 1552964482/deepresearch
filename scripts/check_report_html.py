"""Quick structural sanity check for docs/report.html."""

import re
from pathlib import Path

html = Path("docs/report.html").read_text(encoding="utf-8")
print("size:", len(html))
print("sections:", len(re.findall(r"<section id=", html)))
print("mermaid blocks:", len(re.findall(r'class="mermaid"', html)))
print("tables:", len(re.findall(r"<table>", html)))
print("toc has multicritic:", '#multicritic' in html)
print("section has multicritic:", 'id="multicritic"' in html)
print("closes html:", html.rstrip().endswith("</html>"))
# balance check: every <section opens and closes
print("open sections:", html.count("<section"), "close sections:", html.count("</section>"))
