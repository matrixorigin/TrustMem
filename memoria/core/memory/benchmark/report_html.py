"""Render BenchmarkReport as a self-contained HTML file.

Uses inline CSS + SVG bar charts — no external dependencies.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memoria.core.memory.benchmark.schema import BenchmarkReport, ScenarioResult

_GRADE_COLORS = {
    "S": "#22c55e",
    "A": "#3b82f6",
    "B": "#eab308",
    "C": "#f97316",
    "D": "#ef4444",
}


def _bar(value: float, color: str = "#3b82f6") -> str:
    w = max(0, min(100, value))
    return (
        f'<div style="background:#e5e7eb;border-radius:4px;height:18px;width:100%">'
        f'<div style="background:{color};height:100%;width:{w:.1f}%;border-radius:4px;'
        f'min-width:2px"></div></div>'
    )


def _grade_badge(grade: str) -> str:
    c = _GRADE_COLORS.get(grade, "#6b7280")
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{c};color:#fff;font-weight:700;font-size:14px">{grade}</span>'
    )


def _scenario_row(r: ScenarioResult) -> str:
    sc = r.scorecard
    gc = _GRADE_COLORS.get(r.grade, "#6b7280")
    # Variant details
    variant_html = ""
    for ad in r.assertion_details:
        variants = ad.get("follow_up_variants", [])
        if len(variants) > 1:
            rows = ""
            for v in variants:
                name = html.escape(str(v.get("variant", "?")))
                rows += (
                    f"<tr><td>{name}</td>"
                    f"<td>{v.get('recall', 0):.1f}</td>"
                    f"<td>{v.get('precision', 0):.1f}</td>"
                    f"<td>{v.get('noise_rejection', 0):.1f}</td>"
                    f"<td>{'✅' if v.get('passed') else '❌'}</td></tr>"
                )
            variant_html += (
                f'<details style="margin-top:4px"><summary style="cursor:pointer;'
                f'font-size:12px;color:#6b7280">Agent variants for: '
                f"{html.escape(ad.get('query', ''))}</summary>"
                f'<table class="vtbl"><tr><th>Variant</th><th>Recall</th>'
                f"<th>Precision</th><th>Noise Rej.</th><th>Pass</th></tr>"
                f"{rows}</table></details>"
            )

    return (
        f"<tr>"
        f"<td><b>{html.escape(r.scenario_id)}</b><br>"
        f'<span style="font-size:12px;color:#6b7280">{html.escape(r.title)}</span></td>'
        f"<td>{r.difficulty}</td>"
        f"<td>{_bar(sc.mqs.recall)}<span class='n'>{sc.mqs.recall:.1f}</span></td>"
        f"<td>{_bar(sc.mqs.precision)}<span class='n'>{sc.mqs.precision:.1f}</span></td>"
        f"<td>{_bar(sc.mqs.noise_rejection, '#22c55e')}"
        f"<span class='n'>{sc.mqs.noise_rejection:.1f}</span></td>"
        f"<td style='text-align:center'><b style='color:{gc}'>{r.total_score:.1f}</b></td>"
        f"<td style='text-align:center'>{_grade_badge(r.grade)}</td>"
        f"</tr>"
        f"{'<tr><td colspan=7>' + variant_html + '</td></tr>' if variant_html else ''}"
    )


def _breakdown_chart(title: str, data: dict) -> str:  # type: ignore[type-arg]
    if not data:
        return ""
    rows = ""
    for k, v in sorted(data.items()):
        gc = _GRADE_COLORS.get(
            "S"
            if v >= 90
            else "A"
            if v >= 80
            else "B"
            if v >= 70
            else "C"
            if v >= 60
            else "D",
            "#6b7280",
        )
        rows += (
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            f'<span style="width:80px;font-size:13px">{html.escape(str(k))}</span>'
            f'<div style="flex:1">{_bar(v, gc)}</div>'
            f'<span style="width:40px;text-align:right;font-size:13px">{v:.1f}</span>'
            f"</div>"
        )
    return (
        f'<div style="margin:16px 0"><h3 style="margin:0 0 8px">{html.escape(title)}</h3>'
        f"{rows}</div>"
    )


def render_html_report(report: BenchmarkReport) -> str:
    scenario_rows = "\n".join(_scenario_row(r) for r in report.results)
    gc = _GRADE_COLORS.get(report.overall_grade, "#6b7280")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Memoria Benchmark — {html.escape(report.dataset_id)} {html.escape(report.version)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0 }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f8fafc; color: #1e293b; padding: 24px; max-width: 1200px; margin: 0 auto }}
  h1 {{ font-size: 24px; margin-bottom: 4px }}
  h2 {{ font-size: 18px; margin: 24px 0 12px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px }}
  h3 {{ font-size: 15px; color: #475569 }}
  .card {{ background: #fff; border-radius: 12px; padding: 20px; margin: 12px 0;
           box-shadow: 0 1px 3px rgba(0,0,0,.08) }}
  .hero {{ display: flex; align-items: center; gap: 24px; flex-wrap: wrap }}
  .hero-score {{ font-size: 48px; font-weight: 800; color: {gc} }}
  .hero-meta {{ font-size: 14px; color: #64748b; line-height: 1.8 }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px }}
  th {{ text-align: left; padding: 8px; border-bottom: 2px solid #e2e8f0; color: #64748b;
        font-size: 12px; text-transform: uppercase }}
  td {{ padding: 8px; border-bottom: 1px solid #f1f5f9; vertical-align: top }}
  .n {{ font-size: 11px; color: #94a3b8; float: right; margin-top: 2px }}
  .vtbl {{ font-size: 12px; margin: 4px 0 }}
  .vtbl th {{ font-size: 11px; padding: 4px 6px }}
  .vtbl td {{ padding: 4px 6px }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px }}
</style></head><body>

<h1>Memoria Benchmark Report</h1>
<p style="color:#64748b;margin-bottom:16px">
  {html.escape(report.dataset_id)} {html.escape(report.version)}
  &middot; {report.scenario_count} scenarios
</p>

<div class="card">
  <div class="hero">
    <div>
      <div class="hero-score">{report.overall_score:.1f}</div>
      <div>{_grade_badge(report.overall_grade)}</div>
    </div>
    <div class="hero-meta">
      Overall Memory Quality Score<br>
      {report.scenario_count} scenarios evaluated
    </div>
  </div>
</div>

<div class="grid">
  <div class="card">{_breakdown_chart("By Difficulty", report.by_difficulty)}</div>
  <div class="card">{_breakdown_chart("By Horizon", report.by_horizon)}</div>
  <div class="card">{_breakdown_chart("By Challenge Tag", report.by_tag)}</div>
</div>

<h2>Scenario Results</h2>
<div class="card" style="overflow-x:auto">
<table>
<tr>
  <th>Scenario</th><th>Diff</th><th>Recall</th><th>Precision</th>
  <th>Noise Rej.</th><th>Score</th><th>Grade</th>
</tr>
{scenario_rows}
</table>
</div>

<p style="text-align:center;color:#94a3b8;font-size:12px;margin-top:24px">
  Generated by Memoria Benchmark Framework
</p>
</body></html>"""
