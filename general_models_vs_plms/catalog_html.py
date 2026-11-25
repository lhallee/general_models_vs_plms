"""Generate a standalone interactive HTML explorer for the general_models_vs_plms compilation.

Reads the validated bundle emitted by ``catalog.build_catalog`` and writes one self-contained HTML file
(``docs/compilation.html``) with zero external runtime JavaScript dependencies so humans can filter claims, inspect
the 8-axis confounder matrix, and open full study dossiers offline or on GitHub Pages.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Compiled, confounder-audited evidence on general-purpose LLMs vs protein language models across all protein biology tasks.">
<title>General LLMs vs Protein Language Models - Audited Evidence Explorer</title>
<style>
:root {
  --bg: #f4f6f8; --surface: #ffffff; --ink: #17212b; --muted: #5a6775; --line: #d7dee6; --tint: #eaeff4;
  --a: #0c6676; --a-soft: #dff0f2; --b: #84540f; --b-soft: #f5ead6;
  --ok: #15663d; --ok-bg: #e1f4ea; --warn: #8a5600; --warn-bg: #fdf1d6;
  --bad: #9f2323; --bad-bg: #fde4e4; --na: #5a6775; --na-bg: #edf1f5;
  --body: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f151b; --surface: #172029; --ink: #e5ecf2; --muted: #9aa7b4; --line: #2a3541; --tint: #1e2933;
    --a: #62c6d3; --a-soft: #11353b; --b: #e2b262; --b-soft: #3a2d13;
    --ok: #6ee0a4; --ok-bg: #133524; --warn: #f2be63; --warn-bg: #3b2c10;
    --bad: #f28b8b; --bad-bg: #3f1919; --na: #9aa7b4; --na-bg: #222e3a;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.5; }
a { color: var(--a); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre { font-family: var(--mono); font-size: 0.88em; }
.wrap { max-width: 1380px; margin: 0 auto; padding: 24px clamp(16px, 3vw, 40px) 72px; }
header { display: grid; gap: 12px; margin-bottom: 20px; }
h1 { margin: 0; font-size: clamp(1.5rem, 2.4vw, 2.1rem); line-height: 1.2; }
h2 { margin: 0 0 10px; font-size: 1.2rem; }
h3 { margin: 0 0 8px; font-size: 1.02rem; }
.lede { color: var(--muted); max-width: 100ch; margin: 0; }
.banner {
  background: var(--a-soft); border: 1px solid var(--a); border-radius: 8px;
  padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; justify-content: space-between;
}
.banner strong { color: var(--ink); }
.banner-links { display: flex; flex-wrap: wrap; gap: 10px; }
.btn {
  display: inline-block; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line);
  background: var(--surface); color: var(--ink); font-size: 0.86rem; font-weight: 500; cursor: pointer;
}
.btn.primary { background: var(--a); color: #ffffff; border-color: var(--a); }
.btn.active { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 10px; margin: 16px 0; }
.kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
.kpi .num { font-size: 1.55rem; font-weight: 700; line-height: 1.15; }
.kpi .lbl { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 14px; border-bottom: 1px solid var(--line); padding-bottom: 10px; }
.filters {
  background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px;
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 16px;
}
.filters select, .filters input[type="search"] {
  background: var(--bg); color: var(--ink); border: 1px solid var(--line); border-radius: 6px;
  padding: 6px 10px; font: inherit; font-size: 0.86rem;
}
.filters input[type="search"] { min-width: 220px; flex: 1 1 220px; }
.toggle { display: inline-flex; align-items: center; gap: 6px; font-size: 0.86rem; cursor: pointer; user-select: none; }
.panel { display: none; }
.panel.active { display: block; }
.tablebox { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { background: var(--tint); font-weight: 600; position: sticky; top: 0; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
.pill {
  display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; white-space: nowrap;
}
.st-supported_controlled, .ax-controlled { background: var(--ok-bg); color: var(--ok); }
.st-suggestive_confounded, .ax-uncontrolled { background: var(--warn-bg); color: var(--warn); }
.st-unverified_leaked, .ax-evidenced_concern { background: var(--bad-bg); color: var(--bad); }
.st-queued, .ax-not_applicable, .ax-unread { background: var(--na-bg); color: var(--na); }
.study-link { font-weight: 600; cursor: pointer; color: var(--a); background: none; border: none; padding: 0; font: inherit; text-align: left; }
.study-link:hover { text-decoration: underline; }
.card-grid { display: grid; gap: 12px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
.meta { color: var(--muted); font-size: 0.82rem; margin: 4px 0 8px; }
.modal-backdrop {
  position: fixed; inset: 0; background: rgba(0, 0, 0, 0.55); display: none; align-items: center; justify-content: center;
  padding: 20px; z-index: 100;
}
.modal-backdrop.open { display: flex; }
.modal {
  background: var(--surface); color: var(--ink); border: 1px solid var(--line); border-radius: 10px;
  max-width: 960px; width: 100%; max-height: 88vh; overflow-y: auto; padding: 20px 24px; display: grid; gap: 14px;
}
.modal-head { display: flex; justify-content: space-between; align-items: start; gap: 12px; }
pre.cli { background: var(--tint); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; overflow-x: auto; margin: 0; }
.small { font-size: 0.8rem; color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>General-Purpose LLMs vs Protein Language Models: Confounder-Audited Evidence</h1>
    <p class="lede">
      Interactive compilation of empirical results across literature, lab technical reports, repositories, X probes, and our own controlled experiments.
      Every audited study is graded across eight explicit confounder and leakage axes backed by primary-text citations, separating strictly supported conclusions from confounded or leaked claims.
    </p>
    <div class="banner">
      <div>
        <strong>Looking specifically for our controlled FLIP2, FGFR2, and ProGen2 experiments?</strong>
        <span class="small">Jump directly to our step-by-step experiment cards and embedded figures.</span>
      </div>
      <div class="banner-links">
        <a class="btn primary" href="index.html">Our Experiment Explainer (M1-M14, B1-B2, A1-A6)</a>
        <a class="btn" href="https://github.com/lhallee/general_models_vs_plms/blob/main/RESULTS.md" target="_blank" rel="noopener">Full Tables (RESULTS.md)</a>
        <a class="btn" href="https://github.com/lhallee/general_models_vs_plms/blob/main/COMPILATION.md" target="_blank" rel="noopener">Markdown Index</a>
        <a class="btn" href="https://github.com/lhallee/general_models_vs_plms" target="_blank" rel="noopener">GitHub Repo</a>
      </div>
    </div>
  </header>

  <div class="kpis" id="kpis"></div>

  <div class="tabs" role="tablist">
    <button class="btn active" data-tab="claims">1. Filterable Claims &amp; Synthesis</button>
    <button class="btn" data-tab="matrix">2. 8-Axis Confounder Matrix</button>
    <button class="btn" data-tab="dossiers">3. Study Dossiers &amp; Queued</button>
    <button class="btn" data-tab="agents">4. Agent / CLI Access</button>
  </div>

  <div class="filters" id="filter-bar">
    <label class="toggle"><input type="checkbox" id="f-rigorous"> <strong>Supported (controlled) only</strong></label>
    <select id="f-task"><option value="">All tasks</option></select>
    <select id="f-regime"><option value="">All regimes</option></select>
    <select id="f-strength"><option value="">All evidence statuses</option></select>
    <select id="f-tier"><option value="">All source tiers</option></select>
    <label class="toggle"><input type="checkbox" id="f-wetlab"> Wet-lab only</label>
    <input type="search" id="f-search" placeholder="Search model, target, benchmark, author (e.g. GPT-6, Mythos, ESMC)...">
    <button class="btn" id="f-reset">Reset</button>
  </div>

  <section class="panel active" id="tab-claims">
    <p class="small" id="claims-count"></p>
    <div class="tablebox" style="margin-top:8px;">
      <table>
        <thead>
          <tr>
            <th>Evidence</th>
            <th>Task</th>
            <th>Study</th>
            <th>Regime</th>
            <th>Benchmark</th>
            <th>General LLM</th>
            <th>Protein Baseline</th>
            <th>Notes &amp; Confounders</th>
          </tr>
        </thead>
        <tbody id="claims-body"></tbody>
      </table>
    </div>
  </section>

  <section class="panel" id="tab-matrix">
    <p class="small">Click any study name or cell to open verbatim primary-text evidence for all eight axes. Unread queued sources are excluded from this table.</p>
    <div class="tablebox" style="margin-top:8px;">
      <table>
        <thead>
          <tr>
            <th>Study</th>
            <th>Cutoff</th>
            <th>Homology</th>
            <th>Lit. Recall</th>
            <th>Artifacts</th>
            <th>Oracle Filter</th>
            <th>Compute / Tools</th>
            <th>Baselines</th>
            <th>Data / Wet-Lab</th>
          </tr>
        </thead>
        <tbody id="matrix-body"></tbody>
      </table>
    </div>
  </section>

  <section class="panel" id="tab-dossiers">
    <div class="card-grid" id="dossiers-list"></div>
  </section>

  <section class="panel" id="tab-agents">
    <div class="card-grid">
      <div class="card">
        <h3>Machine-Readable Catalog Files</h3>
        <p>Agents and scripts do not need to scrape Markdown or HTML. Read any of these generated files directly:</p>
        <ul>
          <li><code>catalog/index.json</code> - complete schema, task-level rigorous synthesis, all studies, claims, and verbatim 8-axis audits</li>
          <li><code>catalog/claims.csv</code> - flat claim table with all 8 confounder statuses as columns</li>
          <li><code>catalog/studies.jsonl</code> - one validated study JSON record per line</li>
          <li><code>catalog/studies/&lt;id&gt;.json</code> - canonical per-study source files</li>
          <li><code>studies/&lt;id&gt;.md</code> - human-readable Markdown dossier per study</li>
        </ul>
      </div>
      <div class="card">
        <h3>CLI Queries</h3>
        <pre class="cli"># Validate all study JSON records and rebuild every Markdown, CSV, JSON, and HTML output
py -X utf8 -m general_models_vs_plms.catalog build

# List strictly supported (controlled) claims across all tasks
py -X utf8 -m general_models_vs_plms.catalog query --rigorous-only

# Filter by task, model, or confounder concern as JSON
py -X utf8 -m general_models_vs_plms.catalog query --task inverse_folding --json
py -X utf8 -m general_models_vs_plms.catalog query --model GPT-6 --axis-concern cross_modal_literature_recall --json</pre>
      </div>
    </div>
  </section>
</div>

<div class="modal-backdrop" id="modal-bg" role="dialog" aria-modal="true">
  <div class="modal">
    <div class="modal-head">
      <div>
        <h2 id="modal-title"></h2>
        <div class="meta" id="modal-meta"></div>
      </div>
      <button class="btn" id="modal-close">Close</button>
    </div>
    <div id="modal-body"></div>
  </div>
</div>

<script type="application/json" id="catalog-data">__CATALOG_JSON__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("catalog-data").textContent);
  const V = DATA.vocabularies;
  const STUDIES = new Map(DATA.studies.map((s) => [s.id, s]));
  const AXES = Object.keys(V.confounder_axes);
  const AX_SHORT = { controlled: "Controlled", uncontrolled: "Uncontrolled", evidenced_concern: "Concern", not_applicable: "N/A" };

  function esc(str) {
    return String(str ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // KPIs
  const c = DATA.counts;
  const kpiItems = [
    [c.studies_audited, "Audited studies (full text / repo)"],
    [c.claims_supported_controlled, "Supported (controlled) claims"],
    [c.claims_suggestive_confounded, "Suggestive (confounded) claims"],
    [c.claims_unverified_leaked, "Unverified / likely leaked claims"],
    [c.studies_queued, "Queued unread (no critiques)"],
  ];
  document.getElementById("kpis").innerHTML = kpiItems
    .map(([num, lbl]) => `<div class="kpi"><div class="num">${esc(num)}</div><div class="lbl">${esc(lbl)}</div></div>`)
    .join("");

  // Populate dropdowns
  function fillSelect(id, map) {
    const el = document.getElementById(id);
    for (const [k, label] of Object.entries(map)) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = label;
      el.appendChild(opt);
    }
  }
  fillSelect("f-task", V.tasks);
  fillSelect("f-regime", V.regimes);
  fillSelect("f-strength", V.evidence_strengths);
  fillSelect("f-tier", V.source_tiers);

  // Tabs
  document.querySelectorAll(".tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  });

  // Modal
  const modalBg = document.getElementById("modal-bg");
  function openStudy(id) {
    const s = STUDIES.get(id);
    if (!s) return;
    document.getElementById("modal-title").textContent = s.title;
    document.getElementById("modal-meta").innerHTML =
      `${esc(s.authors)} &middot; ${esc(s.date)} &middot; <code>${esc(s.source_tier)}</code> &middot; <code>${esc(s.read_depth)}</code> &middot; ` +
      s.urls.map((u) => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(u)}</a>`).join(" &middot; ");
    let html = `<div class="card"><h3>Takeaway</h3><p>${esc(s.takeaway)}</p></div>`;
    html += `<div class="card"><h3>Methods</h3><ul>
      <li><strong>Datasets:</strong> ${esc(s.methods.datasets)}</li>
      <li><strong>Protocol:</strong> ${esc(s.methods.evaluation_protocol)}</li>
      <li><strong>Controls:</strong> ${esc(s.methods.controls_included)}</li>
      <li><strong>Compute &amp; Tools:</strong> ${esc(s.methods.compute_and_tools)}</li>
      <li><strong>Metrics:</strong> ${esc(s.methods.metrics)}</li>
      <li><strong>Wet-Lab:</strong> ${esc(s.methods.wet_lab_protocol)}</li>
      <li><strong>Code &amp; Data:</strong> ${esc(s.code_and_data_released)}</li>
    </ul></div>`;
    if (s.read_depth !== "queued_unread") {
      html += `<div class="card"><h3>8-Axis Confounder and Leakage Audit</h3><div class="tablebox"><table><thead><tr><th>Axis</th><th>Status</th><th>Verbatim / Primary-Text Evidence</th></tr></thead><tbody>` +
        AXES.map((ax) => {
          const a = s.confounder_audit[ax];
          return `<tr><td><strong>${esc(V.confounder_axes[ax])}</strong></td><td><span class="pill ax-${esc(a.status)}">${esc(AX_SHORT[a.status])}</span></td><td>${esc(a.evidence)}</td></tr>`;
        }).join("") + `</tbody></table></div></div>`;
    } else {
      html += `<div class="card"><h3>Audit Status</h3><p>Queued unread in full. No confounder critique is assigned until full primary text is verified.</p></div>`;
    }
    if (s.critiques_and_commentary && s.critiques_and_commentary.length) {
      html += `<div class="card"><h3>Attached Critiques &amp; Commentary</h3><ul>` +
        s.critiques_and_commentary.map((cr) =>
          `<li><strong>${esc(cr.author)}</strong> (${esc(cr.date)}, ${esc(cr.source_type)}): <a href="${esc(cr.url)}" target="_blank" rel="noopener">${esc(cr.url)}</a> - ${esc(cr.summary)}</li>`
        ).join("") + `</ul></div>`;
    }
    document.getElementById("modal-body").innerHTML = html;
    modalBg.classList.add("open");
  }
  document.getElementById("modal-close").addEventListener("click", () => modalBg.classList.remove("open"));
  modalBg.addEventListener("click", (e) => { if (e.target === modalBg) modalBg.classList.remove("open"); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") modalBg.classList.remove("open"); });

  function getFilters() {
    return {
      rigorous: document.getElementById("f-rigorous").checked,
      task: document.getElementById("f-task").value,
      regime: document.getElementById("f-regime").value,
      strength: document.getElementById("f-strength").value,
      tier: document.getElementById("f-tier").value,
      wetlab: document.getElementById("f-wetlab").checked,
      q: document.getElementById("f-search").value.trim().toLowerCase(),
    };
  }

  function render() {
    const f = getFilters();
    const matchedClaims = DATA.claims.filter((row) => {
      if (f.rigorous && row.evidence_strength !== "supported_controlled") return false;
      if (!f.rigorous && f.strength && row.evidence_strength !== f.strength) return false;
      if (f.task && row.task !== f.task) return false;
      if (f.regime && row.regime !== f.regime) return false;
      if (f.tier && row.source_tier !== f.tier) return false;
      if (f.wetlab && !row.wet_lab) return false;
      if (f.q) {
        const hay = [row.study_label, row.study_title, row.authors, row.benchmark, row.llm_model, row.llm_score, row.baseline_model, row.baseline_score, row.notes].join(" ").toLowerCase();
        if (!hay.includes(f.q)) return false;
      }
      return true;
    });

    document.getElementById("claims-count").textContent = `Showing ${matchedClaims.length} of ${DATA.claims.length} indexed claims (generated ${DATA.generated_utc}). Click any study label to open its full dossier.`;
    document.getElementById("claims-body").innerHTML = matchedClaims.map((r) => `
      <tr>
        <td><span class="pill st-${esc(r.evidence_strength)}">${esc(r.evidence_strength_label)}</span></td>
        <td>${esc(r.task_label)}</td>
        <td><button class="study-link" data-study="${esc(r.study_id)}">${esc(r.study_label)}</button><div class="small">${esc(r.date)} &middot; ${esc(r.source_tier)}</div></td>
        <td><span class="small">${esc(r.regime_label)}</span></td>
        <td>${esc(r.benchmark)}</td>
        <td><strong>${esc(r.llm_model)}</strong><div class="small">${esc(r.llm_score)}</div></td>
        <td>${esc(r.baseline_model)}<div class="small">${esc(r.baseline_score)}</div></td>
        <td class="small">${esc(r.notes)}</td>
      </tr>`).join("");

    const matchedStudies = DATA.studies.filter((s) => {
      if (f.task && !s.tasks.includes(f.task)) return false;
      if (f.regime && !s.regimes.includes(f.regime)) return false;
      if (f.tier && s.source_tier !== f.tier) return false;
      if (f.wetlab && !s.wet_lab_validated) return false;
      if (f.rigorous && !s.key_results.some((kr) => kr.evidence_strength === "supported_controlled")) return false;
      if (f.q) {
        const hay = [s.short_label, s.title, s.authors, s.takeaway, ...s.models_tested, ...s.plm_baselines].join(" ").toLowerCase();
        if (!hay.includes(f.q)) return false;
      }
      return true;
    });

    const auditedStudies = matchedStudies.filter((s) => s.read_depth !== "queued_unread");
    document.getElementById("matrix-body").innerHTML = auditedStudies.map((s) => `
      <tr>
        <td><button class="study-link" data-study="${esc(s.id)}">${esc(s.short_label)}</button><div class="small">${esc(s.date)}</div></td>
        ${AXES.map((ax) => {
          const st = s.confounder_audit[ax].status;
          return `<td><button class="study-link" data-study="${esc(s.id)}" title="${esc(s.confounder_audit[ax].evidence)}"><span class="pill ax-${esc(st)}">${esc(AX_SHORT[st])}</span></button></td>`;
        }).join("")}
      </tr>`).join("");

    document.getElementById("dossiers-list").innerHTML = matchedStudies.map((s) => `
      <div class="card">
        <div style="display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; align-items:baseline;">
          <h3><button class="study-link" data-study="${esc(s.id)}">${esc(s.title)}</button></h3>
          <span class="small"><code>${esc(s.source_tier)}</code> &middot; <code>${esc(s.read_depth)}</code> &middot; ${esc(s.date)}</span>
        </div>
        <div class="meta">${esc(s.authors)} &middot; Models: ${esc(s.models_tested.slice(0, 5).join(", "))}${s.models_tested.length > 5 ? " ..." : ""}</div>
        <p>${esc(s.takeaway)}</p>
        <div style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
          <button class="btn" data-study="${esc(s.id)}">Open Full Methods &amp; 8-Axis Audit</button>
          <a class="small" href="https://github.com/lhallee/general_models_vs_plms/blob/main/studies/${esc(s.id)}.md" target="_blank" rel="noopener">Markdown dossier</a>
          <a class="small" href="${esc(s.urls[0])}" target="_blank" rel="noopener">Primary source</a>
        </div>
      </div>`).join("");

    document.querySelectorAll("[data-study]").forEach((el) => {
      el.addEventListener("click", () => openStudy(el.dataset.study));
    });
  }

  ["f-rigorous", "f-task", "f-regime", "f-strength", "f-tier", "f-wetlab", "f-search"].forEach((id) => {
    document.getElementById(id).addEventListener("input", render);
    document.getElementById(id).addEventListener("change", render);
  });
  document.getElementById("f-reset").addEventListener("click", () => {
    document.getElementById("f-rigorous").checked = false;
    document.getElementById("f-wetlab").checked = false;
    ["f-task", "f-regime", "f-strength", "f-tier", "f-search"].forEach((id) => { document.getElementById(id).value = ""; });
    render();
  });

  render();
})();
</script>
</body>
</html>
"""


def write_compilation_html(bundle: dict[str, Any], out_path: Path) -> Path:
    """Embed the validated catalog JSON into a standalone interactive HTML explorer."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Escape closing script tags defensively when embedding JSON in HTML.
    embedded = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__CATALOG_JSON__", embedded)
    out_path.write_text(html, encoding="utf-8")
    return out_path
