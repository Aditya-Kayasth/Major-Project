"use strict";

const $ = (sel) => document.querySelector(sel);

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/* Minimal, safe markdown: escapes first, then headings, rules, tables, bold/italic, lists, paragraphs. */
function renderMarkdown(md) {
  const inline = (s) => s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // [Precedent N] / [Precedents 1 and 3] / [Precedents 1, 2] / [Precedent N, <any label>] (server-normalised,
    // e.g. "Para 55-56", "Headnote"): each precedent number links to its card (rank N = Precedent N); the label
    // is shown as written, not linked
    .replace(/\[(Precedents?) (\d+(?:\s*(?:,|and|&amp;|-)\s*\d+)*)(, [^\]<\n]*)?\]/gi, (_, word, nums, label) =>
      `<span class="cite">[${word} ${nums.replace(/\d+/g, (n) => `<a href="#judgment-${n}">${n}</a>`)}${label || ""}]</span>`);
  const lines = escapeHtml(md).split(/\r?\n/);
  let html = "", inList = false, para = [], table = [];
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  const cells = (row) => row.replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => inline(c.trim()));
  const isSeparator = (row) => /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/.test(row);
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !isSeparator(r));
    const [head, ...body] = rows;
    html += `<div class="table-wrap"><table class="md-table"><thead><tr>${cells(head).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
      body.map((r) => `<tr>${cells(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
    table = [];
  };
  for (const raw of lines) {
    const line = raw.trim();
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    const li = line.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
    if (/^\|.*\|$/.test(line)) { flushPara(); closeList(); table.push(line); continue; }
    flushTable();
    if (!line) { flushPara(); closeList(); }
    else if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) { flushPara(); closeList(); html += "<hr>"; }
    else if (h) { flushPara(); closeList(); html += `<h4>${inline(h[2])}</h4>`; }
    else if (li) { flushPara(); if (!inList) { html += "<ul>"; inList = true; } html += `<li>${inline(li[1])}</li>`; }
    else { closeList(); para.push(line); }
  }
  flushTable(); flushPara(); closeList();
  return html;
}

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* non-JSON error */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

/* ---------------------------------------------------------------- tabs */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${btn.dataset.tab}`));
    if (btn.dataset.tab === "evaluation") loadEvaluation();
  });
});

/* ---------------------------------------------------------------- status */
async function loadStatus() {
  try {
    const s = await api("api/status");
    $("#status").innerHTML = [
      `<span class="chip ok"><b>${s.judgments_indexed.toLocaleString()}</b> judgments</span>`,
      `<span class="chip"><b>${s.chunks_indexed.toLocaleString()}</b> passages</span>`,
      `<span class="chip">embed: <b>${escapeHtml(s.embedding_model.split("/").pop())}</b></span>`,
      `<span class="chip">rerank: <b>${escapeHtml(s.reranker_model.split("/").pop())}</b></span>`,
      `<span class="chip ${s.llm_available ? "ok" : "warn"}" title="${escapeHtml(s.llm_route)}">LLM: <b>${s.llm_available ? escapeHtml(s.llm_route.split(" -> ")[0]) : "not configured"}</b></span>`,
    ].join("");
    $("#p-embed").textContent = s.embedding_model;
    $("#p-rerank").textContent = s.reranker_model;
    if (!s.llm_available) { $("#gen-answer").checked = false; $("#gen-answer").disabled = true; }
  } catch (err) {
    $("#status").innerHTML = `<span class="chip warn">Backend offline: ${escapeHtml(err.message)}</span>`;
  }
}

/* ---------------------------------------------------------------- search */
const STAGES = [
  ["metadata_filtering_sec", "Filters"], ["dense_retrieval_sec", "Dense"], ["bm25_retrieval_sec", "BM25"],
  ["rrf_fusion_sec", "Fusion"], ["reranking_sec", "Rerank"], ["parent_expansion_sec", "Passages"],
  ["llm_generation_sec", "LLM"],
];
const ms = (sec) => (sec >= 1 ? `${sec.toFixed(1)} s` : `${Math.round(sec * 1000)} ms`);

function renderTimingLine(t) {
  const parts = STAGES.filter(([k]) => typeof t[k] === "number").map(([k, label]) => `${label} ${ms(t[k])}`);
  const total = t.total_latency_sec ?? t.funnel_latency_sec;
  $("#timing-line").textContent = parts.join(" · ") + (typeof total === "number" ? ` · Total ${ms(total)}` : "");
}

function renderFlags(data, query) {
  const flags = [];
  if (data.low_confidence) {
    flags.push(`<span class="flag critical" role="status">⚠ LOW CONFIDENCE: best relevance ${data.best_relevance?.toFixed(3)} &lt; ${data.min_relevance}. ${data.answer ? "No answer generated." : "An answer would be refused."}</span>`);
  }
  const cc = data.citation_check;
  if (cc && !cc.refused) {
    const unknown = cc.unknown_precedent_numbers.length + cc.unknown_external_citations.length;
    const good = unknown === 0 && cc.citation_precision !== null;
    flags.push(`<span class="flag ${good ? "good" : "warn"}" title="${escapeHtml(JSON.stringify(cc))}">${good ? "✓" : "!"} Citation check: precision ${cc.citation_precision ?? "n/a"} · ${unknown} unknown · cited ${cc.cited_precedents.length}/${cc.n_precedents}${cc.ends_with_verify ? "" : " · missing 'Verify on the official source'"}</span>`);
  } else if (cc && cc.refused) {
    flags.push(`<span class="flag neutral">Refused: insufficient precedent in the indexed corpus</span>`);
  }
  if (data.expanded_query && data.expanded_query.trim() !== query.trim()) {
    flags.push(`<span class="flag neutral">Concordance: ${escapeHtml(data.expanded_query.slice(query.length).trim())}</span>`);
  }
  const f = data.filters || {};
  if (f.applied && Object.keys(f.applied).length) {
    flags.push(`<span class="flag neutral">Filters ${escapeHtml(JSON.stringify(f.applied))} → ${f.matching_docs ?? "?"} judgments</span>`);
  }
  $("#flags").innerHTML = flags.join("");
}

function chip(label, value, title) {
  if (!value && value !== 0) return "";
  return `<span class="mchip" ${title ? `title="${escapeHtml(title)}"` : ""}><span class="mchip-k">${label}</span> ${escapeHtml(value)}</span>`;
}

function renderResults(results) {
  if (!results.length) { $("#results").innerHTML = `<div class="card"><p class="note">No judgments retrieved.</p></div>`; return; }
  $("#results").innerHTML = results.map((r) => {
    const statutes = (r.statutes_cited || "").split("; ").filter(Boolean);
    const shown = statutes.slice(0, 6).join("; ") + (statutes.length > 6 ? ` +${statutes.length - 6}` : "");
    const url = safePdfUrl(r.pdf_url);
    const title = url
      ? `<a class="title-link" href="${url}" target="_blank" rel="noopener">${escapeHtml(r.case_title)}</a>`
      : escapeHtml(r.case_title);
    return `
    <article class="card judgment" id="judgment-${r.rank}">
      <div class="rank">${r.rank}</div>
      <div>
        <div class="result-title">${title}</div>
        <div class="mchips">
          ${chip("Citation", r.citation)}${chip("Date", r.decision_date)}
          ${chip("Bench", r.bench_label, r.judges)}${chip("Author", r.author_judge, r.author_source ? `source: ${r.author_source}` : "")}
          ${chip("Disposal", r.disposal_nature)}${chip("Statutes", shown, statutes.join("; "))}
          ${url ? `<a class="pdf-btn" href="${url}" target="_blank" rel="noopener">Open official PDF</a>` : ""}
        </div>
        <div class="note">Relevance ${r.relevance.toFixed(3)} · Judges: ${escapeHtml(r.judges)}</div>
        ${r.passages.map((p, i) => `
          <div class="passage">
            <span class="role ${p.section_kind === "headnote" ? "headnote" : ""}">${escapeHtml(p.role)}${p.section_kind === "headnote" ? " (editorial)" : ""}</span>
            <span class="note">passage ${i + 1}</span>
            <div class="snippet">${escapeHtml(p.text)}</div>
          </div>`).join("")}
      </div>
    </article>`;
  }).join("");
}

/* Official PDF links: only URLs in the public AWS Open Data bucket are rendered (always via escapeHtml). */
const PDF_PREFIX = "https://indian-supreme-court-judgments.s3.amazonaws.com/";
const safePdfUrl = (u) => (typeof u === "string" && u.startsWith(PDF_PREFIX) ? escapeHtml(u) : "");

/* Sources under the answer: one line per cited precedent (rank N = Precedent N). */
function renderSources(cc, results) {
  const nums = (cc && cc.valid_precedents) || [];
  const rows = nums.map((n) => {
    const r = results.find((x) => x.rank === n);
    if (!r) return "";
    const href = safePdfUrl(r.pdf_url);
    const pdf = href ? ` · <a href="${href}" target="_blank" rel="noopener">Open PDF</a>` : "";
    return `<li><a href="#judgment-${n}">[${n}]</a> ${escapeHtml(r.case_title)}${r.citation ? `, ${escapeHtml(r.citation)}` : ""}${pdf}</li>`;
  }).filter(Boolean);
  return rows.length ? `<div class="sources"><h4>Sources</h4><ul class="sources-list">${rows.join("")}</ul></div>` : "";
}

$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#query").value.trim();
  if (query.length < 3) return;
  const btn = $("#search-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Searching…`;
  $("#search-error").classList.add("hidden");
  const num = (id) => ($(id).value === "" ? null : Number($(id).value));
  try {
    const data = await api("api/search", {
      method: "POST",
      body: JSON.stringify({
        query, top_k: Number($("#topk").value), generate_answer: $("#gen-answer").checked,
        year_from: num("#year-from"), year_to: num("#year-to"), min_bench: num("#min-bench"),
      }),
    });
    renderFlags(data, query);
    renderTimingLine(data.timings || {});
    renderResults(data.results || []);
    if (data.answer) {
      $("#answer").innerHTML = renderMarkdown(data.answer) + renderSources(data.citation_check, data.results || []);
      $("#answer-card").classList.remove("hidden");
    } else { $("#answer-card").classList.add("hidden"); }
    $("#results-area").classList.remove("hidden");
  } catch (err) {
    $("#search-error").textContent = `Search failed: ${err.message}`;
    $("#search-error").classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Search";
  }
});

/* ---------------------------------------------------------------- evaluation charts (plain SVG) */
const CONFIG_ORDER = ["dense", "bm25", "hybrid_rrf", "full", "full_no_prior"];
const SVG_NS = "http://www.w3.org/2000/svg";
const color = (cfg) => `var(--series-${CONFIG_ORDER.indexOf(cfg) + 1})`;

function svgEl(tag, attrs = {}, parent) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

/* Bar with a 4px rounded data-end and a square base (vertical or horizontal). */
function barPath(x, y, w, h, horizontal) {
  const r = Math.min(4, horizontal ? h / 2 : w / 2, horizontal ? w : h);
  if (w <= 0 || h <= 0) return "";
  if (horizontal) return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

const tip = () => $("#tooltip");
function hover(el, html) {
  el.addEventListener("mousemove", (ev) => {
    const t = tip();
    t.innerHTML = html;
    t.classList.remove("hidden");
    t.style.left = `${ev.clientX + 14}px`;
    t.style.top = `${ev.clientY + 12}px`;
  });
  el.addEventListener("mouseleave", () => tip().classList.add("hidden"));
}

function axisY(svg, x0, x1, yScale, ticks, fmt) {
  for (const t of ticks) {
    const y = yScale(t);
    svgEl("line", { x1: x0, x2: x1, y1: y, y2: y, class: "grid" }, svg);
    svgEl("text", { x: x0 - 6, y: y + 4, class: "tick", "text-anchor": "end" }, svg).textContent = fmt(t);
  }
}

function groupedBars(el, categories, series, { yMax = 1, fmt = (v) => v.toFixed(2), catLabel = (c) => c } = {}) {
  const W = 520, H = 250, m = { l: 40, r: 8, t: 10, b: 34 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const y = (v) => m.t + ph - (ph * v) / yMax;
  axisY(svg, m.l, W - m.r, y, [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax), fmt);
  const gw = pw / categories.length, inner = gw * 0.78, bw = (inner - 2 * (series.length - 1)) / series.length;
  categories.forEach((c, ci) => {
    const gx = m.l + ci * gw + (gw - inner) / 2;
    series.forEach((s, si) => {
      const v = s.values[ci];
      if (v === null || v === undefined) return;
      const x = gx + si * (bw + 2), top = y(v);
      const hit = svgEl("rect", { x: x - 1, y: m.t, width: bw + 2, height: ph, fill: "transparent" }, svg);
      svgEl("path", { d: barPath(x, top, bw, m.t + ph - top, false), fill: s.color }, svg);
      hover(hit, `<b>${escapeHtml(s.name)}</b><br>${escapeHtml(catLabel(c))}: ${fmt(v)}`);
    });
    svgEl("text", { x: m.l + ci * gw + gw / 2, y: H - 12, class: "tick", "text-anchor": "middle" }, svg).textContent = catLabel(c);
  });
  el.replaceChildren(svg);
}

function ciBars(el, rows, { xMax = 1, fmt = (v) => v.toFixed(3) } = {}) {
  const W = 520, rowH = 34, m = { l: 190, r: 70, t: 6, b: 24 };
  const H = m.t + m.b + rows.length * rowH;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const pw = W - m.l - m.r, x = (v) => m.l + (pw * v) / xMax;
  for (const t of [0, 0.25, 0.5, 0.75, 1].map((f) => f * xMax)) {
    svgEl("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "grid" }, svg);
    svgEl("text", { x: x(t), y: H - 8, class: "tick", "text-anchor": "middle" }, svg).textContent = fmt(t);
  }
  rows.forEach((r, i) => {
    const cy = m.t + i * rowH + rowH / 2, bh = 12;
    svgEl("text", { x: m.l - 8, y: cy + 4, class: "label", "text-anchor": "end" }, svg).textContent = r.name;
    svgEl("path", { d: barPath(m.l, cy - bh / 2, x(r.value) - m.l, bh, true), fill: r.color }, svg);
    if (r.ci) {
      const [lo, hi] = r.ci;
      svgEl("line", { x1: x(lo), x2: x(hi), y1: cy, y2: cy, class: "whisker" }, svg);
      for (const v of [lo, hi]) svgEl("line", { x1: x(v), x2: x(v), y1: cy - 7, y2: cy + 7, class: "whisker" }, svg);
    }
    svgEl("text", { x: W - m.r + 6, y: cy + 4, class: "value" }, svg).textContent = fmt(r.value);
    const hit = svgEl("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent" }, svg);
    hover(hit, `<b>${escapeHtml(r.name)}</b><br>${r.tip}`);
  });
  el.replaceChildren(svg);
}

function latencyBars(el, rows) {
  const W = 520, rowH = 40, m = { l: 190, r: 70, t: 6, b: 24 };
  const H = m.t + m.b + rows.length * rowH;
  const xMax = Math.max(...rows.map((r) => r.p95 || 0), 1);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const pw = W - m.l - m.r, x = (v) => m.l + (pw * v) / xMax;
  for (const t of [0, 0.25, 0.5, 0.75, 1].map((f) => f * xMax)) {
    svgEl("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "grid" }, svg);
    svgEl("text", { x: x(t), y: H - 8, class: "tick", "text-anchor": "middle" }, svg).textContent = Math.round(t);
  }
  rows.forEach((r, i) => {
    const top = m.t + i * rowH + 6, bh = 11;
    svgEl("text", { x: m.l - 8, y: top + bh + 4, class: "label", "text-anchor": "end" }, svg).textContent = r.name;
    svgEl("path", { d: barPath(m.l, top, x(r.p50) - m.l, bh, true), fill: r.color }, svg);
    svgEl("path", { d: barPath(m.l, top + bh + 2, x(r.p95) - m.l, bh, true), fill: r.color, class: "p95" }, svg);
    svgEl("text", { x: W - m.r + 6, y: top + bh + 4, class: "value" }, svg).textContent = `${Math.round(r.p50)} ms`;
    const hit = svgEl("rect", { x: 0, y: top - 4, width: W, height: rowH, fill: "transparent" }, svg);
    hover(hit, `<b>${escapeHtml(r.name)}</b><br>p50 ${Math.round(r.p50)} ms · p95 ${Math.round(r.p95)} ms`);
  });
  el.replaceChildren(svg);
}

let evaluationLoaded = false;
async function loadEvaluation() {
  if (evaluationLoaded) return;
  try {
    const R = await api("api/results");
    evaluationLoaded = true;
    const cfgs = CONFIG_ORDER.filter((c) => R.configs[c]);
    const name = (c) => R.configs[c].label;
    const run = R.run || {};
    $("#eval-meta").textContent = `Run ${run.timestamp} · commit ${run.git_commit} · ${run.device} · ` +
      `${run.corpus?.judgments?.toLocaleString()} judgments, ${run.corpus?.chunks?.toLocaleString()} passages · ` +
      `queries ${JSON.stringify(run.queries?.counts || {})} · judgment-level ranking, k = 10, 95% bootstrap CIs (1,000 resamples).`;
    $("#eval-legend").innerHTML = cfgs.map((c) =>
      `<span class="legend-item"><span class="swatch" style="background:${color(c)}"></span>${escapeHtml(name(c))}</span>`).join("") +
      `<span class="legend-item note">latency: upper bar p50, lower lighter bar p95</span>`;

    const ks = ["hit@1", "hit@3", "hit@5", "hit@10"];
    groupedBars($("#chart-hitk"), ks, cfgs.map((c) => ({ name: name(c), color: color(c), values: ks.map((k) => R.configs[c].overall[k].value) })),
      { catLabel: (k) => k.replace("hit@", "Hit@") });
    ciBars($("#chart-mrr"), cfgs.map((c) => {
      const o = R.configs[c].overall["mrr@10"];
      return { name: name(c), color: color(c), value: o.value, ci: o.ci, tip: `MRR@10 ${o.value.toFixed(3)}<br>95% CI [${o.ci[0].toFixed(3)}, ${o.ci[1].toFixed(3)}]` };
    }));
    latencyBars($("#chart-latency"), cfgs.map((c) => ({ name: name(c), color: color(c), ...R.configs[c].latency_ms })));
    const types = Object.keys(R.configs[cfgs[0]].by_type);
    groupedBars($("#chart-types"), types, cfgs.map((c) => ({ name: name(c), color: color(c), values: types.map((t) => R.configs[c].by_type[t]["hit@5"]) })),
      { catLabel: (t) => `${t.replace("_", " ")} (n=${R.configs[cfgs[0]].by_type[t].n})` });

    const metrics = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10", "recall@10"];
    let html = `<thead><tr><th>Config</th>${metrics.map((m) => `<th class="num">${m}</th>`).join("")}<th class="num">p50 ms</th><th class="num">p95 ms</th></tr></thead><tbody>`;
    for (const c of cfgs) {
      const o = R.configs[c].overall, l = R.configs[c].latency_ms;
      html += `<tr><td>${escapeHtml(name(c))}</td>${metrics.map((m) => `<td class="num" title="95% CI [${o[m].ci.join(", ")}]">${o[m].value.toFixed(3)}</td>`).join("")}<td class="num">${Math.round(l.p50)}</td><td class="num">${Math.round(l.p95)}</td></tr>`;
    }
    for (const t of types) {
      html += `<tr class="sub"><td colspan="${metrics.length + 3}">${escapeHtml(t)} (n=${R.configs[cfgs[0]].by_type[t].n})</td></tr>`;
      for (const c of cfgs) {
        const o = R.configs[c].by_type[t];
        html += `<tr><td class="indent">${escapeHtml(name(c))}</td>${metrics.map((m) => `<td class="num">${o[m].toFixed(3)}</td>`).join("")}<td></td><td></td></tr>`;
      }
    }
    $("#eval-table").innerHTML = html + "</tbody>";

    const cal = R.calibration || {};
    $("#sweep-table").innerHTML = `<thead><tr><th class="num">Threshold</th><th class="num">Out-of-scope refused</th><th class="num">In-scope refused</th><th class="num">Refused among Hit@1</th></tr></thead><tbody>` +
      (cal.sweep || []).map((s) => `<tr class="${s.threshold === cal.current_min_relevance ? "current" : ""}"><td class="num">${s.threshold}</td><td class="num">${s.oos_refused}</td><td class="num">${s.in_scope_refused}</td><td class="num">${s.hit1_refused}</td></tr>`).join("") +
      `</tbody>`;
    const g = R.generation || {};
    $("#gen-summary").innerHTML = g.status !== "run" ? escapeHtml(g.status || "not run") :
      `<p>${g.n_answers} answers by ${escapeHtml(g.answer_model)}; ${g.n_judged} judged by ${escapeHtml(g.judge)}.</p>
       <p>Judge means (0–10): ${g.judge_means ? Object.entries(g.judge_means).map(([k, v]) => `${escapeHtml(k)} ${v}`).join(" · ") : "not available"}</p>
       <p>Citation precision ${g.citation_precision_mean ?? "n/a"} · answers with unknown citations ${g.answers_with_unknown_citations} · ends with "Verify…" ${g.ends_with_verify}/${g.n_answers} · out-of-scope refusal rate ${g.oos_refusal_rate}</p>`;
    $("#eval-area").classList.remove("hidden");
    $("#eval-error").classList.add("hidden");
  } catch (err) {
    $("#eval-error").textContent = `Could not load results.json: ${err.message}`;
    $("#eval-error").classList.remove("hidden");
  }
}

loadStatus();
