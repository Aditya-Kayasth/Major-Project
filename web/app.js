"use strict";

const $ = (sel) => document.querySelector(sel);

const STAGE_LABELS = {
  metadata_filtering_sec: "Metadata slicing",
  dense_retrieval_sec: "Dense retrieval",
  bm25_retrieval_sec: "BM25 retrieval",
  rrf_fusion_sec: "Rank fusion (RRF)",
  reranking_sec: "Cross-encoder rerank",
  parent_expansion_sec: "Parent expansion",
  llm_generation_sec: "LLM synthesis",
};

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/* Minimal, safe markdown: escapes first, then headings, bold/italic, lists, paragraphs. */
function renderMarkdown(md) {
  const inline = (s) => s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  const lines = escapeHtml(md).split(/\r?\n/);
  let html = "", inList = false, para = [];
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of lines) {
    const line = raw.trim();
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    const li = line.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
    if (!line) { flushPara(); closeList(); }
    else if (h) { flushPara(); closeList(); html += `<h4>${inline(h[2])}</h4>`; }
    else if (li) { flushPara(); if (!inList) { html += "<ul>"; inList = true; } html += `<li>${inline(li[1])}</li>`; }
    else { closeList(); para.push(line); }
  }
  flushPara(); closeList();
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
    if (btn.dataset.tab === "benchmark") loadBenchmark();
  });
});

/* ---------------------------------------------------------------- status */
async function loadStatus() {
  try {
    const s = await api("api/status");
    $("#status").innerHTML = [
      `<span class="chip ok"><b>${s.chunks_indexed.toLocaleString()}</b> chunks</span>`,
      `<span class="chip"><b>${s.parents_indexed.toLocaleString()}</b> sections</span>`,
      `<span class="chip">embed: <b>${escapeHtml(s.embedding_model)}</b></span>`,
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

async function loadExamples() {
  try {
    const examples = await api("api/examples");
    $("#examples").innerHTML = examples.map((q) => `<button type="button" class="example">${escapeHtml(q)}</button>`).join("");
    document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => { $("#query").value = b.textContent; $("#query").focus(); }));
  } catch (_) { /* examples are optional */ }
}

/* ---------------------------------------------------------------- search */
function renderFunnel(counts, expanded, original) {
  const rows = [
    ["Dense candidates", counts.dense], ["BM25 candidates", counts.bm25], ["Fused (RRF)", counts.fused],
    ["Cross-encoder scored", counts.reranked], ["Returned", counts.returned],
  ];
  const max = Math.max(...rows.map((r) => r[1] || 0), 1);
  $("#funnel").innerHTML = rows.map(([label, n]) =>
    `<div class="funnel-row"><span>${label}</span><div class="funnel-bar" style="width:${Math.max(2, (100 * (n || 0)) / max)}%"></div><span>${n ?? "–"}</span></div>`
  ).join("");
  const note = $("#expansion");
  if (expanded && expanded.trim() !== original.trim()) {
    note.innerHTML = `<b>Statutory concordance expansion:</b> ${escapeHtml(expanded)}`;
    note.classList.remove("hidden");
  } else { note.classList.add("hidden"); }
}

function renderTimings(timings) {
  const entries = Object.entries(STAGE_LABELS).filter(([k]) => typeof timings[k] === "number").map(([k, label]) => [label, timings[k] * 1000]);
  const max = Math.max(...entries.map((e) => e[1]), 1);
  const total = (timings.funnel_latency_sec || 0) * 1000 + (timings.llm_generation_sec || 0) * 1000;
  $("#timings").innerHTML = entries.map(([label, ms]) =>
    `<div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar-fill" style="width:${(100 * ms) / max}%"></div></div><span class="ms">${ms.toFixed(1)} ms</span></div>`
  ).join("") + `<div class="bar-row total"><span>Total</span><span></span><span class="ms">${total.toFixed(0)} ms</span></div>`;
}

function scoreBar(label, value, max, color) {
  const pct = Math.max(0, Math.min(100, (100 * value) / max));
  return `<div class="score"><span>${label}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div><span>${value}</span></div>`;
}

function renderResults(results) {
  if (!results.length) { $("#results").innerHTML = `<p class="note">No precedents retrieved.</p>`; return; }
  $("#results").innerHTML = results.map((r) => {
    const authority = r.authority_factor >= 1 ? `+${((r.authority_factor - 1) * 100).toFixed(1)}%` : `${((r.authority_factor - 1) * 100).toFixed(1)}%`;
    return `
    <div class="result">
      <div class="rank">${r.rank}</div>
      <div>
        <div class="result-title">${escapeHtml(r.case_title)}</div>
        <div class="result-meta">${escapeHtml(r.court)}${r.year ? " · " + escapeHtml(r.year) : ""}</div>
        <div class="badges">
          ${r.role ? `<span class="badge role">${escapeHtml(r.role)}</span>` : ""}
          <span class="badge">Dense rank: ${r.dense_rank ?? "–"}</span>
          <span class="badge">BM25 rank: ${r.bm25_rank ?? "–"}</span>
          <span class="badge">Authority prior: ${authority}</span>
        </div>
        ${scoreBar("Relevance score", r.rerank_score, 1.05, "var(--accent)")}
        <details><summary class="note">Matched passage</summary><div class="snippet">${escapeHtml(r.snippet)}</div></details>
      </div>
    </div>`;
  }).join("");
}

$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#query").value.trim();
  if (query.length < 3) return;
  const btn = $("#search-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Searching…`;
  $("#search-error").classList.add("hidden");
  try {
    const data = await api("api/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        top_k: Number($("#topk").value),
        one_per_case: $("#one-per-case").checked,
        generate_answer: $("#gen-answer").checked,
      }),
    });
    renderFunnel(data.funnel_counts || {}, data.expanded_query, query);
    renderTimings(data.timings || {});
    renderResults(data.results || []);
    if (data.answer) {
      $("#answer").innerHTML = renderMarkdown(data.answer);
      $("#answer-card").classList.remove("hidden");
    } else { $("#answer-card").classList.add("hidden"); }
    $("#results-area").classList.remove("hidden");
  } catch (err) {
    $("#search-error").textContent = `Search failed: ${err.message}`;
    $("#search-error").classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Search precedents";
  }
});

/* ---------------------------------------------------------------- benchmark */
const LOWER_IS_BETTER = (metric) => metric.toLowerCase().includes("latency");

function fmt(metric, v) {
  if (v === null || v === undefined) return "N/A";
  if (typeof v !== "number") return escapeHtml(v);
  if (LOWER_IS_BETTER(metric)) return `${v.toFixed(0)} ms`;
  if (/hit|mrr/i.test(metric)) return v.toFixed(3);
  return v.toFixed(2);
}

let benchmarkLoaded = false;
async function loadBenchmark() {
  if (benchmarkLoaded) return;
  try {
    const b = await api("api/benchmark");
    benchmarkLoaded = true;
    const configs = Object.keys(b.summary_table);
    const metrics = configs.length ? Object.keys(b.summary_table[configs[0]]) : [];
    $("#bench-meta").textContent = `${b.queries_count} benchmark queries · collection "${b.collection_name}" · run ${b.timestamp} (${b.file})`;

    let html = `<thead><tr><th>Metric</th>${configs.map((c) => `<th class="num">${escapeHtml(c.replace(/^Config \d+ \(|\)$/g, ""))}</th>`).join("")}</tr></thead><tbody>`;
    for (const m of metrics) {
      const vals = configs.map((c) => b.summary_table[c][m]);
      const nums = vals.filter((v) => typeof v === "number");
      const best = nums.length ? (LOWER_IS_BETTER(m) ? Math.min(...nums) : Math.max(...nums)) : null;
      html += `<tr><td>${escapeHtml(m)}</td>${vals.map((v) => `<td class="num ${v === best && nums.length > 1 ? "best" : ""}">${fmt(m, v)}</td>`).join("")}</tr>`;
    }
    $("#bench-table").innerHTML = html + "</tbody>";

    $("#perquery-table").innerHTML =
      `<thead><tr><th>ID</th><th>Question</th><th>Target judgment</th><th class="num">Rank</th><th>Top-1 retrieved</th></tr></thead><tbody>` +
      b.per_query.map((q) => `<tr>
        <td>${escapeHtml(q.query_id)}</td>
        <td>${escapeHtml(q.query)}</td>
        <td>${escapeHtml(q.target_case)}</td>
        <td class="num ${q.rank === 1 ? "hit" : q.rank ? "" : "miss"}">${q.rank ?? "not in top-5"}</td>
        <td>${escapeHtml(q.top_1_case)}</td></tr>`).join("") + "</tbody>";
    $("#bench-area").classList.remove("hidden");
    $("#bench-error").classList.add("hidden");
  } catch (err) {
    $("#bench-error").textContent = `Could not load benchmark: ${err.message}`;
    $("#bench-error").classList.remove("hidden");
  }
}

loadStatus();
loadExamples();
