const $ = (id) => document.getElementById(id);
const charts = {};
const HISTORY_KEY = "linza-history";
const THEME_KEY = "linza-theme";

const DIM_LABELS = {
  correctness: "Correctness",
  relevance: "Relevance",
  completeness: "Completeness",
  coherence: "Coherence",
  groundedness: "Groundedness",
};
const HITL_KEYS = ["correctness", "relevance", "completeness", "coherence", "groundedness"];

function themeColors() {
  const s = getComputedStyle(document.documentElement);
  return {
    ink: s.getPropertyValue("--ink").trim(),
    muted: s.getPropertyValue("--muted").trim(),
    accent: s.getPropertyValue("--accent").trim(),
    teal: s.getPropertyValue("--teal").trim(),
    rose: s.getPropertyValue("--rose").trim(),
    line: s.getPropertyValue("--line").trim(),
  };
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
  $("themeDark").classList.toggle("active", theme === "dark");
  $("themeLight").classList.toggle("active", theme === "light");
  refreshOpenCharts();
}

function applySavedTheme() {
  setTheme(localStorage.getItem(THEME_KEY) || "dark");
}

function upsertChart(id, config) {
  if (charts[id]) charts[id].destroy();
  const node = $(id);
  if (!node) return;
  charts[id] = new Chart(node, config);
}

function refreshOpenCharts() {
  const last = window.__lastReport;
  if (last) renderResult(last, false);
  renderHistory();
  renderLabelCharts(window.__labels || []);
  if (window.__overview) renderOverview(window.__overview);
}

function pct(value) {
  if (value == null || Number.isNaN(value)) return null;
  return Math.round(Number(value) * 100);
}

function fmt(value) {
  const n = pct(value);
  return n == null ? "—" : `${n}`;
}

function meanDims(dims) {
  const vals = ["correctness", "relevance", "completeness", "coherence"]
    .map((k) => dims[k])
    .filter((v) => v != null);
  if (dims.groundedness != null) vals.push(dims.groundedness);
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

const PRESETS = {
  openrouter: [
    ["Gemma 4 31B", "google/gemma-4-31b-it"],
    ["Gemma 4 free", "google/gemma-4-31b-it:free"],
    ["Llama 3.3 70B", "meta-llama/llama-3.3-70b-instruct"],
    ["GPT-4o mini", "openai/gpt-4o-mini"],
    ["Claude 3.5", "anthropic/claude-3.5-sonnet"],
  ],
  openai: [
    ["gpt-4o-mini", "gpt-4o-mini"],
    ["gpt-4o", "gpt-4o"],
    ["gpt-4.1-mini", "gpt-4.1-mini"],
  ],
  ollama: [
    ["llama3.2", "llama3.2"],
    ["llama3.1", "llama3.1"],
    ["qwen2.5", "qwen2.5"],
    ["mistral", "mistral"],
  ],
};

function applyJudgeView(data) {
  window.__settings = data;
  window.__judgeName = `${data.provider} / ${data.model}`;
  $("healthChip").innerHTML = `<span class="dot"></span><span class="chip-text">${data.provider} · ${data.model}</span>`;
  $("healthChip").title = `${data.provider} / ${data.model}`;
  $("useJudgeLabel").textContent = `G-Eval · ${data.model}`;
  $("orModel").value = data.openrouter_model;
  $("orBase").value = data.openrouter_base_url;
  $("orHint").textContent = data.openrouter_key_set
    ? "Ключ задан. Пустое поле не перезаписывает."
    : "Ключ не задан.";
  $("orKey").placeholder = "API key";
  $("oaModel").value = data.openai_model;
  $("oaBase").value = data.openai_base_url;
  $("oaHint").textContent = data.openai_key_set
    ? "Ключ задан. Пустое поле не перезаписывает."
    : "Ключ не задан.";
  $("oaKey").placeholder = "API key";
  $("olBase").value = data.ollama_base_url;
  $("olModel").value = data.ollama_model;
  selectProvider(data.provider);
}

function selectProvider(name) {
  window.__provider = name;
  document.querySelectorAll(".provider").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.provider === name);
  });
  document.querySelectorAll(".judge-fields").forEach((box) => {
    box.classList.toggle("show", box.dataset.for === name);
  });
}

function fillPresets() {
  for (const [group, items] of Object.entries(PRESETS)) {
    const host = { openrouter: "orPresets", openai: "oaPresets", ollama: "olPresets" }[group];
    $(host).innerHTML = items.map(([label, value]) => `<button type="button" data-value="${value}">${label}</button>`).join("");
    $(host).onclick = (event) => {
      const btn = event.target.closest("button[data-value]");
      if (!btn) return;
      const input = { openrouter: "orModel", openai: "oaModel", ollama: "olModel" }[group];
      $(input).value = btn.dataset.value;
    };
  }
}

async function loadSettings() {
  try {
    applyJudgeView(await (await fetch("/settings")).json());
  } catch {
    $("healthChip").innerHTML = `<span class="dot bad"></span> offline`;
  }
}

async function persistJudge() {
  const payload = {
    provider: window.__provider,
    openrouter_model: $("orModel").value.trim(),
    openrouter_base_url: $("orBase").value.trim(),
    openrouter_api_key: $("orKey").value.trim() || null,
    openai_model: $("oaModel").value.trim(),
    openai_base_url: $("oaBase").value.trim(),
    openai_api_key: $("oaKey").value.trim() || null,
    ollama_base_url: $("olBase").value.trim(),
    ollama_model: $("olModel").value.trim(),
  };
  const res = await fetch("/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`сервер ответил ${res.status}`);
  applyJudgeView(await res.json());
  $("orKey").value = "";
  $("oaKey").value = "";
}

async function saveJudge() {
  const msg = $("judgeMsg");
  $("saveJudge").disabled = true;
  msg.className = "muted";
  msg.textContent = "save…";
  try {
    await persistJudge();
    msg.className = "ok";
    msg.textContent = `Провайдер: ${window.__judgeName}`;
  } catch (err) {
    msg.className = "err";
    msg.textContent = err.message;
  } finally {
    $("saveJudge").disabled = false;
  }
}

async function testJudge() {
  const msg = $("judgeMsg");
  $("testJudge").disabled = true;
  msg.className = "muted";
  msg.textContent = "Health-check…";
  try {
    await persistJudge();
    const data = await (await fetch("/settings/test", { method: "POST" })).json();
    msg.className = data.ok ? "ok" : "err";
    msg.textContent = data.ok
      ? `ok · ${data.provider} / ${data.model} · ${data.sample}`
      : data.message;
  } catch (err) {
    msg.className = "err";
    msg.textContent = err.message;
  } finally {
    $("testJudge").disabled = false;
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  });
  requestAnimationFrame(() => {
    Object.values(charts).forEach((chart) => {
      try {
        chart.resize();
      } catch {
        /* chart not ready */
      }
    });
  });
}

function payloadEval() {
  return {
    question: $("question").value.trim(),
    answer: $("answer").value.trim(),
    reference: $("reference").value.trim() || null,
    context: $("context").value.trim() || null,
    skip_judge: !$("useJudge").checked,
  };
}

function renderResult(data, pushHistory) {
  window.__lastReport = data;
  $("resultEmpty").hidden = true;
  $("resultBody").hidden = false;
  const score = pct(data.overall) ?? 0;
  $("scorePct").textContent = score;
  $("gauge").style.setProperty("--p", score);
  const who = data.judge.fallback
    ? "режим: heuristics (judge fallback)"
    : `G-Eval · ${data.judge.provider} / ${data.judge.model}`;
  $("who").textContent = who;
  $("confLine").textContent = `confidence ${fmt(data.confidence)} · overall ${data.overall}`;
  const flagNames = {
    no_reference: "no reference",
    no_context: "no context",
    high_disagreement: "high disagreement",
    empty_or_too_short: "empty / short",
    judge_fallback: "judge fallback",
    position_bias_detected: "position bias",
    ensemble_disagreement: "ensemble disagreement",
    correctness_capped: "fact over fluency",
  };
  $("flags").innerHTML = Object.entries(data.flags)
    .filter(([, on]) => on)
    .map(([key]) => `<span class="flag on">${flagNames[key] || key}</span>`)
    .join("");

  const dimOrder = ["correctness", "relevance", "completeness", "coherence"];
  if (data.dimensions.groundedness != null) dimOrder.push("groundedness");
  $("dimCards").innerHTML = dimOrder.map((key) => {
    const value = data.dimensions[key];
    return `<div class="metric"><small>${DIM_LABELS[key]}</small><b>${fmt(value)}</b><div class="bar"><i style="width:${pct(value) || 0}%"></i></div></div>`;
  }).join("") + [
    ["ROUGE-L", data.lexical.rouge_l],
    ["BLEU", data.lexical.bleu],
    ["chrF", data.lexical.chrf],
    ["cosine", data.semantic.cosine],
  ].map(([title, value]) => `<div class="metric"><small>${title}</small><b>${fmt(value)}</b><div class="bar"><i style="width:${pct(value) || 0}%"></i></div></div>`).join("");

  $("rationale").textContent = data.judge.rationale || "rationale отсутствует";
  $("human").value = score;
  $("humanOut").textContent = `${score} / 100`;
  setHitlDims(data.dimensions, data.overall);
  drawResultCharts(data);
  if (pushHistory) addHistory(data);
}

function drawResultCharts(data) {
  const c = themeColors();
  const labels = ["Correctness", "Relevance", "Completeness", "Coherence"];
  const values = [
    data.dimensions.correctness,
    data.dimensions.relevance,
    data.dimensions.completeness,
    data.dimensions.coherence,
  ].map((v) => Number(v ?? 0));
  if (data.dimensions.groundedness != null) {
    labels.push("Groundedness");
    values.push(Number(data.dimensions.groundedness));
  }

  upsertChart("radarChart", {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "G-Eval",
        data: values,
        borderColor: c.accent,
        backgroundColor: hexAlpha(c.accent, 0.22),
        pointBackgroundColor: c.teal,
      }],
    },
    options: radarOptions(c),
  });

  upsertChart("layerChart", {
    type: "bar",
    data: {
      labels: ["overall", "judge", "ROUGE-L", "BLEU", "chrF", "cosine"],
      datasets: [{
        label: "слои",
        data: [
          data.overall,
          meanDims(data.dimensions),
          data.lexical.rouge_l ?? 0,
          data.lexical.bleu ?? 0,
          data.lexical.chrf ?? 0,
          data.semantic.cosine ?? 0,
        ],
        backgroundColor: [c.accent, c.teal, c.rose, hexAlpha(c.ink, 0.35), hexAlpha(c.teal, 0.55)],
      }],
    },
    options: barOptions(c),
  });
}

function chartBox(extra) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 240 },
    ...extra,
  };
}

function radarOptions(c) {
  return chartBox({
    plugins: { legend: { display: false } },
    layout: { padding: 6 },
    scales: {
      r: {
        min: 0,
        max: 1,
        ticks: { display: false },
        grid: { color: c.line },
        angleLines: { color: c.line },
        pointLabels: { color: c.muted, font: { size: 11 } },
      },
    },
  });
}

function barOptions(c) {
  return chartBox({
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: c.muted, maxRotation: 0, autoSkip: true }, grid: { display: false } },
      y: { min: 0, max: 1, ticks: { color: c.muted }, grid: { color: c.line } },
    },
  });
}

function hexAlpha(color, a) {
  if (color.startsWith("#") && color.length === 7) {
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  return color;
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function addHistory(data) {
  const rows = readHistory();
  rows.unshift({
    ts: new Date().toLocaleTimeString("ru-RU"),
    question: $("question").value.trim(),
    overall: data.overall,
    who: data.judge.fallback ? "heuristics" : data.judge.model,
  });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(rows.slice(0, 16)));
  renderHistory();
}

function renderHistory() {
  const rows = readHistory();
  $("histTable").innerHTML = rows.length
    ? rows.map((r) => `<tr><td>${r.ts}</td><td>${escapeHtml(r.question).slice(0, 80)}</td><td>${fmt(r.overall)}</td><td>${escapeHtml(r.who)}</td></tr>`).join("")
    : `<tr><td colspan="4" class="muted">пусто</td></tr>`;
  const c = themeColors();
  upsertChart("histChart", {
    type: "line",
    data: {
      labels: rows.slice().reverse().map((r) => r.ts),
      datasets: [{
        label: "overall",
        data: rows.slice().reverse().map((r) => r.overall),
        borderColor: c.accent,
        backgroundColor: hexAlpha(c.accent, 0.16),
        fill: true,
        tension: 0.35,
      }],
    },
    options: chartBox({
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: c.muted, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }, grid: { display: false } },
        y: { min: 0, max: 1, ticks: { color: c.muted }, grid: { color: c.line } },
      },
    }),
  });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

async function evaluate() {
  const btn = $("run");
  btn.disabled = true;
  $("resultEmpty").hidden = false;
  $("resultEmpty").textContent = "считаю…";
  $("resultBody").hidden = true;
  try {
    const res = await fetch("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadEval()),
    });
    if (!res.ok) throw new Error(`сервер ответил ${res.status}`);
    renderResult(await res.json(), true);
  } catch (err) {
    $("resultEmpty").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

async function compare() {
  const btn = $("runPair");
  btn.disabled = true;
  $("pairOut").textContent = "pairwise…";
  try {
    const res = await fetch("/evaluate/pairwise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: $("pq").value.trim(),
        answer_a: $("pa").value.trim(),
        answer_b: $("pb").value.trim(),
        reference: $("pr").value.trim() || null,
      }),
    });
    if (!res.ok) throw new Error(`сервер ответил ${res.status}`);
    const data = await res.json();
    $("pairOut").innerHTML = `
      <p>order AB: <b>${data.first_pass.winner}</b> — ${escapeHtml(data.first_pass.rationale)}</p>
      <p>order BA: <b>${data.swapped_pass.winner}</b> — ${escapeHtml(data.swapped_pass.rationale)}</p>
      <p>position bias: <b>${data.position_bias_detected ? "detected" : "not detected"}</b></p>
      <p>preferred: <b>${data.preferred}</b></p>
    `;
  } catch (err) {
    $("pairOut").innerHTML = `<p class="err">${escapeHtml(err.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
}

async function saveLabel() {
  const btn = $("save");
  const msg = $("saveMsg");
  btn.disabled = true;
  msg.className = "muted";
  msg.textContent = "save…";
  try {
    const res = await fetch("/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: $("question").value.trim(),
        answer: $("answer").value.trim(),
        reference: $("reference").value.trim() || null,
        context: $("context").value.trim() || null,
        human_score: Number($("human").value) / 100,
        case_type: $("caseType").value.trim() || null,
        human_dimensions: {
          correctness: Number($("h-correctness").value) / 100,
          relevance: Number($("h-relevance").value) / 100,
          completeness: Number($("h-completeness").value) / 100,
          coherence: Number($("h-coherence").value) / 100,
          groundedness: $("context").value.trim() ? Number($("h-groundedness").value) / 100 : null,
        },
      }),
    });
    if (!res.ok) throw new Error(`сервер ответил ${res.status}`);
    const data = await res.json();
    msg.className = "ok";
    msg.textContent = `${data.item.id} · n = ${data.total}`;
    await loadLabels();
  } catch (err) {
    msg.className = "err";
    msg.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

async function loadLabels() {
  const data = await (await fetch("/labels/items")).json();
  window.__labels = data.items;
  $("total").textContent = `(${data.total})`;
  $("labelTable").innerHTML = data.items.map((item) => `
    <tr data-id="${escapeHtml(item.id)}">
      <td>${escapeHtml(item.id)}</td>
      <td>${escapeHtml(item.question).slice(0, 90)}</td>
      <td>${fmt(item.human_score)}</td>
    </tr>
  `).join("");
  $("labelTable").onclick = (event) => {
    const row = event.target.closest("tr[data-id]");
    if (!row) return;
    const item = data.items.find((x) => x.id === row.dataset.id);
    if (!item) return;
    $("question").value = item.question;
    $("answer").value = item.answer;
    $("reference").value = item.reference || "";
    $("context").value = item.context || "";
    $("human").value = pct(item.human_score);
    $("humanOut").textContent = `${pct(item.human_score)} / 100`;
    $("caseType").value = item.case_type || "";
    setHitlDims(item.human_dimensions, item.human_score);
    switchTab("eval");
  };
  renderLabelCharts(data.items);
}

function renderLabelCharts(items) {
  const buckets = [0, 0, 0, 0, 0];
  items.forEach((item) => {
    const score = Math.min(4, Math.floor((item.human_score || 0) * 5));
    buckets[score] += 1;
  });
  const c = themeColors();
  upsertChart("labelHist", {
    type: "bar",
    data: {
      labels: ["0–20", "20–40", "40–60", "60–80", "80–100"],
      datasets: [{ data: buckets, backgroundColor: c.teal, label: "меток" }],
    },
    options: barOptions(c),
  });
}

async function runOverview() {
  const btn = $("runOverview");
  btn.disabled = true;
  $("overviewMsg").textContent = "skip_judge · прогон корпуса…";
  try {
    const pack = await (await fetch("/labels/items")).json();
    const res = await fetch("/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skip_judge: true, items: pack.items }),
    });
    if (!res.ok) throw new Error(`сервер ответил ${res.status}`);
    const data = await res.json();
    window.__overview = data;
    renderOverview(data);
    $("overviewMsg").textContent = `n = ${data.overall_vs_human.n}` + (data.semantic_method ? ` · ${data.semantic_method}` : "");
  } catch (err) {
    $("overviewMsg").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

function renderOverview(data) {
  const rows = [
    ["overall vs human", data.overall_vs_human],
    ["judge vs human", data.judge_vs_human],
    ["cosine vs human", data.semantic_vs_human],
    ["lexical vs human", data.lexical_vs_human],
  ];
  $("overviewCards").innerHTML = rows.map(([title, r]) => `
    <div class="metric"><small>${title}</small><b>ρ ${r.spearman ?? "—"}</b>
    <span class="muted">Pearson ${r.pearson ?? "—"} · Kendall ${r.kendall ?? "—"}</span></div>
  `).join("");
  if (data.recommended_weights) {
    const w = data.recommended_weights;
    $("overviewCards").innerHTML += `<div class="metric"><small>веса по Spearman</small><b>${w.weight_semantic} / ${w.weight_lexical} / ${w.weight_judge}</b>
      <span class="muted">cosine / lexical / judge · ρ ${w.spearman}</span></div>`;
  }
  const err = $("errorTables");
  if (err) {
    err.innerHTML = renderErrorBlock("человек высокий — автомат низкий", data.human_high_auto_low)
      + renderErrorBlock("человек низкий — автомат высокий", data.human_low_auto_high);
  }
  const c = themeColors();
  upsertChart("corrChart", {
    type: "bar",
    data: {
      labels: ["overall", "judge", "cosine", "lexical"],
      datasets: [{
        label: "Spearman",
        data: rows.map(([, r]) => r.spearman ?? 0),
        backgroundColor: c.accent,
      }],
    },
    options: chartBox({
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: c.muted, maxRotation: 0 }, grid: { display: false } },
        y: { min: -1, max: 1, ticks: { color: c.muted }, grid: { color: c.line } },
      },
    }),
  });
  upsertChart("biasChart", {
    type: "bar",
    data: {
      labels: ["Verbosity r", "Verbosity ρ", "Disagreement", "High dis."],
      datasets: [{
        label: "bias",
        data: [
          data.biases.verbosity_pearson ?? 0,
          data.biases.verbosity_spearman ?? 0,
          data.biases.mean_disagreement ?? 0,
          data.biases.high_disagreement_rate ?? 0,
        ],
        backgroundColor: [c.rose, c.accent, c.teal, hexAlpha(c.ink, 0.35)],
      }],
    },
    options: chartBox({
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: c.muted, maxRotation: 0 }, grid: { display: false } },
        y: { min: -1, max: 1, ticks: { color: c.muted }, grid: { color: c.line } },
      },
    }),
  });
}

$("tabs").onclick = (event) => {
  const btn = event.target.closest(".tab");
  if (btn) switchTab(btn.dataset.tab);
};
$("themeDark").onclick = () => setTheme("dark");
$("themeLight").onclick = () => setTheme("light");
$("run").onclick = evaluate;
$("runPair").onclick = compare;
$("save").onclick = saveLabel;
$("runOverview").onclick = runOverview;
$("human").oninput = () => {
  $("humanOut").textContent = `${$("human").value} / 100`;
};

function renderHitlDims() {
  const host = $("hitlDims");
  if (!host) return;
  host.innerHTML = HITL_KEYS.map((key) => `
    <label>${DIM_LABELS[key]}</label>
    <div class="range-row">
      <input id="h-${key}" type="range" min="0" max="100" value="90" />
      <b id="h-${key}Out">90</b>
    </div>
  `).join("");
  HITL_KEYS.forEach((key) => {
    $(`h-${key}`).oninput = () => {
      $(`h-${key}Out`).textContent = $(`h-${key}`).value;
    };
  });
}

function setHitlDims(dims, overall) {
  const fallback = pct(overall) ?? 90;
  HITL_KEYS.forEach((key) => {
    const node = $(`h-${key}`);
    const out = $(`h-${key}Out`);
    if (!node || !out) return;
    const value = dims && dims[key] != null ? pct(dims[key]) : fallback;
    node.value = value;
    out.textContent = String(value);
  });
}

function renderErrorBlock(title, rows) {
  if (!rows || !rows.length) return `<p class="muted">${title}: нет случаев с |Δ| ≥ 0.25</p>`;
  const body = rows.map((row) => `
    <tr>
      <td>${escapeHtml(row.id)}</td>
      <td>${escapeHtml(row.case_type || "—")}</td>
      <td>${fmt(row.human)}</td>
      <td>${fmt(row.auto)}</td>
      <td>${row.delta > 0 ? "+" : ""}${Math.round(row.delta * 100)}</td>
      <td>${escapeHtml((row.reasons || []).join(", "))}</td>
    </tr>
  `).join("");
  return `<h3 style="margin-top:18px">${title}</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>id</th><th>тип</th><th>human</th><th>auto</th><th>Δ</th><th>причины</th></tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
}
$("fillBad").onclick = () => {
  $("answer").value = "Столица Франции — Лион, это все знают.";
};
$("healthChip").onclick = () => switchTab("judge");
$("providers").onclick = (event) => {
  const btn = event.target.closest(".provider");
  if (btn) selectProvider(btn.dataset.provider);
};
$("saveJudge").onclick = saveJudge;
$("testJudge").onclick = testJudge;

window.__batchItems = [];
window.__batchReports = [];
let batchAbort = null;

function clip(text, n) {
  const value = String(text || "").replace(/\s+/g, " ").trim();
  return value.length <= n ? value : `${value.slice(0, n)}…`;
}

function setBatchProgress(done, total) {
  $("batchProgress").hidden = total <= 0;
  $("batchBar").style.width = total ? `${Math.round((done / total) * 100)}%` : "0%";
}

function renderBatchTable() {
  const items = window.__batchItems || [];
  const reports = window.__batchReports || [];
  $("batchTable").innerHTML = items.map((item, index) => {
    const report = reports[index];
    const overall = report ? fmt(report.overall) : "—";
    const corr = report ? fmt(report.dimensions.correctness) : "—";
    const who = !report ? "—" : (report.judge.fallback ? "heuristics" : report.judge.model);
    return `<tr data-index="${index}">
      <td>${index + 1}</td>
      <td>${escapeHtml(clip(item.question, 80))}</td>
      <td>${escapeHtml(clip(item.answer, 80))}</td>
      <td>${overall}</td>
      <td>${corr}</td>
      <td>${escapeHtml(who)}</td>
    </tr>`;
  }).join("");
}

function applyImportedItems(items, source) {
  window.__batchItems = items;
  window.__batchReports = [];
  renderBatchTable();
  $("batchMsg").className = "ok";
  $("batchMsg").textContent = `Загружено ${items.length} пар (${source}).`;
  setBatchProgress(0, 0);
}

async function importBatchText(text, filename) {
  const res = await fetch("/evaluate/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, filename: filename || "" }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `сервер ответил ${res.status}`);
  applyImportedItems(data.items, filename || "текст");
}

async function loadBatchSample() {
  $("batchMsg").className = "muted";
  $("batchMsg").textContent = "гружу демо…";
  const res = await fetch("/samples");
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `сервер ответил ${res.status}`);
  applyImportedItems(data.items, "sample_eval.json");
}

function openBatchRow(index) {
  const item = window.__batchItems[index];
  if (!item) return;
  $("question").value = item.question || "";
  $("answer").value = item.answer || "";
  $("reference").value = item.reference || "";
  $("context").value = item.context || "";
  const report = window.__batchReports[index];
  if (report) renderResult(report, false);
  switchTab("eval");
}

async function runBatch() {
  const items = window.__batchItems || [];
  if (!items.length) {
    $("batchMsg").className = "err";
    $("batchMsg").textContent = "Сначала загрузите файл или демо-корпус.";
    return;
  }
  const useJudge = $("batchJudge").checked;
  $("batchRun").disabled = true;
  $("batchStop").hidden = false;
  $("batchMsg").className = "muted";
  window.__batchReports = [];
  renderBatchTable();
  batchAbort = new AbortController();
  try {
    if (!useJudge) {
      $("batchMsg").textContent = `считаю ${items.length} пар без G-Eval…`;
      setBatchProgress(0, items.length);
      const res = await fetch("/evaluate/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items, skip_judge: true }),
        signal: batchAbort.signal,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `сервер ответил ${res.status}`);
      window.__batchReports = data.results || [];
      setBatchProgress(items.length, items.length);
      renderBatchTable();
      $("batchMsg").className = "ok";
      $("batchMsg").textContent = `Готово: ${data.n} пар, средний overall ${fmt(data.mean_overall)}.`;
      return;
    }
    for (let i = 0; i < items.length; i += 1) {
      $("batchMsg").textContent = `G-Eval ${i + 1}/${items.length}…`;
      setBatchProgress(i, items.length);
      const res = await fetch("/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...items[i], skip_judge: false }),
        signal: batchAbort.signal,
      });
      if (!res.ok) throw new Error(`сервер ответил ${res.status} на паре ${i + 1}`);
      window.__batchReports[i] = await res.json();
      renderBatchTable();
    }
    setBatchProgress(items.length, items.length);
    const mean = window.__batchReports.reduce((s, r) => s + r.overall, 0) / items.length;
    $("batchMsg").className = "ok";
    $("batchMsg").textContent = `Готово: ${items.length} пар, средний overall ${fmt(mean)}.`;
  } catch (err) {
    if (err.name === "AbortError") {
      $("batchMsg").className = "muted";
      $("batchMsg").textContent = "Остановлено.";
    } else {
      $("batchMsg").className = "err";
      $("batchMsg").textContent = err.message;
    }
  } finally {
    $("batchRun").disabled = false;
    $("batchStop").hidden = true;
    batchAbort = null;
  }
}

function exportBatch() {
  const items = window.__batchItems || [];
  if (!items.length) return;
  const payload = items.map((item, index) => ({
    ...item,
    report: window.__batchReports[index] || null,
  }));
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "linza-batch.json";
  link.click();
  URL.revokeObjectURL(url);
}

$("batchFile").onchange = async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  $("batchMsg").className = "muted";
  $("batchMsg").textContent = `читаю ${file.name}…`;
  try {
    await importBatchText(await file.text(), file.name);
  } catch (err) {
    $("batchMsg").className = "err";
    $("batchMsg").textContent = err.message;
  }
  event.target.value = "";
};
$("batchSample").onclick = () => loadBatchSample().catch((err) => {
  $("batchMsg").className = "err";
  $("batchMsg").textContent = err.message;
});
$("batchPasteBtn").onclick = () => {
  const show = $("batchText").hidden;
  $("batchText").hidden = !show;
  $("batchPasteActions").hidden = !show;
};
$("batchParse").onclick = () => {
  importBatchText($("batchText").value, "paste.json").catch((err) => {
    $("batchMsg").className = "err";
    $("batchMsg").textContent = err.message;
  });
};
$("batchRun").onclick = runBatch;
$("batchStop").onclick = () => batchAbort && batchAbort.abort();
$("batchExport").onclick = exportBatch;
$("batchTable").onclick = (event) => {
  const row = event.target.closest("tr[data-index]");
  if (row) openBatchRow(Number(row.dataset.index));
};

applySavedTheme();
fillPresets();
renderHitlDims();
loadSettings();
renderHistory();
loadLabels().catch(() => {
  $("total").textContent = "(?)";
});
