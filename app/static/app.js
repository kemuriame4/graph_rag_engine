(() => {
  "use strict";

  const API = "/api/v1";
  const state = {
    response: null,
    activeJobIds: new Set(),
    pollTimer: null,
    currentDrawerDoc: null,
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  const TITLES = {
    ask: "智能问答",
    documents: "文档入库",
    graph: "知识图谱",
    eval: "评测看板",
    traces: "链路 Trace",
  };

  async function api(path, options = {}) {
    const response = await fetch(API + path, options);
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (error) {
        /* ignore */
      }
      throw new Error(detail);
    }
    if (response.status === 204) return null;
    return response.json();
  }

  function toast(message, isError = false) {
    const el = $("#toast");
    el.textContent = message;
    el.classList.toggle("err", isError);
    el.classList.add("show");
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.remove("show"), 3600);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatScore(value) {
    return value === null || value === undefined ? "—" : Number(value).toFixed(3);
  }

  // ---------------- 健康与指标 ----------------
  async function refreshHealth() {
    try {
      const health = await api("/health");
      const dot = $("#health-dot");
      dot.className = "dot ok";
      $("#health-text").textContent =
        `${health.embedding} · ${health.vector} · ${health.sparse}`;
      $("#eval-backend").textContent =
        health.judge === "ragas" ? "Ragas 官方后端" : "内置指标（Ragas 口径）";
    } catch (error) {
      $("#health-dot").className = "dot err";
      $("#health-text").textContent = "服务不可用";
    }
  }

  async function refreshMetrics() {
    try {
      const [stats, metrics] = await Promise.all([api("/stats"), api("/metrics")]);
      $("#m-docs").textContent = stats.documents;
      $("#m-chunks").textContent = stats.chunks;
      $("#m-entities").textContent = stats.entities;
      $("#m-p95").textContent = `${metrics.latency.p95 ?? "-"} ms`;
      $("#trace-hint").textContent =
        `查询 ${metrics.counters.query || 0} · 入库 ${metrics.counters.ingest || 0} · ` +
        `Token ${(metrics.tokens.prompt || 0) + (metrics.tokens.completion || 0)}`;
    } catch (error) {
      /* 静默 */
    }
  }

  // ---------------- 导航 ----------------
  function bindNav() {
    $$(".nav-item").forEach((button) => {
      button.addEventListener("click", () => {
        $$(".nav-item").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        const view = button.dataset.view;
        $$(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
        $("#page-title").textContent = TITLES[view] || view;
        if (view === "documents") loadDocuments();
        if (view === "traces") loadTraces();
        if (view === "eval") loadEvalReports();
      });
    });
  }

  // ---------------- 问答 ----------------
  function addStep(node, detail, elapsed, done = true) {
    const box = $("#live-steps");
    if (box.classList.contains("empty")) {
      box.classList.remove("empty");
      box.innerHTML = "";
    }
    const step = document.createElement("div");
    step.className = "step" + (done ? " done" : "");
    step.innerHTML =
      `<span class="step-dot"></span>` +
      `<div><strong>${escapeHtml(node)}</strong><p>${escapeHtml(detail || "")}</p></div>` +
      `<time>${Number(elapsed).toFixed(1)} ms</time>`;
    box.appendChild(step);
    box.scrollTop = box.scrollHeight;
  }

  function resetSteps(statusText) {
    const box = $("#live-steps");
    box.className = "timeline empty";
    box.textContent = "正在执行状态机…";
    $("#ask-status").textContent = statusText || "运行中";
  }

  async function askRest(payload) {
    return api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  function askStream(payload) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const socket = new WebSocket(`${protocol}//${location.host}${API}/ws/query`);
      const timer = setTimeout(() => {
        if (!settled) {
          settled = true;
          socket.close();
          reject(new Error("流式响应超时"));
        }
      }, 60000);

      socket.onopen = () => socket.send(JSON.stringify(payload));
      socket.onmessage = (event) => {
        let message;
        try {
          message = JSON.parse(event.data);
        } catch (error) {
          return;
        }
        if (message.type === "step") {
          addStep(message.node, message.detail, message.elapsed_ms);
        } else if (message.type === "result") {
          settled = true;
          clearTimeout(timer);
          socket.close();
          resolve(message);
        } else if (message.type === "error") {
          settled = true;
          clearTimeout(timer);
          socket.close();
          reject(new Error(message.detail || "查询失败"));
        }
      };
      socket.onerror = () => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          reject(new Error("WebSocket 连接失败"));
        }
      };
      socket.onclose = () => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          reject(new Error("WebSocket 已关闭"));
        }
      };
    });
  }

  async function ask() {
    const question = $("#question").value.trim();
    if (!question) {
      toast("请先输入问题", true);
      return;
    }
    const payload = {
      question,
      top_k: Math.max(1, Math.min(30, Number($("#opt-topk").value) || 8)),
      enable_hyde: $("#opt-hyde").checked,
      enable_graph: $("#opt-graph").checked,
      tenant: "default",
    };
    const button = $("#btn-ask");
    button.disabled = true;
    button.textContent = "检索中…";
    resetSteps("运行中");
    $("#answer-body").classList.add("empty");
    $("#answer-body").textContent = "正在生成回答…";
    $("#citations").innerHTML = "";
    $("#warnings").innerHTML = "";
    try {
      let data;
      if ($("#opt-stream").checked) {
        try {
          data = await askStream(payload);
        } catch (streamError) {
          data = await askRest(payload);
        }
      } else {
        data = await askRest(payload);
      }
      renderAnswer(data);
      refreshMetrics();
      loadTraces();
    } catch (error) {
      $("#ask-status").textContent = "失败";
      $("#answer-body").classList.remove("empty");
      $("#answer-body").textContent = `查询失败：${error.message}`;
      toast(error.message, true);
    } finally {
      button.disabled = false;
      button.textContent = "发起检索问答";
    }
  }

  function badge(label, value, pass, threshold) {
    const cls = pass ? "pass" : value > 0 ? "warn" : "fail";
    return `<span class="badge ${cls}">${label} ${Number(value).toFixed(2)} / ${threshold}</span>`;
  }

  function renderAnswer(data) {
    state.response = data;
    $("#ask-status").textContent =
      `${data.query_type} · ${data.latency_ms} ms · ${data.prompt_tokens + data.completion_tokens} tok`;

    const body = $("#answer-body");
    body.classList.remove("empty");
    const html = escapeHtml(data.answer).replace(/\[(\d{1,2})\]/g, (match, index) => {
      return `<span class="cite" data-index="${index}">${match}</span>`;
    });
    body.innerHTML = html;

    const reflections = data.reflections || {};
    $("#reflection-badges").innerHTML = [
      badge("Is-Rel", reflections.is_rel || 0, reflections.relevance_pass, "0.22"),
      badge("Is-Sup", reflections.is_sup || 0, reflections.support_pass, "0.72"),
      badge("Is-Use", reflections.is_use || 0, reflections.usefulness_pass, "0.60"),
    ].join("");

    $("#warnings").innerHTML = (data.warnings || [])
      .map((warning) => `<div class="warning">${escapeHtml(warning)}</div>`)
      .join("");

    $("#citations").innerHTML = "";
    (data.citations || []).forEach((citation) => {
      const button = document.createElement("button");
      button.className = "citation";
      button.innerHTML =
        `<b>[${citation.index}] ${escapeHtml(citation.doc_title)}</b>` +
        `<span>${escapeHtml(citation.section || "")}</span>` +
        `<span>${escapeHtml(citation.snippet || "")}</span>`;
      button.addEventListener("click", () => openSource(citation));
      $("#citations").appendChild(button);
    });

    $$(".cite").forEach((cite) => {
      cite.addEventListener("click", () => {
        const index = Number(cite.dataset.index);
        const citation = (data.citations || []).find((item) => item.index === index);
        if (citation) openSource(citation);
      });
    });

    renderRetrieval(data);
  }

  function renderRetrieval(data) {
    const tbody = $("#retrieval-table tbody");
    const docs = data.retrieved || [];
    const trace = data.retrieval || {};
    $("#retrieval-summary").textContent =
      `融合 ${trace.fused_count || 0} · 重排 ${trace.reranked_count || 0} · 召回耗时 ${trace.elapsed_ms || 0} ms` +
      (trace.hyde_query ? " · HyDE 已启用" : "");
    if (!docs.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-row">暂无召回结果</td></tr>';
      return;
    }
    tbody.innerHTML = "";
    docs.forEach((doc, index) => {
      const row = document.createElement("tr");
      row.className = "clickable";
      const channels = (doc.channels || [])
        .map((channel) => `<span class="channel-badge ${escapeHtml(channel)}">${escapeHtml(channel)}</span>`)
        .join("");
      row.innerHTML =
        `<td>${index + 1}</td>` +
        `<td><div class="cell-title">${escapeHtml(doc.doc_title)}</div>` +
        `<div class="cell-sub">${escapeHtml((doc.section_path || []).join(" / ") || "")}</div></td>` +
        `<td>${channels}</td>` +
        `<td>${formatScore(doc.dense_score)}</td>` +
        `<td>${formatScore(doc.sparse_score)}</td>` +
        `<td>${formatScore(doc.graph_score)}</td>` +
        `<td>${formatScore(doc.rrf_score)}</td>` +
        `<td><b>${formatScore(doc.rerank_score)}</b></td>`;
      row.addEventListener("click", () => {
        openSource({
          index: index + 1,
          doc_id: doc.doc_id,
          doc_title: doc.doc_title,
          section: (doc.section_path || []).join(" / "),
          snippet: doc.text,
          page: doc.page_start,
          score: doc.rerank_score,
        });
      });
      tbody.appendChild(row);
    });
  }

  // ---------------- 原文抽屉 ----------------
  function openSource(citation) {
    const drawer = $("#source-drawer");
    drawer.classList.add("open");
    $("#drawer-title").textContent = citation.doc_title || "原文溯源";
    $("#drawer-meta").textContent =
      `${citation.section || ""}${citation.page ? " · 第 " + citation.page + " 页" : ""}`;
    $("#drawer-page").textContent = citation.page ? `第 ${citation.page} 页` : "页码未知";
    $("#drawer-download").href = `${API}/documents/${encodeURIComponent(citation.doc_id)}/file`;
    $("#drawer-search").value = "";

    const terms = collectTerms(state.response?.query || "");
    const snippet = citation.snippet || "暂无原文片段";
    state.currentDrawerDoc = snippet;
    $("#drawer-content").innerHTML = highlightHtml(snippet, terms);

    api(`/documents/${encodeURIComponent(citation.doc_id)}`)
      .then((document) => {
        if (!document.markdown) return;
        state.currentDrawerDoc = document.markdown;
        $("#drawer-content").innerHTML = highlightHtml(document.markdown, terms);
        const first = $("#drawer-content mark");
        if (first) first.scrollIntoView({ block: "center" });
      })
      .catch(() => {
        /* 使用切片文本兜底 */
      });
  }

  function collectTerms(text) {
    const terms = new Set();
    const tokens = String(text).match(/[A-Za-z0-9][A-Za-z0-9._-]{2,}|[\u4e00-\u9fff]{2,6}/g) || [];
    tokens.forEach((token) => {
      if (token.length >= 2) terms.add(token);
    });
    return Array.from(terms).sort((a, b) => b.length - a.length).slice(0, 14);
  }

  function highlightHtml(text, terms) {
    let html = escapeHtml(text);
    terms.forEach((term) => {
      const safe = escapeHtml(term).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      if (!safe) return;
      html = html.replace(new RegExp(safe, "gi"), (match) => `<mark>${match}</mark>`);
    });
    return html;
  }

  function bindDrawer() {
    $("#drawer-close").addEventListener("click", () => $("#source-drawer").classList.remove("open"));
    $("#drawer-search").addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      const term = $("#drawer-search").value.trim();
      if (!term || !state.currentDrawerDoc) return;
      $("#drawer-content").innerHTML = highlightHtml(state.currentDrawerDoc, [term]);
      const first = $("#drawer-content mark.focus") || $("#drawer-content mark");
      if (first) first.scrollIntoView({ block: "center" });
    });
  }

  // ---------------- 文档入库 ----------------
  async function uploadDocument() {
    const input = $("#file-input");
    if (!input.files.length) {
      toast("请选择文件", true);
      return;
    }
    const form = new FormData();
    form.append("file", input.files[0]);
    form.append("tenant", $("#upload-tenant").value.trim() || "default");
    $("#upload-status").textContent = `上传中：${input.files[0].name}`;
    try {
      const job = await api("/documents", { method: "POST", body: form });
      state.activeJobIds.add(job.job_id);
      $("#upload-status").textContent = `已提交任务 ${job.job_id}，正在异步解析…`;
      toast("文档已进入解析队列");
      pollJobs();
    } catch (error) {
      $("#upload-status").textContent = `上传失败：${error.message}`;
      toast(error.message, true);
    }
  }

  async function loadSamples() {
    try {
      await api("/samples/load", { method: "POST" });
      toast("示例文档已提交解析");
      pollJobs();
    } catch (error) {
      toast(error.message, true);
    }
  }

  async function pollJobs() {
    clearTimeout(state.pollTimer);
    try {
      const jobs = await api("/jobs?limit=12");
      renderJobs(jobs);
      const active = jobs.some((job) => ["queued", "parsing", "chunking", "indexing"].includes(job.status));
      if (active) state.pollTimer = setTimeout(pollJobs, 1200);
      else {
        loadDocuments();
        refreshMetrics();
      }
    } catch (error) {
      /* 静默 */
    }
  }

  function renderJobs(jobs) {
    const box = $("#jobs");
    if (!jobs.length) {
      box.innerHTML = "";
      return;
    }
    box.innerHTML = "";
    jobs.forEach((job) => {
      const progress = Math.round((job.progress || 0) * 100);
      const statusText = {
        queued: "排队",
        parsing: "解析中",
        chunking: "语义切片",
        indexing: "索引构建",
        success: "完成",
        error: "失败",
      }[job.status] || job.status;
      const el = document.createElement("div");
      el.className = "job";
      el.innerHTML =
        `<div class="job-top"><b>${escapeHtml(job.filename)}</b><span>${statusText}</span></div>` +
        `<div class="progress"><i style="width:${job.status === "error" ? 100 : progress}%;` +
        `${job.status === "error" ? "background:#fb7185" : ""}"></i></div>` +
        `<div class="job-meta">${job.chunks || 0} 切片 · ${job.entities || 0} 实体` +
        `${job.error ? " · " + escapeHtml(job.error) : ""}</div>`;
      box.appendChild(el);
    });
  }

  async function loadDocuments() {
    try {
      const documents = await api("/documents");
      $("#doc-count").textContent = `共 ${documents.length} 篇`;
      const box = $("#doc-list");
      if (!documents.length) {
        box.className = "doc-list empty";
        box.textContent = "尚未入库文档，可上传或导入内置示例";
        return;
      }
      box.className = "doc-list";
      box.innerHTML = "";
      documents.forEach((doc) => {
        const item = document.createElement("div");
        item.className = "doc-item";
        item.innerHTML =
          `<div class="doc-main"><b>${escapeHtml(doc.title)}</b>` +
          `<span>${escapeHtml(doc.source)} · ${doc.chunk_count} 切片 · ${doc.entity_count} 实体` +
          `${doc.page_count ? " · " + doc.page_count + " 页" : ""}</span></div>` +
          `<div class="doc-actions">` +
          `<a class="btn ghost" href="${API}/documents/${encodeURIComponent(doc.doc_id)}/file" target="_blank">原文</a>` +
          `<button class="btn ghost" data-action="view">详情</button>` +
          `<button class="btn ghost" data-action="delete">删除</button></div>`;
        item.querySelector('[data-action="view"]').addEventListener("click", () => {
          openSource({
            doc_id: doc.doc_id,
            doc_title: doc.title,
            section: "",
            snippet: doc.markdown || "",
            page: null,
          });
        });
        item.querySelector('[data-action="delete"]').addEventListener("click", async () => {
          if (!confirm(`确认删除《${doc.title}》及其索引？`)) return;
          try {
            await api(`/documents/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
            toast("文档已删除");
            loadDocuments();
            refreshMetrics();
          } catch (error) {
            toast(error.message, true);
          }
        });
        box.appendChild(item);
      });
    } catch (error) {
      /* 静默 */
    }
  }

  // ---------------- 知识图谱 ----------------
  async function loadGraph() {
    const entity = $("#graph-entity").value.trim();
    const hops = $("#graph-hops").value;
    const box = $("#graph-canvas");
    if (!entity) {
      toast("请输入实体名称", true);
      return;
    }
    box.innerHTML = '<div class="empty">加载子图…</div>';
    try {
      const graph = await api(`/graph?entity=${encodeURIComponent(entity)}&hops=${hops}`);
      renderGraph(graph);
    } catch (error) {
      box.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
    }
  }

  function nodeColor(type) {
    if (["certification", "standard", "regulation", "policy", "clause"].includes(type)) return "#6366f1";
    if (["product", "component", "spec"].includes(type)) return "#0d9488";
    if (["platform", "organization", "system"].includes(type)) return "#d97706";
    return "#94a3b8";
  }

  function renderGraph(graph) {
    const box = $("#graph-canvas");
    const nodes = graph.nodes || [];
    if (!nodes.length) {
      box.innerHTML = '<div class="empty">未找到实体及其邻域，请尝试其他名称</div>';
      $("#graph-edges").innerHTML = "";
      return;
    }
    const width = 760;
    const height = 420;
    const cx = width / 2;
    const cy = height / 2;
    const positions = new Map();
    const seeds = new Set((graph.seed_entities || []).map((name) => name.toLowerCase().replace(/\s+/g, "")));
    const ringNodes = nodes.filter((node) => !seeds.has(node.id));
    nodes.forEach((node, index) => {
      if (seeds.has(node.id)) {
        const offset = Array.from(seeds).indexOf(node.id);
        positions.set(node.id, { x: cx, y: cy + offset * 60 - 20 });
      }
    });
    const radius = Math.min(width, height) / 2 - 45;
    ringNodes.forEach((node, index) => {
      const angle = (index / ringNodes.length) * Math.PI * 2 - Math.PI / 2;
      positions.set(node.id, {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle) * 0.86,
      });
    });

    const edges = graph.edges || [];
    let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
    edges.forEach((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return;
      svg += `<line class="graph-edge" x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}"></line>`;
      if (edges.length <= 26) {
        svg += `<text class="graph-edge-label" x="${(source.x + target.x) / 2}" y="${(source.y + target.y) / 2 - 3}" text-anchor="middle">${escapeHtml(edge.relation)}</text>`;
      }
    });
    nodes.forEach((node) => {
      const point = positions.get(node.id);
      if (!point) return;
      const radius = 11 + Math.min(9, node.degree || 0);
      const color = nodeColor(node.type);
      svg += `<g class="graph-node" data-entity="${escapeHtml(node.label)}">` +
        `<circle cx="${point.x}" cy="${point.y}" r="${radius}" fill="${color}" opacity="0.9"></circle>` +
        `<text x="${point.x}" y="${point.y + radius + 13}" text-anchor="middle">${escapeHtml(node.label.slice(0, 16))}</text>` +
        `</g>`;
    });
    svg += "</svg>";
    box.innerHTML = svg;
    box.querySelectorAll(".graph-node").forEach((group) => {
      group.addEventListener("click", () => {
        $("#graph-entity").value = group.dataset.entity;
        loadGraph();
      });
    });
    $("#graph-edges").innerHTML = edges
      .slice(0, 40)
      .map((edge) => `<span class="edge-pill"><b>${escapeHtml(edge.source)}</b> —${escapeHtml(edge.relation)}→ <b>${escapeHtml(edge.target)}</b></span>`)
      .join("");
  }

  // ---------------- 评测 ----------------
  async function runEval() {
    const button = $("#btn-eval");
    button.disabled = true;
    button.textContent = "评测运行中…";
    $("#eval-status").innerHTML =
      '<span class="loading"><i></i><i></i><i></i></span> 正在逐条执行 Self-RAG 并计算四项指标…';
    try {
      const path = $("#eval-dataset").value.trim();
      const report = await api("/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(path ? { dataset_path: path } : {}),
      });
      renderEvalMetrics(report);
      $("#eval-status").textContent =
        `报告 ${report.report_id} · ${report.num_samples} 条样本 · 平均 ${report.avg_latency_ms} ms`;
      loadEvalReports();
      toast("评测完成");
    } catch (error) {
      $("#eval-status").textContent = `评测失败：${error.message}`;
      toast(error.message, true);
    } finally {
      button.disabled = false;
      button.textContent = "运行评测";
    }
  }

  function renderEvalMetrics(report) {
    const metrics = [
      ["Faithfulness", report.faithfulness],
      ["Answer Relevance", report.answer_relevance],
      ["Context Recall", report.context_recall],
      ["Context Precision", report.context_precision],
    ];
    $("#eval-metrics").innerHTML = metrics
      .map(([label, value]) => {
        const percent = Math.round(Number(value) * 100);
        return `<div class="score-card"><label>${label}</label><b>${Number(value).toFixed(3)}</b>` +
          `<div class="bar"><i style="width:${Math.max(2, percent)}%"></i></div></div>`;
      })
      .join("");
  }

  async function loadEvalReports() {
    try {
      const reports = await api("/eval/reports?limit=10");
      const tbody = $("#eval-table tbody");
      if (!reports.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-row">暂无评测报告</td></tr>';
        return;
      }
      tbody.innerHTML = "";
      reports.forEach((report) => {
        const row = document.createElement("tr");
        row.className = "clickable";
        row.innerHTML =
          `<td><b>${escapeHtml(report.report_id)}</b></td>` +
          `<td>${report.num_samples}</td>` +
          `<td>${Number(report.faithfulness).toFixed(3)}</td>` +
          `<td>${Number(report.answer_relevance).toFixed(3)}</td>` +
          `<td>${Number(report.context_recall).toFixed(3)}</td>` +
          `<td>${Number(report.context_precision).toFixed(3)}</td>` +
          `<td>${report.p95_latency_ms} ms</td>`;
        row.addEventListener("click", async () => {
          try {
            const detail = await api(`/eval/reports/${encodeURIComponent(report.report_id)}`);
            renderEvalRecords(detail.records || []);
            renderEvalMetrics(detail);
          } catch (error) {
            toast(error.message, true);
          }
        });
        tbody.appendChild(row);
      });
    } catch (error) {
      /* 静默 */
    }
  }

  function renderEvalRecords(records) {
    $("#eval-records").innerHTML = records
      .map((record) => {
        return `<div class="eval-record"><b>${escapeHtml(record.question)}</b>` +
          `<div class="muted">Faith ${record.faithfulness.toFixed(2)} · Rel ${record.answer_relevance.toFixed(2)} · ` +
          `Recall ${record.context_recall.toFixed(2)} · Precision ${record.context_precision.toFixed(2)} · ${record.latency_ms} ms</div></div>`;
      })
      .join("");
  }

  // ---------------- Trace ----------------
  async function loadTraces() {
    try {
      const traces = await api("/traces?limit=30");
      const box = $("#trace-list");
      if (!traces.length) {
        box.className = "trace-list empty";
        box.textContent = "暂无 Trace";
        return;
      }
      box.className = "trace-list";
      box.innerHTML = "";
      traces.forEach((trace) => {
        const item = document.createElement("div");
        item.className = "trace-item";
        const statusCls = trace.status === "success" ? "ok" : "bad";
        const question = (trace.metrics?.attributes?.question || trace.metrics?.attributes?.filename || trace.kind);
        item.innerHTML =
          `<div><b>${escapeHtml(trace.trace_id)}</b><div class="muted">${escapeHtml(String(question))}</div></div>` +
          `<div>${trace.kind} · <span class="${statusCls}">${trace.status}</span> · ${trace.latency_ms} ms</div>`;
        item.addEventListener("click", () => showTraceDetail(trace.trace_id));
        box.appendChild(item);
      });
      if (!$("#trace-detail").dataset.loaded) {
        showTraceDetail(traces[0].trace_id);
      }
    } catch (error) {
      /* 静默 */
    }
  }

  async function showTraceDetail(traceId) {
    try {
      const trace = await api(`/traces/${encodeURIComponent(traceId)}`);
      $("#trace-detail-id").textContent = trace.trace_id;
      const box = $("#trace-detail");
      box.className = "timeline";
      box.dataset.loaded = "1";
      box.innerHTML = "";
      (trace.steps || []).forEach((step) => {
        addStepTo(box, step.node, step.detail, step.elapsed_ms);
      });
      const tokens = trace.metrics?.tokens || {};
      const summary = document.createElement("div");
      summary.className = "warning";
      summary.textContent =
        `总耗时 ${trace.latency_ms} ms · prompt ${tokens.prompt || 0} tok · completion ${tokens.completion || 0} tok` +
        (trace.error ? ` · ${trace.error}` : "");
      box.appendChild(summary);
    } catch (error) {
      toast(error.message, true);
    }
  }

  function addStepTo(box, node, detail, elapsed) {
    const step = document.createElement("div");
    step.className = "step done";
    step.innerHTML =
      `<span class="step-dot"></span>` +
      `<div><strong>${escapeHtml(node)}</strong><p>${escapeHtml(detail || "")}</p></div>` +
      `<time>${Number(elapsed).toFixed(1)} ms</time>`;
    box.appendChild(step);
  }

  // ---------------- 绑定 ----------------
  function bindEvents() {
    bindNav();
    bindDrawer();
    $("#btn-ask").addEventListener("click", ask);
    $("#question").addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") ask();
    });
    $("#btn-clear").addEventListener("click", () => {
      $("#question").value = "";
      $("#answer-body").className = "answer-body empty";
      $("#answer-body").textContent = "等待提问…";
      $("#citations").innerHTML = "";
      $("#reflection-badges").innerHTML = "";
      $("#warnings").innerHTML = "";
      $("#retrieval-table tbody").innerHTML = '<tr><td colspan="8" class="empty-row">暂无召回结果</td></tr>';
      resetSteps("待命");
    });
    $$(".example").forEach((button) => {
      button.addEventListener("click", () => {
        $("#question").value = button.textContent;
        ask();
      });
    });
    $("#btn-upload").addEventListener("click", uploadDocument);
    $("#btn-samples").addEventListener("click", loadSamples);
    $("#btn-graph").addEventListener("click", loadGraph);
    $("#graph-entity").addEventListener("keydown", (event) => {
      if (event.key === "Enter") loadGraph();
    });
    $("#btn-eval").addEventListener("click", runEval);
    $("#eval-dataset").addEventListener("keydown", (event) => {
      if (event.key === "Enter") runEval();
    });
  }

  async function init() {
    bindEvents();
    await refreshHealth();
    await refreshMetrics();
    await Promise.all([loadDocuments(), loadTraces(), loadEvalReports()]);
    pollJobs();
    setInterval(refreshMetrics, 20000);
    setInterval(refreshHealth, 30000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
