/**
 * Lex AI — Frontend Application Logic
 */

const API = "/api";

// ── Authentication ────────────────────────────────────────────────────────────

function login() {
  const userId = document.getElementById("userId").value.trim();
  if (!userId) {
    alert("Please enter a User ID to continue.");
    return;
  }
  localStorage.setItem("user_id", userId);
  window.location.href = "/dashboard";
}

function goAgent() {
  window.location.href = "/agent";
}

// ── History loading ───────────────────────────────────────────────────────────

async function loadHistory() {
  const userId = localStorage.getItem("user_id");
  if (!userId) return;

  try {
    const res = await fetch(`${API}/history?user_id=${encodeURIComponent(userId)}`);
    const data = await res.json();

    const div = document.getElementById("history");
    if (!div) return;

    if (!data.history || data.history.length === 0) {
      div.innerHTML = "<p style='color:#64748b;'>No previous analyses found.</p>";
      return;
    }

    div.innerHTML = data.history
      .map(
        (h) => `
        <div>
          <b>${h.document_name}</b> | OCR Confidence: ${h.ocr_confidence}%<br/>
          <small>${h.created_at}</small>
        </div><hr/>`
      )
      .join("");
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

// ── Document analysis ─────────────────────────────────────────────────────────

async function analyze() {
  const fileInput = document.getElementById("doc");
  if (!fileInput || !fileInput.files.length) {
    alert("Please upload a document before analyzing.");
    return;
  }

  _setLoading(true);
  _clearResults();

  const form = new FormData();
  form.append("document", fileInput.files[0]);

  try {
    const res = await fetch(`${API}/analyze`, {
      method: "POST",
      body: form,
    });

    const data = await res.json();

    if (!res.ok) {
      _showError(data.error || "An unexpected error occurred.");
      return;
    }

    // ── Summary ───────────────────────────────────────────────────────────
    const summaryEl = document.getElementById("summary");
    if (summaryEl) {
      summaryEl.innerText = data.summary || "No summary generated.";
    }

    // ── OCR Metrics ───────────────────────────────────────────────────────
    if (data.ocr_metrics) {
      renderOcrMetrics(data.ocr_metrics);
    }

    // ── Risk Analysis (key: risk_analysis) ───────────────────────────────
    renderRisks(data.risk_analysis || {});

  } catch (err) {
    _showError("Network error — could not reach the server.");
    console.error("Analyze request failed:", err);
  } finally {
    _setLoading(false);
  }
}

// ── OCR Metrics rendering ─────────────────────────────────────────────────────

function renderOcrMetrics(metrics) {
  const panel = document.getElementById("ocr-metrics-panel");
  if (!panel) return;

  // Populate each metric card
  _setText("metric-words",      metrics.word_count  ?? "—");
  _setText("metric-chars",      metrics.char_count  ?? "—");
  _setText("metric-tokens",     metrics.token_count ?? "—");
  _setText("metric-confidence", metrics.confidence != null
    ? metrics.confidence.toFixed(1) + "%"
    : "—"
  );

  const statusEl = document.getElementById("metric-status");
  if (statusEl) {
    const isSuccess = metrics.status === "success";
    statusEl.innerText  = isSuccess ? "✓ Success" : "⚠ Low";
    statusEl.style.color = isSuccess ? "#22d3ee" : "#f59e0b";
  }

  // Animate in the panel
  panel.style.display = "block";
  panel.style.opacity = "0";
  panel.style.transform = "translateY(10px)";
  requestAnimationFrame(() => {
    panel.style.transition = "opacity 0.4s ease, transform 0.4s ease";
    panel.style.opacity    = "1";
    panel.style.transform  = "translateY(0)";
  });
}

// ── Risk rendering ────────────────────────────────────────────────────────────

function renderRisks(riskData) {
  const container = document.getElementById("risks");
  if (!container) return;
  container.innerHTML = "";

  const allRisks = [
    ...(riskData.rule_based || []),
    ...(riskData.ml_based   || []),
  ];

  if (allRisks.length === 0) {
    container.innerHTML = "<p>No significant legal risks detected.</p>";
    return;
  }

  allRisks.forEach((risk) => {
    const levelClass =
      risk.risk_level === "High"   ? "risk-high"   :
      risk.risk_level === "Medium" ? "risk-medium" :
                                     "risk-low";

    const clausePreview =
      risk.clause && risk.clause.length > 200
        ? risk.clause.slice(0, 200) + "…"
        : risk.clause || "";

    // Show category badge if available (rule_based results include it)
    const categoryBadge = risk.category
      ? `<span class="risk-category">${risk.category}</span>`
      : "";

    const card = document.createElement("div");
    card.className = `risk-card ${levelClass}`;
    card.innerHTML = `
      <div class="risk-card-header">
        <div class="risk-level">⚑ ${risk.risk_level}</div>
        ${categoryBadge}
      </div>
      <p><b>Clause:</b> ${clausePreview}</p>
      <p><b>Explanation:</b> ${risk.explanation || "Potential legal ambiguity detected."}</p>
      <div class="risk-source">Source: ${(risk.source || "").replace("_", " ")}</div>
    `;
    container.appendChild(card);
  });
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function _setLoading(isLoading) {
  const btn = document.querySelector(".analyze-btn");
  if (!btn) return;
  btn.disabled    = isLoading;
  btn.textContent = isLoading ? "Analyzing…" : "Analyze";
}

function _clearResults() {
  const summaryEl = document.getElementById("summary");
  if (summaryEl) summaryEl.innerText = "";

  const risksEl = document.getElementById("risks");
  if (risksEl) risksEl.innerHTML = "";

  const metricsPanel = document.getElementById("ocr-metrics-panel");
  if (metricsPanel) metricsPanel.style.display = "none";
}

function _showError(message) {
  const risksEl = document.getElementById("risks");
  if (risksEl) {
    risksEl.innerHTML = `<p style="color:#ef4444;font-weight:600;">⚠ ${message}</p>`;
  }
}

function _setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.innerText = value;
}
