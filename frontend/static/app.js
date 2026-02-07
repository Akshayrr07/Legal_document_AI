const API = "/api";

function login() {
  const userId = document.getElementById("userId").value;
  if (!userId) return alert("User ID required");
  localStorage.setItem("user_id", userId);
  window.location.href = "/dashboard";
}

function goAgent() {
  window.location.href = "/agent";
}

async function loadHistory() {
  const userId = localStorage.getItem("user_id");
  if (!userId) return;

  const res = await fetch(`${API}/history?user_id=${userId}`);
  const data = await res.json();

  const div = document.getElementById("history");
  div.innerHTML = data.history.map(h =>
    `<div>
      <b>${h.document_name}</b> | OCR: ${h.ocr_confidence}<br/>
      <small>${h.created_at}</small>
    </div><hr/>`
  ).join("");
}

async function analyze() {
  const fileInput = document.getElementById("doc");
  if (!fileInput.files.length) return alert("Upload a document");

  const form = new FormData();
  form.append("document", fileInput.files[0]);

  const res = await fetch(`${API}/analyze`, {
    method: "POST",
    body: form
  });

  const data = await res.json();

  document.getElementById("summary").innerText = data.summary || "";
  function renderRisks(riskData) {
  const container = document.getElementById("risks");
  container.innerHTML = "";

  const allRisks = [
    ...(riskData.rule_based || []),
    ...(riskData.ml_based || [])
  ];

  if (allRisks.length === 0) {
    container.innerHTML = "<p>No significant legal risks detected.</p>";
    return;
  }

  allRisks.forEach(risk => {
    const levelClass =
      risk.risk_level === "High" ? "risk-high" :
      risk.risk_level === "Medium" ? "risk-medium" :
      "risk-low";

    const card = document.createElement("div");
    card.className = `risk-card ${levelClass}`;

    card.innerHTML = `
      <div class="risk-level">Risk Level: ${risk.risk_level}</div>
      <p><b>Clause:</b> ${risk.clause.slice(0, 200)}...</p>
      <p><b>Explanation:</b> ${risk.explanation || "Potential legal ambiguity detected."}</p>
      <div class="risk-source">Source: ${risk.source.replace("_", " ")}</div>
    `;

    container.appendChild(card);
  });
}

  renderRisks(data.risks || {});
}
