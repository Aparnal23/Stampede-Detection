function getApiBase() {
  const { protocol, hostname, port } = window.location;
  if (port === "5000") return "";
  if (protocol === "file:") return "http://127.0.0.1:5000";
  const host = hostname || "127.0.0.1";
  return `${protocol}//${host}:5000`;
}

const API = getApiBase();

function formatTime(ts) {
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return "";
  }
}

function colorForLevel(level) {
  switch (level) {
    case "Normal":
      return "normal";
    case "Warning":
      return "warning";
    case "Critical":
      return "critical";
    default:
      return "normal";
  }
}

function renderMetrics(data) {
  const container = document.getElementById("metrics-container");
  container.innerHTML = "";

  if (data.total_predictions === 0) {
    container.innerHTML = `
      <div class="no-data" style="grid-column: 1 / -1;">
        No predictions recorded yet. The system is accumulating data...
      </div>
    `;
    return;
  }

  const accuracyPercent = (data.accuracy * 100).toFixed(1);
  const accuracyColor =
    accuracyPercent >= 80 ? "metric-percent" : 
    accuracyPercent >= 60 ? "metric-percent warning" : 
    "metric-percent critical";

  container.innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Overall Accuracy</div>
      <div class="metric-value">${accuracyPercent}%</div>
      <div class="${accuracyColor}">${data.total_predictions} predictions</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Total Predictions</div>
      <div class="metric-value">${data.total_predictions}</div>
      <div class="metric-percent">Samples evaluated</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Correct Predictions</div>
      <div class="metric-value">${Math.round(data.accuracy * data.total_predictions)}</div>
      <div class="metric-percent">${(data.accuracy * 100).toFixed(2)}%</div>
    </div>
  `;
}

function renderConfusionMatrix(data) {
  if (!data.confusion_matrix || Object.keys(data.confusion_matrix).length === 0) {
    return;
  }

  const section = document.getElementById("confusion-section");
  const container = document.getElementById("confusion-container");
  section.style.display = "block";

  const categories = ["Normal", "Warning", "Critical"];
  let html = `
    <table class="confusion-matrix">
      <thead>
        <tr>
          <th style="text-align: left;">Actual \ Predicted</th>
          ${categories.map((c) => `<th>${c}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
  `;

  for (const actualLabel of categories) {
    html += `<tr><td class="label">${actualLabel}</td>`;
    for (const predLabel of categories) {
      const value = data.confusion_matrix[actualLabel]?.[predLabel] || 0;
      const isCorrect = actualLabel === predLabel;
      const cellClass = isCorrect ? "correct" : "";
      html += `<td class="${cellClass}">${value}</td>`;
    }
    html += `</tr>`;
  }

  html += `
      </tbody>
    </table>
  `;

  container.innerHTML = html;
}

function renderPerClassMetrics(data) {
  if (!data.precision) {
    return;
  }

  const section = document.getElementById("per-class-section");
  const container = document.getElementById("per-class-container");
  section.style.display = "block";

  const categories = ["Normal", "Warning", "Critical"];
  let html = "";

  for (const category of categories) {
    const precision = ((data.precision[category] || 0) * 100).toFixed(1);
    const recall = ((data.recall[category] || 0) * 100).toFixed(1);
    const f1 = ((data.f1_score[category] || 0) * 100).toFixed(1);

    html += `
      <div class="metric-class-card">
        <h4>${category} Level</h4>
        <div class="class-metric">
          <div class="class-metric-label">Precision</div>
          <div class="class-metric-value">${precision}%</div>
        </div>
        <div class="class-metric">
          <div class="class-metric-label">Recall</div>
          <div class="class-metric-value">${recall}%</div>
        </div>
        <div class="class-metric">
          <div class="class-metric-label">F1-Score</div>
          <div class="class-metric-value">${f1}%</div>
        </div>
      </div>
    `;
  }

  container.innerHTML = html;
}

function renderHistory(data) {
  if (!data.history || data.history.length === 0) {
    return;
  }

  const section = document.getElementById("history-section");
  const container = document.getElementById("history-container");
  section.style.display = "block";

  let html = "";
  for (const item of data.history) {
    const isCorrect = item.predicted === item.actual;
    const predictedColor = colorForLevel(item.predicted);
    const actualColor = colorForLevel(item.actual);

    html += `
      <div class="history-item" style="background: ${isCorrect ? "#f0fff4" : "#fff5f5"};">
        <div class="history-time">${formatTime(item.timestamp)}</div>
        <div class="history-labels">
          <span class="label-badge ${predictedColor}">Predicted: ${item.predicted}</span>
          <span class="arrow">→</span>
          <span class="label-badge ${actualColor}">Actual: ${item.actual}</span>
          ${isCorrect ? '<span style="color: #28a745; font-weight: bold; margin-left: 0.5rem;">✓</span>' : '<span style="color: #dc3545; font-weight: bold; margin-left: 0.5rem;">✗</span>'}
        </div>
      </div>
    `;
  }

  container.innerHTML = html;
}

function updateLastUpdatedTime() {
  const now = new Date();
  document.getElementById("last-updated").textContent =
    "Last updated: " + now.toLocaleTimeString();
}

async function fetchAndRenderMetrics() {
  const container = document.getElementById("metrics-container");

  try {
    const response = await fetch(`${API}/api/evaluation/metrics`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    renderMetrics(data);
    renderConfusionMatrix(data);
    renderPerClassMetrics(data);
    renderHistory(data);
    updateLastUpdatedTime();
  } catch (error) {
    container.innerHTML = `
      <div class="status-error" style="grid-column: 1 / -1;">
        Error fetching metrics: ${error.message}<br>
        <small>Make sure the backend server is running.</small>
      </div>
    `;
    console.error("Error fetching metrics:", error);
  }
}

// Initial load
document.addEventListener("DOMContentLoaded", () => {
  fetchAndRenderMetrics();

  // Set up refresh button
  document.getElementById("refresh-btn").addEventListener("click", () => {
    fetchAndRenderMetrics();
  });

  // Auto-refresh every 10 seconds
  setInterval(fetchAndRenderMetrics, 10000);
});
