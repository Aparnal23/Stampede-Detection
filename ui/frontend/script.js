function getApiBase() {
  const { protocol, hostname, port } = window.location;
  if (port === "5000") return "";
  if (protocol === "file:") return "http://127.0.0.1:5000";
  const host = hostname || "127.0.0.1";
  return `${protocol}//${host}:5000`;
}
const API = getApiBase();

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}

function formatTime(ts) {
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return "";
  }
}

function setCrowdLevel(level) {
  const pill = document.getElementById("crowd-level");
  pill.textContent = level;
  pill.dataset.level = level;
}

let maxSeenAuthorityAlertId = -1;

function playAuthorityAlertChime() {
  if (!window.AlertSound) return;
  void window.AlertSound.play().catch(() => {});
}

function authoritySoundsEnabled() {
  const cb = document.getElementById("authority-sound-on");
  return cb && cb.checked;
}

function onNewAlertsForAuthoritySound(items, skipChime = false) {
  const ids = (items || [])
    .map((a) => Number(a.id))
    .filter((n) => Number.isFinite(n));
  const highest = ids.length ? Math.max(...ids) : 0;
  if (maxSeenAuthorityAlertId < 0) {
    maxSeenAuthorityAlertId = highest;
    return;
  }
  if (
    highest > maxSeenAuthorityAlertId &&
    authoritySoundsEnabled() &&
    !skipChime
  ) {
    playAuthorityAlertChime();
  }
  if (highest > maxSeenAuthorityAlertId) maxSeenAuthorityAlertId = highest;
}

let lastAuthorityCrowdLevel = null;

if (window.AlertSound) {
  const authoritySoundCb = document.getElementById("authority-sound-on");
  if (authoritySoundCb) window.AlertSound.unlockFromCheckbox(authoritySoundCb);
}

function renderAlerts(items) {
  const root = document.getElementById("alerts");
  root.replaceChildren();
  if (!items.length) {
    root.appendChild(el(`<p class="empty">No alerts yet.</p>`));
    return;
  }
  for (const a of items) {
    const src =
      a.source === "authority"
        ? '<span class="badge authority">Authority</span>'
        : '<span class="badge system">System</span>';
    const row = el(`
      <article class="alert-item">
        <div class="alert-meta">${src} · ${formatTime(a.ts)}</div>
        <div class="alert-msg">${escapeHtml(a.message)}</div>
      </article>
    `);
    root.appendChild(row);
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

const _lastFeedSrc = {};

function feedAssetUrl(url) {
  if (!url) return "";
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/")) return `${API}${url}`;
  return `${API}/${url}`;
}

// ─── FIX: renderVideoFeeds now sets BOTH data-camera AND data-zone-name
//     so that index.html's _findCardEl(zone.name) can locate the card. ───
function renderVideoFeeds(zones) {
  const root = document.getElementById("video-feeds");
  if (!root || !zones) return;
  if (!zones.length) {
    root.replaceChildren();
    root.appendChild(
      el(
        `<p class="empty">No CCTV zones. Add photos to the <code>cctv/</code> folder and edit <code>cctv_zones.json</code>, then restart the server.</p>`
      )
    );
    return;
  }

  const emptyEl = root.querySelector("p.empty");
  if (emptyEl) emptyEl.remove();

  const seen = new Set();
  for (const z of zones) {
    const cid = z.camera_id;
    seen.add(cid);
    let panel = root.querySelector(`[data-camera="${CSS.escape(cid)}"]`);
    if (!panel) {
      panel = el(`
        <div class="video-panel" data-camera="${escapeHtml(cid)}" data-zone-name="${escapeHtml(z.name)}">
          <div class="video-panel-head">
            <strong class="video-zone-name"></strong>
            <code class="video-cam-id"></code>
            <span class="feed-badge"></span>
          </div>
          <div class="zone-feed-wrap">
            <div class="feed-media"></div>
          </div>
        </div>
      `);
      root.appendChild(panel);
    }

    // Always keep data-zone-name in sync (in case zones were renamed)
    panel.setAttribute("data-zone-name", z.name);
    panel.querySelector(".video-zone-name").textContent = z.name;
    panel.querySelector(".video-cam-id").textContent = cid;

    const state = z.feed_state || "normal";
    const badge = panel.querySelector(".feed-badge");
    badge.innerHTML =
      state === "abnormal"
        ? '<span class="badge system">Abnormal</span>'
        : '<span class="badge authority">Normal</span>';

    const url = feedAssetUrl(z.current_video_url || "");
    const isImage =
      z.feed_media === "image" ||
      /\.(jpe?g|png|gif|webp|svg)(\?|$)/i.test(url);

    // Use the zone-feed-wrap slot (wraps feed-media) so overlay badges work
    const wrap = panel.querySelector(".zone-feed-wrap");
    const slot = panel.querySelector(".feed-media");

    if (isImage && slot.querySelector("video")) slot.replaceChildren();
    if (!isImage && slot.querySelector("img.feed-image")) slot.replaceChildren();

    if (isImage) {
      let img = slot.querySelector("img.feed-image");
      if (!img) {
        slot.replaceChildren();
        img = document.createElement("img");
        img.className = "feed-image";
        img.alt = `${z.name} — CCTV`;
        slot.appendChild(img);
      }
      if (_lastFeedSrc[cid] !== url) {
        _lastFeedSrc[cid] = url;
        img.src = url;
      }
      img.classList.toggle("feed-image--alert", state === "abnormal");
    } else {
      let video = slot.querySelector("video");
      if (!video) {
        slot.replaceChildren();
        video = document.createElement("video");
        video.setAttribute("muted", "");
        video.setAttribute("playsinline", "");
        video.setAttribute("controls", "");
        video.preload = "metadata";
        slot.appendChild(video);
      }
      if (_lastFeedSrc[cid] !== url) {
        _lastFeedSrc[cid] = url;
        video.src = url;
        video.play().catch(() => {});
      }
    }
  }

  root.querySelectorAll(".video-panel").forEach((panel) => {
    const id = panel.getAttribute("data-camera");
    if (id && !seen.has(id)) {
      panel.remove();
      delete _lastFeedSrc[id];
    }
  });
}

// ─── FIX: renderCctvTable now renders all 6 columns to match the
//     index.html table header: Camera | Label | Feed state |
//     People (CSRNet) | Model prediction | Risk score ───
function renderCctvTable(zones) {
  const tb = document.querySelector("#cctv-table tbody");
  if (!tb) return;
  tb.replaceChildren();

  if (!zones || !zones.length) {
    tb.appendChild(
      el(`<tr><td colspan="6" class="empty">No camera data.</td></tr>`)
    );
    return;
  }

  for (const z of zones) {
    const inf   = z.inference || {};
    const state = z.feed_state || "normal";

    const feed =
      state === "abnormal"
        ? '<span class="badge system">Abnormal</span>'
        : '<span class="badge authority">Normal</span>';

    // ── People count cell ──
    let countCell;
    if (inf.status === "done") {
      const cnt = z.cctv_people_estimate;
      countCell =
        cnt > 0
          ? `<span class="count-real">👥 ${cnt}</span>`
          : `<span class="count-zero">0</span>`;
    } else if (inf.status === "running" || inf.status === "queued") {
      countCell = `<span class="count-wait"><span class="zrs-spinner"></span>Analysing…</span>`;
    } else if (inf.status === "error") {
      countCell = `<span style="color:#e74c3c" title="${escapeHtml(inf.error || "")}">⚠ Error</span>`;
    } else {
      countCell = `<span class="count-zero">—</span>`;
    }

    // ── Prediction cell ──
    let predCell  = "—";
    let scoreCell = "—";
    if (inf.status === "done") {
      const pred = inf.overall_prediction || "NORMAL";
      const risk = inf.overall_risk       || "NORMAL";
      const col  =
        pred === "PRE_STAMPEDE"  ? "#e74c3c"
        : risk === "MEDIUM"      ? "#e6900a"
        : "#27ae60";
      predCell  = `<span style="color:${col};font-weight:800">${escapeHtml(pred)}</span>`;
      const pk  = inf.peak_risk ? inf.peak_risk.smooth.toFixed(3) : "—";
      scoreCell = `<span style="color:${col}">${pk}</span>`;
    } else if (inf.status === "running" || inf.status === "queued") {
      predCell  = `<span style="color:#666"><span class="zrs-spinner"></span>…</span>`;
      scoreCell = `<span style="color:#666">—</span>`;
    } else if (inf.status === "error") {
      predCell  = `<span style="color:#e74c3c" title="${escapeHtml(inf.error || "")}">Error</span>`;
      scoreCell = `<span style="color:#e74c3c">—</span>`;
    }

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${escapeHtml(z.camera_id)}</code></td>
      <td>${escapeHtml(z.name)}</td>
      <td>${feed}</td>
      <td>${countCell}</td>
      <td>${predCell}</td>
      <td>${scoreCell}</td>
    `;
    tb.appendChild(tr);
  }
}

function renderPresenceTable(rows) {
  const tb = document.querySelector("#presence-table tbody");
  tb.replaceChildren();
  if (!rows || !rows.length) {
    tb.appendChild(
      el(`<tr><td colspan="3" class="empty">No registrations yet.</td></tr>`)
    );
    return;
  }
  for (const p of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(p.display_name || "—")}</td>
      <td><code>${escapeHtml(p.anon_id)}</code></td>
      <td>${escapeHtml(formatTime(p.ts))}</td>
    `;
    tb.appendChild(tr);
  }
}

async function pollStatus() {
  const res = await fetch(`${API}/api/status`);
  if (!res.ok) throw new Error("status failed");
  const data = await res.json();
  const level = data.level;
  const warningEntered =
    lastAuthorityCrowdLevel !== null &&
    lastAuthorityCrowdLevel !== "Warning" &&
    level === "Warning";
  setCrowdLevel(level);
  lastAuthorityCrowdLevel = level;

  const alertItems = data.alerts || [];
  renderAlerts(alertItems);
  if (warningEntered && authoritySoundsEnabled()) {
    playAuthorityAlertChime();
    onNewAlertsForAuthoritySound(alertItems, true);
  } else {
    onNewAlertsForAuthoritySound(alertItems, false);
  }
  const totalEl = document.getElementById("total-registrations");
  if (totalEl && data.total_registrations != null) {
    totalEl.textContent = `Registered users: ${data.total_registrations}`;
  }
  const zones = data.zones || [];
  renderVideoFeeds(zones);
  renderCctvTable(zones);
  renderPresenceTable(data.presence_detail || []);
}

document
  .getElementById("broadcast-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const feedback = document.getElementById("send-feedback");
    feedback.textContent = "";
    const message = document.getElementById("message").value.trim();
    try {
      const res = await fetch(`${API}/api/authority/broadcast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        feedback.textContent = data.error || "Send failed.";
        return;
      }
      feedback.textContent = data.delivered_to || "Sent.";
      document.getElementById("message").value = "";
      await pollStatus();
    } catch {
      feedback.textContent = "Network error — is the server running?";
    }
  });

function initPublicUrl() {
  const elUrl = document.getElementById("public-url");
  if (elUrl) {
    elUrl.textContent = `${window.location.origin}/user`;
  }
}

async function loadQRCode() {
  try {
    const res = await fetch(`${API}/api/qrcode`);
    if (!res.ok) throw new Error("qrcode failed");
    const data = await res.json();
    const qrImg = document.getElementById("qr-image");
    const regUrl = document.getElementById("registration-url");
    if (qrImg) qrImg.src = data.qr_code;
    if (regUrl) regUrl.textContent = data.registration_url;
  } catch (e) {
    console.error("Could not load QR code:", e);
  }
}

async function uploadVideo() {
  const fileInput = document.getElementById("videoFile");
  const status = document.getElementById("uploadStatus");

  if (!fileInput.files.length) {
    status.innerText = "Please select a file";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch(`${API}/api/upload-video`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (data.status === "uploaded") {
      status.innerText =
        data.inference_message
          ? `✅ ${data.status} — ${data.inference_message}`
          : "✅ Uploaded successfully!";
    } else {
      status.innerText = data.error || "Upload failed";
    }
  } catch (err) {
    status.innerText = "Upload failed";
  }
}

async function init() {
  initPublicUrl();
  loadQRCode();
  try {
    await pollStatus();
    setInterval(pollStatus, 3000);
  } catch {
    setCrowdLevel("Error");
    document.getElementById("alerts").replaceChildren(
      el(
        `<p class="empty">Cannot reach API. Run the Flask app and open <code>http://127.0.0.1:5000/</code>.</p>`
      )
    );
  }
}

init();