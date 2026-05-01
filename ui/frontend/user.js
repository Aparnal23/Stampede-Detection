/** Backend API base. Empty = same origin (when the page is served by Flask on :5000). */
function getApiBase() {
  const { protocol, hostname, port } = window.location;
  if (port === "5000") return "";
  if (protocol === "file:") return "http://127.0.0.1:5000";
  const host = hostname || "127.0.0.1";
  return `${protocol}//${host}:5000`;
}
const API = getApiBase();

function friendlyNetworkError(err) {
  const msg = err && err.message ? String(err.message) : "";
  if (
    err &&
    (err.name === "TypeError" ||
      /network|fetch|load failed|failed to fetch/i.test(msg))
  ) {
    return "Cannot reach the server. In a terminal, run: python backend/app.py (leave it running), then reload this page.";
  }
  return msg || "Something went wrong.";
}

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function formatTime(ts) {
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return "";
  }
}

let maxSeenUserAlertId = -1;
let lastUserCrowdLevel = null;

function playUserAlertChime() {
  if (!window.AlertSound) return;
  void window.AlertSound.play().catch(() => {});
}

function userSoundsEnabled() {
  const cb = document.getElementById("user-sound-on");
  return cb && cb.checked;
}

function onNewUserAlertsSound(items, skipChime = false) {
  const ids = (items || [])
    .map((a) => Number(a.id))
    .filter((n) => Number.isFinite(n));
  const highest = ids.length ? Math.max(...ids) : 0;
  if (maxSeenUserAlertId < 0) {
    maxSeenUserAlertId = highest;
    return;
  }
  if (highest > maxSeenUserAlertId && userSoundsEnabled() && !skipChime) {
    playUserAlertChime();
  }
  if (highest > maxSeenUserAlertId) maxSeenUserAlertId = highest;
}

function setUserCrowdLevel(level) {
  const el = document.getElementById("user-crowd-level");
  if (!el) return;
  el.textContent = level;
  el.dataset.level = level;
}

if (window.AlertSound) {
  const userSoundCb = document.getElementById("user-sound-on");
  if (userSoundCb) window.AlertSound.unlockFromCheckbox(userSoundCb);
}

function renderAlerts(list) {
  const root = document.getElementById("user-alerts");
  root.replaceChildren();
  if (!list.length) {
    root.appendChild(el(`<p class="empty">No alerts yet.</p>`));
    return;
  }
  for (const a of list) {
    const src =
      a.source === "authority"
        ? '<span class="badge authority">Authority</span>'
        : '<span class="badge system">System</span>';
    root.appendChild(
      el(`
      <article class="alert-item alert-item--user">
        <div class="alert-meta">${src} · ${formatTime(a.ts)}</div>
        <div class="alert-msg">${escapeHtml(a.message)}</div>
      </article>
    `)
    );
  }
}

function renderPresenceFeed(items) {
  const root = document.getElementById("presence-list");
  root.replaceChildren();
  if (!items.length) {
    root.appendChild(el(`<p class="empty">No registrations yet.</p>`));
    return;
  }
  for (const p of items) {
    const chip = document.createElement("div");
    chip.className = "presence-chip";
    chip.innerHTML = `
      <span class="presence-chip-name">${escapeHtml(p.display_name || "—")}</span>
      <span class="presence-chip-meta"><code>${escapeHtml(p.anon_id)}</code> · ${escapeHtml(formatTime(p.ts))}</span>
    `;
    root.appendChild(chip);
  }
}

async function pollPublicSnapshot() {
  const res = await fetch(`${API}/api/public/snapshot`);
  if (!res.ok) throw new Error("snapshot");
  const data = await res.json();
  renderPresenceFeed(data.presence || []);
}

async function pollAlerts() {
  const res = await fetch(`${API}/api/user/alerts`);
  if (!res.ok) {
    renderAlerts([]);
    document.getElementById("user-alerts").replaceChildren(
      el(`<p class="empty">Could not load alerts.</p>`)
    );
    return;
  }
  const data = await res.json();
  const crowdLevel = data.crowd_level != null ? data.crowd_level : "Normal";
  const warningEntered =
    lastUserCrowdLevel !== null &&
    lastUserCrowdLevel !== "Warning" &&
    crowdLevel === "Warning";

  setUserCrowdLevel(crowdLevel);
  lastUserCrowdLevel = crowdLevel;

  const list = data.alerts || [];
  renderAlerts(list);

  if (warningEntered && userSoundsEnabled()) {
    playUserAlertChime();
    onNewUserAlertsSound(list, true);
  } else {
    onNewUserAlertsSound(list, false);
  }
}

async function postPresence(name) {
  const res = await fetch(`${API}/api/presence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || "Registration failed");
  }
  return data;
}

function setFeedback(msg, ok = true) {
  const fb = document.getElementById("checkin-feedback");
  fb.textContent = msg;
  fb.dataset.ok = ok ? "1" : "0";
}

document.getElementById("btn-register").addEventListener("click", async () => {
  const name = document.getElementById("user-name").value.trim();
  if (!name) {
    setFeedback("Please enter your name.", false);
    return;
  }
  try {
    const data = await postPresence(name);
    setFeedback(
      `You are registered as ${data.display_name} (${data.anon_id}). You will see alerts here.`,
      true
    );
    await pollPublicSnapshot();
  } catch (e) {
    setFeedback(friendlyNetworkError(e), false);
  }
});

document.getElementById("user-name").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    document.getElementById("btn-register").click();
  }
});

async function init() {
  try {
    await pollPublicSnapshot();
    await pollAlerts();
    setInterval(pollPublicSnapshot, 5000);
    setInterval(pollAlerts, 4000);
  } catch (e) {
    document.getElementById("user-alerts").replaceChildren(
      el(
        `<p class="empty">${escapeHtml(friendlyNetworkError(e))}</p>`
      )
    );
  }
}

init();
