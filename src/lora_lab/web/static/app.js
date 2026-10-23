// Minimal vanilla JS: attribute rows, folder browser, run polling.

function addAttrRow(id = "", question = "", weight = "1") {
  const host = document.getElementById("attr-rows");
  if (!host) return;
  const row = document.createElement("div");
  row.className = "attr-row";
  row.innerHTML = `
    <input name="attr_id" placeholder="id" value="${escapeHtml(id)}">
    <input name="attr_question" placeholder="question the judge answers yes/no" value="${escapeHtml(question)}">
    <input name="attr_weight" placeholder="1" value="${escapeHtml(weight)}">
    <button type="button" class="danger" onclick="this.parentNode.remove()">×</button>`;
  host.appendChild(row);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// -- folder browser -------------------------------------------------------

let browseTarget = null;

function openBrowser(targetId) {
  browseTarget = targetId;
  document.getElementById("browser").classList.add("open");
  loadPath("");
}

function closeBrowser() {
  document.getElementById("browser").classList.remove("open");
}

async function loadPath(path) {
  const list = document.getElementById("browser-list");
  const crumb = document.getElementById("browser-path");
  list.innerHTML = "…";
  const response = await fetch("/api/browse?path=" + encodeURIComponent(path));
  const data = await response.json();
  if (data.error) {
    list.innerHTML = `<div class="err">${escapeHtml(data.error)}</div>`;
    return;
  }
  crumb.textContent = data.path || "Mounted folders";
  let html = "";
  if (!data.path) {
    for (const root of data.roots) {
      html += `<div class="dir" onclick="loadPath('${escapeHtml(root.path)}')">${escapeHtml(root.name)}</div>`;
    }
  } else {
    if (data.parent) {
      html += `<div class="dir" onclick="loadPath('${escapeHtml(data.parent)}')">..</div>`;
    }
    for (const dir of data.dirs) {
      html += `<div class="dir" onclick="loadPath('${escapeHtml(dir.path)}')">
                 <span>${escapeHtml(dir.name)}/</span></div>`;
    }
    html += `<div class="hint">${data.images} image(s) in this folder</div>`;
    document.getElementById("browser-use").dataset.path = data.path;
  }
  list.innerHTML = html;
}

function useBrowserPath() {
  const path = document.getElementById("browser-use").dataset.path;
  if (path && browseTarget) {
    document.getElementById(browseTarget).value = path;
  }
  closeBrowser();
}

// -- LoRA selection -------------------------------------------------------

function toggleAll(checked) {
  document.querySelectorAll("input[data-pick]").forEach((box) => { box.checked = checked; });
}

function selectLatestPerSubject() {
  const best = {};
  document.querySelectorAll("tr[data-subject]").forEach((row) => {
    const subject = row.dataset.subject;
    const epoch = parseInt(row.dataset.epoch || "0", 10);
    if (!best[subject] || epoch > best[subject].epoch) best[subject] = { epoch, row };
  });
  document.querySelectorAll("input[data-pick]").forEach((box) => { box.checked = false; });
  Object.values(best).forEach(({ row }) => {
    const box = row.querySelector("input[data-pick]");
    if (box) box.checked = true;
  });
}

// -- run polling ----------------------------------------------------------

async function pollRun() {
  const root = document.getElementById("run-live");
  if (!root) return;
  try {
    const response = await fetch("/api/run");
    const data = await response.json();
    const bar = document.getElementById("run-bar");
    const label = document.getElementById("run-label");
    const log = document.getElementById("run-log");
    const total = data.progress.total || 0;
    const done = data.progress.done || 0;
    const pct = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
    if (bar) bar.style.width = pct + "%";
    if (label) {
      label.textContent = data.running
        ? `${data.kind} · ${done}/${total || "?"} (${pct}%)`
        : (data.kind ? `last run: ${data.kind}${data.error ? " · " + data.error : " · finished"}` : "idle");
    }
    if (log) {
      log.textContent = (data.log || []).join("\n");
      log.scrollTop = log.scrollHeight;
    }
    document.querySelectorAll("[data-run-button]").forEach((button) => {
      button.disabled = data.running;
    });
    const stop = document.getElementById("run-stop");
    if (stop) stop.disabled = !data.running;
  } catch (error) {
    // transient; keep polling
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("run-live")) {
    pollRun();
    setInterval(pollRun, 3000);
  }
});
