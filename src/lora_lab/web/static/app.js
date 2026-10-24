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

let activeFamily = "";

function updateLoraCounts() {
  const total = document.querySelectorAll("input[data-pick]:checked").length;
  const counter = document.getElementById("selected-count");
  if (counter) counter.textContent = String(total);

  document.querySelectorAll(".lora-group").forEach((group) => {
    const count = group.querySelectorAll("input[data-pick]:checked").length;
    const badge = group.querySelector("[data-group-count]");
    if (badge) {
      badge.textContent = `${count} gewählt`;
      badge.classList.toggle("ok", count > 0);
    }
  });
}

function toggleAll(checked) {
  document.querySelectorAll("input[data-pick]").forEach((box) => {
    const group = box.closest(".lora-group");
    if (!group || !group.hidden) box.checked = checked;
  });
  updateLoraCounts();
}

function toggleGroup(el, checked) {
  const group = el.closest(".lora-group");
  if (!group) return;
  group.querySelectorAll("input[data-pick]").forEach((box) => { box.checked = checked; });
  if (checked) group.open = true;
  updateLoraCounts();
}

function _finalRowInGroup(group) {
  const rows = Array.from(group.querySelectorAll("tr[data-epoch]"));
  if (!rows.length) return null;
  const finalRow = rows.find((r) => parseInt(r.dataset.epoch || "0", 10) === 0);
  if (finalRow) return finalRow;
  return rows.reduce((best, r) => (
    parseInt(r.dataset.epoch || "0", 10) >= parseInt(best.dataset.epoch || "0", 10) ? r : best
  ), rows[0]);
}

function selectGroupFinal(el) {
  const group = el.closest(".lora-group");
  if (!group) return;
  group.querySelectorAll("input[data-pick]").forEach((box) => { box.checked = false; });
  const row = _finalRowInGroup(group);
  if (row) {
    const box = row.querySelector("input[data-pick]");
    if (box) box.checked = true;
  }
  group.open = true;
  updateLoraCounts();
}

function selectFinalPerGroup() {
  document.querySelectorAll("input[data-pick]").forEach((box) => { box.checked = false; });
  document.querySelectorAll(".lora-group").forEach((group) => {
    if (group.hidden) return;
    const row = _finalRowInGroup(group);
    if (row) {
      const box = row.querySelector("input[data-pick]");
      if (box) box.checked = true;
    }
  });
  updateLoraCounts();
}

function selectGroupStep(el, step) {
  const group = el.closest(".lora-group");
  if (!group) return;
  const rows = Array.from(group.querySelectorAll("tr[data-epoch]"));
  rows.forEach((row, idx) => {
    const box = row.querySelector("input[data-pick]");
    if (box) box.checked = ((idx + 1) % step === 0) || (idx === rows.length - 1);
  });
  group.open = true;
  updateLoraCounts();
}

function setGroupSubject(el, subject) {
  const group = el.closest(".lora-group");
  if (!group) return;
  group.querySelectorAll("select[data-subject-select]").forEach((sel) => {
    sel.value = subject;
    const row = sel.closest("tr");
    if (row) row.dataset.subject = subject;
  });
}

function toggleGroupsOpen(open) {
  document.querySelectorAll(".lora-group").forEach((group) => {
    if (!group.hidden) group.open = open;
  });
}

function setFamilyFilter(btn, family) {
  activeFamily = family;
  document.querySelectorAll(".filter-pill").forEach((pill) => {
    pill.classList.toggle("active", pill === btn);
  });
  filterLoras();
}

function filterLoras() {
  const query = (document.getElementById("lora-search")?.value || "").trim().toLowerCase();
  const onlySelected = Boolean(document.getElementById("lora-only-selected")?.checked);

  document.querySelectorAll(".lora-group").forEach((group) => {
    const familyMatch = !activeFamily || group.dataset.family === activeFamily;
    const folderMatch = !query || (group.dataset.folder || "").includes(query);
    let visibleRows = 0;

    group.querySelectorAll("tr[data-epoch]").forEach((row) => {
      const checked = Boolean(row.querySelector("input[data-pick]")?.checked);
      const searchMatch = folderMatch || !query || (row.dataset.search || "").includes(query);
      const show = familyMatch && searchMatch && (!onlySelected || checked);
      row.hidden = !show;
      if (show) visibleRows += 1;
    });

    group.hidden = visibleRows === 0;
    if ((query || onlySelected) && visibleRows > 0) group.open = true;
  });
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
  updateLoraCounts();
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

// -- report gallery, compare & lightbox -----------------------------------

function toggleOpGallery(id) {
  const row = document.getElementById(id);
  if (row) row.hidden = !row.hidden;
}

function filterCompare() {
  const scen = document.getElementById("cmp-scenario")?.value || "";
  const seed = document.getElementById("cmp-seed")?.value || "";
  document.querySelectorAll(".compare-block").forEach((block) => {
    const okScen = !scen || block.dataset.scenario === scen;
    const okSeed = !seed || block.dataset.seed === seed;
    block.hidden = !(okScen && okSeed);
  });
}

function jumpToCompare(scenario) {
  const sel = document.getElementById("cmp-scenario");
  if (sel) {
    sel.value = scenario;
    filterCompare();
  }
  const section = document.getElementById("compare-section");
  if (section) section.scrollIntoView({ behavior: "smooth" });
}

function openImageModal(el) {
  const d = el.dataset;
  const modal = document.getElementById("img-modal");
  if (!modal) return;
  document.getElementById("im-title").textContent = `${d.lora} · ep${d.epoch} · s${d.strength}`;
  document.getElementById("im-img").src = d.path;
  document.getElementById("im-link").href = d.path;
  document.getElementById("im-open-btn").href = d.path;
  document.getElementById("im-lora").textContent = d.lora || "–";
  document.getElementById("im-ep-str").textContent = `Epoche ${d.epoch} · Stärke ${d.strength}`;
  document.getElementById("im-scen-seed").textContent = `${d.scenario} · Seed ${d.seed}`;
  document.getElementById("im-status").textContent = d.status || "–";
  document.getElementById("im-utility").textContent = d.utility || "–";
  document.getElementById("im-identity").textContent = d.identity || "–";
  document.getElementById("im-adherence").textContent = d.adherence || "–";
  document.getElementById("im-quality").textContent = d.quality || "–";
  const adhRow = document.getElementById("im-adherence-details-row");
  const adhVal = document.getElementById("im-adherence-details");
  if (adhRow && adhVal) {
    adhVal.textContent = d.adherenceDetails || "";
    adhRow.hidden = !d.adherenceDetails;
  }
  const leakRow = document.getElementById("im-leakage-row");
  const leakVal = document.getElementById("im-leakage");
  if (leakRow && leakVal) {
    leakVal.textContent = d.leakage || "";
    leakRow.hidden = !d.leakage;
  }
  modal.classList.add("open");
}

function closeImageModal() {
  const modal = document.getElementById("img-modal");
  if (modal) modal.classList.remove("open");
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeImageModal();
});
