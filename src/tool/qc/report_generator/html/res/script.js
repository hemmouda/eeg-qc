// DATA is injected by the template before this script runs.

// ── Info grid ─────────────────────────────────────────────────────────────
const infoItems = [
  ["Patient ID", DATA.patient_id],
  ["Name", DATA.full_name],
  ["Sex", DATA.sex],
  ["Recording Date", DATA.recording_date],
  ["Duration", DATA.duration],
  ["Quality", DATA.quality_up],
];

const ig = document.getElementById("info-grid");
infoItems.forEach(([label, value]) => {
  ig.innerHTML += `<div class="info-card">
    <div class="label">${label}</div>
    <div class="value">${value}</div>
  </div>`;
});

document.getElementById("quality-justification").textContent = DATA.quality_justification;

// ── Remarks ───────────────────────────────────────────────────────────────
const rmk = document.getElementById("remark-list");
if (DATA.remarks.length === 0) {
  rmk.innerHTML = `<div class="no-remarks">No special remarks for this recording.</div>`;
} else {
  DATA.remarks.forEach(remark => {
    rmk.innerHTML += `<div class="remark-card">
      <div class="card-title">${remark.title}</div>
      <div class="card-desc">${remark.description}</div>
    </div>`;
  });
}

// ── Channels ──────────────────────────────────────────────────────────────
const cl = document.getElementById("channel-list");

DATA.channels.forEach((ch, ci) => {
  const item = document.createElement("div");
  item.className = "channel-item";
  item.id = `ch-${ci}`;

  const spectrogramHtml = ch.spectrogram_b64
    ? `<div class="spectrogram-wrap">
             <img src="data:image/png;base64,${ch.spectrogram_b64}" alt="Spectrogram for ${ch.name}">
           </div>`
    : "";

  item.innerHTML = `
    <div class="channel-header" onclick="toggleChannel(${ci})">
      <span class="traffic-light ${ch.quality}" title="${escapeAttr(ch.quality_hover)}"></span>
      <span class="ch-name">${ch.name}</span>
      <span class="ch-epoch-info">Epochs — ${ch.n_epochs} × ${ch.epoch_duration_s}s</span>
      <span class="ch-arrow">⌄</span>
    </div>
    <div class="channel-body">
      ${spectrogramHtml}
      <div class="epoch-timeline" id="timeline-${ci}"></div>
    </div>`;

  cl.appendChild(item);

  renderTimeline(ci);
});

// ── Channel toggle ────────────────────────────────────────────────────────
function toggleChannel(ci) {
  document.getElementById(`ch-${ci}`).classList.toggle("open");
}

// ── Timeline rendering (hour rows -> groups -> dots) ────────────────────
function renderTimeline(ci) {
  const ch = DATA.channels[ci];
  const container = document.getElementById(`timeline-${ci}`);

  ch.hour_rows.forEach(hourRow => {
    const rowEl = document.createElement("div");
    rowEl.className = "hour-row";

    const label = document.createElement("div");
    label.className = "hour-label";
    label.textContent = hourRow.hour_label;
    rowEl.appendChild(label);

    const dotsRow = document.createElement("div");
    dotsRow.className = "epoch-row";

    hourRow.groups.forEach(group => {
      const dot = document.createElement("div");
      dot.className = `epoch-dot ${group.quality}`;
      dot.title = `${group.start_time} – ${group.end_time}`;
      dot.addEventListener("click", () => openModal(group, ch.name));
      dotsRow.appendChild(dot);
    });

    rowEl.appendChild(dotsRow);
    container.appendChild(rowEl);
  });
}

// ── Modal ─────────────────────────────────────────────────────────────────
function openModal(group, chName) {
  document.getElementById("modal-title").textContent =
    `${chName}  ·  ${group.start_time} – ${group.end_time}`;

  const light = document.getElementById("modal-traffic-light");
  light.className = `modal-traffic-light ${group.quality}`;

  const body = document.getElementById("modal-body");

  const cols = [
    ["#", m => m.order],
    ["Start offset (s)", m => m.start_time],
    ["End offset (s)", m => m.end_time],
    ["Median Amp (µV)", m => m.mad_uv],
    ["Peak-to-Peak (µV)", m => m.ptp_uv],
    ["Max Flat Duration (s)", m => m.flatline_max_s],
    ["Clip Fraction", m => m.clip_frac],
    ["Model Predictions", m => renderListOrText(m.predictions),
      "These are the models that predicted an artifact in this epoch."],
    ["Rules Satisfied", m => renderListOrText(m.satisfied_rules),
      "These are the rules that this epoch satisfied."],
  ];

  const thead = cols.map(([label, , hoverText]) => {
    const help = hoverText
      ? `<span class="help-mark" title="${escapeAttr(hoverText)}">?</span>`
      : "";
    return `<th>${label}${help}</th>`;
  }).join("");

  const tbody = group.epochs.map(m =>
    `<tr>${cols.map(([, fn]) => `<td class="mono">${fn(m)}</td>`).join("")}</tr>`
  ).join("");

  body.innerHTML = `
    <table class="micro-table">
      <thead><tr>${thead}</tr></thead>
      <tbody>${tbody}</tbody>
    </table>`;

  document.getElementById("modal-overlay").classList.add("active");
}

function closeModal() {
  document.getElementById("modal-overlay").classList.remove("active");
}

document.getElementById("modal-close")
  .addEventListener("click", closeModal);
document.getElementById("modal-overlay")
  .addEventListener("click", e => { if (e.target === e.currentTarget) closeModal(); });

// ── Helpers ───────────────────────────────────────────────────────────────
function renderListOrText(value) {
  if (typeof value === "string") {
    return `<span class="placeholder-text">${value}</span>`;
  }
  if (Array.isArray(value) && value.length === 0) {
    return `<span class="placeholder-text">None.</span>`;
  }
  return value.join(", ");
}

function escapeAttr(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}