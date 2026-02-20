let allEvents = [];

async function api(url, method = "GET", body = null) {
  const options = { method, headers: { "Content-Type": "application/json" } };
  if (body) options.body = JSON.stringify(body);
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${method} ${url} failed: ${res.status}`);
  return res.json();
}

function badgePrediction(value) {
  const cls = value === "intrusion" ? "badge-danger" : "badge-success";
  return `<span class="badge ${cls}">${value.toUpperCase()}</span>`;
}

function badgeArmed(v) {
  const cls = v ? "badge-danger" : "badge-success";
  const label = v ? "ARMED" : "DISARMED";
  return `<span class="badge ${cls}">${label}</span>`;
}

async function loadEvents() {
  const limit = document.getElementById("limitSel").value;
  allEvents = await api(`/api/events?limit=${limit}`);
  renderFiltered();
}

function renderFiltered() {
  const q = (document.getElementById("searchInput").value || "").trim().toLowerCase();
  const tbody = document.querySelector("#eventsTable tbody");
  tbody.innerHTML = "";

  const rows = allEvents.filter(e => {
    if (!q) return true;
    const hay = `${e.ts} ${e.device_id} ${e.sensor_type} ${e.prediction} ${e.label || ""} ${e.notes || ""}`.toLowerCase();
    return hay.includes(q);
  });

  rows.forEach(e => {
    const tr = document.createElement("tr");
    const sensing = `M:${e.motion} D:${e.door}`;
    tr.innerHTML = `
      <td style="font-size: 13px; color: var(--muted);">${e.ts.replace('T', ' ')}</td>
      <td style="font-weight: 600;">${e.device_id}</td>
      <td><code style="background: var(--glass-bg); padding: 2px 6px; border-radius: 4px;">${sensing}</code></td>
      <td class="muted">${e.sensor_type}</td>
      <td>${badgeArmed(Number(e.system_armed))}</td>
      <td>${badgePrediction(e.prediction)}</td>
      <td style="font-style: italic;">${e.label || "-"}</td>
      <td style="text-align: right;">
        <div class="row" style="justify-content: flex-end; gap: 6px;">
          <button class="btn btn-outline" style="padding: 4px 8px; font-size: 11px;" onclick="setLabel(${e.id}, 'normal')">Mark Normal</button>
          <button class="btn btn-outline" style="padding: 4px 8px; font-size: 11px; border-color: rgba(239,68,68,0.3); color: var(--danger);" onclick="setLabel(${e.id}, 'intrusion')">Mark Intrusion</button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function setLabel(id, label) {
  await api(`/api/label/${id}`, "POST", { label, notes: "" });
  await loadEvents();
}

loadEvents();

