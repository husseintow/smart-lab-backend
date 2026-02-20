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

async function refreshState() {
  const s = await api("/api/state");
  const armed = Number(s.armed);
  document.getElementById("stateArmed").innerHTML = badgeArmed(armed);

  const card = document.getElementById("cardArmed");
  if (armed) {
    card.style.borderColor = "var(--danger)";
    card.classList.add("danger");
    card.classList.remove("success");
  } else {
    card.style.borderColor = "var(--success)";
    card.classList.add("success");
    card.classList.remove("danger");
  }

  document.getElementById("serverTime").textContent = s.server_time || "-";
  document.getElementById("trainedAt").textContent = s.model_trained_at ? s.model_trained_at.split('T')[1] : "-";
}

async function refreshSummary() {
  const x = await api("/api/summary");
  document.getElementById("mTotal").textContent = x.total_events ?? 0;
  document.getElementById("mIntrusion").textContent = x.intrusions ?? 0;
  document.getElementById("mNormal").textContent = x.normal_events ?? 0;
  document.getElementById("lastEventTime").textContent = x.last_event_time ? x.last_event_time.replace('T', ' ') : "-";
}

async function refreshLatest() {
  const rows = await api("/api/events?limit=8");
  const tbody = document.querySelector("#latestTable tbody");
  tbody.innerHTML = "";

  rows.forEach(e => {
    const tr = document.createElement("tr");
    const sensing = `M:${e.motion} D:${e.door}`;
    tr.innerHTML = `
      <td style="font-size: 12px; color: var(--muted);">${e.ts.split('T')[1]}</td>
      <td style="font-weight: 600;">${e.device_id}</td>
      <td><code style="background: var(--glass-bg); padding: 2px 6px; border-radius: 4px;">${sensing}</code></td>
      <td>${badgeArmed(Number(e.system_armed))}</td>
      <td>${badgePrediction(e.prediction)}</td>
      <td class="muted" style="font-size: 12px;">${e.label || "-"}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function armSystem() {
  try {
    await api("/api/arm", "POST", {});
    showToast("System ARMED successfully", "success");
    await refreshAll();
  } catch (err) {
    showToast("Failed to arm system: " + err.message, "error");
  }
}

async function disarmSystem() {
  try {
    await api("/api/disarm", "POST", {});
    showToast("System DISARMED", "info");
    await refreshAll();
  } catch (err) {
    showToast("Failed to disarm system: " + err.message, "error");
  }
}

async function alarmTest() {
  const btn = event.currentTarget;
  const originalText = btn.innerHTML;
  btn.innerHTML = "TRIGGERING...";
  btn.disabled = true;
  try {
    await api("/api/alarm_test", "POST", {});
    showToast("Test alarm command queued", "info");
  } catch (err) {
    showToast("Alarm test failed: " + err.message, "error");
  }
  setTimeout(() => {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }, 2000);
}

async function retrainModel() {
  const btn = event.currentTarget;
  const originalText = btn.innerHTML;
  btn.innerHTML = "RETRAINING...";
  btn.classList.add("loading");
  try {
    await api("/api/retrain", "POST", {});
    showToast("Neural engine retrained", "success");
    await refreshAll();
  } catch (err) {
    showToast("Retraining failed: " + err.message, "error");
  }
  setTimeout(() => {
    btn.innerHTML = originalText;
    btn.classList.remove("loading");
  }, 1000);
}

async function simulateEvent(motion, door, hour) {
  try {
    const res = await api("/api/simulate", "POST", { motion, door, hour });
    const color = res.prediction === 'intrusion' ? 'error' : 'success';
    showToast(`Simulation Result: ${res.prediction.toUpperCase()} (${res.reason})`, color);
    await refreshAll();
  } catch (err) {
    showToast("Simulation failed: " + err.message, "error");
  }
}

async function refreshAll() {
  try {
    await Promise.all([refreshState(), refreshSummary(), refreshLatest()]);
  } catch (err) {
    console.warn("Auto-refresh skipped: " + err.message);
  }
}

refreshAll();
setInterval(refreshAll, 5000);
