async function api(url, method = "GET", body = null) {
  const options = { method, headers: { "Content-Type": "application/json" } };
  if (body) options.body = JSON.stringify(body);
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${method} ${url} failed: ${res.status}`);
  return res.json();
}

function writeOutput(title, obj) {
  const box = document.getElementById("outputBox");
  const time = new Date().toLocaleTimeString();

  const entry = document.createElement("div");
  entry.style.marginBottom = "16px";
  entry.style.borderLeft = "2px solid var(--border)";
  entry.style.paddingLeft = "12px";

  const header = document.createElement("div");
  header.style.color = "var(--primary)";
  header.style.fontWeight = "600";
  header.style.fontSize = "11px";
  header.style.marginBottom = "4px";
  header.textContent = `[${time}] ${title.toUpperCase()}`;

  const content = document.createElement("pre");
  content.style.margin = "0";
  content.style.whiteSpace = "pre-wrap";
  content.style.wordBreak = "break-all";
  content.textContent = JSON.stringify(obj, null, 2);

  entry.appendChild(header);
  entry.appendChild(content);

  if (box.children.length > 0 && box.children[0].textContent.includes("//")) {
    box.innerHTML = "";
  }

  box.prepend(entry);
}

function clearConsole() {
  document.getElementById("outputBox").innerHTML = '<div style="color: var(--muted); opacity: 0.5;">// Console cleared.</div>';
}

function getDeviceId() {
  return (document.getElementById("deviceId").value || "esp32_lab_1").trim();
}

async function armWithDevice() {
  try {
    const out = await api("/api/arm", "POST", { device_id: getDeviceId() });
    showToast(`Command sent to ${getDeviceId()}: ARM`, "success");
    writeOutput("API: ARM", out);
  } catch (err) {
    showToast("Command failed: " + err.message, "error");
  }
}

async function disarmWithDevice() {
  try {
    const out = await api("/api/disarm", "POST", { device_id: getDeviceId() });
    showToast(`Command sent to ${getDeviceId()}: DISARM`, "info");
    writeOutput("API: DISARM", out);
  } catch (err) {
    showToast("Command failed: " + err.message, "error");
  }
}

async function alarmTestWithDevice() {
  try {
    const out = await api("/api/alarm_test", "POST", { device_id: getDeviceId() });
    showToast("Hardware test command queued", "info");
    writeOutput("API: ALARM TEST", out);
  } catch (err) {
    showToast("Command failed: " + err.message, "error");
  }
}

async function testNotification() {
  try {
    const out = await api("/api/test_notification", "POST", {});
    showToast("Telegram gateway test triggered", "info");
    writeOutput("API: NOTIFY TEST", out);
  } catch (err) {
    showToast("Notification failed: " + err.message, "error");
  }
}

async function health() {
  try {
    const out = await api("/api/health");
    showToast("Server health: OK", "success");
    writeOutput("API: HEALTH", out);
  } catch (err) {
    showToast("Health check failed", "error");
  }
}

async function state() {
  try {
    const out = await api("/api/state");
    showToast("System state retrieved", "info");
    writeOutput("API: STATE", out);
  } catch (err) {
    showToast("Failed to read state", "error");
  }
}
