const API_BASE = "http://127.0.0.1:8000";

async function readApiResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `Permintaan gagal (${response.status})`;
    throw new Error(detail);
  }
  return data;
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "sentimenlab-analyze",
    title: "Analisis Sentimen dengan SentimenLab",
    contexts: ["selection"]
  });

  chrome.storage.local.get(["engine"], (res) => {
    if (!res.engine) {
      chrome.storage.local.set({ engine: "indobert" });
    }
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "sentimenlab-analyze" && info.selectionText && tab?.id) {
    chrome.tabs.sendMessage(tab.id, {
      type: "SHOW_SELECTION_CARD",
      text: info.selectionText.trim()
    });
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "CHECK_HEALTH") {
    fetch(`${API_BASE}/health`)
      .then(readApiResponse)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "ANALYZE_SINGLE") {
    chrome.storage.local.get(["engine"], (cfg) => {
      const engine = message.engine || cfg.engine || "indobert";
      fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: message.text,
          engine: engine,
          include_absa: true
        })
      })
        .then(readApiResponse)
        .then((data) => sendResponse({ ok: true, data }))
        .catch((err) => sendResponse({ ok: false, error: err.message }));
    });
    return true;
  }

  if (message.type === "ANALYZE_BATCH") {
    chrome.storage.local.get(["engine"], (cfg) => {
      const engine = message.engine || cfg.engine || "indobert";
      fetch(`${API_BASE}/predict-batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          texts: message.texts,
          engine: engine
        })
      })
        .then(readApiResponse)
        .then((data) => sendResponse({ ok: true, data }))
        .catch((err) => sendResponse({ ok: false, error: err.message }));
    });
    return true;
  }
});