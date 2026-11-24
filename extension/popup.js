document.addEventListener("DOMContentLoaded", () => {
  const statusPill = document.getElementById("status-pill");
  const btnScan = document.getElementById("btn-scan");
  const btnExport = document.getElementById("btn-export-csv");
  const toggleAutoScroll = document.getElementById("toggle-autoscroll");
  const radios = document.querySelectorAll("input[name='engine']");
  const summaryBox = document.getElementById("scan-summary");
  const btnTheme = document.getElementById("btn-theme");
  const themeOrder = ["auto", "dark", "light"];
  const themeLabels = {
    auto: "Otomatis (Sistem)",
    dark: "Gelap (Dark)",
    light: "Terang (Light)",
  };

  function terapkanTemaPopup(theme) {
    const activeTheme = themeOrder.includes(theme) ? theme : "auto";
    document.documentElement.setAttribute("data-theme", activeTheme);
    btnTheme.dataset.theme = activeTheme;
    btnTheme.setAttribute("aria-label", `Tema: ${themeLabels[activeTheme]}`);
    btnTheme.title = `Tema: ${themeLabels[activeTheme]}. Klik untuk beralih.`;
  }

  chrome.storage.local.get(["engine", "autoScan", "theme"], (cfg) => {
    const savedEngine = cfg.engine || "indobert";
    radios.forEach((r) => {
      r.checked = r.value === savedEngine;
    });
    toggleAutoScroll.checked = Boolean(cfg.autoScan);
    terapkanTemaPopup(cfg.theme || "auto");
  });

  btnTheme.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "auto";
    const next = themeOrder[(themeOrder.indexOf(current) + 1) % themeOrder.length];
    terapkanTemaPopup(next);
    chrome.storage.local.set({ theme: next });
  });

  let latestRecords = [];

  function perbaruiTampilanRingkasan(resp) {
    if (!resp || !resp.ok || resp.total === 0) return;
    latestRecords = resp.records || [];
    summaryBox.classList.remove("hidden");
    document.getElementById("sum-total").textContent = resp.total;
    document.getElementById("sum-pos").textContent = resp.counts.POSITIF || 0;
    document.getElementById("sum-net").textContent = resp.counts.NETRAL || 0;
    document.getElementById("sum-neg").textContent = resp.counts.NEGATIF || 0;
  }

  radios.forEach((r) => {
    r.addEventListener("change", () => {
      if (r.checked) chrome.storage.local.set({ engine: r.value });
    });
  });

  toggleAutoScroll.addEventListener("change", () => {
    const enabled = toggleAutoScroll.checked;
    chrome.storage.local.set({ autoScan: enabled });
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "SET_AUTO_SCAN", enabled });
      }
    });
  });

  // Cek status koneksi server FastAPI lokal
  chrome.runtime.sendMessage({ type: "CHECK_HEALTH" }, (resp) => {
    if (resp && resp.ok && resp.data.status === "online") {
      statusPill.textContent = "Server Online";
      statusPill.className = "pill online";
      btnScan.disabled = false;
    } else {
      statusPill.textContent = "Server Offline";
      statusPill.className = "pill offline";
      btnScan.disabled = true;
    }
  });

  // Ambil statistik yang sudah terpindai di tab aktif (jika Auto-Scan aktif)
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]?.id) return;
    chrome.tabs.sendMessage(tabs[0].id, { type: "GET_STATS" }, (resp) => {
      if (!chrome.runtime.lastError && resp) {
        perbaruiTampilanRingkasan(resp);
      }
    });
  });

  // Pindai manual komentar pada tab aktif
  btnScan.addEventListener("click", () => {
    btnScan.textContent = "Memindai Komentar...";
    btnScan.disabled = true;

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]?.id) return;

      chrome.tabs.sendMessage(tabs[0].id, { type: "SCAN_PAGE" }, (resp) => {
        btnScan.textContent = "Pindai Komentar Halaman Ini";
        btnScan.disabled = false;

        if (chrome.runtime.lastError || !resp || !resp.ok) {
          alert("Gagal memindai halaman. Coba muat ulang (F5) halaman marketplace/medsos terlebih dahulu.");
          return;
        }

        perbaruiTampilanRingkasan(resp);
      });
    });
  });

  // Unduh hasil pemindaian halaman ke file CSV
  btnExport.addEventListener("click", () => {
    if (!latestRecords.length) return;

    const header = ["platform", "komentar", "sentimen", "keyakinan_%", "model"];
    const rows = latestRecords.map((r) => [
      `"${String(r.platform).replace(/"/g, '""')}"`,
      `"${String(r.komentar).replace(/"/g, '""')}"`,
      `"${r.sentimen}"`,
      r.keyakinan,
      `"${r.model}"`
    ]);

    const csvContent = "\uFEFF" + [header.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `sentimen_${latestRecords[0].platform}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  });
});