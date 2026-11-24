(() => {
  let autoScanEnabled = false;
  let isScanning = false;
  let scrollDebounceTimer = null;
  const scannedRecords = [];
  let currentTheme = "auto";
  chrome.storage.local.get(["autoScan", "theme"], (res) => {
    autoScanEnabled = Boolean(res.autoScan);
    currentTheme = res.theme || "auto";
  });

  chrome.storage.onChanged.addListener((changes) => {
    if (changes.theme) {
      currentTheme = changes.theme.newValue || "auto";
      const card = document.getElementById("sl-floating-card");
      if (card) card.setAttribute("data-sl-theme", currentTheme);
    }
  });

  function kumpulkanElemenKomentar() {
    const host = window.location.hostname;
    const kandidat = [];
    const seenTexts = new Set(scannedRecords.map((r) => r.komentar));

    function tambahKandidat(el, customText = null) {
      if (!el || el.dataset.slProcessed === "true") return;
      if (el.closest("nav, header, footer, button, #sl-floating-card")) return;

      const teks = (customText || el.innerText || "").replace(/\s+/g, " ").trim();
      if (teks.length < 8 || teks.length > 700) return;
      if (seenTexts.has(teks)) return;

      seenTexts.add(teks);
      kandidat.push({ el, text: teks });
    }

    if (host.includes("shopee")) {
      document.querySelectorAll("[data-cmtid]").forEach((card) => {
        card.querySelectorAll("div, span").forEach((b) => {
          if (b.children.length > 0) return;
          const t = (b.innerText || "").trim();
          if (
            t.length >= 12 &&
            !t.includes("Variasi:") &&
            !t.includes("Variation:") &&
            !t.startsWith("Membantu") &&
            !t.startsWith("Respon Penjual")
          ) {
            tambahKandidat(b, t);
          }
        });
      });

      if (kandidat.length === 0) {
        document.querySelectorAll("div").forEach((div) => {
          const txt = (div.innerText || "").trim();
          if (/^\d{4}-\d{2}-\d{2}.*Variasi:/i.test(txt) && div.parentElement) {
            div.parentElement.querySelectorAll("div").forEach((child) => {
              if (child === div || child.children.length > 1) return;
              const cTxt = (child.innerText || "").trim();
              if (
                cTxt.length >= 15 &&
                !cTxt.includes("Variasi:") &&
                !cTxt.includes("Membantu") &&
                !cTxt.includes("Respon Penjual")
              ) {
                tambahKandidat(child, cTxt);
              }
            });
          }
        });
      }
    } else if (host.includes("tokopedia")) {
      ["[data-testid='lblItemUlasan']", "article p", "[data-testid*='review'] span"].forEach((sel) => {
        document.querySelectorAll(sel).forEach((el) => tambahKandidat(el));
      });
    } else if (host.includes("x.com") || host.includes("twitter.com")) {
      ["article [data-testid='tweetText']", "article div[lang]"].forEach((sel) => {
        document.querySelectorAll(sel).forEach((el) => tambahKandidat(el));
      });
    } else if (host.includes("instagram.com")) {
      ["ul ul span[dir='auto']", "div[role='dialog'] ul span[dir='auto']", "h1[dir='auto']"].forEach((sel) => {
        document.querySelectorAll(sel).forEach((el) => tambahKandidat(el));
      });
    } else if (host.includes("facebook.com")) {
      ["div[role='article'] div[dir='auto'][style*='text-align']", "div[role='article'] div[dir='auto']"].forEach((sel) => {
        document.querySelectorAll(sel).forEach((el) => tambahKandidat(el));
      });
    } else {
      ["p", "article", "[class*='comment']", "[class*='review']"].forEach((sel) => {
        document.querySelectorAll(sel).forEach((el) => tambahKandidat(el));
      });
    }

    return kandidat.slice(0, 60);
  }

  function tempelBadge(el, sentiment, confidence) {
    el.dataset.slProcessed = "true";
    const badge = document.createElement("span");
    const clsMap = {
      POSITIF: "sl-badge-positif",
      NEGATIF: "sl-badge-negatif",
      NETRAL: "sl-badge-netral"
    };
    badge.className = `sl-inline-badge ${clsMap[sentiment] || "sl-badge-netral"}`;
    badge.textContent = `${sentiment} · ${Number(confidence).toFixed(1)}%`;
    el.appendChild(badge);
  }

  function hitungRekapTotal() {
    const counts = { POSITIF: 0, NETRAL: 0, NEGATIF: 0 };
    scannedRecords.forEach((r) => {
      counts[r.sentimen] = (counts[r.sentimen] || 0) + 1;
    });
    return counts;
  }

  function jalankanPemindaian(callback = null) {
    if (isScanning) {
      if (callback) callback({ ok: true, total: scannedRecords.length, counts: hitungRekapTotal(), records: scannedRecords });
      return;
    }

    const kandidat = kumpulkanElemenKomentar();
    if (kandidat.length === 0) {
      if (callback) callback({ ok: true, total: scannedRecords.length, counts: hitungRekapTotal(), records: scannedRecords });
      return;
    }

    isScanning = true;
    const texts = kandidat.map((k) => k.text);
    const platform = window.location.hostname.replace(/^www\./, "");

    chrome.runtime.sendMessage({ type: "ANALYZE_BATCH", texts }, (resp) => {
      isScanning = false;
      if (!resp || !resp.ok) {
        if (callback) callback({ ok: false, error: "Server offline" });
        return;
      }

      resp.data.results.forEach((res, idx) => {
        const item = kandidat[idx];
        if (item) {
          tempelBadge(item.el, res.sentiment, res.confidence);
          scannedRecords.push({
            platform,
            komentar: item.text,
            sentimen: res.sentiment,
            keyakinan: Number(res.confidence).toFixed(1),
            model: resp.data.engine
          });
        }
      });

      if (callback) {
        callback({
          ok: true,
          total: scannedRecords.length,
          counts: hitungRekapTotal(),
          records: scannedRecords
        });
      }
    });
  }

  // Listener Auto-Scan saat pengguna menggulir halaman
  window.addEventListener(
    "scroll",
    () => {
      if (!autoScanEnabled) return;
      clearTimeout(scrollDebounceTimer);
      scrollDebounceTimer = setTimeout(() => {
        jalankanPemindaian();
      }, 900);
    },
    { passive: true }
  );

  // Kartu mengambang untuk analisis teks pilihan
  function tutupKartu() {
    const existing = document.getElementById("sl-floating-card");
    if (existing) existing.remove();
  }

  function tutupTrigger() {
    const trig = document.getElementById("sl-selection-trigger");
    if (trig) trig.remove();
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function tampilkanKartuAnalisis(teks) {
    tutupKartu();
    tutupTrigger();

    const card = document.createElement("div");
    card.id = "sl-floating-card";
    card.setAttribute("data-sl-theme", currentTheme);
    card.innerHTML = `
      <div class="sl-card-top">
        <span class="sl-card-brand">SentimenLab</span>
        <button class="sl-card-close" id="sl-close-btn">&times;</button>
      </div>
      <div class="sl-quote">"${escapeHtml(teks.slice(0, 180))}${teks.length > 180 ? "..." : ""}"</div>
      <div id="sl-card-body" style="font-size:12px; color:#a1a1aa;">Menganalisis teks...</div>
    `;
    document.body.appendChild(card);
    document.getElementById("sl-close-btn").onclick = tutupKartu;

    chrome.runtime.sendMessage({ type: "ANALYZE_SINGLE", text: teks }, (resp) => {
      const body = document.getElementById("sl-card-body");
      if (!body) return;

      if (!resp || !resp.ok) {
        body.innerHTML = `<div style="color:#ef4444;">Gagal terhubung ke server lokal (127.0.0.1:8000). Pastikan <code>start_api.bat</code> sedang berjalan.</div>`;
        return;
      }

      const d = resp.data;
      const colorMap = { POSITIF: "#10b981", NEGATIF: "#ef4444", NETRAL: "#a1a1aa" };
      const accent = colorMap[d.sentiment] || "#a1a1aa";

      const pPos = Number(d.probabilities.POSITIF).toFixed(1);
      const pNet = Number(d.probabilities.NETRAL).toFixed(1);
      const pNeg = Number(d.probabilities.NEGATIF).toFixed(1);

      const absaHtml = (d.absa || [])
        .map(
          (item) => `
          <div class="sl-absa-item" style="flex-direction:column; align-items:flex-start; gap:3px;">
            <div style="display:flex; justify-content:space-between; width:100%;">
              <span><strong>${item.Aspek}</strong></span>
              <span style="color:${colorMap[item.Sentimen] || "#a1a1aa"}; font-weight:600;">
                ${item.Sentimen} (${Number(item["Keyakinan (%)"]).toFixed(1)}%)
              </span>
            </div>
            <div style="font-size:10px; color:#71717a;">"${escapeHtml(item.Klausa)}"</div>
          </div>`
        )
        .join("");

      body.innerHTML = `
        <div class="sl-verdict-box" style="border-left: 3px solid ${accent};">
          <div class="sl-verdict-title">Klasifikasi (${d.engine})</div>
          <div class="sl-verdict-main" style="color:${accent};">${d.sentiment}</div>
          <div class="sl-verdict-sub">Keyakinan: ${Number(d.confidence).toFixed(1)}% · Pos ${pPos}% | Net ${pNet}% | Neg ${pNeg}%</div>
        </div>
        <div class="sl-verdict-title" style="margin-bottom:4px;">Segmentasi Aspek (ABSA)</div>
        ${absaHtml}
      `;
    });
  }

  document.addEventListener("mouseup", (e) => {
    if (e.target.closest("#sl-selection-trigger, #sl-floating-card")) return;

    setTimeout(() => {
      const sel = window.getSelection();
      const teks = sel ? sel.toString().trim() : "";
      tutupTrigger();

      if (teks.length >= 4 && teks.length <= 1000) {
        const range = sel.getRangeAt(0);
        const rect = range.getBoundingClientRect();

        const btn = document.createElement("button");
        btn.id = "sl-selection-trigger";
        btn.setAttribute("data-sl-theme", currentTheme);
        btn.textContent = "Analisis Sentimen";
        btn.style.top = `${Math.max(10, rect.bottom + 6)}px`;
        btn.style.left = `${Math.min(window.innerWidth - 150, Math.max(10, rect.left))}px`;
        btn.onclick = () => tampilkanKartuAnalisis(teks);
        document.body.appendChild(btn);
      }
    }, 10);
  });

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === "SHOW_SELECTION_CARD" && msg.text) {
      tampilkanKartuAnalisis(msg.text);
    }

    if (msg.type === "SET_AUTO_SCAN") {
      autoScanEnabled = Boolean(msg.enabled);
      if (autoScanEnabled) jalankanPemindaian();
      sendResponse({ ok: true });
    }

    if (msg.type === "GET_STATS") {
      sendResponse({
        ok: true,
        total: scannedRecords.length,
        counts: hitungRekapTotal(),
        records: scannedRecords
      });
    }

    if (msg.type === "SCAN_PAGE") {
      jalankanPemindaian((hasil) => sendResponse(hasil));
      return true;
    }
  });
})();