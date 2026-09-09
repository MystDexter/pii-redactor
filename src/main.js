import "./style.css";
import {
  CATEGORIES,
  detectRegex,
  detectNer,
  mergeSpans,
  buildHighlight,
  buildRedacted,
  countByType,
} from "./redactor.js";

const SAMPLE = `Hi, my name is Sarah Chen and I work at Acme Corporation in San Francisco.
You can reach me at sarah.chen@example.com or call (415) 555-0142.
My SSN is 123-45-6789 and the card on file is 4111 1111 1111 1111.
Server logs show a login from 192.168.1.34 last Tuesday.`;

document.querySelector("#app").innerHTML = `
<main class="wrap">
  <header class="head">
    <h1>PII Redactor</h1>
    <p class="tagline">
      Detects names, emails, phone numbers, addresses and other Personally Identifiable Information (PII), then
      redacts them before your text goes anywhere else.
    </p>
    <p class="privacy">
      Runs entirely in your browser. Nothing is uploaded to a server or third party. Reduces exposure;
      does not guarantee complete removal.
    </p>
  </header>

  <section class="panes">
    <div class="pane">
      <div class="pane-head">
        <label for="input">Paste text</label>
        <button id="sample" class="link-btn" type="button">Load sample</button>
      </div>
      <textarea id="input" spellcheck="false"
        placeholder="Paste the text you want to scrub…"></textarea>
      <div class="actions">
        <button id="redact" class="primary" type="button">Redact</button>
        <span id="status" class="status" role="status" aria-live="polite"></span>
      </div>
    </div>

    <div class="pane">
      <div class="pane-head">
        <span class="pane-title">Result</span>
        <button id="copy" class="link-btn" type="button" disabled>Copy</button>
      </div>
      <div class="result-tabs" role="tablist">
        <button id="tab-redacted" class="tab active" type="button">Redacted</button>
        <button id="tab-highlight" class="tab" type="button">Highlighted</button>
      </div>
      <pre id="redacted" class="output" aria-live="polite"></pre>
      <div id="highlight" class="output" hidden></div>
    </div>
  </section>

  <section id="toggles" class="toggles" hidden>
    <span class="toggles-label">Categories detected — uncheck to keep any in the text:</span>
    <div id="toggle-list" class="toggle-list"></div>
  </section>

  <p class="note">
    First run downloads the NER model (<strong>Xenova/bert-base-NER</strong>,
    a quantized model under ~200&nbsp;MB) and caches it in your browser. Regex
    detection (email, phone, SSN, card, IP) works instantly with no download.
  </p>
</main>
`;

// ---------------------
// State + element refs
// ---------------------
const els = {
  input: document.querySelector("#input"),
  redactBtn: document.querySelector("#redact"),
  status: document.querySelector("#status"),
  copyBtn: document.querySelector("#copy"),
  redacted: document.querySelector("#redacted"),
  highlight: document.querySelector("#highlight"),
  toggles: document.querySelector("#toggles"),
  toggleList: document.querySelector("#toggle-list"),
  sampleBtn: document.querySelector("#sample"),
  tabRedacted: document.querySelector("#tab-redacted"),
  tabHighlight: document.querySelector("#tab-highlight"),
};

let state = {
  text: "",
  spans: [],
  enabled: {}, // type -> boolean
};

// ----------
// Rendering
// ----------
function renderOutput() {
  const { text, spans, enabled } = state;
  els.redacted.textContent = buildRedacted(text, spans, enabled);
  els.highlight.innerHTML = buildHighlight(text, spans, enabled);
  els.copyBtn.disabled = spans.length === 0;
}

function renderToggles() {
  const counts = countByType(state.spans);
  const types = Object.keys(CATEGORIES).filter((t) => counts[t]);
  if (types.length === 0) {
    els.toggles.hidden = true;
    els.toggleList.innerHTML = "";
    return;
  }
  els.toggles.hidden = false;
  els.toggleList.innerHTML = types
    .map((type) => {
      const c = CATEGORIES[type];
      return `<label class="toggle pii-${type}">
        <input type="checkbox" data-type="${type}" ${state.enabled[type] ? "checked" : ""}>
        <span class="swatch"></span>
        <span class="toggle-text">${c.label}</span>
        <span class="count">${counts[type]}</span>
      </label>`;
    })
    .join("");
  els.toggleList.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    cb.addEventListener("change", () => {
      state.enabled[cb.dataset.type] = cb.checked;
      renderOutput();
    });
  });
}

function setStatus(msg, kind = "") {
  els.status.textContent = msg;
  els.status.className = `status ${kind}`;
}

// -------------
// Redact flow
// -------------
async function runRedaction() {
  const text = els.input.value;
  if (!text.trim()) {
    setStatus("Paste some text first.", "warn");
    return;
  }

  els.redactBtn.disabled = true;

  // Phase 1: regex is instant.
  const regexSpans = detectRegex(text);

  // Phase 2: NER (may download the model on first run).
  let nerSpans = [];
  try {
    setStatus("Scanning names, orgs and locations…");
    nerSpans = await detectNer(text, onModelProgress);
  } catch (err) {
    console.error(err);
    setStatus(
      "Model failed to load — showing regex-only results (email, phone, SSN, card, IP).",
      "warn",
    );
  }

  const spans = mergeSpans(regexSpans, nerSpans);
  const counts = countByType(spans);

  // Preserve existing toggle choices; default new categories to enabled.
  const enabled = {};
  for (const type of Object.keys(CATEGORIES)) {
    if (counts[type]) enabled[type] = state.enabled[type] ?? true;
  }

  state = { text, spans, enabled };
  renderToggles();
  renderOutput();

  const total = spans.length;
  if (els.status.classList.contains("warn") === false) {
    setStatus(
      total === 0
        ? "No PII detected."
        : `Redacted ${total} item${total === 1 ? "" : "s"}.`,
      total === 0 ? "" : "ok",
    );
  }
  els.redactBtn.disabled = false;
}

// Show model download progress on first run.
let progressReported = false;
function onModelProgress(p) {
  if (p.status === "progress" && p.total) {
    const pct = Math.round((p.loaded / p.total) * 100);
    setStatus(`Downloading model… ${pct}% (first run only)`);
    progressReported = true;
  } else if (p.status === "ready" || p.status === "done") {
    if (progressReported) setStatus("Model ready — scanning…");
  }
}

// -------------------
// Tabs + misc wiring
// -------------------
function showTab(which) {
  const redacted = which === "redacted";
  els.tabRedacted.classList.toggle("active", redacted);
  els.tabHighlight.classList.toggle("active", !redacted);
  els.redacted.hidden = !redacted;
  els.highlight.hidden = redacted;
}

els.tabRedacted.addEventListener("click", () => showTab("redacted"));
els.tabHighlight.addEventListener("click", () => showTab("highlight"));

els.redactBtn.addEventListener("click", runRedaction);

els.sampleBtn.addEventListener("click", () => {
  els.input.value = SAMPLE;
  els.input.focus();
});

els.copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(els.redacted.textContent);
    const prev = els.copyBtn.textContent;
    els.copyBtn.textContent = "Copied ✓";
    setTimeout(() => (els.copyBtn.textContent = prev), 1500);
  } catch {
    setStatus("Copy failed — select the text manually.", "warn");
  }
});

// Cmd/Ctrl+Enter to redact.
els.input.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runRedaction();
});
