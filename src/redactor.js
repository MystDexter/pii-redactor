// PII detection engine — runs entirely client-side.
//
// Two layers:
//   1. Regex layer: deterministic, instant, no download. Catches structured PII
//      (email, phone, SSN, credit card, IP address).
//   2. NER layer: Xenova/bert-base-NER via transformers.js. Catches unstructured
//      PII (names, organizations, locations).
//
// Both run over the ORIGINAL text so character offsets stay consistent. Spans are
// merged afterwards; on overlap the regex span wins (more precise for structured
// data).

import { pipeline, env } from "@huggingface/transformers";

// Allow remote model download from the Hugging Face Hub, disable local file lookup.
env.allowLocalModels = false;

// ---------------------------------------------------------------------------
// Category definitions
// ---------------------------------------------------------------------------
// `layer` is 'regex' or 'ner'. `priority` is used to resolve overlaps between
// regex categories (lower number wins). NER categories always yield to regex.
export const CATEGORIES = {
  // Keyword-anchored secret wins over everything else on overlap.
  secret: {
    label: "Password",
    placeholder: "[PASSWORD]",
    layer: "regex",
    priority: 0,
  },
  email: {
    label: "Email",
    placeholder: "[EMAIL]",
    layer: "regex",
    priority: 1,
  },
  ssn: { label: "SSN", placeholder: "[SSN]", layer: "regex", priority: 2 },
  card: {
    label: "Credit card",
    placeholder: "[CARD]",
    layer: "regex",
    priority: 3,
  },
  ip: { label: "IP address", placeholder: "[IP]", layer: "regex", priority: 4 },
  phone: {
    label: "Phone",
    placeholder: "[PHONE]",
    layer: "regex",
    priority: 5,
  },
  per: { label: "Name", placeholder: "[NAME]", layer: "ner" },
  org: { label: "Organization", placeholder: "[ORG]", layer: "ner" },
  loc: { label: "Location", placeholder: "[LOCATION]", layer: "ner" },
  misc: { label: "Misc", placeholder: "[REDACTED]", layer: "ner" },
};

// ---------------------------------------------------------------------------
// Regex layer
// ---------------------------------------------------------------------------
const PATTERNS = {
  email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
  ssn: /\b\d{3}-\d{2}-\d{4}\b/g,
  // 13–16 digit sequences, optionally grouped by spaces or hyphens (Luhn-checked below).
  card: /\b(?:\d[ -]?){12,18}\d\b/g,
  ip: /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/g,
  // North-American style and international numbers with separators / country code.
  phone: /(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b/g,
};

const SECRET_RE =
  /(?<=\b(?:passwords?|passphrase|passcode|passwd|pwd|pins?|api[\s_-]?keys?|secrets?|tokens?|access[\s_-]?keys?|auth[\s_-]?tokens?|credentials?|login)\b[^\S\n]*(?:is|are|was|=|:)+[^\S\n]*)\S{3,}/gi;

function detectSecrets(text) {
  const spans = [];
  let m;
  SECRET_RE.lastIndex = 0;
  while ((m = SECRET_RE.exec(text)) !== null) {
    let value = m[0];
    // Drop trailing sentence punctuation that isn't part of the secret.
    const trimmed = value.replace(/[.,;:!?)\]}"']+$/, "");
    if (trimmed.length < 3) continue;
    spans.push({
      start: m.index,
      end: m.index + trimmed.length,
      type: "secret",
      text: trimmed,
      source: "regex",
      priority: CATEGORIES.secret.priority,
    });
  }
  return spans;
}

// Luhn checksum — used to keep credit-card matches from flagging arbitrary digit runs.
function luhnValid(value) {
  const digits = value.replace(/\D/g, "");
  if (digits.length < 13 || digits.length > 19) return false;
  let sum = 0;
  let alt = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let n = Number(digits[i]);
    if (alt) {
      n *= 2;
      if (n > 9) n -= 9;
    }
    sum += n;
    alt = !alt;
  }
  return sum % 10 === 0;
}

export function detectRegex(text) {
  const spans = [...detectSecrets(text)];
  for (const [type, pattern] of Object.entries(PATTERNS)) {
    pattern.lastIndex = 0;
    let m;
    while ((m = pattern.exec(text)) !== null) {
      const value = m[0];
      if (type === "card" && !luhnValid(value)) continue;
      spans.push({
        start: m.index,
        end: m.index + value.length,
        type,
        text: value,
        source: "regex",
        priority: CATEGORIES[type].priority,
      });
      // Guard against zero-length matches causing an infinite loop.
      if (m.index === pattern.lastIndex) pattern.lastIndex++;
    }
  }
  return resolveOverlaps(spans);
}

// Keep the highest-priority (lowest number) span when regex matches overlap.
function resolveOverlaps(spans) {
  const sorted = [...spans].sort(
    (a, b) => a.start - b.start || a.priority - b.priority || b.end - a.end,
  );
  const kept = [];
  for (const span of sorted) {
    const clash = kept.find((k) => span.start < k.end && span.end > k.start);
    if (!clash) kept.push(span);
  }
  return kept;
}

// ---------------------------------------------------------------------------
// NER layer
// ---------------------------------------------------------------------------
let nerPromise = null;

// Load the pipeline once. Tries WebGPU, falls back to WASM if unavailable.
export function loadNer(onProgress) {
  if (nerPromise) return nerPromise;
  const model = "Xenova/bert-base-NER";
  nerPromise = pipeline("token-classification", model, {
    device: "webgpu",
    progress_callback: onProgress,
  }).catch(() =>
    pipeline("token-classification", model, { progress_callback: onProgress }),
  );
  return nerPromise;
}

// bert-base-NER handles ~512 tokens at a time. Split long input on whitespace
// boundaries so we never overflow the model, tracking each chunk's char offset.
function chunkText(text, maxLen = 1200) {
  if (text.length <= maxLen) return [{ text, offset: 0 }];
  const chunks = [];
  let offset = 0;
  while (offset < text.length) {
    let end = Math.min(offset + maxLen, text.length);
    if (end < text.length) {
      const ws = text.lastIndexOf(" ", end);
      const nl = text.lastIndexOf("\n", end);
      const cut = Math.max(ws, nl);
      if (cut > offset) end = cut;
    }
    chunks.push({ text: text.slice(offset, end), offset });
    offset = end;
  }
  return chunks;
}

const TAG_TO_TYPE = { PER: "per", ORG: "org", LOC: "loc", MISC: "misc" };

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// bert-base-NER emits BIO-tagged word-piece tokens (B-PER, I-PER, …) with no
// character offsets — only the surface `word`. Merge consecutive tokens of the
// same entity type into whole-word groups, reconstructing the surface text
// (e.g. "A" + "##c" + "##me" -> "Acme").
function aggregate(tokens) {
  const groups = [];
  let cur = null;
  const flush = () => {
    if (cur && cur.words.length) groups.push(cur);
    cur = null;
  };
  for (const tok of tokens) {
    const entity = tok.entity || tok.entity_group || "O";
    if (entity === "O") {
      flush();
      continue;
    }
    const [prefix, tag] = entity.includes("-")
      ? entity.split("-")
      : ["B", entity];
    const type = TAG_TO_TYPE[tag];
    if (!type) {
      flush();
      continue;
    }
    const word = tok.word || "";
    const isPiece = word.startsWith("##");
    const contiguous = cur && cur.type === type && prefix === "I";
    if (contiguous) {
      if (isPiece) cur.words[cur.words.length - 1] += word.slice(2);
      else cur.words.push(word);
      cur.score = Math.min(cur.score, tok.score ?? 1);
    } else {
      flush();
      cur = {
        type,
        words: [isPiece ? word.slice(2) : word],
        score: tok.score ?? 1,
      };
    }
  }
  flush();
  return groups;
}

// Locate each reconstructed entity in the source to recover char offsets.
// A forward-moving cursor maps repeated phrases to successive occurrences;
// whitespace between words is matched loosely so newlines don't break it.
function locate(groups, text, baseOffset) {
  const spans = [];
  let cursor = 0;
  for (const g of groups) {
    const words = g.words.filter(Boolean);
    if (!words.length) continue;
    // Require word boundaries so an entity matches whole words only — e.g. a
    // mislabelled "SS" fragment can't match inside "SSN".
    const body = words.map(escapeRegex).join("\\s+");
    const pre = /\w/.test(words[0][0]) ? "\\b" : "";
    const post = /\w/.test(words[words.length - 1].slice(-1)) ? "\\b" : "";
    const pattern = new RegExp(pre + body + post);
    let m = pattern.exec(text.slice(cursor));
    let start;
    if (m) {
      start = cursor + m.index;
    } else {
      m = pattern.exec(text); // fall back to a search from the beginning
      if (!m) continue;
      start = m.index;
    }
    const end = start + m[0].length;
    spans.push({
      start: start + baseOffset,
      end: end + baseOffset,
      type: g.type,
      text: m[0],
      source: "ner",
      score: g.score,
    });
    cursor = end;
  }
  return spans;
}

export async function detectNer(text, onProgress) {
  const ner = await loadNer(onProgress);
  const spans = [];
  for (const { text: chunk, offset } of chunkText(text)) {
    const tokens = await ner(chunk);
    spans.push(...locate(aggregate(tokens), chunk, offset));
  }
  return spans;
}

// ---------------------------------------------------------------------------
// Merge + render
// ---------------------------------------------------------------------------
// Combine regex and NER spans. A NER span overlapping any regex span is dropped.
export function mergeSpans(regexSpans, nerSpans) {
  const kept = [...regexSpans];
  for (const span of nerSpans) {
    const clash = regexSpans.find(
      (r) => span.start < r.end && span.end > r.start,
    );
    if (!clash) kept.push(span);
  }
  return kept.sort((a, b) => a.start - b.start);
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Build an HTML preview of the original text with enabled spans wrapped in
// coloured <mark> elements. Disabled categories render as plain text.
export function buildHighlight(text, spans, enabled) {
  let html = "";
  let cursor = 0;
  for (const span of spans) {
    if (!enabled[span.type]) continue;
    if (span.start < cursor) continue; // safety: skip any residual overlap
    html += escapeHtml(text.slice(cursor, span.start));
    const label = CATEGORIES[span.type].label;
    html += `<mark class="pii pii-${span.type}" title="${label}">${escapeHtml(
      span.text,
    )}</mark>`;
    cursor = span.end;
  }
  html += escapeHtml(text.slice(cursor));
  return html.replace(/\n/g, "<br>");
}

// Produce the redacted plain text, replacing enabled spans with placeholders.
export function buildRedacted(text, spans, enabled) {
  let out = "";
  let cursor = 0;
  for (const span of spans) {
    if (!enabled[span.type]) continue;
    if (span.start < cursor) continue;
    out += text.slice(cursor, span.start);
    out += CATEGORIES[span.type].placeholder;
    cursor = span.end;
  }
  out += text.slice(cursor);
  return out;
}

// Count detected spans per category (regardless of toggle state).
export function countByType(spans) {
  const counts = {};
  for (const span of spans) counts[span.type] = (counts[span.type] || 0) + 1;
  return counts;
}
