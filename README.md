# PII Redactor

**Live:** https://pii-redactor-app.vercel.app/

A **client-side** PII redactor. Paste text, and it detects names, emails, phone numbers, addresses and other Personally Identifiable Information (PII), then redacts them before the text goes anywhere else. Useful as a pre-processing step before pasting a document into a third-party AI tool, or as a standalone "scrub this before I send it" utility.

Everything runs in the browser. **Nothing is uploaded to a server.**

> Recall isn't perfect — treat this as _reducing exposure_, not _guaranteeing
> removal_.

## Architecture

Two detection layers, both client-side, merged into one set of spans:

1. **Regex layer** — instant, no download. Catches structured PII:
   email, phone, SSN, credit card (Luhn-checked), IP address.
2. **NER layer** — `[Xenova/bert-base-NER](https://huggingface.co/Xenova/bert-base-NER)`
   via [transformers.js](https://github.com/huggingface/transformers.js). A
   quantized model (~under 200 MB) downloaded on first "Redact" click and cached
   by the browser thereafter. Catches unstructured PII: names (PER),
   organizations (ORG), locations (LOC), misc (MISC). English only.

Both layers run over the **original** text so character offsets stay consistent.
On overlap, the **regex span wins** (more precise for structured data).

The model loads on WebGPU when available and falls back to WASM otherwise.

### Code map

- `[src/redactor.js](src/redactor.js)` — detection engine: regex patterns,
  NER pipeline loading + BIO aggregation, span merging, and the
  highlight/redaction renderers. No DOM dependencies.
- `[src/main.js](src/main.js)` — UI: input, Redact button, model-loading status,
  Redacted/Highlighted tabs, per-category toggles, copy-to-clipboard.
- `[src/style.css](src/style.css)` — styling and the per-category colour tokens.

## Run

```bash
npm install
npm run dev
```

Then open the printed local URL. First "Redact" downloads the NER model; regex
detection works immediately with no download.

Build for production:

```bash
npm run build && npm run preview
```

## Usage

1. Paste text (or click **Load sample**).
2. Click **Redact** (or press ⌘/Ctrl+Enter).
3. Read the **Redacted** output and **Copy** it, or switch to **Highlighted**
   to see what was detected, colour-coded by category.
4. Uncheck any category to keep it in the text — recomputes instantly, no model
   re-run.

## Known limits

- Recall isn't perfect. Framed in the UI as "reduces exposure," not a guarantee.
- Common words are sometimes flagged as ORG/MISC. Uncheck the category if noisy.
- First load downloads the model (stated in the UI before the user clicks).
- Multilingual text: swap the model to
  `Xenova/distilbert-base-multilingual-cased-ner-hrl` in `src/redactor.js`
  (larger, slower).
- Some embedded/sandboxed browsers block the Cache API, so the model may
  re-download each session. Normal Chrome/Firefox/Safari cache it after the
  first load.

## Possible next steps

- File upload (`.txt`, `.pdf` via pdf.js).
- Browser extension that redacts before paste, on any site.
