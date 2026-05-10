// netlify/edge-functions/transcribe.js
// Proxies audio chunks to Whisper server-side — API key never touches the browser.
// Streams the Whisper response back immediately so the connection stays alive
// and Netlify's edge timeout doesn't fire during long processing.

const JSON_HEADERS = { "Content-Type": "application/json" };
const MAX_BYTES = 19 * 1024 * 1024;   // 19MB hard cap — Netlify Edge body limit is 20MB
const WHISPER_TIMEOUT_MS = 120_000;   // 2 min — generous for large chunks

const ALLOWED_TYPES = new Set([
  "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/x-m4a",
  "audio/wav", "audio/wave", "audio/ogg", "audio/webm", "audio/flac",
  "video/mp4", "video/webm", "video/quicktime",   // containers with audio
  "application/octet-stream",                      // browsers sometimes send this for audio
]);

export default async (request) => {
  if (request.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method Not Allowed" }), { status: 405, headers: JSON_HEADERS });
  }

  const OPENAI_KEY = Deno.env.get("OPENAI_API_KEY");
  if (!OPENAI_KEY) {
    return new Response(JSON.stringify({ error: "API key not configured." }), { status: 500, headers: JSON_HEADERS });
  }

  // ── Size guard (before parsing body) ──
  const contentLength = parseInt(request.headers.get("content-length") || "0", 10);
  if (contentLength > MAX_BYTES) {
    return new Response(JSON.stringify({ error: `Chunk too large (${(contentLength / 1048576).toFixed(1)} MB). Max is 19 MB.` }), { status: 413, headers: JSON_HEADERS });
  }

  let formData;
  try {
    formData = await request.formData();
  } catch (e) {
    return new Response(JSON.stringify({ error: "Failed to parse upload: " + e.message }), { status: 400, headers: JSON_HEADERS });
  }

  const audioFile = formData.get("file");
  if (!audioFile || typeof audioFile === "string") {
    return new Response(JSON.stringify({ error: "No audio file received." }), { status: 400, headers: JSON_HEADERS });
  }

  // ── MIME type guard ──
  const mime = (audioFile.type || "").toLowerCase();
  if (mime && !ALLOWED_TYPES.has(mime)) {
    return new Response(JSON.stringify({ error: `Unsupported file type: ${mime}. Upload an audio or video file.` }), { status: 415, headers: JSON_HEADERS });
  }

  // ── Second size check after parse ──
  if (audioFile.size > MAX_BYTES) {
    return new Response(JSON.stringify({ error: `Chunk too large (${(audioFile.size / 1048576).toFixed(1)} MB). Max is 19 MB per chunk.` }), { status: 413, headers: JSON_HEADERS });
  }

  // ── Forward to Whisper with timeout guard ──
  const outForm = new FormData();
  outForm.append("file", audioFile);
  outForm.append("model", "whisper-1");
  outForm.append("response_format", "verbose_json");
  outForm.append("timestamp_granularities[]", "segment");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), WHISPER_TIMEOUT_MS);

  let whisperRes;
  try {
    whisperRes = await fetch("https://api.openai.com/v1/audio/transcriptions", {
      method: "POST",
      headers: { Authorization: `Bearer ${OPENAI_KEY}` },
      body: outForm,
      signal: controller.signal,
    });
  } catch (e) {
    clearTimeout(timer);
    const msg = e.name === "AbortError"
      ? "Whisper timed out after 2 minutes. Try a shorter clip or check your connection."
      : "Failed to reach OpenAI: " + e.message;
    return new Response(JSON.stringify({ error: msg }), { status: 503, headers: JSON_HEADERS });
  }
  clearTimeout(timer);

  if (!whisperRes.ok) {
    const errText = await whisperRes.text();
    let errMsg = `Whisper error (${whisperRes.status})`;
    try { errMsg = JSON.parse(errText).error?.message || errMsg; } catch {}
    return new Response(JSON.stringify({ error: errMsg }), { status: whisperRes.status, headers: JSON_HEADERS });
  }

  // ── Stream the Whisper response straight back ──
  // This keeps the Netlify edge connection alive during the full Whisper processing time
  // instead of buffering everything and risking a wall-clock timeout.
  return new Response(whisperRes.body, {
    headers: {
      "Content-Type": "application/json",
      "X-Accel-Buffering": "no",
    },
  });
};
