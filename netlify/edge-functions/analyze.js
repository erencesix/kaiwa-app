// netlify/edge-functions/analyze.js
// Streams GPT-4o response via SSE — no timeout wall.

const JSON_HEADERS = { "Content-Type": "application/json" };
const GPT_TIMEOUT_MS = 180_000; // 3 min — generous for large transcripts

export default async (request) => {
  if (request.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method Not Allowed" }), { status: 405, headers: JSON_HEADERS });
  }

  const OPENAI_KEY = Deno.env.get("OPENAI_API_KEY");
  if (!OPENAI_KEY) {
    return new Response(JSON.stringify({ error: "API key not configured." }), { status: 500, headers: JSON_HEADERS });
  }

  let transcript, segments, language, detectedLanguage;
  try {
    const body = await request.json();
    transcript = body.transcript;
    segments = body.segments || [];
    language = body.language || "id";
    detectedLanguage = body.detectedLanguage || null;
  } catch (e) {
    return new Response(JSON.stringify({ error: "Invalid request: " + e.message }), { status: 400, headers: JSON_HEADERS });
  }

  if (!transcript || !transcript.trim()) {
    return new Response(JSON.stringify({ error: "Transcript is empty." }), { status: 400, headers: JSON_HEADERS });
  }

  const outputLangLabel =
    language === "en" ? "English" :
    language === "both" ? "Indonesian and English" :
    "Indonesian (Bahasa Indonesia)";

  // Normalize detectedLanguage defensively
  const sourceLang = (detectedLanguage || "").trim().toLowerCase() || "the original language";
  const isSourceJapanese = sourceLang.includes("japan");
  const sourceLangLabel = sourceLang === "the original language" ? sourceLang
    : sourceLang.charAt(0).toUpperCase() + sourceLang.slice(1);

  const segmentBlock = segments.length > 0
    ? segments.map(s => `[${s.startFormatted}] ${s.text}`).join("\n")
    : transcript;

  // Cap at 12000 chars — enough for ~60min meeting, leaves room for output tokens
  const cappedSegments = segmentBlock.length > 12000
    ? segmentBlock.substring(0, 12000) + "\n[... truncated ...]"
    : segmentBlock;

  const transcriptSchema = isSourceJapanese
    ? `"transcripts": {
    "translated": "[0:00] [Speaker A]: translated text",
    "ja": "[0:00] [Speaker A]: original japanese text",
    "both": "[0:00] [Speaker A - JP]: japanese\\n[0:00] [Speaker A - ID]: translated\\n\\n[0:05] [Speaker B - JP]: japanese\\n[0:05] [Speaker B - ID]: translated"
  }`
    : `"transcripts": {
    "translated": "[0:00] [Speaker A]: translated text\\n[0:05] [Speaker B]: translated text",
    "ja": "[0:00] [Speaker A]: original ${sourceLangLabel} text\\n[0:05] [Speaker B]: original ${sourceLangLabel} text",
    "both": "[0:00] [Speaker A - ORIG]: original ${sourceLangLabel}\\n[0:00] [Speaker A - TRANS]: translated\\n\\n[0:05] [Speaker B - ORIG]: original ${sourceLangLabel}\\n[0:05] [Speaker B - TRANS]: translated"
  }`;

  const systemPrompt = `You are a meeting analyst and translator. Analyze this meeting transcript and return a JSON object.

Source language: ${sourceLangLabel}
Output language: ${outputLangLabel}

RULES — sections must be DISTINCT, no overlap:
- TRANSCRIPT: Every utterance timestamped. Format: "[M:SS] [Speaker A]: text". Include fillers and short responses.
- CHAPTERS: Time blocks by topic. Title = actual topic. No decisions here.
- KEY POINTS: Only explicit decisions, commitments, action items. WHO + WHAT. Nothing observational.
- HIGHLIGHTS: 2-4 verbatim quotes that are surprising or decisive. Explain why each matters.
- SUMMARY: Past-tense narrative. Specific names/numbers. Do NOT repeat key points or highlights verbatim.

Speaker detection: use real names if mentioned, else Speaker A/B/C. Be consistent.
Translation: natural and contextual, never literal. Match formality level.

CRITICAL: Return ONLY valid JSON. No markdown. No code fences. No text before or after the JSON object.

{
  "speakers": [{"id":"speaker_a","label":"Speaker A","name":null,"role":null,"summary":"their specific contribution"}],
  "chapters": [{"title":"Actual Topic","timestamp":"0:00 - 2:30","summary":"what happened"}],
  "tabs": {
    "summary": [{"point":"theme with specific detail","subPoints":["detail","detail"]}],
    "keyPoints": [{"point":"WHO committed to WHAT","subPoints":["condition or deadline"]}],
    "highlights": [{"speaker":"Speaker A","quote":"exact translated quote","context":"why this matters"}]
  },
  ${transcriptSchema}
}`;

  // ── Call GPT-4o with timeout guard ──
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GPT_TIMEOUT_MS);

  let gptResponse;
  try {
    gptResponse = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_KEY}` },
      signal: controller.signal,
      body: JSON.stringify({
        model: "gpt-4o",
        max_tokens: 8192,    // raised from 4096 — prevents mid-JSON truncation on long meetings
        temperature: 0.1,
        stream: true,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Timestamped transcript:\n${cappedSegments}` },
        ],
      }),
    });
  } catch (e) {
    clearTimeout(timer);
    const msg = e.name === "AbortError"
      ? "Analysis timed out after 3 minutes. The recording may be too long."
      : "Failed to reach OpenAI: " + e.message;
    return new Response(JSON.stringify({ error: msg }), { status: 503, headers: JSON_HEADERS });
  }
  clearTimeout(timer);

  if (!gptResponse.ok) {
    const errText = await gptResponse.text();
    let errMsg = `OpenAI error (${gptResponse.status})`;
    try { errMsg = JSON.parse(errText).error?.message || errMsg; } catch {}
    return new Response(JSON.stringify({ error: errMsg }), { status: gptResponse.status, headers: JSON_HEADERS });
  }

  // ── Stream SSE back to browser ──
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();

  const send = async (obj) => {
    try { await writer.write(enc.encode("data: " + JSON.stringify(obj) + "\n\n")); }
    catch (_) { /* client disconnected */ }
  };

  (async () => {
    let accumulated = "";
    let lineBuf = "";   // proper line buffer — survives TCP chunk boundaries

    try {
      const reader = gptResponse.body.getReader();
      const dec = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        lineBuf += dec.decode(value, { stream: true });

        let nl;
        while ((nl = lineBuf.indexOf("\n")) !== -1) {
          const line = lineBuf.slice(0, nl).trimEnd();
          lineBuf = lineBuf.slice(nl + 1);

          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") continue;

          let parsed;
          try { parsed = JSON.parse(raw); } catch { continue; }

          const token = parsed.choices?.[0]?.delta?.content;
          if (token) {
            accumulated += token;
            await send({ token });
          }
        }
      }

      // Strip any stray markdown fences
      const clean = accumulated
        .replace(/^```json\s*/m, "").replace(/^```\s*/m, "").replace(/\s*```$/m, "").trim();

      // Find the outermost JSON object in case GPT prepended any text
      const jsonStart = clean.indexOf("{");
      const jsonEnd = clean.lastIndexOf("}");
      const jsonStr = jsonStart !== -1 && jsonEnd !== -1
        ? clean.slice(jsonStart, jsonEnd + 1)
        : clean;

      let result;
      try {
        result = JSON.parse(jsonStr);
      } catch (e) {
        await send({ error: "Failed to parse analysis. The meeting may be too long — try a shorter recording. Raw: " + clean.substring(0, 200) });
        return;
      }

      // Ensure all three transcript keys always exist so the frontend never crashes
      if (result.transcripts) {
        result.transcripts.translated = result.transcripts.translated || "";
        result.transcripts.ja = result.transcripts.ja || result.transcripts.translated;
        result.transcripts.both = result.transcripts.both || result.transcripts.translated;
      }

      await send({ done: true, result });
    } catch (err) {
      await send({ error: "Stream error: " + err.message });
    } finally {
      try { await writer.close(); } catch (_) { /* already closed */ }
    }
  })();

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
};
