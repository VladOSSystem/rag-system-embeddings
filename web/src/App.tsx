import { useRef, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:3001";

type Msg = { role: "user" | "assistant"; text: string };
type Citation = { i: number; doc_id: string; page: number; stable_id?: string; score?: number };

export default function App() {
  const [file, setFile] = useState<File | null>(null);

  // Option A fileId (still useful if you keep /upload, but Option B doesn't need it)
  const [uploading, setUploading] = useState(false);

  // Option B doc_id (this should match what you ingested into Qdrant, e.g. cv.pdf)
  const [docId, setDocId] = useState("cv.pdf");
  const [topK, setTopK] = useState(6);
  const [citations, setCitations] = useState<Citation[]>([]);

  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<Msg[]>([]);
  const [streaming, setStreaming] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  // Optional: keep uploadPdf if you want Option A.
  // For Option B, ingestion already happened via python -m rag.ingest (no need to upload).
  async function uploadPdf() {
    if (!file) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);

      const res = await fetch(`${API_BASE}/rag/ingest`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();

      // If you want, auto-set docId based on filename for Option B:
      setDocId(data.doc_id);
    } finally {
      setUploading(false);
    }
  }

  async function send() {
    if (!message.trim() || streaming) return;

    const userText = message.trim();
    setMessage("");

    // reset citations for the next assistant answer
    setCitations([]);

    setChat((c) => [...c, { role: "user", text: userText }, { role: "assistant", text: "" }]);
    setStreaming(true);

    const ac = new AbortController();
    abortRef.current = ac;

    try {
      const res = await fetch(`${API_BASE}/rag/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // ✅ Option B payload:
        body: JSON.stringify({
          message: userText,
          collection: "docs",
          doc_id: docId,   // must match what you ingested, e.g. "cv.pdf"
          top_k: topK,
        }),
        signal: ac.signal,
      });

      if (!res.ok || !res.body) throw new Error(await res.text());

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let buf = "";
      let assistant = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });

        let sep;
        while ((sep = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);

          const line = frame.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;

          const payload = line.slice("data: ".length);
          if (payload === "[DONE]") break;

          // Some frames can be non-JSON; ignore safely
          let event: any;
          try {
            event = JSON.parse(payload);
          } catch {
            continue;
          }

          // ✅ Handle citations event sent before streaming text
          if (event?.type === "citations") {
            setCitations(event.citations ?? []);
            continue;
          }

          // ✅ Streamed text delta (Responses API)
          const delta =
            event?.type === "response.output_text.delta"
              ? event?.delta
              : event?.delta ?? null;

          if (typeof delta === "string" && delta.length) {
            assistant += delta;
            setChat((c) => {
              const copy = [...c];
              copy[copy.length - 1] = { role: "assistant", text: assistant };
              return copy;
            });
          }

          // Optional: if SDK sends final text in another event type, you can add it here later.
        }
      }
    } catch (e: any) {
      setChat((c) => {
        const copy = [...c];
        copy[copy.length - 1] = { role: "assistant", text: `Error: ${e?.message || e}` };
        return copy;
      });
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="title">
          <h1>PDF Chat(RAG)</h1>
          <p>
            {docId ? `Doc: ${docId} • top_k=${topK}` : "Set doc_id (e.g. cv.pdf) and ask a question"}
          </p>
        </div>

        <div className="upload">
          <div className="file-pill">
            <input
              id="pdfFile"
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <label htmlFor="pdfFile">Choose PDF</label>
            <span className="file-name">{file?.name ?? "No file selected"}</span>
          </div>

          <button className="btn primary" onClick={uploadPdf} disabled={!file || uploading}>
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>
      </div>

      {/* Controls for Option B */}
      <div className="topbar" style={{ boxShadow: "none" }}>
        <div className="upload" style={{ width: "100%", justifyContent: "space-between" }}>
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ opacity: 0.75, fontSize: 12 }}>doc_id</span>
              <input
                className="input"
                style={{ width: 260, padding: "10px 12px" }}
                value={docId}
                onChange={(e) => setDocId(e.target.value)}
                placeholder="cv.pdf"
                disabled={streaming}
              />
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ opacity: 0.75, fontSize: 12 }}>top_k</span>
              <input
                className="input"
                style={{ width: 90, padding: "10px 12px" }}
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={streaming}
              />
              {/* Button/link to static CV in public/ (e.g. public/cv.pdf) */}
              <a
                className="btn"
                style={{ marginLeft: 8, textDecoration: "none" }}
                href="/cv.pdf"
                target="_blank"
                rel="noopener noreferrer"
              >
                View CV
              </a>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <span style={{ opacity: 0.65, fontSize: 12 }}>
              (Upload isn’t required for ingestion script.)
            </span>
          </div>
        </div>
      </div>

      <div className="chatPanel">
        <div className="chatScroll">
          {chat.map((m, i) => (
            <div key={i} className={`row ${m.role}`}>
              <div className={`bubble ${m.role}`}>
                {m.text || (m.role === "assistant" && streaming ? "…" : "")}

                {/* Show citations under the LAST assistant message */}
                {m.role === "assistant" && i === chat.length - 1 && citations.length > 0 && (
                  <div className="meta" style={{ marginTop: 10 }}>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>Sources</div>
                    <ul style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 4 }}>
                      {citations.map((c, idx) => (
                        <li key={idx} style={{ opacity: 0.85 }}>
                          {c.doc_id} — p.{c.page}
                          {typeof c.score === "number" ? ` (score ${c.score.toFixed(3)})` : ""}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="composer">
          <input
            className="input"
            placeholder={"Ask about the document…"}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            disabled={streaming}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button className="btn primary" onClick={send} disabled={streaming || !message.trim()}>
            Send
          </button>
          <button className="btn danger" onClick={stop} disabled={!streaming}>
            Stop
          </button>
          
        </div>
      </div>
       <footer style={{ padding: 12, textAlign: "center", opacity: 0.85, fontSize: 13 }}>
              <div style={{ fontWeight: 700 }}>How the app was made</div>
              <div>
                PDFs are chunked and embedded, stored in Qdrant, then retrieved and fed as context
                to an OpenAI Responses model for RAG-style answers.
              </div>
            </footer>
    </div>
  );
}
