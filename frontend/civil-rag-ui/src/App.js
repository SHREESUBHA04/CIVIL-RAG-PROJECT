import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import {
  Upload, Send, Trash2, FileText, Database,
  CheckCircle, XCircle, ChevronRight, Cpu, Layers
} from "lucide-react";
import "./index.css";
import "./App.css";

const API = "http://localhost:8000";

// ─── Helpers ────────────────────────────────────────────────────

function genSessionId() {
  return Math.random().toString(36).slice(2);
}

function ConfidenceDot({ level }) {
  const colors = {
    high:   { bg: "#EAF3DE", dot: "#3B6D11", label: "High confidence" },
    medium: { bg: "#FAEEDA", dot: "#BA7517", label: "Medium confidence" },
    low:    { bg: "#FCEBEB", dot: "#A32D2D", label: "Low confidence" },
    none:   { bg: "#EFEFEB", dot: "#888780", label: "No retrieval" },
  };
  const c = colors[level] || colors.none;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      background: c.bg, borderRadius: 99, padding: "3px 9px", fontSize: 11,
    }}>
      <span style={{
        width: 7, height: 7, borderRadius: "50%",
        background: c.dot, flexShrink: 0,
      }} />
      {c.label}
    </span>
  );
}

function MetaChip({ children }) {
  return (
    <span style={{
      fontSize: 11, padding: "3px 8px", borderRadius: 99,
      border: "0.5px solid var(--gray-200)", color: "var(--gray-600)",
      display: "inline-flex", alignItems: "center", gap: 4,
    }}>
      {children}
    </span>
  );
}

function SourcePill({ name }) {
  const isCSV = name.endsWith(".csv");
  return (
    <span style={{
      fontSize: 11, padding: "3px 9px", borderRadius: 99,
      background: isCSV ? "#EAF3DE" : "var(--blue-light)",
      color: isCSV ? "var(--green)" : "var(--blue)",
      border: `0.5px solid ${isCSV ? "var(--green-mid)" : "var(--blue-mid)"}`,
    }}>
      {name}
    </span>
  );
}

function StatCard({ value, label }) {
  return (
    <div style={{
      background: "white", borderRadius: "var(--radius-md)",
      border: "0.5px solid var(--gray-200)", padding: "10px 12px",
    }}>
      <div style={{ fontSize: 20, fontWeight: 500 }}>{value}</div>
      <div style={{ fontSize: 11, color: "var(--gray-400)", marginTop: 2 }}>{label}</div>
    </div>
  );
}

function DocItem({ name, chunks }) {
  const isCSV = name.endsWith(".csv");
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "7px 8px", borderRadius: "var(--radius-md)",
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: 6, flexShrink: 0,
        background: isCSV ? "var(--green-light)" : "var(--blue-light)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 10, fontWeight: 600,
        color: isCSV ? "var(--green)" : "var(--blue)",
      }}>
        {isCSV ? "CSV" : "PDF"}
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{
          fontSize: 12, color: "var(--gray-900)",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          {name}
        </div>
        {chunks && (
          <div style={{ fontSize: 11, color: "var(--gray-400)" }}>
            {chunks} chunks
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Message bubble ──────────────────────────────────────────────

function Message({ msg }) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", gap: 10 }}>
        <div style={{
          background: "var(--blue)", color: "white",
          borderRadius: "12px 12px 3px 12px",
          padding: "10px 14px", fontSize: 13.5, lineHeight: 1.6,
          maxWidth: "72%",
        }}>
          {msg.content}
        </div>
        <div style={{
          width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
          background: "var(--gray-100)", border: "0.5px solid var(--gray-200)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 11, fontWeight: 500, color: "var(--gray-600)",
        }}>
          U
        </div>
      </div>
    );
  }

  const { answer, sources, route, reflection_passed, retried, confidence } = msg;

  return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
      <div style={{
        width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
        background: "var(--blue)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <Cpu size={14} color="white" />
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          background: "white", border: "0.5px solid var(--gray-200)",
          borderRadius: "3px 12px 12px 12px",
          padding: "12px 16px", fontSize: 13.5, lineHeight: 1.7,
          color: "var(--gray-900)",
        }}>
          <ReactMarkdown>{answer}</ReactMarkdown>
        </div>

        {/* Meta row */}
        <div style={{
          display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8,
          alignItems: "center",
        }}>
          {confidence && <ConfidenceDot level={confidence.level} />}

          {reflection_passed !== undefined && (
            <MetaChip>
              {reflection_passed
                ? <CheckCircle size={10} color="var(--green)" />
                : <XCircle size={10} color="var(--red)" />}
              Reflection {reflection_passed ? "pass" : "fail"}
            </MetaChip>
          )}

          {route && (
            <MetaChip>
              <Layers size={10} />
              {route.toLowerCase()}
            </MetaChip>
          )}

          {retried && (
            <MetaChip>↺ retried</MetaChip>
          )}

          {sources && sources.map(s => (
            <SourcePill key={s} name={s} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [sessionId]                 = useState(genSessionId);
  const [health, setHealth]         = useState(null);
  const [uploading, setUploading]   = useState(false);
  const [uploadMsg, setUploadMsg]   = useState("");
  const [vectors, setVectors]       = useState(0);
  const [docs, setDocs]             = useState([]);
  const fileRef                     = useRef();
  const bottomRef                   = useRef();

  // Poll health on mount
  useEffect(() => {
    const check = async () => {
      try {
        const r = await axios.get(`${API}/health`);
        setHealth(r.data);
        setVectors(r.data.vectors || 0);
        if (r.data.datasets) {
          setDocs(r.data.datasets.map(d => ({ name: d })));
        }
      } catch {
        setHealth(null);
      }
    };
    check();
    const t = setInterval(check, 30000);
    return () => clearInterval(t);
  }, []);

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setMessages(prev => [...prev, { role: "user", content: q }]);
    setInput("");
    setLoading(true);

    try {
      const r = await axios.post(`${API}/ask`, {
        query: q,
        session_id: sessionId,
      });
      setMessages(prev => [...prev, { role: "assistant", ...r.data }]);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setMessages(prev => [...prev, {
        role: "assistant",
        answer: `Error: ${detail}`,
        sources: [], route: "error",
        reflection_passed: false, retried: false,
        confidence: { level: "none" },
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg("Uploading and indexing...");
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await axios.post(`${API}/upload`, form);
      setUploadMsg(`✓ ${r.data.message}`);
      setVectors(r.data.total_vectors);
      setDocs(prev => {
        const names = prev.map(d => d.name);
        if (!names.includes(file.name)) {
          return [...prev, { name: file.name, chunks: r.data.total_chunks }];
        }
        return prev;
      });
    } catch (err) {
      setUploadMsg(`✗ ${err.response?.data?.detail || "Upload failed"}`);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadMsg(""), 4000);
    }
  };

  const clearChat = async () => {
    try {
      await axios.delete(`${API}/clear_session/${sessionId}`);
    } catch {}
    setMessages([]);
  };

  // ─── Render ───────────────────────────────────────────────────

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "260px 1fr",
      height: "100vh",
      background: "var(--gray-50)",
    }}>

      {/* ── Sidebar ─────────────────────────────────────────── */}
      <div style={{
        background: "var(--gray-100)",
        borderRight: "0.5px solid var(--gray-200)",
        display: "flex", flexDirection: "column",
        padding: 16, gap: 14, overflow: "hidden",
      }}>

        {/* Logo */}
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          paddingBottom: 14,
          borderBottom: "0.5px solid var(--gray-200)",
        }}>
          <div style={{
            width: 34, height: 34, borderRadius: 9,
            background: "var(--blue)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Layers size={18} color="white" />
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 500 }}>Civil RAG</div>
            <div style={{ fontSize: 11, color: "var(--gray-400)" }}>
              Engineering Assistant
            </div>
          </div>
        </div>

        {/* API status */}
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          fontSize: 12, color: health ? "var(--green)" : "var(--red)",
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: "50%",
            background: health ? "var(--green)" : "var(--red)",
          }} />
          {health ? "API online" : "API offline"}
        </div>

        {/* Upload */}
        <div>
          <div style={{
            fontSize: 11, color: "var(--gray-400)",
            textTransform: "uppercase", letterSpacing: "0.05em",
            marginBottom: 8,
          }}>
            Documents
          </div>

          <div
            onClick={() => fileRef.current.click()}
            style={{
              background: "white", border: "0.5px dashed var(--gray-200)",
              borderRadius: "var(--radius-md)", padding: "10px 12px",
              textAlign: "center", cursor: uploading ? "not-allowed" : "pointer",
              opacity: uploading ? 0.6 : 1,
            }}
          >
            <Upload size={14} color="var(--gray-400)" style={{ marginBottom: 4 }} />
            <div style={{ fontSize: 12, color: "var(--gray-600)" }}>
              {uploading ? "Indexing..." : "Upload document"}
            </div>
            <div style={{ fontSize: 11, color: "var(--gray-400)", marginTop: 2 }}>
              PDF · DOCX · TXT
            </div>
          </div>

          <input
            ref={fileRef} type="file"
            accept=".pdf,.docx,.txt,.csv"
            style={{ display: "none" }}
            onChange={handleUpload}
          />

          {uploadMsg && (
            <div style={{
              fontSize: 11, marginTop: 6,
              color: uploadMsg.startsWith("✓") ? "var(--green)" : "var(--red)",
            }}>
              {uploadMsg}
            </div>
          )}
        </div>

        {/* Doc list */}
        <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
          {docs.length === 0 && (
            <div style={{ fontSize: 12, color: "var(--gray-400)", padding: "6px 8px" }}>
              No documents indexed yet
            </div>
          )}
          {docs.map(d => <DocItem key={d.name} name={d.name} chunks={d.chunks} />)}
        </div>

        {/* Stats */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <StatCard value={vectors.toLocaleString()} label="Vectors" />
          <StatCard value={messages.filter(m => m.role === "user").length} label="Queries" />
        </div>
      </div>

      {/* ── Main ────────────────────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Topbar */}
        <div style={{
          padding: "12px 20px",
          borderBottom: "0.5px solid var(--gray-200)",
          background: "white",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ fontSize: 14, fontWeight: 500 }}>
            {messages.length === 0 ? "New conversation" : `${messages.filter(m => m.role === "user").length} questions`}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{
              fontSize: 11, padding: "3px 9px", borderRadius: 99,
              border: "0.5px solid var(--gray-200)", color: "var(--gray-600)",
            }}>
              llama-3.3-70b
            </span>
            <span style={{
              fontSize: 11, padding: "3px 9px", borderRadius: 99,
              border: "0.5px solid var(--gray-200)", color: "var(--gray-600)",
            }}>
              Agentic + reflection
            </span>
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                style={{
                  background: "none", border: "0.5px solid var(--gray-200)",
                  borderRadius: "var(--radius-sm)", padding: "4px 8px",
                  cursor: "pointer", display: "flex", alignItems: "center", gap: 4,
                  fontSize: 11, color: "var(--gray-600)",
                }}
              >
                <Trash2 size={11} /> Clear
              </button>
            )}
          </div>
        </div>

        {/* Chat area */}
        <div style={{
          flex: 1, overflowY: "auto",
          padding: "24px 28px",
          display: "flex", flexDirection: "column", gap: 20,
        }}>
          {messages.length === 0 && (
            <div style={{
              margin: "auto", textAlign: "center",
              color: "var(--gray-400)", maxWidth: 380,
            }}>
              <Layers size={32} color="var(--gray-200)" style={{ marginBottom: 12 }} />
              <div style={{ fontSize: 16, fontWeight: 500, color: "var(--gray-600)", marginBottom: 6 }}>
                Civil Engineering Assistant
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                Ask questions about concrete technology, IS codes, mix design,
                or your uploaded datasets.
              </div>
              <div style={{
                display: "flex", flexWrap: "wrap", gap: 8,
                justifyContent: "center", marginTop: 20,
              }}>
                {[
                  "What is water cement ratio?",
                  "Explain IS 456 provisions",
                  "Average strength in dataset",
                  "Types of cement",
                ].map(s => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    style={{
                      background: "white", border: "0.5px solid var(--gray-200)",
                      borderRadius: 99, padding: "6px 14px",
                      fontSize: 12, color: "var(--gray-600)", cursor: "pointer",
                    }}
                  >
                    {s} <ChevronRight size={10} style={{ verticalAlign: "middle" }} />
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <Message key={i} msg={msg} />
          ))}

          {loading && (
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <div style={{
                width: 28, height: 28, borderRadius: "50%",
                background: "var(--blue)",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                <Cpu size={14} color="white" />
              </div>
              <div style={{
                background: "white", border: "0.5px solid var(--gray-200)",
                borderRadius: "3px 12px 12px 12px",
                padding: "10px 16px", display: "flex", gap: 5, alignItems: "center",
              }}>
                {[0, 1, 2].map(i => (
                  <span key={i} style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: "var(--gray-200)",
                    animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
                  }} />
                ))}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div style={{
          padding: "12px 20px",
          borderTop: "0.5px solid var(--gray-200)",
          background: "white",
          display: "flex", gap: 10, alignItems: "flex-end",
        }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a civil engineering question... (Enter to send)"
            rows={1}
            style={{
              flex: 1, background: "var(--gray-100)",
              border: "0.5px solid var(--gray-200)",
              borderRadius: 10, padding: "9px 14px",
              fontSize: 13.5, fontFamily: "inherit",
              resize: "none", outline: "none",
              color: "var(--gray-900)", lineHeight: 1.5,
            }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            style={{
              width: 38, height: 38, borderRadius: 9,
              background: input.trim() && !loading ? "var(--blue)" : "var(--gray-200)",
              border: "none", cursor: input.trim() && !loading ? "pointer" : "not-allowed",
              display: "flex", alignItems: "center", justifyContent: "center",
              flexShrink: 0, transition: "background 0.15s",
            }}
          >
            <Send size={15} color="white" />
          </button>
        </div>
      </div>
    </div>
  );
}