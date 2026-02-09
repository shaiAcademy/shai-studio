
import React, { useState, useEffect, useMemo } from "react";
import axios from "axios";
import {
    IconRocket,
    IconUpload,
    IconImage,
    IconDownload,
    IconLogout,
    IconGlobe,
    IconN8n,
    IconDify
} from "./Icons";

// Import custom service icons
import iconN8n from "../icons/n8n.png";
import iconDify from "../icons/dify.png";

// Import loading animation icons
import iconDownload1 from "../icons/download_1.png";
import iconDownload2 from "../icons/download_2.png";
import iconDownload3 from "../icons/download_3.png";
import iconDownload4 from "../icons/download_4.png";

const loadingIcons = [iconDownload1, iconDownload2, iconDownload3, iconDownload4];

// API Base
const API_BASE = import.meta?.env?.VITE_API_BASE || "/api";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* =========================
   API calls (moved from App.jsx)
========================= */
async function apiStartJob({ kind, prompt, steps, token }) {
    const res = await axios.post(
        `${API_BASE}/generate/${kind}`,
        { prompt, steps },
        { headers: token ? { Authorization: `Bearer ${token}` } : {} }
    );
    return res.data;
}

async function apiJobStatus({ taskId, token }) {
    const res = await axios.get(`${API_BASE}/generate/status/${taskId}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    return res.data;
}

async function apiGetTasks(token) {
    const res = await axios.get(`${API_BASE}/tasks`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    return res.data;
}

async function apiDownloadFile(url, filename, type) {
    if (!url) return;
    let key = url;
    if (url.includes("/api/media/")) {
        key = url.split("/api/media/")[1];
    }
    let downloadUrl = `${API_BASE}/media/download/${key}`;
    if (type === "video") {
        downloadUrl += "?format=mp4";
    }
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/* =========================
   Helpers
========================= */
function getExt(url) {
    if (!url) return "";
    const raw = String(url).trim();
    try {
        const u = new URL(raw, window.location.origin);
        const p = (u.pathname || "").toLowerCase();
        const i = p.lastIndexOf(".");
        return i >= 0 ? p.slice(i + 1) : "";
    } catch {
        const clean = raw.split("#")[0].split("?")[0].toLowerCase();
        const i = clean.lastIndexOf(".");
        return i >= 0 ? clean.slice(i + 1) : "";
    }
}

function inferMediaTypeByUrl(url) {
    const ext = getExt(url);
    if (["webp", "png", "jpg", "jpeg", "gif", "avif"].includes(ext)) return { type: "image", ext };
    if (["mp4", "webm", "mov"].includes(ext)) return { type: "video", ext };
    return { type: "unknown", ext };
}

function RenderMedia({ url, height = 360, controls = true }) {
    if (!url) return null;
    const { type } = inferMediaTypeByUrl(url);

    if (type === "video") {
        return (
            <video
                src={`${url}#t=0.01`}
                controls={controls}
                playsInline
                preload="metadata"
                style={{
                    width: "100%",
                    borderRadius: "16px",
                    height,
                    objectFit: "contain",
                    display: "block",
                }}
            />
        );
    }

    return (
        <img
            src={url}
            alt="result"
            style={{
                width: "100%",
                borderRadius: "16px",
                height,
                objectFit: "contain",
                display: "block",
            }}
        />
    );
}

// Loading animation component - smooth spinner
function LoadingAnimation() {
    return (
        <div className="preview-placeholder" style={{ flexDirection: 'column', gap: '20px' }}>
            <div style={{
                width: '48px',
                height: '48px',
                border: '3px solid #E8E8E8',
                borderTop: '3px solid #49A598',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
            }} />
            <span style={{ color: '#49A598', fontWeight: 500 }}>Генерация...</span>
        </div>
    );
}


export default function Studio({ token, t, lang, setLang, handleLogout }) {
    const [genType, setGenType] = useState("image");
    const [prompt, setPrompt] = useState("");
    const [steps, setSteps] = useState(30);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [tasks, setTasks] = useState([]);
    const [langMenuOpen, setLangMenuOpen] = useState(false);
    const [productsMenuOpen, setProductsMenuOpen] = useState(false);

    const langLabels = { en: "EN", ru: "RU", kz: "KZ" };

    useEffect(() => {
        if (token) {
            fetchTasks(token);
        }
    }, [token]);

    const fetchTasks = async (t) => {
        try {
            const data = await apiGetTasks(t);
            const mapped = data.map(d => ({
                task_id: d.task_id,
                prompt: d.prompt,
                type: d.kind,
                status: d.status,
                url: d.media_url || "",
                created_at: d.created_at
            }));
            setTasks(mapped);
        } catch (err) {
            console.error("Failed to load tasks", err);
            if (err?.response?.status === 401) {
                handleLogout();
            }
        }
    };

    const handleGenerate = async () => {
        setError("");
        if (!prompt.trim()) return setError(t.promptRequired);

        setLoading(true);
        try {
            const kind = genType;
            const start = await apiStartJob({
                kind,
                prompt: prompt.trim(),
                steps: Number(steps) || 30,
                token,
            });

            const taskId = start?.id;
            if (!taskId) throw new Error("Server returned no task id.");

            const pendingTask = {
                task_id: taskId,
                status: start?.status || "IN_QUEUE",
                prompt: prompt.trim(),
                type: kind,
                url: "",
                created_at: new Date().toISOString(),
            };

            setTasks((prev) => [pendingTask, ...prev]);

            const maxWaitMs = 6 * 60 * 1000;
            const intervalMs = 2000;
            const deadline = Date.now() + maxWaitMs;

            let lastStatus = pendingTask.status;
            let mediaUrl = "";

            while (Date.now() < deadline) {
                await sleep(intervalMs);

                const st = await apiJobStatus({ taskId, token });
                lastStatus = st?.status || lastStatus;

                setTasks((prev) =>
                    prev.map((t) => (t.task_id === taskId ? { ...t, status: lastStatus } : t))
                );

                if (lastStatus === "COMPLETED") {
                    mediaUrl = st?.output?.media_url || "";
                    if (mediaUrl) break;
                }

                if (lastStatus === "FAILED" || lastStatus === "TIMED_OUT") {
                    throw new Error(`Generation ${lastStatus}`);
                }
            }

            if (!mediaUrl) throw new Error("Timed out waiting for result (no media_url).");

            setTasks((prev) =>
                prev.map((t) => (t.task_id === taskId ? { ...t, status: "COMPLETED", url: mediaUrl } : t))
            );

            setPrompt("");
        } catch (err) {
            setError(err?.response?.data?.detail || err?.message || "Generation failed.");
        } finally {
            setLoading(false);
        }
    };

    const canGenerate = useMemo(() => {
        return prompt.trim().length > 0 && !loading;
    }, [prompt, loading]);

    const latest = tasks[0];

    return (
        <div className="studio-container">
            {/* Header - Matching Landing Page Style */}
            <header className="header landing-header">
                <div className="header-logo">
                    <span className="header-logo-text">shai.<span>academy</span></span>
                </div>

                <div className="header-actions">
                    {/* Language Switcher */}
                    <div
                        onClick={() => setLangMenuOpen(!langMenuOpen)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: '0 12px',
                            height: '40px',
                            background: '#F6F6F6',
                            borderRadius: '16px',
                            cursor: 'pointer',
                            gap: '8px',
                            position: 'relative'
                        }}
                    >
                        <IconGlobe />
                        <span style={{ fontFamily: 'Manrope', fontWeight: 500, fontSize: '14px', color: '#2C2B2F' }}>{langLabels[lang]}</span>

                        {langMenuOpen && (
                            <div className="lang-menu" style={{ top: '48px' }}>
                                {Object.keys(langLabels).map((l) => (
                                    <button
                                        key={l}
                                        className={`lang-menu-item ${lang === l ? 'active' : ''}`}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setLang(l);
                                            setLangMenuOpen(false);
                                        }}
                                    >
                                        {langLabels[l]}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Products Dropdown */}
                    <div
                        onClick={() => setProductsMenuOpen(!productsMenuOpen)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: '0 12px',
                            height: '40px',
                            backgroundColor: '#F6F6F6',
                            border: '1px solid #49A598',
                            borderRadius: '16px',
                            cursor: 'pointer',
                            gap: '8px',
                            position: 'relative'
                        }}
                    >
                        <span style={{ fontFamily: 'Manrope', fontWeight: 500, fontSize: '14px', color: '#2C2B2F' }}>Продукты</span>
                        <span style={{ fontSize: '10px' }}>{productsMenuOpen ? '▲' : '▼'}</span>

                        {productsMenuOpen && (
                            <div style={{
                                position: 'absolute',
                                top: '48px',
                                left: '0',
                                width: '235px',
                                background: 'white',
                                borderRadius: '16px',
                                boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
                                padding: '8px',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '4px',
                                zIndex: 100
                            }}>
                                <a
                                    href="https://n8n.shai.academy"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '12px',
                                        padding: '12px',
                                        textDecoration: 'none',
                                        color: '#2C2B2F',
                                        borderRadius: '12px',
                                        transition: 'background 0.2s',
                                        fontSize: '14px',
                                        fontWeight: 500
                                    }}
                                    onMouseEnter={(e) => e.currentTarget.style.background = '#F6F6F6'}
                                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                                >
                                    <div style={{
                                        width: '32px',
                                        height: '32px',
                                        borderRadius: '8px',
                                        border: '1px solid #EFEFEF',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        overflow: 'hidden'
                                    }}>
                                        <img src={iconN8n} alt="n8n" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
                                    </div>
                                    {t.loginN8n || "Войти в n8n"}
                                </a>
                                <div style={{ height: '1px', background: '#F6F6F6', margin: '0 12px' }}></div>
                                <a
                                    href="https://dify.shai.academy"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '12px',
                                        padding: '12px',
                                        textDecoration: 'none',
                                        color: '#2C2B2F',
                                        borderRadius: '12px',
                                        transition: 'background 0.2s',
                                        fontSize: '14px',
                                        fontWeight: 500
                                    }}
                                    onMouseEnter={(e) => e.currentTarget.style.background = '#F6F6F6'}
                                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                                >
                                    <div style={{
                                        width: '32px',
                                        height: '32px',
                                        borderRadius: '8px',
                                        border: '1px solid #EFEFEF',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        overflow: 'hidden'
                                    }}>
                                        <img src={iconDify} alt="Dify" style={{ width: '24px', height: '24px', objectFit: 'contain' }} />
                                    </div>
                                    {t.loginDify || "Войти в Dify"}
                                </a>
                            </div>
                        )}
                    </div>

                    <span className="tag tag-success">
                        {t.signedIn}
                    </span>

                    <button className="btn btn-outline btn-sm" onClick={handleLogout}>
                        <IconLogout /> {t.logout}
                    </button>
                </div>
            </header>

            <main className="main-content studio-main">
                {/* Studio Layout */}
                <div className="studio-layout">
                    {/* Left Panel */}
                    <div className="studio-panel">
                        <div className="studio-form">
                            <div className="studio-panel-header">
                                <div>
                                    <h3 className="panel-title">{t.promptYourVision}</h3>
                                    <p className="panel-subtitle">{t.videoNote}</p>
                                </div>

                                <div className="type-switcher">
                                    <span
                                        className={`tag ${genType === 'image' ? 'active' : 'tag-default'}`}
                                        onClick={() => setGenType("image")}
                                    >
                                        {t.image}
                                    </span>
                                    <span
                                        className={`tag ${genType === 'video' ? 'active' : 'tag-default'}`}
                                        onClick={() => setGenType("video")}
                                    >
                                        {t.video}
                                    </span>
                                </div>
                            </div>

                            {/* Prompt Input */}
                            <div className="form-group">
                                <label className="form-label">{t.prompt}</label>
                                <textarea
                                    className="form-input form-textarea"
                                    placeholder={t.promptPlaceholder}
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    rows={6}
                                />
                            </div>

                            {/* Use predefined style for filters/steps */}
                            <div className="studio-filter-row">
                                <span style={{ color: '#80858C' }}>{t.steps}:</span>
                                <input
                                    type="number"
                                    className="filter-input"
                                    value={steps}
                                    onChange={(e) => setSteps(e.target.value)}
                                    min={1}
                                    max={200}
                                />
                            </div>

                            {error && <div className="alert alert-error">{error}</div>}

                            <button
                                className="btn btn-accent btn-lg"
                                onClick={handleGenerate}
                                disabled={!canGenerate}
                                style={{ width: '100%', marginTop: '24px' }}
                            >
                                <IconUpload /> {loading ? t.generating : t.generate}
                            </button>
                        </div>
                    </div>

                    {/* Right Panel - Preview */}
                    <div className="studio-panel">
                        <div className="studio-panel-header">
                            <div className="preview-header">
                                <div className="avatar-ai">
                                    <IconGlobe /* AI Icon replacement */ />
                                </div>
                                <div>
                                    <h3 className="panel-title">{t.livePreview}</h3>
                                    <p className="panel-subtitle">{t.latestMedia}</p>
                                </div>
                            </div>
                        </div>

                        <div className="preview-area">
                            {loading ? (
                                <LoadingAnimation />
                            ) : latest?.url ? (
                                <RenderMedia key={latest.url} url={latest.url} height={360} />
                            ) : (
                                <div className="preview-placeholder">
                                    <div className="preview-placeholder-icon">
                                        <IconImage />
                                    </div>
                                    <span>{t.generateFirst}</span>
                                </div>
                            )}
                        </div>

                        {/* Divider */}
                        <div style={{ height: 1, backgroundColor: '#EFEFEF', width: '100%', margin: '24px 0' }} />

                        {latest && (
                            <div className="preview-info">
                                {/* Info about latest generation */}
                                <div className="preview-info-row">
                                    <span className="preview-info-label">{t.prompt}</span>
                                    <p className="preview-info-value">{latest.prompt}</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* History Section */}
                <section className="history-section">
                    <div className="history-header">
                        <h3>{t.historyTitle}</h3>
                        <span className="tag tag-default">{tasks.length} {tasks.length === 1 ? t.item : t.items}</span>
                    </div>

                    <div className="history-divider" />

                    {!tasks.length && (
                        <div className="empty-state-container">
                            <div className="empty-icon"><IconImage /></div>
                            <p className="empty-state">{t.noTasks}</p>
                        </div>
                    )}

                    <div className="history-grid">
                        {tasks.map((task, idx) => (
                            <div className="history-card" key={`${task.task_id}-${idx}`}>
                                <div className="history-card-media">
                                    {task.url ? (
                                        <RenderMedia key={task.url} url={task.url} height={180} controls={false} />
                                    ) : (
                                        <div className="skeleton" style={{ width: '100%', height: '100%' }} />
                                    )}
                                </div>

                                <div className="history-card-content">
                                    <div className="history-card-tags">
                                        <span className="tag tag-success" style={{ padding: '4px 10px', fontSize: '11px' }}>
                                            {task.type === "video" ? t.video : t.image}
                                        </span>
                                        <span className={`tag ${task.status === 'COMPLETED' ? 'tag-success' : 'tag-default'}`} style={{ padding: '4px 10px', fontSize: '11px' }}>
                                            {task.status}
                                        </span>
                                    </div>

                                    <p className="history-card-prompt" title={task.prompt}>{task.prompt}</p>

                                    <div className="history-card-meta">
                                        <span className="history-card-id">{task.task_id.slice(0, 16)}...</span>

                                        {task.url && (
                                            <button
                                                onClick={() => apiDownloadFile(task.url, `${task.type}-${task.task_id}`, task.type)}
                                                className="btn btn-accent btn-sm"
                                            >
                                                <IconDownload /> {t.download}
                                            </button>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            </main>
            {/* Footer */}
            <footer className="footer">
                <div className="footer-content">
                    <span>© 2025 <span style={{ color: '#80858C' }}>shai.academy</span></span>
                    <span style={{ color: '#49A598' }}>|</span>
                    <span>{t.footerText}</span>
                </div>
            </footer>
        </div>
    );
}
