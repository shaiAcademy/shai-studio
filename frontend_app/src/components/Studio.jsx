
import React, { useState, useEffect, useMemo } from "react";
import { generateImage, generateVideo, getRunpodStatus, apiGetTasks } from "../api";
import AppHeader from "./AppHeader";
import {
    IconUpload,
    IconImage,
    IconDownload,
} from "./Icons";

const API_BASE = import.meta?.env?.VITE_API_BASE || "/api";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));


/* =========================
   Вспомогательные API
========================= */
async function apiStartJob({ kind, prompt, steps, token }) {
    const fn = kind === "video" ? generateVideo : generateImage;
    return fn({ prompt, steps, token });
}

async function apiPollStatus({ taskId, token }) {
    const data = await getRunpodStatus({ taskId, token });
    return data;
}

async function fetchTasks(token) {
    return apiGetTasks(token);
}

async function downloadFile(url, filename, type) {
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
   Хелперы
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

function LoadingAnimation({ label }) {
    return (
        <div className="preview-placeholder" style={{ flexDirection: "column", gap: "20px" }}>
            <div
                style={{
                    width: "48px",
                    height: "48px",
                    border: "3px solid #E8E8E8",
                    borderTop: "3px solid #49A598",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite",
                }}
            />
            <span style={{ color: "#49A598", fontWeight: 500 }}>{label}</span>
        </div>
    );
}


/* =========================
   Studio Component
========================= */
export default function Studio({ token, t, lang, setLang, handleLogout }) {
    const [genType, setGenType] = useState("image");
    const [prompt, setPrompt] = useState("");
    const [steps, setSteps] = useState(30);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [tasks, setTasks] = useState([]);

    useEffect(() => {
        if (token) {
            fetchTasks(token)
                .then((data) => {
                    setTasks(
                        data.map((d) => ({
                            task_id: d.task_id,
                            prompt: d.prompt,
                            type: d.kind,
                            status: d.status,
                            url: d.media_url || "",
                            created_at: d.created_at,
                        }))
                    );
                })
                .catch((err) => {
                    console.error("Failed to load tasks", err);
                    if (err?.response?.status === 401) handleLogout();
                });
        }
    }, [token]);

    const handleGenerate = async () => {
        setError("");
        if (!prompt.trim()) return setError(t.promptRequired);

        setLoading(true);
        try {
            const start = await apiStartJob({
                kind: genType,
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
                type: genType,
                url: "",
                created_at: new Date().toISOString(),
            };

            setTasks((prev) => [pendingTask, ...prev]);

            const deadline = Date.now() + 6 * 60 * 1000;
            const intervalMs = 2000;
            let lastStatus = pendingTask.status;
            let mediaUrl = "";

            while (Date.now() < deadline) {
                await sleep(intervalMs);

                const st = await apiPollStatus({ taskId, token });
                lastStatus = st?.status || lastStatus;

                setTasks((prev) =>
                    prev.map((task) =>
                        task.task_id === taskId ? { ...task, status: lastStatus } : task
                    )
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
                prev.map((task) =>
                    task.task_id === taskId ? { ...task, status: "COMPLETED", url: mediaUrl } : task
                )
            );

            setPrompt("");
        } catch (err) {
            setError(err?.response?.data?.detail || err?.message || "Generation failed.");
        } finally {
            setLoading(false);
        }
    };

    const canGenerate = useMemo(() => prompt.trim().length > 0 && !loading, [prompt, loading]);
    const latest = tasks[0];

    return (
        <div className="studio-container">
            <AppHeader
                lang={lang}
                setLang={setLang}
                t={t}
                rightSlot={
                    <>
                        <span className="tag tag-success">{t.signedIn}</span>
                        <button className="btn btn-outline btn-sm" onClick={handleLogout}>
                            {t.logout}
                        </button>
                    </>
                }
            />

            <main className="main-content studio-main">
                <div className="studio-layout">
                    {/* Left Panel — форма генерации */}
                    <div className="studio-panel">
                        <div className="studio-form">
                            <div className="studio-panel-header">
                                <div>
                                    <h3 className="panel-title">{t.promptYourVision}</h3>
                                    <p className="panel-subtitle">{t.videoNote}</p>
                                </div>

                                <div className="type-switcher">
                                    <span
                                        className={`tag ${genType === "image" ? "active" : "tag-default"}`}
                                        onClick={() => setGenType("image")}
                                    >
                                        {t.image}
                                    </span>
                                    <span
                                        className={`tag ${genType === "video" ? "active" : "tag-default"}`}
                                        onClick={() => setGenType("video")}
                                    >
                                        {t.video}
                                    </span>
                                </div>
                            </div>

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

                            <div className="studio-filter-row">
                                <span style={{ color: "#80858C" }}>{t.steps}:</span>
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
                                style={{ width: "100%", marginTop: "24px" }}
                            >
                                <IconUpload /> {loading ? t.generating : t.generate}
                            </button>
                        </div>
                    </div>

                    {/* Right Panel — превью */}
                    <div className="studio-panel">
                        <div className="studio-panel-header">
                            <div className="preview-header">
                                <div className="avatar-ai" />
                                <div>
                                    <h3 className="panel-title">{t.livePreview}</h3>
                                    <p className="panel-subtitle">{t.latestMedia}</p>
                                </div>
                            </div>
                        </div>

                        <div className="preview-area">
                            {loading ? (
                                <LoadingAnimation label={t.generating} />
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

                        <div style={{ height: 1, backgroundColor: "#EFEFEF", width: "100%", margin: "24px 0" }} />

                        {latest && (
                            <div className="preview-info">
                                <div className="preview-info-row">
                                    <span className="preview-info-label">{t.prompt}</span>
                                    <p className="preview-info-value">{latest.prompt}</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* History */}
                <section className="history-section">
                    <div className="history-header">
                        <h3>{t.historyTitle}</h3>
                        <span className="tag tag-default">
                            {tasks.length} {tasks.length === 1 ? t.item : t.items}
                        </span>
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
                                        <div className="skeleton" style={{ width: "100%", height: "100%" }} />
                                    )}
                                </div>

                                <div className="history-card-content">
                                    <div className="history-card-tags">
                                        <span
                                            className="tag tag-success"
                                            style={{ padding: "4px 10px", fontSize: "11px" }}
                                        >
                                            {task.type === "video" ? t.video : t.image}
                                        </span>
                                        <span
                                            className={`tag ${task.status === "COMPLETED" ? "tag-success" : "tag-default"}`}
                                            style={{ padding: "4px 10px", fontSize: "11px" }}
                                        >
                                            {task.status}
                                        </span>
                                    </div>

                                    <p className="history-card-prompt" title={task.prompt}>
                                        {task.prompt}
                                    </p>

                                    <div className="history-card-meta">
                                        <span className="history-card-id">{task.task_id.slice(0, 16)}...</span>

                                        {task.url && (
                                            <button
                                                onClick={() =>
                                                    downloadFile(
                                                        task.url,
                                                        `${task.type}-${task.task_id}`,
                                                        task.type
                                                    )
                                                }
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

            <footer className="footer">
                <div className="footer-content">
                    <span>© 2025 <span style={{ color: "#80858C" }}>shai.academy</span></span>
                    <span style={{ color: "#49A598" }}>|</span>
                    <span>{t.footerText}</span>
                </div>
            </footer>
        </div>
    );
}
