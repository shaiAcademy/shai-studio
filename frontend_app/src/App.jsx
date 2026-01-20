import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { createTheme, ThemeProvider, CssBaseline } from "@mui/material";
import "./styles.css";

const API_BASE = import.meta?.env?.VITE_API_BASE || "/api";

// shai.academy Theme
const shaiTheme = createTheme({
    typography: {
        fontFamily: "'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    palette: {
        primary: { main: "#123437" },
        secondary: { main: "#49A598" },
        background: { default: "#F1F3F3", paper: "#FFFFFF" },
        text: { primary: "#2C2B2F", secondary: "#495464" },
    },
});

/* =========================
   Translations (EN, RU, KZ)
========================= */
const translations = {
    en: {
        // Header
        signedIn: "Signed in",
        guest: "Guest",
        logout: "Logout",
        launchStudio: "Launch Studio",
        openN8n: "Open n8n",
        n8nRedirecting: "Redirecting...",
        n8nError: "Failed to redirect to n8n",

        // Hero
        heroTitle: "Craft production-grade visuals with AI",
        heroDescription: "Describe your vision and let AI generate stunning images or videos in seconds.",
        startGenerating: "Start generating",
        authenticate: "Authenticate",

        // Auth
        signIn: "Sign in",
        register: "Register",
        login: "Login",
        email: "Email",
        emailPlaceholder: "Enter your email",
        name: "Name",
        namePlaceholder: "Enter your name",
        password: "Password",
        passwordPlaceholder: "Enter your password",
        youAreSignedIn: "You are signed in.",
        authRequired: "Email, Name and Password are required.",
        loginRequired: "Email and Password are required.",
        authFailed: "Auth failed",

        // Flow Registration
        step1Title: "Enter your email",
        step2Title: "What's your name?",
        step3Title: "Create a password",
        next: "Next",
        back: "Back",
        createAccount: "Create Account",

        // Studio
        promptYourVision: "Prompt your vision",
        videoNote: "Video may return .webp (animated) — it will still render correctly.",
        image: "Image",
        video: "Video",
        prompt: "Prompt",
        promptPlaceholder: "Describe what you want to generate...",
        steps: "Steps",
        generate: "Generate",
        generating: "Generating...",
        promptRequired: "Prompt is required.",
        signInFirst: "Please sign in first.",

        // Preview
        livePreview: "Live preview",
        latestMedia: "Latest generated media",
        generateFirst: "Generate your first media to see it here",
        taskId: "Task ID",
        status: "Status",

        // History
        historyTitle: "History & Results",
        items: "items",
        item: "item",
        noTasks: "No tasks yet. Generate your first image/video.",
        download: "Download",

        // Footer
        footerText: "Crafted for product teams",

        // Quick prompts
        quickPrompts: [
            "Cinematic portrait of an astronaut in neon city",
            "Studio photo of a wooden chair on marble floor",
            "Isometric illustration of a smart home dashboard",
            "Moody landscape, misty mountains at sunrise",
        ],
    },
    ru: {
        // Header
        signedIn: "Авторизован",
        guest: "Гость",
        logout: "Выйти",
        launchStudio: "Открыть студию",
        openN8n: "Открыть n8n",
        n8nRedirecting: "Перенаправление...",
        n8nError: "Не удалось перейти в n8n",

        // Hero
        heroTitle: "Создавайте визуальный контент с помощью ИИ",
        heroDescription: "Опишите свою идею и позвольте ИИ создать потрясающие изображения или видео за секунды.",
        startGenerating: "Начать генерацию",
        authenticate: "Авторизация",

        // Auth
        signIn: "Вход",
        register: "Регистрация",
        login: "Вход",
        email: "Электронная почта",
        emailPlaceholder: "Введите email",
        name: "Имя",
        namePlaceholder: "Введите ваше имя",
        password: "Пароль",
        passwordPlaceholder: "Введите пароль",
        youAreSignedIn: "Вы авторизованы.",
        authRequired: "Требуются Email, Имя и Пароль.",
        loginRequired: "Требуются Email и Пароль.",
        authFailed: "Ошибка авторизации",

        // Flow Registration
        step1Title: "Введите ваш email",
        step2Title: "Как вас зовут?",
        step3Title: "Придумайте пароль",
        next: "Далее",
        back: "Назад",
        createAccount: "Создать аккаунт",

        // Studio
        promptYourVision: "Опишите вашу идею",
        videoNote: "Видео может вернуться в формате .webp (анимация) — оно отобразится корректно.",
        image: "Изображение",
        video: "Видео",
        prompt: "Промпт",
        promptPlaceholder: "Опишите, что хотите сгенерировать...",
        steps: "Шаги",
        generate: "Генерировать",
        generating: "Генерация...",
        promptRequired: "Требуется промпт.",
        signInFirst: "Сначала войдите в систему.",

        // Preview
        livePreview: "Предпросмотр",
        latestMedia: "Последний сгенерированный контент",
        generateFirst: "Сгенерируйте первый контент, чтобы увидеть его здесь",
        taskId: "ID задачи",
        status: "Статус",

        // History
        historyTitle: "История и результаты",
        items: "элементов",
        item: "элемент",
        noTasks: "Пока нет задач. Сгенерируйте первое изображение/видео.",
        download: "Скачать",

        // Footer
        footerText: "Создано для продуктовых команд",

        // Quick prompts
        quickPrompts: [
            "Кинематографичный портрет астронавта в неоновом городе",
            "Студийное фото деревянного стула на мраморном полу",
            "Изометрическая иллюстрация панели умного дома",
            "Атмосферный пейзаж, туманные горы на рассвете",
        ],
    },
    kz: {
        // Header
        signedIn: "Кірген",
        guest: "Қонақ",
        logout: "Шығу",
        launchStudio: "Студияны ашу",
        openN8n: "n8n ашу",
        n8nRedirecting: "Қайта бағыттау...",
        n8nError: "n8n-ге өту мүмкін болмады",

        // Hero
        heroTitle: "AI көмегімен визуалды контент жасаңыз",
        heroDescription: "Өз идеяңызды сипаттап, AI-ға бірнеше секундта керемет суреттер немесе бейнелер жасатыңыз.",
        startGenerating: "Генерацияны бастау",
        authenticate: "Авторизация",

        // Auth
        signIn: "Кіру",
        register: "Тіркелу",
        login: "Кіру",
        email: "Электрондық пошта",
        emailPlaceholder: "Email енгізіңіз",
        name: "Аты",
        namePlaceholder: "Атыңызды енгізіңіз",
        password: "Құпия сөз",
        passwordPlaceholder: "Құпия сөзді енгізіңіз",
        youAreSignedIn: "Сіз жүйеге кірдіңіз.",
        authRequired: "Email, Аты және Құпия сөз қажет.",
        loginRequired: "Email және Құпия сөз қажет.",
        authFailed: "Авторизация қатесі",

        // Flow Registration
        step1Title: "Email енгізіңіз",
        step2Title: "Атыңыз қандай?",
        step3Title: "Құпия сөз жасаңыз",
        next: "Келесі",
        back: "Артқа",
        createAccount: "Аккаунт жасау",

        // Studio
        promptYourVision: "Идеяңызды сипаттаңыз",
        videoNote: "Бейне .webp (анимация) форматында қайтарылуы мүмкін — ол дұрыс көрсетіледі.",
        image: "Сурет",
        video: "Бейне",
        prompt: "Промпт",
        promptPlaceholder: "Не жасағыңыз келетінін сипаттаңыз...",
        steps: "Қадамдар",
        generate: "Генерациялау",
        generating: "Генерация...",
        promptRequired: "Промпт қажет.",
        signInFirst: "Алдымен жүйеге кіріңіз.",

        // Preview
        livePreview: "Алдын ала қарау",
        latestMedia: "Соңғы жасалған контент",
        generateFirst: "Мұнда көру үшін алғашқы контентті жасаңыз",
        taskId: "Тапсырма ID",
        status: "Күй",

        // History
        historyTitle: "Тарих және нәтижелер",
        items: "элементтер",
        item: "элемент",
        noTasks: "Әзірге тапсырмалар жоқ. Алғашқы сурет/бейнені жасаңыз.",
        download: "Жүктеу",

        // Footer
        footerText: "Өнім топтары үшін жасалған",

        // Quick prompts
        quickPrompts: [
            "Неон қаласындағы ғарышкердің кинематографиялық портреті",
            "Мәрмәр еденде тұрған ағаш орындықтың студиялық фотосы",
            "Ақылды үй панелінің изометриялық иллюстрациясы",
            "Таңғы тау шыңдарының тұманды пейзажы",
        ],
    },
};

/* =========================
   Helpers: URL -> media type
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

/* =========================
   Download helper (cross-origin safe)
========================= */
/* =========================
   Download helper
========================= */
function downloadFile(url, filename, type) {
    if (!url) return;

    // Extract key from URL
    // URL format: /api/media/path/to/file or full URL
    let key = url;
    if (url.includes("/api/media/")) {
        key = url.split("/api/media/")[1];
    }

    let downloadUrl = `${API_BASE}/media/download/${key}`;

    // Request MP4 conversion for videos
    if (type === "video") {
        downloadUrl += "?format=mp4";
    }

    // Trigger download via new window/iframe or anchor
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.setAttribute("download", filename); // Hint, though Content-Disposition rules
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* =========================
   API calls
========================= */
async function apiRegister({ email, name, password }) {
    const res = await axios.post(`${API_BASE}/auth/register`, { email, name, password });
    return res.data;
}

async function apiLogin({ email, password }) {
    const res = await axios.post(`${API_BASE}/auth/login`, { email, password });
    return res.data;
}

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

/* =========================
   Icons (inline SVG)
========================= */
const IconRocket = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z" />
        <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
        <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
        <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
    </svg>
);

const IconLogin = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
        <polyline points="10 17 15 12 10 7" />
        <line x1="15" y1="12" x2="3" y2="12" />
    </svg>
);

const IconUpload = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
);

const IconDownload = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
);

const IconImage = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <polyline points="21 15 16 10 5 21" />
    </svg>
);

const IconLogout = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
        <polyline points="16 17 21 12 16 7" />
        <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
);

const IconGlobe = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="2" y1="12" x2="22" y2="12" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
);

const IconN8n = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
    </svg>
);

export default function App() {
    const [lang, setLang] = useState(() => localStorage.getItem("shai_lang") || "en");
    const t = translations[lang];

    const [token, setToken] = useState("");
    const [tab, setTab] = useState(() => localStorage.getItem("gen_token") ? 1 : 0);

    const [authMode, setAuthMode] = useState("login");
    const [regStep, setRegStep] = useState(1);
    const [email, setEmail] = useState("");
    const [name, setName] = useState("");
    const [password, setPassword] = useState("");

    const [genType, setGenType] = useState("image");
    const [prompt, setPrompt] = useState("");
    const [steps, setSteps] = useState(30);

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const [tasks, setTasks] = useState([]);

    const [langMenuOpen, setLangMenuOpen] = useState(false);
    const [n8nLoading, setN8nLoading] = useState(false);

    useEffect(() => {
        const saved = localStorage.getItem("gen_token");
        if (saved) {
            setToken(saved);
            fetchTasks(saved);
        }
    }, []);

    const fetchTasks = async (t) => {
        try {
            const data = await apiGetTasks(t);
            // Map backend TaskResponse to frontend structure
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
        }
    };

    useEffect(() => {
        if (token) localStorage.setItem("gen_token", token);
        else localStorage.removeItem("gen_token");
    }, [token]);

    useEffect(() => {
        setRegStep(1);
    }, [authMode]);

    useEffect(() => {
        localStorage.setItem("shai_lang", lang);
    }, [lang]);

    const isAuthed = Boolean(token);
    const latest = tasks[0];

    const canGenerate = useMemo(() => {
        return isAuthed && prompt.trim().length > 0 && !loading;
    }, [isAuthed, prompt, loading]);

    const handleAuth = async () => {
        setError("");
        try {
            let res;
            if (authMode === "register") {
                if (!email || !name || !password) return setError(t.authRequired);
                res = await apiRegister({ email, name, password });
            } else {
                if (!email || !password) return setError(t.loginRequired);
                res = await apiLogin({ email, password });
            }

            if (!res?.access_token) return setError(t.authFailed + ": no access_token returned.");
            setToken(res.access_token);

            // Set SSO cookie for .shai.academy
            const userEmail = email; // captured from state
            document.cookie = `shai_user_email=${userEmail}; domain=.shai.academy; path=/; Max-Age=86400; Secure`;

            fetchTasks(res.access_token);
            setTab(1);
        } catch (err) {
            let msg = err?.response?.data?.detail || err?.message || t.authFailed;
            if (typeof msg !== 'string') {
                msg = JSON.stringify(msg);
            }
            setError(msg);
        }
    };

    const handleLogout = () => {
        setToken("");
        setEmail("");
        setPassword("");
        setName("");
        setTab(0);
    };

    const handleN8nRedirect = async () => {
        if (!token) {
            setError(t.signInFirst);
            return;
        }

        setN8nLoading(true);
        setError("");

        try {
            const response = await axios.post(
                `${API_BASE}/n8n/redirect`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );

            const data = response.data;

            if (data.success && data.redirectUrl) {
                window.location.href = data.redirectUrl;
            } else {
                throw new Error(t.n8nError);
            }
        } catch (err) {
            console.error("Error redirecting to n8n:", err);
            const msg = err?.response?.data?.detail || err?.message || t.n8nError;
            setError(typeof msg === 'string' ? msg : JSON.stringify(msg));
            setN8nLoading(false);
        }
    };

    const handleGenerate = async () => {
        setError("");
        if (!prompt.trim()) return setError(t.promptRequired);
        if (!token) return setError(t.signInFirst);

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

    const langLabels = { en: "EN", ru: "RU", kz: "KZ" };

    return (
        <ThemeProvider theme={shaiTheme}>
            <CssBaseline />
            <div className="app-container">
                {loading && <div className="loading-bar" />}

                {/* Header */}
                <header className="header">
                    <div className="header-logo">
                        <span className="header-logo-text">shai.<span>academy</span></span>
                    </div>

                    <div className="header-actions">
                        {/* Language Switcher */}
                        <div className="lang-switcher">
                            <button
                                className="lang-switcher-btn"
                                onClick={() => setLangMenuOpen(!langMenuOpen)}
                            >
                                <IconGlobe /> {langLabels[lang]}
                            </button>
                            {langMenuOpen && (
                                <div className="lang-menu">
                                    {Object.keys(langLabels).map((l) => (
                                        <button
                                            key={l}
                                            className={`lang-menu-item ${lang === l ? 'active' : ''}`}
                                            onClick={() => {
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

                        <span className={`tag ${isAuthed ? 'tag-success' : 'tag-default'}`}>
                            {isAuthed ? t.signedIn : t.guest}
                        </span>

                        {isAuthed && (
                            <button className="btn btn-outline btn-sm" onClick={handleLogout}>
                                <IconLogout /> {t.logout}
                            </button>
                        )}

                        <button
                            className="btn btn-outline"
                            onClick={() => window.location.href = "https://n8n.shai.academy"}
                            style={{ borderColor: '#FF6D5A', color: '#FF6D5A' }}
                        >
                            <IconN8n /> n8n
                        </button>

                        <button
                            className="btn btn-outline"
                            onClick={() => window.location.href = "https://dify.shai.academy"}
                            style={{ borderColor: '#1C64F2', color: '#1C64F2', marginLeft: '8px' }}
                        >
                            <IconGlobe /> Dify
                        </button>

                        <button
                            className="btn btn-primary"
                            onClick={() => setTab(1)}
                            disabled={!isAuthed}
                        >
                            <IconRocket /> {t.launchStudio}
                        </button>
                    </div>
                </header>

                {/* Main content */}
                <main className="main-content">
                    {/* Hero Section */}
                    <section className="hero">
                        <div className="hero-content">
                            <div className="hero-tags">
                                <span className="tag tag-accent">SHAI.ACADEMY</span>
                                <span className="tag tag-accent">CREATIVE STUDIO</span>
                            </div>

                            <h1>{t.heroTitle}</h1>

                            <p>{t.heroDescription}</p>

                            <div className="hero-actions">
                                <button
                                    className="btn btn-white btn-lg"
                                    onClick={() => setTab(1)}
                                    disabled={!isAuthed}
                                >
                                    <IconRocket /> {t.startGenerating}
                                </button>
                                <button
                                    className="btn btn-outline btn-lg"
                                    style={{ borderColor: 'rgba(255,255,255,0.3)', color: '#fff' }}
                                    onClick={() => setTab(0)}
                                >
                                    {t.authenticate}
                                </button>
                            </div>

                            <div className="quick-prompts">
                                {t.quickPrompts.map((qp) => (
                                    <span
                                        key={qp}
                                        className="quick-prompt"
                                        onClick={() => {
                                            setPrompt(qp);
                                            setTab(1);
                                        }}
                                    >
                                        {qp}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </section>

                    {/* Studio Layout */}
                    <div className="studio-layout">
                        {/* Left Panel */}
                        <div className="studio-panel">
                            {tab === 0 ? (
                                /* AUTH PANEL */
                                <div className="auth-panel">
                                    <h3 className="panel-title">{authMode === "login" ? t.signIn : t.register}</h3>

                                    <div className="auth-switcher">
                                        <span
                                            className={`tag ${authMode === 'login' ? 'tag-success' : 'tag-default'}`}
                                            onClick={() => setAuthMode("login")}
                                            style={{ cursor: 'pointer' }}
                                        >
                                            {t.login}
                                        </span>
                                        <span
                                            className={`tag ${authMode === 'register' ? 'tag-success' : 'tag-default'}`}
                                            onClick={() => setAuthMode("register")}
                                            style={{ cursor: 'pointer' }}
                                        >
                                            {t.register}
                                        </span>
                                    </div>

                                    {error && <div className="alert alert-error">{error}</div>}

                                    {authMode === "login" ? (
                                        <>
                                            <div className="form-group">
                                                <label className="form-label">{t.email}</label>
                                                <input
                                                    type="email"
                                                    className="form-input"
                                                    placeholder={t.emailPlaceholder}
                                                    value={email}
                                                    onChange={(e) => setEmail(e.target.value)}
                                                />
                                            </div>

                                            <div className="form-group">
                                                <label className="form-label">{t.password}</label>
                                                <input
                                                    type="password"
                                                    className="form-input"
                                                    placeholder={t.passwordPlaceholder}
                                                    value={password}
                                                    onChange={(e) => setPassword(e.target.value)}
                                                />
                                            </div>

                                            <button className="btn btn-accent btn-lg" onClick={handleAuth} style={{ width: '100%', marginTop: '8px' }}>
                                                <IconLogin /> {t.signIn}
                                            </button>
                                        </>
                                    ) : (
                                        <div className="reg-flow">
                                            <div className="reg-progress">
                                                <div className={`reg-progress-bar step-${regStep}`} />
                                            </div>

                                            {regStep === 1 && (
                                                <div className="reg-step-content">
                                                    <h4 className="reg-step-title">{t.step1Title}</h4>
                                                    <div className="form-group">
                                                        <input
                                                            type="email"
                                                            className="form-input"
                                                            placeholder={t.emailPlaceholder}
                                                            value={email}
                                                            onChange={(e) => setEmail(e.target.value)}
                                                            autoFocus
                                                        />
                                                    </div>
                                                    <button
                                                        className="btn btn-accent btn-lg"
                                                        disabled={!email.includes('@')}
                                                        onClick={() => setRegStep(2)}
                                                        style={{ width: '100%' }}
                                                    >
                                                        {t.next}
                                                    </button>
                                                </div>
                                            )}

                                            {regStep === 2 && (
                                                <div className="reg-step-content">
                                                    <h4 className="reg-step-title">{t.step2Title}</h4>
                                                    <div className="form-group">
                                                        <input
                                                            type="text"
                                                            className="form-input"
                                                            placeholder={t.namePlaceholder}
                                                            value={name}
                                                            onChange={(e) => setName(e.target.value)}
                                                            autoFocus
                                                        />
                                                    </div>
                                                    <div className="reg-step-actions">
                                                        <button className="btn btn-outline" onClick={() => setRegStep(1)}>{t.back}</button>
                                                        <button
                                                            className="btn btn-accent"
                                                            disabled={name.length < 2}
                                                            onClick={() => setRegStep(3)}
                                                            style={{ flex: 1 }}
                                                        >
                                                            {t.next}
                                                        </button>
                                                    </div>
                                                </div>
                                            )}

                                            {regStep === 3 && (
                                                <div className="reg-step-content">
                                                    <h4 className="reg-step-title">{t.step3Title}</h4>
                                                    <div className="form-group">
                                                        <input
                                                            type="password"
                                                            className="form-input"
                                                            placeholder={t.passwordPlaceholder}
                                                            value={password}
                                                            onChange={(e) => setPassword(e.target.value)}
                                                            autoFocus
                                                        />
                                                    </div>
                                                    <div className="reg-step-actions">
                                                        <button className="btn btn-outline" onClick={() => setRegStep(2)}>{t.back}</button>
                                                        <button
                                                            className="btn btn-accent"
                                                            disabled={password.length < 6}
                                                            onClick={handleAuth}
                                                            style={{ flex: 1 }}
                                                        >
                                                            {t.createAccount}
                                                        </button>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {isAuthed && <div className="alert alert-success">{t.youAreSignedIn}</div>}
                                </div>
                            ) : (
                                /* STUDIO PANEL */
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

                                    {error && <div className="alert alert-error">{error}</div>}

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

                                    <div className="form-group">
                                        <label className="form-label">{t.steps}</label>
                                        <input
                                            type="number"
                                            className="form-input"
                                            value={steps}
                                            onChange={(e) => setSteps(e.target.value)}
                                            min={1}
                                            max={200}
                                        />
                                    </div>

                                    <button
                                        className="btn btn-accent btn-lg"
                                        onClick={handleGenerate}
                                        disabled={!canGenerate}
                                        style={{ width: '100%', marginTop: '8px' }}
                                    >
                                        <IconUpload /> {loading ? t.generating : t.generate}
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Right Panel - Preview */}
                        <div className="studio-panel">
                            <div className="studio-panel-header">
                                <div className="preview-header">
                                    <div className="avatar">AI</div>
                                    <div>
                                        <h3 className="panel-title">{t.livePreview}</h3>
                                        <p className="panel-subtitle">{t.latestMedia}</p>
                                    </div>
                                </div>
                            </div>

                            <div className="preview-area">
                                {latest?.url ? (
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

                            <div className="divider" />

                            {latest && (
                                <div className="preview-info">
                                    <div className="preview-info-row">
                                        <span className="preview-info-label">{t.prompt}</span>
                                        <p className="preview-info-value">{latest.prompt}</p>
                                    </div>
                                    <div className="preview-info-row">
                                        <span className="preview-info-label">{t.taskId}</span>
                                        <code className="preview-info-code">{latest.task_id}</code>
                                    </div>
                                    <div className="preview-info-row">
                                        <span className="preview-info-label">{t.status}</span>
                                        <span className={`tag ${latest.status === 'COMPLETED' ? 'tag-success' : 'tag-default'}`} style={{ padding: '4px 8px', fontSize: '12px' }}>
                                            {latest.status}
                                        </span>
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

                        {!tasks.length && (
                            <p className="empty-state">{t.noTasks}</p>
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
                                                    onClick={() => downloadFile(task.url, `${task.type}-${task.task_id}`, task.type)}
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
                    © 2025 <a href="https://shai.academy">shai.academy</a> | {t.footerText}
                </footer>
            </div>
        </ThemeProvider>
    );
}
