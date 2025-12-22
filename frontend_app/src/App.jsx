import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
    AppBar,
    Toolbar,
    Box,
    Button,
    Container,
    CssBaseline,
    Grid,
    Paper,
    TextField,
    Typography,
    Alert,
    Card,
    CardContent,
    CardActions,
    Chip,
    Divider,
    Stack,
    LinearProgress,
    Avatar,
    Skeleton,
} from "@mui/material";
import LoginIcon from "@mui/icons-material/Login";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DownloadIcon from "@mui/icons-material/Download";
import RocketLaunchIcon from "@mui/icons-material/RocketLaunch";

const API_BASE = import.meta?.env?.VITE_API_BASE || "/api";

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
    // важное: unknown лучше рендерить как img, чтобы animated webp не ломался
    return { type: "unknown", ext };
}

function RenderMedia({ url, height = 360, controls = true }) {
    if (!url) return null;
    const { type } = inferMediaTypeByUrl(url);

    if (type === "video") {
        return (
            <Box
                component="video"
                src={url}
                controls={controls}
                playsInline
                preload="metadata"
                sx={{
                    width: "100%",
                    borderRadius: 2,
                    height,
                    objectFit: "contain",
                    border: "1px solid #e2e8f0",
                    backgroundColor: "#000",
                    display: "block",
                }}
            />
        );
    }

    return (
        <Box
            component="img"
            src={url}
            alt="result"
            sx={{
                width: "100%",
                borderRadius: 2,
                height,
                objectFit: "contain",
                border: "1px solid #e2e8f0",
                display: "block",
            }}
        />
    );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* =========================
   API calls (RunPod serverless gateway)
   - POST /api/auth/login, /api/auth/register -> {access_token}
   - POST /api/generate/{kind} -> {id,status}
   - GET  /api/generate/status/{id} -> {status, output:{media_url}}
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

export default function App() {
    const [token, setToken] = useState("");
    const [tab, setTab] = useState(0); // 0 auth, 1 studio

    const [authMode, setAuthMode] = useState("login"); // login | register
    const [email, setEmail] = useState("");
    const [name, setName] = useState("");
    const [password, setPassword] = useState("");

    const [genType, setGenType] = useState("image"); // image | video
    const [prompt, setPrompt] = useState("");
    const [steps, setSteps] = useState(30);

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    // history
    const [tasks, setTasks] = useState([]);

    useEffect(() => {
        const saved = localStorage.getItem("gen_token");
        if (saved) setToken(saved);
    }, []);

    useEffect(() => {
        if (token) localStorage.setItem("gen_token", token);
        else localStorage.removeItem("gen_token");
    }, [token]);

    const isAuthed = Boolean(token);
    const latest = tasks[0];

    const quickPrompts = [
        "Cinematic portrait of an astronaut in neon city",
        "Studio photo of a wooden chair on marble floor",
        "Isometric illustration of a smart home dashboard",
        "Moody landscape, misty mountains at sunrise",
    ];

    const canGenerate = useMemo(() => {
        return isAuthed && prompt.trim().length > 0 && !loading;
    }, [isAuthed, prompt, loading]);

    const handleAuth = async () => {
        setError("");
        try {
            let res;
            if (authMode === "register") {
                if (!email || !name || !password) return setError("Email, Name and Password are required.");
                res = await apiRegister({ email, name, password });
            } else {
                if (!email || !password) return setError("Email and Password are required.");
                res = await apiLogin({ email, password });
            }

            if (!res?.access_token) return setError("Auth failed: no access_token returned.");
            setToken(res.access_token);
            setTab(1);
        } catch (err) {
            setError(err?.response?.data?.detail || err?.message || "Auth failed.");
        }
    };

    const handleLogout = () => {
        setToken("");
        setEmail("");
        setPassword("");
        setName("");
        setTab(0);
    };

    const handleGenerate = async () => {
        setError("");
        if (!prompt.trim()) return setError("Prompt is required.");
        if (!token) return setError("Please sign in first.");

        setLoading(true);
        try {
            const kind = genType; // "image" | "video"

            // 1) start job => {id,status}
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

            // 2) poll status until COMPLETED and media_url exists
            const maxWaitMs = 6 * 60 * 1000; // 6 minutes
            const intervalMs = 2000; // 2 sec
            const deadline = Date.now() + maxWaitMs;

            let lastStatus = pendingTask.status;
            let mediaUrl = "";

            while (Date.now() < deadline) {
                await sleep(intervalMs);

                const st = await apiJobStatus({ taskId, token });
                lastStatus = st?.status || lastStatus;

                // update status in UI
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

            // 3) finalize task
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

    return (
        <>
            <CssBaseline />

            <AppBar
                position="sticky"
                color="transparent"
                elevation={0}
                sx={{ backdropFilter: "blur(10px)", borderBottom: "1px solid #e2e8f0" }}
            >
                <Toolbar sx={{ display: "flex", gap: 1 }}>
                    <RocketLaunchIcon color="primary" />
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        shai.academy
                    </Typography>

                    <Chip
                        label={isAuthed ? "Signed in" : "Guest"}
                        color={isAuthed ? "success" : "default"}
                        size="small"
                    />

                    {isAuthed ? (
                        <Button variant="outlined" size="small" onClick={handleLogout}>
                            Logout
                        </Button>
                    ) : null}

                    <Button
                        color="primary"
                        variant="contained"
                        size="small"
                        startIcon={<RocketLaunchIcon />}
                        onClick={() => setTab(1)}
                        disabled={!isAuthed}
                    >
                        Launch Studio
                    </Button>
                </Toolbar>
            </AppBar>

            {loading && <LinearProgress color="primary" />}

            <Container maxWidth="lg" sx={{ py: 4, display: "grid", gap: 3 }}>
                {/* HERO */}
                <Paper
                    elevation={0}
                    sx={{
                        p: 4,
                        borderRadius: 4,
                        background:
                            "linear-gradient(135deg, rgba(59,130,246,0.10), rgba(236,72,153,0.12))",
                        border: "1px solid rgba(148,163,184,0.25)",
                        display: "grid",
                        gap: 2,
                    }}
                >
                    <Box display="flex" alignItems="center" gap={1}>
                        <Chip label="shai.academy" color="primary" size="small" />
                        <Chip label="Creative Lab" variant="outlined" size="small" />
                    </Box>

                    <Typography variant="h4" fontWeight={800} sx={{ letterSpacing: -0.5 }}>
                        Craft production-grade visuals with one click.
                    </Typography>

                    <Typography color="text.secondary" maxWidth="720px">
                        Describe the scene, hit Generate, and get a ready-to-ship image or video. (RunPod serverless + polling)
                    </Typography>

                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                        <Button
                            variant="contained"
                            color="primary"
                            startIcon={<RocketLaunchIcon />}
                            onClick={() => setTab(1)}
                            disabled={!isAuthed}
                        >
                            Start generating
                        </Button>
                        <Button variant="outlined" onClick={() => setTab(0)}>
                            Authenticate
                        </Button>
                    </Stack>

                    <Divider sx={{ borderColor: "rgba(148,163,184,0.4)" }} />

                    <Stack direction={{ xs: "column", md: "row" }} spacing={1} flexWrap="wrap" useFlexGap>
                        {quickPrompts.map((qp) => (
                            <Chip
                                key={qp}
                                label={qp}
                                variant="outlined"
                                onClick={() => {
                                    setPrompt(qp);
                                    setTab(1);
                                }}
                                sx={{ backgroundColor: "rgba(255,255,255,0.6)" }}
                            />
                        ))}
                    </Stack>
                </Paper>

                <Grid container spacing={3}>
                    {/* LEFT */}
                    <Grid item xs={12} md={6}>
                        {tab === 0 ? (
                            /* AUTH */
                            <Paper
                                elevation={0}
                                sx={{
                                    p: 3,
                                    borderRadius: 3,
                                    border: "1px solid #e2e8f0",
                                    backgroundColor: "white",
                                    height: "100%",
                                }}
                            >
                                <Stack spacing={2}>
                                    <Typography variant="subtitle1" fontWeight={700}>
                                        {authMode === "login" ? "Sign in" : "Register"}
                                    </Typography>

                                    <Stack direction="row" spacing={1}>
                                        <Chip
                                            label="Login"
                                            color={authMode === "login" ? "primary" : "default"}
                                            variant={authMode === "login" ? "filled" : "outlined"}
                                            onClick={() => setAuthMode("login")}
                                            clickable
                                        />
                                        <Chip
                                            label="Register"
                                            color={authMode === "register" ? "primary" : "default"}
                                            variant={authMode === "register" ? "filled" : "outlined"}
                                            onClick={() => setAuthMode("register")}
                                            clickable
                                        />
                                    </Stack>

                                    {error && <Alert severity="error">{error}</Alert>}

                                    <TextField
                                        label="Email"
                                        type="email"
                                        fullWidth
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                    />

                                    {authMode === "register" && (
                                        <TextField
                                            label="Name"
                                            fullWidth
                                            value={name}
                                            onChange={(e) => setName(e.target.value)}
                                        />
                                    )}

                                    <TextField
                                        label="Password"
                                        type="password"
                                        fullWidth
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                    />

                                    <Button variant="contained" startIcon={<LoginIcon />} onClick={handleAuth} size="large">
                                        {authMode === "login" ? "Sign In" : "Register"}
                                    </Button>

                                    {isAuthed ? <Alert severity="success">You are signed in.</Alert> : null}
                                </Stack>
                            </Paper>
                        ) : (
                            /* STUDIO */
                            <Paper
                                elevation={0}
                                sx={{
                                    p: 3,
                                    borderRadius: 3,
                                    border: "1px solid #e2e8f0",
                                    backgroundColor: "white",
                                    height: "100%",
                                }}
                            >
                                <Stack spacing={2} height="100%">
                                    <Box display="flex" alignItems="center" justifyContent="space-between" gap={2}>
                                        <Box>
                                            <Typography variant="h6" fontWeight={700}>
                                                Prompt your vision
                                            </Typography>
                                            <Typography color="text.secondary" variant="body2">
                                                Video may return .webp (animated) — it will still render correctly.
                                            </Typography>
                                        </Box>

                                        <Stack direction="row" spacing={1}>
                                            <Chip
                                                label="Image"
                                                color={genType === "image" ? "primary" : "default"}
                                                variant={genType === "image" ? "filled" : "outlined"}
                                                onClick={() => setGenType("image")}
                                                clickable
                                            />
                                            <Chip
                                                label="Video"
                                                color={genType === "video" ? "primary" : "default"}
                                                variant={genType === "video" ? "filled" : "outlined"}
                                                onClick={() => setGenType("video")}
                                                clickable
                                            />
                                        </Stack>
                                    </Box>

                                    {error && <Alert severity="error">{error}</Alert>}

                                    <TextField
                                        label="Prompt"
                                        multiline
                                        minRows={6}
                                        fullWidth
                                        value={prompt}
                                        onChange={(e) => setPrompt(e.target.value)}
                                    />

                                    <TextField
                                        label="Steps"
                                        type="number"
                                        fullWidth
                                        value={steps}
                                        onChange={(e) => setSteps(e.target.value)}
                                        inputProps={{ min: 1, max: 200 }}
                                    />

                                    <Button
                                        fullWidth
                                        size="large"
                                        variant="contained"
                                        color="primary"
                                        startIcon={<CloudUploadIcon />}
                                        onClick={handleGenerate}
                                        disabled={!canGenerate}
                                        sx={{ mt: "auto" }}
                                    >
                                        {loading ? "Generating..." : "Generate"}
                                    </Button>
                                </Stack>
                            </Paper>
                        )}
                    </Grid>

                    {/* RIGHT */}
                    <Grid item xs={12} md={6}>
                        <Paper
                            elevation={0}
                            sx={{
                                p: 3,
                                borderRadius: 3,
                                border: "1px solid #e2e8f0",
                                backgroundColor: "white",
                                height: "100%",
                            }}
                        >
                            <Stack spacing={2}>
                                <Box display="flex" alignItems="center" gap={1}>
                                    <Avatar sx={{ bgcolor: "#312e81" }}>AI</Avatar>
                                    <Box>
                                        <Typography fontWeight={700}>Live preview</Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Latest generated media
                                        </Typography>
                                    </Box>
                                </Box>

                                {latest?.url ? (
                                    <RenderMedia key={latest.url} url={latest.url} height={360} />
                                ) : latest ? (
                                    <Skeleton variant="rectangular" height={360} sx={{ borderRadius: 2 }} />
                                ) : (
                                    <Skeleton variant="rectangular" height={360} sx={{ borderRadius: 2 }} />
                                )}

                                <Divider />

                                {latest ? (
                                    <Stack spacing={0.5}>
                                        <Typography variant="body2" color="text.secondary">
                                            Prompt
                                        </Typography>
                                        <Typography>{latest.prompt}</Typography>
                                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                            Task ID: {latest.task_id}
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            Status: {latest.status}
                                        </Typography>
                                        {latest.url ? (
                                            <Typography variant="body2" color="text.secondary">
                                                URL: {latest.url} (ext: {getExt(latest.url) || "n/a"})
                                            </Typography>
                                        ) : null}
                                    </Stack>
                                ) : (
                                    <Typography color="text.secondary">Generate your first media to see it here.</Typography>
                                )}
                            </Stack>
                        </Paper>
                    </Grid>
                </Grid>

                {/* HISTORY */}
                <Paper
                    elevation={0}
                    sx={{
                        p: 3,
                        borderRadius: 3,
                        border: "1px solid #e2e8f0",
                        backgroundColor: "white",
                    }}
                >
                    <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                        <Typography variant="h6" fontWeight={700}>
                            History & Results
                        </Typography>
                        <Chip label={`${tasks.length} item${tasks.length === 1 ? "" : "s"}`} size="small" />
                    </Box>

                    {!tasks.length && (
                        <Typography color="text.secondary">No tasks yet. Generate your first image/video.</Typography>
                    )}

                    <Grid container spacing={2} sx={{ mt: 1 }}>
                        {tasks.map((task, idx) => (
                            <Grid item xs={12} md={4} key={`${task.task_id}-${idx}`}>
                                <Card variant="outlined" sx={{ borderRadius: 2, borderColor: "#e2e8f0" }}>
                                    <CardContent>
                                        <Stack direction="row" alignItems="center" spacing={1}>
                                            <Chip label={task.type === "video" ? "Video" : "Image"} color="primary" size="small" />
                                            <Typography variant="body2" color="text.secondary" noWrap title={task.prompt}>
                                                {task.prompt}
                                            </Typography>
                                        </Stack>

                                        <Box sx={{ mt: 1.5 }}>
                                            {task.url ? (
                                                <RenderMedia key={task.url} url={task.url} height={180} controls={false} />
                                            ) : (
                                                <Skeleton variant="rectangular" height={180} sx={{ borderRadius: 2 }} />
                                            )}
                                        </Box>

                                        <Divider sx={{ my: 1.2 }} />

                                        <Typography variant="body2" color={task.status === "COMPLETED" ? "success.main" : "text.secondary"}>
                                            Status: {task.status}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            {task.task_id}
                                        </Typography>
                                    </CardContent>

                                    <CardActions>
                                        <Button
                                            size="small"
                                            startIcon={<DownloadIcon />}
                                            component="a"
                                            href={task.url || "#"}
                                            download={task.url ? `${task.type}-${task.task_id}` : undefined}
                                            target="_blank"
                                            rel="noreferrer"
                                            disabled={!task.url}
                                        >
                                            Download
                                        </Button>
                                    </CardActions>
                                </Card>
                            </Grid>
                        ))}
                    </Grid>
                </Paper>
            </Container>

            <Box component="footer" sx={{ py: 3, textAlign: "center", color: "text.secondary", fontSize: 14 }}>
                Crafted for product teams • shai.academy
            </Box>
        </>
    );
}
