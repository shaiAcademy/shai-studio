
import React, { useEffect, useState } from "react";
import { createTheme, ThemeProvider, CssBaseline, Box, CircularProgress } from "@mui/material";
import "./styles.css";
import { translations } from "./translations";
import LandingPage from "./components/LandingPage";
import AuthModal from "./components/AuthModal";
import Studio from "./components/Studio";

// shai.academy Theme
const shaiTheme = createTheme({
    typography: {
        fontFamily: "'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        h1: { fontWeight: 700 },
        h2: { fontWeight: 600 },
        h3: { fontWeight: 600 },
    },
    palette: {
        primary: { main: "#123437" },
        secondary: { main: "#49A598" },
        background: { default: "#F1F3F3", paper: "#FFFFFF" },
        text: { primary: "#2C2B2F", secondary: "#495464" },
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: 8,
                },
            },
        },
    },
});

export default function App() {
    const [lang, setLang] = useState(() => localStorage.getItem("shai_lang") || "en");
    const t = translations[lang];

    const [token, setToken] = useState("");
    const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const saved = localStorage.getItem("gen_token");
        if (saved) {
            setToken(saved);
        }
        // Set loading to false after initial setup
        setIsLoading(false);
    }, []);

    useEffect(() => {
        if (token) localStorage.setItem("gen_token", token);
        else localStorage.removeItem("gen_token");
    }, [token]);

    useEffect(() => {
        localStorage.setItem("shai_lang", lang);
    }, [lang]);

    const handleAuthSuccess = (newToken, email) => {
        setToken(newToken);
        // Set SSO cookie for .shai.academy
        if (email) {
            document.cookie = `shai_user_email=${email}; domain=.shai.academy; path=/; Max-Age=86400; Secure`;
        }
    };

    const handleLogout = () => {
        setToken("");
        // Clear SSO cookie
        document.cookie = "shai_user_email=; domain=.shai.academy; path=/; Max-Age=0; Secure";
        // Reload page to clear all state if needed, or just let state update
        window.location.reload();
    };

    return (
        <ThemeProvider theme={shaiTheme}>
            <CssBaseline />
            <div className="app-container">
                {isLoading ? (
                    <Box 
                        display="flex" 
                        justifyContent="center" 
                        alignItems="center" 
                        minHeight="100vh"
                        flexDirection="column"
                        gap={2}
                    >
                        <CircularProgress size={40} />
                        <Box sx={{ color: 'text.secondary', fontSize: '14px' }}>
                            Loading shai.academy...
                        </Box>
                    </Box>
                ) : token ? (
                    <Studio
                        token={token}
                        t={t}
                        lang={lang}
                        setLang={setLang}
                        handleLogout={handleLogout}
                    />
                ) : (
                    <LandingPage
                        t={t}
                        lang={lang}
                        setLang={setLang}
                        onLoginClick={() => setIsAuthModalOpen(true)}
                    />
                )}

                <AuthModal
                    isOpen={isAuthModalOpen}
                    onClose={() => setIsAuthModalOpen(false)}
                    onSuccess={handleAuthSuccess}
                    t={t}
                />
            </div>
        </ThemeProvider>
    );
}
