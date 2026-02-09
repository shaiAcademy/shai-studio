
import React, { useState, useEffect } from "react";
import axios from "axios";
import { IconLogin, IconClose } from "./Icons";

const API_BASE = import.meta?.env?.VITE_API_BASE || "/api";

async function apiRegister({ email, name, password }) {
    const res = await axios.post(`${API_BASE}/auth/register`, { email, name, password });
    return res.data;
}

async function apiLogin({ email, password }) {
    const res = await axios.post(`${API_BASE}/auth/login`, { email, password });
    return res.data;
}

export default function AuthModal({ isOpen, onClose, onSuccess, t }) {
    const [authMode, setAuthMode] = useState("login");
    const [regStep, setRegStep] = useState(1);
    const [email, setEmail] = useState("");
    const [name, setName] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (isOpen) {
            setAuthMode("login");
            setRegStep(1);
            setError("");
        }
    }, [isOpen]);

    if (!isOpen) return null;

    const handleAuth = async () => {
        setError("");
        setLoading(true);
        try {
            let res;
            if (authMode === "register") {
                if (!email || !name || !password) {
                    setLoading(false);
                    return setError(t.authRequired);
                }
                res = await apiRegister({ email, name, password });
            } else {
                if (!email || !password) {
                    setLoading(false);
                    return setError(t.loginRequired);
                }
                res = await apiLogin({ email, password });
            }

            if (!res?.access_token) {
                setLoading(false);
                return setError(t.authFailed + ": no access_token returned.");
            }

            // Success
            onSuccess(res.access_token, email); // Pass email for SSO cookie if needed
            onClose();

        } catch (err) {
            let msg = err?.response?.data?.detail || err?.message || t.authFailed;
            if (typeof msg !== 'string') {
                msg = JSON.stringify(msg);
            }
            setError(msg);
        } finally {
            setLoading(false);
        }
    };


    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <button className="modal-close" onClick={onClose}>
                    <IconClose />
                </button>

                <div className="modal-header">
                    <h3 className="modal-title">{authMode === "login" ? t.signIn : t.register}</h3>
                    <div className="auth-switcher">
                        {/* Switcher logic */}
                        <span
                            className={`tag ${authMode === 'login' ? 'active' : ''}`}
                            onClick={() => setAuthMode('login')}
                        >
                            {t.login}
                        </span>
                        <span
                            className={`tag ${authMode === 'register' ? 'active' : ''}`}
                            onClick={() => setAuthMode('register')}
                        >
                            {t.register}
                        </span>
                    </div>
                </div>

                <div className="modal-body">
                    {error && <div className="alert alert-error">{error}</div>}

                    {authMode === "login" ? (
                        <>
                            {/* Login Form */}
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

                            <button
                                className="btn btn-primary btn-lg btn-block"
                                onClick={handleAuth}
                                disabled={loading}
                            >
                                {loading ? "Loading..." : t.signIn}
                            </button>
                        </>
                    ) : (
                        <div className="reg-flow">
                            {/* Registration Steps */}
                            {/* Simple version for now: all in one or steps? 
                                  App.jsx had steps. Let's keep steps for consistency with design.
                               */
                            }

                            {/* Progress bar */}
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
                                        className="btn btn-primary btn-lg"
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
                                            className="btn btn-primary"
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
                                            className="btn btn-primary"
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


                </div>
            </div>
        </div>
    );
}
