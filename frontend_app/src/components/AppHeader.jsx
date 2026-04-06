/**
 * AppHeader.jsx — общий хедер для LandingPage и Studio.
 *
 * Props:
 *   lang         — текущий язык ("en" | "ru" | "kz")
 *   setLang      — колбэк смены языка
 *   t            — объект переводов
 *   rightSlot    — JSX для правой части (кнопка "Войти" или "Выйти" + статус)
 */
import React, { useState } from "react";
import { IconGlobe } from "./Icons";
import iconN8n from "../icons/n8n.png";
import iconDify from "../icons/dify.png";

const LANG_LABELS = { en: "EN", ru: "RU", kz: "KZ" };

export default function AppHeader({ lang, setLang, t, rightSlot }) {
    const [langMenuOpen, setLangMenuOpen] = useState(false);
    const [productsMenuOpen, setProductsMenuOpen] = useState(false);

    return (
        <header className="header landing-header">
            <div className="header-logo">
                <span className="header-logo-text">
                    shai.<span>academy</span>
                </span>
            </div>

            <div className="header-actions">
                {/* Language Switcher */}
                <div
                    onClick={() => setLangMenuOpen((v) => !v)}
                    style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        padding: "0 12px",
                        height: "40px",
                        background: "#F6F6F6",
                        borderRadius: "16px",
                        cursor: "pointer",
                        gap: "8px",
                        position: "relative",
                    }}
                >
                    <IconGlobe />
                    <span
                        style={{
                            fontFamily: "Manrope",
                            fontWeight: 500,
                            fontSize: "14px",
                            color: "#2C2B2F",
                        }}
                    >
                        {LANG_LABELS[lang]}
                    </span>

                    {langMenuOpen && (
                        <div className="lang-menu" style={{ top: "48px" }}>
                            {Object.keys(LANG_LABELS).map((l) => (
                                <button
                                    key={l}
                                    className={`lang-menu-item ${lang === l ? "active" : ""}`}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setLang(l);
                                        setLangMenuOpen(false);
                                    }}
                                >
                                    {LANG_LABELS[l]}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                {/* Products Dropdown */}
                <div
                    onClick={() => setProductsMenuOpen((v) => !v)}
                    style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        padding: "0 12px",
                        height: "40px",
                        backgroundColor: "#F6F6F6",
                        border: "1px solid #49A598",
                        borderRadius: "16px",
                        cursor: "pointer",
                        gap: "8px",
                        position: "relative",
                    }}
                >
                    <span
                        style={{
                            fontFamily: "Manrope",
                            fontWeight: 500,
                            fontSize: "14px",
                            color: "#2C2B2F",
                        }}
                    >
                        {t.products}
                    </span>
                    <span style={{ fontSize: "10px" }}>{productsMenuOpen ? "▲" : "▼"}</span>

                    {productsMenuOpen && (
                        <div
                            style={{
                                position: "absolute",
                                top: "48px",
                                left: "0",
                                width: "235px",
                                background: "white",
                                borderRadius: "16px",
                                boxShadow: "0px 4px 20px rgba(0,0,0,0.1)",
                                padding: "8px",
                                display: "flex",
                                flexDirection: "column",
                                gap: "4px",
                                zIndex: 100,
                            }}
                        >
                            <ProductLink
                                href="https://n8n.shai.academy"
                                icon={iconN8n}
                                label={t.loginN8n || "Войти в n8n"}
                                alt="n8n"
                            />
                            <div style={{ height: "1px", background: "#F6F6F6", margin: "0 12px" }} />
                            <ProductLink
                                href="https://dify.shai.academy"
                                icon={iconDify}
                                label={t.loginDify || "Войти в Dify"}
                                alt="Dify"
                            />
                        </div>
                    )}
                </div>

                {/* Правый слот — передаётся снаружи (кнопка входа / кнопка выхода) */}
                {rightSlot}
            </div>
        </header>
    );
}

/** Вспомогательный компонент для элемента продуктового дропдауна. */
function ProductLink({ href, icon, label, alt }) {
    return (
        <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "12px",
                textDecoration: "none",
                color: "#2C2B2F",
                borderRadius: "12px",
                transition: "background 0.2s",
                fontSize: "14px",
                fontWeight: 500,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#F6F6F6")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
            <div
                style={{
                    width: "32px",
                    height: "32px",
                    borderRadius: "8px",
                    border: "1px solid #EFEFEF",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    overflow: "hidden",
                }}
            >
                <img src={icon} alt={alt} style={{ width: "24px", height: "24px", objectFit: "contain" }} />
            </div>
            {label}
        </a>
    );
}
