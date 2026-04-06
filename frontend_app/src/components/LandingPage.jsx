
import React from "react";
import { IconPencil } from "./Icons";
import AppHeader from "./AppHeader";

// Изображения лендинга
import imgCenter from "../images_landing/center.png";
import imgLeft from "../images_landing/left.png";
import imgRight from "../images_landing/right.png";

// Иконки чипов
import iconAstronaut from "../icons/astronaut.png";
import iconChair from "../icons/chair.png";
import iconMountain from "../icons/mountain.png";
import iconSmartHome from "../icons/smart_home.png";

const CHIP_ICONS = [iconAstronaut, iconChair, iconSmartHome, iconMountain];

export default function LandingPage({ t, lang, setLang, onLoginClick }) {
    return (
        <div className="landing-container">
            <AppHeader
                lang={lang}
                setLang={setLang}
                t={t}
                rightSlot={
                    <button
                        className="btn"
                        onClick={onLoginClick}
                        style={{
                            marginLeft: "8px",
                            padding: "0 20px",
                            height: "40px",
                            borderRadius: "40px",
                            background: "#49A598",
                            color: "white",
                            fontWeight: 600,
                            fontSize: "16px",
                            border: "none",
                        }}
                    >
                        {t.signIn || "Войти"}
                    </button>
                }
            />

            {/* Hero Section */}
            <section className="landing-hero-wrapper">
                <div className="landing-hero-card">
                    <div className="hero-pill">CREATIVE STUDIO</div>

                    <div className="hero-visuals-container">
                        <div
                            className="hero-card-img card-left"
                            style={{ border: "none", background: "transparent", boxShadow: "none" }}
                        >
                            <div className="play-icon">▶</div>
                            <img src={imgLeft} alt="Visual Left" style={{ borderRadius: "20px" }} />
                        </div>
                        <div
                            className="hero-card-img card-center"
                            style={{ border: "none", background: "transparent", boxShadow: "none" }}
                        >
                            <div className="play-icon">▶</div>
                            <img src={imgCenter} alt="Visual Center" style={{ borderRadius: "20px" }} />
                        </div>
                        <div
                            className="hero-card-img card-right"
                            style={{ border: "none", background: "transparent", boxShadow: "none" }}
                        >
                            <div className="play-icon">▶</div>
                            <img src={imgRight} alt="Visual Right" style={{ borderRadius: "20px" }} />
                        </div>
                    </div>

                    <h1 className="hero-title">{t.heroTitle}</h1>
                    <p className="hero-description">{t.heroDescription}</p>

                    <button
                        className="btn btn-accent btn-lg"
                        onClick={onLoginClick}
                        style={{
                            marginTop: "32px",
                            padding: "16px 32px",
                            borderRadius: "40px",
                            fontSize: "18px",
                        }}
                    >
                        {t.startGenerating} <IconPencil />
                    </button>
                </div>
            </section>

            {/* Chips Section */}
            <section className="landing-bottom">
                <div className="landing-bottom-content">
                    <h2 className="section-title">{t.chipsTitle || "Не знаете, с чего начать?"}</h2>
                    <p className="section-subtitle">{t.chipsSubtitle || "Попробуйте один из вариантов ниже"}</p>
                </div>

                <div className="chips-container">
                    {t.quickPrompts.map((chipPrompt, idx) => (
                        <div className="chip-card" key={idx}>
                            <div className={`chip-icon-box chip-color-${idx % 4}`}>
                                <img
                                    src={CHIP_ICONS[idx % CHIP_ICONS.length]}
                                    alt=""
                                    style={{ width: "24px", height: "24px", mixBlendMode: "screen" }}
                                />
                            </div>
                            <span className="chip-text">{chipPrompt}</span>
                        </div>
                    ))}
                </div>
            </section>

            <footer className="footer landing-footer">
                <div className="footer-content">
                    <span>© 2025 <span style={{ color: "#80858C" }}>shai.academy</span></span>
                    <span style={{ color: "#49A598" }}>|</span>
                    <span>{t.footerText}</span>
                </div>
            </footer>
        </div>
    );
}
