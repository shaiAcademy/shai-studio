
import React, { useState } from "react";
import { IconRocket, IconGlobe, IconN8n, IconPencil, IconLogin, IconDify } from "./Icons";

// Import images
import imgCenter from "../images_landing/center.png";
import imgLeft from "../images_landing/left.png";
import imgRight from "../images_landing/right.png";

// Import icons
import iconAstronaut from "../icons/astronaut.png";
import iconChair from "../icons/chair.png";
import iconMountain from "../icons/mountain.png";
import iconSmartHome from "../icons/smart_home.png";
import iconN8n from "../icons/n8n.png";
import iconDify from "../icons/dify.png";

export default function LandingPage({ t, lang, setLang, onLoginClick }) {
    const [langMenuOpen, setLangMenuOpen] = useState(false);
    const [productsMenuOpen, setProductsMenuOpen] = useState(false);
    const langLabels = { en: "EN", ru: "RU", kz: "KZ" };

    return (
        <div className="landing-container">
            {/* Header */}
            <header className="header landing-header">
                <div className="header-logo">
                    <span className="header-logo-text">shai.<span>academy</span></span>
                </div>

                <div className="header-actions">
                    {/* Language Switcher - First in order per screenshot */}
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
                        <span style={{ fontFamily: 'Manrope', fontWeight: 500, fontSize: '14px', color: '#2C2B2F' }}>{t.products}</span>
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

                    <button
                        className="btn"
                        onClick={onLoginClick}
                        style={{
                            marginLeft: '8px',
                            padding: '0 20px',
                            height: '40px',
                            borderRadius: '40px',
                            background: '#49A598',
                            color: 'white',
                            fontWeight: 600,
                            fontSize: '16px',
                            border: 'none'
                        }}
                    >
                        {t.signIn || "Войти"}
                    </button>
                </div>
            </header>

            {/* Hero Section */}
            <section className="landing-hero-wrapper">
                <div className="landing-hero-card">
                    <div className="hero-pill">
                        CREATIVE STUDIO
                    </div>

                    {/* Visuals - 3 Cards */}
                    <div className="hero-visuals-container">
                        <div className="hero-card-img card-left" style={{ border: 'none', background: 'transparent', boxShadow: 'none' }}>
                            <div className="play-icon">▶</div>
                            <img src={imgLeft} alt="Visual Left" style={{ borderRadius: '20px' }} />
                        </div>
                        <div className="hero-card-img card-center" style={{ border: 'none', background: 'transparent', boxShadow: 'none' }}>
                            <div className="play-icon">▶</div>
                            <img src={imgCenter} alt="Visual Center" style={{ borderRadius: '20px' }} />
                        </div>
                        <div className="hero-card-img card-right" style={{ border: 'none', background: 'transparent', boxShadow: 'none' }}>
                            <div className="play-icon">▶</div>
                            <img src={imgRight} alt="Visual Right" style={{ borderRadius: '20px' }} />
                        </div>
                    </div>

                    <h1 className="hero-title">{t.heroTitle}</h1>
                    <p className="hero-description">{t.heroDescription}</p>

                    <button
                        className="btn btn-accent btn-lg"
                        onClick={onLoginClick}
                        style={{ marginTop: '32px', padding: '16px 32px', borderRadius: '40px', fontSize: '18px' }}
                    >
                        {t.startGenerating} <IconPencil />
                    </button>
                </div>
            </section>

            {/* Bottom Section / Chips */}
            <section className="landing-bottom">
                <div className="landing-bottom-content">
                    <h2 className="section-title">Не знаете, с чего начать?</h2>
                    <p className="section-subtitle">Попробуйте один из вариантов ниже</p>
                </div>

                <div className="chips-container">
                    {/* User mentioned new icons under text. I'll assume they meant inside these chips/cards using custom icons if available, 
                        Method: Check if user uploaded icons. 
                        "также добавил иконки под текст который идет ниже Не знаете, с чего начать?"
                        Let's check for icons directory.
                    */}
                    {t.quickPrompts.map((prompt, idx) => {
                        let IconSrc = iconAstronaut;
                        if (idx === 1) IconSrc = iconChair;
                        if (idx === 2) IconSrc = iconSmartHome;
                        if (idx === 3) IconSrc = iconMountain;

                        return (
                            <div className="chip-card" key={idx}>
                                <div className={`chip-icon-box chip-color-${idx % 4}`}>
                                    <img src={IconSrc} alt="" style={{ width: '24px', height: '24px', mixBlendMode: 'screen' }} />
                                </div>
                                <span className="chip-text">{prompt}</span>
                            </div>
                        );
                    })}
                </div>
            </section>


            {/* Footer */}
            <footer className="footer landing-footer">
                <div className="footer-content">
                    <span>© 2025 <span style={{ color: '#80858C' }}>shai.academy</span></span>
                    <span style={{ color: '#49A598' }}>|</span>
                    <span>{t.footerText}</span>
                </div>
            </footer>
        </div>
    );
}
