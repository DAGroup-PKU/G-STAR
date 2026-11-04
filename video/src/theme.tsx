import { AbsoluteFill, continueRender, delayRender, interpolate, Easing, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import React, { useState } from 'react';

export const C = {
  bg: '#05070d',
  text: '#eef1f7',
  muted: '#9aa6bd',
  dim: '#6d7a94',
  blue: '#5b9dff',
  blueSoft: '#8cc4ff',
  pro: '#f0565b',
  flash: '#ffa02e',
  green: '#7fcf8f',
  yellow: '#f4d27a',
  codeBlue: '#5b9bd5',
  card: 'rgba(255,255,255,0.045)',
  border: 'rgba(255,255,255,0.13)',
};
export const FONT = 'Roboto, -apple-system, "Segoe UI", sans-serif';
export const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';

// Load local fonts once before the first frame renders.
let fontsLoaded = false;
export const useFonts = () => {
  const [handle] = useState(() => (fontsLoaded ? null : delayRender('fonts')));
  React.useEffect(() => {
    if (handle === null) return;
    const faces = [
      new FontFace('Roboto', `url(${staticFile('fonts/roboto.woff2')}) format('woff2')`, { weight: '300 800' }),
      new FontFace('Caveat', `url(${staticFile('fonts/Caveat-Medium.ttf')}) format('truetype')`, { weight: '500' }),
    ];
    Promise.all(faces.map(f => f.load().then(l => document.fonts.add(l))))
      .then(() => { fontsLoaded = true; continueRender(handle); })
      .catch(() => continueRender(handle));
  }, [handle]);
};

export const ease = Easing.bezier(0.2, 0.8, 0.2, 1);
/** 0→1 progress between two frames, eased. */
export const prog = (frame: number, from: number, to: number, e = ease) =>
  interpolate(frame, [from, to], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: e });

export const Background: React.FC = () => (
  <AbsoluteFill style={{ background: `radial-gradient(1200px 800px at 75% 15%, #0e1a36 0%, transparent 60%), radial-gradient(900px 700px at 10% 90%, #0b1430 0%, transparent 55%), ${C.bg}` }} />
);

/** Fades a whole scene in and out. */
export const SceneFade: React.FC<{ children: React.ReactNode; inF?: number; outF?: number }> = ({ children, inF = 12, outF = 12 }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const o = Math.min(prog(frame, 0, inF), 1 - prog(frame, durationInFrames - outF, durationInFrames));
  return <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>;
};

export const Eyebrow: React.FC<{ children: React.ReactNode; color?: string }> = ({ children, color = C.blueSoft }) => (
  <div style={{ fontFamily: FONT, fontSize: 18, letterSpacing: '0.16em', fontWeight: 700, color, textTransform: 'uppercase' }}>{children}</div>
);

export const Headline: React.FC<{ children: React.ReactNode; size?: number; style?: React.CSSProperties }> = ({ children, size = 54, style }) => (
  <div style={{ fontFamily: FONT, fontSize: size, fontWeight: 700, color: C.text, lineHeight: 1.15, letterSpacing: '-0.01em', ...style }}>{children}</div>
);

/** Fade + rise for an element starting at `at`. */
export const Rise: React.FC<{ at: number; children: React.ReactNode; dist?: number; style?: React.CSSProperties; dur?: number }> = ({ at, children, dist = 24, style, dur = 18 }) => {
  const frame = useCurrentFrame();
  const p = prog(frame, at, at + dur);
  return <div style={{ opacity: p, transform: `translateY(${(1 - p) * dist}px)`, ...style }}>{children}</div>;
};

export const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 22, ...style }}>{children}</div>
);

export const fmt = (n: number, d = 0) => n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

/** A thin grid overlay with `n` cells per side. */
export const Grid: React.FC<{ n: number; color?: string; width?: number }> = ({ n, color = 'rgba(255,255,255,0.35)', width = 1 }) => {
  const step = 100 / n;
  return (
    <div style={{
      position: 'absolute', inset: 0, pointerEvents: 'none',
      backgroundImage: `linear-gradient(to right, ${color} ${width}px, transparent ${width}px), linear-gradient(to bottom, ${color} ${width}px, transparent ${width}px)`,
      backgroundSize: `${step}% ${step}%`,
    }} />
  );
};
