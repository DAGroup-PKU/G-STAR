import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { C, Card, Eyebrow, Headline, prog, Rise, SceneFade, FONT } from '../theme';

const SPEED = [
  { ours: 'Flash-1.7B', vs: 'Janus-Pro-7B', x: 21.57, c: C.flash },
  { ours: 'Flash-1.7B', vs: 'BitDance-14B', x: 21.04, c: C.flash },
  { ours: 'Pro-8B', vs: 'Emu3.5-34B', x: 8.36, c: C.pro },
];
// Qwen-Image-Bench overall, paper Table 1 (top entries)
const QIB = [
  { m: 'GPT Image 1', v: 54.07, c: '#5d6678', tag: 'proprietary' },
  { m: 'G-STAR-Pro-8B', v: 50.07, c: C.pro, ours: true },
  { m: 'Emu3.5', v: 49.89, c: '#7f9cc9' },
  { m: 'Qwen-Image', v: 49.23, c: '#7f9cc9' },
  { m: 'Z-Image', v: 49.1, c: '#7f9cc9' },
  { m: 'GLM-Image', v: 48.19, c: '#7f9cc9' },
  { m: 'G-STAR-Flash-1.7B', v: 46.88, c: C.flash, ours: true },
  { m: 'BitDance', v: 45.19, c: '#7f9cc9' },
];
const MIN = 40, MAXV = 55, BW = 520;

export const Compare: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SceneFade inF={8}>
      <AbsoluteFill style={{ padding: '0 120px 150px', justifyContent: 'center' }}>
        <Rise at={0}><Eyebrow>Speed and quality at 1024²</Eyebrow></Rise>
        <Rise at={6}><Headline size={50} style={{ marginTop: 12 }}>Up to <span style={{ color: C.flash }}>21×</span> faster, without giving up quality</Headline></Rise>
        <div style={{ display: 'flex', gap: 40, marginTop: 44 }}>
          <Card style={{ flex: '0 0 640px', padding: '34px 36px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontFamily: FONT, fontSize: 22, color: C.muted, marginBottom: 28 }}>Latency speedup vs. prior discrete AR</div>
            {SPEED.map((s, i) => {
              const p = prog(frame, 8 + i * 8, 34 + i * 8);
              return (
                <div key={s.vs} style={{ marginBottom: 40, fontFamily: FONT }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 23, color: '#cfd6e4' }}>
                    <span><b style={{ color: s.c }}>{s.ours}</b> vs. {s.vs}</span>
                    <b style={{ color: s.c, fontSize: 34, fontVariantNumeric: 'tabular-nums' }}>{(s.x * p).toFixed(2)}×</b>
                  </div>
                  <div style={{ height: 22, background: 'rgba(255,255,255,0.05)', borderRadius: 8, marginTop: 8 }}>
                    <div style={{ height: '100%', width: `${(s.x / 22) * 100 * p}%`, background: s.c, borderRadius: 8 }} />
                  </div>
                </div>
              );
            })}
            <div style={{ fontFamily: FONT, fontSize: 17, color: C.dim }}>Paper Figure 1(d): quality vs. latency, 1024².</div>
          </Card>
          <Card style={{ flex: 1, padding: '28px 32px' }}>
            <div style={{ fontFamily: FONT, fontSize: 22, color: C.muted, marginBottom: 16 }}>Qwen-Image-Bench overall ↑</div>
            {QIB.map((q, i) => {
              const p = prog(frame, 30 + i * 5, 55 + i * 5);
              const glow = q.ours ? prog(frame, 95, 115) : 0;
              return (
                <div key={q.m} style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 17, fontFamily: FONT, opacity: 0.2 + 0.8 * p }}>
                  <div style={{ width: 240, fontSize: 22, color: q.ours ? q.c : '#cfd6e4', fontWeight: q.ours ? 700 : 400 }}>{q.m}{q.tag && <span style={{ fontSize: 15, color: C.dim }}> · {q.tag}</span>}</div>
                  <div style={{ width: BW, height: 34 }}>
                    <div style={{ height: '100%', width: ((q.v - MIN) / (MAXV - MIN)) * BW * p, background: q.c, borderRadius: 6, boxShadow: glow ? `0 0 ${22 * glow}px ${q.c}` : 'none' }} />
                  </div>
                  <div style={{ fontSize: 23, fontWeight: q.ours ? 800 : 500, color: q.ours ? q.c : C.text, fontVariantNumeric: 'tabular-nums' }}>{q.v.toFixed(2)}</div>
                </div>
              );
            })}
            <div style={{ fontFamily: FONT, fontSize: 17, color: C.dim, marginTop: 6 }}>Paper Table 1. Axis starts at 40.</div>
          </Card>
        </div>
      </AbsoluteFill>
    </SceneFade>
  );
};
