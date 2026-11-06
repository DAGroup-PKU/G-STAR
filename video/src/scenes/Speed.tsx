import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { C, Card, Eyebrow, Headline, prog, Rise, SceneFade, FONT } from '../theme';

const STAGES = [
  { name: 'Causal-head baseline', t: 22.23, x: '' },
  { name: 'Semi-AR head', t: 14.93, x: '1.49×' },
  { name: '+ CUDA Graph', t: 5.85, x: '2.55×' },
  { name: '+ Fused Triton kernels', t: 3.37, x: '1.74×' },
  { name: '+ SGLang serving', t: 2.5, x: '1.35×' },
];
const W = 680;

export const Speed: React.FC = () => {
  const frame = useCurrentFrame();
  const big = prog(frame, 60, 80);
  const counter = interpolate(prog(frame, 60, 95), [0, 1], [22.23, 2.5]);
  return (
    <SceneFade outF={8}>
      <AbsoluteFill style={{ padding: '0 120px 150px', justifyContent: 'center' }}>
        <Rise at={0}><Eyebrow>Efficiency · GSTAR-Flash-1.7B · 1024² · 1 × H200</Eyebrow></Rise>
        <Rise at={6}><Headline size={50} style={{ marginTop: 12 }}>From 22.23 s to <span style={{ color: C.blueSoft }}>2.50 s</span> per image</Headline></Rise>
        <div style={{ display: 'flex', gap: 60, marginTop: 56, alignItems: 'center' }}>
          <Card style={{ padding: '40px 40px', display: 'flex', flexDirection: 'column', gap: 34 }}>
            {STAGES.map((s, i) => {
              const p = prog(frame, 8 + i * 10, 24 + i * 10);
              const last = i === STAGES.length - 1;
              return (
                <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 22, opacity: 0.25 + 0.75 * p, fontFamily: FONT }}>
                  <div style={{ width: 290, fontSize: 28, color: last ? C.blueSoft : '#cfd6e4', fontWeight: last ? 700 : 400 }}>{s.name}</div>
                  <div style={{ width: W, height: 44, background: 'rgba(255,255,255,0.04)', borderRadius: 8 }}>
                    <div style={{ width: (s.t / 22.23) * W * p, height: '100%', borderRadius: 8, background: last ? `linear-gradient(90deg, #2f6df6, ${C.blueSoft})` : '#4a5566', boxShadow: last ? '0 0 18px rgba(47,109,246,0.6)' : 'none' }} />
                  </div>
                  <div style={{ width: 110, textAlign: 'right', fontSize: 30, fontWeight: last ? 800 : 500, color: last ? C.blueSoft : C.text, fontVariantNumeric: 'tabular-nums' }}>{s.t.toFixed(2)} s</div>
                  <div style={{ width: 70, fontSize: 21, color: C.green }}>{s.x}</div>
                </div>
              );
            })}
          </Card>
          <div style={{ opacity: big, transform: `scale(${0.9 + 0.1 * big})`, textAlign: 'center', fontFamily: FONT }}>
            <div style={{ fontSize: 170, fontWeight: 800, color: C.blueSoft, lineHeight: 1, fontVariantNumeric: 'tabular-nums' }}>{counter.toFixed(1)}<span style={{ fontSize: 80 }}>s</span></div>
            <div style={{ fontSize: 28, color: C.muted, marginTop: 10 }}>per 1024 × 1024 image</div>
            <div style={{ fontSize: 30, color: C.green, fontWeight: 700, marginTop: 14 }}>8.9× end-to-end</div>
          </div>
        </div>
      </AbsoluteFill>
    </SceneFade>
  );
};
