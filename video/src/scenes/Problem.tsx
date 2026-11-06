import { AbsoluteFill, Img, staticFile, useCurrentFrame } from 'remotion';
import { C, Eyebrow, Grid, Headline, prog, Rise, SceneFade, FONT } from '../theme';

const Panel: React.FC<{ n: number; label: string; steps: string; bar: number; at: number; blur?: number; tone: string }> = ({ n, label, steps, bar, at, blur = 0, tone }) => {
  const frame = useCurrentFrame();
  const p = prog(frame, at, at + 30);
  return (
    <Rise at={at} style={{ width: 520 }}>
      <div style={{ position: 'relative', width: 520, height: 520, borderRadius: 16, overflow: 'hidden', border: `1px solid ${C.border}` }}>
        <Img src={staticFile('gen-frog.webp')} style={{ width: '100%', height: '100%', filter: `blur(${blur}px)`, transform: blur ? 'scale(1.06)' : undefined }} />
        <Grid n={n} color="rgba(255,255,255,0.32)" />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 18, fontFamily: FONT }}>
        <span style={{ fontSize: 30, fontWeight: 700, color: C.text }}>{label}</span>
        <span style={{ fontSize: 30, fontWeight: 700, color: tone, fontVariantNumeric: 'tabular-nums' }}>{steps}</span>
      </div>
      <div style={{ height: 14, borderRadius: 8, background: 'rgba(255,255,255,0.06)', marginTop: 10, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${bar * p * 100}%`, background: tone, borderRadius: 8 }} />
      </div>
    </Rise>
  );
};

export const Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const warn = prog(frame, 50, 70);
  return (
    <SceneFade>
      <AbsoluteFill style={{ alignItems: 'center', paddingTop: 70 }}>
        <Rise at={0}><Eyebrow>The dilemma</Eyebrow></Rise>
        <Rise at={6}><Headline size={50} style={{ marginTop: 12, textAlign: 'center' }}>Compress harder → <span style={{ color: C.blueSoft }}>shorter sequence</span> … <span style={{ color: C.pro, opacity: warn }}>but less capacity</span></Headline></Rise>
        <div style={{ display: 'flex', gap: 140, marginTop: 50, alignItems: 'flex-start' }}>
          <Panel n={64} label="16× · 1 code / token" steps="4,096 steps" bar={1} at={6} tone="#7b879e" />
          <Panel n={16} label="64× · 1 code / token" steps="256 steps" bar={256 / 4096} at={18} tone={C.blueSoft} blur={warn * 9} />
        </div>
        <div style={{ position: 'absolute', left: 1102, width: 470, textAlign: 'center', top: 560, opacity: warn, fontFamily: FONT, fontSize: 22, color: '#ffb3b5', background: 'rgba(10,6,8,0.75)', border: '1px solid rgba(240,86,91,0.5)', padding: '10px 18px', borderRadius: 12 }}>
          one code can't hold a 64 × 64 patch<br /><span style={{ color: C.dim, fontSize: 17 }}>(illustration)</span>
        </div>
      </AbsoluteFill>
    </SceneFade>
  );
};
