import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from 'remotion';
import { C, Card, Eyebrow, fmt, Grid, Headline, prog, Rise, SceneFade, FONT, MONO } from '../theme';

const IMG = 560;
const GROUP_COLORS = [C.yellow, C.green, C.codeBlue, C.yellow]; // as in paper Fig. 2
const SQ = 21, GAP = 4; // code square size

const Stat: React.FC<{ label: string; value: string; sub?: string; tone?: string; badge?: string; badgeOn?: number }> = ({ label, value, sub, tone = C.blueSoft, badge, badgeOn = 0 }) => (
  <Card style={{ padding: '20px 24px', flex: 1, position: 'relative' }}>
    <div style={{ fontFamily: FONT, fontSize: 19, color: C.muted }}>{label}</div>
    <div style={{ fontFamily: FONT, fontSize: 50, fontWeight: 800, color: tone, fontVariantNumeric: 'tabular-nums', marginTop: 4 }}>{value}</div>
    {sub && <div style={{ fontFamily: MONO, fontSize: 16, color: C.dim }}>{sub}</div>}
    {badge && <div style={{ position: 'absolute', top: 16, right: 16, opacity: badgeOn, fontFamily: FONT, fontSize: 15, fontWeight: 700, color: '#0b1a10', background: C.green, padding: '3px 10px', borderRadius: 999 }}>{badge}</div>}
  </Card>
);

export const CoScaling: React.FC = () => {
  const frame = useCurrentFrame();
  const showOthers = prog(frame, 95, 115);
  const pack = prog(frame, 125, 200);
  const morph = prog(frame, 215, 265);
  const s = interpolate(morph, [0, 1], [32, 64]);
  const n = 1024 / s;
  const steps = (1024 / s) ** 2;
  const G = 8 * (s / 32) ** 2;

  // the highlighted cell / 2×2 block on the image
  const cell32 = IMG / 32;
  const bx = 18, by = 11;
  const blockSize = interpolate(prog(frame, 90, 115), [0, 1], [cell32, cell32 * 2]);

  return (
    <SceneFade>
      <AbsoluteFill style={{ padding: '70px 120px 0' }}>
        <Rise at={0}><Eyebrow>How it works · spatial–group co-scaling</Eyebrow></Rise>
        <Rise at={6}><Headline size={50} style={{ marginTop: 12 }}>(s, G) → (αs, α²G): <span style={{ color: C.blueSoft }}>fewer steps, same bits</span></Headline></Rise>

        <div style={{ display: 'flex', gap: 70, marginTop: 40 }}>
          {/* image + grid */}
          <Rise at={10} style={{ position: 'relative', width: IMG, height: IMG, flexShrink: 0 }}>
            <div style={{ position: 'absolute', inset: 0, borderRadius: 16, overflow: 'hidden', border: `1px solid ${C.border}` }}>
              <Img src={staticFile('gen-panda.webp')} style={{ width: '100%', height: '100%', filter: 'brightness(0.8)' }} />
              <div style={{ opacity: 1 - morph }}><Grid n={32} color="rgba(255,255,255,0.28)" /></div>
              <div style={{ opacity: morph }}><Grid n={16} color="rgba(255,255,255,0.45)" width={1.5} /></div>
            </div>
            <div style={{ position: 'absolute', left: bx * cell32, top: by * cell32, width: blockSize, height: blockSize, border: `3px solid ${C.blueSoft}`, boxShadow: `0 0 18px ${C.blue}`, borderRadius: 3 }} />
            <div style={{ position: 'absolute', left: 0, right: 0, bottom: -46, textAlign: 'center', fontFamily: FONT, fontSize: 24, color: C.muted }}>
              {morph < 0.5 ? '32× tokenizer · 32 × 32 grid' : 'effective 64× · 16 × 16 grid'}
            </div>
          </Rise>

          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {/* code groups */}
            <Card style={{ height: 250, position: 'relative', padding: 24 }}>
              <div style={{ fontFamily: FONT, fontSize: 20, color: C.muted }}>
                {pack < 0.5 ? 'Group-wise quantization: G codes per position' : 'Lossless 2×2 packing: one global step, 4G codes'}
              </div>
              {GROUP_COLORS.map((col, k) => {
                const start = { x: (k % 2) * 250, y: 52 + Math.floor(k / 2) * 50 };
                const end = { x: k * 8 * (SQ + GAP), y: 140 };
                const x = interpolate(pack, [0, 1], [start.x, end.x]);
                const y = interpolate(pack, [0, 1], [start.y, end.y]);
                const o = k === 0 ? prog(frame, 20, 35) : showOthers;
                return (
                  <div key={k} style={{ position: 'absolute', left: 24 + x, top: 24 + y, display: 'flex', gap: GAP, opacity: o }}>
                    {Array.from({ length: 8 }, (_, i) => (
                      <div key={i} style={{ width: SQ, height: SQ, borderRadius: 3, background: col, transform: `scale(${prog(frame, 20 + i * 2 + (k ? 80 : 0), 30 + i * 2 + (k ? 80 : 0))})` }} />
                    ))}
                  </div>
                );
              })}
              <div style={{ position: 'absolute', left: 24, top: 200, fontFamily: MONO, fontSize: 18, color: C.dim, opacity: pack }}>
                4 × G = {pack > 0.98 ? 32 : 8} codes → nothing discarded, nothing re-quantized
              </div>
            </Card>

            <div style={{ display: 'flex', gap: 18, marginTop: 22 }}>
              <Stat label="Global AR steps" value={fmt(Math.round(steps))} sub="N = HW / s²" />
              <Stat label="Codes per step" value={String(Math.round(G))} sub="G" />
            </div>
            <div style={{ display: 'flex', gap: 18, marginTop: 18 }}>
              <Stat label="Total codes" value="8,192" sub="N × G" tone={C.text} badge="unchanged" badgeOn={morph} />
              <Stat label="Nominal budget" value="114,688" sub="bits · K = 16,384" tone={C.text} badge="unchanged" badgeOn={morph} />
            </div>
          </div>
        </div>
      </AbsoluteFill>
    </SceneFade>
  );
};
