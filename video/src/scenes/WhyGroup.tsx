import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from 'remotion';
import { C, Card, Eyebrow, Headline, prog, Rise, SceneFade, FONT } from '../theme';

// Four 2×2 sub-tokens, one colour each (as in the "Why group quantization" slide).
const SUB = [C.yellow, C.green, C.codeBlue, C.flash];
const CH = 6, CW = 24, CG = 2; // channels per sub-token vector, cell size, gap
const BAR = CH * (CW + CG);
const SQ = 34; // code square
const lerp = (p: number, a: number, b: number) => a + (b - a) * p;
const stack = (k: number) => ({ x: 10, y: 4 + k * 40 });

const Vec: React.FC<{ color: string; x: number; y: number; o?: number }> = ({ color, x, y, o = 1 }) => (
  <div style={{ position: 'absolute', left: x, top: y, display: 'flex', gap: CG, opacity: o }}>
    {Array.from({ length: CH }, (_, i) => <div key={i} style={{ width: CW, height: 30, borderRadius: 3, background: color, opacity: 0.55 + 0.45 * ((i * 37) % 7) / 6 }} />)}
  </div>
);

const Code: React.FC<{ color: string; x: number; y: number; s?: number; o?: number }> = ({ color, x, y, s = 1, o = 1 }) => (
  <div style={{ position: 'absolute', left: x, top: y, width: SQ, height: SQ, borderRadius: 5, background: color, opacity: o, transform: `scale(${s})`, boxShadow: '0 0 10px rgba(0,0,0,0.4)' }} />
);

const Row: React.FC<{ title: string; step: string; tone: string; at: number; dim?: number; children: React.ReactNode }> = ({ title, step, tone, at, dim = 0, children }) => (
  <Rise at={at}>
    <Card style={{ height: 268, padding: '22px 30px', position: 'relative', opacity: 1 - 0.5 * dim }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: FONT }}>
        <span style={{ fontSize: 28, fontWeight: 700, color: tone }}>{title}</span>
        <span style={{ fontSize: 21, color: C.muted }}>{step}</span>
      </div>
      <div style={{ position: 'relative', height: 172, marginTop: 14 }}>{children}</div>
    </Card>
  </Rise>
);

export const WhyGroup: React.FC = () => {
  const frame = useCurrentFrame();
  // Row A: merge along channel, then quantize the whole vector into one code
  const aMerge = prog(frame, 36, 62);
  const aQuant = prog(frame, 70, 90);
  const aDim = prog(frame, 100, 115);
  // Row B: quantize each sub-token, merge the codes, then keep splitting into more groups
  const bQuant = prog(frame, 110, 128);
  const bMerge = prog(frame, 128, 146);
  const n = frame < 152 ? 4 : frame < 166 ? 8 : 16;
  const groupX = 420;

  const aStep = aQuant > 0.5 ? 'quantize as a whole' : aMerge > 0.1 ? 'merge along channel' : '4 sub-tokens';
  const bStep = frame >= 150 ? 'keep going: more groups' : bMerge > 0.1 ? 'merge along channel' : bQuant > 0.1 ? 'quantize each' : '4 sub-tokens';

  return (
    <SceneFade outF={8}>
      <AbsoluteFill style={{ padding: '70px 120px 0' }}>
        <Rise at={0}><Eyebrow>Why group quantization?</Eyebrow></Rise>
        <Rise at={6}><Headline size={50} style={{ marginTop: 12 }}>Quantize, then merge: <span style={{ color: C.blueSoft }}>a group of codes per position</span></Headline></Rise>

        <div style={{ display: 'flex', gap: 56, marginTop: 40 }}>
          <Rise at={10} style={{ width: 320, flexShrink: 0 }}>
            <div style={{ position: 'relative', width: 320, height: 320, borderRadius: 16, overflow: 'hidden', border: `1px solid ${C.border}` }}>
              <Img src={staticFile('gen-panda.webp')} style={{ width: '100%', height: '100%', objectFit: 'cover', filter: 'brightness(0.85)' }} />
              {SUB.map((col, k) => (
                <div key={k} style={{ position: 'absolute', left: (k % 2) * 160, top: Math.floor(k / 2) * 160, width: 160, height: 160, background: col, opacity: 0.32, border: '2px solid rgba(255,255,255,0.7)', boxSizing: 'border-box' }} />
              ))}
            </div>
            <div style={{ fontFamily: FONT, fontSize: 22, color: C.muted, marginTop: 18, lineHeight: 1.4 }}>
              2 × 2 sub-tokens, merged along the channel, as in Qwen-ViT / GLM-ViT
            </div>
          </Rise>

          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 22 }}>
            <Row title="Merge → Quantize" step={aStep} tone={C.text} at={20} dim={aDim}>
              {SUB.map((col, k) => {
                const s0 = stack(k);
                return <Vec key={k} color={col} x={lerp(aMerge, s0.x, 200 + k * (BAR + 4))} y={lerp(aMerge, s0.y, 66)} o={1 - 0.7 * aQuant} />;
              })}
              <Code color="#8a93a6" x={880} y={64} s={aQuant} o={aQuant} />
              <div style={{ position: 'absolute', left: 930, top: 66, opacity: aQuant, fontFamily: FONT, fontSize: 22, color: '#ffb3b5' }}>1 code for all 4</div>
            </Row>

            <Row title="Quantize → Merge" step={bStep} tone={C.green} at={96}>
              {bMerge < 1 && SUB.map((col, k) => {
                const s0 = stack(k);
                return (
                  <div key={k}>
                    <Vec color={col} x={s0.x} y={s0.y} o={1 - bQuant} />
                    <Code color={col} x={lerp(bMerge, s0.x + BAR + 40, groupX + k * (SQ + 4))} y={lerp(bMerge, s0.y - 2, 62)} s={bQuant} o={bQuant} />
                  </div>
                );
              })}
              {bMerge >= 1 && Array.from({ length: n }, (_, i) => (
                <Code key={`${n}-${i}`} color={SUB[Math.floor(i / (n / 4))]} x={groupX + i * (SQ + 4)} y={62}
                  s={i < n / 2 || n === 4 ? 1 : prog(frame, (n === 8 ? 152 : 166) + i * 0.5, (n === 8 ? 158 : 172) + i * 0.5)} />
              ))}
              <div style={{ position: 'absolute', left: groupX, top: 116, opacity: bMerge, fontFamily: FONT, fontSize: 24, color: C.text }}>
                G = <b style={{ color: C.green, fontVariantNumeric: 'tabular-nums' }}>{n}</b> codes per position
                <span style={{ color: C.dim, fontSize: 20 }}>{interpolate(frame, [150, 152], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }) > 0.5 ? '  ·  4 → 8 → 16 groups' : ''}</span>
              </div>
            </Row>
          </div>
        </div>
      </AbsoluteFill>
    </SceneFade>
  );
};
