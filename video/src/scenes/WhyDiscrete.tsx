import { AbsoluteFill, random, useCurrentFrame } from 'remotion';
import { C, Card, Eyebrow, Headline, prog, Rise, SceneFade, FONT } from '../theme';

const N = 16, SQ = 30, GAP = 6; // tokens per strip

// One row of image tokens; `noisy` jitters the colours.
const Strip: React.FC<{ label: string; color: string; at: number; noisy?: boolean }> = ({ label, color, at, noisy }) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 20, marginTop: 18 }}>
      <div style={{ width: 170, fontFamily: FONT, fontSize: 22, color: '#cfd6e4' }}>{label}</div>
      <div style={{ display: 'flex', gap: GAP }}>
        {Array.from({ length: N }, (_, i) => {
          const p = prog(frame, at + i, at + i + 10);
          const jitter = noisy ? 0.35 + 0.65 * random(`n${i}-${Math.floor(frame / 3)}`) : 1;
          return (
            <div key={i} style={{
              width: SQ, height: SQ, borderRadius: 5, transform: `scale(${p})`,
              background: color, opacity: jitter,
            }} />
          );
        })}
      </div>
    </div>
  );
};

// Loss coverage bar: share of tokens that receive a prediction loss.
const Coverage: React.FC<{ share: number; at: number; tone: string; note: string }> = ({ share, at, tone, note }) => {
  const frame = useCurrentFrame();
  const p = prog(frame, at, at + 30);
  return (
    <div style={{ marginTop: 34, fontFamily: FONT }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 21, color: C.muted }}>
        <span>{note}</span>
        <b style={{ color: tone, fontSize: 30, fontVariantNumeric: 'tabular-nums' }}>{Math.round(share * p * 100)}%</b>
      </div>
      <div style={{ height: 16, borderRadius: 8, background: 'rgba(255,255,255,0.06)', marginTop: 10, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${share * p * 100}%`, background: tone, borderRadius: 8 }} />
      </div>
    </div>
  );
};

export const WhyDiscrete: React.FC = () => {
  const frame = useCurrentFrame();
  const right = prog(frame, 112, 132);
  return (
    <SceneFade inF={8}>
      <AbsoluteFill style={{ padding: '120px 120px 0' }}>
        <Rise at={0}><Eyebrow>Why discrete?</Eyebrow></Rise>
        <Rise at={6}><Headline size={50} style={{ marginTop: 12 }}>One copy of the image. <span style={{ color: C.green, opacity: right }}>Every token supervised.</span></Headline></Rise>
        <div style={{ display: 'flex', gap: 40, marginTop: 46 }}>
          <Rise at={10} style={{ flex: 1 }}>
            <Card style={{ padding: '30px 34px', height: 390 }}>
              <div style={{ fontFamily: FONT, fontSize: 28, fontWeight: 700, color: C.text }}>Diffusion UMM <span style={{ fontSize: 20, color: C.dim, fontWeight: 400 }}>e.g. BAGEL</span></div>
              <div style={{ fontFamily: FONT, fontSize: 20, color: C.muted, marginTop: 6 }}>3 copies of the image · conditions on noisy latents</div>
              <Strip label="Clean VAE" color="#7f9cc9" at={16} />
              <Strip label="Clean SigLIP" color="#a58fd6" at={24} />
              <Strip label="Noisy VAE" color="#7f9cc9" at={32} noisy />
              <Coverage share={0.5} at={60} tone={C.flash} note="Tokens in the loss · masked diffusion (LLaDA)" />
            </Card>
          </Rise>
          <div style={{ flex: 1, opacity: right, transform: `translateY(${(1 - right) * 24}px)` }}>
            <Card style={{ padding: '30px 34px', height: 390, borderColor: `rgba(127,207,143,${0.2 + 0.4 * right})` }}>
              <div style={{ fontFamily: FONT, fontSize: 28, fontWeight: 700, color: C.green }}>Discrete UMM <span style={{ fontSize: 20, color: C.dim, fontWeight: 400 }}>next-token prediction</span></div>
              <div style={{ fontFamily: FONT, fontSize: 20, color: C.muted, marginTop: 6 }}>1 copy of the image · teacher forcing</div>
              <Strip label="Discrete codes" color={C.yellow} at={120} />
              <div style={{ height: 2 * (SQ + 18) }} />
              <Coverage share={1} at={140} tone={C.green} note="Tokens in the loss · teacher forcing" />
            </Card>
          </div>
        </div>
        <Rise at={70} style={{ marginTop: 16, fontFamily: FONT, fontSize: 17, color: C.dim }}>
          Schematic. Loss coverage: masked diffusion supervises about half of the positions per step on average; next-token prediction supervises all of them.
        </Rise>
      </AbsoluteFill>
    </SceneFade>
  );
};
