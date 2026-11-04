import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from 'remotion';
import { C, Eyebrow, fmt, Grid, Headline, prog, Rise, SceneFade, FONT } from '../theme';

const N = 64; // 16× compression on a 1024² image → 64 × 64 = 4,096 tokens
const SIZE = 640;

export const Hook: React.FC = () => {
  const frame = useCurrentFrame();
  // reveal the image token by token (raster order), accelerating
  const done = Math.floor(interpolate(frame, [10, 190], [0, N * N], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: t => t * t }));
  const rows = Math.floor(done / N), col = done % N, cell = SIZE / N;
  const zoom = interpolate(frame, [0, 213], [1.04, 1]);
  const slow = prog(frame, 105, 130);

  return (
    <SceneFade inF={8}>
      <AbsoluteFill style={{ flexDirection: 'row', alignItems: 'center', padding: '0 150px', gap: 110, paddingBottom: 90 }}>
        <div style={{ position: 'relative', width: SIZE, height: SIZE, borderRadius: 18, overflow: 'hidden', border: `1px solid ${C.border}`, boxShadow: '0 30px 90px rgba(0,0,0,0.6)', transform: `scale(${zoom})` }}>
          <Img src={staticFile('gen-lumiere.webp')} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', filter: 'brightness(0.18) blur(3px)' }} />
          <div style={{ position: 'absolute', inset: 0, clipPath: `inset(0 0 ${SIZE - rows * cell}px 0)` }}>
            <Img src={staticFile('gen-lumiere.webp')} style={{ width: '100%', height: '100%' }} />
          </div>
          <div style={{ position: 'absolute', inset: 0, clipPath: `inset(${rows * cell}px ${SIZE - col * cell}px ${SIZE - (rows + 1) * cell}px 0)` }}>
            <Img src={staticFile('gen-lumiere.webp')} style={{ width: '100%', height: '100%' }} />
          </div>
          <Grid n={N} color="rgba(255,255,255,0.12)" />
          {done < N * N && (
            <div style={{ position: 'absolute', left: col * cell - 1, top: rows * cell - 1, width: cell + 2, height: cell + 2, border: `2px solid ${C.blueSoft}`, boxShadow: `0 0 14px ${C.blue}` }} />
          )}
        </div>

        <div style={{ flex: 1 }}>
          <Rise at={4}><Eyebrow>Discrete autoregressive generation</Eyebrow></Rise>
          <Rise at={10}><Headline size={70} style={{ marginTop: 18 }}>One token<br />at a time.</Headline></Rise>
          <Rise at={30} style={{ marginTop: 46 }}>
            <div style={{ fontFamily: FONT, fontSize: 22, color: C.muted }}>sequential steps · 1024² · 16× tokenizer</div>
            <div style={{ fontFamily: FONT, fontSize: 92, fontWeight: 800, color: C.blueSoft, fontVariantNumeric: 'tabular-nums', marginTop: 6 }}>
              {fmt(done)}<span style={{ fontSize: 44, color: C.dim, fontWeight: 500 }}> / 4,096</span>
            </div>
          </Rise>
          <div style={{ opacity: slow, transform: `translateY(${(1 - slow) * 16}px)`, marginTop: 28, display: 'inline-flex', alignItems: 'center', gap: 12, padding: '12px 22px', borderRadius: 999, background: 'rgba(240,86,91,0.12)', border: '1px solid rgba(240,86,91,0.45)', color: '#ffb3b5', fontFamily: FONT, fontSize: 26 }}>
            Long sequence → slow at high resolution
          </div>
        </div>
      </AbsoluteFill>
    </SceneFade>
  );
};
