import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from 'remotion';
import { C, FONT, prog, Rise, SceneFade } from '../theme';

const TILES = ['gen-lumiere', 'gen-taohuayuan', 'gen-egg-tart', 'gen-cinema', 'gen-panda', 'gen-art-deco', 'gen-frog', 'gen-window', 'gen-dog', 'gen-taohuayuan', 'gen-lumiere', 'gen-cinema'];

export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const drift = interpolate(frame, [0, 222], [0, -120]);
  const dim = prog(frame, 40, 80);
  return (
    <SceneFade outF={20}>
      <AbsoluteFill style={{ overflow: 'hidden' }}>
        <div style={{ position: 'absolute', left: -60, top: -40, display: 'grid', gridTemplateColumns: 'repeat(6, 340px)', gap: 12, transform: `translateX(${drift}px) rotate(-4deg)` }}>
          {TILES.map((t, i) => (
            <Img key={i} src={staticFile(`${t}.webp`)} style={{ width: 340, height: 340, borderRadius: 14, objectFit: 'cover', opacity: prog(frame, i * 2, 14 + i * 2) }} />
          ))}
          {TILES.slice(0, 6).map((t, i) => (
            <Img key={`r${i}`} src={staticFile(`${TILES[(i + 3) % TILES.length]}.webp`)} style={{ width: 340, height: 340, borderRadius: 14, objectFit: 'cover', opacity: prog(frame, 12 + i * 2, 26 + i * 2) }} />
          ))}
        </div>
        <AbsoluteFill style={{ background: `rgba(5,7,13,${0.15 + dim * 0.7})` }} />
        <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', paddingBottom: 110, opacity: dim }}>
          <div style={{ fontFamily: 'Caveat, cursive', fontSize: 200, lineHeight: 1, padding: '0 30px', background: `linear-gradient(90deg, ${C.pro}, #ffffff 45%, #6692f1)`, WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent' }}>GSTAR</div>
          <div style={{ fontFamily: FONT, fontSize: 40, color: '#e2e7f0', marginTop: 4 }}>Short sequences. Rich tokens. High-quality images.</div>
          <Rise at={70} style={{ display: 'flex', gap: 22, marginTop: 40, fontFamily: FONT, fontSize: 26, color: C.muted }}>
            <span>32× &amp; effective 64× tokenizers</span><span>·</span><span>semi-AR head</span><span>·</span><span>2.5 s / 1024² image</span>
          </Rise>
          <Rise at={84} style={{ marginTop: 34, fontFamily: FONT, fontSize: 30, color: C.blueSoft }}>github.com/DAGroup-PKU/G-STAR</Rise>
          <Rise at={96} style={{ display: 'flex', gap: 16, marginTop: 30 }}>
            {['logo-pku.png', 'logo-meituan.png'].map(f => <Img key={f} src={staticFile(f)} style={{ height: 44, background: '#fff', borderRadius: 10, padding: '8px 16px' }} />)}
          </Rise>
        </AbsoluteFill>
      </AbsoluteFill>
    </SceneFade>
  );
};
