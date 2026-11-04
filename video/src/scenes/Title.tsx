import { AbsoluteFill, Img, staticFile, useCurrentFrame } from 'remotion';
import { C, FONT, prog, Rise, SceneFade } from '../theme';

export const Title: React.FC = () => {
  const frame = useCurrentFrame();
  const p = prog(frame, 2, 30);
  return (
    <SceneFade>
      <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', paddingBottom: 120 }}>
        <div style={{
          fontFamily: 'Caveat, cursive', fontSize: 230, lineHeight: 1, padding: '0 30px', opacity: p, transform: `scale(${0.92 + 0.08 * p})`,
          background: `linear-gradient(90deg, ${C.pro}, #ffffff 45%, #6692f1)`, WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent',
        }}>GSTAR</div>
        <Rise at={18}><div style={{ fontFamily: FONT, fontSize: 40, color: '#cfd6e4', textAlign: 'center', marginTop: 8 }}>Group-Scaled Token Capacity for<br />Deeply Compressed Autoregressive Generation</div></Rise>
        <Rise at={36} style={{ display: 'flex', gap: 18, marginTop: 44 }}>
          {['logo-pku.png', 'logo-meituan.png'].map(f => (
            <Img key={f} src={staticFile(f)} style={{ height: 52, background: '#fff', borderRadius: 12, padding: '9px 18px' }} />
          ))}
        </Rise>
      </AbsoluteFill>
    </SceneFade>
  );
};
