import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from 'remotion';
import { C, Eyebrow, Headline, prog, Rise, SceneFade, FONT } from '../theme';

const SIZE = 600;
const Wipe: React.FC<{ before: string; after: string; prompt: string; at: number }> = ({ before, after, prompt, at }) => {
  const frame = useCurrentFrame();
  // reveal the edit from left to right, then settle at 55 %
  const split = interpolate(frame, [at + 10, at + 45, at + 75], [100, 0, 45], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return (
    <Rise at={at}>
      <div style={{ position: 'relative', width: SIZE, height: SIZE, borderRadius: 16, overflow: 'hidden', border: `1px solid ${C.border}` }}>
        <Img src={staticFile(after)} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} />
        <div style={{ position: 'absolute', inset: 0, clipPath: `inset(0 ${100 - split}% 0 0)` }}>
          <Img src={staticFile(before)} style={{ width: '100%', height: '100%' }} />
        </div>
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${split}%`, width: 3, marginLeft: -1.5, background: '#fff', boxShadow: '0 0 10px rgba(0,0,0,0.6)' }} />
        <div style={{ position: 'absolute', top: 14, left: 14, padding: '4px 14px', borderRadius: 999, background: 'rgba(5,8,14,0.7)', color: '#fff', fontFamily: FONT, fontSize: 18 }}>Original</div>
        <div style={{ position: 'absolute', top: 14, right: 14, padding: '4px 14px', borderRadius: 999, background: 'rgba(5,8,14,0.7)', color: '#fff', fontFamily: FONT, fontSize: 18 }}>Edited</div>
      </div>
      <div style={{ marginTop: 18, fontFamily: FONT, fontSize: 24, color: '#cfd6e4', display: 'flex', gap: 12, alignItems: 'center' }}>
        <span style={{ color: C.blueSoft, fontWeight: 700 }}>Instruction</span>{prompt}
      </div>
    </Rise>
  );
};

export const Editing: React.FC = () => (
  <SceneFade>
    <AbsoluteFill style={{ padding: '60px 120px 0', alignItems: 'center' }}>
      <Rise at={0}><Eyebrow>Image editing · GEdit-Bench overall 7.45</Eyebrow></Rise>
      <Rise at={6}><Headline size={46} style={{ marginTop: 10 }}>Change the instruction, keep the scene</Headline></Rise>
      <div style={{ display: 'flex', gap: 70, marginTop: 34 }}>
        <Wipe before="edit-watercolor-original.webp" after="edit-watercolor-result.webp" prompt="Turn the photo into a watercolor illustration" at={6} />
        <Wipe before="edit-blackboard-original.webp" after="edit-blackboard-result.webp" prompt="Replace the heading with the G-STAR title" at={14} />
      </div>
    </AbsoluteFill>
  </SceneFade>
);
