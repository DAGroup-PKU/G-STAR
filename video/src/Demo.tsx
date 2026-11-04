import { AbsoluteFill, Sequence, useCurrentFrame } from 'remotion';
import { CAPTIONS } from './captions';
import { Background, C, FONT, prog, useFonts } from './theme';
import { Hook } from './scenes/Hook';
import { Problem } from './scenes/Problem';
import { Title } from './scenes/Title';
import { CoScaling } from './scenes/CoScaling';
import { Heads } from './scenes/Heads';
import { Speed } from './scenes/Speed';
import { Compare } from './scenes/Compare';
import { Editing } from './scenes/Editing';
import { Outro } from './scenes/Outro';

export const FPS = 30;
const s = (sec: number) => Math.round(sec * FPS);

// [component, start (s), end (s)]
const SCENES: [React.FC, number, number][] = [
  [Hook, 0, 7.1],
  [Problem, 7.1, 15.1],
  [Title, 15.1, 19.1],
  [CoScaling, 19.1, 31.1],
  [Heads, 31.1, 43.1],
  [Speed, 43.1, 51.1],
  [Compare, 51.1, 60.1],
  [Editing, 60.1, 66.1],
  [Outro, 66.1, 73.5],
];
export const DURATION = s(73.5);

const Captions: React.FC = () => {
  const frame = useCurrentFrame();
  const t = frame / FPS;
  const cap = CAPTIONS.find(c => t >= c.start - 0.15 && t <= c.end + 0.15);
  if (!cap) return null;
  const o = Math.min(prog(frame, s(cap.start) - 4, s(cap.start) + 4), 1 - prog(frame, s(cap.end) - 4, s(cap.end) + 4));
  return (
    <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 54, pointerEvents: 'none' }}>
      <div style={{
        opacity: o, fontFamily: FONT, fontSize: 36, fontWeight: 500, color: C.text, textAlign: 'center',
        maxWidth: 1500, padding: '10px 26px', borderRadius: 12, background: 'rgba(3,5,10,0.62)',
        textShadow: '0 2px 10px rgba(0,0,0,0.6)',
      }}>{cap.text}</div>
    </AbsoluteFill>
  );
};

export const Demo: React.FC = () => {
  useFonts();
  return (
    <AbsoluteFill style={{ background: C.bg, fontFamily: FONT }}>
      <Background />
      {SCENES.map(([Comp, a, b], i) => (
        <Sequence key={i} from={s(a)} durationInFrames={s(b) - s(a)}>
          <Comp />
        </Sequence>
      ))}
      <Captions />
    </AbsoluteFill>
  );
};
