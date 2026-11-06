import { AbsoluteFill, Sequence, useCurrentFrame } from 'remotion';
import { CAPTIONS } from './captions';
import { Background, C, FONT, prog, useFonts } from './theme';
import { WhyDiscrete } from './scenes/WhyDiscrete';
import { Hook } from './scenes/Hook';
import { Problem } from './scenes/Problem';
import { Title } from './scenes/Title';
import { WhyGroup } from './scenes/WhyGroup';
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
  [WhyDiscrete, 0, 7.5],
  [Hook, 7.5, 11],
  [Problem, 11, 15],
  [Title, 15, 18],
  [WhyGroup, 18, 24.7],
  [CoScaling, 24.7, 31],
  [Heads, 31, 39.5],
  [Speed, 39.5, 44.5],
  [Compare, 44.5, 49.5],
  [Editing, 49.5, 53.5],
  [Outro, 53.5, 58],
];
export const DURATION = s(58);

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
