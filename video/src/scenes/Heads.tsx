import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { C, Card, Eyebrow, Headline, prog, Rise, SceneFade, FONT } from '../theme';

const G = 16, MAX = 6.44, TRACK = 1060;
const x = (t: number) => (t / MAX) * TRACK;

type Lane = { name: string; lat: number; ge: string; color: string; ours?: boolean; note: string };
const LANES: Lane[] = [
  { name: 'Causal AR', lat: 6.44, ge: '0.839', color: C.codeBlue, note: 'G full-head passes' },
  { name: 'Parallel', lat: 1.77, ge: '0.780', color: C.flash, note: 'one pass · codes collide' },
  { name: 'Semi-AR (ours)', lat: 3.41, ge: '0.825', color: C.green, ours: true, note: 'one pass + 1-layer causal sweep' },
];

const Blk: React.FC<{ t0: number; t1: number; y: number; h: number; color: string; now: number; r?: number }> = ({ t0, t1, y, h, color, now, r = 5 }) => {
  if (now <= t0) return null;
  const w = x(Math.min(now, t1)) - x(t0);
  return <div style={{ position: 'absolute', left: x(t0), top: y, width: Math.max(0, w - 2), height: h, background: color, borderRadius: r }} />;
};

export const Heads: React.FC = () => {
  const frame = useCurrentFrame();
  // shared playhead in "head seconds": sweep 0 → 6.44 s over ~6 s of video
  const now = interpolate(frame, [24, 170], [0, MAX], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const highlight = prog(frame, 180, 205);

  return (
    <SceneFade>
      <AbsoluteFill style={{ padding: '0 120px 150px', justifyContent: 'center' }}>
        <Rise at={0}><Eyebrow>Intra-group prediction head</Eyebrow></Rise>
        <Rise at={6}><Headline size={50} style={{ marginTop: 12 }}>Parallel computation. <span style={{ color: C.green }}>Causal consistency.</span></Headline></Rise>

        <div style={{ marginTop: 44, display: 'flex', flexDirection: 'column', gap: 26 }}>
          {LANES.map((l, i) => {
            const done = now >= l.lat;
            const codes: React.ReactNode[] = [];
            const blocks: React.ReactNode[] = [];
            if (i === 0) {
              const w = l.lat / G;
              for (let g = 0; g < G; g++) {
                blocks.push(<Blk key={`b${g}`} t0={g * w} t1={(g + 1) * w} y={12} h={58} color={l.color} now={now} />);
                if (now >= (g + 1) * w) codes.push(<div key={`c${g}`} style={{ position: 'absolute', left: x((g + 1) * w) - 14, top: 80, width: 11, height: 18, borderRadius: 2, background: C.yellow }} />);
              }
            } else if (i === 1) {
              blocks.push(<Blk key="b" t0={0} t1={l.lat} y={12} h={58} color={l.color} now={now} />);
              if (done) for (let g = 0; g < G; g++) codes.push(<div key={`c${g}`} style={{ position: 'absolute', left: x(l.lat) - 14 + (g % 4) * 3, top: 76 + Math.floor(g / 4) * 4, width: 11, height: 18, borderRadius: 2, background: g % 3 === 1 ? C.pro : C.yellow }} />);
            } else {
              const trunk = 1.77, step = (l.lat - trunk) / G;
              blocks.push(<Blk key="b" t0={0} t1={trunk} y={12} h={58} color={l.color} now={now} />);
              for (let g = 0; g < G; g++) {
                blocks.push(<Blk key={`s${g}`} t0={trunk + g * step} t1={trunk + (g + 1) * step} y={36} h={34} color={C.codeBlue} now={now} r={3} />);
                if (now >= trunk + (g + 1) * step) codes.push(<div key={`c${g}`} style={{ position: 'absolute', left: x(trunk + (g + 1) * step) - 9, top: 80, width: 7, height: 18, borderRadius: 2, background: C.yellow }} />);
              }
            }
            const glow = l.ours ? highlight : 0;
            const fade = l.ours ? 1 : 1 - highlight * 0.45;
            return (
              <Rise key={l.name} at={14 + i * 8}>
                <Card style={{ display: 'flex', alignItems: 'center', padding: '26px 30px', gap: 30, opacity: fade, borderColor: glow ? `rgba(127,207,143,${0.25 + glow * 0.6})` : C.border, boxShadow: glow ? `0 0 ${30 * glow}px rgba(127,207,143,0.35)` : 'none' }}>
                  <div style={{ width: 250 }}>
                    <div style={{ fontFamily: FONT, fontSize: 30, fontWeight: 700, color: l.ours ? C.green : C.text }}>{l.name}</div>
                    <div style={{ fontFamily: FONT, fontSize: 18, color: C.dim, marginTop: 4 }}>{l.note}</div>
                  </div>
                  <div style={{ position: 'relative', width: TRACK, height: 108 }}>
                    <div style={{ position: 'absolute', left: 0, right: 0, top: 41, height: 1, background: 'rgba(255,255,255,0.08)' }} />
                    {blocks}{codes}
                    {i === 1 && done && <div style={{ position: 'absolute', left: x(l.lat) + 24, top: 70, fontFamily: FONT, fontSize: 22, color: C.pro, fontWeight: 700 }}>× collision</div>}
                  </div>
                  <div style={{ width: 170, textAlign: 'right', fontFamily: FONT }}>
                    <div style={{ fontSize: 40, fontWeight: 800, color: done ? (l.ours ? C.green : C.text) : C.dim, fontVariantNumeric: 'tabular-nums' }}>{done ? `${l.lat.toFixed(2)} s` : '…'}</div>
                    <div style={{ fontSize: 19, color: C.muted }}>GenEval {l.ge}</div>
                  </div>
                </Card>
              </Rise>
            );
          })}
        </div>
        <Rise at={40} style={{ marginTop: 18, fontFamily: FONT, fontSize: 18, color: C.dim }}>
          Head latency per 512² image, G = 16, matched training (paper Table 4). Timeline split inside Semi-AR is illustrative.
        </Rise>
      </AbsoluteFill>
    </SceneFade>
  );
};
