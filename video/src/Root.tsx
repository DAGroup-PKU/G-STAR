import { Composition } from 'remotion';
import { Demo, DURATION, FPS } from './Demo';

export const Root: React.FC = () => (
  <Composition id="GSTARDemo" component={Demo} durationInFrames={DURATION} fps={FPS} width={1920} height={1080} />
);
