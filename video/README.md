# GSTAR 演示视频源码

项目主页里那段约一分钟的 demo（`docs/assets/demo.mp4`）就是用这里的代码渲染出来的。工程用 [Remotion 4](https://www.remotion.dev/)（React 写画面，逐帧渲染成 MP4）。

- 1920 × 1080，30 fps，73.5 秒，H.264
- 没有配音，字幕直接烧在画面里，另外导出一份 WebVTT 字幕文件
- 所有数字都来自论文；改数字时，主页 `docs/index.html` 里同一个数也要一起改

## 跑起来

需要 Node 18 或更新的版本。第一次渲染时 Remotion 会自动下载一个 Chrome Headless Shell，需要能联网。

```sh
cd video
npm ci            # 按 package-lock.json 装依赖，node_modules 约 1 GB，不入库
npm run studio    # 浏览器里预览，拖时间轴，改代码会热更新
npm run build     # 出片：out/demo.mp4、out/demo-poster.jpg、out/demo.vtt
```

`npm run build` 依次做三件事：

| 命令 | 输出 | 说明 |
| --- | --- | --- |
| `npm run render` | `out/demo.mp4` | H.264，CRF 20 |
| `npm run poster` | `out/demo-poster.jpg` | 第 500 帧（16.7 s，标题画面），主页视频的封面 |
| `npm run vtt` | `out/demo.vtt` | 从 `src/captions.ts` 生成的字幕文件 |

单独跑 `npm run vtt` 之前要先有 `out/` 目录（`mkdir -p out`）。

出片后把三个文件复制到主页：

```sh
cp out/demo.mp4 out/demo-poster.jpg out/demo.vtt ../docs/assets/
```

`out/` 和 `node_modules/` 已写在 `video/.gitignore` 里，不会被提交。

## 文件地图

```
video/
├── package.json         # 依赖版本（Remotion 4.0.539）和上面几个 npm 命令
├── remotion.config.ts   # 渲染设置：JPEG 中间帧、质量 92、覆盖输出
├── scripts/make-vtt.mjs # 把 captions.ts 转成 WebVTT
├── public/              # 图片、logo、字体；代码里用 staticFile('文件名') 引用
└── src/
    ├── index.ts / Root.tsx  # 注册合成 GSTARDemo（分辨率、帧率、总时长）
    ├── Demo.tsx             # 时间轴：每个场景从第几秒到第几秒，字幕层
    ├── captions.ts          # 字幕（画面字幕和 VTT 共用这一份）
    ├── theme.tsx            # 颜色、字体、共用组件（SceneFade、Rise、Card、Grid…）
    └── scenes/              # 每个场景一个文件
```

### 场景和时间

时间轴在 `src/Demo.tsx` 的 `SCENES` 里，单位是秒：

| 场景 | 时间 (s) | 内容 | 主要可改的地方 |
| --- | --- | --- | --- |
| `Hook` | 0 – 7.1 | 1024² 图（16× tokenizer，64×64 个 token）按光栅顺序一个个显现：序列太长、太慢 | `N`、配图 `gen-lumiere.webp` |
| `Problem` | 7.1 – 15.1 | 16×（4,096 步）和 64×（256 步）两块面板：压得更狠序列变短，但每个 token 装不下 | `<Panel>` 的参数、配图 `gen-frog.webp` |
| `Title` | 15.1 – 19.1 | GSTAR 标题和北大、美团 logo | 标题文字 |
| `CoScaling` | 19.1 – 31.1 | 分组量化，压缩 ×2 时组大小 ×4，2×2 无损打包成等效 64× | `G` 的计算、配图 `gen-panda.webp` |
| `Heads` | 31.1 – 43.1 | 三种组内预测头的逐步动画（causal / parallel / semi-AR） | `LANES`：延迟和 GenEval（论文 Table 4） |
| `Speed` | 43.1 – 51.1 | 22.23 s → 2.50 s 的四步加速 | `STAGES`（论文 Figure 5） |
| `Compare` | 51.1 – 60.1 | 和此前离散 AR 模型的加速比、Qwen-Image-Bench 分数条形图 | `SPEED`、`QIB`（论文 Table 1） |
| `Editing` | 60.1 – 66.1 | 两组编辑前后对比，分割线扫过一遍后停在中间 | `<Wipe>` 的图片和编辑指令 |
| `Outro` | 66.1 – 73.5 | 样图墙、标语、logo | `TILES`、标语 |

## 常见改法

**改数字或文字。** 数字都写在各场景文件开头的常量里（`LANES`、`STAGES`、`SPEED`、`QIB` 等）。很多数字在字幕里也出现了一次，例如 2.5 s、21×、7.45、50.07，记得同时改 `src/captions.ts`，以及主页 `docs/index.html` 里对应的地方。

**改字幕。** 只改 `src/captions.ts`，画面字幕和 VTT 都从这里来。每条是 `{ start, end, text }`，单位为秒。文字必须用单引号字符串；如果文字里有单引号，写成 `\'`，否则 `make-vtt.mjs` 的正则会漏掉这一条。字幕时间不要跨场景边界。

**改某个场景的时长。** 场景里的动画按"场景内帧号"计时（30 帧 = 1 秒，每个场景从第 0 帧开始），所以只平移场景不会打乱动画；但如果把场景缩短，靠后的动画就会被截掉。改法：

1. 改 `SCENES` 里这一段的结束时间，后面各段的起止时间依次顺延（首尾要相接）；
2. 改 `SCENES` 下面的 `DURATION`（总时长）；
3. 打开这个场景的文件，检查 `prog(frame, a, b)` 和 `interpolate(frame, [a, b], …)` 里的帧号有没有超出新时长；
4. 同步平移 `captions.ts` 里受影响的字幕时间；
5. 如果封面帧因此落到别的画面，改 `package.json` 里 `poster` 的 `--frame`。

**换图。** 把新图放进 `public/`，改场景里的 `staticFile('…')`。`public/` 里的样图是从 `docs/assets/` 复制过来的，两边保持一致即可。

**加一个场景。** 在 `src/scenes/` 新建组件，最外层包一层 `<SceneFade>`（淡入淡出），然后加进 `Demo.tsx` 的 `SCENES`，按上面的步骤调时间。

**改配色和字体。** 都在 `src/theme.tsx`：颜色在 `C`（`pro` 是 Pro-8B 的红色，`flash` 是 Flash-1.7B 的橙色），字体在 `public/fonts/`，由 `useFonts()` 在第一帧之前加载。

**加配音（目前没有）。** 把音频放进 `public/`，在 `Demo.tsx` 的 `<AbsoluteFill>` 里加 `<Audio src={staticFile('narration.mp3')} />`（从 `remotion` 引入），再按配音调整 `SCENES` 和字幕时间。

改完先用 `npm run studio` 从头到尾拖一遍，再 `npm run build` 出片。
