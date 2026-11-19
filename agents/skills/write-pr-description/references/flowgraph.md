# Drawing an Example's Graph Topology

The graph answers one question, written as its first line: "How does a frame move through
`<example>`?". It shows the parts `main.py` and `main.cpp` connect, at that altitude only.

## Parts

- One box per Neat part the entrypoint creates or connects: the input group
  (`RtspEncodedInput`, `ImageInput`), `SimaDecode`, each `Model` graph, each `Output`, and each
  sender (`VideoSender`, `MetadataSender`). Host code between models, such as postprocessing in
  `utils/`, is one box.
- Name each box as the code names it. Fold the parts inside a Neat group into the group.
- Label every arrow with what moves, in two to four words: "encoded frames", "NV12 frames",
  "BBOX tensor", "boxes as JSON". Label a link with its policy when it has one:
  "latest per stream".
- For many identical streams, draw one stream and write the count on its input box:
  `RtspEncodedInput × N`.
- Draw a boundary, `╭┄╮ ┆ ╰┄╯`, around what runs inside one `Run`, and leave host code and
  Insight outside it.

## Characters and layout

- Put the graph in a fenced code block with no language.
- Boxes use `┌ ┐ └ ┘ ─ │`, lines use `─ │`, joins use `┬ ┴ ├ ┤ ┼`, and arrowheads `► ◄ ▼ ▲`
  touch the box they point into. Use spaces only, no tabs, emoji, or ASCII `+-|` boxes.
- Keep the graph at most 100 columns wide. Run top to bottom, inputs first.
- Put change markers after the name: `VideoSender (new)`.
- Draw graphs of more than four boxes with a short script that places boxes by row and column,
  then check that every line has the same width inside each box.

Under the graph, write one line per part or path worth the reviewer's attention, using the
graph's names.

## Example

```
How does a frame move through high-density-multi-stream-object-detector?

╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
┆ RUN                                                             ┆
┆  ┌──────────────────────┐  encoded frames  ┌──────────────────┐ ┆
┆  │ RtspEncodedInput × N │─────────────────►│ VideoSender × N  │──────┐
┆  └──────────────────────┘  latest/stream   └──────────────────┘ ┆    │
┆             │ encoded frames                                    ┆    │ video
┆             ▼                                                   ┆    │
┆  ┌──────────────────────┐                                       ┆    │
┆  │ SimaDecode × N       │                                       ┆    │
┆  └──────────────────────┘                                       ┆    │
┆             │ NV12, latest/stream                               ┆    │
┆             ▼                                                   ┆    │
┆  ┌──────────────────────┐                                       ┆    │
┆  │ YOLO26 Model         │                                       ┆    │
┆  └──────────────────────┘                                       ┆    │
┆             │ BBOX tensor                                       ┆    │
┆             ▼                                                   ┆    │
┆  ┌──────────────────────┐                                       ┆    │
┆  │ Output "detections"  │                                       ┆    │
┆  └──────────────────────┘                                       ┆    │
╰┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯    │
              ▼ pull                                                   ▼
   ┌──────────────────────┐   boxes as JSON                     ┌──────────────┐
   │ MetadataSender × N   │────────────────────────────────────►│ Insight      │
   └──────────────────────┘                                     └──────────────┘
```

- `RtspEncodedInput` feeds both `VideoSender`, which passes the encoded video to Insight
  unchanged, and `SimaDecode`, whose frames reach the one shared YOLO26 model.
- Both links keep only the latest frame per stream, so a slow stream cannot block the others.
- The application pulls `detections` and sends each stream's boxes on its own `MetadataSender`.
