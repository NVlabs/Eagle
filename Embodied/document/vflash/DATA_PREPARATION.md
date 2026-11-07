# VFlash Data Preparation

[VFlash](README.md) · [Training](TRAINING.md) · [Data Preparation](DATA_PREPARATION.md) · [Evaluation](EVALUATION.md) · [Detailed Results](RESULTS.md)

VFlash training uses JSONL annotation records and a JSON recipe that selects datasets and sampling weights. The records contain image or video observations, human prompts, and assistant answers with physical primitives. The same recipe and answer format train the autoregressive target and its drafter.

## Table of Contents

- [Physical primitives](#physical-primitives)
- [Recipe format](#recipe-format)
- [Data format and media](#data-format)

## Physical Primitives

This implementation represents coordinates with 1,001 number tokens, `<0>` … `<1000>`. Each primitive's coordinate range is quantized linearly with half-up rounding and stated in the prompt.

| Primitive | Answer format | Coordinates |
|:--|:--|:--|
| Point | `<point><x><y></point>` | image, `[0, 1000]` |
| Box | `<ref>label</ref><objects><box><x1><y1><x2><y2></box>…</objects>` | image, `x1 < x2` and `y1 < y2`; an absent category is `<objects>None</objects>` |
| Robot 2D | `<robot_single_2d><u1><v1>…<uP><vP></robot>` | image, `[0, 1000]` |
| Robot 3D | `<robot_single><x1><y1><z1>…<xP><yP><zP></robot>` | camera frame (x right, y down, z forward): x and y in [−5, 5] m, z in [0, 10] m |
| Driving | `<drive><x1><y1><h1>…<xP><yP><hP></drive>` | ego frame: x forward in [−10, 250] m, y left in [−65, 65] m, heading in [−π, π] |
| Dense 3D motion | one `<dense><x1><y1><z1>…</dense>` per query point, one triple per future frame | relative to the first query point: `(k − 500) · s / 500` m for a per-dataset scale `s` (for example 2 m) |

Dual-arm robots write a left-arm atom and then a right-arm atom (`<robot_left_2d>`, `<robot_right_2d>`, `<robot_left>`, `<robot_right>`), each closed by `</robot>`.

Robot, driving and dense trajectories are preceded by an `atomic_decoding` tool call that the target writes as ordinary text. `block_size` counts every token of the primitive, including its start and end tokens. For a single robot or driving trajectory, an optional `count` gives the number of waypoints; for dense motion, `count` gives the number of trajectories. Robot manipulation answers usually state the sub-task in `<action>…</action>` before the call:

```text
<action>Pull the drawer open.</action><tool_call>{"name":"atomic_decoding","arguments":{"block_size":32}}</tool_call><robot_single><521><489><61>…<509><470><85></robot>
```

| Layout | `block_size` |
|:--|:--|
| Single-arm 2D / 3D, `P` waypoints | `2P + 2` / `3P + 2` |
| Dual-arm 2D / 3D | `4P + 4` / `6P + 4` |
| Driving (x, y, heading) | `3P + 2` |
| Dense 3D motion, `F` future frames | `3F + 2` per trajectory, with `count` trajectories |

## Recipe Format

Pass the recipe as the first argument to either VFlash training script. The launcher supplies it to the trainer as `--meta_path`. This synthetic example contains robot trajectory and box entries:

```json
{
  "robot3d_demo": {
    "annotation": ["data/robot3d/part_000.jsonl", "data/robot3d/part_001.jsonl"],
    "root": "data/media",
    "repeat_time": 1.0,
    "data_augment": false,
    "video_reader_backend": "pyav",
    "visual_aug": {
      "prob": 0.3,
      "views": "all",
      "ops": ["motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur", "light"],
      "severity": [1, 10],
      "zoom_max_severity": 5
    }
  },
  "boxes_demo": {
    "annotation": "data/boxes/multiturn.jsonl",
    "root": "data/media",
    "repeat_time": 2.0,
    "data_augment": false,
    "video_reader_backend": "pyav"
  }
}
```

| Field | Description |
|:--|:--|
| `annotation` | Required path or list of paths to JSONL annotation files. |
| `root` | Root for relative image/video paths; defaults to an empty string. |
| `repeat_time` | Defaults to `1.0`. Values of at least `1` multiply the source's sampling weight; values between `0` and `1` select a deterministic subset. |
| `data_augment` | Resize augmentation switch; defaults to `false`. The reference example disables it. |
| `length` | Optional expected row count of a single-file entry; a mismatch is logged. |
| `video_reader_backend` | Optional video decoder of the entry (`decord`, `pyav` or `torchvision`); the reference recipe uses `pyav`. |
| `visual_aug` | Optional photometric augmentation. For each sample, with probability `prob`, one op from `ops` at a severity drawn from `severity` (1–10) is applied to the listed `views` (`"all"` or a list of image/video indexes); `zoom_max_severity` caps zoom blur. Geometry is unchanged, so the labels stay valid. It applies only in VFlash training (`--training_mode ar` or `vflash`). |

## Data Format

Each JSONL row contains `conversations` with `human` and `gpt` turns, plus the media fields for the observation. The `gpt` turns contain the physical primitives defined above. The [shared multimodal record format](../DATA_FORMAT.md) describes additional media and conversation options. The synthetic robot example below is expanded for readability. Store each record on one line in the JSONL file.

```json
{
  "image": [{"video": "videos/episode_000042.mp4", "frame_index": 120}],
  "conversations": [
    {"from": "human", "value": "<image>\nThe input image shows the current scene.\n\nBased on the current observation while the robot is doing \"Open the drawer\", predict the action of the gripper for the next 5 seconds, followed by a 10-point trajectory with 3D coordinates.\n\nCoordinates are expressed in the camera frame. Positive x points right, positive y points down, and positive z points forward. Positions are in meters. Tokens <0> to <1000> represent x and y from -5 to 5 meters and z from 0 to 10 meters."},
    {"from": "gpt", "value": "<action>Pull the drawer open.</action><tool_call>{\"name\":\"atomic_decoding\",\"arguments\":{\"block_size\":32}}</tool_call><robot_single><521><489><61><520><488><64><518><487><67><517><485><70><516><483><73><515><480><76><513><478><78><512><475><80><510><473><83><509><470><85></robot>"}
  ]
}
```

Other answer formats:

```text
2D trajectory: <tool_call>{"name":"atomic_decoding","arguments":{"block_size":10,"count":4}}</tool_call><robot_single_2d><612><540><560><548><498><552><431><550></robot>
Driving:       <tool_call>{"name":"atomic_decoding","arguments":{"block_size":26}}</tool_call><drive><62><500><500><86><500><500><110><501><501><134><501><501><158><502><502><181><503><502><205><504><503><228><505><503></drive>
Boxes:         <ref>cup</ref><objects><box><102><340><218><505></box><box><640><322><731><480></box></objects>
No box:        <ref>dog</ref><objects>None</objects>
Point:         <point><218><365></point>
```

- **Media.** `image` is a path, a list of paths, `{"video": …, "frame_index": N}` (one decoded video frame), `{"file": …}`, or an LMDB reference (`{"lmdb_file": …, "lmdb_key": …}`). Dictionary references accept `"resize": [width, height]`, an exact stretch. Box rows ask for one category per turn of a multi-turn conversation.
- **`atom_loss_masks`** (optional). One string per primitive of the answer, with one `0`/`1` character per coordinate token; `0` removes that token from the loss. Dense motion rows mark occluded points this way.

For packing and training budgets, see [VFlash Training](TRAINING.md#packing-and-training-budget). For the additional ground-truth fields needed by each benchmark, see [Evaluation](EVALUATION.md).
