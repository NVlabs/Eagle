# EagleEmbodied Data Format Documentation

This document describes the supported data formats for training and inference with EagleEmbodied, including plain text, single image, multiple images, interleaved text-image, and video formats.

## Overview

EagleEmbodied supports flexible multimodal data formats that can handle text-only conversations, images, videos, and their combinations. The system automatically processes media placeholders, handles path resolution, and manages memory constraints through intelligent resizing.

## Supported Formats

### 1. Plain Text Format

**Description**: Text-only conversations without any media content.

**Input Format**:
```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is a subset of artificial intelligence..."}
  ]
}
```


**Features**:
- Supports multi-turn conversations
- No media processing required
- Direct text-to-text format conversion

---

### 2. Single Image Format

**Description**: Conversations with a single image reference.

**Input Format**:
```json
{
  "conversations": [
    {"from": "human", "value": "What is in <image-1>?"},
    {"from": "gpt", "value": "This image shows..."}
  ],
  "image": "path/to/image.jpg"
}
```

**Alternative Input Formats**:
```json
{
  "conversations": [...],
  "image": {
    "lmdb_file": "path/to/database.lmdb",
    "lmdb_key": "image_key_123"
  }
}
```

```json
{
  "conversations": [...],
  "image": "https://example.com/image.jpg"
}
```

**Supported Image Sources**:
- **Local file paths**: `"image.jpg"` or `"/absolute/path/image.jpg"`
- **LMDB databases**: `{"lmdb_file": "db.lmdb", "lmdb_key": "key"}`
- **HTTP/HTTPS URLs**: `"https://example.com/image.jpg"`
- **Base64 encoded**: `"data:image/png;base64,..."`
- **File protocol**: `"file:///path/to/image.jpg"`
- **PIL Image objects**: Direct PIL.Image instances

**Placeholder Format**:
- Use `<image-1>` in the conversation text to reference the image
- The placeholder will be automatically replaced with image tokens during processing

**Features**:
- Automatic placeholder detection and replacement
- Support for RGBA images (converted to RGB with white background)
- Automatic path resolution with `media_root` prefix

---

### 3. Multiple Images Format

**Description**: Conversations with multiple images.

**Input Format**:
```json
{
  "conversations": [
    {"from": "human", "value": "Compare <image-1> and <image-2>"},
    {"from": "gpt", "value": "The first image shows... while the second..."}
  ],
  "image_list": [
    "path/to/image1.jpg",
    "path/to/image2.jpg"
  ]
}
```

**Alternative Input Format**:
```json
{
  "conversations": [...],
  "image": [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
  ]
}
```

**Placeholder Format**:
- Use `<image-1>`, `<image-2>`, `<image-3>`, etc. to reference images
- Generic `<image>` placeholders are automatically converted to numbered placeholders
- Missing placeholders are automatically added at the beginning of the first conversation

**Memory Management**:
- Multiple images are automatically resized to fit within `VIDEO_TOTAL_PIXELS` limit (default: ~22M pixels)
- Each image is resized iteratively by 10% until total pixels fit within the limit
- Ensures equal distribution of pixel budget across all images

**Features**:
- Automatic placeholder numbering
- Intelligent image resizing for memory efficiency
- Support for mixed image sources (local files, LMDB, URLs)

---

### 4. Interleaved Text-Image Format

**Description**: Conversations where images are interleaved with text within the same message.

**Input Format**:
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Look at <image-1> and tell me what you see. Also check <image-2>."
    },
    {"from": "gpt", "value": "In the first image..."}
  ],
  "image_list": ["image1.jpg", "image2.jpg"]
}
```

**Alternative Format (Direct Content List)**:
```json
{
  "conversations": [
    {
      "from": "human",
      "value": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "And this one?"},
        {"type": "image", "image": "path/to/image2.jpg"}
      ]
    }
  ]
}
```

**Placeholder Behavior**:
- Placeholders can appear anywhere in the text
- Images are inserted at placeholder positions during processing
- Supports complex interleaving patterns

**Features**:
- Flexible text-image mixing
- Automatic placeholder resolution
- Support for complex multimodal reasoning scenarios

---

### 5. Video Format

**Description**: Conversations with video content.

**Input Format**:
```json
{
  "conversations": [
    {"from": "human", "value": "What happens in <video-1>?"},
    {"from": "gpt", "value": "The video shows..."}
  ],
  "video": "path/to/video.mp4"
}
```

**Alternative Input Format**:
```json
{
  "conversations": [...],
  "video_list": [
    "video1.mp4",
    "video2.mp4"
  ]
}
```

**Video Processing Options**:
```json
{
  "conversations": [...],
  "video": "video.mp4",
  "max_frames": 64,           // Maximum frames to sample (default: 64)
  "fps": 2.0,                 // Target sampling FPS (default: 2.0)
  "video_start": 0.0,         // Start time in seconds (default: 0.0)
  "video_end": 30.0,          // End time in seconds (optional)
  "video_total_pixels": 22000000  // Max total pixels (default: ~22M)
}
```

**Video Frame Sampling**:
- Frames are sampled based on target FPS and max_frames constraints
- Linear sampling: `torch.linspace(0, total_frames-1, nframes)`
- Automatic frame count calculation: `nframes = (total_frames / video_fps) * target_fps`
- Frames are resized if total pixels exceed `video_total_pixels`

**Video Backends**:
- **decord** (preferred): Faster video reading, requires `decord` package
- **torchvision** (fallback): Uses `torchvision.io.read_video`, requires torchvision >= 0.19.0 for HTTP support

**Placeholder Format**:
- Use `<video-1>`, `<video-2>`, etc. in conversation text
- Video placeholders are replaced with frame-by-frame tokens
- Frame timestamps are included in the output when available

**Output Format for Videos**:
- Videos are converted to tensor format: `[T, C, H, W]` (Time, Channels, Height, Width)
- Each frame is processed as an image
- Frame tokens are formatted as: `"Frame-{i+1}-{timestamp:.2f}s: <img><IMG_CONTEXT>...<IMG_CONTEXT></img>"`

**Features**:
- Automatic frame sampling and resizing
- Support for video segments (start/end times)
- Frame-level timestamp tracking
- Memory-efficient processing with pixel budget management
- Support for video lists (multiple videos in one sample)

---

## Data Processing Pipeline

### Step 1: Media Path Resolution
- Images/videos are resolved relative to `media_root` directory
- LMDB paths are handled specially with path normalization
- URL and base64 images are fetched automatically

### Step 2: Placeholder Processing
- Generic placeholders (`<image>`, `<video>`) are converted to numbered ones
- Missing placeholders are automatically added to the first conversation
- Placeholders are matched and replaced with actual media content

### Step 3: Format Conversion
- Input format: `{"conversations": [...], "image": ..., "video": ...}`
- Media is embedded in user messages as content list items

---

## Advanced Features

### Thinking Mode Support
The system supports thinking/reasoning mode with special markers:

- Eagle Embodied uses `/think` for non-empty reasoning and `/no_think` for
  direct answers. Direct-answer targets contain one canonical empty thinking
  block.
- Existing suffixes are normalized independently for every matched human/GPT
  turn instead of being appended again.
- Malformed conversations with unmatched or consecutive human/GPT messages are
  rejected instead of allowing labels to leak across turns.

### LMDB Image Support
Special handling for LMDB-encoded images:
- Supports standard LMDB format
- Special handling for AgiBotWorld dataset format
- Automatic path resolution and key lookup

### Memory Constraints
- **VIDEO_TOTAL_PIXELS**: Environment variable controlling total pixel budget (default: ~22M)
- Multiple images are automatically resized to fit within budget
- Videos are resized frame-by-frame to maintain pixel budget

### Chat Template
The processor supports chat template formatting:
- ChatML format: `<|im_start|>role\ncontent<|im_end|>\n`
- Automatic system message insertion
- Thinking mode suffix support (`/think` or `/no_think`)

---

## Example: Complete Multimodal Sample

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Analyze <image-1> and describe what happens in <video-1>. Also check <image-2>."
    },
    {
      "from": "gpt",
      "value": "The image shows a cat. In the video, a dog is running. The second image shows a bird."
    }
  ],
  "image_list": [
    "images/cat.jpg",
    "images/bird.jpg"
  ],
  "video": "videos/dog_running.mp4"
}
```

---

## Notes

1. **Placeholder Indexing**: Placeholders are 1-indexed (`<image-1>`, not `<image-0>`)
2. **Automatic Resizing**: Multiple images/videos are automatically resized to fit memory constraints
3. **Path Resolution**: All media paths are resolved relative to `media_root` parameter
4. **Missing Placeholders**: If media exists but no placeholder is found, placeholders are automatically added
5. **Generic Placeholders**: Generic `<image>` or `<video>` placeholders are automatically numbered sequentially
6. **Video Processing**: Videos are sampled and resized automatically based on configuration parameters
