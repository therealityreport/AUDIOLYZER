<!-- 54a3ecae-0e6f-4ae6-b8b2-379494f4886e d2158937-b8f6-4034-9870-8789b6bf1dc8 -->
# Streamlit App Revisions to Match Pipeline

## Overview

Transform the single-page Streamlit app into a multipage application with:

- Global episode run selector across all pages
- Canonical episode naming (`shows/{show_slug}/episodes/{episode_id}/`)
- CREATE AUDIO flow with preprocessing integration
- Stage-based pages matching PRD pipeline sequence
- Configurable settings with migration support

## 1. Create UI Operations Module

**File:** `src/show_scribe/ui_ops.py`

Create thin wrapper functions for UI pages:

```python
def ensure_episode_scaffold(show_slug: str, episode_id: str, video_path: Path, preset_path: Path) -> Path
def create_audio(episode_dir: Path, video_path: Path, preset_path: Path) -> dict[str, Path | None]
def run_transcription(episode_dir: Path, config_path: Path | None = None) -> subprocess.CompletedProcess
def run_diarization(episode_dir: Path, config_path: Path | None = None) -> subprocess.CompletedProcess
def run_bleep_detection(episode_dir: Path) -> subprocess.CompletedProcess
def run_speaker_id(episode_dir: Path) -> subprocess.CompletedProcess
def build_exports(episode_dir: Path) -> subprocess.CompletedProcess
def compute_analytics(episode_dir: Path) -> subprocess.CompletedProcess
def get_stage_status(episode_dir: Path) -> dict[str, bool]
```

Implementation notes:

- `create_audio()`: Direct import from `pipeline.audio_ops.extract_and_enhance()`
- Stage runners: Shell out to `scripts/run_pipeline.py` with appropriate flags
- `get_stage_status()`: Check for presence of key artifacts (transcript_raw.json, diarization.json, etc.)

## 2. Settings Management

**File:** `ui/streamlit_app/components/settings_manager.py`

```python
class SettingsManager:
    def load_settings() -> dict[str, Any]
    def save_settings(settings: dict) -> None
    def get_shows_root() -> Path
```

Settings storage: `~/.show_scribe/ui_settings.json`

Default settings:

```json
{
  "shows_root": "~/Documents/VoiceTranscriptTool/shows",
  "voice_bank_path": "~/Documents/VoiceTranscriptTool/voice_bank",
  "default_preset": "configs/reality_tv.yaml",
  "default_show_config": null
}
```

Precedence: UI settings > `SHOWS_ROOT` env var > default

## 3. Global Episode Picker Component

**File:** `ui/streamlit_app/components/episode_picker.py`

```python
def render_episode_picker() -> tuple[str | None, Path | None]:
    """Render sidebar episode selector, return (episode_label, episode_dir)"""
```

Logic:

- Read `shows_root` from settings
- Recursively find all `metadata.json` files under `shows/{show_slug}/episodes/*/`
- Also scan legacy `data/shows/` if it exists
- Display format: `{show_slug} · {episode_id}` (e.g., "RHOBH · S05E01")
- Store selection in `st.session_state["active_episode_dir"]` and `st.session_state["active_episode_label"]`

## 4. Main App Layout

**File:** `ui/streamlit_app/app.py`

Refactor to:

- Remove all page rendering logic (keep only as landing/home page)
- Initialize session state defaults
- Render global episode picker in sidebar
- Display welcome message and quick stats
- Link to page navigation

## 5. Create Multipage Structure

### Page 1: Process Episode

**File:** `ui/streamlit_app/pages/1_Process_Episode.py`

UI sections:

1. **Episode Configuration:**

   - Show slug input (text)
   - Episode ID input (text)
   - "Autofill from filename" button (parses `Show_S##E##` patterns)

2. **Video Selection:**

   - File path picker or text input

3. **Audio Preset (Advanced Accordion):**

   - Preset file path (default: `configs/reality_tv.yaml`)
   - Show vocal separation/enhancement toggles
   - Sample rate, channels, normalization settings

4. **CREATE AUDIO Button:**

   - Creates `shows/{show_slug}/episodes/{episode_id}/` directory
   - Copies/symlinks source video to `{episode_id}.mp4`
   - Writes `metadata.json` with: `source_video_path`, `episode_id`, `show_slug`, `created_at`, `preset`, `show_config`
   - Calls `ui_ops.create_audio()` → produces `audio_extracted.wav`, `audio_vocals.wav`, `audio_enhanced.wav`, `preprocessing_report.json`
   - On success: Sets `st.session_state["active_episode_dir"]` and reruns to populate selector

5. **Existing Audio Selector:**

   - After CREATE AUDIO, show dropdown scoped to current episode dir
   - List: `audio_extracted.wav`, `audio_vocals.wav`, `audio_enhanced.wav`
   - Default: `audio_enhanced.wav`

### Page 2: Transcription

**File:** `ui/streamlit_app/pages/2_Transcription.py`

- Show active episode badge at top
- Check requirements: `audio_processed.wav` or `audio_enhanced.wav` present
- If missing: disabled state with tooltip
- "Run Transcription" button → calls `ui_ops.run_transcription()`
- If `transcript_raw.json` exists: display preview (first 10 segments)

### Page 3: Diarization

**File:** `ui/streamlit_app/pages/3_Diarization.py`

- Show active episode badge
- Check requirements: `transcript_raw.json` present
- "Run Diarization" button → calls `ui_ops.run_diarization()`
- If `diarization.json` exists: display speaker segments table

### Page 4: Bleep Review

**File:** `ui/streamlit_app/pages/3b_Bleep_Review.py`

- Show active episode badge
- Check requirements: `diarization.json` present
- "Detect Bleeps" button → calls `ui_ops.run_bleep_detection()`
- If `bleeps.json` exists: render table with audio preview using `components/audio_player.py`
- Edit bleep word labels inline
- Export buttons: Save to `bleeps.csv` and `bleeps.json`

### Page 5: Speaker ID and Review

**File:** `ui/streamlit_app/pages/4_Speaker_ID_and_Review.py`

- Show active episode badge
- Check requirements: `diarization.json` present
- "Run Speaker ID" button → calls `ui_ops.run_speaker_id()`
- Display matched speakers (green badges for high confidence)
- Queue unknowns for manual labeling
- Integrate snippet labeler flow from current `app.py`

### Page 6: Transcript and Exports

**File:** `ui/streamlit_app/pages/5_Transcript_and_Exports.py`

- Show active episode badge
- Check requirements: speaker ID completed
- "Generate Exports" button → calls `ui_ops.build_exports()`
- Display: `transcript_final.txt`, `transcript_final.srt`, `transcript_final.json`
- Download buttons for each format
- Reuse transcript viewer components from current `app.py`

### Page 7: Analytics

**File:** `ui/streamlit_app/pages/6_Analytics.py`

- Show active episode badge
- "Compute Analytics" button → calls `ui_ops.compute_analytics()`
- If `analytics.json` exists: render charts
  - Speaking time per person (bar chart)
  - Bleep statistics (table + counts)
  - Timeline visualization

### Page 8: Voice Bank

**File:** `ui/streamlit_app/pages/7_Voice_Bank.py`

- Keep existing voice bank viewer from current `app.py`
- Add path configuration from settings
- Display speaker profiles table
- Merge speakers UI (future enhancement placeholder)

### Page 9: Settings

**File:** `ui/streamlit_app/pages/8_Settings.py`

Sections:

1. **Paths Configuration:**

   - Shows root directory
   - Voice bank path
   - Default preset file
   - Default show config path

2. **Environment Check:**

   - FFmpeg installed? (green/red badge)
   - Whisper models cached? (check data/models/)
   - Pyannote models?
   - Links to install scripts

3. **Legacy Data Migration:**

   - If `data/shows/` exists: Show "Migrate" button
   - Copies episodes to new location under `shows_root`

## 6. Shared Components

### Artifact Badge

**File:** `ui/streamlit_app/components/artifact_badge.py`

```python
def render_artifact_badge(artifact_path: Path, label: str) -> None:
    """Render green/yellow/red badge based on artifact presence"""
```

### Episode Badge

**File:** `ui/streamlit_app/components/episode_badge.py`

```python
def render_episode_badge(episode_dir: Path, metadata: dict) -> None:
    """Display episode info: show · episode · created · preset"""
```

### Stage Checklist

**File:** `ui/streamlit_app/components/stage_checklist.py`

```python
def render_stage_checklist(episode_dir: Path) -> None:
    """Show compact stage progress with Resume button"""
```

Display format:

```
✓ Audio Extraction
✓ Transcription
✓ Diarization
⧗ Bleep Detection (in progress)
○ Speaker ID
○ Exports
○ Analytics
```

## 7. Metadata Management

### metadata.json Schema

```json
{
  "episode_id": "RHOBH_S05E01",
  "show_slug": "RHOBH",
  "source_video_path": "/path/to/video.mp4",
  "created_at": "2025-10-23T10:30:00Z",
  "updated_at": "2025-10-23T11:45:00Z",
  "preset": "configs/reality_tv.yaml",
  "show_config": "data/shows/RHOBH/show_config.json",
  "artifacts": {
    "audio_extracted": "audio_extracted.wav",
    "audio_processed": "audio_enhanced.wav",
    "preprocessing_report": "preprocessing_report.json",
    "transcript_raw": "transcript_raw.json",
    "diarization": "diarization.json",
    "transcript_final": "transcript_final.json",
    "bleeps": "bleeps.json",
    "analytics": "analytics.json"
  }
}
```

Functions in `ui_ops.py`:

```python
def write_metadata(episode_dir: Path, data: dict) -> None
def read_metadata(episode_dir: Path) -> dict | None
def update_metadata(episode_dir: Path, updates: dict) -> None
```

## 8. Implementation Order

1. Create `settings_manager.py` component
2. Create `ui_ops.py` module with stub functions
3. Refactor `app.py` to minimal landing page
4. Create `episode_picker.py` component
5. Implement shared components (badges, checklist)
6. Create Page 1 (Process Episode) with CREATE AUDIO
7. Create Pages 2-7 (stage pages) with guardrails
8. Create Page 8 (Settings) with migration
9. Test full workflow end-to-end
10. Remove old single-page code from `app.py`

## Key Files Modified

- `ui/streamlit_app/app.py` - Refactor to landing page
- `ui/streamlit_app/pages/1_Process_Episode.py` - Replace with full flow
- Create: `ui/streamlit_app/pages/2_Transcription.py` through `8_Settings.py`
- Create: `src/show_scribe/ui_ops.py`
- Create: `ui/streamlit_app/components/settings_manager.py`
- Create: `ui/streamlit_app/components/episode_picker.py`
- Create: `ui/streamlit_app/components/artifact_badge.py`
- Create: `ui/streamlit_app/components/episode_badge.py`
- Create: `ui/streamlit_app/components/stage_checklist.py`

## Testing Plan

1. Create test episode with full pipeline run
2. Verify global selector persists across pages
3. Test CREATE AUDIO with and without preprocessing
4. Verify all stage pages render correct states
5. Test Settings page migration from `data/shows/`
6. Verify metadata.json written correctly
7. Test resume capability via stage checklist

### To-dos

- [ ] Create settings_manager.py component for persistent UI configuration
- [ ] Create ui_ops.py with wrapper functions for all pipeline stages
- [ ] Build shared components: episode_picker, artifact_badge, episode_badge, stage_checklist
- [ ] Refactor app.py to minimal landing page with global episode selector
- [ ] Create 1_Process_Episode.py with CREATE AUDIO flow and episode scaffold
- [ ] Create 2_Transcription.py with run button and artifact preview
- [ ] Create 3_Diarization.py with run button and segment display
- [ ] Create 3b_Bleep_Review.py with detection and word label editing
- [ ] Create 4_Speaker_ID_and_Review.py with matching and manual labeling
- [ ] Create 5_Transcript_and_Exports.py with format generation and downloads
- [ ] Create 6_Analytics.py with charts and statistics
- [ ] Create 7_Voice_Bank.py by adapting existing voice bank viewer
- [ ] Create 8_Settings.py with path config, env check, and migration
- [ ] Test full workflow end-to-end with real episode
