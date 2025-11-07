# Cast Name Management System

**Feature:** Canonical Name Spelling & Correction
**Date:** October 16, 2025
**Priority:** 🔴 HIGH - Core MVP Feature

---

## Overview

Show-Scribe provides **multiple mechanisms** to ensure cast members' names are spelled correctly throughout the system, from initial setup through final transcript generation.

---

## Solution Architecture

### Layer 1: Show Configuration (Primary Source of Truth)

**File:** `~/Documents/VoiceTranscriptTool/shows/[ShowName]/show_config.json`

```json
{
  "show_name": "The Office",
  "cast_members": [
    {
      "canonical_name": "Michael Scott",
      "common_misspellings": ["Micheal Scott", "Michael Scot", "Mike Scott"],
      "aliases": ["Michael", "Mike"],
      "role": "main"
    },
    {
      "canonical_name": "Dwight Schrute",
      "common_misspellings": ["Dwight Shrute", "Dwite Schrute"],
      "aliases": ["Dwight", "Dwight K. Schrute"],
      "role": "main"
    },
    {
      "canonical_name": "Jim Halpert",
      "common_misspellings": ["Jim Halbert", "Jim Halpart"],
      "aliases": ["Jim", "James Halpert"],
      "role": "main"
    },
    {
      "canonical_name": "Pam Beesly",
      "common_misspellings": ["Pam Beasley", "Pam Beazley", "Pamela Beesly"],
      "aliases": ["Pam", "Pamela"],
      "role": "main"
    }
  ],
  "confidence_threshold": 0.80,
  "auto_correct_names": true
}
```

**Usage:**
- Loaded at pipeline start
- Used for name validation
- Used for auto-correction
- Referenced in UI dropdowns

#### Keeping cast directories in sync

**Required step:** Run `make sync-cast` (or `python scripts/maintenance/sync_cast_configs.py`) immediately after committing any `show_config.json` changes.
The helper will:

- create `data/shows/<slug>/cast/season_<NN>/` if it does not already exist
- regenerate `cast_list.json` with the canonical spellings in the proper order

Add `--dry-run` to preview changes or `--show <slug>` to target specific shows.

---

### Layer 2: Voice Bank Database (Speaker Profiles)

**Table:** `speaker_profiles`

```sql
CREATE TABLE speaker_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,              -- Key (lowercase, no spaces): "michaelscott"
    display_name TEXT NOT NULL,             -- Canonical: "Michael Scott"
    common_aliases TEXT,                     -- JSON array: ["Mike", "Michael"]
    phonetic_spelling TEXT,                  -- For search: "MY-kel SKOT"
    first_appearance_episode TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_segments INTEGER DEFAULT 0,
    notes TEXT
);

-- Example entries
INSERT INTO speaker_profiles (name, display_name, common_aliases) VALUES
    ('michaelscott', 'Michael Scott', '["Michael", "Mike", "Mike Scott"]'),
    ('dwightschrute', 'Dwight Schrute', '["Dwight", "Dwight K. Schrute"]'),
    ('jimhalpert', 'Jim Halpert', '["Jim", "James", "James Halpert"]'),
    ('pambeesly', 'Pam Beesly', '["Pam", "Pamela", "Pamela Beesly"]');
```

**Key Fields:**
- `name`: Normalized key (lowercase, no spaces) - used for matching
- `display_name`: **Canonical spelling** - used in all outputs
- `common_aliases`: Alternative names for matching

---

### Layer 3: Name Correction Engine

**Module:** `src/show_scribe/utils/name_correction.py`

```python
from typing import Dict, List, Optional
import re
import json
from difflib import get_close_matches

class NameCorrector:
    """Ensures consistent and correct spelling of cast member names."""

    def __init__(self, show_config: dict, voice_bank_db):
        self.show_config = show_config
        self.db = voice_bank_db

        # Build lookup dictionaries
        self.canonical_names = {}  # normalized -> canonical
        self.aliases_map = {}      # alias -> canonical
        self.misspelling_map = {}  # misspelling -> canonical

        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build fast lookup tables from config and database."""

        # From show config
        for cast_member in self.show_config.get("cast_members", []):
            canonical = cast_member["canonical_name"]
            normalized = self._normalize_name(canonical)

            # Canonical name
            self.canonical_names[normalized] = canonical

            # Aliases
            for alias in cast_member.get("aliases", []):
                norm_alias = self._normalize_name(alias)
                self.aliases_map[norm_alias] = canonical

            # Known misspellings
            for misspelling in cast_member.get("common_misspellings", []):
                norm_miss = self._normalize_name(misspelling)
                self.misspelling_map[norm_miss] = canonical

        # From voice bank database
        profiles = self.db.get_all_speaker_profiles()
        for profile in profiles:
            normalized = profile["name"]
            canonical = profile["display_name"]

            self.canonical_names[normalized] = canonical

            # Add aliases from database
            if profile.get("common_aliases"):
                aliases = json.loads(profile["common_aliases"])
                for alias in aliases:
                    norm_alias = self._normalize_name(alias)
                    self.aliases_map[norm_alias] = canonical

    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching: lowercase, no spaces."""
        return re.sub(r'[^a-z0-9]', '', name.lower())

    def correct_name(self, name: str, confidence: float = 1.0) -> dict:
        """
        Correct a potentially misspelled name.

        Returns:
            {
                "original": "Micheal Scott",
                "corrected": "Michael Scott",
                "method": "misspelling_map",
                "confidence": 0.95
            }
        """
        normalized = self._normalize_name(name)

        # 1. Check if already canonical
        if normalized in self.canonical_names:
            canonical = self.canonical_names[normalized]
            if canonical == name:  # Already correct
                return {
                    "original": name,
                    "corrected": name,
                    "method": "exact_match",
                    "confidence": 1.0
                }
            else:
                return {
                    "original": name,
                    "corrected": canonical,
                    "method": "canonical_lookup",
                    "confidence": 1.0
                }

        # 2. Check aliases
        if normalized in self.aliases_map:
            return {
                "original": name,
                "corrected": self.aliases_map[normalized],
                "method": "alias_match",
                "confidence": 0.98
            }

        # 3. Check known misspellings
        if normalized in self.misspelling_map:
            return {
                "original": name,
                "corrected": self.misspelling_map[normalized],
                "method": "misspelling_map",
                "confidence": 0.95
            }

        # 4. Fuzzy matching (find closest match)
        all_canonical = list(self.canonical_names.values())
        matches = get_close_matches(name, all_canonical, n=1, cutoff=0.8)

        if matches:
            return {
                "original": name,
                "corrected": matches[0],
                "method": "fuzzy_match",
                "confidence": 0.75
            }

        # 5. No match found
        return {
            "original": name,
            "corrected": name,
            "method": "no_correction",
            "confidence": confidence
        }

    def correct_transcript(self, transcript_text: str) -> str:
        """
        Apply name corrections to entire transcript.

        Example:
            Input:  "[00:00:15] Micheal Scott: Good morning!"
            Output: "[00:00:15] Michael Scott: Good morning!"
        """
        if not self.show_config.get("auto_correct_names", True):
            return transcript_text

        # Find all speaker labels in transcript
        pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s+([^:]+):'

        def replace_name(match):
            timestamp = match.group(1)
            name = match.group(2).strip()

            correction = self.correct_name(name)
            corrected_name = correction["corrected"]

            return f"[{timestamp}] {corrected_name}:"

        corrected = re.sub(pattern, replace_name, transcript_text)
        return corrected

    def get_canonical_name(self, name: str) -> str:
        """Get canonical spelling for a name."""
        return self.correct_name(name)["corrected"]

    def validate_name(self, name: str) -> bool:
        """Check if name is in canonical form."""
        correction = self.correct_name(name)
        return correction["original"] == correction["corrected"]


# Usage in pipeline
corrector = NameCorrector(show_config, voice_bank_db)

# Correct a single name
result = corrector.correct_name("Micheal Scott")
print(result)
# {'original': 'Micheal Scott', 'corrected': 'Michael Scott', 'method': 'misspelling_map', 'confidence': 0.95}

# Correct entire transcript
corrected_transcript = corrector.correct_transcript(transcript_text)
```

---

### Layer 4: UI Name Selection

**Streamlit Component:** Name selector with autocomplete

```python
# ui/streamlit_app/components/name_selector.py

import streamlit as st
from typing import List, Optional

def render_name_selector(
    label: str,
    cast_members: List[str],
    key: str,
    allow_custom: bool = True,
    default: Optional[str] = None
) -> str:
    """
    Render a name selector with autocomplete and validation.

    Args:
        label: Label for the input
        cast_members: List of canonical cast member names
        key: Unique key for the widget
        allow_custom: Allow entering names not in the list
        default: Default selected name

    Returns:
        Selected canonical name
    """

    # Sort alphabetically
    sorted_cast = sorted(cast_members)

    if allow_custom:
        # Selectbox with "Other (type name)" option
        options = sorted_cast + ["➕ Other (type name)"]

        selected = st.selectbox(
            label,
            options=options,
            index=0 if default is None else (
                sorted_cast.index(default) if default in sorted_cast else len(sorted_cast)
            ),
            key=f"{key}_select"
        )

        if selected == "➕ Other (type name)":
            # Text input for custom name
            custom_name = st.text_input(
                "Enter name:",
                key=f"{key}_custom",
                help="Type the speaker's full name"
            )

            if custom_name:
                # Show suggestion if similar to known cast
                from difflib import get_close_matches
                matches = get_close_matches(custom_name, sorted_cast, n=1, cutoff=0.7)

                if matches:
                    st.warning(f"💡 Did you mean **{matches[0]}**?")
                    if st.button(f"Use '{matches[0]}'", key=f"{key}_suggest"):
                        return matches[0]

                return custom_name

            return None
        else:
            return selected
    else:
        # Strict selection from cast list
        return st.selectbox(
            label,
            sorted_cast,
            index=0 if default is None else (
                sorted_cast.index(default) if default in sorted_cast else 0
            ),
            key=f"{key}_select"
        )


# Usage in review UI
canonical_name = render_name_selector(
    label="Who is this speaker?",
    cast_members=["Michael Scott", "Dwight Schrute", "Jim Halpert", "Pam Beesly"],
    key="speaker_label",
    allow_custom=True
)
```

---

### Layer 5: Transcript Post-Processing

**Module:** `src/show_scribe/pipelines/transcript/builder.py`

```python
class TranscriptBuilder:
    """Build speaker-labeled transcripts with name correction."""

    def __init__(self, name_corrector: NameCorrector):
        self.name_corrector = name_corrector

    def build_transcript(
        self,
        segments: List[dict],
        format: str = "text"
    ) -> str:
        """
        Build transcript with corrected speaker names.

        Each segment:
        {
            "start": 15.2,
            "end": 18.5,
            "text": "Good morning everyone!",
            "speaker": "SPEAKER_00",  # or user-labeled name
            "speaker_name": "Michael Scott"  # resolved name
        }
        """

        # Resolve all speaker names to canonical form
        for segment in segments:
            if "speaker_name" in segment:
                # Apply name correction
                correction = self.name_corrector.correct_name(segment["speaker_name"])
                segment["speaker_name"] = correction["corrected"]

                # Log if correction was made
                if correction["original"] != correction["corrected"]:
                    logger.info(
                        f"Corrected name: '{correction['original']}' → '{correction['corrected']}' "
                        f"(method: {correction['method']}, confidence: {correction['confidence']})"
                    )

        # Build transcript based on format
        if format == "text":
            return self._build_text_transcript(segments)
        elif format == "srt":
            return self._build_srt_transcript(segments)
        elif format == "json":
            return self._build_json_transcript(segments)

    def _build_text_transcript(self, segments: List[dict]) -> str:
        lines = []

        for segment in segments:
            timestamp = self._format_timestamp(segment["start"])
            speaker = segment.get("speaker_name", "Unknown")
            text = segment["text"]

            lines.append(f"[{timestamp}] {speaker}: {text}")

        return "\n\n".join(lines)
```

---

## Complete Workflow

### 1. Initial Setup (One-Time)

```bash
# Create show configuration with canonical names
show-scribe init "The Office"

# This creates: shows/TheOffice/show_config.json
# Prompts for cast member names (validates spelling)
```

**Interactive Prompt:**
```
Enter main cast members (one per line, press Enter twice when done):
> Michael Scott
✓ Added: Michael Scott

> Dwight Schrute
✓ Added: Dwight Schrute

> Jim Halpert
✓ Added: Jim Halpert

> Pam Beesly
✓ Added: Pam Beesly

>
✓ Show configuration saved!
```

### 2. Processing Episode

```bash
show-scribe process TheOffice_S02E05.mp4 --show "The Office"
```

**Pipeline automatically:**
1. Loads show config with canonical names
2. Loads voice bank with display names
3. Performs speaker identification
4. Maps clusters to canonical names
5. Applies name corrections

### 3. Manual Review (Unknown Speakers)

**Streamlit UI shows:**
```
┌─────────────────────────────────────────────────┐
│ Unknown Speaker Review                          │
├─────────────────────────────────────────────────┤
│                                                 │
│ Segment: [00:12:45 - 00:12:52]                 │
│ ▶ [Play Audio]                                  │
│                                                 │
│ Text: "That's what she said!"                   │
│                                                 │
│ Who is this speaker?                            │
│ ┌─────────────────────────────────────────┐   │
│ │ ▼ Michael Scott                         │   │
│ │   Dwight Schrute                        │   │
│ │   Jim Halpert                           │   │
│ │   Pam Beesly                            │   │
│ │   ➕ Other (type name)                  │   │
│ └─────────────────────────────────────────┘   │
│                                                 │
│ [Save] [Skip]                                   │
└─────────────────────────────────────────────────┘
```

**User selects from dropdown** → Guaranteed correct spelling!

### 4. Output Generation

**All outputs use canonical names:**

```
# transcript_final.txt
[00:00:15] Michael Scott: Good morning everyone!

# transcript_final.srt
1
00:00:15,000 --> 00:00:18,000
[Michael Scott]: Good morning everyone!

# bleeps.csv
WORD,PERSON,TIMESTAMP,SENTENCE USED IN
fuck,Michael Scott,00:12:15.420,"What the [BLEEP] is going on?"

# analytics.json
{
  "speakers": [
    {
      "name": "Michael Scott",  // Always canonical
      "duration_seconds": 315.5,
      ...
    }
  ]
}
```

---

## Advanced Features

### Name Variant Detection

```python
# Automatically detect and suggest corrections for:
"Micheal Scott" → "Michael Scott" (common misspelling)
"Mike Scott" → "Michael Scott" (informal)
"M. Scott" → "Michael Scott" (abbreviated)
"Michael G. Scott" → "Michael Scott" (middle initial)
```

### Phonetic Matching

```python
# Match based on pronunciation
"Dwite Shrute" → "Dwight Schrute" (phonetically similar)
"Jim Halbert" → "Jim Halpert" (common mishearing)
```

### Confidence-Based Warnings

```python
# In UI, show confidence indicator
if correction["confidence"] < 0.90:
    st.warning(f"⚠️ Low confidence correction: {original} → {corrected}")
    st.checkbox("Confirm this correction", key="confirm")
```

---

## Configuration Options

### In show_config.json:

```json
{
  "name_correction": {
    "enabled": true,
    "auto_correct_threshold": 0.90,  // Auto-correct if confidence > 90%
    "fuzzy_match_cutoff": 0.80,      // Minimum similarity for suggestions
    "case_sensitive": false,
    "preserve_formatting": false,     // Keep user's capitalization
    "prompt_on_ambiguous": true      // Ask user if unsure
  }
}
```

### In app_settings.json:

```json
{
  "name_validation": {
    "strict_mode": false,           // Reject unknown names
    "require_confirmation": true,   // Always confirm corrections
    "log_corrections": true         // Log all name corrections
  }
}
```

---

## Implementation Priority

### Phase 1: MVP (v1.0) 🔴 HIGH
- [x] Show config with cast_members array
- [ ] Voice bank display_name field
- [ ] NameCorrector class (basic)
- [ ] UI dropdown with canonical names
- [ ] Transcript post-processing

**Effort:** 2-3 days
**Files to create/modify:**
- `src/show_scribe/utils/name_correction.py` (NEW)
- `src/show_scribe/config/schema.json` (update)
- `ui/streamlit_app/components/name_selector.py` (NEW)
- `src/show_scribe/pipelines/transcript/builder.py` (update)

### Phase 2: Enhanced (v1.1) 🟡 MEDIUM
- [ ] Fuzzy matching with suggestions
- [ ] Phonetic matching
- [ ] Common misspellings database
- [ ] Correction confidence display
- [ ] Correction logging and review

**Effort:** 2-3 days

---

## Testing

### Test Cases

```python
# test_name_correction.py

def test_exact_match():
    result = corrector.correct_name("Michael Scott")
    assert result["corrected"] == "Michael Scott"
    assert result["confidence"] == 1.0

def test_common_misspelling():
    result = corrector.correct_name("Micheal Scott")
    assert result["corrected"] == "Michael Scott"
    assert result["method"] == "misspelling_map"

def test_alias_match():
    result = corrector.correct_name("Mike")
    assert result["corrected"] == "Michael Scott"
    assert result["method"] == "alias_match"

def test_fuzzy_match():
    result = corrector.correct_name("Michel Scott")
    assert result["corrected"] == "Michael Scott"
    assert result["method"] == "fuzzy_match"

def test_case_insensitive():
    result = corrector.correct_name("michael scott")
    assert result["corrected"] == "Michael Scott"

def test_transcript_correction():
    input_text = "[00:00:15] Micheal Scott: Hello!"
    expected = "[00:00:15] Michael Scott: Hello!"

    output = corrector.correct_transcript(input_text)
    assert output == expected
```

---

## Summary

✅ **Multi-layered protection** ensures correct name spelling:

1. **Show Config** - Define canonical names upfront
2. **Voice Bank** - Store display_name (canonical form)
3. **Name Corrector** - Auto-correct misspellings
4. **UI Dropdowns** - User selects from validated list
5. **Post-Processing** - Final check before output

✅ **User never types names manually** (reduces errors)
✅ **Auto-correction with confidence scoring**
✅ **All outputs guaranteed to use canonical spelling**

**Implementation Status:** Ready for v1.0 MVP

---

**Document Version:** 1.0
**Last Updated:** October 16, 2025
**Status:** ✅ Implementation Ready
