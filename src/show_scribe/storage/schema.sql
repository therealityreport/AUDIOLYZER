PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS speaker_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    common_aliases TEXT NOT NULL DEFAULT '[]',
    common_misspellings TEXT NOT NULL DEFAULT '[]',
    phonetic_spelling TEXT,
    first_appearance_episode TEXT,
    created_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_segments INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

CREATE TRIGGER IF NOT EXISTS trigger_speaker_profiles_last_updated
AFTER UPDATE ON speaker_profiles
BEGIN
    UPDATE speaker_profiles
    SET last_updated = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS voice_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL,
    embedding_path TEXT NOT NULL,
    source_episode TEXT,
    source_timestamp TEXT,
    audio_sample_path TEXT,
    quality_score REAL,
    created_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (speaker_id) REFERENCES speaker_profiles (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_voice_embeddings_speaker
    ON voice_embeddings (speaker_id);

CREATE TABLE IF NOT EXISTS processing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL UNIQUE,
    episode_path TEXT NOT NULL,
    status TEXT NOT NULL,
    processing_started TIMESTAMP,
    processing_completed TIMESTAMP,
    total_speakers INTEGER,
    auto_identified_speakers INTEGER,
    manual_labeled_speakers INTEGER,
    transcription_confidence REAL,
    diarization_error_rate REAL,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_processing_history_status
    ON processing_history (status, processing_started);

CREATE TABLE IF NOT EXISTS episode_speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    speaker_id INTEGER NOT NULL,
    total_duration_seconds REAL,
    total_word_count INTEGER,
    segment_count INTEGER,
    FOREIGN KEY (speaker_id) REFERENCES speaker_profiles (id) ON DELETE CASCADE,
    FOREIGN KEY (episode_id) REFERENCES processing_history (episode_id) ON DELETE CASCADE,
    UNIQUE (episode_id, speaker_id)
);

CREATE TABLE IF NOT EXISTS sfx_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL CHECK (type IN ('tone', 'mute', 'noise')),
    center_freq_hz REAL,
    bandwidth_hz REAL,
    avg_duration_ms REAL,
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bleep_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('tone', 'mute', 'noise')),
    confidence REAL NOT NULL,
    speaker_cluster_id TEXT,
    person_id INTEGER,
    sentence_text TEXT,
    word_label TEXT NOT NULL DEFAULT '[BLEEP]',
    word_label_source TEXT NOT NULL DEFAULT 'unknown' CHECK (
        word_label_source IN ('manual', 'inferred', 'unknown')
    ),
    word_confidence REAL,
    audio_snippet_path TEXT,
    sfx_profile_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES speaker_profiles (id) ON DELETE SET NULL,
    FOREIGN KEY (sfx_profile_id) REFERENCES sfx_profiles (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_bleep_episode_time
    ON bleep_events (episode_id, start_ms);

CREATE INDEX IF NOT EXISTS idx_bleep_person
    ON bleep_events (person_id);

CREATE TABLE IF NOT EXISTS name_corrections_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_text TEXT NOT NULL,
    corrected_text TEXT NOT NULL,
    method TEXT NOT NULL,
    confidence REAL,
    metadata TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_name_corrections_created
    ON name_corrections_log (created_at);
