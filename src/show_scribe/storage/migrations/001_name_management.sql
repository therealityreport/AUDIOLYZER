-- Ensure speaker profile columns needed for name management exist.
ALTER TABLE speaker_profiles
    ADD COLUMN IF NOT EXISTS display_name TEXT NOT NULL DEFAULT '';

ALTER TABLE speaker_profiles
    ADD COLUMN IF NOT EXISTS common_aliases TEXT NOT NULL DEFAULT '[]';

ALTER TABLE speaker_profiles
    ADD COLUMN IF NOT EXISTS common_misspellings TEXT NOT NULL DEFAULT '[]';

ALTER TABLE speaker_profiles
    ADD COLUMN IF NOT EXISTS phonetic_spelling TEXT;

UPDATE speaker_profiles
SET display_name = CASE
        WHEN display_name IS NULL OR display_name = '' THEN name
        ELSE display_name
    END
WHERE 1 = 1;

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
