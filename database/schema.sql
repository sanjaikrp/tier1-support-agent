-- database/schema.sql
-- Defines the structure of the historical ticket database.
-- This file runs automatically when the PostgreSQL Docker container starts.
-- The Researcher Agent queries this table to find similar past tickets.

-- Drop the table if it already exists (makes re-runs clean and repeatable)
DROP TABLE IF EXISTS tickets;

-- Create the tickets table
CREATE TABLE tickets (
    id                      SERIAL PRIMARY KEY,        -- auto-incrementing unique ID
    category                VARCHAR(20)  NOT NULL,     -- 'bug', 'config', or 'access'
    severity                VARCHAR(20)  NOT NULL,     -- 'low','medium','high','critical'
    title                   TEXT         NOT NULL,     -- short ticket headline
    description             TEXT         NOT NULL,     -- full problem description
    root_cause              TEXT         NOT NULL,     -- the diagnosed root cause
    resolution              TEXT         NOT NULL,     -- how it was fixed
    file_path               TEXT,                      -- relevant source file (bugs/config)
    created_at              TIMESTAMP    DEFAULT NOW(),-- when the ticket was raised
    resolved_at             TIMESTAMP,                 -- when it was resolved
    resolution_time_minutes INT                        -- time taken to resolve
);

-- Create an index on category to speed up the Researcher Agent's filtered queries
CREATE INDEX idx_tickets_category ON tickets (category);

-- Confirmation message when the schema runs successfully
DO $$
BEGIN
    RAISE NOTICE 'tickets table created successfully.';
END $$;
