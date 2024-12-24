-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()),
    accuracy DOUBLE PRECISION NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    source VARCHAR(50) NOT NULL
);

-- Add indexes
CREATE INDEX idx_model_metrics_created_at ON model_metrics(created_at);
CREATE INDEX idx_model_metrics_source ON model_metrics(source); 