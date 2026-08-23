-- ─────────────────────────────────────────────────────────────────────────────
-- infra/scripts/create_partitions.sql
--
-- Monthly-partition maintenance for the `attendance` table (idempotent).
-- Runs daily at 03:00 UTC via the job-partition-maintenance Container Apps
-- job; mirrors backend/db/partitions.py::ensure_partitions so local dev and
-- production share one behaviour:
--   * a DEFAULT catch-all partition (stray dates can never fail an insert)
--   * the current month + the next 4 months pre-created
--
-- CREATE TABLE IF NOT EXISTS ... PARTITION OF is safe to re-run every day;
-- existing partitions are skipped, new months appear as time advances.
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
DECLARE
    m date := date_trunc('month', CURRENT_DATE)::date;
    i int;
    part_name text;
    start_date date;
    end_date date;
BEGIN
    -- Catch-all partition: guarantees inserts never hit a missing partition.
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF attendance DEFAULT',
        'attendance_default'
    );

    FOR i IN 0..4 LOOP
        start_date := m + make_interval(months => i);
        end_date   := start_date + interval '1 month';
        part_name  := 'attendance_' || to_char(start_date, 'YYYY_MM');
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF attendance '
            'FOR VALUES FROM (%L) TO (%L)',
            part_name, start_date, end_date::date
        );
    END LOOP;
END $$;
