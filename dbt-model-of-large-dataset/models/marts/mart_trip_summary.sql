SELECT
    'daily_summary' as summary_type,
    COUNT(*) as total_trips
FROM {{ ref('stg_taxi_trips') }}
WHERE trip_distance > 0
  AND fare_amount >= 0
