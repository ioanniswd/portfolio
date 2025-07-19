SELECT
    {{ distribution_summary('trip_distance') }}
FROM {{ ref('stg_taxi_trips') }}
