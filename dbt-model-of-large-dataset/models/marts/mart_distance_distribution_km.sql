WITH trip_distance_km AS (
    SELECT
        *,
        {{ convert_distance_to_km('trip_distance') }} AS trip_distance_km
    FROM {{ ref('stg_taxi_trips') }}
)

SELECT
    {{ distribution_summary('trip_distance_km') }}
FROM trip_distance_km
