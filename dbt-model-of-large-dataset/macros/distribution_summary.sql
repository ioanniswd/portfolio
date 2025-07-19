{% macro distribution_summary(distance_column) %}
        COUNT(*) AS total_trips,
        AVG({{ distance_column }}) AS avg_trip_distance_km,
        STDDEV_SAMP({{ distance_column }}) AS stddev_trip_distance_km,
        MIN({{ distance_column }}) AS min_trip_distance_km,
        MAX({{ distance_column }}) AS max_trip_distance_km,
        APPROX_QUANTILE({{ distance_column }}, 0.25) AS q1_trip_distance_km,
        APPROX_QUANTILE({{ distance_column }}, 0.5) AS median_trip_distance_km,
        APPROX_QUANTILE({{ distance_column }}, 0.75) AS q3_trip_distance_km
{% endmacro %}
