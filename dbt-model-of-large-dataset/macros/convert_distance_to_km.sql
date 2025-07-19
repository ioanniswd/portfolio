{% macro convert_distance_to_km(distance_column) %}
    ROUND({{ distance_column }} * 1.60934, 2)
{% endmacro %}
