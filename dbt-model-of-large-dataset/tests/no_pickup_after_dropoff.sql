SELECT *
FROM {{ ref('stg_taxi_trips') }}
WHERE pickup_datetime > dropoff_datetime
