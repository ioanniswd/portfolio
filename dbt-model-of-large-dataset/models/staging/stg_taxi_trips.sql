with source as (
    select *
    from {{ source('source_data', 'taxi_trips') }}
),

cleaned as (
    select
        cast("VendorID" as integer) as vendor_id,
        STRPTIME("tpep_pickup_datetime", '%m/%d/%Y %I:%M:%S %p') AS pickup_datetime,
        STRPTIME("tpep_dropoff_datetime", '%m/%d/%Y %I:%M:%S %p') AS dropoff_datetime,
        cast("passenger_count" as integer) as passenger_count,
        nullif(replace("trip_distance", ',', ''), '')::numeric(10, 2) as trip_distance,
        CASE
            WHEN "RatecodeID" == '99' THEN NULL
            ELSE cast("RatecodeID" as integer)
        END AS rate_code_id,
        cast("store_and_fwd_flag" as varchar) as store_and_fwd_flag,
        cast("PULocationID" as integer) as pickup_location_id,
        cast("DOLocationID" as integer) as dropoff_location_id,
        cast("payment_type" as integer) as payment_type,
        nullif(replace("fare_amount", ',', ''), '')::numeric(10, 2) as fare_amount,
        nullif("extra", '')::numeric(10, 2) as extra,
        nullif("mta_tax", '')::numeric(10, 2) as mta_tax,
        nullif("tip_amount", '')::numeric(10, 2) as tip_amount,
        nullif("tolls_amount", '')::numeric(10, 2) as tolls_amount,
        nullif("improvement_surcharge", '')::numeric(10, 2) as improvement_surcharge,
        nullif("total_amount", '')::numeric(10, 2) as total_amount
    from source
    WHERE pickup_datetime <= dropoff_datetime
)

select *
from cleaned
