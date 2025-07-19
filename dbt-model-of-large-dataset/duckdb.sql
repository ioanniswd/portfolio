-- CREATE SCHEMA main;
CREATE TABLE main.taxi_trips AS
SELECT *
FROM read_csv_auto('data/taxi_trips.csv', AUTO_DETECT=TRUE, SAMPLE_SIZE=-1, ALL_VARCHAR=TRUE);

SELECT COUNT(*) FROM main.taxi_trips;
-- Exit
.quit

