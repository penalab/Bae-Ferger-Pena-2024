# Curated Data

This directory contains comma separated files of the data to be included.

## `units_ot.csv` and `units_icx.csv`

List of all the units to be included from each region (OT and ICx).

| Column  | Example      | Description                   |
| ------- | ------------ | ----------------------------- |
| date    | `2023-04-11` | Date of the recording session |
| owl     | 33           | Owl ring number               |
| channel | 5            | Channel number                |

## `srf_dates_ot.csv` and `srf_dates_icx.csv`

Two-column list that maps recording session dates to the correct date where
Spatial Receptive Fields (SRF) have been recorded (without moving electrodes
between days). Separation into two days was necessary for time constrains.

| Column      | Example      | Description                   |
| ----------- | ------------ | ----------------------------- |
| competition | `2023-05-23` | Competition recording session |
| srf         | `2023-05-22` | Related SRF recording session |

