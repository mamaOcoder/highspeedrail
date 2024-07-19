# High-Speed Passenger Rail Analysis

## Notes

### Running the get_data.py file
The get_data.py file calls the Google Maps Distance Matrix API. This requires an API key to be saved in a '.env' file. To obtain a key, you must first have a Google Developers account and create a project on the Google Cloud Console. See this [Getting Started](https://developers.google.com/maps/get-started) tutorial for help.

## Data Sources
- Metropolitan Statistical Area Populations: [US Census Bureau MSA Population Totals](https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-metro-and-micro-statistical-areas.html)
- IATA codes: [Bureau of Transportation Statistics World Airport Codes List](https://www.bts.gov/topics/airlines-and-airports/world-airport-codes)
- Flight Data:
    - Passenger Counts: [Bureau of Transportation Statistics Domestic Flights](https://www.transtats.bts.gov/TableInfo.asp?gnoyr_VQ=GEE&QO_fu146_anzr=Nv4%20Pn44vr45&V0s1_b0yB=D)
    