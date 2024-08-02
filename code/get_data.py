# -*- coding: utf-8 -*-
"""
This code loads data from multiple sources for 
"""

import pandas as pd
import os
import re
from dotenv import load_dotenv
import time
import pickle
import googlemaps

import logging

def setup_logging():
    """Configure logging."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f'logs/get_data-{timestamp}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_filename, filemode='w')

def get_metro_pop():
    """
    Load CSV and return a dateframe of metro populations.
    
    The data was pulled from the Census Bureau's website and saved locally as a CSV file.
    This function loads the CSV into a dataframe and performs some cleaning.
    
    Returns:
        pd.DataFrame: Cleaned dataframe containing metro population data.
    """
    # Load CSV file into a DataFrame
    # Define the custom header names
    header_names = ['MetroArea', 'Base_pop', '2020_pop', '2021_pop', '2022_pop', '2023_pop']
    full_pop_df = pd.read_csv('../data/cbsa-met-est2023-pop.csv',skiprows=6, nrows=424, names=header_names)
    
    # Only need the most recent population count
    pop_df = full_pop_df[['MetroArea','2023_pop']].rename(columns={'2023_pop':'Population'})
    
    # Clean dataframe
    # Strip the '.' at the beginning of the 'MetroArea' column
    pop_df.loc[:,'MetroArea'] = pop_df['MetroArea'].str.lstrip('.')

    # Drop divisions of the metro areas
    pop_df = pop_df[~pop_df['MetroArea'].str.endswith('Division')].reset_index(drop=True)

    # Remove "Metro Area" from the city
    pop_df.loc[:,'MetroArea'] = pop_df['MetroArea'].str.replace(r'Metro Area', '').str.rstrip()

    # Grab the "main" city from the metro area for using for coordinates (listed first)
    # For example, the DC metro area includes northern Virginia and parts of Maryland 
    #   (Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area) but we just take DC for our calculations
    
    mainCity = pop_df['MetroArea'].str.split(',').str.get(0).str.split(r'[-/]').str.get(0)
    mainState = pop_df['MetroArea'].str.split(',').str.get(1).str.split('-').str.get(0)
    pop_df['MainCity'] = mainCity + ',' + mainState.str.rstrip()
    
    # Manually fix cities some city names
    pop_df.loc[pop_df['MainCity']=='Winston, NC','MainCity'] = 'Winston-Salem, NC'
    pop_df.loc[pop_df['MainCity']=='Barnstable Town, MA','MainCity'] = 'Barnstable, MA'
    pop_df.loc[pop_df['MainCity']=='Amherst Town, MA','MainCity'] = 'Amherst, MA'
    pop_df.loc[pop_df['MainCity']=='Urban Honolulu, HI','MainCity'] = 'Honolulu, HI'
    
    # Convert the population to int
    pop_df.loc[:,'Population'] = pop_df['Population'].str.replace(',','').astype(int)
    
    return pop_df

def get_geos(cities, logger, max_retries=3, overwrite=False):
    """
    Call the Google Maps Geocoder API to get lat/lon geo coordinates.
    
    This function collects the geo coordinates for the main city of the MSA.

    Args:
        cities (list of str): List of city names.
        logger (logging.Logger): Logger instance.
        max_retries: Maximum number of retries if API call fails (default is 3).
        overwrite (bool): Whether to rerun the API query if data already exists.

    Returns:
        pd.DataFrame: Updated population dataframe containing geo-coords for the MSA.
    """
    # Google provides $300 credits of free service and querying Google Maps can use that up, so
    # avoid rerunning the API calls if the data is saved as a pickle
    geo_pickle = '../data/pickled/geos_df.pickle'
    if os.path.isfile(geo_pickle) and not overwrite:
        try:
            with open(geo_pickle, 'rb') as file:
                geo_df = pickle.load(file)
            logger.info("Loading geo data from pickled file.")
            return geo_df
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            pass
        
    # Load API key from env 
    load_dotenv()
    gm_api = os.getenv('GOOGLE_MAPS_API_KEY')
    if not gm_api:
        raise ValueError("API Key not found. Please set the GOOGLE_MAPS_API_KEY environment variable.")

    gmaps = googlemaps.Client(key=gm_api)
    
    geos_list = []
    
    for i,city in enumerate(cities):
        try:
            # Call the geocoder for the city
            result = gmaps.geocode(city)
            
            # Will look into these scenarios after
            if len(result)==0:
                logger.warning(f"No results found for city: {city}")
                continue
            else:
                # Parse just lat/lng fields
                geos_list.append({
                        'MainCity': city,
                        'lat': result[0]['geometry']['location']['lat'],
                        'lng': result[0]['geometry']['location']['lng']
                    })
                
            # Pause to respect rate limiting
            time.sleep(0.1)
            
            if i%25==0:
                print(f"Completed {i} out of {len(cities)} cities.")
                logger.info(f"Completed {i} out of {len(cities)} cities.")
                
                # Save partial results
                with open(f'../data/pickled/temp/geo_results_partial_{i}.pickle', 'wb') as file:
                    pickle.dump(geos_list, file)
                
            
        except googlemaps.exceptions.ApiError as e:
            logger.error(f"API error for city {city}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for city {city}: {e}")
            
    geo_df = pd.DataFrame(geos_list)
   
    # Save to pickle
    with open(f'../data/pickled/geos_df.pickle', 'wb') as file:
        pickle.dump(geo_df, file)
        
    return geo_df

def query_google_dm(origins, destinations, logger, max_retries=3):
    """
    Call the Google Maps Distance Matrix API to get distance and duration.
    
    This function collects the distance (meters) and duration (seconds) between origins and destinations
    based on the current road network and average time-independent traffic conditions.

    Args:
        origins: List of string values of origin cities.
        destinations: List of string values of destination cities to calculate.
        logger (logging.Logger): Logger instance.
        max_retries: Maximum number of retries if API call fails (default is 3).

    Returns:
        list of dict: List of dictionaries with results for city pairs, including distance and duration.
    """
    
    # Load API key from env 
    load_dotenv()
    gm_api = os.getenv('GOOGLE_MAPS_API_KEY')
    if not gm_api:
        raise ValueError("API Key not found. Please set the GOOGLE_MAPS_API_KEY environment variable.")


    gmaps = googlemaps.Client(key=gm_api)

    attempt = 0
    while attempt < max_retries:
        try:
            response = gmaps.distance_matrix(origins, destinations)
            break  # Break out of the loop if successful
        except googlemaps.exceptions.ApiError as e:
            # Handle different types of errors
            error_message = str(e)
            if "OVER_QUERY_LIMIT" in error_message:
                logger.warning("Exceeded query limit. Waiting before retrying...")
                time.sleep(10)  # Wait before retrying
                attempt += 1
                continue  # Retry
            elif "REQUEST_DENIED" in error_message:
                logger.error("Request denied by the server.")
                return []  # Return
            elif "INVALID_REQUEST" in error_message:
                logger.error("Invalid request. Check your input parameters.")
                return []  # Return
            elif "MAX_ELEMENTS_EXCEEDED" in error_message or "MAX_DIMENSIONS_EXCEEDED" in error_message:
                logger.error("Exceeded maximum elements or dimensions per query.")
                return []  # Return 
            elif "OVER_DAILY_LIMIT" in error_message:
                logger.error("Over daily limit. Check billing status or usage cap.")
                return []  # Return 
            elif "UNKNOWN_ERROR" in error_message:
                logger.warning("Unknown error. Retrying.")
                time.sleep(2)
                attempt += 1
                continue  # Retry
            else:
                logger.warning(f"Unhandled error: {error_message}")
                attempt += 1
                continue  # Retry
        
        except Exception as e:
            logger.warning(f"Error: {e}")
            attempt += 1
            if attempt < max_retries:
                logger.info(f"Retrying... Attempt {attempt} of {max_retries}")
                continue
            else:
                logger.error(f"Failed after {attempt} attempts.")
                return []  # Return empty list if max retries reached

    results = []
    # Extract origins and destinations
    origins = response['origin_addresses']
    destinations = response['destination_addresses']

    # Loop through the rows in the response
    for i, row in enumerate(response['rows']):
        origin = origins[i]
        for j, element in enumerate(row['elements']):
            dest = destinations[j]

            #skip if origin and destination are the same
            if origin == dest:
                continue
                
            if element['status'] == 'OK':
                distance = element['distance']['value']
                duration = element['duration']['value']
            else:
                distance = None
                duration = None
                logger.warning(f"No valid response for {origin} to {dest}: {element['status']}")
            
            results.append({
                'Origin': origin,
                'Destination': dest,
                'Distance_meters': distance,
                'Duration_seconds': duration
            })
    
    return results

def get_city_distances(cities, overwrite=False, logger=None):
    """
    Batch process cities to get distances and durations using Google Maps Distance Matrix API.
    
    Google Maps Distance Matrix requests are limited to a maximum of 100 elements,
    where the number of origins times the number of destinations defines the number of elements.
    This function batches the cities list into origins and destinations lists of 10 elements
    to submit to the API request.

    Args:
        cities (list of str): List of city names.
        overwrite (bool): Whether to rerun the API query if data already exists.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: DataFrame containing distance and duration between cities.
    """
    
    # Google provides $300 credits of free service and querying Google Maps can use that up, so
    # avoid rerunning the API calls if the data is saved as a pickle
    dist_pickle = '../data/pickled/distance_df.pickle'
    if os.path.isfile(dist_pickle) and not overwrite:
        try:
            with open(dist_pickle, 'rb') as file:
                dist_df = pickle.load(file)
            logger.info("Loading data from pickled file.")
            return dist_df
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            pass
        
    batch_size = 10
    dm_results = []
    
    # Track progress
    total_batches = (len(cities) // batch_size + (len(cities) % batch_size > 0)) ** 2
    completed_batches = 0
    
    logger.info(f"Starting Google Maps Distance Matrix API calls for {total_batches} batches.")
    
    # Create pairs of batches
    for i in range(0, len(cities), batch_size):
        for j in range(0, len(cities), batch_size):
            origins = cities[i:i + batch_size]
            destinations = cities[j:j + batch_size]

            # get distance matrix
            try:
                batch_results = query_google_dm(origins, destinations, logger)
                # add batch results to list for all dm results
                dm_results += batch_results
            except Exception as e:
                logger.error(f"Error processing batch {i}-{j}: {e}")
                # Save partial results
                with open(f'../data/pickled/dm_results_partial_{i}_{j}.pickle', 'wb') as file:
                    pickle.dump(dm_results, file)
                continue

            completed_batches += 1
            if completed_batches%100 == 0 or completed_batches == total_batches:
                logger.info(f"Completed {completed_batches} out of {total_batches} batches")
                print(f"Completed {completed_batches} out of {total_batches} batches")
            
            time.sleep(1)  # Sleep to avoid hitting rate limits
            
        # Save partial results
        with open(f'../data/pickled/dm_results_partial_{i}_{j}.pickle', 'wb') as file:
            pickle.dump(dm_results, file)

    # Convert the list of results to a DataFrame
    dist_df = pd.DataFrame(dm_results)
    
    # Clean the DataFrame to remove 'USA' from the city name
    dist_df.loc[:,'Origin'] = dist_df['Origin'].str.replace(', USA','')
    dist_df.loc[:,'Destination'] = dist_df['Destination'].str.replace(', USA','')
    
    # Clean the DataFrame to remove zipcode from the city name
    dist_df.loc[:,'Origin'] = dist_df['Origin'].str.replace(r'\d+','', regex=True)
    dist_df.loc[:,'Destination'] = dist_df['Destination'].str.replace(r'\d+','', regex=True)
    
    # Save distance in miles
    dist_df['Distance_miles'] = dist_df['Distance_meters'] * 0.000621371
    dist_df['Duration_minutes'] = round(dist_df['Duration_seconds'] / 60)
    
    dist_df = dist_df[['Origin','Destination','Distance_meters','Distance_miles','Duration_seconds','Duration_minutes']]
    
    # Save the DataFrame as a pickle file
    with open(dist_pickle, 'wb') as file:
        pickle.dump(dist_df, file)
    
    return dist_df  
 
def get_airport_codes(msa_df):
    """
    Match airport IATA codes to metropolitan area.
    
    This function loads a CSV of airport codes pulled from the Bureau of Transportation's website
    and merges it with the MSAs.

    Args:
        msa_df (pd.DataFrame): DataFrame with metropolitan statistical area (MSA) names and population.

    Returns:
        pd.DataFrame: DataFrame with MSA and airport IATA codes.
    """
    # Load CSV file into a DataFrame
    airports_df = pd.read_csv('../data/BTS_Airports.csv')
    airports_df[['City', 'State']] = airports_df['City'].str.split(', ', n=1, expand=True)
    airports_df = airports_df.loc[airports_df['State'].str.len()==2]
    
    iata_list = []

    for i, row in msa_df.iterrows():
        maincity = row['MainCity']
        printinfo=False
        
        states = re.split(r'[\-/]', row['MetroArea'].split(',')[-1].strip())
        cities = re.split(r'[\-/]', row['MetroArea'].split(',')[0].strip())
        
        city_match = pd.DataFrame()

        for city, state in [(c, s) for c in cities for s in states if c!='']:
            cm = airports_df.loc[((airports_df['City'].str.contains(city)) | airports_df['Airport'].str.contains(city)) & (airports_df['State'].str.contains(state))]
            city_match = pd.concat([city_match, cm])

        if len(city_match) == 0:
            found = False
            # Check if airport city list in MSA (but not main)
            # Need to do reverse lookup
            for j, r in airports_df.iterrows():
                ap_city = r['City'].strip()
                ap_state = r['State'].strip()
                if '/' in ap_city:
                    apc_list = ap_city.split('/')
                    for c in apc_list:
                        ap_city_match = (c in row['MetroArea']) & (ap_state in row['MetroArea'])
                        if ap_city_match:
                            iata_list.append({
                                'AirportCode': r['Code'],
                                'MetroArea': row['MetroArea'],
                                'MainCity': maincity
                            })
                            found = True
                else:
                    ap_city_match = (ap_city in row['MetroArea']) & (ap_state in row['MetroArea'])
                    if ap_city_match:
                        iata_list.append({
                            'AirportCode': r['Code'],
                            'MetroArea': row['MetroArea'],
                            'MainCity': maincity
                        })
                        found = True

            # if not found:
            #     print(maincity)
            #     print('***********************')
        else:   
            # MSA could have multiple airports
            for k, m in city_match.iterrows():
                iata_list.append({
                    'AirportCode': m['Code'],
                    'MetroArea': row['MetroArea'],
                    'MainCity': maincity
                    })
            
    # Manually found airport/MSA pairs
    iata_list.append({'AirportCode': 'HYA', 'MetroArea': 'Barnstable Town, MA', 'MainCity': 'Barnstable Town, MA'})
    iata_list.append({'AirportCode': 'CWA', 'MetroArea': 'Wausau, WI', 'MainCity': 'Wausau, WI',})
   
    # Convert the list of results to a DataFrame
    iata_df = pd.DataFrame(iata_list).drop_duplicates()
    
    return iata_df
 
def get_flight_data(iata_df):
    """
    Get flight data for metro areas
    
    This function loads a CSV of domestic non-stop flight data reported by both U.S. and foreign air carriers 
    and parses it for passenger flight information relevant to MSAs.

    Args:
        iata_df (pd.DataFrame): DataFrame with MSA and airport codes.

    Returns:
        pd.DataFrame: DataFrame with passenger flight information.
    """
    # Load CSV file into a DataFrame
    segments_df = pd.read_csv('../data/T_T100D_SEGMENT_ALL_CARRIER.csv')
    
    # We are only interested in passenger flights, so remove all others (AIRCRAFT_CONFIG=1)
    pass_flight_df = segments_df.loc[segments_df['AIRCRAFT_CONFIG']==1]
    
    # We aren't interested in helicopter flights
    pass_flight_df = pass_flight_df.loc[pass_flight_df['AIRCRAFT_GROUP']!=3]
    
    # Only keep flights with origin and destination in a MSA
    msa_airports = iata_df['AirportCode']
    msa_flight_df = pass_flight_df.loc[pass_flight_df['ORIGIN'].isin(msa_airports) & pass_flight_df['DEST'].isin(msa_airports)]
    msa_flight_df = msa_flight_df.loc[msa_flight_df['ORIGIN']!=msa_flight_df['DEST']]
    
    # Remove flights with no passengers
    msa_flight_df = msa_flight_df.loc[msa_flight_df['PASSENGERS']!=0]
    
    # Add City_Pair based on MSA dataframe
    origin_mc = msa_flight_df['ORIGIN'].apply(lambda x: iata_df.loc[iata_df['AirportCode'] == x, 'MainCity'].values[0])
    dest_mc = msa_flight_df['DEST'].apply(lambda x: iata_df.loc[iata_df['AirportCode'] == x, 'MainCity'].values[0])
    msa_flight_df['OriginMSA'] = origin_mc
    msa_flight_df['DestMSA'] = dest_mc
    msa_flight_df['CityPair'] = [tuple(sorted(pair)) for pair in zip(origin_mc, dest_mc)]
    
    # Not interested in flights within an MSA
    msa_flight_df = msa_flight_df.loc[msa_flight_df['CityPair'].apply(lambda x: x[0]==x[1])==False].reset_index(drop=True)
    
    # Save the DataFrame as a pickle file
    full_pickle = '../data/pickled/flights_df_all_fields.pickle'
    with open(full_pickle, 'wb') as file:
        pickle.dump(msa_flight_df, file)
        
    filtered_df = msa_flight_df[['DEPARTURES_SCHEDULED','DEPARTURES_PERFORMED','SEATS','PASSENGERS','DISTANCE','RAMP_TO_RAMP',
                                  'AIR_TIME','UNIQUE_CARRIER','AIRLINE_ID','UNIQUE_CARRIER_NAME','REGION','CARRIER_GROUP_NEW',
                                  'ORIGIN','ORIGIN_CITY_NAME','ORIGIN_CITY_MARKET_ID','DEST','DEST_CITY_NAME','DEST_CITY_MARKET_ID',
                                  'AIRCRAFT_GROUP','AIRCRAFT_TYPE','MONTH','YEAR','DISTANCE_GROUP','CLASS','CityPair']]
    # Save the DataFrame as a pickle file
    flight_pickle = '../data/pickled/flights_df.pickle'
    with open(flight_pickle, 'wb') as file:
        pickle.dump(filtered_df, file)
        
    return(filtered_df)

def get_gdp():
    """
    Get GDP data for each Metropolitan Statistical Area.
    
    This function loads a CSV of GDP by MSA from the Bureau of Economic Analysis' website.

    Returns:
        pd.DataFrame: DataFrame with MSA and GDP in thousands of dollars.
    """
    
    gdp_df = pd.read_csv('../data/CAGDP2_MSA_2017_2022.csv')
    
    # Remove empty rows
    gdp_df = gdp_df.loc[gdp_df['GeoName'].notnull()]
    
    # Clean MSA names
    gdp_df.loc[:, 'GeoName'] = gdp_df['GeoName'].str.replace(r'(Metropolitan Statistical Area)', '').str.strip()
    gdp_df.loc[:, 'GeoName'] = gdp_df['GeoName'].str.replace('*', '').str.strip()
    
    # Get only total GDP
    gdp_df = gdp_df.loc[gdp_df['LineCode']==1]
    
    # Rename DataFrame columns to prepare to merge with population
    gdp_df = gdp_df[['GeoName','2022']].rename(columns={'GeoName':'MetroArea', '2022':'GDP_thousands_dollars'})
    mainCity = gdp_df['MetroArea'].str.split(',').str.get(0).str.split(r'[-/]').str.get(0)
    mainState = gdp_df['MetroArea'].str.split(',').str.get(1).str.split('-').str.get(0)
    gdp_df['MainCity'] = mainCity + ',' + mainState.str.rstrip()
    
    # Fix some mismatched names to match pop_df
    # Manually fix cities some city names
    gdp_df.loc[gdp_df['MainCity']=='Winston, NC','MainCity'] = 'Winston-Salem, NC'
    gdp_df.loc[gdp_df['MainCity']=='Poughkeepsie, NY', 'MainCity'] = 'Kiryas Joel, NY'
    gdp_df.loc[gdp_df['MainCity']=='California, MD', 'MainCity'] = 'Lexington Park, MD'
    gdp_df.loc[gdp_df['MainCity']=='The Villages, FL', 'MainCity'] = 'Wildwood, FL'
    gdp_df.loc[gdp_df['MainCity']=='Barnstable Town, MA', 'MainCity'] = 'Barnstable, MA'
    gdp_df.loc[gdp_df['MainCity']=='Urban Honolulu, HI','MainCity'] = 'Honolulu, HI'
    
    # Convert GDP to numeric value
    gdp_df['GDP_thousands_dollars'] = pd.to_numeric(gdp_df['GDP_thousands_dollars'], errors='coerce')
    
    return(gdp_df)
    
def make_msa_df(pop_df, geo_df, gdp_df, tti_df):
    """
    Combine data for overall information for MSAs
    
    This function combines population, geo coordinates and GDP data into a single dataframe.

    Args:
        pop_df (pd.DataFrame): DataFrame with metropolitan statistical area (MSA) names and population.
        geo_df (pd.DataFrame): DataFrame with metropolitan statistical area (MSA) names and geo-coords.
        gdp_df (pd.DataFrame): DataFrame with metropolitan statistical area (MSA) names and GDP.
        tti_df (pd.DataFrame): DataFrame with metropolitan statistical area (MSA) names and TTI data.

    Returns:
        pd.DataFrame: DataFrame with all MSA data.
    """
    
    msa_df = pd.merge(pop_df, geo_df, on='MainCity', how='left')
    
    msa_df = pd.merge(msa_df, gdp_df[['MainCity','GDP_thousands_dollars']], on='MainCity', how='left')
    
    msa_df = pd.merge(msa_df, tti_df[['MSAmatch','TravelTimeIndexValue']], left_on='MainCity', right_on='MSAmatch', how='left')
    msa_df.drop(columns=['MSAmatch'], inplace=True)
    
    # The MetroArea values are the ones from the population data (2023). 
    # Note that the GDP data was from 2022 and I found that there are
    # a few new MSAs.
    msa_df = msa_df.loc[msa_df['GDP_thousands_dollars'].notnull()]
    
    # Save the DataFrame as a pickle file
    msa_pickle = '../data/pickled/msa_df.pickle'
    with open(msa_pickle, 'wb') as file:
        pickle.dump(msa_df, file)
        
    return msa_df
    
def get_tti(pop_df):
    header_names = ['AreaGroup', 'MetroArea', 'StateCode', 'PopulationGroup', 'Year', 'Population_thousands', 
                'PopulationRank', 'AutoCommuters','FreewayDailyVehicle_miles_thousands','ArterialStreetDailyVehicle_miles_thousands',
                'ValueOfTime','CommercialValueOfTime','AverageStateGasCost','AverageStateDieselCost','CongestedTravel',
                'CongestedSystem','NumberOfRushHours','TotalGallons_thousands','TotalGallonsRank','GallonsPerAutoCommuter',
                'GallonsPerAutoCommuterRank','TotalDelay','TotalDelayRank','DelayPerAutoCommuter','DelayPerAutoCommuterRank',
                'TravelTimeIndexValue','TravelTimeIndexRank','CommuterStressIndexValue','CommuterStressIndexRank',
                'FreewayPlanningTimeIndexValue','FreewayPlanningTimeIndexRank','AnnualCongestionCostTotalDollars_millions',
                'AnnualCongestionCostRank','AnnualCongestionCostPerAutoCommuter','AnnualCongestionCostPerAutoCommuterRank',
                'TruckTotalDelay_thousands','TruckTotalDelayRank','TruckTotalGallons','TruckTotalGallonsRank','TruckAnnualCost_millions',
                'TruckAnnualCostRank','AnnualCO2ExcessDueCongestion_tons','AnnualCO2ExcessDueCongestionRank',
                'AnnualCO2ExcessDueAllTravel_tons','AnnualCO2ExcessDueAllTravelRank','AnnualCO2ExcessTruckCongestion_tons',
                'AnnualCO2ExcessTruckCongestionRank','AnnualCO2ExcessTruckTravel_tons','AnnualCO2ExcessTruckTravelRank',
                'PopulationRankAll','WastedFuelRank','WastedFuelPerCommuterRank','AnnualDelayRank','DelayPerCommuterRank',
                'TravelTimeIndexRankAll','AnnualCongCostRank','CostPerCommuterRank','AnnualTruckDelayRank','WastedFuelTrucksRank',
                'TruckCongCostRank']
    tti_df = pd.read_excel('../data/complete-data-2023-umr-by-tti.xlsx',skiprows=4, names=header_names)
    
    tti_map = {}
    for i, row in pop_df.iterrows():
        found = False
        maincity = row['MainCity']
        
        states = re.split(r'[\-/]', row['MetroArea'].split(',')[-1].strip())
        cities = re.split(r'[\-/]', row['MetroArea'].split(',')[0].strip())

        for city, state in [(c, s) for c in cities for s in states if c!='']:
            cm = tti_df.loc[(tti_df['MetroArea'].str.contains(city)) & (tti_df['MetroArea'].str.contains(state))]
            if len(cm)>0:
                found=True
                tti_map[cm['MetroArea'].tolist()[0]] = maincity
                break
                

        if not found:
            for ma in list(set(tti_df['MetroArea'].tolist())):
                last_space_index = ma.rfind(' ')
                itt_city = re.split(r'[\-/]', ma[:last_space_index])
                itt_state = re.split(r'[\-/]', ma[last_space_index + 1:])
                
                for city, state in [(c, s) for c in itt_city for s in itt_state if c!='']:
                    cm = (city in row['MetroArea']) & (state in row['MetroArea'])
                    if cm:
                        found=True
                        tti_map[ma] = maincity
                        break
                if found:
                    break
        # if not found:
        #     print(maincity)
        
    tti_df['MSAmatch'] = tti_df['MetroArea'].map(tti_map)
    
    tti_df = tti_df.loc[tti_df['Year']==2022]
    
     # Save the DataFrame as a pickle file
    tti_pickle = '../data/pickled/tti_df.pickle'
    with open(tti_pickle, 'wb') as file:
        pickle.dump(tti_df, file)
        
    return tti_df
    
def main():
    """Main function to compile data."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get population data for metro areas
    pop_df = get_metro_pop()
    logger.info("Completed processing population CSV.")
    
    # Get GDP for the metro areas
    gdp_df = get_gdp()
    logger.info("Completed processing GDP data.")
    
    # Get geo-coords for cities
    cities = pop_df['MainCity'].tolist()
    geo_df = get_geos(cities, logger=logger)
    logger.info("Completed API calls to Google Maps Geocoder for coordinates.")
    
    # Get TTI data
    tti_df = get_tti(pop_df)
    logger.info("Completed processing tti data.")
    
    # Combine pop_df, geo_df and gdp_df and tti_dfmake a single msa_df file
    msa_df = make_msa_df(pop_df,geo_df,gdp_df,tti_df)
    logger.info("Completed merging MSA data.")
    
    # Get distance and duration 
    cities = msa_df['MainCity'].tolist()
    dist_df = get_city_distances(cities, logger=logger)
    logger.info("Completed API calls to Google Maps Distance Matrix for distances.")
    
    # Get airport codes for metro areas
    iata_df = get_airport_codes(msa_df)
    logger.info("Completed processing airport IATA codes.")
    
    # Get flight info for metro areas
    flights_df = get_flight_data(iata_df)
    logger.info("Completed processing flight data.")
    
    
    

if __name__ == "__main__":
    main()
