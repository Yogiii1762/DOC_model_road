# file: Doc_Model.py
#Description : This program reads an input file to extract latitude and longitude data, which it sends to the HERE API to retrieve route information. It then processes the JSON response to obtain necessary attributes, calculates the distance between consecutive points, and saves the data to a CSV file. Additionally, the program generates a report and a graph of the vehicle's path.It cleans the data automatically.
# author: Yogeswaran Amsavalli 
# version: 3.2.2
# date: 2024-05-09

#Import the necessary libraries.
import json
from jsonpath_ng import jsonpath, parse
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2, asin
from pyproj import Geod
from haversine import haversine
import matplotlib.pyplot as plt
import os
import requests


def process_file(filepath, output_directory):

#importing Json file from API_call (Replace with your own API key)
    url = "https://routematching.hereapi.com/v8/match/routelinks"
    api_key = "Replcae with your HERE Maps API Key"


    #To access single file from the computer: Add the filepath here


    with open(filepath, "r") as file:
        lat_lon_data = file.read().strip()

    headers = {
        "Content-Type": "application/json"
    }

    params = {
        "apikey": api_key,
        "mode": "fastest;truck;traffic:disabled",
        "routeMatch": "1",
        "attributes": ["ADAS_ATTRIB_FCn(*)", "APPLICABLE_SPEED_LIMIT(*)", "TRAFFIC_SIGN_FCn(*)", "TRAFFIC_PATTERN_FCn(*)", "ARCHIVED_WEATHER(*)"]
    }

    response = requests.post(url, headers=headers, params=params, data=lat_lon_data)

    if response.status_code == 200:
        data1= response.json()
        print("Getting Response from API Successful")
    else:
        print("Error:", response.status_code, response.text)


    #This Functiuon reads the coordinates from the file and returns the start and end coordinates for calculation
    
    def read_coordinates(filepath):
        with open(filepath, "r") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]  # Skip empty lines

        # Skip the first line (header)
        lines = lines[1:]

        # Initialize start and end coordinates
        start_lat, start_lon, end_lat, end_lon = None, None, None, None

        # Iterate over the lines
        for line in lines:
            # Replace ", " with "," and split the line into latitude and longitude
            try:
                lat, lon = map(float, line.replace(", ", ",", ).split(","))
                # If this is the first valid line, set the start coordinates
                if start_lat is None and start_lon is None:
                    start_lat, start_lon = lat, lon
                # Update the end coordinates
                end_lat, end_lon = lat, lon
            except ValueError:
                # If the line could not be split into two floats, skip it
                continue

        return start_lat, start_lon, end_lat, end_lon

    start_lat, start_lon, end_lat, end_lon = read_coordinates(filepath)

    
    #Uncomment the below code to read the JSON file from the computer and specify the path of the file in the filePath1 variable: 
    
    # filePath1 = "/home/yogi1762/MScThesis_dOC-generator_2024/MScThesis_dOC-generator_2024/StartupLibrary/Test_Responses/Oslo.json"
    # data1= data1
    #Extracting the data from the JSON file:
    # with open(filePath1, 'r') as file:
    #     data1 = json.load(file)
    # start_lat,start_lon = add start coordinates here
    # end_lat,end_lon = add end coordinates here

    # Exrtracting data from json and appending to lists for calculations
    Latitude = []
    Longitude = []
    Elevation = []
    Slopes = []
    Curvatures = []
    Headings = []
    Ref_Nodes = []
    NonRef_Nodes =[]

    jsonpath_expr = parse("$.response.route[0].leg[0].link[*].linkId")
    LinkID = [int(match.value) for match in jsonpath_expr.find(data1)]

    #Link data from HERE API
    link_length = parse("$.response.route[0].leg[0].link[*].length")
    link_length_Match = [float(match.value) for match in link_length.find(data1)]
    Distance_HERE= data1['response']['route'][0]['summary']['distance']


    attributes = ['HPY', 'HPZ', 'HPX', 'SLOPES', 'CURVATURES','HEADINGS','REFNODE_LINKCURVHEADS','NREFNODE_LINKCURVHEADS']

    attr_matches = {}

    # Iterate over attributes and find matches
    for attr in attributes:
        jsonpath_expr = parse(f"$.response.route[0].leg[0].link[*].attributes.ADAS_ATTRIB_FCN[0].{attr}")
        matches = [json.loads(match.value) for match in jsonpath_expr.find(data1)]
        attr_matches[attr] = matches

    # Extend lists maintaining order
    for i in range(len(attr_matches[attributes[0]])):
        for attr in attributes:
            matches = attr_matches[attr]
            if i < len(matches):
                if attr == 'HPY':
                    Latitude.append(matches[i])
                elif attr == 'HPZ':
                    Elevation.append(matches[i])
                elif attr == 'HPX':
                    Longitude.append(matches[i])
                elif attr == 'SLOPES':
                    Slopes.append(matches[i])
                elif attr == 'CURVATURES':
                    Curvatures.append(matches[i])
                elif attr == 'HEADINGS':
                    Headings.append(matches[i])
                elif attr == 'REFNODE_LINKCURVHEADS':
                    Ref_Nodes.append(matches[i])
                elif attr == 'NREFNODE_LINKCURVHEADS':
                    NonRef_Nodes.append(matches[i])
            else:
                # If no match found for a particular index, extend with [0, 0, 0]
                if attr == 'REFNODE_LINKCURVHEADS':
                    Ref_Nodes.append([0, 0, 0])
                elif attr == 'NREFNODE_LINKCURVHEADS':
                    NonRef_Nodes.append([0, 0, 0])
                else:
                    # For other attributes, extend with None or any other default value if necessary
                    pass

    #Creating the Lists for the data and Replacing the missing data with 0                

    for i in range(len(Curvatures)):
        for j in range(len(Curvatures[i])):
            if Curvatures[i][j]== 1000000000 :
                Curvatures[i][j]=0

    for i in range(len(Curvatures)):
        if not Curvatures[i]:
            if LinkID[i] > 0:
                Curvatures[i] = [Ref_Nodes[i][1], NonRef_Nodes[i][1]]
                Headings[i] = [Ref_Nodes[i][2], NonRef_Nodes[i][2]]
            elif LinkID[i] < 0:
                Curvatures[i] = [NonRef_Nodes[i][1], Ref_Nodes[i][1]]
                Headings[i] = [NonRef_Nodes[i][2], Ref_Nodes[i][2]]

        else :
            if LinkID[i] > 0:
                Curvatures[i].insert(0,Ref_Nodes[i][1])
                Curvatures[i].append(NonRef_Nodes[i][1])
                Headings[i].insert(0,Ref_Nodes[i][2])
                Headings[i].append(NonRef_Nodes[i][2])
            elif LinkID[i] < 0:
                Curvatures[i].insert(0,NonRef_Nodes[i][1])
                Curvatures[i].append(Ref_Nodes[i][1])
                Headings[i].insert(0,NonRef_Nodes[i][2])
                Headings[i].append(Ref_Nodes[i][2])




    #Creation of Lists Ends here
    #Scaling of the data for calculation:

    flip_data = [Latitude, Longitude, Elevation, Slopes, Curvatures, Headings]
    scales =     [1e7,          1e7,      1e2,      1e3,     1e6,       1e3]  

    for i in range(len(LinkID)):
        for data_list, scale in zip(flip_data, scales):
            data_list[i] = list(np.cumsum([float(x) for x in data_list[i]]))
            data_list[i] = [x / scale for x in data_list[i]]
            if int(LinkID[i]) < 0:
                data_list[i] = np.flip(data_list[i]).tolist()

    #Slicing Start and Final:

    def find_nearest_point(input_points, latitudes, longitudes):
        min_distance = float('inf')
        nearest_point = None
        nearest_index = None
        for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
            point = (lat, lon)
            distance = haversine(input_points, point)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
                nearest_index = i
        return nearest_point, nearest_index
    if start_lat is not None and start_lon is not None and end_lat is not None and end_lon is not None:
        input_points_start = (start_lat, start_lon)
        input_points_end = (end_lat, end_lon)
        start_point, start_index = find_nearest_point(input_points_start, Latitude[0], Longitude[0])
        end_point, end_index = find_nearest_point(input_points_end, Latitude[-1], Longitude[-1])
        
        #Start point and end point variables can be accessed if needed 

    #Slicing the data based on the start and end points and slicing operation has been written long for understanding: can be made shorter//


        if start_index == len(Latitude[0]) - 1:
            Latitude[0] = Latitude[0][start_index-1:]
            Longitude[0] = Longitude[0][start_index-1:]
            Latitude[0][0] = start_lat
            Longitude[0][0] = start_lon
            Elevation[0] = Elevation[0][start_index-1:]
            Slopes[0] = Slopes[0][start_index-1:]
            Curvatures[0] = Curvatures[0][start_index-1:]
            Headings[0] = Headings[0][start_index-1:]
            
        elif start_index == 0:
            Latitude[0]=Latitude[0][start_index:]
            Longitude[0]=Longitude[0][start_index:]
            Elevation[0]=Elevation[0][start_index:]
            Slopes[0]=Slopes[0][start_index:]
            Curvatures[0]=Curvatures[0][start_index:]
            Headings[0]=Headings[0][start_index:]
            
        else:
            Latitude[0]=Latitude[0][start_index:]
            Longitude[0]=Longitude[0][start_index:]
            Elevation[0]=Elevation[0][start_index:]
            Slopes[0]=Slopes[0][start_index:]
            Curvatures[0]=Curvatures[0][start_index:]
            Headings[0]=Headings[0][start_index:]
            

        if end_index == 0:
            Latitude[-1]= Latitude[-1][:2]
            Longitude[-1]= Longitude[-1][:2]
            Latitude[-1][-1]= end_lat
            Longitude[-1][-1]= end_lon
            Elevation[-1]= Elevation[-1][:2]
            Slopes[-1]= Slopes[-1][:2]
            Curvatures[-1]= Curvatures[-1][:2]
            Headings[-1]= Headings[-1][:2]
            
        elif end_index == len(Latitude[-1]) - 1:
            Latitude[-1]= Latitude[-1][:]
            Longitude[-1]= Longitude[-1][:]
            Elevation[-1]= Elevation[-1][:]
            Slopes[-1]= Slopes[-1][:]
            Curvatures[-1]= Curvatures[-1][:]
            Headings[-1]= Headings[-1][:]
        else:
            Latitude[-1]= Latitude[-1][:end_index+1]
            Longitude[-1]= Longitude[-1][:end_index+1]
            Latitude[-1][-1]= end_lat
            Longitude[-1][-1]= end_lon
            Elevation[-1]= Elevation[-1][:end_index+1]
            Slopes[-1]= Slopes[-1][:end_index+1]
            Curvatures[-1]= Curvatures[-1][:end_index+1]
            Headings[-1]= Headings[-1][:end_index+1]

    else:        
        print("Start and end points not provided. Skipping slicing operation.")
    
    #Distance Calculation Functions :
    
    def haver(point1, point2):

        earth_radius = 6371e3
        lat1, lon1, elev1 = radians(point1[0]), radians(point1[1]), point1[2]
        lat2, lon2, elev2 = radians(point2[0]), radians(point2[1]), point2[2]
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) * sin(dlat / 2) + cos(lat1) * cos(lat2) * sin(dlon / 2) * sin(dlon / 2)
        c = 2 * asin(sqrt(a))
        horizontal_distance = earth_radius * c
        vertical_distance = abs(elev2 - elev1)
        inclined_distance = sqrt(horizontal_distance ** 2 + vertical_distance ** 2)
        return horizontal_distance, inclined_distance

    def Geo_distance(point1, point2):

        geod = Geod(ellps='WGS84')
        ele1 = point1[2]
        ele2 = point2[2]
        _, _, distance = geod.inv(point1[1], point1[0], point2[1], point2[0])
        inclined_distance = sqrt(distance ** 2 + (ele2 - ele1) ** 2)
        return distance, inclined_distance
    
    #calculation Execution

    distance_list_hav = []
    distance_list_hav_inclined = []
    distance_list_geo = []
    distance_list_geo_inclined = []
    for i in range(len(Latitude)):
        link_hav = []
        link_hav_inlcined=[]
        link_geo = []
        
        link_geo_inclined = []
        distance_list_hav.append(link_hav)  
        distance_list_geo.append(link_geo)
        distance_list_geo_inclined.append(link_geo_inclined)
        distance_list_hav_inclined.append(link_hav_inlcined)

        for j in range(len(Latitude[i])-1):  
            
            point1 = (Latitude[i][j], Longitude[i][j], Elevation[i][j])
            point2 = (Latitude[i][j+1], Longitude[i][j+1], Elevation[i][j+1])
            hav,hav_inclined = haver(point1, point2)
            link_hav.append(hav)
            link_hav_inlcined.append(hav_inclined)
            geodesic, geodesic_inclined= Geo_distance(point1, point2)
            link_geo.append(geodesic)
            link_geo_inclined.append(geodesic_inclined)

    #Link Length Calculation below :
    Haversine_distance =[]
    Haversine_distance_inclined = []
    Geodesic_distance = []
    Geodesic_distance_inclined = []

    for i in range(len(Latitude)):
        Haversine_distance.append(sum(distance_list_hav[i]))
        Haversine_distance_inclined.append(sum(distance_list_hav_inclined[i]))
        Geodesic_distance.append(sum(distance_list_geo[i]))
        Geodesic_distance_inclined.append(sum(distance_list_geo_inclined[i]))


    Total_haversine_distance = sum(Haversine_distance)
    Total_haversine_distance_inclined = sum(Haversine_distance_inclined)
    Total_geodesic_distance = sum(Geodesic_distance)
    Total_geodesic_distance_inclined = sum(Geodesic_distance_inclined)

    #Model Attributes

    #Speed Limit

    links_parse = parse("$.response.route[0].leg[0].link[*]")
    speed_limit_parse = parse("$.attributes.APPLICABLE_SPEED_LIMIT[0].APPLICABLE_SPEED_LIMIT")
    speed_limits = []

    for link in links_parse.find(data1):
        speed_limit = speed_limit_parse.find(link.value)
        speed_limits.append(int(speed_limit[0].value) if speed_limit else None)


    speed_limit_updated = speed_limits.copy()

    for i in range(len(speed_limits)):
        # If the current speed limit is None
        if speed_limits[i] is None:
            # If it's the first element, find the next available speed limit
            if i == 0:
                j = 1
                while speed_limits[j] is None and j < len(speed_limits) - 1:
                    j += 1
                speed_limit_updated[i] = speed_limits[j]
            else:
                # Otherwise, use the previous speed limit
                speed_limit_updated[i] = speed_limits[i - 1]

    #Get from the LINKS* Tree:

    links_parse = parse("$.response.route[0].leg[0].link[*]")

    Traffic_Condition = []
    Traffic_Sign = []   

    Free_Flow_Speed = []

    Wind_Direction = []
    Wind_Velocity = []  

    for i,link in enumerate(links_parse.find(data1)):
        condition_parse = parse("$.attributes.TRAFFIC_SIGN_FCN[0].CONDITION_TYPE")
        condition = condition_parse.find(link.value)
        Traffic_Condition.append(int(condition[0].value) if condition else None)

        sign_parse = parse("$.attributes.TRAFFIC_SIGN_FCN[0].TRAFFIC_SIGN_TYPE")
        sign = sign_parse.find(link.value)
        Traffic_Sign.append(int(sign[0].value) if sign else None)

        free_flow_speed_parse = parse("$.attributes.TRAFFIC_PATTERN_FCN[0].FREE_FLOW_SPEED")
        free_flow_speed = free_flow_speed_parse.find(link.value)
        Free_Flow_Speed.append(int(free_flow_speed[0].value) if free_flow_speed else speed_limit_updated[i])

        wind_direction_parse = parse("$.attributes.ARCHIVED_WEATHER[0].WIND_DIRECTION")
        wind_direction = wind_direction_parse.find(link.value)
        Wind_Direction.append(float(wind_direction[0].value) if wind_direction else None)

        wind_velocity_parse = parse("$.attributes.ARCHIVED_WEATHER[0].WIND_VELOCITY")
        wind_velocity = wind_velocity_parse.find(link.value)
        Wind_Velocity.append(float(wind_velocity[0].value) if wind_velocity else 0)


    #Creating a list of lists as Latitudes, Longitudes, Elevation etc. to match the length 


    Speed_parse = dict(zip(LinkID, speed_limit_updated))
    speed_limits_list = [[Speed_parse[link_id]] * len(lat) for link_id, lat in zip(LinkID, Latitude)]
    speed_limits_list_float = [[float(value) / 3.6 for value in sublist] for sublist in speed_limits_list]

    traffic_signal_parse = dict(zip(LinkID, Traffic_Condition))
    traffic_signals = [[traffic_signal_parse[link_id]] + [0]*(len(lat)-1) for link_id, lat in zip(LinkID, Latitude)]

    Traffic_sign_parse = dict(zip(LinkID, Traffic_Sign))
    traffic_signs = [[Traffic_sign_parse[link_id]] + [0]*(len(lat)-1) for link_id, lat in zip(LinkID, Latitude)]

    free_flow_speed_parse = dict(zip(LinkID, Free_Flow_Speed))
    Free_Flow_Speed = [[free_flow_speed_parse[link_id]] * len(lat) for link_id, lat in zip(LinkID, Latitude)]
    Free_Flow_Speed_float = [[float(value) / 3.6 for value in sublist] for sublist in Free_Flow_Speed]

    wind_direction_parse = dict(zip(LinkID, Wind_Direction))
    Wind_Direction = [[wind_direction_parse[link_id]] + [0]*(len(lat)-1) for link_id, lat in zip(LinkID, Latitude)]

    wind_velocity_parse = dict(zip(LinkID, Wind_Velocity))
    Wind_Velocity = [[wind_velocity_parse[link_id]] + [0]*(len(lat)-1) for link_id, lat in zip(LinkID, Latitude)]
    wind_velocity_float = [[float(value) / 3.6 for value in sublist] for sublist in Wind_Velocity]

    # Flattening of Lists for DOC File 

    flat_latitudes = [item for i, sublist in enumerate(Latitude) for item in sublist[:-1] if i != len(Latitude) - 1] + Latitude[-1]
    flat_longitudes = [item for i, sublist in enumerate(Longitude) for item in sublist[:-1] if i != len(Longitude) - 1] + Longitude[-1]
    flat_elevations = [item for i, sublist in enumerate(Elevation) for item in sublist[:-1] if i != len(Elevation) - 1] + Elevation[-1]
    flat_slopes= [item for i, sublist in enumerate(Slopes) for item in sublist[:-1] if i != len(Slopes) - 1] + Slopes[-1]
    flat_curvatures = [item for i, sublist in enumerate(Curvatures) for item in sublist[:-1] if i != len(Curvatures) - 1] + Curvatures[-1]
    flat_headings = [item for i, sublist in enumerate(Headings) for item in sublist[:-1] if i != len(Headings) - 1] + Headings[-1]
    flat_speed_limits = [item for i, sublist in enumerate(speed_limits_list_float) for item in sublist[:-1] if i != len(speed_limits_list_float) - 1] + speed_limits_list_float[-1]
    flat_free_flow_speed = [item for i, sublist in enumerate(Free_Flow_Speed_float) for item in sublist[:-1] if i != len(Free_Flow_Speed_float) - 1] + Free_Flow_Speed_float[-1]
    flat_traffic_signals = [item for i, sublist in enumerate(traffic_signals) for item in sublist[:-1] if i != len(traffic_signals) - 1] + traffic_signals[-1]
    flat_traffic_signs = [item for i, sublist in enumerate(traffic_signs) for item in sublist[:-1] if i != len(traffic_signs) - 1] + traffic_signs[-1]
    flat_wind_direction = [item for i, sublist in enumerate(Wind_Direction) for item in sublist[:-1] if i != len(Wind_Direction) - 1] + Wind_Direction[-1]
    flat_wind_velocity = [item for i, sublist in enumerate(wind_velocity_float) for item in sublist[:-1] if i != len(wind_velocity_float) - 1] + wind_velocity_float[-1]

    #Sgnals and signs filtering 

    Traffic_signal =[flat_traffic_signals[i] == 16 for i in range(len(flat_traffic_signals))]
    Stop_sign=[flat_traffic_signs[i] == 20 for i in range(len(flat_traffic_signs))]
    Yield_sign=[flat_traffic_signs[i] == 42 for i in range(len(flat_traffic_signs))]
    pedestrian_crossing=[flat_traffic_signs[i] == 41 for i in range(len(flat_traffic_signs))]

    print("Extracting the attribues from the JSON file is successful")



    #Distance between Consecutive Points after flattening the lists:
    def calculate_distances(points):
        wgs84_geod = Geod(ellps='WGS84') 
        distances = []
        inclined_distances = []

        for i in range(len(points)-1):
            lon1, lat1, ele1 = points[i]
            lon2, lat2, ele2 = points[i+1]

            # Calculate horizontal distance
            az12, az21, dist = wgs84_geod.inv(lon1, lat1, lon2, lat2)
            distances.append(dist)

            # Calculate inclined distance
            ele_diff = abs(ele2 - ele1)
            inclined_dist = sqrt(dist**2 + ele_diff**2)
            inclined_distances.append(inclined_dist)

        return distances, inclined_distances

    print("Distance Calculation Successful")
    
    #CSV FILE Creation Section :


    flat_distance, flat_inclined_distance = calculate_distances(list(zip(flat_longitudes, flat_latitudes, flat_elevations)))
    flat_distance.insert(0, 0)
    flat_inclined_distance.insert(0, 0)
    distance_model = np.cumsum(flat_inclined_distance)

    check_distance=  np.round(flat_distance,2)
    check_distance = sum(check_distance)
    Total_flat_distance = sum(flat_distance)    
    Total_flat_inclined_distance = sum(flat_inclined_distance)  

    
    #CSV FILE and Folder Creation Section 
    csv_files_directory = os.path.join(output_directory, 'CSV_Files')
    os.makedirs(csv_files_directory, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_file_path = os.path.join(csv_files_directory, base_name + '.csv')

    Test = pd.DataFrame({'Linklenghth': link_length_Match,   'Geodesic Distance': Geodesic_distance, 'Geodesic Distance Inclined': Geodesic_distance_inclined})
    Test.to_csv('Test.csv', index=False)


    Model = pd.DataFrame({'Latitude': flat_latitudes, 'Longitude': flat_longitudes, 'Elevation': flat_elevations, 'Slopes': flat_slopes, 'Curvatures': flat_curvatures, 'Headings': flat_headings, 'Distance': distance_model,
                        'Speed Limits': flat_speed_limits, 'free_flow_speed': flat_free_flow_speed,'Traffic Signals': Traffic_signal, 'Stop Sign': Stop_sign, 'Yield Sign': Yield_sign, 'Pedestrian Crossing': pedestrian_crossing, 'Wind Direction': flat_wind_direction, 'Wind Velocity': flat_wind_velocity})
    Model.to_csv(output_file_path, index=False)
    print(f"Model File Saved")
    

    #Printing Function/ creating Report :
    reports_directory = os.path.join(output_directory, 'Reports')
    os.makedirs(reports_directory, exist_ok=True)

    with open(os.path.join(reports_directory, base_name+ 'Distance Report.txt'), 'w') as f:
        # print("Distance from HERE MAPS:", Distance_HERE)
        print(f"\n HAVERSINE:{Total_haversine_distance} Difference: {Total_haversine_distance - Distance_HERE}\n Error: {((Total_haversine_distance - Distance_HERE)/Distance_HERE)*100}%", file=f)
        print(f"\n HAVERSINE Inclined:{Total_haversine_distance_inclined} Difference: {Total_haversine_distance_inclined - Distance_HERE}\n Error: {((Total_haversine_distance_inclined - Distance_HERE)/Distance_HERE)*100}%", file=f)
        # # print(f"\n Geodesic Distance:{Total_geodesic_distance} Difference: {Total_geodesic_distance - Distance_HERE}\n Error: {((Total_geodesic_distance - Distance_HERE)/Distance_HERE)*100}%")
        # # print(f"\n Geodesic Distance Inclined:{Total_geodesic_distance_inclined} Difference: {Total_geodesic_distance_inclined - Distance_HERE}\n Error: {((Total_geodesic_distance_inclined - Distance_HERE)/Distance_HERE)*100}%")
        print(f"\n GEODESIC:{Total_flat_distance} Difference: {Total_flat_distance - Distance_HERE}\n Error: {((Total_flat_distance - Distance_HERE)/Distance_HERE)*100}%", file=f)
        print(f"\n GEODESIC Inclined:{Total_flat_inclined_distance} Difference: {Total_flat_inclined_distance - Distance_HERE}\n Error: {((Total_flat_inclined_distance - Distance_HERE)/Distance_HERE)*100}%", file=f)
        # print(f"\n Geodesic Distance:{total_geo_flat} Difference: {total_geo_flat - Distance_HERE}\n Error: {((total_geo_flat - Distance_HERE)/Distance_HERE)*100}%")
        # print(f"\n rounded distance:{check_distance} Difference: {check_distance - Distance_HERE}\n Error: {((check_distance - Distance_HERE)/Distance_HERE)*100}%")    


    # #Plotting the Graphs

    # plt.plot(flat_longitudes, flat_latitudes)
    # # # plt.plot(np.cumsum(dist_vinc_int),interpol_HPZ,  'x-', label='interpolated HPZ')
    # # plt.axes('equal')
    # plt.xlabel('Longitudes')
    # plt.ylabel('Latitudes')
    # # plt.legend()
    # plt.title('Vehicle Path')
    # plt.savefig(os.path.join(reports_directory, base_name + 'Vehicle Path.png'))
    # # plt.show()
    pass

# The Program Execution Starts from here : Incase if you are using json file directly, Comment this function 
def process_files_in_directory(directory, output_directory, extension=".txt"):
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            try:
                print(f"Processing file: {filename}")
                process_file(file_path, output_directory)  # Call your function here
            except Exception as e:
                print(f"Failed to process file: {filename}. Error: {str(e)}")

input_directory = "C:/path/to/folder" # Specify the input txt files folder here 
output_directory = input_directory # Specify the output folder by default it will be inside the input directory 
process_files_in_directory(input_directory, output_directory)