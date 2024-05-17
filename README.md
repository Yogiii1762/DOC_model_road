# **Advancing Vehicle Path Estimation Using Geospatial Data Analysis**

## **Overview**

This repository contains the code developed for my Master's Thesis, which focuses on advancing vehicle path estimation using geospatial data analysis. The primary goal of this project is to extract detailed road data from GPS traces and create a deterministic operating cycle (DOC) model. This model is used for various applications, including residual range estimation for vehicles.

## **Features**

- **Extract Road Data**: The code extracts road data by processing latitude and longitude GPS traces from a vehicle.
- **Generate DOC Model**: It returns a deterministic operating cycle (DOC) model in CSV format, which includes attributes such as elevation, slope, speed limit, and more.
- **Distance Calculation**: Implements accurate distance calculation using Haversine and Vincenty formulas.
- **Energy Consumption Analysis**: Facilitates analysis of energy consumed by the vehicle and estimation of remaining travel distance based on available battery or fuel.

## **Installation**

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/Yogiii1762/doc_model_road.git
```
## **Requirements**

Ensure you have the following dependencies installed:

- Python 3.x
- pandas
- requests
- numpy
- geopy
- json
- Matplotlib
Also look out for other dependencies if you dont have them by default.

You can install the required packages using pip:

```bash
pip install pandas requests numpy geopy json matplotlib
```

## **USAGE**
## **Extracting Road Data**
To extract road data and generate the DOC model, you will need a HERE Maps API key. Run the script DOC_model.py:
- **api_key**: Your HERE Maps API key
- **input**: Path to the input CSV file containing latitude and longitude GPS traces.
- **output**: Path to the output CSV file where the DOC model will be saved.(Optional : by default the input directory)

## **Input and Output**
- **Input file** : The input should be a text file and should contain gps traces like below

``` bash
Latitude, Longitude
x1, y1
x2,y2
xn,yn
```
## **Methodology**
- **Data Collection**: The script collects GPS traces (latitude, longitude, and optionally elevation) from a CSV file.
- **Distance Calculation**: It calculates the distance between consecutive GPS points using Haversine and Vincenty formulas.
- **Road Data Extraction**: It uses HERE Maps API to extract road attributes such as speed limit, slope, and curvature based on the GPS traces.
- **DOC Model Generation**: The extracted data is compiled into a deterministic operating cycle (DOC) model in CSV format.
## **Applications**
- **Residual Range Estimation**: The DOC model can be used to estimate the remaining travel distance with the available energy in the battery or fuel.
- **Energy Consumption Analysis**: Analyzes the energy consumed by the vehicle based on accurate distance estimation and road conditions.
- **Simulation and Testing**: Provides detailed road conditions and vehicle dynamics for simulation and testing purposes.
