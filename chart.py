import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, ttest_ind

# File paths
property_values_path = "/Users/amanagrawal/Desktop/Data Sci 2025/property values by neighborhood.csv"
neighborhoods_path = "//Users/amanagrawal/Desktop/Data Sci 2025/usneighborhoods.csv"
parks_geojson_path = "/Users/amanagrawal/Desktop/Data Sci 2025/USA_Parks.geojson"
merged_output_path = "/Users/amanagrawal/Desktop/Data Sci 2025/merged_property_values.geojson"
# Step 1: Load datasets
print("Loading datasets...")
property_values_data = pd.read_csv(property_values_path)
neighborhoods_data = pd.read_csv(neighborhoods_path)
parks_gdf = gpd.read_file(parks_geojson_path)

# Step 2: Convert neighborhoods to GeoDataFrame
print("Converting neighborhoods to GeoDataFrame...")
neighborhoods_data['geometry'] = neighborhoods_data.apply(
    lambda row: Point(row['lng'], row['lat']), axis=1
)
neighborhoods_gdf = gpd.GeoDataFrame(neighborhoods_data, geometry='geometry', crs="EPSG:4326")

# Step 3: Prepare property values dataset
print("Preparing property values dataset...")
property_values_data.rename(columns={'RegionName': 'neighborhood', 'Prop_Values': 'Property_Value'}, inplace=True)

# Step 4: Merge neighborhoods with property values
print("Merging neighborhoods with property values...")
merged_gdf = neighborhoods_gdf.merge(property_values_data, on='neighborhood', how='inner')

# Step 5: Use centroids for park polygons
print("Converting park polygons to centroids...")
parks_gdf['geometry'] = parks_gdf.geometry.centroid
parks_gdf = parks_gdf.set_geometry('geometry')

# Step 6: Reproject to Cartesian coordinates for distance calculations
print("Reprojecting to Cartesian coordinates...")
merged_gdf = merged_gdf.to_crs(epsg=3395)
parks_gdf = parks_gdf.to_crs(epsg=3395)

# Step 7: Calculate nearest park distances
print("Calculating nearest park distances...")
property_coords = np.array(list(merged_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
park_coords = np.array(list(parks_gdf.geometry.apply(lambda geom: (geom.x, geom.y))))
park_tree = cKDTree(park_coords)
distances, _ = park_tree.query(property_coords)
merged_gdf['nearest_park_distance'] = distances

# Step 8: Reproject back to WGS84
print("Reprojecting back to WGS84...")
merged_gdf = merged_gdf.to_crs(epsg=4326)

# Step 9: Save the updated GeoDataFrame
print("Saving merged GeoDataFrame...")
merged_gdf.to_file(merged_output_path, driver='GeoJSON')
print(f"Merged GeoDataFrame with park proximity saved to: {merged_output_path}")

# Step 10: Create subgroups and analyze differences
print("Analyzing differences across multiple cutoffs...")
cutoff_points = [500, 1000, 1500, 2000]  # Cutoff distances in meters
analysis_results = []

for cutoff in cutoff_points:
    # Create subgroups
    merged_gdf['Proximity_Group'] = np.where(merged_gdf['nearest_park_distance'] <= cutoff, 'Closer', 'Farther')
    
    # Calculate means
    closer_mean = merged_gdf[merged_gdf['Proximity_Group'] == 'Closer']['Property_Value'].mean()
    farther_mean = merged_gdf[merged_gdf['Proximity_Group'] == 'Farther']['Property_Value'].mean()
    
    # Perform t-test
    closer_values = merged_gdf[merged_gdf['Proximity_Group'] == 'Closer']['Property_Value']
    farther_values = merged_gdf[merged_gdf['Proximity_Group'] == 'Farther']['Property_Value']
    t_stat, p_value = ttest_ind(closer_values, farther_values, nan_policy='omit')
    
    # Store results
    analysis_results.append({
        'Cutoff (meters)': cutoff,
        'Closer Mean': closer_mean,
        'Farther Mean': farther_mean,
        'T-Statistic': t_stat,
        'P-Value': p_value
    })

# Convert results to DataFrame
analysis_df = pd.DataFrame(analysis_results)

# Step 11: Visualization of differences
print("Visualizing differences across cutoff points...")
plt.figure(figsize=(10, 6))
plt.plot(analysis_df['Cutoff (meters)'], analysis_df['Closer Mean'], label='Closer Mean')
plt.plot(analysis_df['Cutoff (meters)'], analysis_df['Farther Mean'], label='Farther Mean')
plt.xlabel('Cutoff Distance (meters)')
plt.ylabel('Mean Property Value (USD)')
plt.title('Property Values by Proximity to Parks')
plt.legend()
plt.grid(True)
plt.show()

# Step 12: Display analysis results
import ace_tools as tools; tools.display_dataframe_to_user(name="Analysis of Park Proximity", dataframe=analysis_df)

print("Analysis complete.")