import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import os
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore', message='Geometry is in a geographic CRS')

# Set page configuration
st.set_page_config(
    page_title="Baltimore Amenity & Transit Optimizer",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Baltimore Amenity & Transit Optimization Tool")
st.markdown("""
This application helps optimize the placement of amenities and bus stops in Baltimore, MD. 
You can select a census block group to define your study area, then iteratively place different 
types of amenities, and finally optimize bus stop locations based on the amenities.
""")

# Initialize session state for storing data between interactions
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'optimize_bus_stops' not in st.session_state:
    st.session_state.optimize_bus_stops = False
if 'selected_block_group' not in st.session_state:
    st.session_state.selected_block_group = None
if 'study_area' not in st.session_state:
    st.session_state.study_area = None
if 'placed_amenities' not in st.session_state:
    st.session_state.placed_amenities = None
if 'bus_stops' not in st.session_state:
    st.session_state.bus_stops = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'vacant_lots' not in st.session_state:
    st.session_state.vacant_lots = None
if 'census_blocks' not in st.session_state:
    st.session_state.census_blocks = None
if 'amenities' not in st.session_state:
    st.session_state.amenities = None
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []
if 'distance_df' not in st.session_state:
    st.session_state.distance_df = None
if 'distance_df_amenities' not in st.session_state:
    st.session_state.distance_df_amenities = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "amenity_optimization"

# Create tabs for the two main components
tab1, tab2 = st.tabs(["Amenity Optimization", "Bus Stop Optimization"])

# Function to load data
@st.cache_data
def load_data():
    """Load all necessary datasets for the application"""
    try:
        # Define data file paths in the data folder
        data_dir = "data"
        census_data_file = os.path.join(data_dir, "baltimore_city_census_data.csv")
        census_blocks_file = os.path.join(data_dir, "tl_2020_24_bg20.shp")  # TIGER/Line shapefile for MD block groups
        vacant_lots_file = os.path.join(data_dir, "Geo_dataframe_lots_sorted.shp")
        amenities_file = os.path.join(data_dir, "ammenities_cleaned.gpkg")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            st.warning(f"Creating data directory: {data_dir}")
            os.makedirs(data_dir)
            
        # Check if files exist
        missing_files = []
        if not os.path.exists(census_data_file):
            missing_files.append("baltimore_city_census_data.csv")
        
        # Check if census blocks shapefile directory exists
        if not os.path.exists(census_blocks_file):
            # Try checking for individual shapefile components
            shapefile_path = os.path.join(data_dir, "tl_2020_24_bg20.shp")
            if not os.path.exists(shapefile_path):
                missing_files.append("tl_2020_24_bg20 census blocks shapefile")
        
        if not os.path.exists(vacant_lots_file):
            missing_files.append("Geo_dataframe_lots_sorted.shp")
        
        if not os.path.exists(amenities_file):
            missing_files.append("ammenities_cleaned.gpkg")
        
        if missing_files:
            st.error(f"Missing required data files in the 'data' folder: {', '.join(missing_files)}")
            st.info("Please place the required data files in the 'data' folder and restart the application.")
            return None, None, None
        
        # Load datasets
        # Load census data from CSV
        census_data = pd.read_csv(census_data_file)
        
        # Load census block geometries from shapefile
        if os.path.exists(os.path.join(data_dir, "tl_2020_24_bg20.shp")):
            census_blocks_geo = gpd.read_file(os.path.join(data_dir, "tl_2020_24_bg20.shp"))
        else:
            census_blocks_geo = gpd.read_file(census_blocks_file)
        
        # Check if the CSV has a GEOID column for joining with geometries
        if 'GEOID' not in census_data.columns:
            potential_geoid_cols = [col for col in census_data.columns 
                                  if 'geoid' in col.lower() or 'id' in col.lower() or 'tract' in col.lower()]
            if potential_geoid_cols:
                census_data['GEOID'] = census_data[potential_geoid_cols[0]]
                st.info(f"Using {potential_geoid_cols[0]} as GEOID column")
        
        # Filter census blocks to just Baltimore City if needed
        if 'COUNTYFP' in census_blocks_geo.columns:
            # Baltimore City FIPS code is 510
            baltimore_blocks = census_blocks_geo[census_blocks_geo['COUNTYFP'] == '510']
            if not baltimore_blocks.empty:
                census_blocks_geo = baltimore_blocks
                st.info(f"Filtered to {len(census_blocks_geo)} Baltimore City census blocks")
        
        # Join census data with geometries
        # Make sure GEOIDs are strings for joining
        census_blocks_geo['GEOID'] = census_blocks_geo['GEOID'].astype(str)
        census_data['GEOID'] = census_data['GEOID'].astype(str)
        
        census_blocks = census_blocks_geo.merge(census_data, on='GEOID', how='inner')
        st.info(f"Joined census data with geometries, resulting in {len(census_blocks)} census blocks")
        
        # Load vacant lots from shapefile
        vacant_lots = gpd.read_file(vacant_lots_file)
        
        # Check for vacant indicator column
        if 'vacant' not in vacant_lots.columns:
            st.warning("No 'vacant' column found in vacant lots data. Assuming all lots are vacant.")
            vacant_lots['vacant'] = 1.0
        else:
            # Filter to only vacant lots
            vacant_lots = vacant_lots[vacant_lots['vacant'] == 1.0]
        
        # Load amenities data
        amenities = gpd.read_file(amenities_file)
        
        return census_blocks, vacant_lots, amenities
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load OSM graph for network analysis
@st.cache_data
def load_graph():
    """Load OpenStreetMap graph for Baltimore"""
    try:
        place_name = "Baltimore, Maryland, United States"
        graph = ox.graph_from_place(place_name, network_type='walk')
        return graph
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return None

# Helper function to define study area based on selected block group
def define_study_area(census_blocks, selected_block_id, radius_miles=3):
    """
    Define study area as all census blocks within a given radius of the selected block
    
    Parameters:
        census_blocks: GeoDataFrame of census blocks
        selected_block_id: ID of the selected census block
        radius_miles: Radius in miles to define the study area
    
    Returns:
        GeoDataFrame of census blocks in the study area
    """
    # Get the selected block
    selected_block = census_blocks[census_blocks['GEOID'] == selected_block_id]
    
    if selected_block.empty:
        st.error(f"Selected block ID {selected_block_id} not found")
        return None
    
    # Get centroid of selected block
    centroid = selected_block.geometry.centroid.iloc[0]
    
    # Convert radius from miles to meters
    radius_meters = radius_miles * 1609.34
    
    # Project to a CRS that uses meters
    census_blocks_proj = census_blocks.to_crs("EPSG:32618")  # UTM Zone 18N for Baltimore
    centroid_proj = gpd.GeoSeries([centroid], crs=census_blocks.crs).to_crs("EPSG:32618")
    
    # Create buffer around centroid
    buffer = centroid_proj.buffer(radius_meters)
    
    # Find all blocks that intersect with the buffer
    intersects = census_blocks_proj.geometry.intersects(buffer.iloc[0])
    study_area = census_blocks.loc[intersects]
    
    return study_area

# Function to prepare vacant lots
def prepare_vacant_lots(vacant_lots, graph):
    """
    Prepare vacant lots for analysis by finding nearest network nodes
    
    Parameters:
        vacant_lots: GeoDataFrame of vacant lots
        graph: OSMnx graph
    
    Returns:
        GeoDataFrame of vacant lots with added info
    """
    proj_crs = "EPSG:32618"  # UTM Zone 18N for Baltimore
    graph_crs = graph.graph['crs']

    # Project and calculate centroid
    vacant_lots_proj = vacant_lots.to_crs(proj_crs)
    vacant_lots_proj['centroid'] = vacant_lots_proj.geometry.centroid

    # Convert centroids back to lat/lon for graph node lookup
    centroids_latlon = gpd.GeoSeries(vacant_lots_proj['centroid'], crs=proj_crs).to_crs(graph_crs)
    vacant_lots = vacant_lots.to_crs(graph_crs)
    vacant_lots['centroid'] = centroids_latlon
    vacant_lots['lon'] = centroids_latlon.x
    vacant_lots['lat'] = centroids_latlon.y

    # Find nearest graph node
    vacant_lots['nearest_node'] = ox.distance.nearest_nodes(graph, X=vacant_lots['lon'], Y=vacant_lots['lat'])
    vacant_lots_proj['nearest_node'] = vacant_lots['nearest_node']

    return vacant_lots_proj

# Function to prepare census blocks
def prepare_census_blocks(census_blocks, graph):
    """
    Prepare census blocks for analysis by finding nearest network nodes
    
    Parameters:
        census_blocks: GeoDataFrame of census blocks
        graph: OSMnx graph
    
    Returns:
        GeoDataFrame of census blocks with added info
    """
    proj_crs = "EPSG:32618"  # UTM Zone 18N for Baltimore
    graph_crs = graph.graph['crs']

    # Project and calculate centroid
    census_blocks_proj = census_blocks.to_crs(proj_crs)
    census_blocks_proj['centroid'] = census_blocks_proj.geometry.centroid

    # Convert centroids back to lat/lon for node matching
    centroids_latlon = gpd.GeoSeries(census_blocks_proj['centroid'], crs=proj_crs).to_crs(graph_crs)
    census_blocks = census_blocks.to_crs(graph_crs)
    census_blocks['centroid'] = centroids_latlon
    census_blocks['lon'] = centroids_latlon.x
    census_blocks['lat'] = centroids_latlon.y

    # Add coordinates + nearest nodes to projected version
    census_blocks_proj['lon'] = census_blocks['lon']
    census_blocks_proj['lat'] = census_blocks['lat']
    census_blocks_proj['nearest_node'] = ox.distance.nearest_nodes(graph, X=census_blocks_proj['lon'], Y=census_blocks_proj['lat'])

    return census_blocks_proj

# Function to compute distances using Dijkstra tree
def compute_fast_distances_dijkstra_tree(graph, census_blocks_proj, vacant_lots_proj):
    """
    Compute walking distances from each census block to each vacant lot
    
    Parameters:
        graph: OSMnx graph
        census_blocks_proj: Projected GeoDataFrame of census blocks
        vacant_lots_proj: Projected GeoDataFrame of vacant lots
    
    Returns:
        DataFrame with distances
    """
    # Project graph
    graph_proj = ox.project_graph(graph)
    nodes_proj, _ = ox.graph_to_gdfs(graph_proj, nodes=True)

    # Add nearest_node to census blocks (lat/lon → graph)
    census_blocks_proj['nearest_node'] = ox.distance.nearest_nodes(
        graph, X=census_blocks_proj['lon'], Y=census_blocks_proj['lat']
    )

    # Show progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Computing distances...")

    distances = []
    # Precompute node geometry lookup for offset calculation
    node_geom_lookup = nodes_proj.geometry.to_dict()

    # Loop through each census block
    for i, (idx, block) in enumerate(census_blocks_proj.iterrows()):
        block_id = idx
        block_node = block['nearest_node']

        # Get all shortest paths from this block node
        all_paths = nx.single_source_dijkstra_path_length(graph_proj, source=block_node, weight='length')

        # Loop through each individual vacant lot
        for lot_idx, lot in vacant_lots_proj.iterrows():
            lot_node = lot['nearest_node']
            lot_centroid = lot['centroid']

            # Get precomputed network distance (or skip if unreachable)
            net_dist = all_paths.get(lot_node, np.inf)
            if not np.isfinite(net_dist):
                continue

            # Euclidean offset from node to lot centroid
            node_geom = node_geom_lookup[lot_node]
            eucl_offset = node_geom.distance(lot_centroid)

            total_dist = net_dist + eucl_offset

            distances.append({
                'census_block_index': block_id,
                'vacant_lot_index': lot_idx,
                'vacant_node': lot_node,
                'network_dist_m': net_dist,
                'euclidean_offset_m': eucl_offset,
                'total_dist_m': total_dist
            })
        
        # Update progress
        progress = (i + 1) / len(census_blocks_proj)
        progress_bar.progress(progress)
        progress_text.text(f"Computing distances... {i+1}/{len(census_blocks_proj)} blocks processed")

    progress_text.text("Distance computation complete!")
    time.sleep(1)
    progress_text.empty()
    progress_bar.empty()

    return pd.DataFrame(distances)

# Function for amenity placement optimization using genetic algorithm
def optimize_amenity_placement_genetic_constrained(
    distance_df_all,
    distance_df_existing,
    census_blocks,
    vacant_lots,
    ammenities_gdf,
    amenity_group,
    demographic_priority=None,
    num_new_amenities=5,
    generations=100,
    population_size=50,
    mutation_rate=0.1,
    verbose=True
):
    """
    Optimize amenity placement using genetic algorithm with constraints
    
    Parameters:
        distance_df_all: DataFrame with distances to all vacant lots
        distance_df_existing: DataFrame with distances to existing amenities
        census_blocks: GeoDataFrame of census blocks
        vacant_lots: GeoDataFrame of vacant lots
        ammenities_gdf: GeoDataFrame of existing amenities
        amenity_group: Type of amenity to optimize
        demographic_priority: Demographic variable to prioritize
        num_new_amenities: Number of new amenities to place
        generations: Number of generations for genetic algorithm
        population_size: Population size for genetic algorithm
        mutation_rate: Mutation rate for genetic algorithm
        verbose: Whether to show verbose output
        
    Returns:
        best_lots_df: GeoDataFrame of optimal vacant lots for new amenities
        baseline: Total distance before optimization
        best_score: Total distance after optimization
    """
    all_lots = distance_df_all['vacant_lot_index'].unique()
    existing_lots = distance_df_existing['vacant_lot_index'].unique() if not distance_df_existing.empty else []
    candidate_lots = list(set(all_lots) - set(existing_lots))

    # Map lot index to census block
    lot_to_block = {}
    for i, row in vacant_lots.iterrows():
        for j, block in census_blocks.iterrows():
            if row.geometry.within(block.geometry):
                lot_to_block[i] = j
                break

    # Demographic block targeting
    demographic_block_id = None
    if demographic_priority and demographic_priority in census_blocks.columns:
        demographic_block_id = census_blocks[demographic_priority].idxmax()
        lots_in_demo_block = [i for i, b in lot_to_block.items() if b == demographic_block_id]
        if not lots_in_demo_block:
            demographic_block_id = None  # Skip if no lots in that block

    # Current total distance
    baseline = 0
    if not distance_df_existing.empty:
        baseline_series = distance_df_existing.groupby('census_block_index')['total_dist_m'].min()
        baseline = baseline_series.sum()
    else:
        # If no existing amenities, use max possible distance
        baseline = float('inf')

    def evaluate(combo):
        combo_df = distance_df_all[distance_df_all['vacant_lot_index'].isin(combo)]
        if distance_df_existing.empty:
            merged = combo_df
        else:
            merged = pd.concat([distance_df_existing, combo_df])
        dist = merged.groupby('census_block_index')['total_dist_m'].min()
        return dist.sum()

    def is_valid_combo(combo):
        used_blocks = set()
        for lot_id in combo:
            block_id = lot_to_block.get(lot_id, None)
            if block_id is not None:
                if block_id in used_blocks:
                    return False
                used_blocks.add(block_id)
        return True

    # Generate population
    population = []
    for _ in range(population_size):
        base = np.random.choice(candidate_lots, num_new_amenities, replace=False).tolist()
        if demographic_block_id and lots_in_demo_block:
            base[0] = np.random.choice(lots_in_demo_block)
        population.append(base)

    best_score = np.inf
    best_solution = None
    
    # Set up progress display
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Running genetic optimization...")
    
    for gen in range(generations):
        scores = []
        for individual in population:
            if not is_valid_combo(individual):
                continue
            score = evaluate(individual)
            scores.append((score, individual))

        if not scores:
            continue

        scores.sort()
        top = scores[:population_size // 2]

        if top[0][0] < best_score:
            best_score = top[0][0]
            best_solution = top[0][1]

        # Update progress
        progress = (gen + 1) / generations
        progress_bar.progress(progress)
        progress_text.text(f"Running genetic optimization... Generation {gen+1}/{generations}, Best score: {best_score:.2f}")

        # Crossover
        new_population = []
        while len(new_population) < population_size:
            p1 = top[np.random.randint(len(top))][1]
            p2 = top[np.random.randint(len(top))][1]
            child = list(set(p1[:num_new_amenities // 2] + p2[:num_new_amenities // 2]))
            while len(child) < num_new_amenities:
                candidate = np.random.choice(candidate_lots)
                if candidate not in child:
                    child.append(candidate)
            if is_valid_combo(child):
                new_population.append(child)

        # Mutation
        for i in range(len(new_population)):
            if np.random.random() < mutation_rate:
                idx = np.random.randint(0, num_new_amenities - 1)
                replacement = np.random.choice(candidate_lots)
                while replacement in new_population[i]:
                    replacement = np.random.choice(candidate_lots)
                new_population[i][idx] = replacement
                if not is_valid_combo(new_population[i]):
                    new_population[i][idx] = replacement  # fallback

        population = new_population

    progress_text.text("Optimization complete!")
    time.sleep(1)
    progress_text.empty()
    progress_bar.empty()

    best_lots_df = vacant_lots.loc[best_solution].copy()
    best_lots_df['assigned_amenity_group'] = amenity_group

    return best_lots_df, baseline, best_score

# Helper function to get suggested number of amenities to add
def get_suggested_amenity_count(census_blocks, ammenities_gdf, amenity_group):
    """Calculate suggested number of amenities to add based on population"""
    # Default ratios (people per amenity) for estimating how many to add
    DEFAULT_AMENITY_RATIOS = {
        'hospital': 15000,
        'pharmacy': 7500,
        'school': 3000,
        'university': 20000,
        'fresh food': 3000,
        'transit': 2500,
        'restaurant': 1500,
        'personal care': 2000,
        'recreation': 2500,
        'bank services': 4000,
        'lodging': 10000,
        'civic': 7500
    }
    
    # Estimate total population
    if 'Total_Population' in census_blocks.columns:
        total_pop = census_blocks['Total_Population'].sum()
    else:
        total_pop = census_blocks['Male_Population'].sum() + census_blocks['Female_Population'].sum()

    # Determine ratio and suggestion
    default_ratio = DEFAULT_AMENITY_RATIOS.get(amenity_group, 5000)
    suggested_total = int(np.ceil(total_pop / default_ratio))
    num_existing = len(ammenities_gdf[ammenities_gdf['amenity_group'] == amenity_group])
    suggested_new = max(suggested_total - num_existing, 0)

    if suggested_new == 0 and amenity_group in ['hospital', 'pharmacy', 'school']:
        suggested_new = 1
        
    return suggested_new, num_existing, total_pop

# Function to find optimal bus stop locations
def find_optimal_bus_stops(demographic_data, amenities_data, eligible_block_groups, n_bus_stops=20,
                          amenity_weight=0.5, drive_weight=0.25, income_weight=0.25):
    """
    Find optimal bus stop locations based on weighted criteria
    
    Parameters:
        demographic_data: DataFrame with census demographic data
        amenities_data: GeoDataFrame of amenities
        eligible_block_groups: List of eligible census block groups
        n_bus_stops: Number of bus stops to place
        amenity_weight: Weight for amenity criterion
        drive_weight: Weight for drive-alone commuting criterion
        income_weight: Weight for income criterion
        
    Returns:
        GeoDataFrame of optimal bus stop locations
    """
    # Set up progress display
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Finding optimal bus stops...")
    
    # Convert eligible block groups to strings for comparison
    eligible_block_groups = [str(block) for block in eligible_block_groups]
    
    # Filter demographic data to eligible block groups
    census_data = demographic_data[demographic_data['GEOID'].isin(eligible_block_groups)]
    
    # Ensure consistent CRS between datasets
    target_crs = census_data.crs
    if amenities_data.crs != target_crs:
        amenities_data = amenities_data.to_crs(target_crs)
    
    # Count amenities in each census block using spatial join
    progress_text.text("Counting amenities per census block...")
    progress_bar.progress(0.1)
    
    # Fix any invalid geometries
    census_data.geometry = census_data.geometry.buffer(0)
    amenities_data.geometry = amenities_data.geometry.buffer(0)
    
    # Create a dissolve of all eligible census blocks
    eligible_area = census_data.dissolve().buffer(0)
    
    # Clip amenities to eligible area
    amenities_in_area = gpd.clip(amenities_data, eligible_area)
    
    # Using intersects predicate for spatial join
    spatial_join = gpd.sjoin(amenities_in_area, census_data, predicate="intersects")
    
    # Count amenities by block
    amenity_counts = spatial_join.groupby("GEOID").size().reset_index(name="amenity_count")
    
    # Join counts back to main dataframe
    census_data = census_data.merge(amenity_counts, on="GEOID", how="left")
    census_data["amenity_count"] = census_data["amenity_count"].fillna(0)
    
    progress_text.text("Creating weighted features for clustering...")
    progress_bar.progress(0.3)
    
    # Create features for clustering
    # Ensure no zeros in income to avoid division by zero
    census_data.loc[census_data["Per_Capita_Income"] <= 0, "Per_Capita_Income"] = 1
    
    # Create inverse income (so lower income areas get higher scores)
    census_data["income_inverse"] = 1 / census_data["Per_Capita_Income"]
    
    # Select and prepare features
    features = census_data[["amenity_count", "Commute_Public_Transit", "income_inverse"]].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    # Apply weights
    features_scaled["amenity_count"] *= amenity_weight
    features_scaled["Commute_Public_Transit"] *= drive_weight
    features_scaled["income_inverse"] *= income_weight
    
    # Calculate combined score
    census_data["combined_score"] = (
        features_scaled["amenity_count"] +
        features_scaled["Commute_Public_Transit"] +
        features_scaled["income_inverse"]
    )
    
    # Adjust number of bus stops if fewer eligible areas than requested
    if len(census_data) < n_bus_stops:
        n_bus_stops = min(len(census_data), n_bus_stops)
    
    progress_text.text(f"Applying K-means clustering to find {n_bus_stops} optimal bus stop locations...")
    progress_bar.progress(0.5)
    
    # Extract coordinates and scores for clustering
    points_for_clustering = np.array(list(zip(
        census_data.geometry.centroid.x,
        census_data.geometry.centroid.y,
        census_data["combined_score"]
    )))
    
    # Apply K-means
    points_for_clustering = np.nan_to_num(points_for_clustering)
    kmeans = KMeans(n_clusters=n_bus_stops, random_state=42).fit(points_for_clustering)
    
    # Add cluster labels
    census_data["cluster"] = kmeans.labels_
    
    # Add cluster info to amenities
    # First, for each amenity, find the nearest census block
    progress_text.text("Assigning clusters to amenities...")
    progress_bar.progress(0.7)
    
    # Create an array of block centroids for KDTree
    block_centroids = np.array([(point.x, point.y) for point in census_data.geometry.centroid])
    tree = KDTree(block_centroids)
    
    # For each amenity, find the nearest block
    amenity_centroids = np.array([(point.x, point.y) for point in amenities_in_area.geometry.centroid])
    nearest_blocks = tree.query(amenity_centroids, k=1)[1]
    
    # Assign cluster to each amenity
    amenities_in_area["cluster"] = [census_data.iloc[i]["cluster"] for i in nearest_blocks]
    
    # Now use amenity density to place stops
    progress_text.text("Finding optimal bus stop locations based on amenity density...")
    progress_bar.progress(0.8)
    
    bus_stops = []
    
    for i in range(n_bus_stops):
        # Get all amenities in this cluster
        cluster_amenities = amenities_in_area[amenities_in_area["cluster"] == i]
        cluster_blocks = census_data[census_data["cluster"] == i]
        
        if len(cluster_amenities) > 0:
            # If we have amenities in this cluster, use them for placement
            try:
                from scipy.stats import gaussian_kde
                
                # Extract amenity coordinates
                amenity_points = np.array([
                    (point.x, point.y) for point in cluster_amenities.geometry.centroid
                ])
                
                # Generate a grid of points over the cluster area
                x_min, y_min, x_max, y_max = cluster_blocks.total_bounds
                grid_size = 50  # number of points in each dimension
                x_grid = np.linspace(x_min, x_max, grid_size)
                y_grid = np.linspace(y_min, y_max, grid_size)
                xx, yy = np.meshgrid(x_grid, y_grid)
                grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
                
                # Calculate density at each grid point
                kde = gaussian_kde(amenity_points.T)
                density = kde(grid_points.T)
                
                # Find the point with highest density
                best_idx = np.argmax(density)
                best_x, best_y = grid_points[best_idx]
                
                # Ensure the point is within eligible area
                best_point = Point(best_x, best_y)
                if not eligible_area.contains(best_point)[0]:
                    # If outside, find closest point within eligible area
                    # For simplicity, fall back to weighted centroid of amenities
                    coords_weighted = np.average(
                        amenity_points, 
                        weights=[cluster_amenities.iloc[i].get("amenity_count", 1) 
                                for i in range(len(cluster_amenities))],
                        axis=0
                    )
                    best_x, best_y = coords_weighted
            
            except Exception as e:
                # Simpler approach - weighted centroid of amenities
                amenity_points = np.array([
                    (point.x, point.y) for point in cluster_amenities.geometry.centroid
                ])
                best_x, best_y = np.mean(amenity_points, axis=0)
            
            # Get census block that contains this point
            containing_blocks = [
                block for block in cluster_blocks.itertuples() 
                if block.geometry.contains(Point(best_x, best_y))
            ]
            
            if containing_blocks:
                geoid = containing_blocks[0].GEOID
                per_capita_income = containing_blocks[0].Per_Capita_Income
                transit_riders = containing_blocks[0].Commute_Public_Transit
            else:
                # If point is not in any block, use the nearest block's attributes
                dists = [Point(best_x, best_y).distance(block.geometry) 
                         for block in cluster_blocks.itertuples()]
                nearest_idx = np.argmin(dists)
                nearest_block = cluster_blocks.iloc[nearest_idx]
                geoid = nearest_block["GEOID"]
                per_capita_income = nearest_block["Per_Capita_Income"]
                transit_riders = nearest_block["Commute_Public_Transit"]
            
            bus_stops.append({
                "cluster_id": i,
                "geometry": Point(best_x, best_y),
                "score": np.max(density) if 'density' in locals() else len(cluster_amenities),
                "amenity_count": len(cluster_amenities),
                "transit_riders": transit_riders,
                "per_capita_income": per_capita_income,
                "GEOID": geoid
            })
            
        elif not cluster_blocks.empty:
            # If no amenities, fall back to highest-scoring block
            best_block = cluster_blocks.loc[cluster_blocks["combined_score"].idxmax()]
            bus_stops.append({
                "cluster_id": i,
                "geometry": best_block.geometry.centroid,
                "score": best_block["combined_score"],
                "amenity_count": best_block.get("amenity_count", 0),
                "transit_riders": best_block.get("Commute_Public_Transit", 0),
                "per_capita_income": best_block.get("Per_Capita_Income", 0),
                "GEOID": best_block["GEOID"]
            })
    
    progress_text.text("Creating bus stops GeoDataFrame...")
    progress_bar.progress(0.9)
    
    # Create GeoDataFrame of optimal bus stop locations
    bus_stops_gdf = gpd.GeoDataFrame(bus_stops, geometry="geometry", crs=census_data.crs)
    
    progress_text.text("Bus stop optimization complete!")
    progress_bar.progress(1.0)
    time.sleep(1)
    progress_text.empty()
    progress_bar.empty()
    
    return bus_stops_gdf, census_data, amenities_in_area

# Create an interactive map using folium
def create_interactive_map(census_blocks, vacant_lots=None, amenities=None, bus_stops=None, selected_block=None):
    """
    Create an interactive map with all data layers
    
    Parameters:
        census_blocks: GeoDataFrame of census blocks
        vacant_lots: GeoDataFrame of vacant lots
        amenities: GeoDataFrame of amenities
        bus_stops: GeoDataFrame of bus stops
        selected_block: GeoID of selected census block
        
    Returns:
        Folium map object
    """
    # Ensure consistent CRS (convert to WGS84 for folium)
    census_blocks = census_blocks.to_crs('EPSG:4326')
    
    if vacant_lots is not None:
        vacant_lots = vacant_lots.to_crs('EPSG:4326')
    
    if amenities is not None:
        amenities = amenities.to_crs('EPSG:4326')
    
    if bus_stops is not None:
        bus_stops = bus_stops.to_crs('EPSG:4326')
    
    # Create map centered on Baltimore
    m = folium.Map(
        location=[39.2904, -76.6122],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add census blocks as polygons
    census_blocks_layer = folium.FeatureGroup(name="Census Blocks")
    
    for idx, row in census_blocks.iterrows():
        popup_content = f"""
        <b>GEOID:</b> {row['GEOID']}<br>
        <b>Population:</b> {row.get('Total_Population', 'N/A')}<br>
        <b>Income:</b> ${int(row.get('Per_Capita_Income', 0)):,}<br>
        <b>Transit Commuters:</b> {int(row.get('Commute_Public_Transit', 0))}
        """
        
        # Highlight selected block
        if selected_block and row['GEOID'] == selected_block:
            color = 'red'
            fill_color = 'red'
            fill_opacity = 0.4
            weight = 3
        else:
            color = 'blue'
            fill_color = 'blue'
            fill_opacity = 0.1
            weight = 1
        
        # Convert geometry to GeoJSON
        geo_j = folium.GeoJson(
            row['geometry'].__geo_interface__,
            style_function=lambda x: {
                'color': color,
                'fillColor': fill_color,
                'fillOpacity': fill_opacity,
                'weight': weight
            }
        )
        
        # Add popup
        popup = folium.Popup(popup_content, max_width=300)
        geo_j.add_child(popup)
        
        census_blocks_layer.add_child(geo_j)
    
    m.add_child(census_blocks_layer)
    
    # Add vacant lots as markers
    if vacant_lots is not None:
        vacant_lots_layer = folium.FeatureGroup(name="Vacant Lots")
        
        for idx, row in vacant_lots.iterrows():
            popup_content = f"Vacant Lot ID: {idx}"
            
            # Get centroid
            centroid = row['geometry'].centroid
            
            # Add marker
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=3,
                color='gray',
                fill=True,
                fill_color='gray',
                fill_opacity=0.7,
                popup=popup_content
            ).add_to(vacant_lots_layer)
        
        m.add_child(vacant_lots_layer)
    
    # Add amenities with different colors for different groups
    if amenities is not None:
        # Create a marker cluster for amenities
        amenities_layer = folium.FeatureGroup(name="Amenities")
        
        # Define color mapping for amenity groups
        amenity_colors = {
            'hospital': 'red',
            'pharmacy': 'purple',
            'school': 'blue',
            'university': 'darkblue',
            'fresh food': 'green',
            'transit': 'orange',
            'restaurant': 'cadetblue',
            'personal care': 'pink',
            'recreation': 'darkgreen',
            'bank services': 'darkpurple',
            'lodging': 'lightblue',
            'civic': 'darkred'
        }
        
        # Add markers for each amenity
        for idx, row in amenities.iterrows():
            amenity_group = row.get('amenity_group', 'other')
            color = amenity_colors.get(amenity_group, 'gray')
            
            popup_content = f"""
            <b>Amenity Type:</b> {amenity_group}<br>
            <b>Name:</b> {row.get('name', 'N/A')}
            """
            
            # Get centroid
            centroid = row['geometry'].centroid
            
            # Add marker
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_content
            ).add_to(amenities_layer)
        
        m.add_child(amenities_layer)
    
    # Add bus stops
    if bus_stops is not None:
        bus_stops_layer = folium.FeatureGroup(name="Bus Stops")
        
        for idx, row in bus_stops.iterrows():
            popup_content = f"""
            <b>Bus Stop ID:</b> {idx}<br>
            <b>Census Block:</b> {row['GEOID']}<br>
            <b>Nearby Amenities:</b> {int(row['amenity_count'])}<br>
            <b>Transit Riders:</b> {int(row['transit_riders'])}<br>
            <b>Income:</b> ${int(row['per_capita_income']):,}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['geometry'].y, row['geometry'].x],
                radius=8,
                color='black',
                fill=True,
                fill_color='black',
                fill_opacity=0.7,
                popup=popup_content
            ).add_to(bus_stops_layer)
        
        m.add_child(bus_stops_layer)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Main function for the amenity optimization tab
def run_amenity_optimization_tab():
    """Handle all UI and logic for the amenity optimization tab"""
    st.header("1. Amenity Placement Optimization")
    
    # Load data if not already loaded
    if st.session_state.census_blocks is None or st.session_state.vacant_lots is None or st.session_state.amenities is None:
        with st.spinner("Loading data..."):
            st.session_state.census_blocks, st.session_state.vacant_lots, st.session_state.amenities = load_data()
    
    if st.session_state.graph is None:
        with st.spinner("Loading street network..."):
            st.session_state.graph = load_graph()
    
    # If data loading failed, show error and return
    if st.session_state.census_blocks is None or st.session_state.vacant_lots is None or st.session_state.amenities is None or st.session_state.graph is None:
        st.error("Failed to load required data. Please check the data files and try again.")
        return
    
    # Step 1: Select census block group
    st.subheader("Step 1: Select Census Block Group")
    
    # Create a dropdown to select census block group
    census_block_options = st.session_state.census_blocks['GEOID'].astype(str).tolist()
    
    if st.session_state.selected_block_group is None:
        default_index = 0
    else:
        default_index = census_block_options.index(st.session_state.selected_block_group) if st.session_state.selected_block_group in census_block_options else 0
    
    selected_block = st.selectbox(
        "Select a census block group to define your study area:",
        census_block_options,
        index=default_index
    )
    
    # Define study area if block selected or changed
    if st.session_state.selected_block_group != selected_block:
        st.session_state.selected_block_group = selected_block
        with st.spinner("Defining study area..."):
            st.session_state.study_area = define_study_area(
                st.session_state.census_blocks, 
                selected_block,
                radius_miles=3
            )
    
    # Step 2: Display study area and prepare for optimization
    if st.session_state.study_area is not None:
        st.subheader("Step 2: Study Area & Data Preparation")
        
        # Show statistics about the study area
        col1, col2 = st.columns(2)
        with col1:
            total_pop = st.session_state.study_area['Total_Population'].sum() if 'Total_Population' in st.session_state.study_area.columns else "N/A"
            st.metric("Total Population", f"{total_pop:,}" if isinstance(total_pop, (int, float)) else total_pop)
            
            total_blocks = len(st.session_state.study_area)
            st.metric("Census Blocks", total_blocks)
        
        with col2:
            avg_income = st.session_state.study_area['Per_Capita_Income'].mean() if 'Per_Capita_Income' in st.session_state.study_area.columns else "N/A"
            st.metric("Avg. Per Capita Income", f"${avg_income:,.2f}" if isinstance(avg_income, (int, float)) else avg_income)
            
            # Count vacant lots in the study area
            if 'vacant_lots_in_study_area' not in st.session_state:
                # Ensure consistent CRS
                vacant_lots_for_join = st.session_state.vacant_lots.to_crs(st.session_state.study_area.crs)
                
                # Find points in polygons
                vacant_lots_in_area = gpd.sjoin(
                    vacant_lots_for_join,
                    st.session_state.study_area,
                    predicate='within'
                )
                
                st.session_state.vacant_lots_in_study_area = vacant_lots_in_area
            
            st.metric("Vacant Lots", len(st.session_state.vacant_lots_in_study_area))
        
        # Display interactive map
        st.subheader("Study Area Map")
        
        # Calculate existing amenities in the study area
        amenities_in_area = gpd.sjoin(
            st.session_state.amenities.to_crs(st.session_state.study_area.crs),
            st.session_state.study_area,
            predicate='within'
        ).drop_duplicates()
        
        # Add any placed amenities
        if st.session_state.placed_amenities is not None:
            amenities_in_area = pd.concat([amenities_in_area, st.session_state.placed_amenities])
        
        # Create and display the map
        map = create_interactive_map(
            st.session_state.study_area,
            st.session_state.vacant_lots_in_study_area,
            amenities_in_area,
            selected_block=selected_block
        )
        folium_static(map, width=800, height=500)
        
        # Step 3: Compute distance matrices if not already done
        if st.session_state.distance_df is None:
            st.subheader("Step 3: Compute Distance Matrices")
            
            if st.button("Compute Distance Matrices"):
                with st.spinner("Preparing vacant lots..."):
                    vacant_lots_proj = prepare_vacant_lots(st.session_state.vacant_lots_in_study_area, st.session_state.graph)
                
                with st.spinner("Preparing census blocks..."):
                    census_blocks_proj = prepare_census_blocks(st.session_state.study_area, st.session_state.graph)
                
                with st.spinner("Computing distances..."):
                    st.session_state.distance_df = compute_fast_distances_dijkstra_tree(
                        st.session_state.graph,
                        census_blocks_proj,
                        vacant_lots_proj
                    )
                
                with st.spinner("Computing distances to existing amenities..."):
                    if len(amenities_in_area) > 0:
                        amenities_proj = prepare_vacant_lots(amenities_in_area, st.session_state.graph)
                        st.session_state.distance_df_amenities = compute_fast_distances_dijkstra_tree(
                            st.session_state.graph,
                            census_blocks_proj,
                            amenities_proj
                        )
                    else:
                        st.session_state.distance_df_amenities = pd.DataFrame(columns=[
                            'census_block_index', 'vacant_lot_index', 'vacant_node',
                            'network_dist_m', 'euclidean_offset_m', 'total_dist_m'
                        ])
                
                st.success("Distance matrices computed successfully!")
        
        # Step 4: Run optimization
        if st.session_state.distance_df is not None:
            st.subheader("Step 4: Run Amenity Placement Optimization")
            
            # Select amenity type
            available_groups = sorted([
                g for g in st.session_state.amenities['amenity_group'].dropna().unique()
                if isinstance(g, str)
            ])
            
            amenity_group = st.selectbox(
                "Select amenity type to optimize:",
                available_groups
            )
            
            # Select demographic priority
            demographic_columns = [col for col in st.session_state.study_area.columns 
                                  if col not in ['geometry', 'GEOID', 'centroid', 'lon', 'lat', 'nearest_node']]
            
            demographic_priority = st.selectbox(
                "Select demographic variable to prioritize (optional):",
                ["None"] + demographic_columns
            )
            
            if demographic_priority == "None":
                demographic_priority = None
            
            # Get suggested amenity count
            suggested_new, num_existing, total_pop = get_suggested_amenity_count(
                st.session_state.study_area,
                amenities_in_area,
                amenity_group
            )
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Study Area Population", f"{total_pop:,}")
            with col2:
                st.metric(f"Existing {amenity_group} Amenities", num_existing)
            with col3:
                st.metric("Suggested New Amenities", suggested_new)
            
            # Input for number of amenities to add
            num_new_amenities = st.number_input(
                f"Number of new '{amenity_group}' amenities to add:",
                min_value=1,
                max_value=20,
                value=suggested_new
            )
            
            # Advanced options in expander
            with st.expander("Advanced Optimization Settings"):
                generations = st.slider(
                    "Number of generations:",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
                
                population_size = st.slider(
                    "Population size:",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10
                )
                
                mutation_rate = st.slider(
                    "Mutation rate:",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.1,
                    step=0.01
                )
            
            # Run optimization button
            if st.button("Run Optimization"):
                with st.spinner("Running optimization..."):
                    # If we have already placed some amenities, update the distance_df_amenities
                    if st.session_state.placed_amenities is not None:
                        # Ensure we don't duplicate already placed amenities
                        if 'assigned_amenity_group' in st.session_state.placed_amenities.columns:
                            existing_of_type = st.session_state.placed_amenities[
                                st.session_state.placed_amenities['assigned_amenity_group'] == amenity_group
                            ]
                            
                            # Prepare these for distance calculation
                            if len(existing_of_type) > 0:
                                existing_proj = prepare_vacant_lots(existing_of_type, st.session_state.graph)
                                census_blocks_proj = prepare_census_blocks(st.session_state.study_area, st.session_state.graph)
                                
                                existing_distances = compute_fast_distances_dijkstra_tree(
                                    st.session_state.graph,
                                    census_blocks_proj,
                                    existing_proj
                                )
                                
                                # Combine with existing amenities distances
                                st.session_state.distance_df_amenities = pd.concat([
                                    st.session_state.distance_df_amenities,
                                    existing_distances
                                ])
                    
                    # Run optimization
                    best_lots_df, baseline, best_score = optimize_amenity_placement_genetic_constrained(
                        st.session_state.distance_df,
                        st.session_state.distance_df_amenities,
                        st.session_state.study_area,
                        st.session_state.vacant_lots_in_study_area,
                        amenities_in_area,
                        amenity_group,
                        demographic_priority,
                        num_new_amenities,
                        generations,
                        population_size,
                        mutation_rate
                    )
                    
                    # Add the new amenities to placed_amenities
                    if st.session_state.placed_amenities is None:
                        st.session_state.placed_amenities = best_lots_df
                    else:
                        st.session_state.placed_amenities = pd.concat([
                            st.session_state.placed_amenities,
                            best_lots_df
                        ])
                    
                    # Update optimization history
                    st.session_state.optimization_history.append({
                        'amenity_group': amenity_group,
                        'num_placed': len(best_lots_df),
                        'baseline_distance': baseline,
                        'optimized_distance': best_score,
                        'improvement': baseline - best_score if baseline != float('inf') else "N/A"
                    })
                    
                    st.session_state.optimization_complete = True
                
                # Show results
                st.success(f"Successfully placed {len(best_lots_df)} new {amenity_group} amenities!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Distance Before", f"{baseline/len(st.session_state.study_area):.1f} m" if baseline != float('inf') else "N/A")
                with col2:
                    st.metric("Average Distance After", f"{best_score/len(st.session_state.study_area):.1f} m")
                with col3:
                    improvement = (baseline - best_score)/len(st.session_state.study_area) if baseline != float('inf') else 0
                    st.metric("Average Improvement", f"{improvement:.1f} m", delta=f"{improvement:.1f}")
                
                # Update the map with new amenities
                all_amenities = pd.concat([amenities_in_area, best_lots_df])
                
                map = create_interactive_map(
                    st.session_state.study_area,
                    st.session_state.vacant_lots_in_study_area,
                    all_amenities,
                    selected_block=selected_block
                )
                folium_static(map, width=800, height=500)
            
            # If we've completed at least one optimization, show history
            if st.session_state.optimization_complete:
                st.subheader("Optimization History")
                
                history_df = pd.DataFrame(st.session_state.optimization_history)
                st.table(history_df)
                
                # Button to move to bus stop optimization
                if st.button("Proceed to Bus Stop Optimization"):
                    st.session_state.active_tab = "bus_stop_optimization"
                    st.experimental_rerun()

# Main function for the bus stop optimization tab
def run_bus_stop_optimization_tab():
    """Handle all UI and logic for the bus stop optimization tab"""
    st.header("2. Bus Stop Placement Optimization")
    
    # Check if amenity optimization is complete
    if not st.session_state.optimization_complete:
        st.warning("Please complete amenity optimization first before proceeding to bus stop optimization.")
        
        if st.button("Go to Amenity Optimization"):
            st.session_state.active_tab = "amenity_optimization"
            st.experimental_rerun()
            
        return
    
    st.subheader("Optimize Bus Stop Locations")
    st.markdown("""
    This step will find optimal locations for bus stops based on:
    - Proximity to amenities (especially those you've just placed)
    - Public transit commuting patterns
    - Income levels (prioritizing lower-income areas)
    """)
    
    # Input parameters for bus stop optimization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_bus_stops = st.number_input(
            "Number of bus stops to place:",
            min_value=5,
            max_value=50,
            value=15,
            step=1
        )
    
    with col2:
        amenity_weight = st.slider(
            "Amenity weight:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col3:
        income_weight = st.slider(
            "Income weight (prioritize lower income):",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.1
        )
    
    # Calculate transit weight to sum to 1.0
    transit_weight = 1.0 - amenity_weight - income_weight
    st.write(f"Transit commuting weight: {transit_weight:.1f}")
    
    # Run optimization button
    if st.button("Optimize Bus Stop Locations"):
        with st.spinner("Running bus stop optimization..."):
            # Combine existing amenities with placed amenities
            amenities_in_area = gpd.sjoin(
                st.session_state.amenities.to_crs(st.session_state.study_area.crs),
                st.session_state.study_area,
                predicate='within'
            ).drop_duplicates()
            
            if st.session_state.placed_amenities is not None:
                all_amenities = pd.concat([amenities_in_area, st.session_state.placed_amenities])
            else:
                all_amenities = amenities_in_area
            
            # Get eligible block group IDs
            eligible_block_groups = st.session_state.study_area['GEOID'].tolist()
            
            # Run optimization
            bus_stops_gdf, census_data, amenities_in_area = find_optimal_bus_stops(
                st.session_state.study_area,
                all_amenities,
                eligible_block_groups,
                n_bus_stops=n_bus_stops,
                amenity_weight=amenity_weight,
                drive_weight=transit_weight,
                income_weight=income_weight
            )
            
            # Store the results
            st.session_state.bus_stops = bus_stops_gdf
            st.session_state.optimize_bus_stops = True
        
        st.success(f"Successfully placed {len(bus_stops_gdf)} bus stops!")
        
        # Display map with bus stops
        st.subheader("Bus Stop Locations Map")
        
        map = create_interactive_map(
            st.session_state.study_area,
            None,  # No need to show vacant lots here
            all_amenities,
            bus_stops_gdf,
            selected_block=st.session_state.selected_block_group
        )
        folium_static(map, width=800, height=500)
        
        # Display bus stop details
        st.subheader("Bus Stop Details")
        
        # Format bus stop data for display
        display_cols = [
            'cluster_id', 'GEOID', 'amenity_count', 
            'transit_riders', 'per_capita_income'
        ]
        
        display_df = bus_stops_gdf[display_cols].copy()
        display_df.columns = [
            'Cluster ID', 'Census Block', 'Nearby Amenities',
            'Transit Riders', 'Per Capita Income'
        ]
        
        # Format income
        display_df['Per Capita Income'] = display_df['Per Capita Income'].apply(lambda x: f"${int(x):,}")
        
        st.table(display_df)
        
        # Visualization
        st.subheader("Visualizations")
        
        # Create optimized bus stop visualizations
        with st.spinner("Creating visualizations..."):
            # Create figures
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Convert to Web Mercator for basemap
            census_data_web = census_data.to_crs(epsg=3857)
            amenities_web = amenities_in_area.to_crs(epsg=3857)
            bus_stops_web = bus_stops_gdf.to_crs(epsg=3857)
            
            # Plot census blocks colored by amenity count
            census_data_web.plot(column="amenity_count", cmap="Purples", 
                            legend=True, ax=ax, legend_kwds={'label': "Amenity Count"})
            
            # Plot optimal bus stops
            bus_stops_web.plot(color="black", edgecolors='white', markersize=100, ax=ax, label="Optimal Bus Stops")

            # Plot amenities with color by group if possible
            if 'amenity_group' in amenities_web.columns:
                unique_amenity_groups = sorted(amenities_web['amenity_group'].unique())
                
                # Create a custom color map
                n_colors = len(unique_amenity_groups)
                colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
                
                # Create handles for the legend
                legend_elements = []
                
                # Plot each amenity group with its own color and add to legend
                for i, amenity_type in enumerate(unique_amenity_groups):
                    # Get subset of amenities for this type
                    subset = amenities_web[amenities_web['amenity_group'] == amenity_type]
                    
                    # Plot with consistent color
                    color = colors[i]
                    subset.plot(color=color, markersize=20, alpha=0.7, ax=ax)
                    
                    # Add to legend elements
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=10, label=amenity_type))
                
                # Add bus stops to legend elements
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='black', markersize=10, label="Bus Stops"))
                
                # Add the custom legend
                ax.legend(handles=legend_elements, title="Amenity Types", 
                         loc='upper right', fontsize=10)
            else:
                # Fallback if amenity_group column doesn't exist
                amenities_web.plot(color='blue', markersize=20, alpha=0.7, ax=ax, label="Amenities")
                plt.legend(loc='upper right')
            
            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            
            # Turn off axes (remove grid numbers)
            ax.axis('off')
            
            plt.title("Amenity Density and Bus Stop Locations in Baltimore", fontsize=16)
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
        
        # Download button for bus stop locations
        bus_stop_csv = bus_stops_gdf.drop(columns=['geometry']).to_csv().encode('utf-8')
        st.download_button(
            label="Download Bus Stop Data (CSV)",
            data=bus_stop_csv,
            file_name='optimal_bus_stops.csv',
            mime='text/csv',
        )

# Main app logic
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Baltimore_Seal.svg/1200px-Baltimore_Seal.svg.png", width=100)
        st.title("Optimization Controls")
        
        # Navigation
        selected_tab = st.radio(
            "Navigation",
            ["Amenity Optimization", "Bus Stop Optimization"]
        )
        
        if selected_tab == "Amenity Optimization":
            st.session_state.active_tab = "amenity_optimization"
        else:
            st.session_state.active_tab = "bus_stop_optimization"
        
        # Reset button
        if st.button("Reset Application"):
            # Reset all session state variables
            for key in st.session_state.keys():
                if key != "graph" and key != "census_blocks" and key != "vacant_lots" and key != "amenities":
                    st.session_state[key] = None
            
            st.session_state.optimization_complete = False
            st.session_state.optimize_bus_stops = False
            st.session_state.optimization_history = []
            st.session_state.active_tab = "amenity_optimization"
            
            st.success("Application reset successfully!")
            st.experimental_rerun()
    
    # Run the selected tab
    if st.session_state.active_tab == "amenity_optimization":
        with tab1:
            run_amenity_optimization_tab()
    else:
        with tab2:
            run_bus_stop_optimization_tab()

if __name__ == "__main__":
    main()