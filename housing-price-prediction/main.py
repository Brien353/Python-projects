from config.clust_config import cat_cl_feat, num_cl_feat
from src.graph import box_plt, corr_map_plt, dist_plt,clust_plot
from src.tools import corr_mat, data_statistics, feat_eng, general_desc, load_housig_data, num_cat_cols_lst
from src.pippe import projection_pipe
from sklearn.cluster import HDBSCAN
if __name__ == "__main__":
    data_path = "/home/briennavarro/Python-projects/housing-price-prediction/data/raw/Housing.csv"
    print("Running program...")

    try:
        # 1. Load Data
        print("Data is going to be loaded...")
        df = load_housig_data(data_path)
        print("Data loaded successfully.")

        # 2. Exploratory Data Analysis (EDA)
        print("\n--- Generating Statistics Resume ---")
        general_desc(df)

        print("\n--- Generating EDA Visualizations ---")
        # Generate basic distribution and boxplots for a key numerical feature (e.g., price)
        dist_plt(df, "price")
        box_plt(df, "price")

        # Generate correlation matrix and plot it
        print("\n--- Calculating and plot the correlation matrix ---")

        print("\n--- Showing the correlation matrix ---")
        correlation_matrix = corr_mat(df)
        print(correlation_matrix)

        print("\n---Displaying the correlation matrix----")
        corr_map_plt(correlation_matrix)

        # 3. Feature Engineering
        print("\n--- Feature Engineering ---")
        df_engineered = feat_eng(df)

        # 4. Dimensionality Reduction / Embedding Pipeline
        print("\n--- Running Projection Pipeline (UMAP)---")
        X = df_engineered.drop(columns=["price"])
        y = df_engineered["price"]
        # Instantiate and run your pipeline
        pipeline = projection_pipe()
        embedding = pipeline.fit_transform(X)
        print(f"Projected high-dimensional data into shape: {embedding.shape}")


        #Clustering
        print("\n---Performing clustering----")
        clustering = HDBSCAN(
        min_cluster_size = 20,
        min_samples = 5,
        metric='euclidean',
        cluster_selection_method='eom')
        print("\n---Clustering was performed succesfully---")

        #Labeling

        print("\n---Labeling using HDBSCAN---")
        cluster_labels = clustering.fit_predict(embedding)
        print("\n---Labeling was performed succesfully")

        print("\n---Plotting clustering---")
        clust_plot(embedding,cluster_labels)

        #Plot price variation among clusters

        print("\n---Calculating house price profiles---")
        profile_df = X.copy()
        profile_df['Cluster'] = cluster_labels
        profile_df['price'] = y
        profile_df = profile_df[profile_df['Cluster'] != -1]
        numeric_cols = profile_df.select_dtypes(include=['number']).columns
        print("\n---Showing Houses Profiles (Mean values per cluster) ---")
        housing_profiles = profile_df[numeric_cols].groupby('Cluster').mean()
        print(housing_profiles)

        print("\n" + "="*60)
        print("HOUSING MARKET SEGMENTATION INSIGHTS")
        print("="*60)

        print("The UMAP + HDBSCAN pipeline successfully uncovered three distinct")
        print("housing tiers based on structural features.\n")

        for cluster_id in housing_profiles.index:
            row = housing_profiles.loc[cluster_id]
            
            if cluster_id == 0:
                tier_name = "Cluster 0: Entry-Level & Budget Homes"
                desc = "Smallest footprint, single-bathroom starter homes with minimal parking."
            elif cluster_id == 1:
                tier_name = "Cluster 1: Premium Multi-Story Luxury Estates"
                desc = "Largest footprint, heavily vertical (3.7+ stories), high-end premium market."
            elif cluster_id == 2:
                tier_name = "Cluster 2: Spacious Suburban Family Homes"
                desc = "Highest bedroom/bathroom counts, spread horizontally. High utility for families."
            else:
                tier_name = f"Cluster {cluster_id}"
                desc = "Unclassified structural tier."

            print(f"{tier_name}")
            print(f"Avg Price:       ${row['price']:,.2f}")
            print(f"Avg Area:        {row['area']:,.1f} sq ft")
            print(f"Bed/Bath Ratio:  {row['bedrooms']:,.1f} beds / {row['bathrooms']:,.1f} baths")
            print(f"Stories/Parking: {row['stories']:,.1f} stories / {row['parking']:,.1f} spots")
            print(f"Profile Summary: {desc}")
            print("-" * 60)

            








    except FileNotFoundError:
        print(f"Error: The file at {data_path} could not be found.")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")