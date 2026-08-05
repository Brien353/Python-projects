import os
import sys
import matplotlib.pyplot as plt

# Local imports
from config.general_settings import PROCESSED_DATA_FOLDER, RAW_DATA_PATH
from sklearn.cluster import HDBSCAN
from src.data_utils import corr_mat, feat_eng, general_desc, load_data,outlier_imp
from src.models.pipelines import projection_pipe
from src.viz_utils import box_plt, clust_plot, corr_map_plt, dist_plt


def print_header(title: str) -> None:
    """Helper function to print clean, consistent headers to the console."""
    print(f"\n{'=' * 60}")
    print(f"{title.upper()}")
    print(f"{'=' * 60}")


def main():
    try:
        print_header("Program Initialized")

        # ==========================================
        # 1. DATA LOADING & EDA
        # ==========================================
        print_header("Exploratory Data Analysis")
        print("Loading data...")
        df = load_data(RAW_DATA_PATH)
        print("Data loaded successfully.")

        print("\n--- Generating Statistics Summary ---")
        general_desc(df)

        print("\n--- Generating EDA Visualizations ---")
        dist_plt(df, "price")
        plt.show()
        box_plt(df, "price")
        plt.show()

        print("\n--- Calculating and Displaying Correlation Matrix ---")
        correlation_matrix = corr_mat(df)
        print(correlation_matrix)
        corr_map_plt(correlation_matrix)
        plt.show()


        # ==========================================
        # 2. FEATURE ENGINEERING & UMAP projection
        # ==========================================
        print_header("Feature Engineering & Projection")
        df_engineered = feat_eng(df)

        # Separate features and target
        X = df_engineered.drop(columns=["price"])

        print("Running Projection Pipeline (UMAP)...")
        pipeline = projection_pipe()
        embedding = pipeline.fit_transform(X)
        print(f"Projected high-dimensional data into shape: {embedding.shape}")


        #============================================
        # 3. OUTLIER DETECTION
        #============================================
        
        # ==========================================
        # 3. CLUSTERING (HDBSCAN)
        # ==========================================
        print_header("Clustering & Labeling")
        clustering = HDBSCAN(
            min_cluster_size=20,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clustering.fit_predict(embedding)
        print("Labeling performed successfully.")

        print("\nPlotting clusters...")
        clust_plot(embedding, cluster_labels)
        plt.show()

        # ==========================================
        # 4. PROFILING & SEGMENTATION INSIGHTS
        # ==========================================
        print_header("Housing Market Segmentation Insights")
        
        # Add cluster labels to dataframe
        df_engineered['Cluster'] = cluster_labels
        
        # Split clustered data from anomalies (-1)
        clustered_df = df_engineered[df_engineered['Cluster'] != -1]
        numeric_cols = clustered_df.select_dtypes(include=['number']).columns
        housing_profiles = clustered_df[numeric_cols].groupby('Cluster').mean()
        
        print("The UMAP + HDBSCAN pipeline successfully uncovered distinct housing tiers.\n")

        tier_descriptions = {
            0: ("Entry-Level & Budget Homes", "Smallest footprint, single-bathroom starter homes with minimal parking."),
            1: ("Premium Multi-Story Luxury Estates", "Largest footprint, heavily vertical (3.7+ stories), high-end premium market."),
            2: ("Spacious Suburban Family Homes", "Highest bedroom/bathroom counts, spread horizontally. High utility for families.")
        }

        for cluster_id, row in housing_profiles.iterrows():
            tier_title, desc = tier_descriptions.get(
                cluster_id, 
                (f"Cluster {cluster_id}", "Unclassified structural tier.")
            )

            print(f"Cluster {cluster_id}: {tier_title}")
            print(f"Description:     {desc}")
            print(f"Avg Price:       ${row['price']:,.2f}")
            print(f"Avg Area:        {row['area']:,.1f} sq ft")
            print(f"Bed/Bath Ratio:  {row['bedrooms']:,.1f} beds / {row['bathrooms']:,.1f} baths")
            print(f"Stories/Parking: {row['stories']:,.1f} stories / {row['parking']:,.1f} spots")
            print("-" * 60)

        # ==========================================
        # 6. EXPORT
        # ==========================================

        print_header("Exporting Segmented Data")
        
        # Ensure processed data directory exists before saving
        os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)
        
        export_filename = "Housing_segmented.csv"
        full_export_path = os.path.join(PROCESSED_DATA_FOLDER, export_filename)
        
        df_engineered.to_csv(full_export_path, index=False)
        print(f"Segmented data successfully exported to: {full_export_path}")
        print("\nReady for Supervised Machine Learning Algorithms.")

    except FileNotFoundError:
        print(f"\n[ERROR]: The file at {RAW_DATA_PATH} could not be found.")
        print("Please check your config.py paths.")
    except Exception as e:
        print(f"\n[ERROR]: An unexpected error occurred during execution:\n{e}")


if __name__ == "__main__":
    main()