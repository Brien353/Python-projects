""" Module that performs EDA, outlier detection and data cleaninig pipeline  so that ML models can use the information for proper training"""
import os

import matplotlib.pyplot as plt
from config.general_settings import PROCESSED_DATA_FOLDER, RAW_DATA_PATH
from sklearn.cluster import HDBSCAN
from src.data_utils import (
    clean_df,
    corr_mat,
    feat_eng,
    general_desc,
    load_data,
    num_cat_cols_lst,
    tuckey_outlier_detection,
)
from src.models.pipelines import projection_pipe
from src.viz_utils import box_plt, clust_plot, corr_map_plt, dist_plt, sc_plot


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

        print_header("Data loading")


        df = load_data(RAW_DATA_PATH)

        print("Data loaded successfully.")



        print_header("Data summary")

        general_desc(df)

        print("Data summary created successfully.")


        print_header("Initializing EDA for both numerical and categorical variables")

        print_header("Initializing numerical variables distribution Visualization")
        n_cols , _ = num_cat_cols_lst(df)

        for col in n_cols : 
            dist_plt(df, f"{col}",show_kde= True, log_scale= False)
            plt.title(f"Variable : {col} distribution")
            plt.show()

        print_header("Initializing numerical variables boxplot Visualization")

        for col in n_cols:
            box_plt(df, f"{col}", log_scale= False)
            plt.title(f"Variable : {col} box plot")
            plt.show()
    
        print_header("Calculating and Displaying Correlation Matrix ")

        correlation_matrix = corr_mat(df)

        print(correlation_matrix)

        corr_map_plt(correlation_matrix)
        plt.show()

        #===============================================
        # 2. Data cleaninig
        #===============================================

        import sys

        print_header('Data cleaninig')
        ds,o_num = tuckey_outlier_detection(df,'price', 1.5)
        print(ds,o_num)
        sys.exit()
        
        sc_plot(df,'area','price')
        plt.title("Area-Price scatter plot for extreme value trimming")
        plt.show()

        df, deleted_reg = clean_df(df)

        print(f'The percentage of deleted elements are : {deleted_reg : .2f}%')

        # ==========================================
        # 3. FEATURE ENGINEERING & UMAP projection
        # ==========================================

        print_header("Initializing Feature engineering")
        df_engineered = feat_eng(df)

        # Separate features and target
        X = df_engineered.drop(columns=["price"])

        
        print_header("Running UMAP projection pipeline")

        pipeline = projection_pipe()
        embedding = pipeline.fit_transform(X)

        print(f"Data projected succesfully, projected data dimensions are: {embedding.shape}")

        # ==========================================
        # 4. CLUSTERING (HDBSCAN)
        # ==========================================

        print_header("Initializing Clustering")

        print_header("Initializing clustering HDSCAN algorithm")

        clustering = HDBSCAN(
            min_cluster_size=30,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        print("HDBSCAN clustering was performed succesfully.")

        print_header("Labeling dataset using HDBSCAN clustering algorithm")

        cluster_labels = clustering.fit_predict(embedding)

        print("Labeling performed successfully.")

        print_header("Plotting clusters")

        clust_plot(embedding, cluster_labels)
        plt.show()

        print('Clustering visualization ended succesfully')

        # ==========================================
        # 5. PROFILING & SEGMENTATION INSIGHTS
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
            1: ("Mid-Tier Family Residences", "Moderate footprint, multi-bathroom homes with lower story heights and balanced utility."),
            2: ("Premium Vertical Luxury Estates", "Larguest footprint, multy-story layout (3.8 stories average), highest price tag and area")
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