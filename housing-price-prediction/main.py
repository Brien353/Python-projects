from src.data_loader import load_housig_data, data_statistics
from src.data_loader import general_desc
from src.graph import dist_plt, box_plt

if __name__ == '__main__':
    data_path  = '/home/briennavarro/Python-projects/housing-price-prediction/data/raw/Housing.csv'
    print('Running pipeline...')

    try :
        #Loading data
        df = load_housig_data(data_path)
        print('Data loaded successfully')
        box_plt(df, 'price',lg_scale_bol=True)
    except Exception as e:
        print(f'Error loading the data {e}')