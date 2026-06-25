from src.data_loader import load_housig_data

if __name__ == '__main__':
    data_path  = '/home/briennavarro/Python-projects/housing-price-prediction/data/raw/Housing.csv'
    print('Running pipeline...')

    try : 
        df = load_housig_data(data_path)
        print('Data loaded successfully')
    except Exception as e:
        print(f'Error loading the data {e}')