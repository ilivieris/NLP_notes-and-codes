import pandas as pd


def load_dataset():

    filepath_dict = {'yelp': 'Data/yelp_labelled.txt',
                    'amazon': 'Data/amazon_cells_labelled.txt',
                    'imdb': 'Data/imdb_labelled.txt'}


    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

    return pd.concat(df_list)

