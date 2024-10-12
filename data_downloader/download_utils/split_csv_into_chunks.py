"""
Splits a .csv file into chunks

"""

import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_file",
    )

    parser.add_argument(
        "--number_of_chunks",
    )

    args = parser.parse_args()

    assert args.csv_file is not None, \
        'csv_file must be specified'
    
    assert args.number_of_chunks is not None, \
        'number_of_chunks must be specified'
    
    filepath = args.csv_file
    filename = os.path.basename(filepath)
    filedir = os.path.dirname(filepath)
    n_chunks = int(args.number_of_chunks)

    print(f'Splitting csv file in chunks of {n_chunks} records...')

    df = pd.read_csv(filepath)

    # split DataFrame into chunks
    list_df = [df[i:i+n_chunks] for i in range(0,len(df),n_chunks)]

    filename_no_extention = filename.split(os.extsep)[0]

    print(f'Total csv length: {len(df)}')
    for idx, cur_df in enumerate(list_df):
        output_filename = filename_no_extention + "_" + str(idx) + ".csv"
        print(f'[*] Length of {output_filename}: {len(cur_df)}')
        cur_df.to_csv(os.path.join(filedir, output_filename), index=False)


    print('Done.')

