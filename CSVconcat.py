import pandas as pd

filename='features_part_1.csv'
filename2='features_part_2.csv'
saveName='features_12.csv'

csv1 = pd.read_csv(filename,error_bad_lines=False,header=None)
csv2 = pd.read_csv(filename2,error_bad_lines=False,header=None)
csv3 = pd.concat([csv1,csv2])

csv3.to_csv(saveName, sep=',', index=False)


