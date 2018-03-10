import pandas as pd

standing_filename="/home/konroy/Documents/konroy/learning data mining with python/leagues_NBA_2013_standings_expanded-standings.csv"
standing=pd.read_csv(standing_filename,skiprows=[0,1])
print(standing)