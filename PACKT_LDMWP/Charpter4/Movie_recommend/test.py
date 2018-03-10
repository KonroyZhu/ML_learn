import pandas as pd
import os

data_folder="/home/konroy/Documents/konroy/learning data mining with python/ml-100k"
movie_name_filename=os.path.join(data_folder,"u.item")
movie_name_data=pd.read_csv(movie_name_filename,delimiter="|",header=None,encoding="mac-roman")

movie_name_data.columns=["MovieID","Title","Release Date","Video Release","IMDB","<UNK>","Action","Adventure","Animation",
                         "Children's","Comedy","Crime","Documentary",
                         "Drama","Fantasy","Film-Noir","Horror","Musical","Mestery","Romance","Sci-Fi","Thriller","Wat","Western"]
# print(movie_name_data[:5])

def get_movie_name(movie_id):
    title_object=movie_name_data[movie_name_data["MovieID"]==movie_id]["Title"]
    # print(title_object.values[0])
    title=title_object.values[0]
    return title


Movie=get_movie_name(3)
print(Movie)