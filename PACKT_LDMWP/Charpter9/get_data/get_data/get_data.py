
titles = {}#To save the id for the books by each writers
#The number unfound pages and captcha required pages is abandended
# titles['tarkington'] = [1098, 15855, 1983, 297, 402, 5798,
#                         8740, 980, 1158, 1611, 2326, 30092,
#                         483, 5949, 8867, 13275, 18259, 2595,
#                         3428, 5756, 6401, 9659]##Done!
#
# titles['dickens'] = [24022, 1392, 1414, 2324, 580,
#                      786, 888, 963, 27924, 1394, 1415, 15618,
#                      25985, 588, 807, 914, 967, 30127, 1400,
#                      1421, 16023, 28198, 644, 809, 917, 968, 1023,
#                      1406, 1422, 17879, 30368, 675, 810, 924, 98,
#                      1289, 1413, 1423, 17880, 32241, 699, 821, 927]
# #
# # titles['nesbit'] = [34219, 23661, 28804, 4378, 778, 20404, 28725,
# #                     33028, 4513, 794] ##Done!
#
# titles['doyle'] = [2349, 11656, 1644, 22357, 2347, 290, 34627,
#                    8394, 26153, 12555, 1661, 23059, 2348, 294, 355,
#                    5260, 8727, 10446, 126, 17398, 2343, 2350, 3070,
#                    356, 5317, 903, 10581, 13152, 2038, 2344, 244, 32536,
#                    423, 537, 108, 139, 2097, 2345, 24951, 32777, 4295,
#                    7964, 11413, 1638, 21768, 2346, 2845, 3289, 439, 834]
#
# titles['twain'] = [ 1213, 245, 30092, 3176, 3179, 3183, 3189, 74,
#                    86, 1086, 142, 2572, 3173, 3177, 3180, 3186, 3192,
#                    76, 91, 119, 1837, 2895, 3174, 3178, 3181, 3187, 3432,
#                    8525]

titles['burton'] = [2400, 5760, 6036, 7111, 8821,
                    18506, 4658, 5761, 6886, 7113]

titles['gaboriau'] = [1748, 1651, 2736, 3336, 4604, 4002, 2451,
                      305, 3802, 547]
assert len(titles) == 2

# # assert len(titles['tarkington']) == 22
# assert len(titles['dickens']) == 43
# # assert len(titles['nesbit']) == 10
# assert len(titles['doyle']) == 50
# assert len(titles['twain']) == 28
assert len(titles['burton']) == 10
assert len(titles['gaboriau']) == 10



#To create proper folders for each writer
#To get the article from the author and bookid
import requests
from time import sleep
import os
data_folder="/home/konroy/Documents/konroy/learning data mining with python/author-atrribute"
for author in titles:
    print("Downloading titles from {author}".format(author=author))
    # Make author's folder if not exists
    author_folder = os.path.join(data_folder, author)
    if not os.path.exists(author_folder):
        os.makedirs(author_folder)
    for bookid in titles[author]:
        print(bookid)
        url="http://www.gutenberg.org/files/"+str(bookid)+"/"+str(bookid)+".txt"
        print(" - " + url)
        filename = os.path.join(author_folder, "{id}.txt".format(id=bookid))

        # #Get the articles
        r=requests.get(url)
        article=r.text
        print(article)

        #Make the text file named 'bookid'(filename) by the following two line
        f=open(filename,'w')
        f.write(article)

        #Advoid reaching the lowest time limit
        sleep(60 * 10)
print("Download complete")