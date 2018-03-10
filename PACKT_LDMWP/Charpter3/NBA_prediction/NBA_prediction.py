import pandas as pd
import  numpy as np

def count_magnitude(series):#to tell weather the score is 3-digit or 2-digit
    list=[]
    for i in series:
        num = 0
        for j in i:
            num += 1
        list.append(num)
    return list

def is_equal_magnitude():#judge if the magnitude is the same and return the boolean value in a list
    same_magnitude=[]
    num_v_l=count_magnitude(dataset["VisitorPTS:"])
    # print(num_v_l)
    num_h_l=count_magnitude(dataset["Home TeamPTS:"])
    # print(num_h_l)
    judge=zip(num_v_l,num_h_l)
    for data in judge:
        same_magnitude.append(data[0]==data[1])
    # print(same_magnitude)
    return same_magnitude


'''
# dataset["HomeWin:"]=dataset["VisitorPTS:"]<dataset["Home TeamPTS:"]
# when comparing a 2-digit to a 3-digit(like 95 to 103)pandas will return a reverse value
'''
def Home_Win():#if the score is the same in magnitude pandas can compare them (if not the result need to be reversed)
    list=[]
    for i in range(1237):
        if(dataset.ix[i]["Same Magnitude:"]):
            list.append(dataset.ix[i]["VisitorPTS:"]<dataset.ix[i]["Home TeamPTS:"])
        else:
            list.append(not dataset.ix[i]["VisitorPTS:"]<dataset.ix[i]["Home TeamPTS:"])
    return list
'''
# print(standing[standing["Team"]==home_team]["Rk"].values)#enpty list is included
'''
def get_rank(series):
    try:
        return series["Rk"].values[0]
    except:
        return -1
def accuracy_test(x):
    #bring in the decision tree
    from sklearn.tree import DecisionTreeClassifier
    clf=DecisionTreeClassifier(random_state=14)
    # x_homehigher=dataset[["HomeLastWin","VisitorLastWin","HomeRanksHigher"]].values
    # print(x_previouswins)

    #test the acurracy
    from sklearn.cross_validation import cross_val_score
    scores=cross_val_score(clf,x,y_true,scoring='accuracy')
    print("Accuracy: {0:.1f}%".format(np.mean(scores)*100))


if __name__ == '__main__':
##load the data
    data_filename='/home/konroy/Documents/konroy/learning data mining with python/leagues_NBA_2014_games_games.csv'
    dataset=pd.read_csv(data_filename)

##clean the data and reset the title for each list
    dataset.columns=["Date:","Time:","Score Type:","Visitor Team:","VisitorPTS:","Home Team:","Home TeamPTS:","OT?:","Notes:"]

##extract new feature
    #part1:  which team had won the other
    #1: Same Magnitude---for 'HomeWin'
    dataset["Same Magnitude:"]=is_equal_magnitude()
    #2: HomeWin----judge weather home team win
    dataset["HomeWin:"]=Home_Win()
    # print(dataset.ix[21:25])
    y_true=dataset["HomeWin:"].values

    #3: HomeLastWin   & VisitorLastWin---mark down weather the team had won in the last game
    dataset["HomeLastWin"]=False
    dataset["VisitorLastWin"]=False
    from collections import defaultdict
    won_last=defaultdict(int)
    for index,row in dataset.iterrows():#index:from 0 to 1236   row: the data from dataset
        home_team=row["Home Team:"]
        visitor_team=row["Visitor Team:"]

        won_last[home_team] = row["HomeWin:"]
        won_last[visitor_team] = not row["HomeWin:"]

        row["HomeLastWin"]=won_last[home_team]
        row["VisitorLastWin"]=won_last[visitor_team]
        #put the boolean value in dataset
        # dataset.ix[index]["HomeLastWin"]=row["HomeLastWin"]
        # dataset.ix[index]["VisitorLastWin"] = row["VisitorLastWin"]
        dataset.ix[index]=row
    #test
    # print(dataset.ix[21:25])
    # test part1
    x=dataset[["HomeLastWin","VisitorLastWin"]].values
    accuracy_test(x)


    #part2: which team's rank is heigher
    #4: HomeTeamRanksHigher---by refering to some other data(from 2013), figure out  the higher team rank between visitor and home
    standing_filename="/home/konroy/Documents/konroy/learning data mining with python/leagues_NBA_2013_standings_expanded-standings.csv"
    standing=pd.read_csv(standing_filename,skiprows=[0,1])
    # print(standing)
    dataset["HomeRanksHigher"]=0
    for index,row in dataset.iterrows():
        home_team=row["Home Team:"]
        visitor_team=row["Visitor Team:"]
        #some of the teams has change their name
        if home_team=="New Orleans Pelicans":
            home_team="New Orleans Hornets"
        elif visitor_team=="New Orleans Pelicans":
            visitor_team="New Orleans Hornets"


        #campare the rank of home_team and visitor_team and update the values
        home_rank=get_rank(standing[standing["Team"]==home_team])#return the rank from standing that the Team(name)==home_team
        visitor_rank=get_rank(standing[standing["Team"]==visitor_team])#return the rank from standing that the Team(name)==visitor_team
        row["HomeRanksHigher"]=int(home_rank>visitor_rank)
        dataset.ix[index]=row
    # print(dataset.ix[2:5])
    # test part2
    x=dataset[["HomeLastWin","VisitorLastWin","HomeRanksHigher"]].values
    accuracy_test(x)


    #part3: home won last(without considering home team or visitor team)
    #5: HomeTeamWonLast---the situation for the last match
    last_match_winner=defaultdict(int)
    dataset["HomeTeamWonLast"]=0
    for index,row in dataset.iterrows():
        home_team = row["Home Team:"]
        visitor_team = row["Visitor Team:"]
        #as weather the team is playing in its own court is not taken into consideration, we can queue up the team by the letter in the team name
        teams=tuple(sorted([home_team,visitor_team]))#sort by the letters
        row["HomeTeamWonLast"]=1 if last_match_winner[teams]==row["Home Team:"] else 0
        dataset.ix[index]=row
        #update the dict last_match_winner
        winner=row["Home Team:"] if row["HomeWin:"] else row["Visitor Team:"]
        last_match_winner[teams]=winner
    print(dataset.ix[22:25])

    #test part3
    x=dataset[["HomeRanksHigher","HomeTeamWonLast"]].values
    accuracy_test(x)










'''
#sample of using zip and compareing
list1=[1,2,3]
list2=[4,5,6]
list3=zip(list1,list2)
for i in list3:
    print (i[0]>i[1])
'''

'''
def to_int(series):
    list=[]
    for pts in series:
        try:
            list.append(int(pts))
        except:
            list.append(-1)
    series=list
'''