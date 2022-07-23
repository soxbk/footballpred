# Data Preprocessing

import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt

# dataSet = pd.read_csv('NFLGameStatsO.csv')
# teamGameStats = pd.DataFrame( columns = ['GameId', 'GameDate', 'Team', 'PassingYards', 'RushingYards', 'PassingYardsAllowedDefense', 'RushingYardsAllowedDefense', 'TurnoversCreated', '3rdDownsConverted', 'Sacks', 'Points', 'WonGame'])
# teamGameStats = teamGameStats.apply(pd.to_numeric).dropna()
# for row in  range(len(dataSet)):
#     if dataSet.loc[row, 'game_id'] not in set(teamGameStats['GameId']):
#         if teamGameStats.empty == False:
#           if teamGameStats.loc[homeTeamIndex, 'Points'] > teamGameStats.loc[awayTeamIndex, 'Points']:
#             teamGameStats.loc[homeTeamIndex, 'WonGame'] = 1
#             teamGameStats.loc[awayTeamIndex, 'WonGame'] = 0
#           if teamGameStats.loc[homeTeamIndex, 'Points'] < teamGameStats.loc[awayTeamIndex, 'Points']:
#             teamGameStats.loc[awayTeamIndex, 'WonGame'] = 1
#             teamGameStats.loc[homeTeamIndex, 'WonGame'] = 0
#         teamGameStatsNewRow = pd.DataFrame(
#             {'GameId': [dataSet.loc[row, 'game_id']], 'Team': [dataSet.loc[row, 'home_team']], 'GameDate': [dataSet.loc[row, 'game_date']], 'PassingYards': 0, 'RushingYards': 0,'PassingYardsAllowedDefense': 0, 'RushingYardsAllowedDefense': 0,
#              'TurnoversCreated': 0, '3rdDownsConverted': 0, 'Sacks': 0, 'Points': 0})
#         teamGameStats = pd.concat([teamGameStats, teamGameStatsNewRow], ignore_index=True)
#         teamGameStats.reset_index()
#         homeTeamIndex = teamGameStats.last_valid_index()
#         teamGameStatsNewRow = pd.DataFrame(
#             {'GameId': [dataSet.loc[row, 'game_id']], 'Team': [dataSet.loc[row, 'away_team']], 'GameDate': [dataSet.loc[row, 'game_date']], 'PassingYards': 0, 'RushingYards': 0, 'PassingYardsAllowedDefense': 0, 'RushingYardsAllowedDefense': 0,
#              'TurnoversCreated': 0, '3rdDownsConverted': 0, 'Sacks': 0, 'Points': 0})
#         teamGameStats = pd.concat([teamGameStats, teamGameStatsNewRow], ignore_index=True)
#         teamGameStats.reset_index()
#         awayTeamIndex = teamGameStats.last_valid_index()
#     if dataSet.loc[row, 'home_team'] == dataSet.loc[row, 'posteam']:
#         if dataSet.loc[row, 'play_type'] == 'pass':
#           teamGameStats.loc[homeTeamIndex, 'PassingYards'] = teamGameStats.loc[homeTeamIndex, 'PassingYards'] + dataSet.loc[row, 'yards_gained']
#           teamGameStats.loc[awayTeamIndex, 'PassingYardsAllowedDefense'] = teamGameStats.loc[awayTeamIndex, 'PassingYardsAllowedDefense'] + dataSet.loc[row, 'yards_gained']
#         if dataSet.loc[row, 'play_type'] == 'run':
#           teamGameStats.loc[homeTeamIndex, 'RushingYards'] = teamGameStats.loc[homeTeamIndex, 'RushingYards'] + dataSet.loc[row, 'yards_gained']
#           teamGameStats.loc[awayTeamIndex, 'RushingYardsAllowedDefense'] = teamGameStats.loc[awayTeamIndex, 'RushingYardsAllowedDefense'] + dataSet.loc[row, 'yards_gained']
#         if dataSet.loc[row, 'fumble'] == 1 or dataSet.loc[row, 'interception'] == 1:
#           teamGameStats.loc[awayTeamIndex, 'TurnoversCreated'] = teamGameStats.loc[awayTeamIndex, 'TurnoversCreated'] + dataSet.loc[row, 'fumble'] + dataSet.loc[row, 'interception']
#         if dataSet.loc[row, 'third_down_converted'] == 1:
#           teamGameStats.loc[homeTeamIndex, '3rdDownsConverted'] += 1
#         if dataSet.loc[row, 'sack'] == 1:
#           teamGameStats.loc[awayTeamIndex, 'Sacks'] += 1
#         if dataSet.loc[row, 'total_home_score'] >= 0:
#           teamGameStats.loc[homeTeamIndex, 'Points'] = dataSet.loc[row, 'total_home_score']
#     if dataSet.loc[row, 'away_team'] == dataSet.loc[row, 'posteam']:  
#         if dataSet.loc[row, 'play_type'] == 'pass':
#           teamGameStats.loc[awayTeamIndex, 'PassingYards'] = teamGameStats.loc[awayTeamIndex, 'PassingYards'] + dataSet.loc[row, 'yards_gained']
#           teamGameStats.loc[homeTeamIndex, 'PassingYardsAllowedDefense'] = teamGameStats.loc[homeTeamIndex, 'PassingYardsAllowedDefense'] + dataSet.loc[row, 'yards_gained']
#         if dataSet.loc[row, 'play_type'] == 'run':
#           teamGameStats.loc[awayTeamIndex, 'RushingYards'] = teamGameStats.loc[awayTeamIndex, 'RushingYards'] + dataSet.loc[row, 'yards_gained']
#           teamGameStats.loc[homeTeamIndex, 'RushingYardsAllowedDefense'] = teamGameStats.loc[homeTeamIndex, 'RushingYardsAllowedDefense'] + dataSet.loc[row, 'yards_gained']
#         if dataSet.loc[row, 'fumble'] == 1 or dataSet.loc[row, 'interception'] == 1:
#           teamGameStats.loc[homeTeamIndex, 'TurnoversCreated'] = teamGameStats.loc[homeTeamIndex, 'TurnoversCreated'] + dataSet.loc[row, 'fumble'] + dataSet.loc[row, 'interception']
#         if dataSet.loc[row, 'third_down_converted'] == 1:
#           teamGameStats.loc[awayTeamIndex, '3rdDownsConverted'] += 1
#         if dataSet.loc[row, 'sack'] == 1:
#           teamGameStats.loc[homeTeamIndex, 'Sacks'] += 1
#         if dataSet.loc[row, 'total_home_score'] >= 0:
#           teamGameStats.loc[awayTeamIndex, 'Points'] = dataSet.loc[row, 'total_away_score']
# # Setting Win/Loss on last game
# if teamGameStats.iloc[-1 , -2] > teamGameStats.iloc[-2, -2]:
#             teamGameStats.iloc[-1, -1] = 1
#             teamGameStats.iloc[-2, -1] = 0
# if teamGameStats.iloc[-1, -2] < teamGameStats.iloc[-2, -2]:
#             teamGameStats.iloc[-2, -1] = 1
#             teamGameStats.iloc[-1, -1] = 0
# #Dropping all rows with nan values
# teamGameStats = teamGameStats.dropna()
teamGameStats = pd.read_csv('../teamGameStats.csv')
X = teamGameStats.iloc[:, 4 : -2].values
y = teamGameStats.iloc[:, -1].values


# Splitting train and test data

from sklearn.model_selection import train_test_split
import sys
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

np.set_printoptions(threshold=sys.maxsize)
print(type(y_train[0]))
print((X_test))

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# print(X_train)

# print(X_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu' ))

ann.add(tf.keras.layers.Dense(units=6, activation='relu' ))

ann.add(tf.keras.layers.Dense(units=6, activation='relu' ))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid' ))

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

y_train = np.asarray(y_train).astype(np.int64)

ann.fit(X_train, y_train, batch_size = 32, epochs = 33)

def annML(team):
    teamWinPercentage = ann.predict(sc.transform([team])) * 100
    return teamWinPercentage

    # print(ann.predict(sc.transform([['PassingYards': 0, 'RushingYards': 0, 'PassingYardsAllowedDefense': 0, 'RushingYardsAllowedDefense': 0,
    #              'TurnoversCreated': 0, '3rdDownsConverted': 0, 'Sacks': 0, 'Points': 0]])) > 0.5)
# print(ann.predict(sc.transform([[258.0, 118.0, 250, 120, 1.2, 4, 2]])) * 100)
# print(ann.predict(sc.transform([[389.0, 123.0, 300, 100, 0.8, 5, 2]])) * 100)
# print(teamGameStats.iloc[:, 3:6])

# # y_pred = ann.predict(X_test):
# # y_pred = (y_pred > 0.5)
# # y_test = (y_test > 0.5)
# # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# # from sklearn.metrics import confusion_matrix, accuracy_score
# # # print(type(y_pred[0]))
# # # print(type(y_test))
# # # print(type(y_test[0]))
# # # print(type(y_pred))
# # # # y_test = np.asarray(y_test).astype(np.int64)
# # # print((y_test))

# # cm = confusion_matrix(y_test, y_pred)
# # print(cm)
# # accuracy = accuracy_score(y_test, y_pred) * 100
# # print(accuracy)

# # # Visualizing Accuracy

# # y = np.array([accuracy, (100 - accuracy )])
# # pieLabels = ['Predicition Successful' , 'Prediction Unsuccessful']
# # explodeVar = [0.3, 0]

# # plt.pie(y, labels = pieLabels, shadow = True, explode = explodeVar)
# # plt.title('Accucary Score')
# # plt.savefig(r'C:\Users\sox_b\PycharmProjects\NFL_Predictions\myFootballSite\predictor\templates\predictor\predictionAccuracyPiechart.jpg') 

# # # Passing Yards and Rushing Yards side by side

# import seaborn as sns
# # # x = np.array(teamGameStats.iloc[:, 3].values)

# # # print(np.where(x == 639))
# # # print(np.where(x == 3220.0))
# # # print(teamGameStats.iloc[1393,])
# # # print(teamGameStats.iloc[1392,])
# # # y = np.array(teamGameStats.iloc[:, 4].values)
# # # y = np.asarray(y).astype(np.int64)
# # # x = np.asarray(x).astype(np.int64)
# # # random_collection = [x, y]
# # # print(np.max(x))
  
# # # Create a figure instance
# fig = plt.figure()
  
# # Create an axes instance
# ax = fig.gca()
  
# # Create the violinplot
# plt.title('Passing Yards/Game and Rushing Yards/Game by Occurrences')
# sns.violinplot(ax = ax, data = teamGameStats.iloc[:, 1:3])
# # violinplot = ax.violinplot(random_collection)
# plt.savefig(r'C:\Users\sox_b\PycharmProjects\NFL_Predictions\myFootballSite\predictor\templates\predictor\violinplot.jpg') 

# # plt.show()
# # # plt.scatter(x, y)
# # # plt.show()



# # x = teamGameStats['PassingYards'] + teamGameStats['RushingYards']
# # y = np.array(teamGameStats.iloc[:, -2].values)
# # random_collection = [x, y]
# # plt.title('Points Scored x Total Yards')
# # plt.xlabel('Total Yards') #x label
# # plt.ylabel('Points Scored') #y label
# # plt.scatter(x, y)
# # plt.savefig(r'C:\Users\sox_b\PycharmProjects\NFL_Predictions\myFootballSite\predictor\templates\predictor\Points Scored x Total Yards.jpg') 
# # plt.show()
