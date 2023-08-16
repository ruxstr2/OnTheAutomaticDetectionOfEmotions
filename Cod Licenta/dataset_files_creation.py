# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:25:02 2023

@author: ruxan
"""

import pandas as pd
df = pd.read_csv('train_dev_test.csv', header=None)
df.rename(columns={0: 'text', 1: 'classes', 2: 'id'}, inplace=True) #add header
df.to_csv('all_emotions.csv', index=False) #save to new file

#create the Ekman emotions file
0 - admiration 
1 - amusement 
2 - anger 
3 - annoyance 
4 - approval 
5 - caring 
6 - confusion 
7 - curiosity 
8 - desire
9 - disappointment 
10 - disapproval 
11 - disgust 
12 - embarassement 
13 - excitement 
14 - fear 
15 - gratitude 
16 - grief 
17 - joy
18 - love 
19 - nervousness 
20 - optimism 
21 - pride 
22 - realization 
23 - relief 
24 - remorse 
25 - sadness 
26 - surprise 
27 - neutral

#Ekman mapping
0 - anger : 2,3,10
1 - disgust : 11
2 - fear : 14,19
3 -  joy : 0,1,4,5,8,13,15,17,18,20,21,23
4 - sadness : 9,12,16,24,25
5 - surprise : 6,7,22,26
6 - neutral : 27

import csv

header_to_write = ['text','classes','id']
file1 = open("all_emotions.csv", encoding = "utf8")
file2 = open("all_emotions_ekman.csv",'w',encoding = "utf8")

csvreader = csv.reader(file1)
csvwriter = csv.writer(file2)
csvwriter.writerow(header_to_write)

header = []
header = next(csvreader)

anger = [2,3,10]
disgust = [11]
fear = [14,19]
joy = [0,1,4,5,8,13,15,17,18,20,21,23]
sadness = [9,12,16,24,25]
surprise = [6,7,22,26]
neutral = [27]

ambiguous = []

for row in csvreader:
    to_write = [row[0]]
    #classes = row[1]
    #classes.split(",")
    #for i in range(len(classes)):
     #   classes[i] = int(classes[i])
    classes = [int(x) for x in row[1].split(",")]
        
    classes_to_write = set()
    
    for c in classes:
        if c in anger:
            classes_to_write.add(0)
        if c in disgust:
            classes_to_write.add(1)
        if c in fear:
            classes_to_write.add(2)
        if c in joy:
            classes_to_write.add(3)
        if c in sadness:
            classes_to_write.add(4)
        if c in surprise:
            classes_to_write.add(5)
        if c in neutral:
            classes_to_write.add(6)
        
    if len(classes_to_write) > 1:
         ambiguous.append(row[2])
    if len(classes_to_write) == 0:
        raise Exception("No classes found at " + row[2])
        
    classes_to_write = [str(x) for x in classes_to_write]
    classes_to_write = ",".join(classes_to_write)
    to_write.append(classes_to_write)
    to_write.append(row[2])
    csvwriter.writerow(to_write)
    
    
file1.close()
file2.close()
            
#Ekman file for no ambiguities

file1 = open("all_emotions.csv", encoding = "utf8")
file3 = open("all_emotions_ekman_unambiguous.csv",'w',encoding = "utf8")

csvreader = csv.reader(file1)
csvwriter = csv.writer(file3)
csvwriter.writerow(header_to_write)

header = []
header = next(csvreader)


for row in csvreader:
    to_write = [row[0]]
    classes = [int(x) for x in row[1].split(",")]        
    classes_to_write = set()
    
    for c in classes:
        if c in anger:
            classes_to_write.add(0)
        if c in disgust:
            classes_to_write.add(1)
        if c in fear:
            classes_to_write.add(2)
        if c in joy:
            classes_to_write.add(3)
        if c in sadness:
            classes_to_write.add(4)
        if c in surprise:
            classes_to_write.add(5)
        if c in neutral:
            classes_to_write.add(6)
        
    if len(classes_to_write) > 1:
         continue
    if len(classes_to_write) == 0:
        raise Exception("No classes found at " + row[2])
        
    classes_to_write = [str(x) for x in classes_to_write]
    classes_to_write = ",".join(classes_to_write)
    to_write.append(classes_to_write)
    to_write.append(row[2])
    csvwriter.writerow(to_write)
    
    
file1.close()
file3.close() 

  
 
#Sentiment grouping
0 - neutral : 27
1 - positive : 1, 13, 17, 18, 8, 20, 5, 21, 0, 15, 23, 4
2 - negative : 14, 19, 24, 12, 9, 25, 16, 11, 2, 3, 10
3 - ambiguous : 22, 26, 7, 6

header_to_write = ['text','classes','id']
file1 = open("all_emotions.csv", encoding = "utf8")
file4 = open("all_sentiments.csv",'w',encoding = "utf8")

csvreader = csv.reader(file1)
csvwriter = csv.writer(file4)
csvwriter.writerow(header_to_write)

header = []
header = next(csvreader)

neutral = [27]
positive = [1, 13, 17, 18, 8, 20, 5, 21, 0, 15, 23, 4]
negative = [14, 19, 24, 12, 9, 25, 16, 11, 2, 3, 10]
ambiguous_s = [22, 26, 7, 6]

ambiguous_s_count = []

for row in csvreader:
    to_write = [row[0]]
    #classes = row[1]
    #classes.split(",")
    #for i in range(len(classes)):
     #   classes[i] = int(classes[i])
    classes = [int(x) for x in row[1].split(",")]
        
    classes_to_write = set()
    
    for c in classes:
        if c in neutral:
            classes_to_write.add(0)
        if c in positive:
            classes_to_write.add(1)
        if c in negative:
            classes_to_write.add(2)
        if c in ambiguous_s:
            classes_to_write.add(3)
        
    if len(classes_to_write) > 1:
         ambiguous_s_count.append(row[2])
    if len(classes_to_write) == 0:
        raise Exception("No classes found at " + row[2])
        
    classes_to_write = [str(x) for x in classes_to_write]
    classes_to_write = ",".join(classes_to_write)
    to_write.append(classes_to_write)
    to_write.append(row[2])
    csvwriter.writerow(to_write)
    
    
file1.close()
file4.close()

#Sentiment grouping unambiguous
-4200

header_to_write = ['text','classes','id']
file1 = open("all_emotions.csv", encoding = "utf8")
file5 = open("all_sentiments_unambiguous.csv",'w',encoding = "utf8")

csvreader = csv.reader(file1)
csvwriter = csv.writer(file5)
csvwriter.writerow(header_to_write)

header = []
header = next(csvreader)

neutral = [27]
positive = [1, 13, 17, 18, 8, 20, 5, 21, 0, 15, 23, 4]
negative = [14, 19, 24, 12, 9, 25, 16, 11, 2, 3, 10]
ambiguous_s = [22, 26, 7, 6]

ambiguous_s_count = []

for row in csvreader:
    to_write = [row[0]]
    #classes = row[1]
    #classes.split(",")
    #for i in range(len(classes)):
     #   classes[i] = int(classes[i])
    classes = [int(x) for x in row[1].split(",")]
        
    classes_to_write = set()
    
    for c in classes:
        if c in neutral:
            classes_to_write.add(0)
        if c in positive:
            classes_to_write.add(1)
        if c in negative:
            classes_to_write.add(2)
        if c in ambiguous_s:
            classes_to_write.add(3)
        
    if len(classes_to_write) > 1:
         continue
    if len(classes_to_write) == 0:
        raise Exception("No classes found at " + row[2])
        
    classes_to_write = [str(x) for x in classes_to_write]
    classes_to_write = ",".join(classes_to_write)
    to_write.append(classes_to_write)
    to_write.append(row[2])
    csvwriter.writerow(to_write)
    
    
file1.close()
file5.close()

----
header_to_write = ['text','classes','id']
file1 = open("all_emotions.csv", encoding = "utf8")
file6 = open("all_emotions_unambiguous.csv",'w',encoding = "utf8")

csvreader = csv.reader(file1)
csvwriter = csv.writer(file6)
csvwriter.writerow(header_to_write)

header = []
header = next(csvreader)


ambiguous = []
for row in csvreader:
    to_write = [row[0]]
    #classes = row[1]
    #classes.split(",")
    #for i in range(len(classes)):
     #   classes[i] = int(classes[i])
    classes = [int(x) for x in row[1].split(",")]
        
    if len(classes) > 1:
         ambiguous.append(row[2])
         continue
        
    to_write.append(row[1])
    to_write.append(row[2])
    csvwriter.writerow(to_write)
    
file1.close()
file6.close()

