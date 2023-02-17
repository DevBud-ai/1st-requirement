# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:53:01 2023

@author: ASUS
"""
import json
f = open("C:\\Users\\ASUS\\Downloads\\UIED-master\\data\\output\\merge\\111.json")

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list\
#FIRST DATA
ids=[]
bbox=[]
hw=[]
text=[]
textid=[]
parent_child={}
for i in data["compos"]:
    ids.append(i['id'])
    pos=i['position']
    y1=pos['column_max']
    x1=pos['row_max']
    x=pos['row_min']
    y=pos['column_min']

    bbox.append([x,y,x1,y1])
    hw.append([i['height'],i['width']])
    if 'text_content' in i.keys() :
        text.append(i['text_content'])
        textid.append(i['id'])
    if 'children' in i.keys():
        parent_child[i['id']]=i['children']

F1 = open("C:\\Users\\ASUS\\Downloads\\UIED-master\\data\\output\\merge\\Home(1).json")

# returns JSON object as
# a dictionary
Data = json.load(F1)
Ids=[]
Bbox=[]
Hw=[]
Text=[]
Textid=[]
Parent_child={}
for i in Data["compos"]:
    Ids.append(i['id'])
    pos=i['position']
    y1=pos['column_max']
    x1=pos['row_max']
    x=pos['row_min']
    y=pos['column_min']
    Bbox.append([x,y,x1,y1])
    Hw.append([i['height'],i['width']])
    if 'text_content' in i.keys() :
        Text.append(i['text_content'])
        Textid.append(i['id'])
    if 'children' in i.keys():
        Parent_child[i['id']]=i['children']
"""first=[]
for r in range(0,len(bbox)):
    i=bbox[r]
    j=hw[r]
    initial=[i[0],i[1]+j[0],i[0]+j[1],i[1]]
    first.append(initial)
    
Second=[]
for r in range(0,len(Bbox)):
    i=bbox[r]
    #j=hw[r]
    intial=
    """
def Similarity_Check(word,idx):
    l=Similarity([word,text1[idx if idx <= len(ids1)-1 else len(ids1)-1]])
    Klist=[0.0] * len(ids1)

    if l >= 0.99:
        match_idx = idx if idx <= len(ids1)-1 else len(ids1)-1
        b_check=Bbox_Check(idx,match_idx)
        probability=1.0
        score=l
        inter_analyse=[match_idx,b_check,probability,score]
    else:
        Klist[idx if idx <= len(ids1)-1 else len(ids1)-1]
        flag=False
        if idx == 0:
            Next=idx+1
            while(Next != len(ids1)):
                L=Similarity([word,text1[Next]])
                Klist[Next]=L
                if L >= 0.99  and (abs(idx - Next)<=3):
                    flag=True
                    break
                Next=Next+1
            if flag==True:
                    match_idx=Next
                    b_check=Bbox_Check(idx,match_idx)
                    probability=1.0
                    score=1.0
                    inter_analyse=[match_idx,b_check,probability,score]
            else:
                inter_analyse=list(highest_prob(Klist,idx))
        elif idx == len(ids)-1:
            previous=idx-1 if idx<=len(ids1) else len(ids1)-1
            while(previous != -1):
                L=Similarity([word,text1[previous]])
                Klist[previous]=L
                if L >= 0.99  and (abs(len(ids)-1 - previous)<=3):
                    flag=True
                    break
                previous=previous-1

            if flag==True:
                    match_idx=previous
                    b_check=Bbox_Check(idx,match_idx)
                    probability=1.0
                    score=1.0
                    inter_analyse=[match_idx,b_check,probability,score]
            else:
                inter_analyse=list(highest_prob(Klist,idx))
        else:
            previous=(idx if idx<=len(ids1)-1 else len(ids1)-1)-1
            Next=(idx if idx<len(ids1)-1 else len(ids1)-2)+1
            while(previous!=-1 or Next !=len(ids1)):#-1,len(ids1)

                if (previous != -1):
                    l1=Similarity([word,text1[previous]])
                    Klist[previous]=l1
                if (Next != len(ids1)):
                    l2=Similarity([word,text1[Next]])
                    Klist[Next]=l2
                if (l1 >=0.99 and abs((idx if idx<=len(ids1)-1 else len(ids1)-1) - previous)<=3) or (abs((idx if idx<=len(ids1)-1 else len(ids1)-1) - Next)<=3 and l2>= 0.99):
                    flag=True
                    break
                if (previous != -1):
                    previous=previous-1
                if (Next != len(ids1)):
                    Next=Next+1
            if flag==True:
                if (l1>l2):
                    match_idx=previous
                    b_check=Bbox_Check(idx,match_idx)
                    probability=1.0
                    score=1.0
                    inter_analyse=[match_idx,b_check,probability,score]
                else:
                    match_idx=Next
                    b_check=Bbox_Check(idx,match_idx)
                    probability=1.0
                    score=1.0
                    inter_analyse=[match_idx,b_check,probability,score]
            else:
                inter_analyse=list(highest_prob(Klist,idx))
    return(inter_analyse)

Combined_Analyse=[]
for i in range(0,len(text)):
    word=text[i]
    idx=i
    Combined_Analyse.append(Similarity_Check(word,idx))

# visualization 
"""
from PIL import Image
from skimage.metrics import structural_similarity
import cv2
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
image_ao = Image.open("Home.png")
image_a = image_ao.resize((1200, 1200))
arrA = np.array(image_a)

for i in range(0,len(row)): 
    x=column[i]
    y=row[i]
    w=width[i]
    h=height[i]
    if i==3:
        cv2.rectangle(arrA,(x,y), (x+w,y+h), (0,255,0), 5)
        
    cv2.rectangle(arrA,(x,y), (x+w,y+h), (255,0,0), 5)
    cv2.imwrite('differ.png',arrA)
    """
    
#GUI test

import json
f = open("C:\\Users\\ASUS\\Downloads\\UIED-master\\data\\output\\merge\\Home.json")

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list\
#FIRST DATA
Ids=[]
Bbox=[]
Hw=[]
Text=[]
Textid=[]
Parent_child={}
for i in data["compos"]:
    if i['class'] == "Compo":
        Ids.append(i['id'])
        pos=i['position']
        y1=pos['column_max']
        x1=pos['row_max']
        x=pos['row_min']
        y=pos['column_min']
    
        Bbox.append([x,y,x1,y1])
        hw.append([i['height'],i['width']])
        if 'text_content' in i.keys() :
            Text.append(i['text_content'])
            Textid.append(i['id'])
        if 'children' in i.keys():
            Parent_child[i['id']]=i['children']

F1 = open("C:\\Users\\ASUS\\Downloads\\UIED-master\\data\\output\\merge\\Home(1).json")

# returns JSON object as
# a dictionary
Data = json.load(F1)
Ids1=[]
Bbox1=[]
Hw1=[]
Text1=[]
Textid1=[]
Parent_child1={}
for i in Data["compos"]:
    if i['class'] == 'Compo':
        Ids1.append(i['id'])
        pos=i['position']
        y1=pos['column_max']
        x1=pos['row_max']
        x=pos['row_min']
        y=pos['column_min']
        Bbox1.append([x,y,x1,y1])
        Hw1.append([i['height'],i['width']])
        if 'text_content' in i.keys() :
            Text1.append(i['text_content'])
            Textid1.append(i['id'])
        if 'children' in i.keys():
            Parent_child1[i['id']]=i['children']

# text word present check 

gui_box=[]
for r in range(0,len(Ids)):
    i=Bbox[r]
    j=hw[r]
    initial=[i[0],i[1]+j[0],i[0]+j[1],i[1]]
    gui_box.append(initial) 
Text_box=[]
for r in range (0,len(text)):
    x=row[r]
    y=column[r]
    w=width[r]
    h=height[r]
    Text_box.append([x,y+h,x+w,y])
gui_box2=[]
for r in range(0,len(Ids1)):
    i=Bbox1[r]
    j=Hw1[r]
    initial=[i[0],i[1]+j[0],i[0]+j[1],i[1]]
    gui_box2.append(initial) 

Text_box2=[]
for r in range (0,len(text1)):
    x=row1[r]
    y=column1[r]
    w=width1[r]

class Solution:
   def isRectangleOverlap(self, R1, R2):
      if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
         return (False,0)
      else:
          left = max(R1[0], R2[0])
          right = min(R1[2], R2[2])
          bottom = min(R1[1], R2[1])
          top = max(R1[3], R2[3])
          In_width=right-left
          In_height=bottom-top
          In_Area=In_width*In_height
          width=R2[2]-R2[0]
          height=R2[1]-R2[3]
          Area=width*height
          In_cent=In_Area/Area * 100
          return (True,In_cent)    


overlap=[]
ob = Solution()
for R1 in range(0,len(gui_box)):
   inter_list=[]
   for R2 in range(0,len(Text_box)):
      over_lap,In_cent=ob.isRectangleOverlap(gui_box[R1],Text_box[R2])
      if over_lap=='True' and In_cent>=80:
         inter_list.append([R2,1])
      elif(over_lap=='True' and (In_cent >=50 and In_cent < 80)):
         inter_list.append([R2,2])
   overlap.append(inter_list)


overlap_2=[]
ob = Solution()
for R1 in range(0,len(gui_box2)):
   inter_list=[]
   for R2 in range(0,len(Text_box2)):
      over_lap,In_cent=ob.isRectangleOverlap(gui_box2[R1],Text_box2[R2])
      if over_lap=='True' and In_cent>=80:
         inter_list.append([R2,1])
      elif(over_lap=='True' and (In_cent >=50 and In_cent < 80)):
         inter_list.append([R2,2])
   overlap_2.append(inter_list)
   
#gui boxes are present equ;

if len(gui_box)== len(gui_box2):
    print("Equal Implementation of gui boxes")
else:
    print("gui boxes are not present equally")

gui_cmpar=[]
for i in range(0,len(gui_box)):
    inter_list=[]
    for j in range(0,len(gui_box2)):
         over_lap,In_cent=ob.isRectangleOverlap(gui_box[i],gui_box2[j])
         if over_lap==True and In_cent>=80:
             w1=Hw[i][1]
             h1=Hw[i][0]
             w2=Hw[j][1]
             h2=Hw[j][0]
             w_diff = abs(w1 - w2)
             h_diff= abs(h1 - h2)
             if w_diff <= 3:
                 check_1 = 1
             if w_diff > 3:
                 check_1 = 2
             if h_diff <= 3:
                 check_2 = 1
             if h_diff > 3:
                 check_2 = 2
             total_diff= check_1 * check_2
             total_diff=total_diff if total_diff !=4 else 3
             inter_list.append([j,In_cent,total_diff])
    gui_cmpar.append(inter_list)

for i in range(0,len(gui_cmpar)):
    if len(gui_cmpar[i])>1:
        min_value=0
        index=0
        for j in range(0,len(gui_cmpar[i])):
            J=gui_cmpar[i][j]
            value=J[1]/J[2]
            if value>min_value:
                min_value=value
                index=j
        gui_cmpar[i]=gui_cmpar[i][j]
    if len(gui_cmpar[i])==0:
        gui_cmpar[i]='xx'

for i in range(0,len(overlap)):
    if len(overlap)==1:
        ref=gui_cmpar[i][0]
        cmp2=overlap_2[ref]
        cmp1=overlap[i]
        for i in cmp1:
            ck=[]
            if i in cmp2 and len(cmp1)==len(cmp2):
               ck.append(1)
            else:
                
        
        
        
        
        
        