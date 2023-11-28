import numpy as np
import math
import copy
import statistics as stat
from collections import defaultdict

class info:
    name = ""
    avg = 0
    gain = 0
    labels = []

    def __init__(self, name, avg):
        self.name = name
        self.avg = avg

    def print(self):
        print("Name: ", self.name)
        print("Average: ", self.avg)
        print("Gain: ", self.gain)

class label:
    name = ""
    g = 0
    h = 0
    total = 0
    def __init__(self, name, g, h):
        self.name = name
        self.g = g
        self.h = h
        self.total = g + h

    def print(self):
        print("Label: ", self.name)
        print("Gammas: ", self.g, "\nHadrons: ", self.h)

class Node:
    name = ""
    labels = []
    left = None
    right = None
    entropyL = 0
    entropyR = 0
    index = 0
    avg = 0
    theclass = ""

    def __init__(self, name, labels, avg, index, theclass = ""):
        self.name = name
        self.labels = copy.deepcopy(labels)
        self.entropyL = entropy(labels[0].g, labels[0].h)
        self.entropyR = entropy(labels[1].g, labels[1].h)  
        self.avg = avg
        self.purityL = math.sqrt(labels[0].g ** 2 + labels[0].h ** 2)
        self.purityR = math.sqrt(labels[1].g ** 2 + labels[1].h ** 2)
        self.index = index
        self.theclass = theclass


    def print(self):
        print("Attribute: ", self.name)
        print("Labels: ", self.labels[0].print(), "Entropy: ", self.entropyL, "\n", self.labels[1].print(), "Entropy: ", self.entropyR, "\n")
        print("Average: ", self.avg)
        print("Purities: ", self.purityL, self.purityR, "\n")
        print("Class: ", self.theclass, "\n\n")

    #Prints the tree in preorder fashion
    def printtree(self):
        if self:
            self.print()
            if self.left:
                self.left.printtree()
            if self.right:
                self.right.printtree()

    #This function takes the leaf nodes after the tree is finished and gives them leaves that are defined by their classes instead of their attributes. This is to evaluate test data into either types.
    def last(self):
        if self:
            if self.left:
                self.left.last()
            if self.right:
                self.right.last()
            if self.left is None and self.right is None:
                if self.labels[0].g > self.labels[0].h:
                    theclass = "Gamma"
                else:
                    theclass = "Hadron"
                self.left = Node(self.labels[0].name, self.labels, self.avg, self.index, theclass)
                if self.labels[1].g > self.labels[1].h:
                    theclass = "Gamma"
                else:
                    theclass = "Hadron"
                self.right = Node(self.labels[1].name, self.labels, self.avg, self.index, theclass)
        

cases = []
junior = []
attrnames = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
attrstats = []
gslist = []
hslist = []
listofattr = []

for i in range(len(attrnames)):
    listofattr.append(info(attrnames[i], 0))
    listofattr[i].print()

attrnames.pop(len(attrnames) - 1)
count = 0
gamma = 0
hadron = 0

infile = open('magic04sample.data', 'r')
gorh = []
for line in infile:
    junior = []
    
    for i in line.split(','):
        junior.append(i)
        if i.lstrip('-').replace('.','',1).isdigit() and not i.startswith("00"):
            junior[count] = float(junior[count])
        if i == 'g\n':
            gamma += 1
            gslist.append(junior)
            gorh.append("g")
        elif i == 'h\n':
            hadron += 1
            hslist.append(junior)
            gorh.append("h")
        count += 1

    cases.append(junior)
    count = 0

infile.close()

numAttr = len(cases[0]) - 1
numCases = len(cases)

#finding the averages
for i in range(len(attrnames)):
    newlist = np.array(cases)
    col = newlist[:,i]
    col = list(col)
    col = [float(i) for i in col]
    listofattr[i].avg = sum(col)/len(col)

for i in range(len(listofattr)):
    listofattr[i].print()

i = len(cases) - 1
[j.pop(10) for j in cases]
listofattr.pop(10)

copylistofattr = copy.deepcopy(listofattr)

#splits the numeric values into whether they are above or below average
for i in range(len(cases)):
    for j in range(len(cases[i])):
        if cases[i][j] >= listofattr[j].avg:
            cases[i][j] = ">=avg"
        else:
            cases[i][j] = "<avg"


def findParent(numAttr, numCases, cases, listattr, gorh):
    h = 0
    attr = []
    while h < numAttr:
        i = 1
        here = -1
        labels = []
        names = []

        labelclass = label("", 0, 0)

        while i < numCases:
            if (names.count(cases[i][h]) == 0):
                labelclass = label(cases[i][h], 0, 0)
                names.append(cases[i][h])
                labels.append(labelclass)
                here += 1
            else:
                for k in range(len(labels)):
                    if labels[k].name == cases[i][h]:
                        here = k
            if gorh[i-1] == "g":
                labels[here].g += 1
            elif gorh[i-1] == "h":
                labels[here].h += 1
            i += 1
        listattr[h].labels = copy.deepcopy(labels)
        
        listattr[h].labels[0].print()
        listattr[h].labels[1].print()
        
        h += 1
    return listattr
    
def entropy(yes, no):
    if yes == 0 and no == 0:
        return 0
    if yes == 0:
        e = -(no/(yes + no)) * math.log(no/(yes + no), 2)
    if no == 0:
        e = -(yes/(yes + no)) * math.log(yes/(yes + no), 2)
    if yes != 0 and no != 0:
        e = -(yes/(yes + no)) * math.log(yes/(yes + no), 2) - (no/(yes + no)) * math.log(no/(yes + no), 2)
    return e

def findGains(listofattr, numCases, eParent):
    numCases -= 1
    sum = 0
    maxGain = 0
    maxGainIndex = 0
    for i in range(len(listofattr)-1):
        gain = eParent
        for j in range(len(listofattr[i].labels)):
            gain -= ((listofattr[i].labels[j].g  + listofattr[i].labels[j].h)/numCases) * entropy(listofattr[i].labels[j].g, listofattr[i].labels[j].h)
        listofattr[i].gain = gain
        sum += listofattr[i].gain
        listofattr[i].print()
        if listofattr[i].gain > maxGain:
            maxGain = listofattr[i].gain
            maxGainIndex = i
        print("\n")
    return maxGainIndex

listofattr = copy.deepcopy(findParent(numAttr, numCases, cases, listofattr, gorh))

for i in range(len(listofattr)):
    listofattr[i].print()
    for j in range(len(listofattr[i].labels)):
        listofattr[i].labels[j].print()
    print("\n")

eParent = entropy(gamma, hadron)
print(eParent)
gains = []
newParentIndex = findGains(listofattr, numCases, eParent)
eParent = max(entropy(listofattr[newParentIndex].labels[0].g, listofattr[newParentIndex].labels[0].h), entropy(listofattr[newParentIndex].labels[1].g, listofattr[newParentIndex].labels[1].h))
print(listofattr[newParentIndex].name)

root = Node(listofattr[newParentIndex].name, listofattr[newParentIndex].labels, listofattr[newParentIndex].avg, newParentIndex)

#Turns the lists taken from data from the file into Nodes and returns the newly made Node along with the parent Node's index in listofattr
def fill(cases, listofattr, numAttr, newParentIndex, root, eParent, label):
    if len(listofattr) > 1:
        i = numAttr

        newlabelgorh = []

        [newlabelgorh.append(j[newParentIndex]) for j in cases]

        [j.pop(newParentIndex) for j in cases]
        listofattr.pop(newParentIndex)
        
        listofA = []
        listofB = []
        gorhA = []
        gorhB = []
        for j in range(len(newlabelgorh)):

            if newlabelgorh[j] == label:
                listofA.append(cases[j])
                gorhA.append(gorh[j])
        listofattr = copy.deepcopy(findParent(len(listofA[0]), len(listofA), listofA, listofattr, gorhA))
        gains = []
        newParentIndex = findGains(listofattr, len(listofA), eParent)
        eParent = max(entropy(listofattr[newParentIndex].labels[0].g, listofattr[newParentIndex].labels[0].h), entropy(listofattr[newParentIndex].labels[1].g, listofattr[newParentIndex].labels[1].h))
        print("fill--------------")
        newnode = Node(listofattr[newParentIndex].name, listofattr[newParentIndex].labels, listofattr[newParentIndex].avg, newParentIndex)
        print(eParent)
        print(listofattr[newParentIndex].name)
        listofattr[newParentIndex].labels[0].print()
        
        listofattr[newParentIndex].labels[1].print()

        print("end of fill-----------------------")
    return Node(listofattr[newParentIndex].name, listofattr[newParentIndex].labels, listofattr[newParentIndex].avg, newParentIndex), newParentIndex
        
#This function is used to form the two child nodes of a parent and returns the parent index so we can refer to it from listofattr.
def pairup(root, newParentIndex):
    if len(listofattr) > 1: #Making sure that we do not extract from an empty list
        
        i = 0
        node1, newParentIndex = fill(cases, listofattr, len(listofattr), newParentIndex, root, eParent, ">=avg")
        node2, newParentIndex = fill(cases, listofattr, len(listofattr), newParentIndex, root, eParent, "<avg")
        root.left = copy.deepcopy(node1)
        root.right = copy.deepcopy(node2)
        i += 1
        return newParentIndex
    return

#this function is used to construct the whole tree. NewParentIndex is updated each time so that we know which attribute will be implemented into Node form.
def construct(root, newParentIndex):
    newParentIndex = pairup(root, newParentIndex)
    newParentIndex = pairup(root.left, newParentIndex)
    newParentIndex = pairup(root.right, newParentIndex)
    newParentIndex = pairup(root.left.left, newParentIndex)
    newParentIndex = pairup(root.left.right, newParentIndex)

construct(root, newParentIndex)

root.last()

root.printtree()


for i in range(len(listofattr)):
    listofattr[i].print()
    for j in range(len(listofattr[i].labels)):
        listofattr[i].labels[j].print()
    print("\n")


#we have set up the sample tree, and now we test it with a test set
test = []
infile = open('magic04test.data', 'r')
gorh = []
loops = 0
for line in infile:
    junior = []
    
    for i in line.split(','):
        junior.append(i)
        if i.lstrip('-').replace('.','',1).isdigit() and not i.startswith("00"):
            junior[count] = float(junior[count])
        if i == 'g\n':
            gamma += 1
            gslist.append(junior)
            gorh.append("Gamma")
        elif i == 'h\n':
            hadron += 1
            hslist.append(junior)
            gorh.append("Hadron")
        count += 1

    test.append(junior)
    count = 0
    loops += 1

infile.close()

treecopy = copy.deepcopy(root)

#this prediction function goes down the tree and chooses which leaf node to go to according to the values of the test set
def predict(root, test, attrnames):
    if (root.left is None and root.right is None) or (root.theclass == "Gamma" or root.theclass == "Hadron"):
        return root.theclass
    else:
        if test[attrnames.index(root.name)] >= root.avg:
            return predict(root.left, test, attrnames)
        else:
            return predict(root.right, test, attrnames)

#now we calculate the accuracy of the algorithm on the test set
results = []
correct = 0
wrong = 0
for i in range(len(test)):
    results.append(predict(root, test[i], attrnames))
    if results[i] == gorh[i]:
        correct += 1
    else:
        wrong += 1
accuracy = correct/(correct + wrong)
print("Accuracy: ", accuracy)



#we will be redoing the prior block for the case of the sample data set
cases = []
infile = open('magic04sample.data', 'r')
gorh = []
loops = 0
for line in infile:
    junior = []
    
    for i in line.split(','):
        junior.append(i)
        if i.lstrip('-').replace('.','',1).isdigit() and not i.startswith("00"):
            junior[count] = float(junior[count])
        if i == 'g\n':
            gamma += 1
            gslist.append(junior)
            gorh.append("Gamma")
        elif i == 'h\n':
            hadron += 1
            hslist.append(junior)
            gorh.append("Hadron")
        count += 1

    cases.append(junior)
    count = 0
    loops += 1

infile.close()

results = []
correct = 0
wrong = 0
for i in range(len(cases)):
    results.append(predict(root, cases[i], attrnames))
    if results[i] == gorh[i]:
        correct += 1
    else:
        wrong += 1
accuracy = correct/(correct + wrong)
print("Accuracy: ", accuracy)
