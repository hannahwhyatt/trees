"""
SCC.461 (Lancaster University): Decision Tree Classifier
21/12/2021

"""
import random
import copy

class _TreeNode:
    
    """ Superclass to DecisionTree subclass """

    def __init__(
        self, 
        midpoint = None,            # midpoint of the previous split (or self split, if nodeType ="root")
        nodeDirection = None,       # node originating from left or right side of a split
        nodeType = "root",          # root, branch or leaf node
        data = None,                # data to be split (if root or branch) or classed (if leaf)
        decisionFeature = None,     # feature used for the previous split (or self split if nodeType == "root"
        currentDepth = 0,           # the depth of the node within the tree
        maxDepth = None,            # maximum depth of the tree
        minLeafSize = 1             # minimum number of items within a leaf
        ):

        self.midpoint = midpoint
        self.nodeDirection = nodeDirection
        self.nodeType = nodeType 
        self.data = data
        self.decisionFeature = decisionFeature
        self.currentDepth = currentDepth
        self.maxDepth = maxDepth
        self.minLeafSize = minLeafSize

        self.left = None            # for branch or root nodes, left and right are pointers for either side of the split
        self.right = None

    def _GiniOfData(self, data):

        """ Finds the gini impurity of a data set (one side of a split)"""

        # save the size of the data set
        sizeData = len(data)

        # make a list containing only the labels
        labelcol = []
        for i in data:
            labelcol.append(i[-1])      

        # save a set of the labels names that are present in the dataset
        categories = set(labelcol)

        # iterate through the set of label names to find the corresponding term within the gini impurity forumla:

        terms = []                              # initialises a list for terms
        for category in categories:             # for each label:
            count = labelcol.count(category)        # count how many are present
            term = (count/sizeData)**2              # calculate the gini impurity term
            terms.append(term)                      # append to the list of terms

        # return the gini impurity of the data set
        return 1 - sum(terms)

    def _GiniOfSplit(self, data, dataLeft, dataRight):

        """ Find the gini impurity of a given split in a data set """

        # save the size of the full dataset 
        lenData = len(data)

        # save the sizes of each side of the split
        sizeLeft = len(dataLeft)
        sizeRight = len(dataRight)

        # find the gini impurity of each side
        giniLeft = self._GiniOfData(dataLeft)
        giniRight = self._GiniOfData(dataRight) 

        # return the total gini impurity of the split    
        return (sizeLeft/lenData)*giniLeft + (sizeRight/lenData)*giniRight
  
    def _findBestSplit(self):

        """ Finds the best split of a given dataset using Gini impurity. 
        Possible splits tested are constrained by minimum size of a leaf node when provided"""

        # get the size of the data set
        lenData = len(self.data)

        # save a list of labels
        labelcol = []
        for item in self.data:
            labelcol.append(item[-1])

        # If there is only one item in the data, it cannot be split
        # therefore, function returns "None" for all output 
        if lenData == 1:
            return [None]*5        
        
        # if the data are labelled the same, data cannot be split
        # therefore, function returns "None" for all output
        if len(set(labelcol)) == 1:
            return [None]*5



        # g is the lowest gini impurity found. Initialise g as 1    
        g = 1
        
        # For all features:
        for feature in range(len(self.data[0]) - 1):

            # if all values of the given feature are the same, the data cannot be split;
                # continue to the next feature
            values = [item[feature] for item in self.data]
            if len(set(values)) == 1:
                continue

            # order the data by the given feature  
            ordData = sorted(self.data, key = lambda col: (col[feature]))

            # loop through all possible splits of the feature             
            for div in range(self.minLeafSize, (lenData - self.minLeafSize + 1)):
                
                dataLeft = ordData[:div]    # Data to the left of the divide
                dataRight = ordData[div:]   # Data to the right of the divide
                
                # ignore splits that fall between items with the same value for the given feature
                feature = int(feature)
                if dataLeft[-1][feature] == dataRight[0][feature]:
                    continue

                # find the gini index of the split
                gIndex = self._GiniOfSplit(ordData, dataLeft, dataRight)

                # find the midpoint of the split 
                midpoint = (float(dataLeft[-1][feature]) + float(dataRight[0][feature]))*0.5
                
                # if the split returns a gini impurity less than previously found, save 
                # the relevant information about the split. If the gini impurity is equal 
                # to the least found so far, randomly decide whether or not to update 
                # the best split.
                if gIndex == g:
                    r = random.random()
                    if r < 0.5:
                        g = copy.copy(gIndex)
                        split = midpoint
                        left = dataLeft             # data to the left of the split
                        right = dataRight           # data to the right of the split
                        decisionFeature = feature   # feature (column index) by which the split is made    
                    else: 
                        continue
                elif gIndex < g:
                    g = gIndex
                    split = midpoint
                    left = dataLeft             
                    right = dataRight          
                    decisionFeature = feature   
        
        # if no possible splits are found following iteration through all features,
        # function returns "None" for all output
        if g == 1:
            return [None]*5

        # return relevant information about the best split
        return (round(g, 5), split, left, right, decisionFeature)
        
    def fit(self):

        """ 
        Fits a decision tree using class variable 'data' as training data.
        - Recursively fits a decision tree for each left and right branch node.
        - Depth of the decision tree is constrained by class variable 'maxDepth' when provided.
        
        """

        # if the maximum depth has been reached, create a leaf:
        if self.currentDepth == self.maxDepth:

            # determine the class of he leaf from the contents:

            # create a list containing the only labels in the leaf
            classLabels = []
            for item in self.data:
                classLabels.append(item[-1])
            # find the most commonly occuring label in the list and assign to the leaf 'content' variable
            self.classLabel = max(set(classLabels), key = classLabels.count) 
            self.nodeType = "leaf"
        
        # if the maximum depth has not been reached, attempt to split the data:
        else:
            
            # find the best split
            gini, split, newLeft, newRight, decisionFeature = self._findBestSplit()

            # if the data cannot be split, create a leaf:
            if split is None:

                # create a list containing the only labels in the leaf
                classLabels = []
                for item in self.data:
                    classLabels.append(item[-1])
                # find the most commonly occuring label in the list and assign to the leaf 
                self.classLabel = max(set(classLabels), key = classLabels.count) 
                
                self.nodeType = "leaf"

            # if the data were sucessfully split:
            else:

                # save the information about the split to the node
                if self.nodeType == "root":
                    self.midpoint = split
                    self.decisionFeature = decisionFeature

                L = _TreeNode(nodeDirection = "left", nodeType = "branch", data = newLeft, 
                                currentDepth = self.currentDepth + 1,
                                maxDepth = self.maxDepth, 
                                minLeafSize = self.minLeafSize)
                R = _TreeNode(nodeDirection = "right", nodeType = "branch", data = newRight, 
                                currentDepth = self.currentDepth + 1,
                                maxDepth = self.maxDepth, 
                                minLeafSize = self.minLeafSize)


                self.midpoint = split
                self.decisionFeature = decisionFeature

                self.left = L
                self.right = R

                # recursively fit trees the left and right sides of the split
                self.left.fit()
                self.right.fit()

    def pred(self, newData):

        """ Predict the classes of unlabelled data """

        # initialise and empty list to conatin the predicted classes
        predictions = []

        # ieratively run each item in the new dataset through the decision tree:
        for item in newData:
            
            # start at the root node
            node = copy.deepcopy(self)
            
            # while the node we are considering is not a leaf node:
            while node.nodeType != "leaf":

                # find the split point and deciding feature
                split, feature = node.midpoint, node.decisionFeature

                # find the next node in the tree, depending on which side of the split the item belongs to
                if item[feature] <= split:
                    node = node.left
                else:
                    node = node.right

            # once a leaf node is reached, save the class to the list of predictions 
            # before moving on to the next item in the data set
            predictions.append(node.classLabel)

        # once all items have been classified, return the list of predictions
        return predictions

    def printTree(self):

        """ Print the decision tree in text format """

        # identify the node type and print revelenat information:
        # if the node type is a root or a branch, recursively print the left and right nodes
        if self.nodeType != "leaf":
            
            if self.nodeType == "root":
                print("[root] depth = " + str(self.currentDepth))

            print("|   "*self.currentDepth + "|--- feature "+ str(self.decisionFeature) +
                    " <= " + str(round(self.midpoint, 2)))
            self.left.printTree()
            print("|   "*self.currentDepth + "|--- feature "+ str(self.decisionFeature) + 
                    " >  " + str(round(self.midpoint, 2)))
            self.right.printTree()


        # if the node type is a leaf, print the relevant information    
        elif self.nodeType == "leaf":
            print("|   "*self.currentDepth + "|--- class: " + str(self.classLabel))

    def getDepth(self):

        """ returns the maximimum depth of the tree as an integer """

        ## if the node is a leaf or an empty root, return the current depth
        ## otherwise recursively find the depths of left and right branchs
        ## and return the maximum depth found.
        if self.left == None and self.right == None:    
            return self.currentDepth                       
        else:                                          
            leftDepth = self.left.getDepth()            
            rightDepth = self.right.getDepth()
            maxDepth = max(leftDepth, rightDepth)       
        return maxDepth                                 


class DecisionTree(_TreeNode):

    """ Decision tree classifier

    Subclass of 'TreeNode' where:
    - nodeType = "root"
    - separate arguments for data and labels

    """
    def __init__(self, data, labels, maxDepth = None, minLeafSize = 1):

        # merge training data and training data labels
        labelledData = []
        for item, label in zip(data, labels):
            i = item[:]
            i.append(label)
            labelledData.append(i)

        # initialises super class        
        _TreeNode.__init__(self, 
        midpoint = None,            
        nodeDirection = None,       
        nodeType = "root",         
        data = labelledData,                
        decisionFeature = None,    
        currentDepth = 0,
        maxDepth = maxDepth,
        minLeafSize = minLeafSize
        )

