# Coursera Data Science Specialization
# Practical Machine Learning Course
# Week 4 Project
# John James
# April 22, 2016

##############################################################################
#                           SPLIT DATA                                       #
##############################################################################
## ---- splitData
library(caret)
set.seed(1030)
inTrain       <- createDataPartition(y=trainFile$classe, p=0.70, list = FALSE)
trainData     <- trainFile[inTrain,]      #Training Data
testData      <- trainFile[-inTrain,]     #Testing Data
valData       <- testFile                 #Validation Data
## ---- end