# Coursera Data Science Specialization
# Practical Machine Learning Course
# Week 4 Project
# John James
# April 22, 2016

##############################################################################
#                             SCALE AND CENTER                               #
##############################################################################
## ---- centerScale
library(funModeling)
library(caret)
#Select non-predictor (NP) variables that should be excluded from centering and scaling
nonPredictors <- c("X", "user_name", "raw_timestamp_part_1", 
                   "raw_timestamp_part_2", "cvtd_timestamp",
                   "new_window", "num_window")

#Remove non-predictors from data sets
trainDataCS   <- trainData[, -which(names(trainData) %in% nonPredictors)]
testDataCS    <- testData[, -which(names(testData) %in% nonPredictors)]
valDataCS     <- valData[, -which(names(valData) %in% nonPredictors)]

#Center and scale independent variables based upon means from training set.
centerScale   <- preProcess(trainDataCS, method=c("center", "scale"))
trainDataCS   <- predict(centerScale, trainDataCS[-153])
testDataCS    <- predict(centerScale, testDataCS[-153])
valDataCS     <- predict(centerScale, valDataCS[-153])

#Add outcome variable back to data sets
trainDataCS   <- data.frame(trainDataCS,classe=trainData$classe)
testDataCS    <- data.frame(testDataCS,classe=testData$classe)
valDataCS     <- data.frame(valDataCS,problem_id=valData$problem_id)
## ---- end

##############################################################################
#                         CLEAN TRAINING DATA                                #
##############################################################################
## ---- cleanData
#Identify columns in the training set with over 80% NA values
trainDataCSStatus <- df_status(trainDataCS)
naVars      <- subset(trainDataCSStatus, trainDataCSStatus$p_na > 80)

#Remove them from training, test and validation sets
trainDataCS <- trainDataCS[, !(names(trainDataCS) %in% naVars[,"variable"])]
testDataCS  <- testDataCS[, !(names(testDataCS) %in% naVars[,"variable"])]
valDataCS   <- valDataCS[, !(names(valDataCS) %in% naVars[,"variable"])]

#Summarize dimensions of training set
trainObservations  <- nrow(trainDataCS)
trainVariables     <- ncol(trainDataCS)
## ---- end

##############################################################################
#                                 PLOT                                       #
##############################################################################
## ---- corrPlot
#Render correlation plot
library(corrplot)
corMatrix <- cor(trainDataCS[-53])
corrplot::corrplot(corMatrix, order = "alphabet", method = "color", type = "lower", 
          tl.cex = 0.8, tl.col = rgb(0, 0, 0))
## ---- end