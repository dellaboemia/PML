# Coursera Data Science Specialization
# Practical Machine Learning Course
# Week 4 Project
# John James
# April 22, 2016


##############################################################################
#                             ENVIRONMENT                                    #
##############################################################################
## ---- environment
library(adabag)
library(C50)
library(class)
library(e1071)
library(earth)
library(elmNN)
library(foreach)
library(gbm)
library(gpls)
library(ipred)
library(klaR)
library(mda)
library(nnet)
library(party)
library(doParallel)
library(pls)
library(plyr)
library(R.cache)
library(randomForest)
library(rpart)
library(rrcov)
library(RRF)
library(RSNNS)
library(RWeka)
library(vbmp)
library(wsrf)
library(xgboost)
## ---- end


##############################################################################
#                             READ DATA                                      #
##############################################################################
## ---- readData
trainUrl   <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl    <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainUrl, destfile = "trainFile.csv")
download.file(testUrl, destfile = "testFile.csv") 

trainFile  <- read.csv("trainFile.csv", na.strings=c('#DIV/0', '', 'NA'))
testFile   <- read.csv("testFile.csv", na.strings=c('#DIV/0', '', 'NA'))

## ---- end
