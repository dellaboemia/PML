# Coursera Data Science Specialization
# Practical Machine Learning Course
# Week 4 Project
# John James
# April 22, 2016

##############################################################################
#                               EXPLORE                                      #
##############################################################################
## ---- exploreData
library(funModeling)  

#Gather basic dimensions
observations <- dim(trainData)[1]
variables    <- dim(trainData)[2]

#Summarize zeros, NAs, data type, and number of unique values
trainDataStatus <- df_status(trainData)
naVars      <- subset(trainDataStatus, trainDataStatus$p_na > 80)
minNaVars   <- min(naVars$q_na)
pctNaVars   <- min(naVars$p_na)
numNaVars   <- nrow(naVars)
## ---- end

