# Coursera Data Science Specialization
# Practical Machine Learning Course
# Week 4 Project
# John James
# April 22, 2016


library(caret)
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

## ---- ensemble2
#Predict Best models
pred.AdaBoost.M1	<- 	predict(model.AdaBoost.M1,testDataCS)
pred.gbm	<- 	predict(model.gbm,testDataCS)
pred.xgbTree	<- 	predict(model.xgbTree,testDataCS)
pred.treebag	<- 	predict(model.treebag,testDataCS)
pred.rf	<- 	predict(model.rf,testDataCS)
pred.RRF	<- 	predict(model.RRF,testDataCS)
pred.wsrf	<- 	predict(model.wsrf,testDataCS)
pred.knn	<- 	predict(model.knn,testDataCS)
pred.C5.0	<- 	predict(model.C5.0,testDataCS)
pred.C5.0Tree	<- 	predict(model.C5.0Tree,testDataCS)

#Combine Best predictors into data frame.
pred.DF <- data.frame( pred.AdaBoost.M1,
                       pred.gbm,
                       pred.xgbTree,
                       pred.treebag,
                       pred.rf,
                       pred.RRF,
                       pred.wsrf,
                       pred.knn,
                       pred.C5.0,
                       pred.C5.0Tree,
                       classe = testDataCS$classe)

#Train all predictors
set.seed(1030)
model.ensemble2 <- train(classe~., data=pred.DF, trControl=ctrl, method="rf")
## ---- end