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

## ---- ensemble1
pred.nb	<- 	predict(model.nb,testDataCS)
pred.AdaBoost.M1	<- 	predict(model.AdaBoost.M1,testDataCS)
pred.gbm	<- 	predict(model.gbm,testDataCS)
pred.xgbTree	<- 	predict(model.xgbTree,testDataCS)
pred.pls	<- 	predict(model.pls,testDataCS)
pred.simpls	<- 	predict(model.simpls,testDataCS)
pred.widekernelpls	<- 	predict(model.widekernelpls,testDataCS)
pred.fda	<- 	predict(model.fda,testDataCS)
pred.mda	<- 	predict(model.mda,testDataCS)
pred.QdaCov	<- 	predict(model.QdaCov,testDataCS)
pred.pda	<- 	predict(model.pda,testDataCS)
pred.treebag	<- 	predict(model.treebag,testDataCS)
pred.rf	<- 	predict(model.rf,testDataCS)
pred.RRF	<- 	predict(model.RRF,testDataCS)
pred.wsrf	<- 	predict(model.wsrf,testDataCS)
pred.knn	<- 	predict(model.knn,testDataCS)
pred.avNNet	<- 	predict(model.avNNet,testDataCS)
pred.gcvEarth	<- 	predict(model.gcvEarth,testDataCS)
pred.multinom	<- 	predict(model.multinom,testDataCS)
pred.C5.0	<- 	predict(model.C5.0,testDataCS)
pred.C5.0Tree	<- 	predict(model.C5.0Tree,testDataCS)
pred.ctree	<- 	predict(model.ctree,testDataCS)
pred.rpart	<- 	predict(model.rpart,testDataCS)

#Combine all predictors into data frame.
pred.DF <- data.frame(pred.nb,
                       pred.AdaBoost.M1,
                       pred.gbm,
                       pred.xgbTree,
                       pred.pls,
                       pred.simpls,
                       pred.widekernelpls,
                       pred.fda,
                       pred.mda,
                       pred.QdaCov,
                       pred.pda,
                       pred.treebag,
                       pred.rf,
                       pred.RRF,
                       pred.wsrf,
                       pred.knn,
                       pred.avNNet,
                       pred.gcvEarth,
                       pred.multinom,
                       pred.C5.0,
                       pred.C5.0Tree,
                       pred.ctree,
                       pred.rpart, classe = testDataCS$classe)

#Train all predictors
set.seed(1030)
model.ensemble1 <- train(classe~., data=pred.DF, trControl=ctrl, method="rf")
## ---- end