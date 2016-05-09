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

## ---- validation
predV.nb	<- 	predict(model.nb,valDataCS)
predV.AdaBoost.M1	<- 	predict(model.AdaBoost.M1,valDataCS)
predV.gbm	<- 	predict(model.gbm,valDataCS)
predV.xgbTree	<- 	predict(model.xgbTree,valDataCS)
predV.pls	<- 	predict(model.pls,valDataCS)
predV.simpls	<- 	predict(model.simpls,valDataCS)
predV.widekernelpls	<- 	predict(model.widekernelpls,valDataCS)
predV.fda	<- 	predict(model.fda,valDataCS)
predV.mda	<- 	predict(model.mda,valDataCS)
predV.QdaCov	<- 	predict(model.QdaCov,valDataCS)
predV.pda	<- 	predict(model.pda,valDataCS)
predV.treebag	<- 	predict(model.treebag,valDataCS)
predV.rf	<- 	predict(model.rf,valDataCS)
predV.RRF	<- 	predict(model.RRF,valDataCS)
predV.wsrf	<- 	predict(model.wsrf,valDataCS)
predV.knn	<- 	predict(model.knn,valDataCS)
predV.avNNet	<- 	predict(model.avNNet,valDataCS)
predV.gcvEarth	<- 	predict(model.gcvEarth,valDataCS)
predV.multinom	<- 	predict(model.multinom,valDataCS)
predV.C5.0	<- 	predict(model.C5.0,valDataCS)
predV.C5.0Tree	<- 	predict(model.C5.0Tree,valDataCS)
predV.ctree	<- 	predict(model.ctree,valDataCS)
predV.rpart	<- 	predict(model.rpart,valDataCS)

#Combine all predictors into data frame.
pred.DFV <- data.frame(pred.nb = predV.nb,
                       pred.AdaBoost.M1=      predV.AdaBoost.M1,
                       pred.gbm = predV.gbm,
                       pred.xgbTree = predV.xgbTree,
                       pred.pls = predV.pls,
                       pred.simpls = predV.simpls,
                       pred.widekernelpls = predV.widekernelpls,
                       pred.fda = predV.fda,
                       pred.mda = predV.mda,
                       pred.QdaCov = predV.QdaCov,
                       pred.pda = predV.pda,
                       pred.treebag = predV.treebag,
                       pred.rf = predV.rf,
                       pred.RRF = predV.RRF,
                       pred.wsrf = predV.wsrf,
                       pred.knn = predV.knn,
                       pred.avNNet = predV.avNNet,
                       pred.gcvEarth = predV.gcvEarth,
                       pred.multinom = predV.multinom,
                       pred.C5.0 = predV.C5.0,
                       pred.C5.0Tree = predV.C5.0Tree,
                       pred.ctree = predV.ctree,
                       pred.rpart = predV.rpart)

#Calculate predictions
predV.ensemble1 <- predict(model.ensemble1, pred.DFV)
solution <- data.frame(problem_id = valDataCS$problem_id, classe = predV.ensemble1)
## ---- end