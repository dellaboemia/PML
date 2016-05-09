# Coursera Data Science Specialization
# Practical Machine Learning Course
# Week 4 Project
# John James
# April 22, 2016


##############################################################################
#                           CREATE MODELS                                    #
##############################################################################

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

## ---- modelData
#Setup parallel processing
noCores <- detectCores()-1

##############################################################################
#                           BAYESIAN MODELS                                  #
##############################################################################

#                             NAIVE BAYES                                    #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running nb model"); print("Running nb model")
model.nb<- train(classe~., data=trainDataCS, trControl=ctrl, method="nb")
stopCluster(cl)
save.image()

##############################################################################
#                           BOOSTING ALGORITHMS                              #
##############################################################################

#                             ADABOOST.M1                                    #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 9)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running AdaBoost.M1 model")
model.AdaBoost.M1<- train(classe~., data=trainDataCS, trControl=ctrl, method="AdaBoost.M1")
stopCluster(cl)
save.image()

#                                   GBM                                     #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 9)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running gbm model")
model.gbm<- train(classe~., data=trainDataCS, trControl=ctrl, method="gbm")
stopCluster(cl)
save.image()

#                                     xgbTree                                         #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 12)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running xgbTree model")
model.xgbTree<- train(classe~., data=trainDataCS, trControl=ctrl, method="xgbTree")
stopCluster(cl)
save.image()


##############################################################################
#                           PARTIAL LEAST SQUARES                            #
##############################################################################

#                      PARTIAL LEAST SQUARES                                #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 1)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running pls model")
model.pls<- train(classe~., data=trainDataCS, trControl=ctrl, method="pls")
stopCluster(cl)
save.image()

#                      PARTIAL LEAST SQUARES (simpls)                        #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 1)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running simpls model")
model.simpls<- train(classe~., data=trainDataCS, trControl=ctrl, method="simpls")
stopCluster(cl)
save.image()

#                               WIDER KERNEL PLS                        #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 1)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running widekernelpls model")
model.widekernelpls<- train(classe~., data=trainDataCS, trControl=ctrl, method="widekernelpls")
stopCluster(cl)
save.image()

##############################################################################
#                           DISCRIMINANT ANALYSIS                            #
##############################################################################
#                         FLEXIBLE DISCRIMINANT ANALYSIS                     #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running fda model")
model.fda<- train(classe~., data=trainDataCS, trControl=ctrl, method="fda")
stopCluster(cl)
save.image()

#                         MIXTURE DISCRIMINANT ANALYSIS                     #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running mda model")
model.mda<- train(classe~., data=trainDataCS, trControl=ctrl, method="mda")
stopCluster(cl)
save.image()

#                 ROBUST QUANTILE DISCRIMINANT ANALYSIS                     #
#Set Seeds
set.seed(1030);
# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3)
#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running QdaCov model")
model.QdaCov<- train(classe~., data=trainDataCS, trControl=ctrl, method="QdaCov")
stopCluster(cl)
save.image()

#                         PENALIZED DISCRIMINANT ANALYSIS                     #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running pda model") 
model.pda<- train(classe~., data=trainDataCS, trControl=ctrl, method="pda")
stopCluster(cl)
save.image()

#                         REGULARIZED DISCRIMINANT ANALYSIS                     #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 9)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running rda model") 
model.rda<- train(classe~., data=trainDataCS, trControl=ctrl, method="rda")
stopCluster(cl)
save.image()

##############################################################################
#                               BAGGED METHODS                               #
##############################################################################
#                                 TREEBAG                                    #
# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running treebag model") 
model.treebag<- train(classe~., data=trainDataCS, trControl=ctrl, method="treebag")
stopCluster(cl)
save.image()

##############################################################################
#                              ENSEMBLE METHODS                              #
##############################################################################
#                             PARALLEL RANDOM FOREST                         #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running parRF model")
model.parRF<- train(classe~., data=trainDataCS, trControl=ctrl, method="parRF")
stopCluster(cl)
save.image()

#                              RANDOM FOREST                                 #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running rf model")
model.rf<- train(classe~., data=trainDataCS, trControl=ctrl, method="rf")
stopCluster(cl)
save.image()

#                         REGULARIZED RANDOM FOREST                        #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running RRF model")
model.RRF<- train(classe~., data=trainDataCS, trControl=ctrl, method="RRF")
stopCluster(cl)
save.image()

#                         WEIGHTED SUBSPACE RANDOM FOREST                        #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running wsrf model")
model.wsrf<- train(classe~., data=trainDataCS, trControl=ctrl, method="wsrf")
stopCluster(cl)
save.image()

##############################################################################
#                           INSTANCE BASED METHODS                           #
##############################################################################
#                            K NEAREST NEIGHBOR                              #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 22)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running knn model")
model.knn<- train(classe~., data=trainDataCS, trControl=ctrl, method="knn")
stopCluster(cl)
save.image()


##############################################################################
#                              NEURAL NETWORKS                               #
##############################################################################
#                     MODEL AVERAGED NEURAL NETWORK                         #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 9)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running avNNet model")
model.avNNet<- train(classe~., data=trainDataCS, trControl=ctrl, method="avNNet")
stopCluster(cl)
save.image()

#                         Multi-Layer Perceptron                            #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("mlp")
model.mlp<- train(classe~., data=trainDataCS, trControl=ctrl, method="mlp")
stopCluster(cl)
save.image()

##############################################################################
#           #Multivariate Adaptive Regression Splines (MARS)                 #
##############################################################################
#               Multivariate Adaptive Regression Splines (MARS)              #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 1)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running gcvEarth model")
model.gcvEarth<- train(classe~., data=trainDataCS, trControl=ctrl, method="gcvEarth")
stopCluster(cl)
save.image()

#                     Logistic and Multinomial Regression                    #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running multinom model")
model.multinom<- train(classe~., data=trainDataCS, trControl=ctrl, method="multinom")
stopCluster(cl)
save.image()

##############################################################################
#                           DECISION BASED METHODS                           #
##############################################################################
#                                     C5.0                                   #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 4)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running C5.0 model")
model.C5.0<- train(classe~., data=trainDataCS, trControl=ctrl, method="C5.0")
stopCluster(cl)
save.image()

#                                     C5.0Tree                              #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running C5.0Tree model")
model.C5.0Tree<- train(classe~., data=trainDataCS, trControl=ctrl, method="C5.0Tree")
stopCluster(cl)
save.image()

#                                     CTREE                              #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running ctree model")
model.ctree<- train(classe~., data=trainDataCS, trControl=ctrl, method="ctree")
stopCluster(cl)
save.image()

#                                     J48                                #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running J48 model")
model.J48<- train(classe~., data=trainDataCS, trControl=ctrl, method="J48")
stopCluster(cl)
save.image()

#                                     JRIP                                #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running JRip model")
model.JRip<- train(classe~., data=trainDataCS, trControl=ctrl, method="JRip")
stopCluster(cl)
save.image()

#                                     RPART                                #
#Set Seeds
set.seed(1030); seeds <- vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 3)
seeds[[31]]<- sample.int(n=1000, 1)

# Train Control
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, seeds=seeds)

#Train Model
cl <- makeCluster(noCores); registerDoParallel(cl); print("Running rpart model")
model.rpart<- train(classe~., data=trainDataCS, trControl=ctrl, method="rpart")
stopCluster(cl)
save.image()
## ---- end