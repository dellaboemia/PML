# Coursera Data Science Specialization
# Practical Machine Learning Course
# Week 4 Project
# John James
# April 22, 2016


library(caret)
## ---- compareModels
library(mlbench)
comparison <- resamples(list(XGB=model.xgbTree,
                             PARRF = model.parRF,
                             RF= model.rf,
                             WSRF=model.wsrf,
                             C5.0 = model.C5.0,
                             ESMBL1 = model.ensemble1,
                             ESMBL2 = model.ensemble2))


## ---- end

## ---- boxPlots
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(comparison, scales=scales)
## ---- end


## ---- pairWise
diffs <- diff(comparison)
summary(diffs)
## ---- end