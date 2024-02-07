library(dplyr)
library(rpart)
library(rpart.plot)
library(xgboost)
library(plyr)
library(doParallel)
library(pROC)
library(ggplot2)
library(gbm)
library(randomForest)
library(ranger)
library(tidyverse)

cores=delightgbmcores=detectCores()
registerDoParallel(cores=cores)


rm(list=ls())

df <- read.csv("./data/output/df.csv")
df.train <- read.csv("./data/output/df.train.csv")
df.test <- read.csv("./data/output/df.test.csv")

df.train$class <- factor(df.train$class, levels = c(0, 1), labels = c("bad", "good"))
df.test$class <- factor(df.test$class, levels = c(0, 1), labels = c("bad", "good"))

# gbm wants target variable to be 0 and 1

rfe_columns <- read.csv("./data/output/rfe_columns.csv")
rfe_columns <- c("class", rfe_columns$x)

df.train <- df.train[, rfe_columns]
df.test <- df.test[, rfe_columns]

rf <- randomForest(class ~ ., 
                   data = df.train)

saveRDS(rf, file = "./data/output/rf/rf.RDS")

  
getOOBSampleSize <- function(nObs) {
  sample(1:nObs, size = nObs, replace = T) %>%
    unique() %>%
    length()
}   

nReps <- 10000
nObs  <- 2000
time0 <- Sys.time()
results <- replicate(nReps, getOOBSampleSize(nObs = nObs))
time1 <- Sys.time()
time1 - time0

results %>% 
  enframe() %>%
  ggplot(aes(value)) +
  geom_histogram(binwidth = 10, col = "black", fill = "pink") +
  labs(
    title = "OOB sample size distribution",
    subtitle = paste0("nObs = ", nObs, ", nReps = ", nReps),
    caption = "Source: own simulations"
  )

results %>% 
  enframe() %>%
  ggplot(aes(name, value)) +
  geom_line(col = "pink")

print(rf)

plot(rf)

# appropriate number of trees 120

rf2 <- 
  randomForest(class ~ .,
               data = df.train,
               ntree = 100,
               sampsize = nrow(df.train),
               mtry = 8,
               # minimum number of obs in the terminal nodes
               nodesize = 100,
               # we also generate predictors importance measures,
               importance = TRUE)

saveRDS(rf2, file = "./data/output/rf/rf2.RDS")

parameters_rf <- expand.grid(mtry = 2:15)
ctrl_oob <- trainControl(method = "oob", classProbs = TRUE)

set.seed(123456789)

rf3 <-
  train(class ~ .,
        data = df.train,
        method = "rf",
        ntree = 100,
        nodesize = 100,
        tuneGrid = parameters_rf,
        trControl = ctrl_oob,
        importance = TRUE)

rf3
saveRDS(rf3, file = "./data/output/rf/rf3.RDS") 

plot(rf3$results$mtry,
     rf3$results$Accuracy, type = "b")  

plot(rf3$results$mtry,
     rf3$results$Kappa, type = "b") 
  
  
# mtry = 10
modelLookup("ranger")

parameters_ranger <- 
  expand.grid(mtry = 4:15,
              # split rule
              splitrule = "gini",
              # minimum size of the terminal node
              min.node.size = c(50, 100, 150))

ctrl_cv5 <- trainControl(method = "cv", 
                         number =    5,
                         classProbs = T)

set.seed(123456789)
rf3a <- 
  train(class ~ ., 
        data = df.train, 
        method = "ranger", 
        num.trees = 100, # default = 500
        # numbers of processor cores to use in computations
        num.threads = 3,
        # impurity measure
        importance = "impurity",
        # parameters
        tuneGrid = parameters_ranger, 
        trControl = ctrl_cv5)

saveRDS(rf3a, file = "./data/output/rf/rf3a.RDS") 

rf3a

plot(rf3a)




pred.train.rf <- predict(rf, 
                         df.train, 
                         type = "prob")[, "good"]
ROC.train.rf  <- roc(as.numeric(df.train$class == "good"), 
                     pred.train.rf)


pred.test.rf  <- predict(rf, 
                         df.test, 
                         type = "prob")[, "good"]
ROC.test.rf   <- roc(as.numeric(df.test$class == "good"), 
                     pred.test.rf)


pred.train.rf2 <- predict(rf2, 
                          df.train, 
                          type = "prob")[, "good"]
ROC.train.rf2  <- roc(as.numeric(df.train$class == "good"), 
                      pred.train.rf2)


pred.test.rf2  <- predict(rf2, 
                          df.test, 
                          type = "prob")[, "good"]
ROC.test.rf2   <- roc(as.numeric(df.test$class == "good"), 
                      pred.test.rf2)


pred.train.rf3 <- predict(rf3, 
                          df.train, 
                          type = "prob")[, "good"]
ROC.train.rf3  <- roc(as.numeric(df.train$class == "good"), 
                      pred.train.rf3)


pred.test.rf3  <- predict(rf3, 
                          df.test, 
                          type = "prob")[, "good"]
ROC.test.rf3   <- roc(as.numeric(df.test$class == "good"), 
                      pred.test.rf3)


pred.train.rf3a <- predict(rf3a, 
                           df.train, 
                           type = "prob")[, "good"]
ROC.train.rf3a  <- roc(as.numeric(df.train$class == "good"), 
                       pred.train.rf3a)


pred.test.rf3a  <- predict(rf3a, 
                           df.test, 
                           type = "prob")[, "good"]
ROC.test.rf3a   <- roc(as.numeric(df.test$class == "good"), 
                       pred.test.rf3a)

list(
  ROC.train.rf   = ROC.train.rf,
  ROC.test.rf    = ROC.test.rf,
  ROC.train.rf2  = ROC.train.rf2,
  ROC.test.rf2   = ROC.test.rf2,
  ROC.train.rf3  = ROC.train.rf3,
  ROC.test.rf3   = ROC.test.rf3,
  ROC.train.rf3a = ROC.train.rf3a,
  ROC.test.rf3a  = ROC.test.rf3a
) %>%
  pROC::ggroc(alpha = 0.5, linetype = 1, size = 1) + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color = "grey", 
               linetype = "dashed") +
  labs(title = paste0("Gini TEST: ",
                      "rf = ", 
                      round(100 * (2 * auc(ROC.test.rf) - 1), 1), "%, ",
                      "rf2 = ", 
                      round(100 * (2 * auc(ROC.test.rf2) - 1), 1), "%, ",
                      "rf3 = ", 
                      round(100 * (2 * auc(ROC.test.rf3) - 1), 1), "%, ",
                      "rf3a = ", 
                      round(100 * (2 * auc(ROC.test.rf3a) - 1), 1), "% "),
       subtitle =  paste0("Gini TRAIN: ",
                          "rf = ", 
                          round(100 * (2 * auc(ROC.train.rf) - 1), 1), "%, ",
                          "rf2 = ", 
                          round(100 * (2 * auc(ROC.train.rf2) - 1), 1), "%, ",
                          "rf3 = ", 
                          round(100 * (2 * auc(ROC.train.rf3) - 1), 1), "%, ",
                          "rf3a = ", 
                          round(100 * (2 * auc(ROC.train.rf3a) - 1), 1), "% ")) +
  theme_bw() + coord_fixed() +
  scale_color_brewer(palette = "Paired")

getacc




  