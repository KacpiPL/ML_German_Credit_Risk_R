library(dplyr)
library(rpart)
library(rpart.plot)
library(xgboost)
library(plyr)
library(doParallel)
library(pROC)
library(ggplot2)
library(gbm)

cores=delightgbmcores=detectCores()
registerDoParallel(cores=cores)


rm(list=ls())

df <- read.csv("./data/output/df.csv")
df.train <- read.csv("./data/output/df.train.csv")
df.test <- read.csv("./data/output/df.test.csv")

df <- df[,-1]
df.train <- df.train[,-1]
df.test <- df.test[,-1]


# gbm wants target variable to be 0 and 1

rfe_columns <- read.csv("./data/output/rfe_columns.csv")
rfe_columns <- c("class", rfe_columns$x)

df.train <- df.train[, rfe_columns]
df.test <- df.test[, rfe_columns]

set.seed(123456789)

gbm.1 <- 
  gbm(class ~ .,
      data = df.train,
      distribution = "bernoulli",
      # total number of trees
      n.trees = 500,
      # number of variable interactions - actually depth of the trees
      interaction.depth = 4,
      # shrinkage parameter - speed (pace) of learning
      shrinkage = 0.01,
      verbose = FALSE)

gbm.1 %>% saveRDS("./data/output/gbm/gbm.1.rds")

# predict train
df.pred.train.gbm <- predict(gbm.1,
                                  df.train, 
                                  type = "response",
                                  n.trees = 500)
# predict test
df.pred.test.gbm <- predict(gbm.1,
                             df.test, 
                             type = "response",
                             n.trees = 500)

# Convert probabilities to class predictions (0 or 1) based on a 0.5 threshold
df.pred.train.class <- ifelse(df.pred.train.gbm > 0.5, 1, 0)
df.pred.train.class <- factor(df.pred.train.class, levels = c(0, 1))

df.train$class <- factor(df.train$class, levels = c(0, 1))


df.pred.test.class <- ifelse(df.pred.test.gbm > 0.5, 1, 0)
df.pred.test.class <- factor(df.pred.test.class, levels = c(0, 1))

df.test$class <- factor(df.test$class, levels = c(0, 1))

# Confusion Matrix for Training data
confusionMatrix(df.pred.train.class, df.train$class)

# Confusion Matrix for Test data
confusionMatrix(df.pred.test.class, df.test$class)

# Gini coefficient (using AUC)
# For Training data
auc.train <- roc(df.train$class, df.pred.train.gbm)$auc
gini.train <- 2 * auc.train - 1

# For Test data
auc.test <- roc(df.test$class, df.pred.test.gbm)$auc
gini.test <- 2 * auc.test - 1

# Print Gini coefficients
print(paste("Gini Coefficient for Training Data: ", gini.train))
print(paste("Gini Coefficient for Test Data: ", gini.test))

##### grid search #####




