library(dplyr)
library(rpart)
library(rpart.plot)
library(xgboost)
library(plyr)
library(doParallel)
library(pROC)

rm(list=ls())

df <- read.csv("./data/output/df.csv")

df.train <- read.csv("./data/output/df.train.csv")
df.test <- read.csv("./data/output/df.test.csv")

# Delete the id from the dataset
df <- df[,-1]
df.train <- df.train[,-1]
df.test <- df.test[,-1]


cores=detectCores()
registerDoParallel(cores=cores)

# Let us examine frequencies of the target variable values:
# good - 1
# bad - 0

# table(df$class)
# table(df$class)/length(df$class)
# 
# # Let us also examine frequencies of the target variable inside the training and the testing set.
# table(df.train$class)/length(df.train$class)
# table(df.test$class)/length(df.test$class)
# 
# # Create the tree where all columns as x
# model1.formula <- class ~ .
# 
# risk.tree1 <- 
#   rpart(model1.formula, # model formula
#         data = df.train, # data
#         method = "class") # type of the tree: classification
# 
# risk.tree1
# rpart.plot(risk.tree1)  
# fancyRpartPlot(risk.tree1)



# Scaling NOT needed:
## decision tree
## random forest
## gmb
## XGB
## ADA - if applied to decision tree no needed, if svm needed

# Start with xgb

colnames(df)
rm(list=ls())

df <- read.csv("./data/output/df.csv")
df.train <- read.csv("./data/output/df.train.csv")
df.test <- read.csv("./data/output/df.test.csv")

df <- df[,-1]
df.train <- df.train[,-1]
df.test <- df.test[,-1]

df$class <- factor(df$class, levels = c(0, 1), labels = c("bad", "good"))
df.train$class <- factor(df.train$class, levels = c(0, 1), labels = c("bad", "good"))
df.test$class <- factor(df.test$class, levels = c(0, 1), labels = c("bad", "good"))

table(df.train$class)
table(df.test$class)

# colsample_bytree - # rule of thumb (number of predictors)
sqrt(49)/49

# min_child_weight - 0.5 ... 0.1% of number of observations
0.005 * 2000
0.01 * 2000

# Hyperpatrameter tuning
# Example using caret for grid search
# xgb_grid <- expand.grid(
#   nrounds = seq(20, 100, 10),
#   max_depth = seq(5, 9, 1),
#   gamma = c(0, 1),
#   eta = c(0.05, 0.1, 0.2),               # +
#   colsample_bytree = c(0.5, 0.75, 1),    # +
#   min_child_weight = c(10, 15, 20),      # +
#   subsample = c(0.5, 0.75, 1)
# )


xgb_grid <- expand.grid(
  nrounds = 100,
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 6, 9),
  gamma = 0,
  colsample_bytree = c(0.5, 0.75, 1),
  min_child_weight = c(1, 2, 3),
  subsample = c(0.5, 0.75, 1)
)

# parameters_xgb <- expand.grid(nrounds = seq(20, 80, 10),
#                               max_depth = c(8),
#                               eta = c(0.25), 
#                               gamma = 1,
#                               colsample_bytree = c(0.2),
#                               min_child_weight = c(150),
#                               subsample = 0.8)

train_control <- trainControl(method = "cv", 
                              number = 5,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

xgb_model <- train(class ~ .,
                   data = df.train,
                   method = "xgbTree",
                   trControl = train_control,
                   tuneGrid  = xgb_grid)

xgb_model

results <- xgb_model$results

# Model evaluation
predictions <- predict(xgb_model, df.test)
conf_matrix <- confusionMatrix(predictions, df.test$class)
print(conf_matrix)

ROC.train <- pROC::roc(df.train$class, 
                       predict(xgb_model,
                               df.train, type = "prob")[, "good"])

ROC.test <- pROC::roc(df.test$class, 
                       predict(xgb_model,
                               df.test, type = "prob")[, "good"])

plot(ROC.train)
plot(ROC.test)

predictions2 <- predict(xgb_model, df.train)
conf_matrix <- confusionMatrix(predictions2, df.train$class)
print(conf_matrix)

# Feature importance
importance_matrix <- xgb.importance(feature_names = colnames(train_data), model = xgb_model)
xgb.plot.importance(importance_matrix)


##### xgb na pca
xgb_model_pca <- train(class ~ .,
                   data = df_train_pca,
                   method = "xgbTree",
                   trControl = train_control,
                   tuneGrid  = xgb_grid)

results_pca <- xgb_model$results

predictions_pca <- predict(xgb_model_pca, df_test_pca)
conf_matrix_pca <- confusionMatrix(predictions_pca, df_test_pca$class)
print(conf_matrix_pca)


# xgb na rfe

df.train.rfe <- df.train[, c("class", rfe_columns)]
df.test.rfe <- df.test[, c("class", rfe_columns)]

xgb_model_rfe <- train(class ~ .,
                       data = df.train.rfe,
                       method = "xgbTree",
                       trControl = train_control,
                       tuneGrid  = xgb_grid)

results_rfe <- xgb_model_rfe$results

predictions_rfe <- predict(xgb_model_rfe, df.test.rfe)

conf_matrix_rfe <- confusionMatrix(predictions_rfe, df.test.rfe$class)


print(conf_matrix)
print(conf_matrix_pca)
print(conf_matrix_rfe)



