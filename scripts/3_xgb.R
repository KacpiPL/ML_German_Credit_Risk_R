library(dplyr)
library(rpart)
library(rpart.plot)
library(xgboost)
library(plyr)
library(doParallel)
library(pROC)
library(ggplot2)
library(caret)

cores=detectCores()
registerDoParallel(cores=cores)

# Start with xgb
rm(list=ls())

df <- read.csv("./data/output/df.csv")
df.train <- read.csv("./data/output/df.train.csv")
df.test <- read.csv("./data/output/df.test.csv")

# Xgb requires target variable as factor - let's change it
df$class <- factor(df$class, levels = c(0, 1), labels = c("bad", "good"))
df.train$class <- factor(df.train$class, levels = c(0, 1), labels = c("bad", "good"))
df.test$class <- factor(df.test$class, levels = c(0, 1), labels = c("bad", "good"))

# Read columns chosen by rfe
rfe_columns <- read.csv("./data/output/rfe_columns.csv")

# Add "class" to the list of columns chosen for analysis
rfe_columns <- c("class", rfe_columns$x)

# Modify dfs to have columns chosen by rfe
df.train <- df.train[, rfe_columns]
df.test <- df.test[, rfe_columns]

# colsample_bytree - # rule of thumb (number of predictors)
sqrt(35)/35

# min_child_weight - 0.5 ... 0.1% of number of observations
0.005 * 2000  # 10
0.01 * 2000   # 20

##### Hyperparameter tuning #####
set.seed(12345678)
# Tune nrounds
xgb_grid <- expand.grid(nrounds = seq(20, 80, 10),
                              max_depth = c(8),
                              eta = c(0.25),
                              gamma = 1,
                              colsample_bytree = c(0.17),
                              min_child_weight = c(10),
                              subsample = 0.8)

train_control <- trainControl(method = "cv", 
                              number = 5,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

xgb_model.1 <- train(class ~ .,
                   data = df.train,
                   method = "xgbTree",
                   trControl = train_control,
                   tuneGrid  = xgb_grid)
xgb_model.1

xgb_model.1 %>% saveRDS("./data/output/xgb/xgb_model.1.rds")

# best for nrounds = 40

# Tune max_depth and min_child_weight
xgb_grid <- expand.grid(nrounds = 40,
                        max_depth = c(5, 15, 1),
                        eta = c(0.25),
                        gamma = 1,
                        colsample_bytree = c(0.17),
                        min_child_weight = seq(10, 200, 2),
                        subsample = 0.8)
set.seed(12345678)
xgb_model.2 <- train(class ~ .,
                     data = df.train,
                     method = "xgbTree",
                     trControl = train_control,
                     tuneGrid  = xgb_grid)
xgb_model.2

xgb_model.2 %>% saveRDS("./data/output/xgb/xgb_model.2.rds")

# The best for max_depth = 5, min_child_weight = 12

# Tune colsample_bytree
xgb_grid <- expand.grid(nrounds = 70,
                        max_depth = 5,
                        eta = c(0.25),
                        gamma = 1,
                        colsample_bytree = seq(0.05, 0.7, 0.01),
                        min_child_weight = 12,
                        subsample = 0.8)

set.seed(12345678)
xgb_model.3 <- train(class ~ .,
                     data = df.train,
                     method = "xgbTree",
                     trControl = train_control,
                     tuneGrid  = xgb_grid)
xgb_model.3

xgb_model.3 %>% saveRDS("./data/output/xgb/xgb_model.3.rds")

# The best for colsample_bytree = 0.65

# Tune subsample
xgb_grid <- expand.grid(nrounds = 70,
                        max_depth = 15,
                        eta = c(0.25),
                        gamma = 1,
                        colsample_bytree = 0.65,
                        min_child_weight = 14,
                        subsample = seq(0.6, 0.9, 0.05))

set.seed(12345678)
xgb_model.4 <- train(class ~ .,
                     data = df.train,
                     method = "xgbTree",
                     trControl = train_control,
                     tuneGrid  = xgb_grid)
xgb_model.4

xgb_model.4 %>% saveRDS("./data/output/xgb/xgb_model.4.rds")
# The best for subsample = 0.9

# Change learning rate and number of trees
xgb_grid <- expand.grid(nrounds = 140,
                        max_depth = 15,
                        eta = c(0.12),
                        gamma = 1,
                        colsample_bytree = 0.3,
                        min_child_weight = 14,
                        subsample = 0.9)

set.seed(12345678)
xgb_model.5 <- train(class ~ .,
                     data = df.train,
                     method = "xgbTree",
                     trControl = train_control,
                     tuneGrid  = xgb_grid)
xgb_model.5

xgb_model.5 %>% saveRDS("./data/output/xgb/xgb_model.5.rds")

# Change learning rate and number of trees again
xgb_grid <- expand.grid(nrounds = 280,
                        max_depth = 15,
                        eta = c(0.06),
                        gamma = 1,
                        colsample_bytree = 0.3,
                        min_child_weight = 14,
                        subsample = 0.9)

set.seed(12345678)
xgb_model.6 <- train(class ~ .,
                     data = df.train,
                     method = "xgbTree",
                     trControl = train_control,
                     tuneGrid  = xgb_grid)
xgb_model.6

xgb_model.6 %>% saveRDS("./data/output/xgb/xgb_model.6.rds")

source("./scripts/getAccuracyAndGini.R")

models <- c("1":"6")

sapply(paste0("xgb_model.", models),
       function(x) getAccuracyAndGini(model = get(x),
                                      data = df.test,
                                      target_variable = "class",
                                      predicted_class = "good")
)


# xgb_model.6 has all values highest - we choose this

# Calculate ROC
ROC.train <- pROC::roc(df.train$class, 
                       predict(xgb_model.6,
                               df.train, type = "prob")[, "good"])

ROC.test  <- pROC::roc(df.test$class, 
                       predict(xgb_model.6,
                               df.test, type = "prob")[, "good"])

cat("AUC for train = ", pROC::auc(ROC.train), 
    ", Gini for train = ", 2 * pROC::auc(ROC.train) - 1, "\n", sep = "")

cat("AUC for test = ", pROC::auc(ROC.test), 
    ", Gini for test = ", 2 * pROC::auc(ROC.test) - 1, "\n", sep = "")


# Chart of xgb ROC
list(
  ROC.train   = ROC.train,
  ROC.test    = ROC.test
) %>%
  pROC::ggroc(alpha = 0.5, linetype = 1, size = 1) + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color = "grey", 
               linetype = "dashed") +
  labs(subtitle = paste0("Gini TRAIN: ",
                         "xgb = ", 
                         round(100 * (2 * auc(ROC.train) - 1), 1), "%, ",
                         "Gini TEST: ",
                         "xgb = ", 
                         round(100 * (2 * auc(ROC.test) - 1), 1), "%, "
                         )) +
  theme_bw() + coord_fixed() +
  scale_color_brewer(palette = "Paired")

