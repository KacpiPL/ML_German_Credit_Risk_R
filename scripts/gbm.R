library(dplyr)
library(rpart)
library(rpart.plot)
library(xgboost)
library(plyr)
library(doParallel)
library(pROC)
library(ggplot2)
library(gbm)
library(caret)

cores=delightgbmcores=detectCores()
registerDoParallel(cores=cores)


rm(list=ls())

df <- read.csv("./data/output/df.csv")
df.train <- read.csv("./data/output/df.train.csv")
df.test <- read.csv("./data/output/df.test.csv")

df <- df[,-1]
df.train <- df.train[,-1]
df.test <- df.test[,-1]

# Make factor
df$class <- factor(df$class, levels = c(0, 1), labels = c("bad", "good"))
df.train$class <- factor(df.train$class, levels = c(0, 1), labels = c("bad", "good"))
df.test$class <- factor(df.test$class, levels = c(0, 1), labels = c("bad", "good"))

rfe_columns <- read.csv("./data/output/rfe_columns.csv")
rfe_columns <- c("class", rfe_columns$x)

df.train <- df.train[, rfe_columns]
df.test <- df.test[, rfe_columns]

# define model formula - all variables apart from class
model_formula <- NA
model_formula <- "class_1 ~"

for (i in 2:length(colnames(df.train))){
  if(i == 2){
    model_formula <- paste(model_formula, colnames(df.train)[i])
  } else {
    model_formula <- paste(model_formula, " + ", colnames(df.train)[i])
  }
}

model_formula <- as.formula(model_formula)

# gbm wants target variable to be 0 and 1
# Define new variable - class_1 with 0 and 1
df.train$class_1 <- (df.train$class == "good") * 1
df.test$class_1 <- (df.test$class == "good") * 1

##### Simple GBM #####
set.seed(123456789)
gbm.1 <- 
  gbm(model_formula,
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

source("./scripts/getAccuracyAndGini2.R")

# Training set
getAccuracyAndGini2(data = data.frame(class = df.train$class,
                                      pred = df.pred.train.gbm),
                    predicted_probs = "pred",
                    target_variable = "class",
                    target_levels = c("good", "bad"),
                    predicted_class = "good")

# Test set
getAccuracyAndGini2(data = data.frame(class = df.test$class,
                                      pred = df.pred.test.gbm),
                    predicted_probs = "pred",
                    target_variable = "class",
                    target_levels = c("good", "bad"),
                    predicted_class = "good")

ROC.train.gbm <- pROC::roc(df.train$class, 
                           df.pred.train.gbm)

ROC.test.gbm  <- pROC::roc(df.test$class, 
                           df.pred.test.gbm)

cat("AUC for train = ", pROC::auc(ROC.train.gbm), 
    ", Gini for train = ", 2 * pROC::auc(ROC.train.gbm) - 1, "\n", sep = "")

cat("AUC for test = ", pROC::auc(ROC.test.gbm), 
    ", Gini for test = ", 2 * pROC::auc(ROC.test.gbm) - 1, "\n", sep = "")


list(
  ROC.train.gbm = ROC.train.gbm,
  ROC.test.gbm  = ROC.test.gbm
) %>%
  pROC::ggroc(alpha = 0.5, linetype = 1, size = 1) + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color = "grey", 
               linetype = "dashed") +
  labs(subtitle = paste0("Gini TRAIN: ",
                         "gbm = ", 
                         round(100 * (2 * auc(ROC.train.gbm) - 1), 1), "%, ",
                         "Gini TEST: ",
                         "gbm = ", 
                         round(100 * (2 * auc(ROC.test.gbm) - 1), 1), "%, "
  )) +
  theme_bw() + coord_fixed() +
  scale_color_brewer(palette = "Paired")

##### Hyperparameter tuning #####










