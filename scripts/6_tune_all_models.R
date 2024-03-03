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
library(workflows)
library(parsnip)
library(recipes)
library(dials)
library(rsample)
library(tune)
library(yardstick)

rm(list=ls())

cores=delightgbmcores=detectCores()
registerDoParallel(cores=cores)

source("./scripts/tune_functions.R")

df <- read.csv("./data/output/df.csv")
df.train <- read.csv("./data/output/df.train.csv")
df.test <- read.csv("./data/output/df.test.csv")

# Make factor
df$class <- factor(df$class, levels = c(0, 1), labels = c("bad", "good"))
df.train$class <- factor(df.train$class, levels = c(0, 1), labels = c("bad", "good"))
df.test$class <- factor(df.test$class, levels = c(0, 1), labels = c("bad", "good"))

rfe_columns <- read.csv("./data/output/rfe_columns.csv")
rfe_columns <- c("class", rfe_columns$x)

df.train <- df.train[, rfe_columns]
df.test <- df.test[, rfe_columns]

# define parameters of models
models <- list(
  xgboost = list(
    model_name = 'xgboost',
    
    model_1 = boost_tree(mode = "classification", engine = "xgboost", learn_rate = tune()),
    
    model_2_args = list(tree_depth = tune(), 
                        loss_reduction = tune(), 
                        trees = tune(),
                        stop_iter = tune(),
                        min_n = tune()),
    
    rec_spec = recipe(class ~ ., df.train),
    
    grid_1 = set_seed_grid(grid_latin_hypercube(learn_rate(), size = 10), 123),
    
    grid_2 = set_seed_grid(as_tibble(expand.grid(
      tree_depth = c(6, 9, 12),
      loss_reduction = c(0.5, 1, 2),
      trees = c(100, 300, 500),
      stop_iter = seq(10, 50, by=10),
      min_n = seq(25, 100, by=25)
    )), 123)
  ),
  random_forest = list(
    model_name = 'random_forest',
    model_1 = rand_forest(mode = "classification", engine = "ranger", mtry = 5, trees = tune()),
    model_2_args = list(min_n = tune()),
    
    rec_spec = recipe(class ~ ., df.train),
    
    grid_1 = set_seed_grid(grid_latin_hypercube(trees(), size = 50), 123),
    
    grid_2 = set_seed_grid(as_tibble(expand.grid(
      min_n = c(75, 100, 150, 300)
    )), 123)
  ),
  decision_tree = list(
    model_name = 'decision_tree',
    model_1 = decision_tree(mode = "classification", engine = "rpart", cost_complexity = tune()),
    model_2_args = list(tree_depth = tune(), min_n = tune()),
    
    rec_spec = recipe(class ~ ., df.train),
    
    grid_1 = set_seed_grid(grid_latin_hypercube(cost_complexity(), size = 10), 123),
    grid_2 = set_seed_grid(as_tibble(expand.grid(
      tree_depth = c(6, 9, 12),
      min_n = seq(25, 100, by=25)
    )), 123)
  )
)

##### xgb ##### 
# xgb tune
result_list_xgb <- list(tune_two_stages(models, "xgboost"))
result_list_xgb <- result_list_xgb[[1]]

# xgb summarise
tuned_params_xgb <- c("trees", "min_n", "tree_depth", "loss_reduction", "stop_iter", "learn_rate")
results <- result_list_xgb$result

final_results_xgb <- summary_xgb(results, tuned_params_xgb)
write.csv(final_results_xgb, "./data/output/xgb/final_results_xgb.csv")

#write.csv(all_results_xgb, "./data/output/xgb/all_results_xgb.csv")

##### rf ##### 
# random_forest tune
result_list_random_forest <- list(tune_two_stages(models, "random_forest"))
result_list_random_forest <- result_list_random_forest[[1]]

# rf summarise
tuned_params_random_forest <- c("trees", "min_n")
results <- result_list_random_forest$result

final_results_random_forest <- summary_random_forest(results, tuned_params_random_forest)
write.csv(final_results_random_forest, "./data/output/rf/final_results_random_forest.csv")

##### decision_tree ##### 
# decision_tree tune
result_decision_tree <- list(tune_two_stages(models, "decision_tree"))
result_decision_tree <- result_decision_tree[[1]]

# decision_tree summarise
tuned_params_decision_tree <- c("cost_complexity", "tree_depth", "min_n")
results <- result_decision_tree$result

final_results_decision_tree <- summary_decision_tree(results, tuned_params_decision_tree)
write.csv(final_results_decision_tree, "./data/output/xgb/final_results_decision_tree.csv")



