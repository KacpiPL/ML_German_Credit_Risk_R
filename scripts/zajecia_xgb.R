
# cross validation
ctrl_cv3 <- trainControl(method = "cv", 
                         number = 3,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)

# 1 nrounds
parameters_xgb <- expand.grid(nrounds = seq(20, 80, 10),
                              max_depth = c(8),
                              eta = c(0.25), 
                              gamma = 1,
                              colsample_bytree = c(0.2),
                              min_child_weight = c(150),
                              subsample = 0.8)

datausa.xgb <- train(model1.formula,
                     data = datausa.train,
                     method = "xgbTree",
                     trControl = ctrl_cv3,
                     tuneGrid  = parameters_xgb)

datausa.xgb
datausa.xgb %>% saveRDS(here("output", "datausa.xgb.rds"))

# 2 max_depth
parameters_xgb2 <- expand.grid(nrounds = 80,
                               max_depth = seq(5, 15, 2),
                               eta = c(0.25), 
                               gamma = 1,
                               colsample_bytree = c(0.2),
                               min_child_weight = seq(200, 1000, 200),
                               subsample = 0.8)

set.seed(123456789)
datausa.xgb2 <- train(model1.formula,
                      data = datausa.train,
                      method = "xgbTree",
                      trControl = ctrl_cv3,
                      tuneGrid  = parameters_xgb2)

datausa.xgb2
datausa.xgb2 %>% saveRDS(here("output", "datausa.xgb2.rds"))


# 3 colsample_bytree
parameters_xgb3 <- expand.grid(nrounds = 80,
                               max_depth = 9,
                               eta = c(0.25), 
                               gamma = 1,
                               colsample_bytree = seq(0.1, 0.8, 0.1),
                               min_child_weight = 200,
                               subsample = 0.8)
set.seed(123456789)
datausa.xgb3 <- train(model1.formula,
                      data = datausa.train,
                      method = "xgbTree",
                      trControl = ctrl_cv3,
                      tuneGrid  = parameters_xgb3)
datausa.xgb3

datausa.xgb3 %>% saveRDS(here("output", "datausa.xgb3.rds"))

# 4 subsample
parameters_xgb4 <- expand.grid(nrounds = 80,
                               max_depth = 9,
                               eta = c(0.25), 
                               gamma = 1,
                               colsample_bytree = 0.7,
                               min_child_weight = 200,
                               subsample = c(0.6, 0.7, 0.75, 0.8, 0.85, 0.9))

set.seed(123456789)
datausa.xgb4 <- train(model1.formula,
                      data = datausa.train,
                      method = "xgbTree",
                      trControl = ctrl_cv3,
                      tuneGrid  = parameters_xgb4)
datausa.xgb4
datausa.xgb4 %>% saveRDS(here("output", "datausa.xgb4.rds"))

# 5
parameters_xgb5 <- expand.grid(nrounds = 160,
                               max_depth = 9,
                               eta = 0.12, 
                               gamma = 1,
                               colsample_bytree = 0.7,
                               min_child_weight = 200,
                               subsample = 0.9)

set.seed(123456789)
datausa.xgb5 <- train(model1.formula,
                      data = datausa.train,
                      method = "xgbTree",
                      trControl = ctrl_cv3,
                      tuneGrid  = parameters_xgb5)
datausa.xgb5
datausa.xgb5 %>% saveRDS(here("output", "datausa.xgb5.rds"))

# 6
parameters_xgb6 <- expand.grid(nrounds = 320,
                               max_depth = 9,
                               eta = 0.06, 
                               gamma = 1,
                               colsample_bytree = 0.7,
                               min_child_weight = 200,
                               subsample = 0.9)

set.seed(123456789)
datausa.xgb6 <- train(model1.formula,
                      data = datausa.train,
                      method = "xgbTree",
                      trControl = ctrl_cv3,
                      tuneGrid  = parameters_xgb6)
datausa.xgb6
datausa.xgb6 %>% saveRDS(here("output", "datausa.xgb6.rds"))



