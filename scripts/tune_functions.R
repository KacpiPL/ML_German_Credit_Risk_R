##### function to tune models
tune_two_stages <- function(models, model_name) {
  # Extract parameters based on the model_name
  model_params <- models[[model_name]]
  
  # Stage 1 tuning
  set.seed(123)
  wflw_model_1 <- workflow() %>%
    add_model(model_params$model_1) %>%
    add_recipe(model_params$rec_spec)
  
  set.seed(123)
  # ilość cv
  cv_folds <- vfold_cv(df_train, v = 10)
  
  tune_stage_1 <- tune_grid(
    wflw_model_1,
    resamples = cv_folds,
    grid      = model_params$grid_1,
    metrics   = metric_set(roc_auc),
    control   = control_grid(verbose = TRUE)
  )
  
  # Choose the best model
  best_params_model_1 <- tune_stage_1 %>% collect_metrics() %>% arrange(-mean) %>%
    filter(row_number() == 1)
  
  print(best_params_model_1)
  # Stage 2 tuning
  set.seed(123)
  param_name <- names(model_params$grid_1)[1]
  param_value <- best_params_model_1[[1]]
  
  # Set up model_2
  model_params$model_2_args[[param_name]] <- param_value
  model_2 <- model_params$model_1 %>%
    set_args(
      !!!model_params$model_2_args  
    )
  
  wflw_model_2 <- wflw_model_1 %>%
    update_model(model_2)
  
  # Tune stage 2
  set.seed(123)
  tune_stage_2 <- tune_grid(
    wflw_model_2,
    resamples = cv_folds,
    grid      = model_params$grid_2,
    metrics   = metric_set(roc_auc),
    control   = control_grid(verbose = TRUE)
  )
  
  all_results <- tune_stage_2 %>% collect_metrics() %>% arrange(-mean) 
  
  result <- all_results %>% filter(row_number() <= 3)
  result[[param_name]] <- best_params_model_1[[1]]
  
  return(list(result = result, all_results = all_results))
}


# function to generate roc auc plot
plot_roc <- function(ROC.train, ROC.test){
  
  roc_plot <- list(
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
  
  return(roc_plot)
}

# function to summarize xgb results
summary_xgb <- function(results, tuned_params_xgb){
  
  results_df <- data.frame()
  
  for (i in 1:3) {
    tuned_params <- results[i, tuned_params_xgb]
    
    rounded_params <- round(as.numeric(tuned_params), 2)
    
    xgb_model = boost_tree(mode = "classification",
                           engine = "xgboost",
                           trees = rounded_params[1],
                           min_n = rounded_params[2],
                           tree_depth = rounded_params[3],
                           loss_reduction = rounded_params[4],
                           stop_iter = rounded_params[5],
                           learn_rate = rounded_params[6])
    
    rec_spec <- recipe(class ~ ., df_train)
    
    wflw_model <- workflow() %>%
      add_model(xgb_model) %>%
      add_recipe(rec_spec)
    
    trained_model <- wflw_model %>% fit(data = df_train)
    
    # Generate predictions - train
    predictions_train <- predict(trained_model, new_data = df_train, type = "prob")
    
    ROC.train <- pROC::roc(df_train$class,
                           predictions_train$.pred_good)
    
    auc_train <- auc(ROC.train) 
    gini_train <- round(100 * (2 * auc(ROC.train) - 1), 1)
    
    # Generate predictions - test
    predictions_test <- predict(trained_model, new_data = df_test, type = "prob")
    
    ROC.test <- pROC::roc(df_test$class,
                          predictions_test$.pred_good)
    
    auc_test <- auc(ROC.test)
    gini_test <- round(100 * (2 * auc(ROC.test) - 1), 1)
    
    # sgenerate and save roc plot
    roc_plot <- plot_roc(ROC.train, ROC.test)
    path <- paste("./plots/xgb_", i, ".png", sep="")
    ggsave(path, roc_plot, width = 10, height = 8, dpi = 300)
    
    iteration_df <- data.frame(
      iteration = i,
      trees = rounded_params[1],
      min_n = rounded_params[2],
      tree_depth = rounded_params[3],
      loss_reduction = rounded_params[4],
      stop_iter = rounded_params[5],
      learn_rate = rounded_params[6],
      
      auc_train = auc_train,
      gini_train = gini_train,
      
      auc_test = auc_test,
      gini_test = gini_test,
      
      models_spec = paste(names(tuned_params), "=", rounded_params, collapse = ", "),
      model_name = "xgboost"
    )
    
    results_df <- rbind(results_df, iteration_df)
  }
  
  return(results_df)
}


##### summary random_forest ##### 
summary_random_forest <- function(results, tuned_params_random_forest){
  
  results_df <- data.frame()
  
  for (i in 1:3) {
    tuned_params <- results[i, tuned_params_random_forest]
    
    rounded_params <- round(as.numeric(tuned_params), 2)
    
    random_forest_model = rand_forest(
      mode = "classification",
      engine = "ranger",
      mtry = 5,
      trees = rounded_params[1],
      min_n = rounded_params[2]
    )
    
    rec_spec <- recipe(class ~ ., df_train)
    
    wflw_model <- workflow() %>%
      add_model(random_forest_model) %>%
      add_recipe(rec_spec)
    
    trained_model <- wflw_model %>% fit(data = df_train)
    
    # Generate predictions - train
    predictions_train <- predict(trained_model, new_data = df_train, type = "prob")
    
    ROC.train <- pROC::roc(df_train$class,
                           predictions_train$.pred_good)
    
    auc_train <- auc(ROC.train) 
    gini_train <- round(100 * (2 * auc(ROC.train) - 1), 1)
    
    # Generate predictions - test
    predictions_test <- predict(trained_model, new_data = df_test, type = "prob")
    
    ROC.test <- pROC::roc(df_test$class,
                          predictions_test$.pred_good)
    
    auc_test <- auc(ROC.test)
    gini_test <- round(100 * (2 * auc(ROC.test) - 1), 1)
    
    iteration_df <- data.frame(
      iteration = i,
      trees = rounded_params[1],
      min_n = rounded_params[2],
      
      auc_train = auc_train,
      gini_train = gini_train,
      
      auc_test = auc_test,
      gini_test = gini_test,
      
      models_spec = paste(names(tuned_params), "=", rounded_params, collapse = ", "),
      model_name = "random_forest"
    )
    
    results_df <- rbind(results_df, iteration_df)
  }
  
  return(results_df)
}


##### summary decision_tree ##### 
summary_decision_tree <- function(results, tuned_params_decision_tree){
  
  results_df <- data.frame()
  
  for (i in 1:3) {
    tuned_params <- results[i, tuned_params_decision_tree]
    rounded_params <- round(as.numeric(tuned_params), 2)
    
    decision_tree_model = decision_tree(mode = "classification", 
                                        engine = "rpart", 
                                        cost_complexity = rounded_params[1],
                                        tree_depth = rounded_params[2],
                                        min_n = rounded_params[3])
    
    rec_spec <- recipe(class ~ ., df_train)
    
    wflw_model <- workflow() %>%
      add_model(decision_tree_model) %>%
      add_recipe(rec_spec)
    
    trained_model <- wflw_model %>% fit(data = df_train)
    
    # Generate predictions - train
    predictions_train <- predict(trained_model, new_data = df_train, type = "prob")
    
    ROC.train <- pROC::roc(df_train$class,
                           predictions_train$.pred_good)
    
    auc_train <- auc(ROC.train) 
    gini_train <- round(100 * (2 * auc(ROC.train) - 1), 1)
    
    # Generate predictions - test
    predictions_test <- predict(trained_model, new_data = df_test, type = "prob")
    
    ROC.test <- pROC::roc(df_test$class,
                          predictions_test$.pred_good)
    
    auc_test <- auc(ROC.test)
    gini_test <- round(100 * (2 * auc(ROC.test) - 1), 1)
    
    iteration_df <- data.frame(
      iteration = i,
      cost_complexity = rounded_params[1],
      tree_depth = rounded_params[2],
      min_n = rounded_params[3],
      
      auc_train = auc_train,
      gini_train = gini_train,
      
      auc_test = auc_test,
      gini_test = gini_test,
      
      models_spec = paste(names(tuned_params), "=", rounded_params, collapse = ", "),
      model_name = "decision_tree"
    )
    
    results_df <- rbind(results_df, iteration_df)
  }
  
  return(results_df)
}