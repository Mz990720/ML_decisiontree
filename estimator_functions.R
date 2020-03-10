# 估计量函数 
# Given a set of features of n observations, actions taken by each observation, 
# and reward acquired by each observation, calculate
# all Gammas (estimates of rewards for each observation, for each action) using 
# doubly robust/augmented inverse propensity weighting,
# direct method/causal forest regression, and
# inverse propensity scores/inverse propensity weighting.
# Returns gammas for both training data and test data; the rewards of the test 
# data are estimated using models which were learned on the training data.
# All probability and mean reward estimates are derived using random forest models,
# either direct random forests or causal random forests, depending on the value of 
# the 'direct'parameter.
# 
# inputs:
#   features: features for each observation
#   rewards: observed rewards
#   actions: actions taken in the dataset
#   k: number of actions
#   train_prob_type: whether the probability for each action in the training
#     data is known ('exact') or unknown and needs to be estimated ('est').
#     In randomized control trials, the probabilities are usually known; for 
#     other real-world data, the probabilities are usually unknown.
#   probs: training probabilities, if they are known. Otherwise, pass in 0.
#   test_features: test features for each observation
#   test_rewards: test observed rewards
#   test_actions: test actions taken in the dataset
#   test_prob_type: whether the probability for each action in the training
#     data is known ('exact') or unknown and needs to be estimated ('est').
#   test_probs: test probabilities, if they are known. Otherwise, pass in 0.
#   prob_clip: the minimum probability in the dataset. Required to prevent numerical issues.
#   f: number of training folds
#   direct: whether to also include gammas which were calculated using direct random forests,
#     as well as causal random forests
#   verbose: whether to print out additional information
#
# returns:
#   list of gammas: list of matrices of estimated rewards. 
#     the list will have 5 gammas if "direct" == FALSE, or 
#     8 gammas if "direct" == TRUE (because of the 3 additional gammas which 
#     can be calculated using the means derived from direct random forest estimation).
calculate.all.Gammas = function(features, rewards, actions, k, train_prob_type, probs, 
                                test_features, test_actions, test_rewards,  
                                test_prob_type = 'exact', test_probs = 0, prob_clip = 0.001, f = 5, direct = FALSE, verbose = FALSE) {
  
  if (length(unique(actions))<k){
    stop('Not enough data to train the model.')
  }
  
  # actions are 0:(k-1), add one for R indexing
  actions = actions + 1
  test_actions = test_actions + 1
  
  if (!(train_prob_type %in% c('est', 'exact'))) {
    stop(paste('Invalid train_prob_type:', train_prob_type))
  }
  
  # calculate e_prob: probability estimates
  if (train_prob_type == 'est') {
    e_probs = est.prob.rf.with.test(features, actions, k, f, test_features, verbose = verbose)
    e_prob = e_probs$train
    t_e_prob = e_probs$test
  } else if (train_prob_type == 'exact') {
    e_prob = probs
    t_e_prob = test_probs
  }
  e_prob[e_prob < prob_clip] = prob_clip
  t_e_prob[t_e_prob < prob_clip] = prob_clip
  
  
  # calculate training/test means from estimated probabilities
  if (direct) {
    e_means_direct = est.rwd.rf.with.test(features, rewards, actions, k, e_prob, test_features, test_actions, t_e_prob, 'direct', verbose = verbose)
    e_mean_direct = e_means_direct$train
    t_e_mean_direct = e_means_direct$test
  }
  
  e_means_causal = est.rwd.rf.with.test(features, rewards, actions, k, e_prob, test_features, test_actions, t_e_prob, 'causal', verbose = verbose)
  e_mean_causal = e_means_causal$train
  t_e_mean_causal = e_means_causal$test
  
  # calculate the gamma used for evaluation using the true probabilities (TODO: Clip if needed!)
  e_prob_eval = probs
  t_e_prob_eval = test_probs
  
  if (train_prob_type == 'est' &&
      test_prob_type == 'est') {
    e_prob_eval = e_prob
    t_e_prob_eval = t_e_prob
    e_mean_eval = e_mean_causal
    t_e_mean_eval = t_e_mean_causal
  }
  else if (train_prob_type == 'exact') { # no need to recompute if we're training using exact probabilities
    e_mean_eval = e_mean_causal
    t_e_mean_eval = t_e_mean_causal
  } else {
    e_means_eval = est.rwd.rf.with.test(features, rewards, actions, k, e_prob_eval, test_features, test_actions, t_e_prob_eval, 'causal', verbose = verbose)
    e_mean_eval = e_means_eval$train
    t_e_mean_eval = e_means_eval$test
  }
  
  AIPW_eval   = est.gamma(test_rewards, test_actions, k, t_e_prob_eval, t_e_mean_eval, 'DR')
  
  
  # calculate all relevant gammas
  AIPW_causal = est.gamma(rewards, actions, k, e_prob, e_mean_causal, 'AIPW')
  IPW         = est.gamma(rewards, actions, k, e_prob, e_mean_causal, 'IPW')
  DM_causal   = est.gamma(rewards, actions, k, e_prob, e_mean_causal, 'DM')
  
  DM_causal_test = est.gamma(test_rewards, test_actions, k, t_e_prob, t_e_mean_causal, 'DM')
  if (direct) {
    AIPW_direct = est.gamma(rewards, actions, k, e_prob, e_mean_direct, 'AIPW')
    DM_direct   = est.gamma(rewards, actions, k, e_prob, e_mean_direct, 'DM')
    DM_direct_test = est.gamma(test_rewards, test_actions, k, t_e_prob, t_e_mean_direct, 'DM')
  }
  
  if (direct) {
    return(list(
      AIPW_causal = AIPW_causal,
      AIPW_direct = AIPW_direct,
      IPW = IPW,
      DM_causal = DM_causal,
      DM_direct = DM_direct,
      DM_causal_test = DM_causal_test,
      DM_direct_test = DM_direct_test,
      AIPW_eval = AIPW_eval
    ))
  } else {
    return(list(
      AIPW_causal = AIPW_causal,
      IPW = IPW,
      DM_causal = DM_causal,
      DM_causal_test = DM_causal_test,
      AIPW_eval = AIPW_eval
    ))
    
  }
}



# Given a set of features of n observations, actions taken by each observation, 
# and reward acquired by each observation, calculate
# a training and test Gamma (estimate of rewards for each observation, for each action) 
# using either doubly robust/augmented inverse propensity weighting,
# direct method/causal forest regression, or
# inverse propensity scores/inverse propensity weighting.
# Returns gammas for both training data and test data; the rewards of the test 
# data are estimated using models which were learned on the training data.
# Probability and mean reward estimates are derived using either random forest models,
# either direct random forests or causal random forests, depending on the value of 
# the 'direct'parameter.
# 
# inputs:
#   features: features for each observation
#   rewards: observed rewards
#   actions: actions taken in the dataset
#   k: number of actions
#   prob_estimation: how to estimate probabilities of each action being taken using 
#     the features of the data. "mlr" - multinomial logistic regression, "rf" - 
#     random forest, 'exact' - exact probabilities are known (as in randomized control
#     trials)
#   exact_probs: training probabilities, if they are known. Otherwise, pass in 0.
#   mu_estimation: how to estimate average rewards, per observation per action. 
#      'lasso' - lasso model, 'rf' - random forest
#   method: how to estimate gamma from the means and the probabilities.
#      'DR' - doubly robust, 'IPW' - inverse propensity weighting', 'DM' - direct method
#   forest_type: whether to use 'direct' or 'causal' random forest to estimate rewards
#   test_features: test features for each observation
#   test_rewards: test observed rewards
#   test_actions: test actions taken in the dataset
#   test_prob_type: whether the probability for each action in the training
#     data is known ('exact') or unknown and needs to be estimated ('est').
#   test_exact_probs: test probabilities, if they are known. Otherwise, pass in 0.
#   t_method: how to estimate gamma from the means and the probabilities.
#      'DR' - doubly robust, 'IPW' - inverse propensity weighting', 'DM' - direct method
#      if unspecified, will default to 'method'
#   f: number of training folds
#   prob_clip: the minimum probability in the dataset. Required to prevent numerical issues.
#
# returns:
#   list of two gammas, the first being the gamma of the training data, the second being the 
#   gamma of the test data.
calculate.Gamma.with.test = function(features, rewards, actions, k, prob_estimation = 'None', exact_probs = 0, 
                                     mu_estimation = 'None', forest_type = 'direct', method = 'DR', 
                                     test_features = 0, test_actions = 0, test_rewards = 0, test_prob_type = "None", 
                                     test_exact_probs = 0, t_method = "None", f=5, prob_clip = 0.001){
  if (length(unique(actions))<k){
    stop('Not enough data to train the model.')
  }
  actions = actions + 1
  test_actions = test_actions + 1
  
  if (t_method == "None") {
    t_method = method
  }
  
  # Estimate probability of each action
  # if (method == 'CFR' && t_method == "CFR") {
  #   e_prob = 0 # we don't need probs for CFR
  #   t_e_prob = 0
  # }
  if (test_prob_type == 'exact') {
    
    #calculate e_prob
    if (prob_estimation == 'mlr'){
      e_prob = est.prob.mlr(features, actions, k, f)
    }
    else if (prob_estimation == 'rf') {
      e_prob = est.prob.rf(features, actions, k, f)
    }
    else if (prob_estimation == 'exact') {
      if (all(exact_probs == 0)) {
        exact_probs = test_exact_probs
      }
      if (class(exact_probs) == "numeric") {
        e_prob = matrix(rep(exact_probs, nrow(features)), ncol = length(exact_probs), byrow = T)
      } else {
        e_prob = as.matrix(exact_probs)
        stopifnot(nrow(e_prob) == nrow(features))
      }
    }
    else {
      stop('Invalid probability estimation method.')
    }
    
    # calculate t_e_prob
    if (all(exact_probs == 0)) {
      exact_probs = test_exact_probs
    }
    if (class(exact_probs) == "numeric") {
      t_e_prob = matrix(rep(exact_probs, nrow(test_features)), ncol = length(exact_probs), byrow = T)
    } else {
      t_e_prob = as.matrix(test_exact_probs)
      stopifnot(nrow(t_e_prob) == nrow(test_features))
    }
  }
  
  
  # calculate e_prob and t_e_prob simultaneously
  else if (test_prob_type == 'est') {
    if (prob_estimation == 'mlr'){
      e_probs = est.prob.mlr.with.test(features, actions, k, f, test_features)
      e_prob = e_probs$train
      t_e_prob = e_probs$test
    }
    else if (prob_estimation == 'rf') {
      
      e_probs = est.prob.rf.with.test(features, actions, k, f, test_features)
      e_prob = e_probs$train
      t_e_prob = e_probs$test
    }
    else if (prob_estimation == 'exact') {
      if (class(exact_probs) == "numeric") {
        e_prob = matrix(rep(exact_probs, nrow(features)), ncol = length(exact_probs), byrow = T)
        t_e_prob = matrix(rep(exact_probs, nrow(test_features)), ncol = length(exact_probs), byrow = T)
      } else {
        e_prob = as.matrix(exact_probs)
        t_e_prob = as.matrix(test_exact_probs)
        stopifnot(nrow(e_prob) == nrow(features))
        stopifnot(nrow(t_e_prob) == nrow(test_features))
      }
    }
    else {
      stop('Invalid probability estimation method.')
    }
    t_e_prob[t_e_prob < prob_clip] = prob_clip
  }
  
  else {
    stop(paste("Invalid test probability estimation method:", test_prob_type))
  }
  
  e_prob[e_prob < prob_clip] = prob_clip
  
  #Estimate rewards
  if (method == 'IPS' && t_method == "IPS") {
    e_mean = 0 # we don't need means for IPS
  }
  else if (mu_estimation == 'lasso'){
    e_means = est.rwd.lasso.with.test(features, rewards, actions, k, f, test_features)
    e_mean = e_means$train
    t_e_mean = e_means$test
  }
  else if (mu_estimation == 'rf'){
    # e_mean = est.rwd.rf(features, rewards, actions, k, e_prob)
    e_means = est.rwd.rf.with.test(features, rewards, actions, k, e_prob, test_features, test_actions, t_e_prob, forest_type)
    e_mean = e_means$train
    t_e_mean = e_means$test
  }
  else {
    stop('Invalid reward estimation method.')
  }
  
  #Calculate Gamma
  gamma = est.gamma(rewards, actions, k, e_prob, e_mean, method)
  test_gamma = est.gamma(test_rewards, test_actions, k, t_e_prob, t_e_mean, t_method)
  
  return(list(train = gamma, test = test_gamma))  
}


# Given a set of features of n observations, actions taken by each observation, 
# and reward acquired by each observation, calculate
# a Gamma (estimate of rewards for each observation, for each action) 
# using either doubly robust/augmented inverse propensity weighting,
# direct method/causal forest regression, or
# inverse propensity scores/inverse propensity weighting.
# Probability and mean reward estimates are derived using either random forest models,
# either direct random forests or causal random forests, depending on the value of 
# the 'direct'parameter.
# 
# inputs:
#   features: features for each observation
#   rewards: observed rewards
#   actions: actions taken in the dataset
#   k: number of actions
#   prob_estimation: how to estimate probabilities of each action being taken using 
#     the features of the data. "mlr" - multinomial logistic regression, "rf" - 
#     random forest, 'exact' - exact probabilities are known (as in randomized control
#     trials)
#   exact_probs: training probabilities, if they are known. Otherwise, pass in 0.
#   mu_estimation: how to estimate average rewards, per observation per action. 
#      'lasso' - lasso model, 'rf' - random forest
#   method: how to estimate gamma from the means and the probabilities.
#      'DR' - doubly robust, 'IPW' - inverse propensity weighting', 'DM' - direct method
#   f: number of training folds
#   prob_clip: the minimum probability in the dataset. Required to prevent numerical issues.
#
# returns:
#   gamma: estimates of rewards for each observation, for each action
calculate.Gamma = function(features, rewards, actions, k, prob_estimation = 'None', exact_probs = 0, 
                           mu_estimation = 'None', method = 'DR', 
                           f=5, prob_clip = 0.001){
  if (length(unique(actions))<k){
    stop('Not enough data to train the model.')
  }
  actions = actions + 1
  
  # Estimate probability of each action
  if (prob_estimation == 'mlr'){
    e_prob = est.prob.mlr(features, actions, k, f)
  }
  else if (prob_estimation == 'rf') {
    e_prob = est.prob.rf(features, actions, k, f)
  }
  else if (prob_estimation == 'exact') {
    if (class(exact_probs) == "numeric") {
      e_prob = matrix(rep(exact_probs, nrow(features)), ncol = length(exact_probs), byrow = T)
    } else {
      e_prob = as.matrix(exact_probs)
      stopifnot(nrow(e_prob) == nrow(features))
    }
  }
  else 
  {
    stop('Invalid probability estimation method.')
  }
  
  e_prob[e_prob < prob_clip] = prob_clip
  # saveRDS(e_prob, paste('e_prob', prob_estimation, '.RDS', sep="_"))
  # e_prob = readRDS(paste('e_prob', prob_estimation, '.RDS', sep="_"))
  
  #Estimate rewards
  if (method == 'IPS') {
    e_mean = 0 # we don't need means for IPS
  }
  else if (mu_estimation == 'lasso'){
    e_mean = est.rwd.lasso(features, rewards, actions, k, f)
  }
  else if (mu_estimation == 'rf'){
    e_mean = est.rwd.rf(features, rewards, actions, k, e_prob)
  }
  else {
    stop('Invalide reward estimation method.')
  }
  
  #Calculate Gamma
  gamma = est.gamma(rewards, actions, k, e_prob, e_mean, method)
  
  return(gamma)  
}


# Given an estimate of probabilities and an estimate of means, calculate
# a gamma (estimate of rewards for each observation, for each action) using 
# either doubly robust/augmented inverse propensity weighting,
# direct method/causal forest regression, or
# inverse propensity scores/inverse propensity weighting.
# inputs:
#   rewards: observed rewards
#   actions: actions taken in the dataset
#   k: number of actions
#   e_prob: estimate of probabilities for each action to be taken
#   e_mean: estimate of the mean rewards
#   eval: evaluation method
#
# returns:
#   gamma: matrix of estimated rewards
est.gamma = function(rewards, actions, k, e_prob, e_mean, eval = 'DR'){
  n = length(actions)
  actions_aug = matrix(0, nrow = n, ncol = k)
  actions_aug[cbind(X1=1:n, X2=actions)] = 1
  if (eval == 'DR' || eval == 'AIPW'){
    Gamma = e_mean + (rewards - e_mean) / (e_prob) * actions_aug
  }
  else if (eval == 'DM' || eval == 'CFR'){
    Gamma = e_mean
  }
  else if (eval == 'IPS' || eval == 'IPW'){
    Gamma = rewards / (e_prob) * actions_aug
  }
  else {
    print('Not a valid policy evaluation estimator')
    return(0)
  }
  return(Gamma)
}

# estimate action probabilities using multinomial logistic regression
est.prob.mlr = function(features, actions, k, f){
  n = length(actions)
  features = data.frame(features)
  e_prob = matrix(NA, nrow = n, ncol = k)
  for (i in 1:f) {
    inds = (floor((n/f)*(i-1)) + 1):(floor((n/f)*i))
    train_data = data.frame(features[-inds,], actions = actions[-inds])
    prob_model = multinom(actions~., data = train_data, trace = 0)
    e_prob[inds,] = predict(object = prob_model, newdata = features[inds,], type = 'prob')
  }
  return(e_prob)
}

# estimate action probabilities using multinomial logistic regression
# for both training and test data; use model trained on training data
# for test data
est.prob.mlr.with.test = function(features, actions, k, f, t_features) {
  n = length(actions)
  features = data.frame(features)
  e_prob = matrix(NA, nrow = n, ncol = k)
  for (i in 1:f) {
    inds = (floor((n/f)*(i-1)) + 1):(floor((n/f)*i))
    train_data = data.frame(features[-inds,], actions = actions[-inds])
    prob_model = multinom(actions~., data = train_data, trace = 0)
    e_prob[inds,] = predict(object = prob_model, newdata = features[inds,], type = 'prob')
  }
  train_data = data.frame(features, actions = actions)
  prob_model = multinom(actions~., data = train_data, trace = 0)
  t_e_prob = predict(object = prob_model, newdata = t_features, type = 'prob')
  return(list(train = e_prob, test = t_e_prob))
}

# estimate action probabilities using random forest
est.prob.rf = function(features, actions, k, f){
  n = length(actions)
  features = data.frame(features)
  e_prob = matrix(NA, nrow = n, ncol = k)
  for (i in 1:k){
    train_data = data.frame(features, actions = as.factor(as.numeric(actions==i)))
    prob_model = randomForest(actions~. , data = train_data)
    e_prob[,i] = predict(prob_model, train_data , type = 'prob')[,'1']
  }
  e_prob = e_prob/(rowSums(e_prob)+1e-15) #regularization
  return(e_prob)
}

# estimate action probabilities using random forest
# for both training and test data; use model trained on training data
# for test data
est.prob.rf.with.test = function(features, actions, k, f, t_features, verbose){
  n = length(actions)
  features = data.frame(features)
  e_prob = matrix(NA, nrow = n, ncol = k)
  t_e_prob = matrix(NA, nrow = nrow(t_features), ncol = k)
  for (i in 1:k){
    if (verbose) {
      cat(paste('fitting action', i, 'prob random forest...\n'))
    }
    train_data = data.frame(features, actions = as.factor(as.numeric(actions==i)))
    prob_model = randomForest(actions~. , data = train_data)
    e_prob[,i] = predict(prob_model, train_data , type = 'prob')[,'1']
    t_e_prob[,i] = predict(prob_model, data.frame(t_features), type = 'prob')[,'1']
  }
  e_prob = e_prob/(rowSums(e_prob)+1e-15) #regularization
  t_e_prob = t_e_prob/(rowSums(t_e_prob)+1e-15) #regularization
  return(list(train = e_prob, test = t_e_prob))
}

# estimate average rewards for every action, for every observation,
# using lasso
est.rwd.lasso = function(features, rewards, actions, k, f) {
  n = length(rewards)
  features = data.frame(features)
  e_mean = matrix(NA, nrow = n, ncol = k)
  for (i in 1:f) {
    inds = (floor((n/f)*(i-1)) + 1):(floor((n/f)*i))
    features_tmp = features[-inds,]
    rewards_tmp = rewards[-inds]
    actions_tmp = actions[-inds]
    for (j in 1:k){
      mean_model = cv.glmnet(as.matrix(features_tmp[actions_tmp==j,]),as.matrix(rewards_tmp[actions_tmp==j]))
      e_mean[inds,j] = predict(mean_model, newx = as.matrix(features[inds,]), s = "lambda.min")
    }
  }
  return(e_mean)
}

# estimate average rewards for every action, for every observation,
# using lasso for both training and test data; use model trained on training data
# for test data
est.rwd.lasso.with.test = function(features, rewards, actions, k, f, t_features) {
  n = length(rewards)
  features = data.frame(features)
  e_mean = matrix(NA, nrow = n, ncol = k)
  for (i in 1:f) {
    inds = (floor((n/f)*(i-1)) + 1):(floor((n/f)*i))
    features_tmp = features[-inds,]
    rewards_tmp = rewards[-inds]
    actions_tmp = actions[-inds]
    for (j in 1:k){
      mean_model = cv.glmnet(as.matrix(features_tmp[actions_tmp==j,]),as.matrix(rewards_tmp[actions_tmp==j]))
      e_mean[inds,j] = predict(mean_model, newx = as.matrix(features[inds,]), s = "lambda.min")
    }
  }
  t_e_mean = matrix(NA, nrow = nrow(t_features), ncol = k)
  for (j in 1:k){
    mean_model = cv.glmnet(as.matrix(features[actions==j,]),as.matrix(rewards[actions==j]))
    t_e_mean[,j] = predict(mean_model, newx = as.matrix(t_features), s = "lambda.min")
  }
  return(list(train = e_mean, test = t_e_mean))
}


# estimate average rewards for every action, for every observation,
# using random forest 
est.rwd.rf = function(features, rewards, actions, k, e_prob, verbose = F){
  n = length(rewards)
  
  #Step 1 : estimate the total mean of rewards
  rewards_train = data.frame(features, rewards = rewards)
  total_mean_model = randomForest(rewards~. , data = rewards_train)
  if (verbose) {
    cat('fitting mean random forest...\n')
  }
  mean_hat = predict(total_mean_model , rewards_train)
  
  #Step 2 : estimate the "difference" between one action and the rest
  tau_mtrx = matrix(NA, nrow = n, ncol = k)
  for (i in 1:k){
    if (verbose) {
      cat(paste('fitting action', i, 'mean causal forest...\n'))
    }
    tau_model = causal_forest(X = as.matrix(features), Y = as.matrix(rewards),
                              W = as.matrix(as.numeric(actions==i)),
                              Y.hat = as.matrix(mean_hat),
                              W.hat = as.matrix(e_prob[,i]))
    tau_mtrx[,i] = predict(tau_model, features)$predictions
  }
  
  #Step 3 : calculate reward for each action
  e_mean = mean_hat + tau_mtrx * (1 - e_prob)
  
  return(e_mean)
}


# estimate average rewards for every action, for every observation,
# using random forest for both training and test data; use model trained on training data
# for test data
# can use either direct random forest or causal random forest
est.rwd.rf.with.test = function(features, rewards, actions, k, e_prob, t_features, t_actions=0, t_e_prob, 
                                forest_type, verbose = F){
  n = length(rewards)
  
  #Step 1 : estimate the total mean of rewards
  rewards_train = data.frame(features, rewards = rewards)
  if (forest_type == 'causal') {
    if (verbose) {
      cat('fitting mean random forest...\n')
    }
    total_mean_model = randomForest(rewards~. , data = rewards_train)
    mean_hat = predict(total_mean_model , rewards_train)
    t_mean_hat = predict(total_mean_model , data.frame(t_features))
  }
  
  #Step 2 : estimate the "difference" between one action and the rest
  tau_mtrx = matrix(NA, nrow = n, ncol = k)
  t_tau_mtrx = matrix(NA, nrow = nrow(t_features), ncol = k)
  for (i in 1:k){
    if (forest_type == 'causal') {
      if (verbose) {
        cat(paste('fitting action', i, 'mean causal forest...\n'))
      }
      tau_model = causal_forest(X = as.matrix(features), Y = as.matrix(rewards),
                                W = as.matrix(as.numeric(actions==i)),
                                Y.hat = as.matrix(mean_hat),
                                W.hat = as.matrix(e_prob[,i]))
      tau_mtrx[,i] = predict(tau_model, features)$predictions
      t_tau_mtrx[,i] = predict(tau_model, t_features)$predictions
    }
    else if (forest_type == 'direct') {
      if (verbose) {
        cat(paste('fitting action', i, 'random forest...\n'))
      }
      train_data = data.frame(features[actions == i, ], rwds = rewards[actions == i])
      tau_model = randomForest(rwds ~ ., data = train_data)
      tau_mtrx[,i] = predict(tau_model, data.frame(features))
      t_tau_mtrx[,i] = predict(tau_model, data.frame(t_features))
    }
  }
  
  #Step 3 : calculate reward for each action
  
  if (forest_type == 'causal') {
    e_mean = mean_hat + tau_mtrx * (1 - e_prob)
    t_e_mean = t_mean_hat + t_tau_mtrx * (1 - t_e_prob)
  }
  else if (forest_type == 'direct') {
    e_mean = tau_mtrx
    t_e_mean = t_tau_mtrx
  }
  else stop('Forest type not implemented')
  
  if (verbose) {
    cat('done\n')
  }
  return(list(train = e_mean, test=t_e_mean))
}

##### Random Forest direct regression method
regression.rf.with.test = function(features, actions, rewards, k, t_features){
  actions = actions + 1
  n = length(actions)
  n_train = nrow(features)
  n_test = nrow(t_features)
  features = data.frame(features)
  est_rwds_train = matrix(NA, nrow = n_train, ncol = k)
  est_rwds = matrix(NA, nrow = n_test, ncol = k)
  for (i in 1:k){
    train_data = data.frame(features[actions == i, ], rwds = rewards[actions == i])
    prob_model = randomForest(rwds~. , data = train_data)
    est_rwds_train[,i] = predict(prob_model, features)
    est_rwds[,i] = predict(prob_model, t_features)
  }
  return(list(train = est_rwds_train, test = est_rwds))
}
