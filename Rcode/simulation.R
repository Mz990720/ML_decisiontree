# Main code used to generate simulated data and run policy learning.

# modify the path to the source directory; alternatively, uncomment the lower
# line if using RStudio
# setwd('path/to/source/directory') è¿™é‡Œè¿™ä¸ªåœ°å€çœŸçš„æ²¡é—®é¢˜å�?
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\install_packages.R")
source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\estimator_functions.R")
source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\generate_data.R")
source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\evaluation.R")
source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\tree_visualization.R")
sourceCpp('C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\Rport.cpp')
options(scipen = 100)

prob_method <- 'mlr'  # 'mlr':multi-class logistic regression; 'rf':random forest;  æ­¤ç¬¦å·ä¸ºèµ‹å€? 
rwd_method <- 'rf'  # 'lasso':lasso; 'rf': generalized random forest;
level_of_tree <- 2  # decide the level of learnt tree
jump_step <- 1  # step length  

n_best <- 10000 # number of observations to calculate best tree
n_test <- 15000 # number of observations to test learned trees
n_array = c(2000) # array of number of observations to learn trees on  
p <- 10 # number of features
k <- 3 # number of actions

out_filename = 'simulation_output2'

# generate best tree
print('Generate data')
features_best = generate.features(n_best, p)
rewardmat_best = get.ellipse.rewards.mtrx(features_best)  
tree_best = learn(features_best, rewardmat_best, level_of_tree, jump_step)  
#tree_best_greedy = learn_greedy(features_best, rewardmat_best, level_of_tree, jump_step)   #也可以直接在这里比较得了？？�?

# generate test dataset
features_test = generate.features(n_test, p)
rewardmat_test = get.ellipse.rewards.mtrx(features_test)   

for (n in n_array) {
  print(paste('Number of data points:', n))
  # repeat n_trials times for avg results
  n_trials = 1
  method_names = c('CAIPWL_opt', 'CAIPWL_skip', 'CAIPWL_greedy', 'IPWL_opt', 'IPWL_skip', 'IPWL_greedy', 
                   'CAIPWL_direct', 'DM_direct', 'DM_causal', 'random')
  results = matrix(0, nrow = n_trials, ncol = length(method_names))
  colnames(results) = method_names
  
  
  for (i in 1:n_trials) {
    # generate ellipse data (basically Main.R on the ellipse data)
    print(paste('trial', i))
    
    features = generate.features(n, p)
    actions = generate.ellipse.actions(features)
    rewards = generate.ellipse.rewards(features, actions, noise_level = 2)
    
    # perform estimation (IPS/DR)
    dr1 = calculate.Gamma(features, rewards, actions, k, prob_method, mu_estimation = rwd_method, method = 'DR')
    
    Gammas = calculate.all.Gammas(features, rewards, actions, probs = 0,
                                  test_features = features[1:2, ], test_actions=actions[1:2], test_rewards=rewards[1:2],
                                  0, k = k, train_prob_type = 'est', test_prob_type = 'est', prob_clip = 0.001, direct = TRUE)
    rewardmat_DR1 = Gammas$AIPW_causal
    rewardmat_IPS1 = Gammas$IPW
    rewardmat_DR_direct = Gammas$AIPW_direct
    rewardmat_DM_direct = Gammas$DM_direct
    rewardmat_DM_causal = Gammas$DM_causal
    
    rewardmat_DR = calculate.Gamma(features, rewards, actions, k, prob_method, mu_estimation = rwd_method, method = 'DR')
    rewardmat_IPS = calculate.Gamma(features, rewards, actions, k, prob_method, mu_estimation = rwd_method, method = 'IPS')
    
    # learn
    tree_DR = learn(features, rewardmat_DR, level_of_tree, jump_step)
    tree_IPS = learn(features, rewardmat_IPS, level_of_tree, jump_step)
    tree_DR_direct = learn(features, rewardmat_DR_direct, level_of_tree, jump_step)
    tree_DM_direct = learn(features, rewardmat_DM_direct, level_of_tree, jump_step)
    tree_DM_causal = learn(features, rewardmat_DM_causal, level_of_tree, jump_step)
    
    tree_skip_DR = learn(features, rewardmat_DR, level_of_tree, 10)
    tree_skip_IPS = learn(features, rewardmat_IPS, level_of_tree, 10)
  
    #tree_greedy_DR = learn_greedy(features, rewardmat_DR, level_of_tree, jump_step)
    #tree_greedy_IPS = learn_greedy(features, rewardmat_IPS, level_of_tree, jump_step)
    
    # evaluation (learnt tree vs best tree)
    results[i, 1] = compute.regret(tree_DR, tree_best, features_test, rewardmat_test)
    results[i, 2] = compute.regret(tree_skip_DR, tree_best, features_test, rewardmat_test)
    #results[i, 3] = compute.regret(tree_greedy_DR, tree_best, features_test, rewardmat_test)
    results[i, 4] = compute.regret(tree_IPS, tree_best, features_test, rewardmat_test)
    results[i, 5] = compute.regret(tree_skip_IPS, tree_best, features_test, rewardmat_test)
    #results[i, 6] = compute.regret(tree_greedy_IPS, tree_best, features_test, rewardmat_test)
    results[i, 7] = compute.regret(tree_DR_direct, tree_best, features_test, rewardmat_test)
    results[i, 8] = compute.regret(tree_DM_direct, tree_best, features_test, rewardmat_test)
    results[i, 9] = compute.regret(tree_DM_causal, tree_best, features_test, rewardmat_test)
  
    print(results[i, ])
    # visualization
    # visualize(tree_DR, TRUE, paste(i, "DR", "learnt.png", sep = '_'))
    visualize(tree_DR, as.character(1:p), as.character(1:k), paste('simulation_tree', i, "DR.png",  sep = '_'), TRUE)
  }
  results[, 10] = compute.random.regret(k, tree_best, features_test, rewardmat_test)
  print('Final results:')
  print(apply(results, 2, mean))
  write.csv(results, paste(out_filename, n, '.csv', sep="_"), row.names = F)
}

