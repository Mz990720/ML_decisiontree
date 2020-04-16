# Main code used to generate simulated data and run policy learning.

# modify the path to the source directory; alternatively, uncomment the lower
# line if using RStudio
# setwd('path/to/source/directory') 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\install_packages.R")
source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\estimator_functions.R")
source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\generate_data.R")
sourceCpp('C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\Rport_1.cpp')
source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\evaluation.R")
source("C:\\Users\\zhangbokang\\Desktop\\2019summer\\code\\tree_visualization.R")
options(scipen = 100)

prob_method <- 'mlr'  # 'mlr':multi-class logistic regression; 'rf':random forest;
rwd_method <- 'rf'  # 'lasso':lasso; 'rf': generalized random forest;
level_of_tree <- 2  # decide the level of learnt tree
jump_step <- 1  # step length

n_best <- 100 # number of observations to calculate best tree
#n_test <- 15000 # number of observations to test learned trees
n_array = c(2000) # array of number of observations to learn trees on
p <- 10 # number of features
k <- 3 # number of actions
#list1 <- list(c(2,5,3),21.3,sin)
jumps <- list(2,5,10) # a list to store the jump stemps 
out_filename = 'simulation_output4'

# generate best tree
#features_best = generate.features(n_best, p)
#rewardmat_best = get.ellipse.rewards.mtrx(features_best)
#tree_best_greedy = learn_greedy(features_best, rewardmat_best, level_of_tree, jump_step)

# generate test dataset
#features_test = generate.features(n_test, p)
#rewardmat_test = get.ellipse.rewards.mtrx(features_test)
#for(j in jumps) 
#{
  #print(j)
  tree_greedy = tree_rwd_sum = tree_opt_rwd_sum = 0
  # repeat n_trials times for avg results
  n_trials = 1  # run n_trials to average
  method_names = c('CAIPWL_opt', 'CAIPWL_skip', 'CAIPWL_greedy', 'IPWL_opt', 'IPWL_skip', 'IPWL_greedy', 
                   'CAIPWL_direct', 'DM_direct', 'DM_causal', 'random')
  results = matrix(0, nrow = 1, ncol = length(method_names)) #ÃÃˆÃ–Â»Â·Ã…Ã’Â»ÃÃ�?
  colnames(results) = method_names
  
    #for(i in 1:n_trials){
    #generate ellipse data (basically Main.R on the ellipse data)
      
    features = generate.features(n_best, p)

    actions = generate.ellipse.actions(features)
    rewards = generate.ellipse.rewards(features, actions, noise_level = 2) #Ã¦â€”Â¢Ã§â€Â¨Ã¨Â¿â„¢Ã¤Â¸ÂªÃ¥Â­Â¦Ã¤Â¹Â�? Ã¤Â¹Å¸Ã§â€Â¨Ã¨Â¿â„¢Ã¤Â¸ÂªÃ¦Âµâ€¹Ã¨Â¯â�?? 
    rewardmat = get.ellipse.rewards.mtrx(features)
    #tree_best = learn(features, rewardmat, level_of_tree, jump_step)
    # perform estimation (IPS/DR)
    dr1 = calculate.Gamma(features, rewards, actions, k, prob_method, mu_estimation = rwd_method, method = 'DR')
    
    Gammas = calculate.all.Gammas(features, rewards, actions, probs = 0,
                                  test_features = features[1:2, ], test_actions=actions[1:2], test_rewards=rewards[1:2],
                                  0, k = k, train_prob_type = 'est', test_prob_type = 'est', prob_clip = 0.001, direct = TRUE)
    rewardmat_DR = Gammas$AIPW_causal
    #rewardmat_IPS = Gammas$IPW
    #rewardmat_DR_direct = Gammas$AIPW_direct
    #rewardmat_DM_direct = Gammas$DM_direct
    #rewardmat_DM_causal = Gammas$DM_causal
    
    rewardmat_DR = calculate.Gamma(features, rewards, actions, k, prob_method, mu_estimation = rwd_method, method = 'DR')
    #rewardmat_IPS = calculate.Gamma(features, rewards, actions, k, prob_method, mu_estimation = rwd_method, method = 'IPS')
    
    # learn
    #tree_DR = learn(features, rewardmat_DR, level_of_tree, jump_step)
    #tree_IPS = learn(features, rewardmat_IPS, level_of_tree, jump_step)
    #tree_DR_direct = learn(features, rewardmat_DR_direct, level_of_tree, jump_step)
    #tree_DM_direct = learn(features, rewardmat_DM_direct, level_of_tree, jump_step)
    #tree_DM_causal = learn(features, rewardmat_DM_causal, level_of_tree, jump_step)
    
    #tree_skip_DR = learn(features, rewardmat_DR, level_of_tree, j)
    #tree_skip_IPS = learn(features, rewardmat_IPS, level_of_tree, 10)
  
    #tree_greedy_DR = learn_greedy(features, rewardmat_DR, level_of_tree, jump_step)
    #tree_greedy_IPS = learn_greedy(features, rewardmat_IPS, level_of_tree, jump_step)
    
    # evaluation (learnt tree vs best tree)
    tree_greedy = learn_greedy(features, rewardmat_DR, level_of_tree, jump_step)
    print(tree_greedy)
    #tree_rwd = learn(features, rewardmat_DR, level_of_tree, j)
    #tree_opt_rwd = learn(features, rewardmat_DR, level_of_tree, jump_step)
    #print(tree_rwd)
    #print(tree_opt_rwd)
    #tree_rwd_sum = tree_rwd_sum + tree_rwd
    #tree_opt_rwd_sum = tree_opt_rwd_sum + tree_opt_rwd
    #}
    #tree_rwd_sum = tree_rwd_sum /3
    #tree_opt_rwd_sum = tree_opt_rwd_sum/3
    #y[flag] = tree_rwd_sum / tree_opt_rwd_sum
    #x[flag] = j/n_best 
    #flag = flag + 1 
#}

#plot(x[1:3],y[1:3],main="test",type="l",xlab="jump_step/n",ylab="R2/R1")

#画个�?
    #results[1, 1] = newcompute.regret(tree_DR, tree_best, features, rewards)
    #results[1, 2] = newcompute.regret(tree_skip_DR, tree_best, features, rewards)
    #results[1, 3] = compute.regret(tree_greedy_DR, tree_best, features, rewards)
    #results[1, 4] = newcompute.regret(tree_IPS, tree_best, features, rewards)
    #results[1, 5] = newcompute.regret(tree_skip_IPS, tree_best, features, rewards)
    #results[1, 6] = compute.regret(tree_greedy_IPS, tree_best, features, rewards)
    #results[1, 7] = compute.regret(tree_DR_direct, tree_best, features, rewards)
    #results[1, 8] = compute.regret(tree_DM_direct, tree_best, features, rewards)
    #results[1, 9] = compute.regret(tree_DM_causal, tree_best, features, rewards)
  
    #print(results[1, ])
    # visualization
    # visualize(tree_DR, TRUE, paste(i, "DR", "learnt.png", sep = '_'))
    #visualize(tree_DR, as.character(1:p), as.character(1:k), paste('simulation_tree', 1, "DR.png",  sep = '_'), TRUE)
  
  #results[, 10] = compute.random.regret(k, tree_best, features, rewards)
  #print('Final results:')
  #print(apply(results, 2, mean))
  #write.csv(results, paste(out_filename, 1, '.csv', sep="_"), row.names = F)

