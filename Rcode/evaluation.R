
# Given a learned tree and a set of features, use the tree policy
# to choose an action for each observation (row of feature matrix)
# 
# inputs:
#   features: features for each observation
#   tree: learned tree (as from the function "learn")
#
# returns:
#   vector of actions, one for each row of 'feature' matrix   for each sample there is a action 
classify.via.tree<-function(features, tree){
  tree_level = as.integer(log2(nrow(tree)+1))-1
  num = nrow(features)
  tree <- tree[order(tree[1]),]
  ret <- apply(X = features,MARGIN = 1,FUN = function(row){
    nid = 1
    while (nid<2**tree_level)  # the max is tree_level - 1
    {
      if(row[tree[nid,2]+1]<tree[nid,3]){
        # the index for feats plus one, because the index starts at 1 while feat index starts at 0.
        nid=nid*2
      }else{
        nid=nid*2+1
      }
    }
    if(tree[nid,2]!=-1) stop('tree node -> i shold be -1 at leaf nodes.')
    return(tree[nid,3])
  })
  return(ret)
}

# given a matrix of actions representing several different policies 
# and a Gamma reward matrix, compute the means, SDs, differences, and p-values
# of each policy compared with all other policies.
#
# inputs:
#   actions: matrix of actions where each column represents a different policy
#   gamma: a gamma matrix (estimate of rewards for each action) 
#
# returns:
#   list of 
#     1. mean reward/SD of reward for each action
#     2. difference matrix: difference of the average reward of policy i and j
#     3. p value matrix: whether policy i achieved a statistically significantly
#        higher reward than policy j
#     4. SD matrix: SD of the differences of rewards between policies i and j
get.results.from.actions = function(actions, gamma) {
  n = nrow(actions)
  nc = ncol(actions)
  rw_per_act = actions
  for (i in 1:nc) {
    rw_per_act[,i] = gamma[cbind(1:n, actions[, i])]
  }
  
  r_means = apply(rw_per_act, 2, mean)
  r_sds = apply(rw_per_act, 2, sd)
  
  r_pval = matrix(0, nrow = nc, ncol = nc, dimnames = list(colnames(actions), colnames(actions)))
  r_diff = r_pval
  r_sdmat = r_pval
  
  for (i in 1:nc) {
    for (j in 1:nc) {
      if (i != j) {
        z_i = rw_per_act[, i] - rw_per_act[, j]
        r_diff[j, i] = mean(z_i)
        r_pval[j, i] = t.test(z_i, mu=0)$p.value
        r_sdmat[j, i] = sd(z_i)
      }
    }
  }
  return(list(means_sd = data.frame(means = r_means, sds = r_sds),
              # sds = data.frame(sds = r_sds),
              diffs = r_diff, 
              p_vals = r_pval,
              sdmat = r_sdmat))
}


# helper function to calculate the total reward given a tree policy
# inputs:
#   tree: tree policy learned from data
#   features: feature matrix
#   rewards: reward matrix
#
# returns:
#   (scalar) total rewards when using the tree policy
get.sumrwd = function(tree, features, rewards){
  w = classify.via.tree(features, tree)+1
  n = nrow(features)
  sink("D:\\My.txt",append=TRUE,split=TRUE)
  #print(w) # should be different
  #print(n)
  sink()
  return(sum(rewards[cbind(1:n,w)]))
}


# helper function to calculate difference in rewards given a tree and action vector
# inputs:
#   tree: decision tree for a policy
#   actions2: actions to compare the tree policy to
#   features: feature matrix
#   rewards: reward matrix
#
# returns:
#   (scalar) difference between rewards based on tree and rewards based on actions2
get.diff.rwds = function(tree1, actions2, features, rewards){
  w1 = classify.via.tree(features, tree1)+1
  w2 = actions2#classify.via.tree(features, tree2)+1
  n = nrow(features)
  return(rewards[cbind(1:n,w1)] - rewards[cbind(1:n,w2)])
}


# helper function to calculate difference in rewards given two different action vectors
# inputs:
#   actions1, actions2: vectors of actions - must not be zero indexed!
#   features: feature matrix
#   rewards: reward matrix
#
# returns:
#   (scalar) difference between rewards based on actions1 and rewards based on actions2
get.diff.rwds.from.actions = function(actions1, actions2, features, rewards){
  w1 = actions1#classify.via.tree(features, tree1)+1
  w2 = actions2#classify.via.tree(features, tree2)+1
  n = nrow(features)
  return(rewards[cbind(1:n,w1)] - rewards[cbind(1:n,w2)])
}


evaluation.gamma = function(features, est_rewards){
  real_rewards = get.square.rewards.mtrx(features)
  return(sum((real_rewards - est_rewards)**2))
}



# given an optimal tree and reward matrix and a different tree, calculate the regret
# of the tree's policy
# inputs:
#   tree: decision tree to compute regret for
#   tree_opt: optimal decision tree for the problem
#   features: features for each observation
#   true_rewardmat: the true rewards (or an accurate estimate of) for each action,
#       for each data point
#
# returns:
#   (scalar) regret of "tree" policy
compute.regret = function(tree, tree_opt, features, true_rewardmat) {

  tree_rwd = get.sumrwd(tree, features, true_rewardmat)
  tree_opt_rwd = get.sumrwd(tree_opt, features, true_rewardmat)
  n = nrow(features)
  return((tree_opt_rwd - tree_rwd)/n)
}


newcompute.regret = function(tree, tree_opt, features, true_rewardmat)
{
  tree_rwd = get.sumrwd(tree, features, true_rewardmat)
  tree_opt_rwd = get.sumrwd(tree_opt, features, true_rewardmat)
  n = nrow(features)
  return((tree_opt_rwd - tree_rwd)/n)
}


# given an optimal tree and reward matrix, calculate the regret
# of a random policy 
# inputs:
#   k: number of actions
#   tree_opt: optimal decision tree for the problem
#   features: features for each observation
#   true_rewardmat: the true rewards (or an accurate estimate of) for each action,
#       for each data point
#
# returns:
#   (scalar) regret of random policy

compute.random.regret = function(k, tree_opt, features, true_rewardmat) {
  tree_opt_rwd = get.sumrwd(tree_opt, features, true_rewardmat)
  
  n = nrow(features)
  w = sample(k,n, replace = T)
  rand_rwd = sum(true_rewardmat[cbind(1:n,w)])
  
  return((tree_opt_rwd - rand_rwd)/n)
}
