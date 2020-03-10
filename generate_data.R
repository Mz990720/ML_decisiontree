f1 = 6
f2 = 8

# pa1_r12 = .15
# pa3_r12 = .15
pa1_r12 = 0.2
pa3_r12 = 0.2

pa1_r3 = 0.4
pa3_r3 = 0.4



generate.data = function(n, p, k, method = 'rdmtree'){
  if (method == 'rdmtree'){
    features=generate.features(n, p)
    tree_origin=generate.tree(p,k,l = 2)
    actions=generate.actions(n,k)
    rewards <- generate.rewards(features, tree_origin, actions, k)
  }
  else if(method == 'fxtree'){
    features=generate.features(n, p)
    tree_origin=read.table('tree_origin.txt')
    actions=generate.actions(n,k)
    rewards <- generate.rewards(features,tree_origin, actions, k)
  }
  else if(method == 'square'){
    features=generate.features(n, p)
    actions=generate.square.actions(features)
    rewards=generate.square.rewards(features, actions)
    tree_origin = data.frame(node_id = 1:7, i=c(0,1,1,-1,-1,-1,-1),b=c(0.5,0.5,0.5,2,0,1,2))
  }
  else{
    stop('Invalid generation method.')
  }
  training_data <- data.frame(features,actions,rewards)
  write.table(training_data,file = 'training_data.txt',sep = ' ',col.names = FALSE,row.names = FALSE)  #这里没有输出 
  write.table(tree_origin,file = 'tree_origin.txt',sep = ' ',col.names = FALSE,row.names = FALSE)
}

## Generate features
generate.features<-function(n, p){
    tmp=runif(n*p)
    return(matrix(data = tmp,nrow = n,ncol = p))
}

## Generate uniformly random actions
generate.actions<-function(n, k){
    return(sample(0:(k-1),n,replace = TRUE))
}

## Generate tree-based rewards
generate.rewards<-function(features, tree, actions, num_actions, noise_level = 0.1){
    correct <- classify.via.tree(features, tree)
    return(1-abs(correct - actions)/num_actions + rnorm(length(actions))*noise_level**0.5)
}

# get a matrix of the number of correct actions
get.rewards.mtrx<-function(features, tree, num_actions){
    correct <- classify.via.tree(features, tree)
    n = nrow(features)
    rewards = matrix(NA, nrow = n, ncol = num_actions)
    for (i in 1:n){
        for (j in 1:num_actions){
            rewards[i,j] = 1-abs(correct[i] - j)/num_actions
        }
    }
    return(rewards)
}

## Generate square actions
generate.square.actions <- function(features){
  n = nrow(features)
  best_actions = get.square.best.actions(features)
  tmp = runif(n)
  actions = rep(NA, n)
  for (i in 1:n){
    if (best_actions[i]==3){
      if (tmp[i]<=0.4) actions[i] = 1
      else if (tmp[i]>0.6) actions[i] = 3
      else actions[i] = 2
    }
    else{
      if (tmp[i]<=0.15) actions[i] = 1
      else if(tmp[i]>0.85) actions[i] = 3
      else actions[i] = 2
    }
  }
  return(actions-1)
}

# get the best actions for the square test case
get.square.best.actions <- function(features){
  n = nrow(features)
  actions = rep(NA, n)
  for (i in 1:n){
    if ((features[i,x1]**2+features[i,x2]**2 <= 0.25) || ((1-features[i,x1])**2+(1-features[i,x2])**2<=0.25)){
      actions[i] = 3
    }
    else if (features[i,x1]<=0.5 && features[i,x2]>=0.5){
      actions[i] = 1
    }
    else{
      actions[i] = 2
    }
  }
  return(actions)
}

## Generate rewards from actions in the square test case, with added noise
generate.square.rewards <- function(features, actions, noise_level = 1){
  n = length(actions)
  actions = actions + 1
  rewards = rep(NA,n)
  for (i in 1:n){
    if ((features[i,x1]**2+features[i,x2]**2 <= 0.25) || ((1-features[i,x1])**2+(1-features[i,x2])**2<=0.25)){
      rewards[i] = actions[i] - 2 + rnorm(1)*noise_level**0.5
    }
    else if (features[i,x1]<=0.5 && features[i,x2]>=0.5){
      rewards[i] = 4 - actions[i] + rnorm(1)*noise_level**0.5
    }
    else{
      rewards[i] = 2 - abs(actions[i]-2)/4 + rnorm(1)*noise_level**0.5
    }
  }
  return(rewards)
} 

## Generate rewards in the square test case
get.square.rewards.mtrx <- function(features){
  n = nrow(features)
  rewards = matrix(NA, nrow = n, ncol = 3)
  for (i in 1:n){
    if ((features[i,x1]**2+features[i,x2]**2 <= 0.25) || ((1-features[i,x1])**2+(1-features[i,x2])**2<=0.25)){
      for (j in 1:3) rewards[i,j] = j - 2
    }
    else if (features[i,x1]<=0.5 && features[i,x2]>=0.5){
      for (j in 1:3) rewards[i,j] = 4 - j
    }
    else{
      for (j in 1:3) rewards[i,j] = 2 - abs(j-2)/4
    }
  }
  return(rewards)
} 


## Generate ellipse test case actions
generate.ellipse.actions <- function(features){
  n = nrow(features)
  best_actions = get.ellipse.best.actions(features)
  tmp = runif(n)
  actions = rep(NA, n)
  for (i in 1:n){
    if (best_actions[i]==3){
      if (tmp[i]<= pa1_r3) actions[i] = 1
      else if (tmp[i] > (1 - pa3_r3)) actions[i] = 3
      else actions[i] = 2
    }
    else{
      if (tmp[i] <= pa1_r12) actions[i] = 1
      else if(tmp[i] > (1 - pa3_r12)) actions[i] = 3
      else actions[i] = 2
    }
  }
  return(actions-1)
}

## Generate best ellipse test case actions based on features
get.ellipse.best.actions <- function(features){
  n = nrow(features)
  actions = rep(NA, n)
  for (i in 1:n){
    if (((features[i,f1]/0.6)**2 + (features[i,f2]/0.35)**2 <= 1) || (((1-features[i,f1])/0.4)**2 + ((1-features[i,f2])/0.35)**2 <= 1)){
      actions[i] = 3
    }
    else if (features[i,f1] <= 0.6 && features[i,f2] >= 0.35){
      actions[i] = 1
    }
    else{
      actions[i] = 2
    }
  }
  return(actions)
}

## Generate rewards in ellipse based on features and actions, with noise added
generate.ellipse.rewards <- function(features, actions, noise_level = 1){
  n = length(actions)
  actions = actions + 1
  rewards = rep(NA,n)
  for (i in 1:n){
    if (((features[i,f1]/0.6)**2 + (features[i,f2]/0.35)**2 <= 1) || (((1-features[i,f1])/0.4)**2 + ((1-features[i,f2])/0.35)**2 <= 1)){
      rewards[i] = 1.5*(actions[i] - 2) + rnorm(1)*noise_level
    }
    else if (features[i,f1] <= 0.6 && features[i,f2] >= 0.35){
      rewards[i] = 4 - actions[i] + rnorm(1)*noise_level
    }
    else{
      rewards[i] = 2 - abs(actions[i]-2)/2 + rnorm(1)*noise_level
    }
  }
  return(rewards)
} 

## Generate rewards in ellipse test case
get.ellipse.rewards.mtrx <- function(features){
  n = nrow(features)
  rewards = matrix(NA, nrow = n, ncol = 3)
  for (i in 1:n){
    if (((features[i,f1]/0.6)**2 + (features[i,f2]/0.35)**2 <= 1) || (((1-features[i,f1])/0.4)**2 + ((1-features[i,f2])/0.35)**2 <= 1)){
      for (j in 1:3) rewards[i,j] = 1.5*(j - 2)
    }
    else if (features[i,f1] <= 0.6 && features[i,f2] >= 0.35){
      for (j in 1:3) rewards[i,j] = 4 - j
    }
    else{
      for (j in 1:3) rewards[i,j] = 2 - abs(j-2)/2
    }
  }
  return(rewards)
} 

# Other utilities
generate.tree<-function(p, k, l){
  feature_ind <- sample(x = 0:(p-1),size = 2**l-1,replace = TRUE)
  threshold <- runif(2**l-1)
  leaf_labels<-sample(0:(k-1),size = 2**l,replace = TRUE)
  tree <- cbind(1:(2**(l+1)-1),c(feature_ind,rep(-1,2**l)),c(threshold,leaf_labels))
  return(data.frame(tree))
}

# visualize the actions plotted against features (for square test case)
plot2Ddata = function(features, actions){
  data.tmp = data.frame(x1 = features[,1],x2=features[,2],cl=as.factor(actions))
  png('special_data.png')
  ggplot(data.tmp,aes(x=x1,y=x2,colour=cl))+geom_point()
  dev.off()
}
