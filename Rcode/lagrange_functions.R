subtract.cost.from.gamma = function(gamma, costs) {
  
  if (ncol(gamma) != length(costs)) {
    stop(paste('improper number of costs:', length(costs)))
  }
  if (any(costs < 0)) {
    stop('Some costs are negative')
  }
  
  cost_mat = matrix(rep(costs, nrow(gamma)), nrow = nrow(gamma), byrow = T)
  return(gamma - cost_mat)
}