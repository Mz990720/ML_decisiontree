library(data.tree)
library(DiagrammeR)
library(rsvg)
library(DiagrammeRsvg)



# visualize a tree and save it to a pdf file
#
# inputs: 
#   data: tree array, as returned by "learn"
#   feat_names: array of feature names
#   action_names: array of action names
#   filename: full path specifying where to save the figure. Must end in '.pdf'
#   isPruned: whether the tree is already pruned
#   
visualize = function(data, feat_names, action_names, filename, isPruned = TRUE) {
  if (isPruned) {
    visualize_basic(compress(data), feat_names, action_names, filename)
  } else {
    visualize_basic(data, feat_names, action_names, filename)
  }
}

# prune a tree so it is visualized cleanly
compress = function(data) {
  matrix <- data
  #matrix <- matrix[order(matrix[,1],decreasing=F),]
  totalNumber <- nrow(matrix)
  leavesNumber <- 2^as.integer(log2(totalNumber))

  for (a in (totalNumber-leavesNumber):min((totalNumber-leavesNumber),2)) {
    if ((matrix[2*a,3] == matrix[2*a+1,3])) {
      matrix[a,2] <- -1
      #matrix[a,3] <- matrix[[2*a,3]]
      matrix[a,3] <- matrix[2*a,3]
    }
  }
  matrix
}

# helper function to visualize a tree
visualize_basic = function(data, feat_names, action_names, filename) {
  matrix <- data
  matrix <- matrix[order(matrix[,1],decreasing=F),]
  totalNumber <- nrow(matrix)
  leavesNumber <- 2^as.integer(log2(totalNumber))

  list <- list(Node$new(paste0(feat_names[matrix[1,2]+1]," < ",round(matrix[1,3],4))))
  
  #assign each n to a list

  for (j in 2:totalNumber) {
  if (matrix[j,2] != -1) {
    if (j%%2 == 0){
      list[[j]] <- Node$new(paste0(feat_names[matrix[j,2]+1]," < ",round(matrix[j,3],4)))
    # } else if (j == 3){
    #   list[[j]] <- Node$new(paste0(feat_names[matrix[j,2]+1]," = ",round(matrix[j,3],4),' '))
    } else {
      list[[j]] <- Node$new(paste0(feat_names[matrix[j,2]+1]," < ",round(matrix[j,3],4),' '))
    }
  } else {
    list[[j]] <- Node$new(action_names[matrix[j,3]+1])
  }
}

#  for (j in 2:totalNumber) {
#    if (matrix[j,2] != -1) {
#      list[[j]] <- Node$new(paste0(j,":","X",matrix[j,2]," < ",round(matrix[j,3],4)))
#    } else {
#      list[[j]] <- Node$new(matrix[j,3])
#    }
#  }
  
  #build hierarchy
  for (a in (totalNumber-leavesNumber):1) {
    if (matrix[a,2] != -1) {
      list[[a]]$AddChildNode(list[[2*a]])
      list[[a]]$AddChildNode(list[[2*a+1]])
    }
  }
  
  #Styling
  SetNodeStyle(list[[1]], style = "filled,rounded", shape = "box", fillcolor = "LightBlue", 
               fontcolor = "Black", fontname = "helvetica", tooltip = GetDefaultTooltip)
  Do(list[[1]]$leaves, function(node) SetNodeStyle(node, shape = "egg", fillcolor = "Thistle"))
  
  #save as pdf
  export_graph(ToDiagrammeRGraph(list[[1]]), filename)
}


