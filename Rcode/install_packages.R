# install all of the R packages needed to run code pipeline

pkgLoad <- function( packages = "pipeline" ) {
  
  if( length( packages ) == 1L && packages == "pipeline" ) {
    packages <- c( 
      "data.tree","DiagrammeR",
      "grf", "nnet", 
      "randomForest", "Rcpp" 
    )
  }
  
  packagecheck <- match( packages, utils::installed.packages()[,1] )
  
  packagestoinstall <- packages[ is.na( packagecheck ) ]
  
  if( length( packagestoinstall ) > 0L ) {
    utils::install.packages( packagestoinstall
    )
  } else {
    # print( "All requested packages already installed" )
  }
  
  for( package in packages ) {
    suppressPackageStartupMessages(
      library( package, character.only = TRUE, quietly = TRUE )
    )
  }
  
}

pkgLoad()
