#include <Rcpp.h>
#include "policy.h"
using namespace std;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::DataFrame learn(Rcpp::NumericMatrix feats_R, Rcpp::NumericMatrix reward_R, int layer, int step) {

	//read data & init
	nSample = feats_R.rows();
	nFeat = feats_R.cols();
	nClass = reward_R.cols();
	level = layer;
	skip_node = step;
	//allocate();
	init();

	for (int i = 0; i < nSample; i++)
		for (int j = 0; j < nFeat; j++)
			set_feats(j,i,feats_R(i, j));
	for (int i = 0; i < nSample; i++)
		for (int j = 0; j < nClass; j++)
			set_rewards(i,j, reward_R(i, j));

	
	sort_feat();
	

	// learn the tree.
	learn_from_data(1, 1);
	// Initialize Rcpp containers and put the tree into the vectors.
	int nNode = (1 << (level + 1)) - 1;
	Rcpp::NumericVector v1(nNode);
	Rcpp::NumericVector v2(nNode);
	Rcpp::NumericVector v3(nNode);
	for(int i=0; i< nNode;i++)
	{
		v1[i] = i+1;
		v2[i] = i+1>=(1<<level)?-1:get_tree_feat(i+1);
		v3[i] = get_tree_thre(i+1);
	}

    free_memory();
	// return a dataframe object.
	return Rcpp::DataFrame::create(Rcpp::Named("node_id") = v1, Rcpp::Named("i") = v2, Rcpp::Named("b") = v3);
}


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::DataFrame learn_greedy(Rcpp::NumericMatrix feats_R, Rcpp::NumericMatrix reward_R, int layer, int step)
{

	nSample = feats_R.rows();
	nFeat = feats_R.cols();
	nClass = reward_R.cols();
	level = layer;
	skip_node = step;
	int nNode = (1 << (level + 1)) - 1;

	greedy_init();

	for (int i = 0; i < nSample; i++)
        for (int j = 0; j < nFeat; ++j)
        	set_feats(i,j,feats_R(i, j)); //the greedy should be i,j
            //feats[i][j] = feats_R(i, j);
    for (int i = 0; i < nSample; i++)
        for (int j = 0; j < nClass; ++j)
        	set_rewards(i,j, reward_R(i, j));

    printf("before learn\n");
    learn_from_data_greedy(1, 1, feats, rewards, nSample);
    printf("after learn\n");
   
	// return a dataframe object.s
	Rcpp::NumericVector v1(nNode);
	Rcpp::NumericVector v2(nNode);
	Rcpp::NumericVector v3(nNode);
	for(int i=0; i< nNode;i++)
	{
		v1[i] = i+1;
		if(i+1<=(1<<level)-1) //judge the node if it is the leaf
		{
			v2[i] = get_greedytree_feat(i+1);
		}
		else{
			v2[i]=-1;
		}
		v3[i] = get_greedytree_thre(i+1);
	} 
	//free_memory_greedy();
	free(feats);
	free(rewards);
	delete[] greedytree;
	return Rcpp::DataFrame::create(Rcpp::Named("node_id") = v1, Rcpp::Named("i") = v2, Rcpp::Named("b") = v3);
}
