#ifndef _POLICY_H_
#define _POLICY_H_

#include <algorithm>
#include <iostream>

#ifdef _WIN32
#include "getopt.h"
#else
#include <unistd.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <bitset>
#include <vector>
#include <map>
#include <ctime>

#define NO_MAP 0
#define HASHING 2
using namespace std;

extern int mapping;

const double INF = 1e18 + 10;
const int MAXN = 20000;
const int MAXNODE = 20000;
extern int MAXF;
extern int MAXK;
extern int MAXHASH;

extern int nSample, nFeat, nClass, level, skip_node;
extern int cur_feat, total;
extern int **orderFeat;
extern bool *jump_leaf;
extern double **sum_reward, *leaf_thre;
extern double **feats, **rewards, best_reward,best_reward_greedy;

struct greedytreeNode
{
    int feat, row;
    double threshold = 0;
    double reward = 0;
    bool end = 0;
};
extern greedytreeNode *greedytree;

struct treeNode
{
    int action, feat;
    double threshold;
    bitset<MAXN> vis;// set the node for sample selection  01 shows if it is still in
};
extern treeNode tree[];

struct hashNode
{
    bitset<MAXN> vis;
    int level;
    double reward;
};

extern vector<hashNode> *hash_Nodes;

unsigned int hashing(bitset<MAXN> &, int);
bool equal(bitset<MAXN> &, bitset<MAXN> &, int);
double search_hash(bitset<MAXN> &, int, unsigned int);

//interface start
void set_feats(int, int, double);
void set_rewards(int, int, double);
int get_tree_feat(int);
double get_tree_thre(int);
int get_greedytree_feat(int);
double get_greedytree_thre(int);
//end

void sort_feat();
void init();
double leaf_learning(int, unsigned int);
double learn_from_data(int, int, bool = true);

template <typename T>
T *allocate(int);
template <typename T>
T **allocate(int, int);
template <typename T>
void delete_array(T*);
template <typename T>
void delete_array(T**, int);
void free_memory();
void free_memory_greedy();
void greedy_init();
void learn_from_data_greedy(int i, int layer, double **father_feats, double **father_rewards, int father_row);
void findbestslice(int i, int node_row, double **node_feats, double **node_rewards);
void copyfeats(double thre, int ft, double **father_feats, double **father_rewards, int father_row, double **left_feats, double **left_rewards, int &left_row, double **right_feats, double **right_rewards, int &right_row);
void sort_feat_greedy(int **orderFeat, int node_row);
void sumofreward(double *rewardsum, double **node_rewards, int node_row);
void findbest(double **a, int row, int column, int i);
bool  comp(int a, int b);
void printtree(int i);



#endif
