#include <algorithm>
#include <iostream>
#include <cstring>

using namespace std;
const int MAXNODE = (1 << 10) + 100;
int MAXF = 80; // max feature
const double INF = 1e18 + 10;
const int MAXN = 1e6 + 10;
int MAXK = 16; // max class
int nSample, nFeat, nClass, level, skip_node,skip;
int cur_feat;
double **rewards; //存放rewards
double **feats; //存放特征值以及各个 
double best_reward = 0;
double best_reward_skip = 0;
double *rewardsum;

struct treeNode
{
    int feat, row;
    double threshold = -INF;  //划分点
    double reward = 0;
    bool end = 0;
};
treeNode tree[MAXNODE];

bool cmp(int a, int b)
{
    return feats[a][cur_feat] <= feats[b][cur_feat];  
}
void sort_feat(int **orderFeat, int node_row)
{
    for (int i = 0; i < nFeat; i++)
    {
        cur_feat = i;
        sort(orderFeat[i], orderFeat[i] + node_row, cmp);  
    }
}
template <typename T>
T *allocate(int m)
{
    T *a = new T[m]();
    return a;
}
template <typename T>
T **allocate(int m, int n)
{
    T **a = new T *[m];
    for (int i = 0; i < m; i++)
    {
        a[i] = new T[n]();
    }
    return a;
}
template <typename T>
void delete_array(T *a)
{
    delete[] a;
}
template <typename T>
void delete_array(T **a, int m)
{
    for (int i = 0; i < m; i++)
        delete[] a[i];
}
void init()
{
    feats = allocate<double>(MAXF, MAXN);
    rewards = allocate<double>(MAXN, MAXK);
    memset(tree, 0, sizeof tree);
}
void findbest(double **a, int row, int column, int i)
{
    double max = -INF;
    for (int q = 0; q < row; ++q)
        for (int w = 0; w < column; ++w)
            if (a[q][w] > max)
            {
                max = a[q][w];
                tree[i].feat = q;
                tree[i].row = w;
            }
    tree[i].reward = max;
}
void sumofreward(double *rewardsum, double **node_rewards, int node_row)
{
    for (int k = 0; k < nClass; ++k)
        for (int n = 0; n < node_row; ++n)
            rewardsum[k] += node_rewards[n][k];
}
void  findbestslice_skip(int k,int i, int node_row, double **node_feats, double **node_rewards)
{
	feats = node_feats; //在sort_feat中
    double **cur_best;  //对于每一种划分，找到一种最大的reward，放入cur_best二维数组中    
    cur_best = allocate<double>(nFeat, node_row);
    int **orderFeat;
    orderFeat = allocate<int>(nFeat, node_row);
    for (int p = 0; p < nFeat; ++p)
        for (int n = 0; n < node_row; ++n)
            orderFeat[p][n] = n; //存放的是第p个特征 第n个大/小的smaple标记号 
	
    sort_feat(orderFeat, node_row);
    double **sum_up, **sum_down; //nSample*nClass
    sum_down = allocate<double>(node_row, nClass);
    sum_up = allocate<double>(node_row, nClass);
    bool flag = 0;  //防止出现所有元素都相同的情况
    double *rewardsum;
    rewardsum = allocate<double>(nClass);
    sumofreward(rewardsum, node_rewards, node_row);
    for (int p = 0; p < nFeat; ++p)
        for (int n = 0; n < node_row - 1; n=n+k)
        {
        	if(n >= node_row-1) //越界 进行调整 
        	{
        		n = node_row-2;
			}
            int pos = orderFeat[p][n];
            int new_pos = orderFeat[p][n + 1];
            if (node_feats[p][pos] == node_feats[p][new_pos])  //发现了相同的特征值  
            {
                for (int k = 0; k < nClass; ++k)
                {
                    if (!n)
                        sum_down[n][k] = node_rewards[pos][k];
                    else
                        sum_down[n][k] = sum_down[n - 1][k] + node_rewards[pos][k];
                    sum_up[n][k] = rewardsum[k] - sum_down[n][k];
                }
                cur_best[p][n] = 0;
                continue;
            }
            for (int k = 0; k < nClass; ++k)
            {
                if (!n)
                    sum_down[n][k] = node_rewards[pos][k];
                else
                    sum_down[n][k] = sum_down[n - 1][k] + node_rewards[pos][k];
                sum_up[n][k] = rewardsum[k] - sum_down[n][k];
            }
            double best_up, best_down;
            best_up = *max_element(sum_up[n], sum_up[n] + nClass);
            best_down = *max_element(sum_down[n], sum_down[n] + nClass);
            cur_best[p][n] = best_up + best_down;
            flag = 1;
        }

    if (flag)
    {
        findbest(cur_best, nFeat, node_row - 1, i);
        int bigger = orderFeat[tree[i].feat][tree[i].row + 1];
        tree[i].row = orderFeat[tree[i].feat][tree[i].row];
        tree[i].threshold = (node_feats[tree[i].row][tree[i].feat] + node_feats[bigger][tree[i].feat]) / 2;
    }
    else
    {
        tree[i].end = 1;
        tree[i].threshold = INF;
        tree[i].feat = 0; //默认在X0
        tree[i].reward = *max_element(rewardsum, rewardsum + nClass);
    }
    delete_array<double>(rewardsum);
    delete_array<int>(orderFeat, nFeat);
    delete_array<double>(cur_best, node_row);
    delete_array<double>(sum_up, node_row);
    delete_array<double>(sum_down, node_row);
}
void findbestslice(int i, int node_row, double **node_feats, double **node_rewards)
{
    feats = node_feats; //在sort_feat中
    double **cur_best;  //对于每一种划分，找到一种最大的reward，放入cur_best二维数组中    
    cur_best = allocate<double>(nFeat, node_row);
    int **orderFeat;
    orderFeat = allocate<int>(nFeat, node_row);
    for (int p = 0; p < nFeat; ++p)
        for (int n = 0; n < node_row; ++n)
            orderFeat[p][n] = n; //存放的是第p个特征 第n个大/小的smaple标记号 
	
    sort_feat(orderFeat, node_row);
    double **sum_up, **sum_down; //nSample*nClass
    sum_down = allocate<double>(node_row, nClass);
    sum_up = allocate<double>(node_row, nClass);
    bool flag = 0;  //防止出现所有元素都相同的情况
    double *rewardsum;
    rewardsum = allocate<double>(nClass);
    sumofreward(rewardsum, node_rewards, node_row);
    for (int p = 0; p < nFeat; ++p)
        for (int n = 0; n < node_row - 1; ++n)
        {
            int pos = orderFeat[p][n];
            int new_pos = orderFeat[p][n + 1];
            if (node_feats[p][pos] == node_feats[p][new_pos])  //发现了相同的特征值  
            {
                for (int k = 0; k < nClass; ++k)
                {
                    if (!n)
                        sum_down[n][k] = node_rewards[pos][k];
                    else
                        sum_down[n][k] = sum_down[n - 1][k] + node_rewards[pos][k];
                    sum_up[n][k] = rewardsum[k] - sum_down[n][k];
                }
                cur_best[p][n] = 0;
                continue;
            }
            for (int k = 0; k < nClass; ++k)
            {
                if (!n)
                    sum_down[n][k] = node_rewards[pos][k];
                else
                    sum_down[n][k] = sum_down[n - 1][k] + node_rewards[pos][k];
                sum_up[n][k] = rewardsum[k] - sum_down[n][k];
            }
            double best_up, best_down;
            best_up = *max_element(sum_up[n], sum_up[n] + nClass);
            best_down = *max_element(sum_down[n], sum_down[n] + nClass);
            cur_best[p][n] = best_up + best_down;
            flag = 1;
        }
	
    if (flag)
    {
        findbest(cur_best, nFeat, node_row - 1, i);
        int bigger = orderFeat[tree[i].feat][tree[i].row + 1];
        tree[i].row = orderFeat[tree[i].feat][tree[i].row];
        tree[i].threshold = (node_feats[tree[i].row][tree[i].feat] + node_feats[bigger][tree[i].feat]) / 2;
    }
    else
    {
        tree[i].end = 1; //若节点内部元素全部相同 则直接结束 标记  
        tree[i].threshold = INF;
        tree[i].feat = 0; //默认在X0
        tree[i].reward = *max_element(rewardsum, rewardsum + nClass);
    }
    delete_array<double>(rewardsum);
    delete_array<int>(orderFeat, nFeat);
    delete_array<double>(cur_best, node_row);
    delete_array<double>(sum_up, node_row);
    delete_array<double>(sum_down, node_row);
}

void copyfeats(double thre, int ft, double **father_feats, double **father_rewards, int father_row, double **left_feats, double **left_rewards, int &left_row, double **right_feats, double **right_rewards, int &right_row)
{
    left_row = right_row = 0;
    for (int n = 0; n < father_row; ++n)
    {
        if (father_feats[n][ft] < thre)
        {
        	
            for (int p = 0; p < nFeat; ++p)
                left_feats[left_row][p] = father_feats[n][p];
            for (int k = 0; k < nClass; ++k)
                left_rewards[left_row][k] = father_rewards[n][k];
            //memcpy(left_feats[left_row], father_feats[n], sizeof(left_feats[left_row]));
            // memcpy(left_rewards[left_row], father_rewards[n], sizeof(father_rewards[n]));
            left_row++;
        }
        else
        {
            for (int p = 0; p < nFeat; ++p)
                right_feats[right_row][p] = father_feats[n][p];
            for (int k = 0; k < nClass; ++k)
                right_rewards[right_row][k] = father_rewards[n][k];
            //memcpy(right_feats[right_row], father_feats[n], sizeof(father_feats[n]));
            //memcpy(right_rewards[right_row], father_rewards[n], sizeof(father_rewards[n]));
            right_row++;
        }
    }
    delete_array<double>(father_feats, father_row);
    delete_array<double>(father_rewards, father_row);
}
void learn_from_data_skip(int k,int i, int layer, double **father_feats, double **father_rewards, int father_row)
{
	printf("in\n");
	if (layer == level)
    {
        tree[i].end = 1;
        findbestslice_skip(k,i, father_row, father_feats, father_rewards);  //通过skip的方式查照最优划分点 
        return;
    }
    findbestslice_skip(k,i, father_row, father_feats, father_rewards);
    if (tree[i].end)
        return;
    int ft = tree[i].feat;
    int left = i << 1, right = (i << 1) + 1;
    double **left_feats, **right_feats, **left_rewards, **right_rewards;
    left_feats = allocate<double>(nSample, nFeat);
    right_feats = allocate<double>(nSample, nFeat);
    left_rewards = allocate<double>(nSample, nClass);
    right_rewards = allocate<double>(nSample, nClass);
    int left_row, right_row;
    copyfeats(tree[i].threshold, ft, father_feats, father_rewards, father_row, left_feats, left_rewards, left_row, right_feats, right_rewards, right_row);
    learn_from_data_skip(k,left, layer + 1, left_feats, left_rewards, left_row);
    learn_from_data_skip(k,right, layer + 1, right_feats, right_rewards, right_row);
}
void learn_from_data(int i, int layer, double **father_feats, double **father_rewards, int father_row)
{
    if (layer == level)
    {
        tree[i].end = 1;
        findbestslice(i, father_row, father_feats, father_rewards);  //father_row是从上一层继承来的 
        return;
    }
    findbestslice(i, father_row, father_feats, father_rewards);
    if (tree[i].end)
        return;
    int ft = tree[i].feat;
    int left = i << 1, right = (i << 1) + 1;
    double **left_feats, **right_feats, **left_rewards, **right_rewards;
    left_feats = allocate<double>(nSample, nFeat);
    right_feats = allocate<double>(nSample, nFeat);
    left_rewards = allocate<double>(nSample, nClass);
    right_rewards = allocate<double>(nSample, nClass);
    int left_row, right_row;
    copyfeats(tree[i].threshold, ft, father_feats, father_rewards, father_row, left_feats, left_rewards, left_row, right_feats, right_rewards, right_row);
    learn_from_data(left, layer + 1, left_feats, left_rewards, left_row);
    learn_from_data(right, layer + 1, right_feats, right_rewards, right_row);
}
void printtree(int i = 1)
{
    cout << endl
         << "x" << tree[i].feat << "<" << tree[i].threshold << " ";
    if (tree[i].end)
        best_reward += tree[i].reward;
    else
    {
        printtree(i << 1);
        printtree((i << 1) + 1);
    }
}
void printtree_skip(int i = 1)
{
    cout << endl
         << "x" << tree[i].feat << "<" << tree[i].threshold << " ";
    if (tree[i].end)
    	
        best_reward_skip += tree[i].reward;
    else
    {
        printtree_skip(i << 1);
        printtree_skip((i << 1) + 1);
    }
}
void greedy()
{
    learn_from_data(1, 1, feats, rewards, nSample);
    printtree_skip();
}
void greedy_skip(int k)
{
	printf("greedy for skip\n");
	learn_from_data_skip(k,1, 1, feats, rewards, nSample);
	printf("learned\n");
    printtree_skip();
}
int main()
{
	int choose;
    cout << "nFeat:\tnSample:\tnClass:\t" << endl;
    cin >> nFeat >> nSample >> nClass;
    init();
    cout << "input: feats(" << nSample << "*" << nFeat << ")" << endl;
    for (int i = 0; i < nSample; i++)
        for (int j = 0; j < nFeat; ++j)
            cin >> feats[i][j];
    cout << "input: rewards(" << nSample << "*" << nClass << ")" << endl;
    for (int i = 0; i < nSample; i++)
        for (int j = 0; j < nClass; ++j)
            cin >> rewards[i][j];
    cout << " input: level:";
    cin >> level;
    cout << " 1 greedy 2 greedy_skip" << endl;
    cin >> choose;
    if(choose==1)
    {
    	greedy(); 
   		cout << endl
         	<< "best reward=" << best_reward;
	}
	else
	{
		cout << " input skip(k):";
    	cin >> skip; 
    	greedy_skip(skip);
    	cout << endl
    	     << "best reward(skip)=" << best_reward_skip;
	}   
}
