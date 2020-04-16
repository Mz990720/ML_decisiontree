#include <algorithm>
#include <iostream>
#include <cstring>

using namespace std;
const int MAXNODE = (1 << 10) + 100;
int MAXF = 80; // max feature
const double INF = 1e18 + 10;
const int MAXN = 1e6 + 10;
int MAXK = 16; // max class
int nSample, nFeat, nClass, level, skip_node;
int cur_feat;
double **rewards;
double **feats;
double best_reward = 0;
double allbudget;
double *rewardsum;
int left_action,right_action;
double budgetcost[3];


struct budget_reward{
	double cons_rewards;
	double cons_budget;
};

struct treeNode
{
    int feat, row;
    double threshold = -INF;
    double reward = 0;
    bool end = 0;
    double Budget;
};
treeNode tree[MAXNODE];

struct thelist
{
    int start;
    int end;
    treeNode queue[MAXNODE];
};
struct thelist Queue;



void Queuepush(treeNode x)
{
	Queue.end++;
	Queue.queue[Queue.end-1] = x;
}

struct treeNode Queuepop()
{
	if(Queue.start == Queue.end) //队列空的 
	{
		printf("false!"); 
	}
	Queue.start++;
	return Queue.queue[Queue.start-1];
}

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
    memset(Queue.queue, 0, sizeof Queue.queue);
    Queue.end = Queue.start = 0;
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
void findbestslice(int i, int node_row, double **node_feats, double **node_rewards,double budget)
{
    feats = node_feats; 
    struct budget_reward **cur_best;
	struct budget_reward cu; 
    cur_best = allocate<struct budget_reward>(nFeat, node_row);
    int **orderFeat;
    orderFeat = allocate<int>(nFeat, node_row);
    for (int p = 0; p < nFeat; ++p)
        for (int n = 0; n < node_row; ++n)
            orderFeat[p][n] = n;
    sort_feat(orderFeat, node_row);
    double **sum_up, **sum_down; //nSample*nClass
    sum_down = allocate<double>(node_row, nClass);
    sum_up = allocate<double>(node_row, nClass);
    bool flag = 0; 
    double *rewardsum;
    rewardsum = allocate<double>(nClass);
    sumofreward(rewardsum, node_rewards, node_row);
    for (int p = 0; p < nFeat; ++p)
        for (int n = 0; n < node_row - 1; ++n)
        {
            int pos = orderFeat[p][n];
            int new_pos = orderFeat[p][n + 1];
            if (node_feats[p][pos] == node_feats[p][new_pos])
            {
                for (int k = 0; k < nClass; ++k)
                {
                    if (!n)
                        sum_down[n][k] = node_rewards[pos][k];
                    else
                        sum_down[n][k] = sum_down[n - 1][k] + node_rewards[pos][k];
                    sum_up[n][k] = rewardsum[k] - sum_down[n][k];
                }
                cur_best[p][n].cons_rewards = 0;
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
            cur_best[p][n].cons_rewards = best_up + best_down;  //直接将这个换为另一种数据结构 换为结构体二维数组 其中存放最好的rewards  
            flag = 1;
        }
    if (flag)
    {
        findbest(cur_best, nFeat, node_row - 1, i); //这一步的函数需要更改 改为结合budget的   在这一步函数中找到该个点最好的划分点  
        int bigger = orderFeat[tree[i].feat][tree[i].row + 1];
        int rank = tree[i].row; // the rank in this feat
        // before  the row is just the rank  now it is the real
        tree[i].row = orderFeat[tree[i].feat][tree[i].row];
        tree[i].threshold = (node_feats[tree[i].row][tree[i].feat] + node_feats[bigger][tree[i].feat]) / 2; // the breakpoint
    	//need to store the action
    	left_action = right_action = 0; 
    	double *sumup; // up shows left_action down shows right_action
    	sumup = allocate<double>(nClass);
    	double *sumdown;
    	sumdown = allocate<double>(nClass);
    	for(int j=0; j<=rank; j++)
        {
        	sumup[0] = sumup[0] + node_rewards[orderFeat[tree[i].feat][j]][0];
        } 
        for(int j=0; j<=rank; j++)
        {
        	sumup[1] = sumup[1] + node_rewards[orderFeat[tree[i].feat][j]][1];
        }
       
        for(int j=0; j<=rank; j++)
        {
        	sumup[2] = sumup[2] + node_rewards[orderFeat[tree[i].feat][j]][2];
        }
       
        for(int j=rank+1; j<node_row; j++)
        {
        	sumdown[0] = sumdown[0] + node_rewards[orderFeat[tree[i].feat][j]][0];
        }
       
        for(int j=rank+1; j<node_row; j++)
        {
        	sumdown[1] = sumdown[1] + node_rewards[orderFeat[tree[i].feat][j]][1];
        }
       
        for(int j=rank+1; j<node_row; j++)
        {
        	sumdown[2] = sumdown[2] + node_rewards[orderFeat[tree[i].feat][j]][2];
        }
        
        for(int k=1;k<nClass;k++)
        {
        	if(sumup[left_action]<sumup[k])
        	{
        		left_action = k;
        	}
        }
        for(int k=1; k<nClass;k++)
        {
        	if( sumdown[right_action] < sumdown[k] )
        		right_action = k;
        }
        printf("%d %d \n",left_action,right_action );
    	//tree[2*i].threshold = left_action;
    	//tree[2*i+1].threshold = right_action;
    }
    else
    {
        tree[i].end = 1;
        tree[i].threshold = INF;
        tree[i].feat = 0; //榛樿鍦╔0
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


//之后的参数直接调用该树节点的成员 
void learn_from_data(int i, int layer, double **father_feats, double **father_rewards, int father_row,double budget)
{
    if (layer == level)
    {
        tree[i].end = 1;
        findbestslice(i, father_row, father_feats, father_rewards,budget);
        return;
    }
    findbestslice(i, father_row, father_feats, father_rewards,budget);
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
void greedy()
{
    learn_from_data(1, 1, feats, rewards, nSample,allbudget);  
    printtree();
}
int main()
{
    cout << "nFeat:\tnSample:\tnClass:\t" << endl;
    cin >> nFeat >> nSample >> nClass;
    init();
    cout << "input: feats(" << nSample << "*" << nFeat << ")" << endl;
	feats[0][0] = 8;
	feats[0][1] = 2;
	feats[0][2] = 4;
	feats[1][0] = 3;
	feats[1][1] = 8;
	feats[1][2] = 1;
	rewards[0][0] = 8;
	rewards[0][1] = 19;
	rewards[0][2] = 4;
	rewards[1][0] = 10.5;
	rewards[1][1] = 9;
	rewards[1][2] = 11.5;
	rewards[2][0] = 17;
	rewards[2][1] = 10;
	rewards[2][2] = 11;
	
	budgetcost[0] = 6;
	budgetcost[1] = 7;
	budgetcost[2] = 8;
	
	/*for (int i = 0; i < nSample; i++)
        for (int j = 0; j < nFeat; ++j)
            cin >> feats[i][j];
    cout << "input: rewards(" << nSample << "*" << nClass << ")" << endl;
    for (int i = 0; i < nSample; i++)
        for (int j = 0; j < nClass; ++j)
            cin >> rewards[i][j];*/
    cout << " input: level:";
    cin >> level;
    cout << " input: budget:";
    cin >> allbudget;
    greedy();
    cout << endl
         << "best reward=" << best_reward;
	 
}
