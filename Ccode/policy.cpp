#include "policy.h"

//config start
int mapping = 0;
int MAXMEM = 7;// max memory of PC. 7GB default.
//end

int MAXF = 80;// max feature
int MAXK = 16;// max class
int MAXHASH = 1e6 + 100;

int nSample, nFeat, nClass, level, skip_node = 1;
int cur_feat, total;
int **orderFeat;
bool *jump_leaf;
double **sum_reward, *leaf_thre;
double **feats, **rewards, best_reward,best_reward_greedy;

//hash start
int cur_hashnode = 0;
int maxn_hashnode = (int)(MAXMEM*1e9 / (MAXN / 8.0 + 13));
treeNode tree[MAXNODE];
greedytreeNode *greedytree;
vector<hashNode> *hash_nodes;
unsigned int hashing(bitset<MAXN> &vis, int level)
{
	unsigned int hash = 1;
	int pos = 0;
	while (pos < nSample)
	{
		unsigned int tem = 0;
		for (int i = 0; i < 32; i++)
		{
			tem <<= 1;
			tem |= vis[pos];
			pos++;
			if (pos == nSample)
				break;
		}
		if (tem != 0)
		{
			hash *= (tem | 1);
		}
	}
	hash = hash * level * level * level;
	return hash % MAXHASH;
}
bool equal(bitset<MAXN> &a, bitset<MAXN> &b, int n)
{
	for (int i = 0; i < n; i++)
		if (a[i] ^ b[i])
			return 0;
	return 1;
}
double search_hash(bitset<MAXN> &vis, int level, unsigned int hash_code)
{
	for (unsigned int i = 0; i < hash_nodes[hash_code].size(); i++)
	{
		hashNode tem = hash_nodes[hash_code][i];
		if (equal(tem.vis, vis, nSample) && tem.level == level)
		{
			return tem.reward;
		}
	}
	return -INF;
}
inline void add_hash(bitset<MAXN> &vis, int level, double reward, int hash_code)
{
	if (++cur_hashnode > maxn_hashnode) return;
	hash_nodes[hash_code].push_back({vis, level, reward});
}
//end

//interface start
void set_feats(int j, int i, double val)
{

	feats[j][i] = val;
}
void set_rewards(int i, int j, double val)
{
	rewards[i][j] = val;
}
int get_greedytree_feat(int i)
{
	return (int)greedytree[i].feat;
}
int get_tree_feat(int i)
{
	return (int)tree[i].feat;
}
double get_tree_thre(int i)
{
	return tree[i].threshold;
}
double get_greedytree_thre(int i)
{
	return greedytree[i].threshold;
}
//end

bool cmp(int a, int b)
{
	return feats[cur_feat][a] < feats[cur_feat][b];  
}
void sort_feat()
{
	for (int i = 0; i < nFeat; i++)
	{
		cur_feat = i;
		sort(orderFeat[i], orderFeat[i] + nSample, cmp);
	}
}

void sort_feat_greedy(int **orderFeat, int node_row)
{
 
  for (int i = 0; i < nFeat; i++) 
  {
    cur_feat = i;
    int temp;
	for(int j=0; j < node_row;j++)
	{
		for(int k=j+1;k<node_row;k++)
		{
			if(orderFeat[i][k]<orderFeat[i][j])
			{
				temp = orderFeat[i][j];
				orderFeat[i][j] = orderFeat[i][k];
				orderFeat[i][k]=temp;
			}
		}
	}
    //sort(orderFeat[i], orderFeat[i] + node_row, cmp);  the sort function here has some error
  }
}

template <typename T>
T *allocate(int m)
{
	T* a = new T[m]();
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
void delete_array(T* a)
{
    delete[] a;
}
template <typename T>
void delete_array(T** a,int m)
{
    for(int i=0 ;i < m ; i++)
        delete[] a[i];
}

void init()
{
	jump_leaf = allocate<bool>(MAXN);
	orderFeat = allocate<int>(MAXF, MAXN);
	sum_reward = allocate<double>(MAXN, MAXK);
	leaf_thre = allocate<double>(MAXN);
	feats = allocate<double>(MAXF, MAXN);
	rewards = allocate<double>(MAXN, MAXK);
	hash_nodes = allocate<vector<hashNode>>(MAXHASH);

	cur_hashnode = 0;
	for (int i = 0; i < MAXHASH; i++)
		hash_nodes[i].clear();
	memset(tree, 0, sizeof tree);
	for (int i = 0; i < nSample; i++)
		tree[1].vis.set(i);
	for (int i = 0; i < nFeat; i++)
		for (int j = 0; j < nSample; j++)
			orderFeat[i][j] = j;
}

/*void free_memory()
{
    //free memory
    delete [] jump_leaf;
    delete [] orderFeat;
    delete [] sum_reward;
    delete [] leaf_thre;
    delete [] feats;
    delete [] rewards;
    delete [] hash_nodes;
}*/
void free_memory_greedy()
{
	
	
	printf("7\n");
    for (int i = 0; i < MAXN; i++)
      delete[] feats[i];
  	printf("6\n");
    for (int i = 0; i < MAXN; i++)
      delete[] rewards[i];
    printf("5\n");
    delete[] greedytree;
    printf("8\n");
}

void free_memory()
{
    delete_array(jump_leaf);
    delete_array(orderFeat, MAXF);
    delete_array(sum_reward, MAXN);
    delete_array(leaf_thre);
    delete_array(feats, MAXF);
    delete_array(rewards, MAXN);
    delete_array(hash_nodes);
}

double leaf_learning(int i, unsigned int hash_code)
{
	double best_reward = -INF;
	// left child and right child
	int left = i << 1, right = (i << 1) + 1;
	for (int k = 0; k < nFeat; k++)
	{
		int count = 1, total = 0, cur_skip = 0; 
		//find the reward matrix  
		for (int j = 0; j < nSample;)
		{
			// from biggest feature to smallest
			int pos = orderFeat[k][j], new_pos = 0;
			j++;	
			if (!tree[i].vis[pos])   
				continue;
			for (int l = 0; l < nClass; l++)
				sum_reward[count][l] = rewards[pos][l] + sum_reward[count - 1][l];
			total++; 
			// adding the node with same feature
			while (j < nSample)
			{
				new_pos = orderFeat[k][j];
				while (j < nSample && !tree[i].vis[new_pos])
				{
					j++;
					new_pos = orderFeat[k][j];
				}
				if (j == nSample || feats[k][pos] != feats[k][new_pos])
					break;
				total++;
				for (int l = 0; l < nClass; l++)
					sum_reward[count][l] = rewards[new_pos][l] + sum_reward[count][l];
				j++;
			}
			// leaf_thre[i] is the threshold when cutting between i-1 and i
			leaf_thre[count] = (feats[k][pos] + (j == nSample ? 0 : feats[k][new_pos])) / 2;
			// to decide jumping at node count
			jump_leaf[count] = total / skip_node >= cur_skip;
			if (jump_leaf[count])
				cur_skip++;
			count++;
		}
		// calculate the best reward
		for (int j = 0; j < count; j++)
		{
			if (!jump_leaf[j])    //jump
				continue;
			double up_best = -INF, dw_best = -INF;
			double left_action = 0, right_action = 0;
			for (int l = 0; l < nClass; l++)
			{
				if (up_best < sum_reward[j][l])
				{
					up_best = sum_reward[j][l];
					right_action = l;
				}
				if (dw_best < sum_reward[count - 1][l] - sum_reward[j][l])
				{
					dw_best = sum_reward[count - 1][l] - sum_reward[j][l];
					left_action = l;
				}
			}
			if (best_reward < up_best + dw_best)
			{
				best_reward = up_best + dw_best;
				tree[i].threshold = j ? leaf_thre[j] : feats[k][orderFeat[k][0]] + 1;
				tree[i].feat = k;
				tree[left].action = tree[left].threshold = left_action;
				tree[right].action = tree[right].threshold = right_action;
			}
		}
	}
	// if there is no node, set everything to -1
	if (best_reward == -INF)
	{
		best_reward = 0;
		tree[i].threshold = -1;
		tree[i].feat = -1;
		tree[left].action = tree[left].threshold = -1;
		tree[right].action = tree[right].threshold = -1;
	}
	if (mapping == 2)
		add_hash(tree[i].vis, 1, best_reward, hash_code);
	return best_reward;
}
double learn_from_data(int i, int layer, bool memory)
{
	pair<bitset<MAXN>, int> key;
	unsigned int hash_code = 0;
	if (memory && mapping == 2)
	{
		hash_code = hashing(tree[i].vis, level - layer + 1);
		double reward = search_hash(tree[i].vis, level - layer + 1, hash_code);
		if (reward != -INF)
			return reward;
	}
	// father of leaf node
	if (layer >= level)
		return leaf_learning(i, hash_code);

	double cur_best = -INF, best_feat = -1, best_thre = -1;
	treeNode best_left, best_right;
	// left child, right child
	int left = i << 1, right = left + 1;
	for (int k = 0; k < nFeat; k++)
	{
		int cur_skip = 0, total = 0;
		tree[left].vis = tree[i].vis;
		tree[right].vis.reset();
		double a = learn_from_data(left, layer + 1);
		double b = learn_from_data(right, layer + 1);
		if (cur_best < a + b)
		{
			cur_best = a + b;
			best_feat = k;
			best_left = tree[left];
			best_right = tree[right];
			best_thre = feats[k][orderFeat[k][0]] + 1;
		}
		// separate the nodes from big feature node to samll one
		for (int j = 0; j < nSample;)
		{
			int pos = orderFeat[k][j], new_pos = 0;
			j++;
			if (!tree[left].vis[pos])
				continue;
			tree[left].vis.reset(pos);
			tree[right].vis.set(pos);
			total++;
			// jump the node with same feature
			while (j < nSample)
			{
				new_pos = orderFeat[k][j];
				while (j < nSample && !tree[left].vis[new_pos])
				{
					j++;
					new_pos = orderFeat[k][j];
				}
				if (j == nSample || feats[k][pos] != feats[k][new_pos])
					break;

				tree[left].vis.reset(new_pos);
				tree[right].vis.set(new_pos);
				total++;
				j++;
			}
			// jump node
			if (cur_skip > total / skip_node && j != nSample)
				continue;

			a = j == nSample ? 0 : learn_from_data(left, layer + 1);
			b = learn_from_data(right, layer + 1);

			cur_skip++;
			if (a + b > cur_best)
			{
				cur_best = a + b;
				best_feat = k;
				best_thre = j == nSample ? feats[k][pos] : ((feats[k][pos] + feats[k][new_pos]) / 2);
				best_left = tree[left];
				best_right = tree[right];
			}
		}
	}

	if (cur_best == -INF)
		cur_best = 0;
	tree[i].feat = best_feat;
	tree[i].threshold = best_thre;
	tree[left] = best_left;
	tree[right] = best_right;
	learn_from_data(left, layer + 1, false);
	learn_from_data(right, layer + 1, false);

	if (mapping == 2)
		add_hash(tree[i].vis, level - layer + 1, cur_best, hash_code);
	return cur_best;
}

void greedy_init()
{
    feats = allocate<double>(MAXN, MAXF);
    rewards = allocate<double>(MAXN, MAXK);
    greedytree = new greedytreeNode[MAXNODE];
    memset(greedytree, 0, sizeof greedytree);

}


void learn_from_data_greedy(int i, int layer, double **father_feats, double **father_rewards, int father_row)
{
    if (layer == level)
    {
        greedytree[i].end = 1;
        findbestslice(i, father_row, father_feats, father_rewards);
        return;
    }
    //This is the breakpoint
    printf("find best\n");
    findbestslice(i, father_row, father_feats, father_rewards);
    if (greedytree[i].end)
        return;
    int ft = greedytree[i].feat;
    int left = i << 1, right = (i << 1) + 1;
    double **left_feats, **right_feats, **left_rewards, **right_rewards;
    left_feats = allocate<double>(nSample, nFeat);
    right_feats = allocate<double>(nSample, nFeat);
    left_rewards = allocate<double>(nSample, nClass);
    right_rewards = allocate<double>(nSample, nClass);
    int left_row, right_row;
    copyfeats(greedytree[i].threshold, ft, father_feats, father_rewards, father_row, left_feats, left_rewards, left_row, right_feats, right_rewards, right_row);
    printf("the next left\n");
    learn_from_data_greedy(left, layer + 1, left_feats, left_rewards, left_row);
    printf("the next right\n");
    learn_from_data_greedy(right, layer + 1, right_feats, right_rewards, right_row);
    printf("to delete\n");
    for (int i = 0; i < nSample; i++)
    {

        delete[] left_feats[i];
        delete[] right_feats[i];
        delete[] left_rewards[i];
        delete[] right_rewards[i];
    }
    printf("delete over\n");
    return;
}

void findbestslice(int i, int node_row, double **node_feats, double **node_rewards)
{
	int left_action,right_action;
    feats = node_feats; //在sort_feat?
    double **cur_best;  //对于每一种划分，找到一种最大的reward，放入cur_best?
    cur_best = allocate<double>(nFeat, node_row);
    int **orderFeat;
    orderFeat = allocate<int>(nFeat, node_row);
    for (int p = 0; p < nFeat; ++p)
        for (int n = 0; n < node_row; ++n)
            orderFeat[p][n] = n;

    //printf("before sort\n");
    sort_feat_greedy(orderFeat, node_row);  
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
        int bigger = orderFeat[greedytree[i].feat][greedytree[i].row + 1];
        int rank = greedytree[i].row; // the rank in this feat
        // before  the row is just the rank  now it is the real
        greedytree[i].row = orderFeat[greedytree[i].feat][greedytree[i].row];
        greedytree[i].threshold = (node_feats[greedytree[i].row][greedytree[i].feat] + node_feats[bigger][greedytree[i].feat]) / 2; // the breakpoint
    	//need to store the action
    	left_action = right_action = 0; 
    	double *sumup; // up shows left_action down shows right_action
    	sumup = allocate<double>(nClass);
    	double *sumdown;
    	sumdown = allocate<double>(nClass);
    	for(int j=0; j<=rank; j++)
        {
        	sumup[0] = sumup[0] + node_rewards[orderFeat[greedytree[i].feat][j]][0];
        } 
        for(int j=0; j<=rank; j++)
        {
        	sumup[1] = sumup[1] + node_rewards[orderFeat[greedytree[i].feat][j]][1];
        }
       
        for(int j=0; j<=rank; j++)
        {
        	sumup[2] = sumup[2] + node_rewards[orderFeat[greedytree[i].feat][j]][2];
        }
       
        for(int j=rank+1; j<node_row; j++)
        {
        	sumdown[0] = sumdown[0] + node_rewards[orderFeat[greedytree[i].feat][j]][0];
        }
       
        for(int j=rank+1; j<node_row; j++)
        {
        	sumdown[1] = sumdown[1] + node_rewards[orderFeat[greedytree[i].feat][j]][1];
        }
       
        for(int j=rank+1; j<node_row; j++)
        {
        	sumdown[2] = sumdown[2] + node_rewards[orderFeat[greedytree[i].feat][j]][2];
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
    	greedytree[2*i].threshold = left_action;
    	greedytree[2*i+1].threshold = right_action;      
    }
    else
    {
    
        greedytree[i].end = 1;
        greedytree[i].reward = *max_element(rewardsum, rewardsum + nClass);
    }   
    printf("before findbest delete\n");
    delete[] rewardsum;
    for (int i = 0; i < nFeat; i++)
    {
    	delete[] orderFeat[i];
    	//delete[] cur_best[i];
    }
  	for (int i = 0; i < node_row; i++)
        delete[] sum_up[i];
    //delete_array<double>(sum_down, node_row);
    for (int i = 0; i < node_row; i++)
        delete[] sum_down[i];
    for (int i = 0; i < nFeat; i++)
        delete[] cur_best[i];
    printf("findbest delete\n");
    return;
}

bool comp(int a, int b)
{
	printf("in the comp\n");
    return feats[a][cur_feat] < feats[b][cur_feat];
}

void findbest(double **a, int row, int column, int i)
{
    double max = -INF;
    for (int q = 0; q < row; ++q)
        for (int w = 0; w < column; ++w)
            if (a[q][w] > max)
            {
                max = a[q][w];
                greedytree[i].feat = q;
                greedytree[i].row = w;
            }
    greedytree[i].reward = max;
}
void sumofreward(double *rewardsum, double **node_rewards, int node_row)
{
    for (int k = 0; k < nClass; ++k)
        for (int n = 0; n < node_row; ++n)
            rewardsum[k] += node_rewards[n][k];
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

void printtree(int i = 1)
{
    /*cout << endl
         << "x" << greedytree[i].feat << "<" << greedytree[i].threshold << " ";*/
    if (greedytree[i].end)
        best_reward_greedy += greedytree[i].reward;
    else
    {
        printtree(i << 1);
        printtree((i << 1) + 1);
    }
}