#include "policy.h"


double true_tree(double data[], int nodes_num)
{
    queue<int *> nodes;
    int *all_data = new int[nSample]{0};
    for (int m = 0; m < nSample; m++)
    {
        all_data[m] = 1;
    }
    nodes.push(all_data);

    int **data_list = new int *[nodes_num];
    for (int i = 0; i < nodes_num; i++)
    {
        data_list[i] = new int[nSample]{0};
    }

    int *data_left;
    int *data_right;
    for (int n = 0; n < (nodes_num >> 1); n++)
    {
        data_left = data_list[n * 2];
        data_right = data_list[n * 2 + 1];

        int *all_data = nodes.front();
        nodes.pop();
        int feature_idx = data[n * 3 + 1];
		//cout << feature_idx << ' ';
		if (feature_idx != -1) {
			double feature_threshold = data[n * 3 + 2];

			for (int m = 0; m < nSample; m += 1)
			{
				if (all_data[m] == 0)
				{
					continue;
				}
				if (feats[feature_idx][m] < feature_threshold)
				{
					data_left[m] = 1;
				}
				else
				{
					data_right[m] = 1;
				}
			}
		}
        nodes.push(data_left);
        nodes.push(data_right);
    }

    double reward = 0;
    int k = 0;
    int k_node = nodes_num >> 1;
    while (!nodes.empty())
    {
        k = data[k_node * 3 + 2];
        int *all_data = nodes.front();
        nodes.pop();
        for (int i = 0; i < nSample; i++)
        {
            // cout << all_data[i];
            if (all_data[i] == 1)
            {
                reward += rewards[i][k];
            }
        }
        // cout << endl;
        k_node++;
    }

    for (int i = 0; i < nodes_num; i++)
    {
        delete[] data_list[i];
    }
    delete[] data_list;
    delete[] all_data;

    return reward;
}

double execute_test(int idx)
{
    sprintf(test, "%d", idx);
    read();
    init();
    learn_from_data(1, 1);

    int nodes_num = (1 << (level + 1)) - 1;
    double *result = new double[nodes_num * 3];

    for (int i = 0; i < nodes_num; i++)
    {
        result[i * 3] = i + 1;
        result[i * 3 + 1] = tree[i + 1].feat;
        result[i * 3 + 2] = tree[i + 1].threshold;
        //cout << i + 1 << ' ' << tree[i + 1].feat << ' ' << tree[i + 1].threshold << endl;
    }

    double answer = true_tree(result, nodes_num);
    delete[] result;
    return answer;
}

void test_run(int begin_idx, int end_idx, int jump, char *answer_path, char *f_p = NULL)
{
	if (f_p)
	{
		strncpy(path, f_p, 50);
	}
    skip_node = jump;
    std::ifstream answer_file;
    answer_file.open(answer_path);
    int x;
    double y;
    double answer;
    do
    {
        do
        {
            answer_file >> x >> y;
        } while (x < begin_idx);
        answer = execute_test(x);
        if (y == answer)
        {
            cout << "test case " << x << " Accepted! " << endl;
        }
        else
        {
            cout << "test case " << x
                 << " Failed! Answer is " << y
                 << ". But calculation result is " << answer << "." << endl;
        }
    } while (x + 1 < end_idx);
}

void test_100_run(int jump)
{
    skip_node = jump;
    double answer = execute_test(100);
    cout << answer << endl;
}

int main(int argc, char *argv[])
{
    // getopts(argc, argv);
    char answer_path[50];

    strncpy(answer_path, "../test_data/answer.txt", 50);
    test_run(1, 46, 1, answer_path);
    cout << endl;
    strncpy(answer_path, "../test_data/answer_jump.txt", 50);
    test_run(39, 46, 2, answer_path);
    cout << endl;
    strncpy(answer_path, "../test_data/answer_jump.txt", 50);
    test_run(46, 47, 4, answer_path);
    cout << endl;
    strncpy(answer_path, "../test_data/answer_jump.txt", 50);
    test_run(47, 48, 2, answer_path);
    cout << endl;
    strncpy(answer_path, "../test_data/answer_jump.txt", 50);
    test_run(48, 54, 5, answer_path);
    cout << endl;
    test_100_run(1);
    test_100_run(10);

    char f_p[50] = "01Feature/";
    strncpy(answer_path, "../test_data/01Feature/answer.txt", 50);
    test_run(1, 9, 1, answer_path, f_p);
    cout << endl;

    return 0;
}
