#include "ptree_core.h"

#include "APTree.h"
#include "json.h"
#include "json_io.h"
#include "model.h"
#include "state.h"

#include <cassert>
#include <sstream>

PTreeOutput PTreeFit(
    arma::vec R,
    arma::vec Y,
    arma::mat X,
    arma::mat Z,
    arma::mat H,
    arma::vec portfolio_weight,
    arma::vec loss_weight,
    arma::vec stocks,
    arma::vec months,
    arma::vec unique_months,
    arma::vec first_split_var,
    arma::vec second_split_var,
    size_t num_stocks,
    size_t num_months,
    size_t min_leaf_size,
    size_t max_depth,
    size_t num_iter,
    size_t num_cutpoints,
    double eta,
    bool equal_weight,
    bool no_H,
    bool abs_normalize,
    bool weighted_loss,
    double lambda_mean,
    double lambda_cov,
    double lambda_mean_factor,
    double lambda_cov_factor,
    bool early_stop,
    double stop_threshold,
    double lambda_ridge,
    double a1,
    double a2,
    arma::mat list_K,
    bool random_split)
{
    PTreeOutput output;

    std::map<size_t, size_t> months_list;
    assert(num_months == unique_months.n_elem);
    for (size_t i = 0; i < num_months; i++)
    {
        months_list[static_cast<size_t>(unique_months(i))] = i;
    }

    State state(
        X, Y, R, Z, H, portfolio_weight, loss_weight, stocks, months,
        first_split_var, second_split_var, num_months, months_list, num_stocks,
        min_leaf_size, max_depth, num_cutpoints, equal_weight, no_H,
        abs_normalize, weighted_loss, eta, lambda_mean, lambda_cov,
        lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold,
        lambda_ridge, a1, a2, list_K, random_split);

    APTreeModel model(lambda_cov);

    arma::umat Xorder(X.n_rows, X.n_cols, arma::fill::zeros);
    for (size_t i = 0; i < X.n_cols; i++)
    {
        Xorder.col(i) = arma::sort_index(X.col(i));
    }

    APTree root(state.num_months, 1, state.num_obs_all, 1, 0, &Xorder);
    root.setN(X.n_rows);

    model.initialize_portfolio(state, &root);
    model.initialize_regressor_matrix(state);

    bool break_flag = false;
    std::vector<double> criterion_values;
    output.all_criterion.clear();

    arma::vec temp_vec;
    for (size_t iter = 0; iter < num_iter; iter++)
    {
        root.grow(break_flag, model, state, iter, criterion_values);

        temp_vec.set_size(criterion_values.size());
        for (size_t i = 0; i < criterion_values.size(); i++)
        {
            temp_vec(i) = criterion_values[i];
        }
        output.all_criterion.push_back(temp_vec);

        if (break_flag)
        {
            break;
        }
    }

    arma::vec leaf_node_index;
    arma::mat all_leaf_portfolio;
    arma::mat leaf_weight;
    arma::mat ft;
    arma::mat ft_benchmark;

    model.calculate_factor(root, leaf_node_index, all_leaf_portfolio, leaf_weight,
                           ft, ft_benchmark, state);

    if (!ptree_is_quiet())
    {
        std::cout << "fitted tree " << std::endl;
        std::cout.precision(3);
        std::cout << root << std::endl;
    }

    std::stringstream trees;
    trees.precision(10);
    trees.str(std::string());
    trees << root;

    json j = tree_to_json(root);

    output.tree = trees.str();
    output.json = j.dump(4);
    output.leaf_weight = leaf_weight;
    output.leaf_id = leaf_node_index;
    output.ft = ft;
    output.ft_benchmark = ft_benchmark;
    output.portfolio = all_leaf_portfolio;

    return output;
}

arma::vec PTreePredict(arma::mat X, const std::string &json_string, arma::vec months)
{
    size_t N = X.n_rows;
    arma::vec leaf_index(N);

    size_t dim_theta = 0;
    json temp = json::parse(json_string);
    temp.at("dim_theta").get_to(dim_theta);

    APTree root(dim_theta);
    std::string json_mut = json_string;
    json_to_tree(json_mut, root);

    APTreeModel model(1.0);
    model.predict_AP(X, root, months, leaf_index);

    return leaf_index;
}
