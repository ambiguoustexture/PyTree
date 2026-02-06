#ifndef GUARD_ptree_core_h
#define GUARD_ptree_core_h

#include "common.h"

#include <string>
#include <vector>

struct PTreeOutput
{
    std::string tree;
    std::string json;
    arma::mat leaf_weight;
    arma::vec leaf_id;
    arma::mat ft;
    arma::mat ft_benchmark;
    arma::mat portfolio;
    std::vector<arma::vec> all_criterion;
};

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
    bool random_split);

arma::vec PTreePredict(arma::mat X, const std::string &json_string, arma::vec months);

#endif
