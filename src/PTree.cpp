#include "common.h"
#include "ptree_core.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List PTree_cpp(arma::vec R, arma::vec Y, arma::mat X, arma::mat Z, arma::mat H, arma::vec portfolio_weight, arma::vec loss_weight, arma::vec stocks, arma::vec months, arma::vec unique_months, arma::vec first_split_var, arma::vec second_split_var, size_t num_stocks, size_t num_months, size_t min_leaf_size = 100, size_t max_depth = 5, size_t num_iter = 30, size_t num_cutpoints = 4, double eta = 1.0, bool equal_weight = false, bool no_H = false, bool abs_normalize = false, bool weighted_loss = false, double lambda_mean = 0, double lambda_cov = 0, double lambda_mean_factor = 0, double lambda_cov_factor = 0, bool early_stop = false, double stop_threshold = 0.95, double lambda_ridge = 0, double a1 = 0.05, double a2 = 1.0, arma::mat list_K = arma::zeros(2, 2), bool random_split = false)
{
    PTreeOutput output = PTreeFit(
        R, Y, X, Z, H, portfolio_weight, loss_weight, stocks, months,
        unique_months, first_split_var, second_split_var, num_stocks, num_months,
        min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight,
        no_H, abs_normalize, weighted_loss, lambda_mean, lambda_cov,
        lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold,
        lambda_ridge, a1, a2, list_K, random_split);

    Rcpp::List all_criterion = Rcpp::List::create();
    for (size_t i = 0; i < output.all_criterion.size(); i++)
    {
        all_criterion.push_back(output.all_criterion[i], std::to_string(i));
    }

    Rcpp::StringVector output_tree(1);
    output_tree(0) = output.tree;

    Rcpp::StringVector json_output(1);
    json_output[0] = output.json;

    return Rcpp::List::create(
        Rcpp::Named("tree") = output_tree,
        Rcpp::Named("leaf_weight") = output.leaf_weight,
        Rcpp::Named("leaf_id") = output.leaf_id,
        Rcpp::Named("ft") = output.ft,
        Rcpp::Named("ft_benchmark") = output.ft_benchmark,
        Rcpp::Named("portfolio") = output.portfolio,
        Rcpp::Named("json") = json_output,
        Rcpp::Named("all_criterion") = all_criterion);
}
