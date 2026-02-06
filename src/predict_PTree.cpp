#include "common.h"
#include "ptree_core.h"

// [[Rcpp::export]]
Rcpp::List predict_PTree_cpp(arma::mat X, Rcpp::StringVector json_string, arma::vec months)
{
    if (json_string.size() < 1)
    {
        Rcpp::stop("json_string is empty");
    }
    std::string json = Rcpp::as<std::string>(json_string[0]);
    arma::vec leaf_index = PTreePredict(X, json, months);
    return Rcpp::List::create(
        Rcpp::Named("leaf_index") = leaf_index);
}
