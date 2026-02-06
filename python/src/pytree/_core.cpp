#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ptree_core.h"

#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static arma::vec as_vec(const py::array_t<double, py::array::forcecast> &arr)
{
    auto buf = arr.request();
    if (buf.ndim != 1)
    {
        throw std::runtime_error("Expected 1D array");
    }
    arma::vec out(static_cast<double *>(buf.ptr), buf.shape[0]);
    return out;
}

static arma::mat as_mat(const py::array_t<double, py::array::f_style | py::array::forcecast> &arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2)
    {
        throw std::runtime_error("Expected 2D array");
    }
    arma::mat out(static_cast<double *>(buf.ptr), buf.shape[0], buf.shape[1]);
    return out;
}

static py::array_t<double> to_numpy(const arma::mat &mat)
{
    py::array_t<double, py::array::f_style> arr(
        {static_cast<py::ssize_t>(mat.n_rows), static_cast<py::ssize_t>(mat.n_cols)});
    std::memcpy(arr.mutable_data(), mat.memptr(), sizeof(double) * mat.n_rows * mat.n_cols);
    return arr;
}

static py::array_t<double> to_numpy(const arma::vec &vec)
{
    py::array_t<double> arr({static_cast<py::ssize_t>(vec.n_elem)});
    std::memcpy(arr.mutable_data(), vec.memptr(), sizeof(double) * vec.n_elem);
    return arr;
}

PYBIND11_MODULE(_core, m)
{
    m.def("set_num_threads",
          [](int n)
          {
#ifdef _OPENMP
              omp_set_num_threads(n);
#else
              (void)n;
#endif
          },
          py::arg("n"),
          "Set the number of OpenMP threads.");

    m.def("get_max_threads",
          []() -> int
          {
#ifdef _OPENMP
              return omp_get_max_threads();
#else
              return 1;
#endif
          },
          "Get the maximum number of OpenMP threads.");

    m.def("fit",
          [](const py::array_t<double, py::array::forcecast> &R,
             const py::array_t<double, py::array::forcecast> &Y,
             const py::array_t<double, py::array::f_style | py::array::forcecast> &X,
             const py::array_t<double, py::array::f_style | py::array::forcecast> &Z,
             const py::array_t<double, py::array::f_style | py::array::forcecast> &H,
             const py::array_t<double, py::array::forcecast> &portfolio_weight,
             const py::array_t<double, py::array::forcecast> &loss_weight,
             const py::array_t<double, py::array::forcecast> &stocks,
             const py::array_t<double, py::array::forcecast> &months,
             const py::array_t<double, py::array::forcecast> &unique_months,
             const py::array_t<double, py::array::forcecast> &first_split_var,
             const py::array_t<double, py::array::forcecast> &second_split_var,
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
             const py::array_t<double, py::array::f_style | py::array::forcecast> &list_K,
             bool random_split)
          {
              // Copy arrays to Armadillo types while GIL is held
              auto r = as_vec(R);
              auto y = as_vec(Y);
              auto x = as_mat(X);
              auto z = as_mat(Z);
              auto h = as_mat(H);
              auto pw = as_vec(portfolio_weight);
              auto lw = as_vec(loss_weight);
              auto st = as_vec(stocks);
              auto mo = as_vec(months);
              auto um = as_vec(unique_months);
              auto fsv = as_vec(first_split_var);
              auto ssv = as_vec(second_split_var);
              auto lk = as_mat(list_K);

              PTreeOutput output;
              {
                  py::gil_scoped_release release;
                  output = PTreeFit(
                      r, y, x, z, h, pw, lw, st, mo, um, fsv, ssv,
                      num_stocks, num_months, min_leaf_size, max_depth,
                      num_iter, num_cutpoints, eta, equal_weight, no_H,
                      abs_normalize, weighted_loss, lambda_mean, lambda_cov,
                      lambda_mean_factor, lambda_cov_factor, early_stop,
                      stop_threshold, lambda_ridge, a1, a2, lk, random_split);
              }

              py::dict result;
              result["tree"] = output.tree;
              result["json"] = output.json;
              result["leaf_weight"] = to_numpy(output.leaf_weight);
              result["leaf_id"] = to_numpy(output.leaf_id);
              result["ft"] = to_numpy(output.ft);
              result["ft_benchmark"] = to_numpy(output.ft_benchmark);
              result["portfolio"] = to_numpy(output.portfolio);

              py::list crit;
              for (const auto &vec : output.all_criterion)
              {
                  crit.append(to_numpy(vec));
              }
              result["all_criterion"] = crit;
              return result;
          },
          py::arg("R"),
          py::arg("Y"),
          py::arg("X"),
          py::arg("Z"),
          py::arg("H"),
          py::arg("portfolio_weight"),
          py::arg("loss_weight"),
          py::arg("stocks"),
          py::arg("months"),
          py::arg("unique_months"),
          py::arg("first_split_var"),
          py::arg("second_split_var"),
          py::arg("num_stocks"),
          py::arg("num_months"),
          py::arg("min_leaf_size"),
          py::arg("max_depth"),
          py::arg("num_iter"),
          py::arg("num_cutpoints"),
          py::arg("eta"),
          py::arg("equal_weight"),
          py::arg("no_H"),
          py::arg("abs_normalize"),
          py::arg("weighted_loss"),
          py::arg("lambda_mean"),
          py::arg("lambda_cov"),
          py::arg("lambda_mean_factor"),
          py::arg("lambda_cov_factor"),
          py::arg("early_stop"),
          py::arg("stop_threshold"),
          py::arg("lambda_ridge"),
          py::arg("a1"),
          py::arg("a2"),
          py::arg("list_K"),
          py::arg("random_split"));

    m.def("predict",
          [](const py::array_t<double, py::array::f_style | py::array::forcecast> &X,
             const std::string &json_string,
             const py::array_t<double, py::array::forcecast> &months)
          {
              auto x = as_mat(X);
              auto mo = as_vec(months);
              arma::vec leaf_index;
              {
                  py::gil_scoped_release release;
                  leaf_index = PTreePredict(x, json_string, mo);
              }
              return to_numpy(leaf_index);
          },
          py::arg("X"),
          py::arg("json_string"),
          py::arg("months"));
}
