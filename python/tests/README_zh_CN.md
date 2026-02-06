# PyTree 测试

该目录包含 Python 绑定的最小冒烟测试，以及公开数据快照的校验。

## 依赖
- CMake >= 3.18
- Armadillo 头文件与库（Ubuntu 可用 `libarmadillo-dev`）
- OpenMP 编译工具链（g++）
- Python 依赖：`numpy`、`pybind11`、`scikit-build-core`

## 构建与运行（离线友好）
若无法访问 PyPI，可安装到本地目录并设置 `PYTHONPATH`：
```bash
python3 -m pip install ./python --no-build-isolation --no-deps --target python/.venv
export PYTHONPATH=python/.venv

python3 python/tests/test_smoke_fit_predict.py
python3 python/tests/test_validation.py
python3 python/tests/test_determinism.py
python3 python/tests/test_public_data.py
python3 python/tests/compare_public_data.py
```

说明：这里的 `python/.venv` 只是本地目标目录，并非真正的 virtualenv。
如遇用户站点包干扰，可设置 `PYTHONNOUSERSITE=1`，或移除 user
site-packages 中旧的 `pytree` 包。

## 使用 pytest（可选）
部分环境会自动加载 pytest 插件并触发网络请求，可禁用自动加载：
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=python/.venv \
  python3 -m pytest -q python/tests/test_smoke_fit_predict.py
```

## 覆盖内容（每个测试的检查点）
- `test_smoke_fit_predict.py`：合成数据的端到端 fit/predict；检查
  `ft == portfolio @ leaf_weight` 一致性与 `from_json` 回读。
- `test_validation.py`：输入校验（NaN、长度不一致、索引越界）。
- `test_determinism.py`：在 `OMP_NUM_THREADS=1` 下的确定性。
- `test_public_data.py`：公开数据一致性（行列、无 NaN）。
- `compare_public_data.{R,py}`：R/Python 统计量一致性。

## R 与 Python 统计对比
若已安装 R，可生成参考统计并对比：
```bash
Rscript python/tests/compare_public_data.R
PYTHONPATH=python/.venv python3 python/tests/compare_public_data.py
```

## 数据说明
公开数据目录：
`python/tests/data/P-Tree-Public-Data-main`
