# H-RGP

本仓库提供 H-RGP 主模型、五个对比模型及相应训练和评估代码。三个预见期为 1、3、7 天，五个 seed 为 0、1、2、3、4。

## 下载与数据

在仓库页面点击 **Code → Download ZIP**，解压后进入代码目录。也可复制 **Code** 菜单中的仓库地址，用 Git 克隆。

GitHub 下载包包含代码和运行配置，不包含水文中心的流量、雨量站、水位、入库与出库观测，不包含由这些资料生成的模型输入或逐日预测文件。使用这些资料须取得数据提供方许可，仓库没有实测数据下载地址。

ERA5-Land、Copernicus GLO-30 DEM 和 SoilGrids 可从各自官方平台获取，链接见 [数据说明](DATA.md)。公开数据需要结合经授权的实测资料、预处理和特征构建才能形成论文使用的输入；本仓库不提供从公开原始产品自动重建全部输入的程序。输入格式见 [输入文件说明](docs/INPUT_FORMAT.md)。

## 使用顺序

安装适合本机 CPU/CUDA 的 PyTorch，并使用 Python 3.10 安装依赖。授权输入放在本地 `data` 目录；预检、训练和结果目录均不应上传。

```bash
python -m pip install -r requirements.txt
python run.py --action verify
python run.py --action plan --data data --output runs
python run.py --action check-data --data data
python run.py --action preflight --data data --output preflight --seeds 0 --leads 1 3 7 --device cpu
python run.py --action train --data data --output runs --device cuda
python run.py --action evaluate --evaluate-test --data data --output runs --device cuda
python summarize.py --runs runs --split test --output summary
```

没有研究数据也能校验代码和查看任务计划。训练、前向/反向预检及数值评价需要实际输入文件。默认不启动训练，评价测试日期需要显式加 `--evaluate-test`。

五个基线为 LSTM、无库情 LSTM、MC-LSTM、可微 HBV 和标准 HBV。标准 HBV 每个 seed 率定一次，再按三个预见期的日期评价连续模拟结果。LSTM 的 15/90 天历史窗口按验证集 NSE 选择。指标按全部五个 seed 汇总为均值和样本标准差。

本实验使用对应历史日期的已知降水，H-RGP 还使用有效入库观测作为辅助训练标签。评价期曾用于方法选择，不能视为完全独立且未参与开发的测试样本。代码采用 [MIT 许可证](LICENSE)；数据共享仍遵循提供方的授权条件。
