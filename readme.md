## 环境配置

```bash
conda create -n anomal_test python=3.13
conda activate anomal_test
pip3 install torch torchvision # 请替换为对应版本的命令，查询链接附后
pip3 install anomalib
anomalib install -v
```
[PyTorch安装命令查询](https://pytorch.org/get-started/locally/)

## 脚本使用方法

### train_single.py

简介：单模型训练

参数：
- name: 训练模型名称
- input: 训练集数据目录（仅正样本）
- error: 可选，验证集数据目录（仅负样本）

示例：`python -u train_single.py --name AMD_R5 --input datasets/AMD-INTEL/Sub_AMD/R5`

备注：
- 模型保存在当前目录下的weights目录中，以模型名称为子目录名。
- error目录中的图像最好与训练图像相似但不同类，例如AMD_R5与AMD_R9、INTEL_Ultra5与INTEL_Ultra7，可避免模型过于敏感，提高预测精度。


### train_batch.py

简介：多模型训练

参数：
- input: 训练集数据目录

示例：`python -u train_batch.py --input datasets/mixed`

备注：
- 自动检测input目录下的第一层子目录，将其目录名作为模型名，以其中图片作为训练数据。
- 请不要在input目录下嵌套其他目录层级，例如dataset/mixed目录下包含AMD_AIMAX、AMD_AIMAX+、AMD_R5等目录，这些目录下即为训练用图像。
- 脚本会自动采样其他目录中的图像生成error目录，训练完成后自动删除。
- 模型保存在当前目录下的weights目录中，以模型名称为子目录名。


### predict.py

简介：预测

参数：
- name: 使用模型名称
- input: 待预测数据目录
- output: 可选，结果保存目录，若不提供则默认为'./results'
- ckpt: 可选，模型文件路径，若不提供则默认为'./weights/{name}/model.ckpt'

示例：`python -u predict.py --name AMD_R5 --ckpt weights/AMD_R5/model.ckpt --input datasets/AMD-INTEL/Sub_AMD/R5_AI`

备注：
- 结果保存在output目录中，以模型名称为子目录名。