# 深大开源模型 -> Sc_vision

在尽可能不修改Sc_vision原处理逻辑的情况下加入深大开源的模型

---
## 环境配置要求

### 系统与依赖
- **操作系统**：Ubuntu 20.04 / 22.04 LTS
- **C++ 标准**：C++17 及以上
- **CMake**：>= 3.10
- **OpenCV**：>= 4.5.0
- **Intel OpenVINO Toolkit**：>= 2024.0 (推荐 2024.x 或 2025.x LTS)

### OpenVINO 安装配置 (Ubuntu APT 快速安装)

```bash
# 1. 安装 GPG 密钥与 Intel APT 源
sudo apt update
sudo apt install -y gnupg curl
curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor -o /usr/share/keyrings/intel-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/intel-archive-keyring.gpg] https://apt.repos.intel.com/openvino/2025 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2025.list

# 2. 安装 OpenVINO Runtime
sudo apt update
sudo apt install -y openvino-2025

# 3. 加载 OpenVINO 环境变量 (可写入 ~/.bashrc 或 ~/.zshrc)
source /opt/intel/openvino_2025/setupvars.sh
```

---

## 基本说明
1.保留了‘fivepoints_infer’,以便独立测试模型效果。
具体使用方法详见[深大原仓库README](https://github.com/SZURPVision/RuneDetectionModel)。
原仓库涉及到的模型类型均已保留，不需要重复转换。

2.我没保留模型预热部分，后续看看对处理的影响大不大

3.保留了auto_buff的测试程序，可以用来做测试

4.我还没编译过，所以不知道还有什么bug,之后会检查和修掉，也望能有有志之士共勉

---

## 致谢

深圳大学RobotPilots战队