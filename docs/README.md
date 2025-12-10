## 🚀 Installation

```bash

uv venv myenv --python 3.12 --seed
source myenv/bin/activate

# 步骤1：进入docs目录
cd docs

# 步骤2：安装依赖（使用uv）
uv pip install -r requirements-docs.txt

# 安装 sphinx-autobuild（如果没在 requirements 文件里）
uv pip install sphinx-autobuild

# 从 docs 目录运行：
sphinx-autobuild ./source ./_build/html --port 8000

# 步骤1：清理旧文件
make clean

# 步骤2：构建HTML
make html

# 步骤3：本地预览
python -m http.server -d _build/html/

浏览器访问：http://localhost:8000

🌍 Internationalization
国际化翻译流程（以中文为例）

# 步骤1：提取可翻译文本（生成 .pot）
sphinx-build -b gettext source _build/gettext

# 步骤2：生成/更新中文 .po 文件
sphinx-intl update -p _build/gettext -l zh_CN

# 步骤3：人工翻译 .po 文件
# 用文本编辑器打开 source/locale/zh_CN/LC_MESSAGES/*.po
# 在 msgstr "" 里填入中文翻译

# 步骤4：编译并构建中文文档
make intl

# 步骤5：查看效果
python -m http.server -d _build/html


浏览器访问：

英文版： http://localhost:8000
中文版： http://localhost:8000/zh-cn

```
