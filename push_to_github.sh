#!/bin/bash
# 推送代码到 GitHub 的完整脚本

# 1. 首先在 GitHub 上创建仓库:
#    访问 https://github.com/new
#    仓库名: llm-benchmark
#    描述: A high-performance load testing tool for self-deployed large language models
#    选择 Public
#    不要初始化任何文件

# 2. 设置远程仓库地址（请将 yourusername 替换为你的 GitHub 用户名）
echo "请输入你的 GitHub 用户名:"
read username

git remote add origin https://github.com/$username/llm-benchmark.git

# 3. 推送代码
git push -u origin master

echo "推送完成！访问 https://github.com/$username/llm-benchmark 查看你的仓库"