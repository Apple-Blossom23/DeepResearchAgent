#!/usr/bin/env bash
# 生产环境启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${RED}🚀 启动生产环境...${NC}"

# 设置环境变量
export ENV="prod"

# 显示当前环境
echo -e "${YELLOW}📋 当前环境: ${ENV}${NC}"

# 启动应用
echo -e "${CYAN}▶️  启动应用...${NC}"
python run.py

# 清理环境变量（可选）
# unset ENV


