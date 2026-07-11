# Travel Planning Assistant

一个面向中文旅行场景的智能旅游规划助手，支持多轮对话、行程规划、交通方式比较、天气查询、酒店推荐、地点检索，以及语音/图片等多模态输入。

项目包含：

- `backend/`：FastAPI 后端，负责意图识别、状态记忆、工具调用、回答生成
- `frontend/`：React + Vite 前端，提供接近 ChatGPT 的对话式交互界面

## 核心能力

- 多轮旅行对话与上下文记忆
- 行程规划：按目的地、天数、预算、人数、偏好生成建议
- 交通比较：如高铁 / 飞机 / 自驾对比
- 天气查询与天气驱动的行程调整
- 酒店、景点、餐饮候选推荐
- 小红书旅行经验整合
- 语音输入与图片理解
- 结构化调试面板，便于观察意图、规划、工具结果和状态

## 项目结构

```text
TravelPlanningAssistant/
├─ backend/
│  ├─ app/
│  │  ├─ api/          # API 路由
│  │  ├─ core/         # 配置管理
│  │  ├─ schemas/      # 请求/响应模型
│  │  ├─ services/     # Agent、LLM、记忆、旅行编排
│  │  ├─ tools/        # 地图、天气、搜索、酒店等工具封装
│  │  └─ rag/          # 检索增强相关模块
│  ├─ .env.example
│  └─ requirements.txt
├─ frontend/
│  ├─ src/
│  │  ├─ services/     # 前端 API 调用
│  │  ├─ styles/       # 样式
│  │  ├─ App.jsx
│  │  └─ main.jsx
│  ├─ .env.example
│  └─ package.json
├─ requirements.txt
└─ README.md
```

## 技术栈

### 后端

- FastAPI
- Pydantic / pydantic-settings
- httpx
- Redis（可选，用于记忆）
- SQLAlchemy / PyMySQL（部分扩展能力）

### 前端

- React 18
- Vite

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd TravelPlanningAssistant
```

### 2. 配置 Python 环境

建议使用虚拟环境：

```bash
python -m venv .venv
```

Windows 激活：

```bash
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果你只想安装后端依赖，也可以使用：

```bash
pip install -r backend/requirements.txt
```

### 3. 配置后端环境变量

请基于 `backend/.env.example` 或自行新建根目录 `.env` 文件。

推荐至少配置：

```env
QWEN_API_KEY=your_qwen_api_key
TENCENT_MAP_API_KEY=your_tencent_map_key
WEATHER_API_KEY=your_weather_key
SEARCH_API_KEY=your_search_key
JUSTONEAPI_TOKEN=your_justoneapi_token
REDIS_URL=redis://127.0.0.1:6379/0
```

说明：

- 项目配置默认从根目录 `.env` 读取
- 不要把真实密钥提交到 GitHub
- 可以保留 `.env.example` 作为示例模板

### 4. 启动后端

推荐进入 `backend` 目录启动：

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

如果你想在项目根目录启动，也可以使用：

```bash
uvicorn app.main:app --reload --app-dir backend --host 0.0.0.0 --port 8000
```

后端健康检查：

- `GET http://localhost:8000/health`
- `GET http://localhost:8000/api/v1/health`

### 5. 配置前端环境变量

在 `frontend/` 下创建 `.env`，参考：

```env
VITE_API_BASE_URL=http://localhost:8000
```

或者直接参考：

- `frontend/.env.example`

### 6. 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认前端开发地址通常为：

- `http://localhost:5173`

## API 简介

### 文本对话

`POST /api/v1/chat`

示例请求：

```json
{
  "session_id": "default",
  "message": "帮我规划杭州3天2晚，预算5000，喜欢自然风景"
}
```

### 多模态对话

`POST /api/v1/chat/multimodal`

支持：

- 文本
- 图片文件
- 音频文件

## 使用建议

你可以直接输入类似下面的内容：

- `帮我规划杭州 3 天 2 晚，预算 5000，喜欢自然风景`
- `从济南去北京玩 3 天，偏好历史建筑，预算 3000`
- `想去成都吃美食和逛博物馆，2 人 4 天怎么安排`
- `比较一下去上海旅游坐高铁还是飞机更合适`

为了得到更准确的结果，建议尽量提供：

- 出发地
- 目的地
- 天数
- 人数
- 预算
- 日期
- 偏好（如自然 / 人文 / 美食 / 亲子 / 拍照）

## 上传到 GitHub 前的安全建议

### 1. 不要上传这些文件

请确保以下内容不会被提交：

- `.env`
- `backend/.env`
- `frontend/.env`
- 各类本地密钥文件
- 虚拟环境目录 `.venv/`
- `node_modules/`
- 上传文件目录、缓存目录、日志文件

### 2. 只保留示例配置

建议保留：

- `backend/.env.example`
- `frontend/.env.example`

并把里面的值写成占位符，例如：

```env
QWEN_API_KEY=your_qwen_api_key
TENCENT_MAP_API_KEY=your_tencent_map_key
```

### 3. 如果你已经把密钥写进代码或 `.env`

在上传前请立即处理：

- 把真实密钥从代码和配置文件中删掉
- 改成环境变量读取
- 重新生成这些密钥（如果它们曾经泄露过）

### 4. 推荐检查方式

在提交前，手动检查这些文件是否包含真实信息：

- 根目录 `.env`
- `backend/.env`
- `frontend/.env`
- `backend/app/core/config.py` 对应的实际配置来源
- 任何包含 `key` / `token` / `secret` / `password` 的文件

## 当前仓库更适合公开展示的内容

适合上传到 GitHub 的是：

- 源代码
- `.env.example`
- `README.md`
- `.gitignore`
- 不含真实数据的示例配置

不建议上传：

- 真实 API Key
- 真实 Redis / MySQL 凭据
- 本地测试上传的图片或音频
- 大型缓存、索引和虚拟环境

## 后续可继续完善

- 增加 Docker 部署说明
- 增加单元测试与接口测试
- 增加数据库初始化脚本
- 增加更完整的系统架构图
- 增加线上演示地址和截图

## License

如果你准备公开发布，建议补充许可证，例如 MIT。
目前如未额外声明，可在上传前自行添加 `LICENSE` 文件。
