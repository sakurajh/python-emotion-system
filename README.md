<div align="center">

# 🎓 情绪识别教学系统

**FastAPI 全栈教学平台 —— 摄像头实时情绪识别 · 在线考试 · 学情画像 · AI 助手**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.x-4479A1?style=flat-square&logo=mysql&logoColor=white)](https://www.mysql.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![AI](https://img.shields.io/badge/LLM-豆包·火山方舟-00C4B4?style=flat-square)](https://www.volcengine.com/)

</div>

---

## ✨ 功能模块

### 😊 实时情绪识别
- 浏览器摄像头视频流实时推送到后端（`/video_feed`）
- OpenCV 做人脸检测与表情分析，情绪数据实时回传前端图表（`/api/get_data`）
- 课堂情绪分布可视化，辅助教师了解学生状态

### 📝 在线考试系统
- **教师端**：可视化出题（`/teacher/exam_builder`）、发布考试、查看成绩
- **试卷导入**：支持上传 Word / PDF 自动解析题目（PyMuPDF + RapidOCR + python-docx）
- **学生端**：在线答题、提交（`/student/exam/{exam_id}`）、查看成绩
- 考试列表 / 详情 / 修改 / 删除全流程 API

### 📊 学情画像
- 按学生检索学习数据（`/api/profiler/search_user`）
- AI 生成个性化学习报告（`/api/profiler/generate_report`）
- 教师端班级整体分析（`/api/teacher/analyze_class`）

### 🤖 AI 助手
- 内置对话助手（`/api/chat`），接入火山方舟 · 豆包大模型
- 支持多轮对话，可用于答疑与学习引导

### 🛡️ 用户与后台
- 学生 / 教师 / 管理员 三角色
- 注册登录（密码 SHA-256 加盐哈希存储）
- 管理后台：用户管理、行为分析、平台统计（`/admin`、`/api/admin/*`）

## 🛠 技术栈

| 层次 | 技术 |
|---|---|
| Web 框架 | FastAPI + Jinja2 模板 |
| 数据库 | MySQL + SQLAlchemy ORM |
| 计算机视觉 | OpenCV（摄像头采集 / 人脸 / 情绪） |
| 文档解析 | PyMuPDF（PDF）、python-docx（Word）、RapidOCR（OCR 识别） |
| 大模型 | 火山方舟豆包（OpenAI 兼容接口） |
| 前端 | HTML + CSS + JS 模板渲染，实时图表 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

项目**不再在代码里硬编码任何密钥**，运行前请设置以下环境变量：

| 变量 | 说明 | 必填 |
|---|---|---|
| `ARK_API_KEY` | 火山方舟 API Key | AI 功能必填 |
| `ARK_ENDPOINT_ID` | 推理接入点 ID | AI 功能必填 |
| `DB_USER` | MySQL 用户名 | ✅ |
| `DB_PASS` | MySQL 密码 | ✅ |
| `DB_HOST` | MySQL 主机（默认 `localhost`） | 可选 |
| `DB_PORT` | MySQL 端口（默认 `3306`） | 可选 |

**Windows（CMD / PowerShell）**

```bat
set ARK_API_KEY=你的Key
set ARK_ENDPOINT_ID=你的接入点ID
set DB_USER=root
set DB_PASS=你的密码
set DB_HOST=127.0.0.1
set DB_PORT=3306
```

**Linux / macOS**

```bash
export ARK_API_KEY=你的Key
export ARK_ENDPOINT_ID=你的接入点ID
export DB_USER=root DB_PASS=你的密码
```

> 参考 [.env.example](.env.example)（仅作说明用途，代码直接读取系统环境变量）。

### 3. 启动

```bash
python main.py
```

浏览器打开 `http://127.0.0.1:8000` 即可。

## 📁 项目结构

```text
python-emotion-system/
├── main.py             # FastAPI 主程序（路由 / 模型 / 业务逻辑）
├── requirements.txt    # 依赖清单
├── templates/          # Jinja2 页面模板
├── static/             # 前端静态资源（CSS / JS / 图表）
└── .env.example        # 环境变量示例（不含真实密钥）
```

## 🔌 API 一览（节选）

| 分类 | 接口 |
|---|---|
| 认证 | `POST /api/register` · `POST /api/login` · `POST /api/teacher/register` |
| 情绪识别 | `GET /video_feed` · `GET /api/get_data` |
| 考试 | `POST /api/exam/publish` · `POST /api/exam/submit` · `GET /api/exam/results/{id}` |
| 学情 | `GET /api/profiler/search_user` · `POST /api/profiler/generate_report` |
| AI | `POST /api/chat` |
| 管理 | `GET /api/admin/stats` · `GET /api/admin/users` · `GET /api/admin/user_detail/{name}` |

## ⚠️ 安全提示

- 项目曾因早期版本在代码中硬编码密钥而存在泄露风险，**已改为环境变量读取**；如你部署了旧版本，请务必在控制台**更换 API Key 与数据库密码**
- 生产部署建议加 HTTPS、限制上传文件类型与大小、给管理接口加权限校验

## 📄 License

MIT
