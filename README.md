# 城市交通GPS轨迹聚类与出行模式分析平台

基于Python Dash框架 + Plotly + Mapbox + NumPy构建的城市交通GPS轨迹分析工具，支持从原始轨迹导入到模式挖掘的完整分析流程。

## ✨ 功能特性

### 数据导入与预处理
- CSV格式GPS轨迹导入，必需列：user_id / timestamp / longitude / latitude
- 自动检测时间格式并统一为UTC，可选列：speed / heading / mode
- 噪声过滤：速度>200km/h剔除；连续两点距离>5km且间隔<10秒剔除
- 轨迹分段：同用户按时间间隔>30分钟切割为独立Trip
- 轨迹平滑：滑动窗口均值滤波，窗口大小可调
- 自动统计：总用户数、轨迹段数、时间跨度、空间Bounding Box

### 停留点(Stay Point)检测
- 距离阈值D（默认200米）+ 时间阈值T（默认20分钟）双参数算法
- 自动区分真实停留和微停留(transit_stop，如等红灯，<2分钟不计入停留)
- 地图可视化：大圆标注，半径正比于停留时长，悬停显示时段

### 轨迹相似度计算
- **DTW (Dynamic Time Warping)**：动态时间规整，支持FastDTW近似加速
  - 规则：轨迹数>500条时强制启用FastDTW
- **Frechet距离**：离散Frechet，人遛狗距离，对空间形态敏感
- **LCSS (Longest Common SubSequence)**：空间ε+时间δ阈值，归一化相似度
  - 空间阈值默认：数据集平均点间距×3（自动设定）
- **EDR (Edit Distance on Real sequence)**：实序列编辑距离，插入/删除/替换
- 距离矩阵计算显示进度条，结果可视化为热力图

### 轨迹聚类分析
- **DBSCAN**：eps邻域半径 + min_samples最小邻域点数
  - 提供k-距离图辅助选参，拐点处自动推荐eps值
- **OPTICS**：DBSCAN改进版，输出可达性图(Reachability Plot)
- **谱聚类 Spectral Clustering**：
  - 距离矩阵→高斯核相似度→拉普拉斯矩阵→特征值分解→KMeans
  - 自动通过eigengap(特征值间隔)推荐最佳K值
- 自动计算**轮廓系数(Silhouette Score)**，<0.2时警告聚类质量差
- 结果：地图同簇同色渲染、噪声灰色标注、统计侧边栏

### 出行模式挖掘
- **OD分析**：500m×500m空间网格化，构建OD矩阵
  - 地图弧线OD流向图，线宽正比流量，显示前N高频OD对
- **通勤模式识别**：工作日早7-9点+晚17-19点高频OD>60%工作日
  - 输出用户、起终点坐标、置信度
- **频繁路径挖掘**：
  - GPS点→空间网格序列→前缀树Trie→统计频繁子序列
  - 网格尺寸自动根据采样间隔推荐（避免过细/过粗）
- **时段分布**：出行小时(0-23)柱状图、星期分布(一-日)、时长分布直方图

### 热点区域与异常检测
- **热点发现**：全量GPS点DBSCAN空间聚类（eps=200m, min_samples=50）
  - 凸包边界多边形+半透明填充，统计：中心/面积/密度/用户数/高峰时
  - 支持热力图(Heatmap)模式：网格化密度渐变色渲染
- **异常轨迹检测**：
  - DBSCAN标记的噪声轨迹
  - 偏离簇平均路径>2倍标准差
  - 同OD对出行时长>3倍平均值(绕路)
  - 异常红色高亮，侧边栏分类筛选(绕路/偏离/噪声)

### 地图可视化交互
- 底图切换：路网图/卫星图/浅色(Carto Positron)/深色(Carto Darkmatter)
- 轨迹动画回放：点击轨迹后按时间顺序播放，速度(1x/2x/5x/10x)可调
- 时间范围滑块：只显示指定时间窗口内的轨迹
- 空间框选：Plotly工具栏支持矩形选择（drawrect）
- 图层独立开关：轨迹层/停留点层/热力层/OD流向层/热点层/异常层
- 限制：一次最多渲染2000条，超出提示筛选/抽样

### 统计Dashboard & 导出
- 核心指标：平均出行距离(km)、平均时长(min)、日均次数、轮廓系数
- 统计图：时段分布、星期分布、距离分布、聚类占比饼图、交通模式占比
- 所有图表支持Plotly工具栏PNG导出
- CSV报表导出：轨迹统计、停留点、聚类结果、OD矩阵、热点区域数据
- 独立底部Dashboard，可折叠

## 🚀 快速开始

### 环境要求
- Python ≥ 3.9
- pip

### 安装依赖
```bash
cd traj-cluster
pip install -r requirements.txt
```

### 启动应用
```bash
python app.py
```

打开浏览器访问： http://127.0.0.1:8050

### 使用流程
1. **导入数据**：左侧"📊 数据"Tab
   - 点击"🎲 生成示例数据"快速体验（20用户×5工作日）
   - 或拖拽/选择CSV文件（必需列：user_id, timestamp, longitude, latitude）
   - 调整分段阈值、平滑参数 → 点"▶ 执行预处理"

2. **停留点检测**：切换到"📍 停留点"Tab
   - 设置距离阈值D(米)、时间阈值T(分钟)
   - 点"▶ 检测停留点" → 查看统计和列表

3. **相似度计算**：切换到"📐 相似度"Tab
   - 选择度量方法（DTW推荐，轨迹多选FastDTW加速）
   - 选择轨迹范围（建议先sample100体验）
   - 点"▶ 计算距离矩阵" → 查看热力图

4. **聚类分析**：切换到"🎯 聚类"Tab
   - 选择算法（DBSCAN推荐开始，查看k-距离图辅助选参）
   - 点"▶ 执行聚类" → 查看轮廓系数、图例、地图彩色渲染

5. **模式挖掘**：切换到"🔍 模式挖掘"Tab
   - 设置网格大小（留空自动推荐）、最小支持度
   - 点"▶ 挖掘出行模式"
   - 在右侧图层勾选"OD流向"、"热力图"、"热点"、"异常"查看可视化

6. **统计与导出**：右侧"📊 统计"Tab查看图表
   - "💾 导出"Tab下载CSV报表和PNG图片

## 📁 项目结构
```
traj-cluster/
├── app.py                          # Dash应用主入口
├── requirements.txt                # 依赖清单
├── assets/
│   ├── __init__.py
│   └── custom.css                  # 自定义样式
├── core/                           # 核心算法模块
│   ├── __init__.py
│   ├── utils.py                    # Haversine距离、平滑等工具
│   ├── preprocessing.py            # CSV导入、去噪、分段、Trip类
│   ├── stay_points.py              # 停留点检测算法
│   ├── similarity.py               # DTW/FastDTW/Frechet/LCSS/EDR
│   ├── clustering.py               # DBSCAN/OPTICS/谱聚类+轮廓系数
│   ├── pattern_mining.py           # OD/通勤/频繁路径/时空分布
│   ├── hotspot_anomaly.py          # 热点DBSCAN+异常轨迹识别
│   ├── serialization.py            # Trip↔Dict序列化(用于dcc.Store)
│   ├── sample_generator.py         # 示例轨迹数据生成器
│   └── visualization.py            # 地图渲染+统计图表
├── callbacks/                      # Dash交互回调
│   ├── __init__.py
│   ├── data_callbacks.py           # 数据面板、参数面板显隐、Tab切换
│   ├── analysis_callbacks.py       # 停留点/相似度/聚类/模式挖掘流程
│   └── dashboard_callbacks.py      # 地图/时间/动画/图表/导出
├── layouts/
│   ├── __init__.py
│   └── app_layout.py               # 整体UI布局定义
└── exports/                        # 自动创建的导出目录
```

## 📋 业务规则实现清单

| # | 规则 | 实现位置 |
|---|------|----------|
| 1 | 轨迹数>500时强制使用FastDTW | `core/similarity.py:compute_distance_matrix` |
| 2 | 地图最多渲染2000条轨迹，超出抽样 | `core/visualization.py:add_trajectories_to_map` |
| 3 | LCSS/EDR空间阈值=平均点间距×3 | `core/similarity.py:compute_avg_point_spacing` |
| 4 | 停留时长<T的微停留标记transit_stop | `core/stay_points.py:detect_stay_points_for_trip` |
| 5 | 频繁路径网格尺寸自动匹配采样间隔 | `core/pattern_mining.py:recommend_grid_size` |
| 6 | 聚类完成计算轮廓系数，<0.2警告 | `core/clustering.py:_make_result` |

## ⚠️ 性能说明
- DTW是O(n²·m²)复杂度，大量轨迹请务必使用**FastDTW+抽样**
- 建议先在**100条以内**样本调参，确认参数合理后再扩大规模
- 聚类、频繁路径计算量随数据量上升较快，建议分批处理
- Mapbox底图无需Token（使用了开源样式），如需自定义可自行配置

## 📦 核心依赖
| 库 | 用途 |
|----|------|
| Dash | 交互式Web框架 |
| Plotly | 可视化图表、Mapbox地图 |
| NumPy / Pandas | 向量化计算、数据处理 |
| scikit-learn | DBSCAN/OPTICS/谱聚类/轮廓系数 |
| SciPy | 拉普拉斯矩阵、特征值分解 |
| GeoPy / Shapely | Haversine距离、凸包计算 |
| Kaleido | 图表PNG导出 |

## 🔧 CSV数据格式
```csv
user_id,timestamp,longitude,latitude,speed,heading,mode
user_001,2024-06-03 07:23:00,121.4785,31.2356,35.2,78.4,car
user_001,2024-06-03 07:23:30,121.4792,31.2361,36.8,80.1,car
...
```

---
© Trajectory Clustering Platform
