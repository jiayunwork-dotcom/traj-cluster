import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import PIL.Image
except Exception:
    pass

import dash
from dash import html
import dash_bootstrap_components as dbc

from layouts.app_layout import create_app_layout
from callbacks.data_callbacks import register_data_callbacks
from callbacks.analysis_callbacks import register_analysis_callbacks
from callbacks.dashboard_callbacks import register_map_callbacks, register_dashboard_callbacks


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True,
    title="城市交通GPS轨迹聚类分析平台",
)

app.layout = create_app_layout()

register_data_callbacks(app)
register_analysis_callbacks(app)
register_map_callbacks(app)
register_dashboard_callbacks(app)

server = app.server


def print_banner():
    print()
    print("=" * 70)
    print("  🚗 城市交通GPS轨迹聚类与出行模式分析平台")
    print("  Trajectory Clustering & Pattern Mining System")
    print("=" * 70)
    print()
    print("  功能模块:")
    print("  ✅ 数据导入与预处理 (CSV导入/噪声过滤/轨迹分段/平滑)")
    print("  ✅ 停留点检测 (距离D+时间T双阈值, 区分微停留)")
    print("  ✅ 轨迹相似度计算 (DTW/FastDTW/Frechet/LCSS/EDR)")
    print("  ✅ 轨迹聚类 (DBSCAN+eps推荐/OPTICS+可达性图/谱聚类+eigengap)")
    print("  ✅ 出行模式挖掘 (OD矩阵/通勤识别/频繁路径/时空分布)")
    print("  ✅ 热点区域发现 (空间DBSCAN/凸包可视化/热力图)")
    print("  ✅ 异常轨迹检测 (DBSCAN噪声/路径偏离/绕路异常)")
    print("  ✅ 交互式地图 (底图切换/动画回放/图层控制/时间滑块)")
    print("  ✅ 统计Dashboard & 结果导出 (CSV/PNG)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    print_banner()
    port = int(os.environ.get("PORT", 8050))
    print(f"  🌐 启动中... 请在浏览器打开: http://127.0.0.1:{port}")
    print("=" * 70)
    print()
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )
