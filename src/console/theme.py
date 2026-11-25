"""总控界面主题：深色车载风格颜色与字体。"""

# 背景层级
BG_ROOT = "#12151a"
BG_CARD = "#1c2128"
BG_CARD_HOVER = "#262c36"
BG_CARD_ACTIVE = "#2d3748"
BG_PANEL = "#161a20"

# 文字
FG_PRIMARY = "#e6edf3"
FG_SECONDARY = "#8b949e"
FG_MUTED = "#6e7681"

# 强调色
ACCENT_BLUE = "#4a9eff"
ACCENT_GREEN = "#3fb950"
ACCENT_YELLOW = "#d29922"
ACCENT_GRAY = "#6e7681"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#a371f7"

# 字体
FONT_TITLE = ("Microsoft YaHei", 14, "bold")
FONT_SUBTITLE = ("Microsoft YaHei", 11)
FONT_CARD_TITLE = ("Microsoft YaHei", 11, "bold")
FONT_CARD_STATUS = ("Microsoft YaHei", 9)
FONT_METRIC_LABEL = ("Microsoft YaHei", 9)
FONT_METRIC_VALUE = ("Microsoft YaHei", 14, "bold")
FONT_LOG = ("Consolas", 9)
FONT_HINT = ("Microsoft YaHei", 9)

# 卡片状态对应颜色
STATUS_COLORS = {
    "idle": ACCENT_GRAY,
    "starting": ACCENT_YELLOW,
    "running": ACCENT_GREEN,
    "stopping": ACCENT_YELLOW,
    "error": ACCENT_RED,
}

STATUS_LABELS = {
    "idle": "待机",
    "starting": "启动中",
    "running": "运行中",
    "stopping": "停止中",
    "error": "异常",
}