"""
进化记录文档
============
记录策略进化过程、发现、决策依据
"""

from pathlib import Path

DOC_PATH = Path("output/evolution_log.md")

def log_evolution(version, title, findings, config, metrics):
    """记录进化版本"""
    content = f"""
# 策略进化 v{version}: {title}
## 发现
{findings}

## 配置
- 选股: {config.get('select', 'N/A')}
- 持仓: {config.get('top_n', 'N/A')}
- 周期: {config.get('hold_days', 'N/A')}
- 过滤: {config.get('filter', '无')}

## 绩效
- 年化: {metrics.get('annual', 'N/A')}
- 夏普: {metrics.get('sharpe', 'N/A')}
- 胜率: {metrics.get('win_rate', 'N/A')}
- 回撤: {metrics.get('max_drawdown', 'N/A')}

---
"""
    with open(DOC_PATH, 'a', encoding='utf-8') as f:
        f.write(content)

def init_log():
    """初始化文档"""
    header = """# 科创板智能选股策略 - 进化记录

## 进化历程

| 版本 | 日期 | 策略 | 年化 | 夏普 | 胜率 |
|------|------|------|------|------|------|
"""
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.write(header)