# CanvasXpress 本地化安装说明

## 📦 文件信息

已成功下载CanvasXpress JavaScript库到本地项目：

- **CSS文件**: `canvasXpress.css` (227 KB)
- **JS文件**: `canvasXpress.min.js` (3.5 MB)
- **演示页面**: `canvasxpress_demo.html`

## 🚀 使用方法

### 1. 在Django模板中引入CanvasXpress

```html
<!-- 在HTML head中引入CSS -->
<link rel="stylesheet" href="/static/canvasxpress/canvasXpress.css" type="text/css"/>

<!-- 在HTML body底部引入JavaScript -->
<script type="text/javascript" src="/static/canvasxpress/canvasXpress.min.js"></script>
```

### 2. 基础使用示例

```html
<canvas id="myChart" width="600" height="400"></canvas>

<script>
var data = {
    "y": {
        "vars": ["变量1", "变量2"],
        "smps": ["样本1", "样本2", "样本3"],
        "data": [[10, 20, 30], [35, 25, 15]]
    }
};

var config = {
    "graphType": "Bar",
    "title": "我的图表"
};

new CanvasXpress("myChart", data, config);
</script>
```

### 3. 在Django视图中的集成

```python
# views.py
def chart_view(request):
    context = {
        'chart_data': {
            "y": {
                "vars": ["A", "B", "C"],
                "smps": ["X", "Y", "Z"],
                "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            }
        }
    }
    return render(request, 'chart.html', context)
```

```html
<!-- chart.html -->
<canvas id="chart" width="600" height="400"></canvas>

<script>
const djangoData = {{ chart_data|safe }};
new CanvasXpress("chart", djangoData, {"graphType": "Heatmap"});
</script>
```

## 🎨 支持的图表类型

- **基础图表**: Bar, Line, Area, Scatter, Pie, Doughnut
- **统计图表**: Boxplot, Histogram, Density, Violin
- **高级图表**: Heatmap, Network, Sankey, Treemap, Radar
- **科学图表**: Genome, Correlation, Kaplan-Meier, Venn
- **3D图表**: Scatter3D, Surface, Contour

## 📊 NMR系统中的应用建议

1. **结果可视化**: 展示NMR预测结果的分布
2. **模型比较**: 对比不同模型的性能指标
3. **数据探索**: 交互式探索NMR数据集
4. **统计报告**: 生成包含图表的分析报告

## 🔗 相关链接

- 官方文档: https://canvasxpress.org/docs.html
- 示例代码: https://canvasxpress.org/examples.html
- API参考: https://canvasxpress.org/api.html

## ⚡ 性能优化

- 已压缩的.min.js文件适合生产环境
- 支持响应式设计，自动适配屏幕尺寸
- 可处理百万级数据点
- 内置缓存和优化机制

---

**安装日期**: 2026-01-07  
**版本**: 最新稳定版  
**状态**: ✅ 可用