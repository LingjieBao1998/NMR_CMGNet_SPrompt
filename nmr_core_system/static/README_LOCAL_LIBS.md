# 本地化前端依赖库

## 📦 已下载的库文件

### 1. Bootstrap 5.1.3
- **文件**: `bootstrap/bootstrap.min.css` (160 KB)
- **文件**: `bootstrap/bootstrap.bundle.min.js` (76 KB)
- **用途**: 前端框架和样式
- **原地址**: https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/

### 2. jQuery 3.6.0
- **文件**: `js/jquery.min.js` (87 KB)
- **用途**: JavaScript库依赖
- **原地址**: https://code.jquery.com/jquery-3.6.0.min.js

### 3. DataTables 1.13.6
- **文件**: `datatables/jquery.dataTables.min.js` (85 KB)
- **文件**: `datatables/dataTables.bootstrap5.min.css` (12 KB)
- **文件**: `datatables/dataTables.bootstrap5.min.js` (2.4 KB)
- **用途**: 表格插件，提供搜索、排序、分页功能
- **原地址**: https://cdn.datatables.net/1.13.6/

### 4. CanvasXpress
- **文件**: `canvasxpress/canvasXpress.css` (222 KB)
- **文件**: `canvasxpress/canvasXpress.min.js` (3.5 MB)
- **用途**: 数据可视化图表库
- **原地址**: https://www.canvasxpress.org/dist/

### 5. Plotly.js 2.27.0
- **文件**: `plotly/plotly.min.js` (3.5 MB)
- **用途**: 交互式数据可视化图表库
- **原地址**: https://cdn.jsdelivr.net/npm/plotly.js@2.27.0/dist/plotly.min.js
- **下载日期**: 2026-01-07

## 🔧 在模板中的使用方式

### 替换CDN链接为本地链接

**之前 (CDN)**:
```html
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
```

**现在 (本地)**:
```html
<link href="/static/bootstrap/bootstrap.min.css">
<script src="/static/datatables/jquery.dataTables.min.js"></script>
```

### 完整的本地依赖引用

```html
<!-- CSS -->
<link rel="stylesheet" href="/static/bootstrap/bootstrap.min.css">
<link rel="stylesheet" href="/static/datatables/dataTables.bootstrap5.min.css">
<link rel="stylesheet" href="/static/canvasxpress/canvasXpress.css">

<!-- JavaScript -->
<script src="/static/js/jquery.min.js"></script>
<script src="/static/datatables/jquery.dataTables.min.js"></script>
<script src="/static/datatables/dataTables.bootstrap5.min.js"></script>
<script src="/static/bootstrap/bootstrap.bundle.min.js"></script>
<script src="/static/canvasxpress/canvasXpress.min.js"></script>
<script src="/static/plotly/plotly.min.js"></script>
```

### Plotly.js 使用示例

```html
<!-- 引入Plotly.js -->
<script src="/static/plotly/plotly.min.js"></script>

<!-- 创建图表容器 -->
<div id="myChart" style="width:100%;height:400px;"></div>

<script>
// 创建简单的折线图
const trace1 = {
  x: [1, 2, 3, 4],
  y: [10, 15, 13, 17],
  type: 'scatter'
};

const data = [trace1];
Plotly.newPlot('myChart', data);
</script>
```

## ✅ 优势

1. **离线可用**: 不依赖外部CDN，提高系统稳定性
2. **加载速度**: 本地文件加载更快，减少网络延迟
3. **安全性**: 避免第三方CDN的潜在安全风险
4. **一致性**: 确保所有用户使用相同版本的库
5. **稳定性**: 避免外部CDN服务中断影响系统功能

## 📋 维护说明

- **更新日期**: 2026-01-07
- **版本**: 最新稳定版
- **状态**: ✅ 可用

如需更新这些库，请从官方网站下载最新版本并替换相应文件。

---

**总大小**: ~7.7 MB (未压缩)
**节省**: 避免了多个外部HTTP请求，提高系统稳定性和加载速度