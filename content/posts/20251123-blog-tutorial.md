---
title: "博客搭建指南"
date: 2025-11-23
tags: ["tutorial", "hugo", "papermod"]
katex: true
---

# 个人博客搭建指南
欢迎阅读本博客搭建指南！本文将指导你如何使用 Hugo 和 PaperMod 主题来创建一个漂亮且功能强大的个人博客。

## 🚀 一、安装 Hugo
```bash
brew install hugo
```

查看是否安装成功：
```bash
hugo version
```

## 📂 二、创建你的博客项目

```bash
hugo new site myblog
cd myblog
```

## 🎨 三、安装 PaperMod 主题
```bash
git init
git submodule add https://github.com/adityatelange/hugo-PaperMod themes/PaperMod
```
然后修改 `hugo.toml`：
```toml
theme = "PaperMod"
```

## 📝 四、最终配置
```toml
baseURL = "https://{{github_username}}.github.io/"
languageCode = "zh-cn"
title = "My Blog"
theme = "PaperMod"

[params]
  defaultTheme = "auto"
  showReadingTime = true
  showCodeCopyButtons = true
  showToc = true

[outputs]
  home = ["HTML", "RSS", "JSON"]
```

这个配置：
- 自动支持亮/暗模式
- 首页支持 RSS + JSON（给搜索引擎 & 搜索插件用）
- 显示阅读时间
- 代码块支持一键复制
- 文章目录支持

## ✍️ 五、创建你的第一篇文章
```bash
hugo new posts/hello-world.md
```
打开生成的： `content/posts/hello-world.md` 文件，编辑内容：
```markdown
---
title: "Hello World"
date: 2025-01-01
tags: ["tech"]
---

使用 PaperMod 的第一篇 Hugo 博文！

```python
print("code highlight works")
```

## 💻 六、本地预览
```bash
hugo server
```
打开浏览器访问 `http://localhost:1313`，不出意外的话此时博客已正常工作。

## 📝 七、添加公式支持
PaperMod 主题默认不支持 LaTeX 公式渲染，我们可以通过添加 KaTeX 来实现这一功能。下面是具体步骤：
### 1️⃣ 新增 extended_head.html

在 `layouts/partials/extended_head.html` 中新增引入katex，具体如下
```html
{{ if .Params.katex}}{{ partial "katex.html" . }}{{ end }}
```

### 2️⃣ 创建 `kaTex.html`

在 `layouts/partials` 目录下创建 `kaTex.html` 文件，并添加以下内容：

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css" integrity="sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X" crossorigin="anonymous">

<!-- The loading of KaTeX is deferred to speed up page rendering -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js" integrity="sha384-g7c+Jr9ZivxKLnZTDUhnkOnsh30B4H0rpLUpJ4jAIKs4fnJI+sEnkvrMWph2EDg4" crossorigin="anonymous"></script>

<!-- To automatically render math in text elements, include the auto-render extension: -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js" integrity="sha384-mll67QQFJfxn0IYznZYonOWZ644AWYC+Pt2cHqMaRhXVrursRwvLnLaebdGIlYNa" crossorigin="anonymous"
    onload="document.addEventListener('DOMContentLoaded', function() { renderMathInElement(document.body, { delimiters: [ {left: '$$', right: '$$', display: true}, {left: '\\(', right: '\\)', display: false}, {left: '$', right: '$', display: false} ] }); });"></script>

```

### 3️⃣ 新增内容页的参数配置
当我们基于参数 .Params.katex 启用插件时。我们可以在页面的 Front Matter 中设置 katex: true 。

```markdown
---

title: "How to Add LaTeX Support in Hugo"

...

katex: true

---
```

### 4️⃣ 使用公式
在文章中，你可以使用以下语法来编写数学公式：
- 行内公式：使用单个美元符号包裹，例如 `$E=mc^2$`，渲染效果$E=mc^2$。
- 块级公式：使用双美元符号包裹，例如：
```markdown
$$
\int_a^b f(x) \,dx = F(b) - F(a)
$$
```
渲染效果为：
$$
\int_a^b f(x) \,dx = F(b) - F(a)
$$

这样，博客就支持 LaTeX 数学公式的渲染了！

## 📦 八、部署到 GitHub Pages
我们将使用 GitHub Actions 来自动化部署 Hugo 博客到 GitHub Pages。这样每次写好Markdown文章并推送到 GitHub 后，GitHub Actions 会自动构建并部署博客。操作步骤如下：

### 1️⃣ 创建 GitHub 仓库
创建一个新的 GitHub 仓库，命名为 `{{github_username}}.github.io`。

### 2️⃣ 配置 GitHub 仓库
1. 打开github仓库，进入 `Settings` -> `Pages`，选择 `Source` 为 `GitHub Actions`。

2. 打开 `Settings` -> `Actions` -> `General`，更改workflow权限设置为 `Read and write permissions`，以允许 GitHub Actions 推送到 gh-pages 分支。

### 3️⃣ 添加 GitHub Actions 工作流
添加GitHub Action工作流文件 `.github/workflows/hugo.yml`：
```yaml
name: Deploy Hugo

on:
  push:
    branches:
      - main

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true
          fetch-depth: 0

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: 'latest'
          extended: true

      - name: Build
        run: hugo --minify

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./public
```
### 4️⃣ 推送代码到 GitHub
1. 关联你的本地仓库到 GitHub：
```bash
git remote add origin https://github.com/{{github_username}}/{{github_username}}.github.io.git
``` 
2. 提交并推送代码到 GitHub：
```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```
3. 等待 GitHub Actions 运行完成后，访问 `https://{{github_username}}.github.io` 查看你的博客！

## 🎉 九、写在最后

博客到此搭建完成，后期写好文章后，先在本地预览，确认无误后提交到 GitHub 即可自动部署更新。祝大家写作愉快！🎉

## 参考资料
- [Hugo 官方文档](https://gohugo.io/documentation/)
- [GitHub Pages 文档](https://docs.github.com/en/pages)
- [Github Actions 文档](https://docs.github.com/en/actions)
- [Hugo PaperMod 公式渲染](https://fisherdaddy.com/posts/hugo/)