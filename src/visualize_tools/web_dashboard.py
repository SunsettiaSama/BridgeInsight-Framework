"""
VibDash — 基于标准库 HTTPServer + SSE 的轻量 Web 可视化仪表盘。

零额外依赖（跨进程 push() 需要 requests，已是常见库）。

用法
----
# 1. 独立服务器（推荐）：在一个终端启动，其他脚本任意时间注入
#    python -m src.visualize_tools.web_dashboard
#    python -m src.visualize_tools.web_dashboard --port 5679 --cols 2

# 2. 同进程启动 + 推送
from src.visualize_tools.web_dashboard import WebDashboard
dash = WebDashboard(port=5678, cols=3).start()
dash.push(fig_ts,   page="Sample_001", slot=0, title="时程",   page_cols=2)
dash.push(fig_psd,  page="Sample_001", slot=1, title="PSD")
dash.wait()   # 阻塞，Ctrl+C 退出

# 3. 跨进程推送（服务须已在运行）
from src.visualize_tools.web_dashboard import push
push(fig, page="Sample_002", slot=0, title="时程", port=5678, page_cols=2)

# 4. PlotLib 集成
ploter.show_web(page="fig4_22", cols=2)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import socketserver
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 内嵌 HTML 模板
# 占位符 COLS_VALUE 在运行时被替换为服务器默认列数
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VibDash</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --sidebar:230px;
  --bg:#0f1117;
  --surface:#1c1e26;
  --border:#2a2d3a;
  --accent:#4f8ef7;
  --new:#f7a94f;
  --text:#d0d4e0;
  --dim:#55596a;
  --btn-bg:#22253a;
  --btn-hover:#2e3350;
}
body{display:flex;height:100vh;background:var(--bg);color:var(--text);
  font-family:system-ui,sans-serif;font-size:13px;overflow:hidden}

/* ── 侧边栏 ── */
#sidebar{width:var(--sidebar);min-width:var(--sidebar);display:flex;
  flex-direction:column;border-right:1px solid var(--border);background:var(--surface)}
#sb-header{padding:14px 16px 10px;border-bottom:1px solid var(--border)}
#sb-title{font-size:16px;font-weight:700;letter-spacing:.4px;color:var(--accent)}
#conn{font-size:11px;color:var(--dim);margin-top:4px}
#sb-search{
  margin:8px 10px 4px;padding:5px 10px;
  background:var(--bg);border:1px solid var(--border);border-radius:5px;
  color:var(--text);font-size:12px;width:calc(100% - 20px);outline:none}
#sb-search::placeholder{color:var(--dim)}
#pages-list{flex:1;overflow-y:auto;padding:4px 0}

.page-item{
  padding:9px 16px;cursor:pointer;
  border-left:3px solid transparent;transition:all .12s;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  color:var(--dim)}
.page-item:hover{background:rgba(255,255,255,.04);color:var(--text)}
.page-item.active{
  border-left-color:var(--accent);color:var(--text);
  background:rgba(79,142,247,.10)}
.page-item.filtered{display:none}
.new-dot{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--new);margin-left:7px;vertical-align:middle;
  animation:blink 1s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* ── 主区域 ── */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#topbar{
  padding:8px 14px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:7px;flex-shrink:0;flex-wrap:wrap}

.nav-btn{
  background:var(--btn-bg);border:1px solid var(--border);
  color:var(--text);border-radius:5px;padding:4px 11px;
  cursor:pointer;font-size:12px;transition:background .12s;white-space:nowrap;
  user-select:none}
.nav-btn:hover:not(:disabled){background:var(--btn-hover)}
.nav-btn:disabled{opacity:.3;cursor:default}
.nav-btn.dl-btn{color:var(--new)}

#page-title{
  font-size:13px;font-weight:600;color:var(--text);
  flex:1;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  min-width:0}

.cols-group{display:flex;gap:3px}
.col-btn{
  background:var(--btn-bg);border:1px solid var(--border);
  color:var(--dim);border-radius:4px;padding:3px 9px;
  cursor:pointer;font-size:11px;transition:all .12s;user-select:none}
.col-btn:hover{background:var(--btn-hover);color:var(--text)}
.col-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}

#page-count{font-size:11px;color:var(--dim);white-space:nowrap}

#grid-wrap{flex:1;overflow-y:auto;padding:12px}
.grid{display:grid;gap:10px;align-content:start}

/* ── 格子 ── */
.cell{
  background:var(--surface);border:1px solid var(--border);
  border-radius:8px;overflow:hidden;transition:border-color .2s}
.cell:hover{border-color:var(--accent)}
.cell.empty{
  min-height:100px;display:flex;align-items:center;
  justify-content:center;color:var(--dim);font-size:11px}
.cell img{width:100%;display:block;cursor:zoom-in}
.cell .cell-label{
  padding:5px 10px;font-size:11px;color:var(--dim);
  border-top:1px solid var(--border)}

/* ── 空态提示 ── */
#empty-hint{
  display:flex;align-items:center;justify-content:center;
  height:100%;color:var(--dim);font-size:14px;
  flex-direction:column;gap:10px}
#empty-hint code{
  background:var(--border);padding:5px 12px;border-radius:5px;
  font-size:12px;color:var(--accent)}

/* ── 灯箱 ── */
#lightbox{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);
  align-items:center;justify-content:center;z-index:9999}
#lightbox.open{display:flex}
#lb-img{max-width:86vw;max-height:88vh;border-radius:6px;
  box-shadow:0 0 40px rgba(0,0,0,.7)}
.lb-arrow{
  position:absolute;top:50%;transform:translateY(-50%);
  background:rgba(255,255,255,.13);border:none;color:#fff;
  font-size:24px;padding:12px 17px;cursor:pointer;border-radius:6px;
  transition:background .15s;z-index:10;line-height:1}
.lb-arrow:hover{background:rgba(255,255,255,.28)}
#lb-prev{left:16px}
#lb-next{right:16px}
#lb-close{
  position:absolute;top:14px;right:16px;
  background:rgba(255,255,255,.13);border:none;color:#fff;
  font-size:16px;padding:6px 13px;cursor:pointer;border-radius:6px;
  transition:background .15s;z-index:10}
#lb-close:hover{background:rgba(255,255,255,.28)}
#lb-save{
  position:absolute;top:14px;right:78px;
  background:rgba(255,255,255,.13);border:none;color:#fff;
  font-size:16px;padding:6px 13px;cursor:pointer;border-radius:6px;
  transition:background .15s;z-index:10}
#lb-save:hover{background:rgba(255,255,255,.28)}
#lb-caption{
  position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
  background:rgba(0,0,0,.65);color:#ccc;padding:6px 18px;
  border-radius:20px;font-size:12px;white-space:nowrap;pointer-events:none}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sb-header">
    <div id="sb-title">VibDash</div>
    <div id="conn">● 连接中…</div>
  </div>
  <input id="sb-search" type="text" placeholder="搜索页面…"
         oninput="filterPages(this.value)">
  <div id="pages-list"></div>
</div>

<div id="main">
  <div id="topbar">
    <button class="nav-btn" id="btn-prev" onclick="prevPage()" disabled>&#8592; 上一页</button>
    <span id="page-title">—</span>
    <button class="nav-btn" id="btn-next" onclick="nextPage()" disabled>下一页 &#8594;</button>
    <button class="nav-btn dl-btn" onclick="downloadPage()">&#8595; 下载</button>
    <div class="cols-group">
      <button class="col-btn" onclick="setColsOverride(1)">1</button>
      <button class="col-btn" onclick="setColsOverride(2)">2</button>
      <button class="col-btn" onclick="setColsOverride(3)">3</button>
      <button class="col-btn" onclick="setColsOverride(4)">4</button>
    </div>
    <span id="page-count"></span>
  </div>
  <div id="grid-wrap">
    <div id="empty-hint">
      <span>等待图像注入…</span>
      <code>push(fig, page="name", slot=0)</code>
    </div>
    <div class="grid" id="grid" style="display:none"></div>
  </div>
</div>

<!-- 灯箱 -->
<div id="lightbox">
  <button class="lb-arrow" id="lb-prev" onclick="lbStep(-1)">&#8249;</button>
  <img id="lb-img" src="" alt="">
  <button class="lb-arrow" id="lb-next" onclick="lbStep(1)">&#8250;</button>
  <button id="lb-save" onclick="lbDownload()">&#8595; 保存</button>
  <button id="lb-close" onclick="closeLightbox()">&#10005;</button>
  <div id="lb-caption"></div>
</div>

<script>
const DEFAULT_COLS = COLS_VALUE;
let currentPage  = null;
let colsOverride = null;   // null = 使用页面自身 cols

const pagesList = [];      // newest-first 插入顺序
const pageCols  = {};      // {name: cols}
const pageSlots = {};      // {name: {slot: {image, title}}} — 本地缓存

let lbSlots = [];          // 灯箱当前页所有 slot
let lbIdx   = 0;

// ── 列数管理 ──────────────────────────────────────────────
function getActiveCols() {
  return colsOverride || pageCols[currentPage] || DEFAULT_COLS;
}
function applyPageCols(cols) {
  document.getElementById('grid').style.gridTemplateColumns =
    'repeat(' + cols + ', 1fr)';
  document.querySelectorAll('.col-btn').forEach(b =>
    b.classList.toggle('active', parseInt(b.textContent) === cols));
}
function setColsOverride(n) {
  colsOverride = n;
  applyPageCols(n);
}

// ── 格子管理 ──────────────────────────────────────────────
function ensureCells(maxSlot) {
  const grid = document.getElementById('grid');
  for (let i = 0; i <= maxSlot; i++) {
    if (!document.getElementById('slot-' + i)) {
      const cell = document.createElement('div');
      cell.className = 'cell empty';
      cell.id = 'slot-' + i;
      cell.innerHTML = '<span>slot ' + i + '</span>';
      grid.appendChild(cell);
    }
  }
}
function setSlot(slot, image, title) {
  document.getElementById('empty-hint').style.display = 'none';
  const grid = document.getElementById('grid');
  grid.style.display = '';
  ensureCells(slot);
  const cell = document.getElementById('slot-' + slot);
  cell.className = 'cell';
  cell.innerHTML =
    '<img src="' + image + '" alt="' + title +
    '" onclick="openLightbox(' + slot + ')">' +
    '<div class="cell-label">[' + slot + '] ' + title + '</div>';
}

// ── 翻页 ─────────────────────────────────────────────────
function switchPage(name) {
  if (currentPage === name) return;
  currentPage  = name;
  colsOverride = null;   // 切页时重置列数覆盖

  document.querySelectorAll('.page-item').forEach(el => {
    const active = el.dataset.page === name;
    el.classList.toggle('active', active);
    if (active) { const d = el.querySelector('.new-dot'); if (d) d.remove(); }
  });

  document.getElementById('page-title').textContent = name;
  const idx = pagesList.indexOf(name);
  document.getElementById('page-count').textContent =
    (idx + 1) + ' / ' + pagesList.length;
  document.getElementById('btn-prev').disabled = (idx >= pagesList.length - 1);
  document.getElementById('btn-next').disabled = (idx <= 0);

  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  grid.style.display = 'none';
  document.getElementById('empty-hint').style.display = '';
  applyPageCols(pageCols[name] || DEFAULT_COLS);

  if (pageSlots[name]) {
    renderSlots(pageSlots[name]);
  } else {
    fetch('/api/page/' + encodeURIComponent(name))
      .then(r => r.json())
      .then(data => { pageSlots[name] = data; renderSlots(data); });
  }
}
function renderSlots(data) {
  Object.entries(data)
    .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
    .forEach(([slot, info]) => setSlot(parseInt(slot), info.image, info.title));
}

function prevPage() {
  const i = pagesList.indexOf(currentPage);
  if (i < pagesList.length - 1) switchPage(pagesList[i + 1]);
}
function nextPage() {
  const i = pagesList.indexOf(currentPage);
  if (i > 0) switchPage(pagesList[i - 1]);
}

// ── 侧边栏 ───────────────────────────────────────────────
function addPage(name, cols, isNew) {
  if (pagesList.includes(name)) return;
  pagesList.unshift(name);
  pageCols[name] = cols || DEFAULT_COLS;

  const list = document.getElementById('pages-list');
  const item = document.createElement('div');
  item.className = 'page-item';
  item.dataset.page = name;
  item.innerHTML = name + (isNew && currentPage ? '<span class="new-dot"></span>' : '');
  item.onclick = () => switchPage(name);
  list.insertBefore(item, list.firstChild);

  // 更新所有页面的 Prev/Next 状态
  if (currentPage) {
    const idx = pagesList.indexOf(currentPage);
    document.getElementById('btn-prev').disabled = (idx >= pagesList.length - 1);
  }
  if (!currentPage) switchPage(name);
}

function markDirty(page) {
  if (page === currentPage) return;
  const item = document.querySelector('[data-page="' + page + '"]');
  if (item && !item.querySelector('.new-dot'))
    item.insertAdjacentHTML('beforeend', '<span class="new-dot"></span>');
}

function filterPages(q) {
  q = q.toLowerCase();
  document.querySelectorAll('.page-item').forEach(el =>
    el.classList.toggle('filtered', q !== '' && !el.dataset.page.toLowerCase().includes(q)));
}

// ── 下载 ─────────────────────────────────────────────────
function downloadPage() {
  if (!currentPage) return;
  const slots = pageSlots[currentPage] || {};
  Object.entries(slots).forEach(([slot, info]) => {
    const a = document.createElement('a');
    a.href = info.image;
    a.download = currentPage.replace(/[\/\\:*?"<>|]/g, '_') + '_slot' + slot + '.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });
}

// ── 灯箱 ─────────────────────────────────────────────────
function openLightbox(slotIdx) {
  if (!currentPage) return;
  const slots = pageSlots[currentPage] || {};
  lbSlots = Object.entries(slots).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
  lbIdx   = lbSlots.findIndex(([s]) => parseInt(s) === slotIdx);
  if (lbIdx < 0) lbIdx = 0;
  _renderLightbox();
  document.getElementById('lightbox').classList.add('open');
}
function _renderLightbox() {
  if (!lbSlots.length) return;
  const [slot, info] = lbSlots[lbIdx];
  document.getElementById('lb-img').src = info.image;
  document.getElementById('lb-caption').textContent =
    '[' + slot + '] ' + info.title + '  (' + (lbIdx + 1) + ' / ' + lbSlots.length + ')';
  document.getElementById('lb-prev').style.visibility = lbIdx > 0 ? '' : 'hidden';
  document.getElementById('lb-next').style.visibility =
    lbIdx < lbSlots.length - 1 ? '' : 'hidden';
}
function lbStep(dir) {
  lbIdx = Math.max(0, Math.min(lbSlots.length - 1, lbIdx + dir));
  _renderLightbox();
}
function lbDownload() {
  if (!lbSlots.length) return;
  const [slot, info] = lbSlots[lbIdx];
  const a = document.createElement('a');
  a.href = info.image;
  a.download = currentPage.replace(/[\/\\:*?"<>|]/g,'_') + '_slot' + slot + '.png';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}

// ── 键盘快捷键 ────────────────────────────────────────────
document.addEventListener('keydown', e => {
  const lb = document.getElementById('lightbox').classList.contains('open');
  if      (e.key === 'Escape')      { if (lb) closeLightbox(); }
  else if (e.key === 'ArrowLeft')   { if (lb) lbStep(-1);  else prevPage(); }
  else if (e.key === 'ArrowRight')  { if (lb) lbStep(1);   else nextPage(); }
});

// ── SSE ──────────────────────────────────────────────────
const es = new EventSource('/stream');

es.addEventListener('update', e => {
  const d = JSON.parse(e.data);
  if (!pageSlots[d.page]) pageSlots[d.page] = {};
  pageSlots[d.page][d.slot] = {image: d.image, title: d.title};
  if (d.page === currentPage) setSlot(parseInt(d.slot), d.image, d.title);
  else markDirty(d.page);
});

es.addEventListener('new_page', e => {
  const d = JSON.parse(e.data);
  addPage(d.page, d.cols, true);
});

es.onopen = () => {
  const el = document.getElementById('conn');
  el.textContent = '● 已连接'; el.style.color = '#4caf50';
};
es.onerror = () => {
  const el = document.getElementById('conn');
  el.textContent = '● 连接断开'; el.style.color = '#e94560';
};

// ── 初始加载 ─────────────────────────────────────────────
fetch('/api/pages')
  .then(r => r.json())
  .then(ps => {
    // ps newest-first；逆序 addPage 使最新的最终在顶
    [...ps].reverse().forEach(name => {
      fetch('/api/meta/' + encodeURIComponent(name))
        .then(r => r.json())
        .then(meta => addPage(name, meta.cols, false));
    });
  });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP 服务器（标准库，线程化）
# ---------------------------------------------------------------------------

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """每个请求独立线程，SSE 长连接不阻塞其他请求。"""
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    """请求处理器，通过类变量 dashboard 引用 WebDashboard 实例。"""
    dashboard: "WebDashboard" = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path

        if path == '/':
            self._html()
        elif path == '/stream':
            self._sse()
        elif path == '/api/pages':
            with self.dashboard._lock:
                pages = list(self.dashboard._pages.keys())
            self._json(pages)
        elif path.startswith('/api/page/'):
            name = urllib.parse.unquote(path[len('/api/page/'):])
            with self.dashboard._lock:
                data = dict(self.dashboard._pages.get(name, {}))
            self._json(data)
        elif path.startswith('/api/meta/'):
            name = urllib.parse.unquote(path[len('/api/meta/'):])
            with self.dashboard._lock:
                meta = dict(self.dashboard._meta.get(name, {'cols': self.dashboard._cols}))
            self._json(meta)
        elif path.startswith('/api/download/'):
            # /api/download/<page>/<slot>
            rest  = urllib.parse.unquote(path[len('/api/download/'):])
            parts = rest.split('/', 1)
            if len(parts) == 2:
                page, slot = parts
                with self.dashboard._lock:
                    info = self.dashboard._pages.get(page, {}).get(slot)
                if info:
                    self._png(info['image'], f'{page}_slot{slot}.png')
                else:
                    self.send_error(404)
            else:
                self.send_error(400)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/push':
            length = int(self.headers.get('Content-Length', 0))
            body   = json.loads(self.rfile.read(length))
            self.dashboard._store_and_broadcast(body)
            self._json({'ok': True})
        else:
            self.send_error(404)

    # ── 响应工具 ─────────────────────────────────────────────────────────
    def _html(self):
        html = self.dashboard._html_cache.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def _png(self, data_uri: str, filename: str):
        prefix = 'data:image/png;base64,'
        b64    = data_uri[len(prefix):]
        raw    = base64.b64decode(b64)
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header(
            'Content-Disposition',
            f'attachment; filename="{filename}"',
        )
        self.end_headers()
        self.wfile.write(raw)

    def _sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        q: queue.Queue = queue.Queue()
        self.dashboard._clients.append(q)
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    self.wfile.write(msg.encode('utf-8'))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b': ping\n\n')
                    self.wfile.flush()
        except Exception:
            pass
        finally:
            if q in self.dashboard._clients:
                self.dashboard._clients.remove(q)

    def log_message(self, fmt, *args):
        pass


# ---------------------------------------------------------------------------
# WebDashboard
# ---------------------------------------------------------------------------

class WebDashboard:
    """
    本地 Web 仪表盘服务器。

    参数
    ----
    port      : 监听端口（默认 5678）
    cols      : 默认格子列数（可被 per-page page_cols 覆盖，默认 3）
    auto_open : 启动后是否自动打开浏览器（默认 True）
    """

    def __init__(self, port: int = 5678, cols: int = 3, auto_open: bool = True):
        self._port      = port
        self._cols      = cols
        self._auto_open = auto_open

        # 数据存储
        self._pages: Dict[str, Dict[str, dict]] = {}
        self._meta:  Dict[str, dict]             = {}   # {page: {cols: int, ...}}
        self._clients: List[queue.Queue]          = []
        self._lock    = threading.Lock()
        self._server: Optional[_ThreadedHTTPServer] = None

        self._html_cache = _HTML_TEMPLATE.replace('COLS_VALUE', str(cols))

    # ── 服务生命周期 ──────────────────────────────────────────────────────

    def start(self) -> "WebDashboard":
        _Handler.dashboard = self
        self._server = _ThreadedHTTPServer(('localhost', self._port), _Handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        url = f'http://localhost:{self._port}'
        print(f'[VibDash] 服务已启动 → {url}')
        if self._auto_open:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        return self

    def stop(self):
        if self._server:
            self._server.shutdown()
            print('[VibDash] 服务已停止')

    def wait(self):
        """阻塞当前线程（保持进程存活），Ctrl+C 退出。"""
        print('[VibDash] 按 Ctrl+C 退出…')
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    # ── 图像注入（同进程）────────────────────────────────────────────────

    def push(
        self,
        fig,
        page: str,
        slot: int,
        title: Optional[str] = None,
        dpi: int = 150,
        page_cols: Optional[int] = None,
    ):
        """
        将 matplotlib Figure 注入指定页面的指定 slot。

        fig 可以是 Figure 对象，也可以是 (main_fig, aux_figs) 元组——仅取 main_fig。
        page_cols : 该页面的网格列数（仅首次创建页面时生效）。
        """
        main_fig = fig[0] if isinstance(fig, tuple) else fig
        image    = _fig_to_data_uri(main_fig, dpi)
        title    = title or f'slot {slot}'

        self._store_and_broadcast({
            'page'     : page,
            'slot'     : str(slot),
            'image'    : image,
            'title'    : title,
            'page_cols': page_cols,
        })

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _store_and_broadcast(self, payload: dict):
        page      = payload['page']
        slot      = str(payload['slot'])
        image     = payload['image']
        title     = payload.get('title', f'slot {slot}')
        page_cols = payload.get('page_cols')

        with self._lock:
            is_new = page not in self._pages
            if is_new:
                self._pages[page] = {}
                self._meta[page]  = {'cols': page_cols if page_cols is not None else self._cols}
            self._pages[page][slot] = {'image': image, 'title': title}

        if is_new:
            self._broadcast('new_page', {
                'page': page,
                'cols': self._meta[page]['cols'],
            })
        self._broadcast('update', {
            'page' : page,
            'slot' : slot,
            'image': image,
            'title': title,
        })

    def _broadcast(self, event: str, data: dict):
        msg = f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'
        for q in list(self._clients):
            q.put(msg)


# ---------------------------------------------------------------------------
# 跨进程 push()（HTTP POST，需要 requests）
# ---------------------------------------------------------------------------

def push(
    fig,
    page: str,
    slot: int,
    title: Optional[str] = None,
    port: int = 5678,
    dpi: int = 150,
    page_cols: Optional[int] = None,
):
    """
    从任意脚本/进程向已运行的 VibDash 服务推送一张图。

    服务须已在运行（python -m src.visualize_tools.web_dashboard），
    否则会收到清晰的连接拒绝提示。

    参数
    ----
    fig       : matplotlib.Figure 或 (main_fig, aux_figs) 元组
    page      : 页面名称（同名多次 push 累积到同一页）
    slot      : 格子编号（0-based）
    title     : 格子标题（默认 "slot N"）
    port      : VibDash 服务端口（默认 5678）
    dpi       : 渲染分辨率（默认 150）
    page_cols : 该页面列数（仅首次创建页面时生效）
    """
    import requests

    main_fig = fig[0] if isinstance(fig, tuple) else fig
    image    = _fig_to_data_uri(main_fig, dpi)
    title    = title or f'slot {slot}'

    url = f'http://localhost:{port}/push'
    try:
        resp = requests.post(url, json={
            'page'     : page,
            'slot'     : str(slot),
            'image'    : image,
            'title'    : title,
            'page_cols': page_cols,
        }, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f'[VibDash] 无法连接到 localhost:{port}，'
            '请先启动服务：python -m src.visualize_tools.web_dashboard'
        )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _fig_to_data_uri(fig, dpi: int = 150) -> str:
    """将 matplotlib Figure 序列化为 base64 data URI。"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'


# ---------------------------------------------------------------------------
# 全局单例（供 PlotLib.show_web() 使用）
# ---------------------------------------------------------------------------

_global_dashboard: Optional[WebDashboard] = None


def _get_or_start_dashboard(port: int, cols: int) -> WebDashboard:
    """返回全局 dashboard 实例，若未启动则自动启动。"""
    global _global_dashboard
    if _global_dashboard is None or _global_dashboard._port != port:
        _global_dashboard = WebDashboard(port=port, cols=cols).start()
    return _global_dashboard


# ---------------------------------------------------------------------------
# 独立服务器入口（python -m src.visualize_tools.web_dashboard）
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VibDash 独立服务器 — 启动后任意脚本可通过 push() 注入图像',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--port',       type=int,  default=5678, help='监听端口')
    parser.add_argument('--cols',       type=int,  default=3,    help='默认网格列数')
    parser.add_argument('--no-browser', action='store_true',     help='不自动打开浏览器')
    args = parser.parse_args()

    dash = WebDashboard(
        port=args.port,
        cols=args.cols,
        auto_open=not args.no_browser,
    )
    dash.start()
    dash.wait()
