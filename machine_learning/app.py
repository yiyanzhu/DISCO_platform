"""
ML Platform (Strict SISSO Logic + Line Ending Fix)
- [SSH] 回滚到 SISSO 的相对路径逻辑 (使用 ~/path)，确保与你的环境兼容
- [Fix] 强制 slurm.sh 使用 Linux 换行符 (\n)，防止 Windows 上传导致脚本无法执行
- [UI] 修复弹窗遮挡问题 (zIndex)
"""

import base64
import os
import io
import sys
import time
import json
import joblib
import requests
from pathlib import Path
import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx, no_update, ALL, MATCH
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
import ase.io
import plotly.graph_objects as go
import zipfile
import traceback
from sklearn.model_selection import train_test_split

# Ensure local package import works when run as script
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from machine_learning.config import MLConfigManager
from machine_learning.data_loader import MLTrainDataBuilder
from services.common.templates import load_slurm_template
from services.config.loader import load_config, get_remote_server, get_queue_defaults

# Crystal Toolkit
import crystal_toolkit.components as ctc
from crystal_toolkit.settings import SETTINGS

# ----------------------------------------------------------------------------
# 0. 配置与基础数据
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_TEMPLATE_DIR = ROOT_DIR / "services" / "machine_learning"
GLOBAL_CONFIG = load_config()
REMOTE_CONFIG = get_remote_server(GLOBAL_CONFIG)
QUEUE_CONFIG = get_queue_defaults(GLOBAL_CONFIG)
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")


def resolve_elements_csv():
    candidates = []
    cfg_name = GLOBAL_CONFIG.get("local_paths", {}).get("elements_csv", "elements_properties_all.csv")
    candidates.append(Path(cfg_name))
    candidates.append(ROOT_DIR / cfg_name)
    candidates.append(ROOT_DIR / "discriptor" / cfg_name)
    candidates.append(Path(__file__).resolve().parent / cfg_name)
    for p in candidates:
        if p.exists():
            return p
    print("Warning: elements_properties CSV not found; atom features will be empty.")
    return None


ELEMENTS_CSV_PATH = resolve_elements_csv()
ELEMENTS_DF = pd.read_csv(ELEMENTS_CSV_PATH) if ELEMENTS_CSV_PATH else None

# 可用特征列（数值型）
AVAILABLE_FEATURES = []
if ELEMENTS_DF is not None:
    AVAILABLE_FEATURES = [c for c in ELEMENTS_DF.select_dtypes(include=["number"]).columns if c not in {"atomic_number"}]
    # 截取常用前若干个，避免下拉过长
    if len(AVAILABLE_FEATURES) > 20:
        AVAILABLE_FEATURES = AVAILABLE_FEATURES[:20]
if not AVAILABLE_FEATURES:
    AVAILABLE_FEATURES = ["atomic_number", "atomic_radius", "density"]

CONFIG_MANAGER = MLConfigManager()
TRADITIONAL_MODEL_OPTIONS = list(CONFIG_MANAGER.config.get("models", {}).get("traditional", {}).keys()) or ["xgb", "rf"]
GNN_MODEL_OPTIONS = list(CONFIG_MANAGER.config.get("models", {}).get("gnn", {}).keys()) or ["schnet", "dimenet_pp"]


def submit_job_via_backend(module: str, command: str, files: list[dict], remote_subdir: str = "ml"):
    if not BACKEND_BASE_URL:
        return None, "BACKEND_BASE_URL not set"
    try:
        resp = requests.post(
            f"{BACKEND_BASE_URL}/jobs/",
            json={
                "module": module,
                "command": command,
                "files": files,
                "remote_subdir": remote_subdir,
            },
            timeout=30,
        )
        if resp.status_code >= 300:
            return None, f"Backend rejected: {resp.text}"
        data = resp.json()
        return data, None
    except Exception as exc:
        return None, str(exc)


def refresh_job_via_backend(pk: int):
    if not BACKEND_BASE_URL:
        return None, "BACKEND_BASE_URL not set"
    try:
        resp = requests.post(f"{BACKEND_BASE_URL}/jobs/{pk}/refresh/", timeout=20)
        if resp.status_code >= 300:
            return None, f"Backend refresh failed: {resp.text}"
        return resp.json(), None
    except Exception as exc:
        return None, str(exc)

class MLFeatureBuilder:
    @staticmethod
    def extract_features(structures, indices_str, selected_features):
        if ELEMENTS_DF is None:
            return None, "缺少元素特征表 (elements_properties_all.csv)"

        data_rows = []
        try:
            indices = [int(i)-1 for i in indices_str.strip().split()]
        except: return None, "索引格式错误"
        
        if not selected_features: return None, "未选择特征"

        for struct_info in structures:
            fname = struct_info['filename']
            s_obj = parse_structure_content(struct_info['content'])
            if not s_obj: continue

            row = {'filename': os.path.splitext(fname)[0]}
            valid = True
            for i, atom_idx in enumerate(indices):
                if atom_idx >= len(s_obj):
                    valid = False; break
                sym = s_obj[atom_idx].specie.symbol
                props_row = ELEMENTS_DF[ELEMENTS_DF['symbol'].str.lower() == sym.lower()]
                for feat in selected_features:
                    if not props_row.empty and feat in props_row.columns:
                        val = props_row.iloc[0][feat]
                        row[f"Atom{i+1}_{feat}"] = 0.0 if pd.isna(val) else float(val)
                    else:
                        row[f"Atom{i+1}_{feat}"] = 0.0
            if valid: data_rows.append(row)
        
        if not data_rows: return None, "提取失败"
        return pd.DataFrame(data_rows), f"提取成功: {len(data_rows)} 行"

# ----------------------------------------------------------------------------
# 1. SSH 管理 (复用通用 SSHManager + 换行符修复)
# ----------------------------------------------------------------------------
from services.remote_server.ssh_manager import SSHManager as BaseSSHManager


class RealSSHManager:
    """Thin wrapper to keep legacy interface while delegating to shared SSHManager."""

    def __init__(self, hostname, username, password, port=22):
        self._base = BaseSSHManager(hostname=hostname, port=int(port), username=username, password=password)

    def connect(self):
        ok, msg = self._base.connect()
        if ok:
            self._base.open_sftp()
        return ok, msg

    def mkdir_remote(self, dir_name):
        return self._base.mkdir_remote(dir_name)

    def write_remote_file(self, dir_name, filename, content):
        remote_path = f"{dir_name}/{filename}"
        clean_content = str(content).replace('\r\n', '\n')
        return self._base.write_remote_file(remote_path, clean_content)

    def write_remote_binary(self, dir_name, filename, data: bytes):
        """Write binary payloads (e.g., joblib) to remote using SFTP."""
        try:
            if not getattr(self._base, "sftp", None):
                return False, "SFTP 未连接"
            remote_path = f"{dir_name}/{filename}"
            with self._base.sftp.file(remote_path, "wb") as f:
                f.write(data)
            return True, f"写入文件成功: {remote_path}"
        except Exception as e:
            return False, f"写入文件失败: {e}"

    def exec_command(self, cmd):
        ret, out, err = self._base.exec_command(cmd)
        return out, err

    def submit_job_slurm(self, dir_name):
        return self._base.submit_job_slurm(dir_name)

    def check_job_status(self, job_id):
        exists, _ = self._base.query_slurm_status(job_id)
        return "RUNNING" if exists else "COMPLETED"

    def download_file(self, dir_name, filename):
        success, content = self._base.read_remote_file(f"{dir_name}/{filename}")
        return content if success else None

    def close(self):
        self._base.close()

# ----------------------------------------------------------------------------
# 2. 配置与解析
# ----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "hostname": REMOTE_CONFIG.get("hostname", ""),
    "username": REMOTE_CONFIG.get("username", ""),
    "password": REMOTE_CONFIG.get("password", ""),
    "port": REMOTE_CONFIG.get("port", 22),
}


def load_template_file(name: str) -> str:
    candidates = [
        ML_TEMPLATE_DIR / name,
        Path(__file__).resolve().parent / "templates" / name,
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(f"Template {name} not found in {candidates}")


TRAIN_SCRIPT_TEMPLATE = load_template_file("train_script.py")
try:
    SLURM_TEMPLATE = load_slurm_template(path=ML_TEMPLATE_DIR / "slurm.sh")
except FileNotFoundError:
    SLURM_TEMPLATE = load_template_file("slurm.sh")

def parse_structure_content(content):
    if not content: return None
    try:
        if "data_" in content[:500] or "_cell_" in content[:1000]:
            return Structure.from_str(content, fmt="cif")
        return Structure.from_str(content, fmt="poscar")
    except:
        try:
            atoms = ase.io.read(io.StringIO(content))
            return AseAtomsAdaptor.get_structure(atoms)
        except: return None


def build_corr_warning(full_df: pd.DataFrame, target_col: str):
    """For traditional models: compute Pearson corr on numeric feature cols (excluding filename/target),
    return plotly heatmap figure and warning text for |corr| > 0.8."""
    feature_df = full_df.drop(columns=[c for c in ["filename", target_col] if c in full_df.columns])
    feature_df = feature_df.select_dtypes(include=[np.number])
    if feature_df.shape[1] < 2:
        return None, "特征列不足，无法计算相关性。"

    corr = feature_df.corr(method="pearson").fillna(0)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmin=-1, zmax=1, colorbar=dict(title="Pearson")))
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40), height=480, template="plotly_white")

    high_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i):
            val = corr.iloc[i, j]
            if abs(val) >= 0.8:
                high_pairs.append((cols[i], cols[j], float(val)))

    if high_pairs:
        warn_text = "以下特征对相关性 |r| ≥ 0.8，请留意共线性: " + "; ".join([f"{a} vs {b}: {v:.2f}" for a, b, v in high_pairs])
    else:
        warn_text = "未发现 |r| ≥ 0.8 的特征对。"

    warning_block = dbc.Alert(warn_text, color="warning" if high_pairs else "success", className="mt-2 mb-1")
    heatmap_block = dcc.Graph(figure=fig, style={"height": "520px"})
    return heatmap_block, warning_block

# ----------------------------------------------------------------------------
# 3. UI
# ----------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"], suppress_callback_exceptions=True)
server = app.server

new_batch_uploader = dcc.Upload(
    id="new-batch-uploader",
    children=html.Div([
        html.I(className="bi bi-cloud-upload", style={"fontSize": "2rem"}),
        html.Div("拖入结构文件"),
        html.Div("生成新批次", className="text-muted small")
    ]),
    className="upload-container",
    multiple=True
)

settings_card = dbc.Card([
    dbc.CardHeader("2. 配置"),
    dbc.CardBody([
        dbc.Label("特征 (Atom Features)", className="small fw-bold"),
        dcc.Dropdown(
            id="inp-features",
            options=[{'label': f, 'value': f} for f in AVAILABLE_FEATURES],
            value=AVAILABLE_FEATURES[:3],
            multi=True,
            className="mb-2"
        ),
        dbc.Label("模型 (Model)", className="small fw-bold"),
        dcc.Dropdown(
            id="inp-model",
            options=[{'label': f"Trad | {m.upper()}", 'value': m} for m in TRADITIONAL_MODEL_OPTIONS] +
                    [{'label': f"GNN | {m.upper()}", 'value': m} for m in GNN_MODEL_OPTIONS],
            value=(TRADITIONAL_MODEL_OPTIONS[0] if TRADITIONAL_MODEL_OPTIONS else (GNN_MODEL_OPTIONS[0] if GNN_MODEL_OPTIONS else 'xgb')),
            clearable=False
        ),
        html.Hr(),
        dbc.Button("合并数据并生成任务", id="btn-generate", color="primary", className="w-100"),
        html.Div(id="log-gen", style={"height": "60px", "overflowY": "scroll", "backgroundColor": "#111", "color": "#0f0", "fontSize": "11px", "padding": "5px", "marginTop": "5px", "whiteSpace": "pre-wrap"})
    ])
])

modal = dbc.Modal([
    dbc.ModalHeader("预览与提交"),
    dbc.ModalBody(dbc.Tabs([
        dbc.Tab(label="train_script.py", label_class_name="fw-semibold text-dark", children=[
            html.Label("train_script.py", className="fw-bold mb-1"),
            dcc.Textarea(
                id="editor-script",
                style={"width": "100%", "height": "380px", "fontFamily": "monospace", "backgroundColor": "#f8f9fa", "color": "#111", "border": "1px solid #dee2e6"}
            )
        ]),
        dbc.Tab(label="slurm.sh", label_class_name="fw-semibold text-dark", children=[
            html.Label("slurm.sh", className="fw-bold mb-1"),
            dcc.Textarea(
                id="editor-slurm",
                style={"width": "100%", "height": "380px", "fontFamily": "monospace", "backgroundColor": "#f8f9fa", "color": "#111", "border": "1px solid #dee2e6"}
            )
        ]),
        dbc.Tab(label="config.json", label_class_name="fw-semibold text-dark", children=[
            html.Label("config.json", className="fw-bold mb-1"),
            dcc.Textarea(
                id="editor-config",
                style={"width": "100%", "height": "380px", "fontFamily": "monospace", "whiteSpace": "pre", "backgroundColor": "#f8f9fa", "color": "#111", "border": "1px solid #dee2e6"}
            )
        ]),
        dbc.Tab(label="train_data.csv", label_class_name="fw-semibold text-dark", children=[
            html.Label("train_data.csv", className="fw-bold mb-1"),
            dcc.Textarea(
                id="editor-data",
                style={"width": "100%", "height": "380px", "fontFamily": "monospace", "whiteSpace": "pre", "backgroundColor": "#f8f9fa", "color": "#111", "border": "1px solid #dee2e6"}
            )
        ]),
    ])),
    dbc.ModalFooter([
        html.Div(id="log-sub", className="me-auto small text-muted"),
        dbc.Button("拉取状态", id="btn-pull-status", outline=True, color="warning", className="me-2"),
        dbc.Button("取消", id="btn-close-modal", className="me-2"),
        dbc.Button("提交", id="btn-submit-modal", color="primary")
    ])
], id="modal-file-editor", size="xl", backdrop="static", style={"zIndex": 100000})

left_panel = [
    dbc.Card([dbc.CardHeader("1. 新建 (New Batch)", className="bg-primary text-white py-2"), dbc.CardBody([new_batch_uploader], className="p-2")], className="mb-3"),
    html.Div(settings_card, className="mb-3")
]

right_panel = [
    dbc.Card([
        dbc.CardHeader(["工作区", dbc.Button("清空", id="btn-reset-all", color="link", size="sm", className="float-end text-danger text-decoration-none py-0")]), 
        dbc.CardBody([html.Div(id="batches-container", className="row g-2"), html.Div("请拖入结构...", id="empty-placeholder", className="text-center text-muted py-5")], className="p-2")
    ], className="mb-3"),
    dbc.Card([dbc.CardHeader("4. 结果"), dbc.CardBody(html.Div(id='result-display'))])
]

ctc.register_crystal_toolkit(app=app, layout=dbc.Container([
    modal, dcc.Store(id='store-batches-data', data=[]), dcc.Store(id='store-job-info', data={}), dcc.Interval(id='interval-job-monitor', interval=5000),
    dbc.NavbarSimple(
        brand="🤖 ML Feature Building",
        color="white", className="mb-3 shadow-sm",
        children=[dbc.NavItem(dbc.NavLink("Reset", href="/", external_link=True))]
    ), 
    dbc.Row([dbc.Col(left_panel, width=3), dbc.Col(right_panel, width=9)])
], fluid=True, style={"minHeight": "100vh"}))

# ----------------------------------------------------------------------------
# 4. 回调函数
# ----------------------------------------------------------------------------

# 1. 批次创建
@app.callback(
    Output("store-batches-data", "data"), Output("batches-container", "children"), Output("empty-placeholder", "style"), Output("new-batch-uploader", "contents"),
    Input("new-batch-uploader", "contents"), Input("btn-reset-all", "n_clicks"),
    State("new-batch-uploader", "filename"), State("store-batches-data", "data")
)
def manage_batches_safe(contents, n_reset, filenames, current_data):
    if ctx.triggered_id == "btn-reset-all": return [], [], {"display": "block"}, None
    try:
        if current_data is None: current_data = []
        updated_data = current_data
        if contents:
            new_structs = []
            for c, f in zip(contents, filenames):
                try:
                    decoded = base64.b64decode(c.split(",")[1])
                    if f.endswith('.zip'):
                        with zipfile.ZipFile(io.BytesIO(decoded)) as z:
                            for n in z.namelist():
                                if not n.endswith('/') and not n.startswith('__MACOSX'):
                                    new_structs.append({'filename': os.path.basename(n), 'content': z.read(n).decode('utf-8', errors='ignore')})
                    else:
                        new_structs.append({'filename': f, 'content': decoded.decode('utf-8', errors='ignore')})
                except Exception as e: print(f"File Error: {e}")
            if new_structs: updated_data.append({"id": len(current_data), "structures": new_structs})

        children = []
        for batch in updated_data:
            b_id = batch['id']
            structs = batch['structures']
            init_struct = None
            if structs: init_struct = parse_structure_content(structs[0]['content'])
            if init_struct is None: init_struct = Structure(Lattice.cubic(3.0), ["H"], [[0,0,0]])

            ctc_view = ctc.StructureMoleculeComponent(init_struct, id=f"viewer-batch-{b_id}", color_scheme="VESTA")
            
            card = dbc.Col(dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([html.Strong(f"Batch #{b_id+1}"), html.Span(f"{len(structs)}", className="badge bg-secondary ms-1")], width="auto"),
                        dbc.Col([dcc.Dropdown(id={'type': 'struct-sel', 'index': b_id}, options=[{'label': s['filename'], 'value': i} for i, s in enumerate(structs)], value=0, clearable=False, style={"fontSize": "12px"})], width=4),
                        dbc.Col([dbc.Input(id={'type': 'indices-inp', 'index': b_id}, placeholder="Index (e.g. 1 2)", size="sm")], width=3),
                        dbc.Col([dcc.Upload(id={'type': 'csv-up', 'index': b_id}, children=html.Div("CSV", id={'type': 'csv-lbl', 'index': b_id}, className="badge bg-secondary"), style={"cursor": "pointer"})], width=True)
                    ], className="g-1 align-items-center")
                ]),
                dbc.CardBody(html.Div(ctc_view.layout(), style={"height": "300px", "width": "100%", "position": "relative"}))
            ], className="shadow-sm border-0 mb-3"), width=12, lg=6)
            children.append(card)
        return updated_data, children, {"display": "none"} if children else {"display": "block"}, None
    except: return no_update, no_update, no_update, None

# 2. 动态视图
@app.callback([Output(f"viewer-batch-{i}", "data") for i in range(20)], Input({'type': 'struct-sel', 'index': ALL}, 'value'), State("store-batches-data", "data"))
def update_view(vals, data):
    outs = [no_update] * 20
    if not ctx.triggered_id: return outs
    idx = ctx.triggered_id['index']
    if idx < len(data):
        try:
            s = parse_structure_content(data[idx]['structures'][vals[idx]]['content'])
            if s: outs[idx] = s
        except: pass
    return outs

# 3. CSV
@app.callback(Output({'type': 'csv-lbl', 'index': MATCH}, 'children'), Output({'type': 'csv-lbl', 'index': MATCH}, 'className'), Input({'type': 'csv-up', 'index': MATCH}, 'contents'))
def csv_lbl(c): return ("OK", "badge bg-success") if c else no_update

# 4. 生成与提交
@app.callback(
    [Output("log-gen", "children"), Output("editor-script", "value"), Output("editor-slurm", "value"), Output("editor-config", "value"), Output("editor-data", "value"), Output("modal-file-editor", "is_open"), 
     Output("store-job-info", "data"), Output("log-sub", "children"), Output("result-display", "children")],
    [Input("btn-generate", "n_clicks"), Input("btn-close-modal", "n_clicks"), Input("btn-submit-modal", "n_clicks"), Input("interval-job-monitor", "n_intervals"), Input("btn-pull-status", "n_clicks")],
    [State("store-batches-data", "data"), State({'type': 'csv-up', 'index': ALL}, 'contents'), State({'type': 'indices-inp', 'index': ALL}, 'value'),
     State("inp-features", "value"), State("inp-model", "value"), State("log-gen", "children"), 
     State("editor-script", "value"), State("editor-slurm", "value"), State("editor-config", "value"), State("editor-data", "value"), State("store-job-info", "data"), State("log-sub", "children")]
)
def task_manager(n_gen, n_cl, n_sub, n_int, n_pull, batches, csvs, idxs, feats, model, l_gen, scr, slu, cfg, dat, job, l_sub):
    trig = ctx.triggered_id
    
    if trig == "btn-generate":
        if not batches:
            return (l_gen or "")+"\n无批次数据", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        is_gnn = model in GNN_MODEL_OPTIONS
        logs, dfs, graph_dataset = [], [], []
        data_builder = MLTrainDataBuilder(ELEMENTS_DF if ELEMENTS_DF is not None else pd.DataFrame())

        for i, b in enumerate(batches):
            idx_str = idxs[i] if i < len(idxs) else None
            csv_content = csvs[i] if i < len(csvs) else None

            if is_gnn:
                if not csv_content:
                    logs.append(f"Batch {i+1}: GNN 需要 CSV 目标列")
                    continue
                try:
                    c_df = pd.read_csv(io.StringIO(base64.b64decode(csv_content.split(',')[1]).decode('utf-8')))
                    if 'filename' not in c_df:
                        c_df.rename(columns={c_df.columns[0]: 'filename'}, inplace=True)
                    c_df['filename'] = c_df['filename'].astype(str).apply(lambda x: x.split('.')[0])
                    if len(c_df.columns) < 2:
                        logs.append(f"Batch {i+1}: CSV 缺少目标列（至少需要2列：id, target）")
                        continue
                    target_col = c_df.columns[1]
                    targets_map = dict(zip(c_df['filename'], c_df[target_col]))
                    part = data_builder.build_graph_dataset(b['structures'], targets_map, parse_structure_func=parse_structure_content) if data_builder else []
                    if part:
                        graph_dataset.extend(part)
                        logs.append(f"Batch {i+1}: 图数据 {len(part)} 条 (目标列 {target_col})")
                    else:
                        logs.append(f"Batch {i+1}: 图数据生成失败")
                except Exception as e:
                    logs.append(f"CSV/GNN Error: {e}")
                continue

            # 传统模型路径
            if not idx_str:
                logs.append(f"Batch {i+1}: 无索引")
                continue
            feat_df, msg = MLFeatureBuilder.extract_features(b['structures'], idx_str, feats)
            logs.append(f"Batch {i+1}: {msg}")
            if feat_df is not None:
                if csv_content:
                    try:
                        c_df = pd.read_csv(io.StringIO(base64.b64decode(csv_content.split(',')[1]).decode('utf-8')))
                        if 'filename' not in c_df: c_df.rename(columns={c_df.columns[0]: 'filename'}, inplace=True)
                        c_df['filename'] = c_df['filename'].astype(str).apply(lambda x: x.split('.')[0])

                        if len(c_df.columns) < 2:
                            logs.append(f"Batch {i+1}: CSV 缺少目标列（至少需要2列：id, target[, extra...]）")
                            continue

                        target_col = c_df.columns[1]  # 按描述符逻辑，第二列为目标
                        extra_cols_csv = list(c_df.columns[2:])

                        merge_df = pd.merge(feat_df, c_df, on='filename')

                        # 重排: filename + 结构特征 + CSV额外特征 + 目标列
                        feat_cols = list(feat_df.columns)
                        extra_cols_merged = [c for c in extra_cols_csv if c in merge_df.columns]
                        ordered_cols = feat_cols + extra_cols_merged + [target_col]
                        # 去重以防万一
                        ordered_cols = [c for i, c in enumerate(ordered_cols) if c not in ordered_cols[:i]]
                        merge_df = merge_df[ordered_cols]
                        feat_df = merge_df
                        logs.append(f"Batch {i+1}: 目标列 '{target_col}' 放到最后，额外特征 {extra_cols_merged}")
                    except Exception as e:
                        logs.append(f"CSV Error: {e}")
                dfs.append(feat_df)
        
        if is_gnn:
            if not graph_dataset:
                return (l_gen or "")+"\nGNN 数据生成失败", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
            train_set, test_set = train_test_split(graph_dataset, test_size=0.2, random_state=42)
            payload = (train_set, None, test_set, None)
            buf = io.BytesIO()
            joblib.dump(payload, buf)
            pkl_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            data_preview = f"GNN dataset: total {len(graph_dataset)}, train {len(train_set)}, test {len(test_set)}"
            corr_view = dbc.Alert("GNN 模型跳过相关性检查", color="info", className="mt-2 mb-0")
            model_params = CONFIG_MANAGER.get_model_params("gnn", model) or {}
            config_payload = json.dumps({"model_name": model, "params": model_params}, indent=4)
            job_payload = {"job_name": f"ml_job_{int(time.time())}", "dataset_type": "graph", "pkl_b64": pkl_b64}
            job_name = job_payload["job_name"]
        else:
            if not dfs:
                return (l_gen or "")+"\n失败", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

            full_df = pd.concat(dfs).fillna(0)
            data_preview = full_df.to_csv(index=False)
            # Pearson correlation check (traditional models only)
            target_col = full_df.columns[-1]
            heatmap_block, warning_block = build_corr_warning(full_df, target_col)
            corr_view = html.Div([
                html.H6("特征相关性 (Pearson)", className="fw-bold mt-2"),
                warning_block if warning_block else html.Div("无相关性结果"),
                heatmap_block if heatmap_block else html.Div()
            ], className="p-2 bg-light border rounded")
            model_params = CONFIG_MANAGER.get_model_params("traditional", model) or {}
            config_payload = json.dumps({"model_name": model, "params": model_params}, indent=4)
            job_payload = {"job_name": f"ml_job_{int(time.time())}", "dataset_type": "tabular"}
            job_name = job_payload["job_name"]

        fmt_kwargs = {
            "job_name": job_name,
            "nodes": QUEUE_CONFIG.get("nodes", 1),
            "ntasks": QUEUE_CONFIG.get("ntasks_per_node", 30),
            "time_limit": QUEUE_CONFIG.get("time_limit", "01:00:00"),
            "partition": QUEUE_CONFIG.get("partition", "vasp"),
            # Support shared template placeholders
            "command": "python train_script.py",
            "email_directive": "",
            # Support legacy local template placeholder
            "env_name": "ml_env",
        }
        slurm_content = SLURM_TEMPLATE.format(**fmt_kwargs)
        script_content = TRAIN_SCRIPT_TEMPLATE.replace("{model_name}", model)

        return (
            (l_gen or "")+"\n" + " | ".join(logs) if logs else (l_gen or ""),
            script_content,
            slurm_content,
            config_payload,
            data_preview if data_preview else "生成的 train_data 为空",
            True,
            job_payload,
            no_update,
            corr_view,
        )

    elif trig == "btn-close-modal":
            return no_update, no_update, no_update, no_update, no_update, False, no_update, no_update, no_update

    elif trig == "btn-submit-modal":
        dataset_type = (job or {}).get("dataset_type", "tabular")
        pkl_b64 = (job or {}).get("pkl_b64")

        cfg_content = cfg
        if not cfg_content:
            params = CONFIG_MANAGER.get_model_params("gnn", model) if dataset_type == "graph" else CONFIG_MANAGER.get_model_params("traditional", model) or {}
            cfg_content = json.dumps({"model_name": model, "params": params}, indent=4)

        if not scr or not slu or (not dat and dataset_type != "graph"):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, "文件内容缺失，请重新生成", no_update
        if dataset_type == "graph" and not pkl_b64:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, "缺少图数据文件，请重新生成", no_update

        use_backend = bool(BACKEND_BASE_URL) and dataset_type != "graph"
        if use_backend:
            files = [
                {"name": "train_script.py", "content": scr},
                {"name": "slurm.sh", "content": slu},
                {"name": "config.json", "content": cfg_content},
                {"name": "train_data.csv", "content": dat},
            ]
            data, err = submit_job_via_backend(
                module="machine_learning",
                command="bash train_script.py",
                files=files,
                remote_subdir="ml",
            )
            if err or not data:
                return no_update, no_update, no_update, no_update, no_update, no_update, no_update, f"后端提交失败: {err}", no_update

            job_info = {
                "id": data.get("job_id"),
                "dir": data.get("remote_dir"),
                "status": "sub",
                "dataset_type": dataset_type,
                "backend_pk": data.get("id"),
            }
            log_msg = f"已通过 Django 后端提交: job={job_info['id']} dir={job_info['dir']}"
            return no_update, no_update, no_update, no_update, no_update, False, job_info, log_msg, no_update

        ssh = RealSSHManager(**DEFAULT_CONFIG)
        ok, msg = ssh.connect()
        if not ok:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, f"连接失败: {msg}", no_update

        rd = f"ML_{int(time.time())}"
        ok_mk, msg_mk = ssh.mkdir_remote(rd)
        if not ok_mk:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, f"{msg_mk}", no_update

        log_steps = [f"目录: {rd}"]
        ok1, m1 = ssh.write_remote_file(rd, "train_script.py", scr)
        log_steps.append(m1)
        ok2, m2 = ssh.write_remote_file(rd, "slurm.sh", slu)
        log_steps.append(m2)
        if dataset_type == "graph":
            raw = base64.b64decode(pkl_b64)
            ok3, m3 = ssh.write_remote_binary(rd, "train_data.pkl", raw)
        else:
            ok3, m3 = ssh.write_remote_file(rd, "train_data.csv", dat)
        log_steps.append(m3)
        ok4, m4 = ssh.write_remote_file(rd, "config.json", cfg_content)
        log_steps.append(m4)

        if ok1 and ok2 and ok3 and ok4:
            ok_s, jid = ssh.submit_job_slurm(rd)
            if ok_s:
                return no_update, no_update, no_update, no_update, no_update, False, {"id": jid, "dir": rd, "status": "sub", "dataset_type": dataset_type}, "\n".join(log_steps)+f"\n成功 Job: {jid}", no_update
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, "\n".join(log_steps)+f"\n提交失败: {jid}", no_update

        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, "\n".join(log_steps)+"\n上传中断", no_update

    elif trig in ["interval-job-monitor", "btn-pull-status"]:
        if not isinstance(job, dict) or not job.get("status"):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        if job.get("status") == "sub":
            if job.get("backend_pk"):
                refreshed, err = refresh_job_via_backend(job.get("backend_pk"))
                if err or not refreshed:
                    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, f"刷新失败: {err}", no_update
                job_status = refreshed.get("status")
                job['status'] = "done" if job_status == "COMPLETED" else job_status.lower()
            else:
                ssh = RealSSHManager(**DEFAULT_CONFIG)
                if ssh.connect()[0]:
                    if ssh.check_job_status(job['id']) == "COMPLETED":
                        job['status'] = "done"
                else:
                    return no_update, no_update, no_update, no_update, no_update, no_update, job, "连接失败，稍后再试", no_update

            if job.get("status") in ["done", "COMPLETED"]:
                ssh = RealSSHManager(**DEFAULT_CONFIG)
                if ssh.connect()[0]:
                    res = ssh.download_file(job['dir'], "results.csv")
                    if not res:
                        res = ssh.download_file(f"{job['dir']}/outputs/machine_learning", "results.csv")
                    ssh.close()
                    if res:
                        df = pd.read_csv(io.StringIO(res))
                        fig = go.Figure(data=go.Scatter(x=df['y_true'], y=df['y_pred'], mode='markers', text=df.get('id')))
                        fig.add_shape(type="line", x0=df['y_true'].min(), y0=df['y_true'].min(), x1=df['y_true'].max(), y1=df['y_true'].max(), line=dict(dash="dash", color="green"))
                        return no_update, no_update, no_update, no_update, no_update, no_update, job, "完成!", html.Div([dcc.Graph(figure=fig), dbc.Table.from_dataframe(df.head(), striped=True)])

    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

if __name__ == "__main__":
    app.run_server(debug=True, port=8052)