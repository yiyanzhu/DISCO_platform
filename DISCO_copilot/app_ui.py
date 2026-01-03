import streamlit as st
import os
import json
import pandas as pd
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from real_tools import real_structure_builder, real_energy_calculator

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="DISCO-Pilot: Hierarchical Multi-Agent System",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 以优化论文截图效果
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .agent-box {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .success-metric {
        font-size: 1.2rem;
        font-weight: bold;
        color: #059669;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar: 配置
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/atom-editor.png", width=80)
    st.title("系统配置")
    
    # API Key 管理
    env_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_key = st.text_input("DeepSeek API Key", value=env_key, type="password")
    
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("### 🤖 智能体状态")
    st.info("Supervisor: Ready")
    st.info("Modeling Agent: Ready")
    st.info("Calculation Agent: Ready")

# ==========================================
# 核心逻辑函数
# ==========================================
def get_llm(api_key):
    return ChatOpenAI(
        model="deepseek-chat", 
        openai_api_key=api_key, 
        openai_api_base="https://api.deepseek.com",
        temperature=0.1
    )

def run_supervisor_planning(user_request, llm):
    """Supervisor 规划阶段"""
    prompt = f"""
    你是一个计算化学研究主管。用户的需求是: "{user_request}"
    
    请提取出用户想要研究的表面(Surface)和吸附物(Adsorbate)。
    然后，根据化学知识，列出需要在该表面上测试的吸附位点(Sites)。
    对于 fcc(111) 面，常见的位点有: top, bridge, fcc (hollow), hcp (hollow)。
    
    请严格以 JSON 格式返回，不要包含 Markdown 格式标记，格式如下:
    {{
        "surface": "Pt(111)",
        "adsorbate": "O",
        "sites": ["top", "bridge", "fcc", "hcp"]
    }}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
    return json.loads(content)

def generate_final_report(user_request, results, llm):
    """Supervisor 报告阶段"""
    results_str = ""
    for res in results:
        if res.get("energy") is not None:
            results_str += f"- 位点 {res['site']}: 能量 {res['energy']:.4f} eV\n"
        else:
            results_str += f"- 位点 {res['site']}: 计算失败\n"

    prompt = f"""
    所有计算已完成。请根据以下结果写一份简短的科学报告，指出哪个位点最稳定（能量最低）。
    请使用专业的学术语气。
    
    任务: {user_request}
    结果:
    {results_str}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ==========================================
# 主界面
# ==========================================
st.markdown('<h1 class="main-header">DISCO-Pilot: Hierarchical Multi-Agent System</h1>', unsafe_allow_html=True)
st.markdown("### 🧪 自动化计算化学研究平台")

# 初始化 Session State
if "plan" not in st.session_state:
    st.session_state.plan = None
if "results" not in st.session_state:
    st.session_state.results = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# 1. 用户输入区
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("请输入研究目标", value="研究Pt(111)面上的O原子的最稳定的吸附构型")
    with col2:
        st.write("") # Spacer
        st.write("")
        start_btn = st.button("🚀 开始规划 (Supervisor)", use_container_width=True, type="primary")

# 2. 规划阶段
if start_btn and api_key:
    with st.spinner("🧠 Supervisor (DeepSeek) 正在思考并制定研究计划..."):
        try:
            llm = get_llm(api_key)
            plan_data = run_supervisor_planning(user_input, llm)
            st.session_state.plan = plan_data
            st.session_state.results = None # 重置结果
            st.session_state.logs = []
            st.success("规划完成！")
        except Exception as e:
            st.error(f"规划失败: {e}")

if st.session_state.plan:
    st.markdown('<div class="sub-header">📋 研究计划 (Research Plan)</div>', unsafe_allow_html=True)
    
    plan = st.session_state.plan
    
    # 显示计划详情
    c1, c2, c3 = st.columns(3)
    c1.metric("表面模型", plan.get("surface"))
    c2.metric("吸附分子", plan.get("adsorbate"))
    c3.metric("待计算位点数", len(plan.get("sites", [])))
    
    # 转换为 DataFrame 显示
    df_plan = pd.DataFrame({
        "Task ID": [f"task_{s}" for s in plan["sites"]],
        "Site Type": plan["sites"],
        "Status": ["Pending"] * len(plan["sites"])
    })
    st.table(df_plan)
    
    # 执行按钮
    if st.button("✅ 批准并执行 (Execute Agents)", type="primary"):
        st.session_state.results = []
        
        # 创建进度容器
        progress_container = st.container()
        
        with progress_container:
            st.markdown('<div class="sub-header">⚙️ 智能体执行中 (Agent Execution)</div>', unsafe_allow_html=True)
            
            # 使用 st.status 展示详细过程
            with st.status("正在协调多智能体协作...", expanded=True) as status:
                
                total_tasks = len(plan["sites"])
                cols = st.columns(total_tasks)
                
                results_data = []
                
                for i, site in enumerate(plan["sites"]):
                    task_col = cols[i]
                    with task_col:
                        st.markdown(f"**Task: {site}**")
                        
                        # --- Modeling Agent ---
                        st.write("🔨 Modeling...")
                        time.sleep(0.5) # UI 效果
                        struct_path = real_structure_builder(plan["surface"], plan["adsorbate"], site)
                        
                        if struct_path:
                            st.success("Modeled")
                            
                            # --- Calculation Agent ---
                            st.write("🧮 Calculating...")
                            energy = real_energy_calculator(struct_path)
                            
                            if energy is not None:
                                st.success(f"E = {energy:.2f} eV")
                                results_data.append({
                                    "site": site,
                                    "energy": energy,
                                    "path": struct_path
                                })
                            else:
                                st.error("Calc Failed")
                                results_data.append({"site": site, "energy": None, "path": struct_path})
                        else:
                            st.error("Model Failed")
                            results_data.append({"site": site, "energy": None, "path": None})
                            
                status.update(label="✅ 所有智能体任务已完成！", state="complete", expanded=False)
                
            st.session_state.results = results_data

# 3. 结果与报告阶段
if st.session_state.results:
    st.markdown('<div class="sub-header">📊 最终报告 (Final Report)</div>', unsafe_allow_html=True)
    
    # 生成文字报告
    if api_key:
        with st.spinner("✍️ Supervisor 正在汇总数据并撰写报告..."):
            llm = get_llm(api_key)
            report = generate_final_report(user_input, st.session_state.results, llm)
            
            st.markdown("### 📝 智能体分析结论")
            st.markdown(f"""
            <div class="agent-box">
                {report}
            </div>
            """, unsafe_allow_html=True)
    
    # 数据可视化
    st.markdown("### 📈 能量数据对比")
    df_res = pd.DataFrame(st.session_state.results)
    
    # 找出最优
    if not df_res["energy"].isnull().all():
        min_idx = df_res["energy"].idxmin()
        best_site = df_res.loc[min_idx, "site"]
        best_energy = df_res.loc[min_idx, "energy"]
        
        col_metric, col_chart = st.columns([1, 2])
        with col_metric:
            st.metric("最稳定位点", best_site)
            st.metric("最低吸附能", f"{best_energy:.4f} eV")
        
        with col_chart:
            st.bar_chart(df_res.set_index("site")["energy"])
    
    # 下载区域
    st.markdown("### 📂 结构文件下载")
    file_cols = st.columns(len(st.session_state.results))
    for i, res in enumerate(st.session_state.results):
        with file_cols[i]:
            if res["path"] and os.path.exists(res["path"]):
                with open(res["path"], "r") as f:
                    file_content = f.read()
                st.download_button(
                    label=f"📥 {res['site']}.xyz",
                    data=file_content,
                    file_name=os.path.basename(res["path"]),
                    mime="chemical/x-xyz"
                )

# Footer
st.markdown("---")
st.caption("Powered by DeepSeek, LangChain, ASE & Streamlit | DISCO-Pilot v1.0")
