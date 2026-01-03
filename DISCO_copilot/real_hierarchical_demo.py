import os
import operator
import json
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 导入真实的工具
from real_tools import real_structure_builder, real_energy_calculator

# ==========================================
# 配置 DeepSeek API
# ==========================================
# 请在这里直接填入您的 DeepSeek API Key
API_KEY = "sk-63404d56c125456e8d5e78cd60d2decc" 
BASE_URL = "https://api.deepseek.com"

# 如果上面没填，尝试从环境变量读取
if "在这里填入" in API_KEY:
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        API_KEY = env_key

if not API_KEY or "在这里填入" in API_KEY:
    print("⚠️ 警告: 未配置 API Key。请在代码中填入 API_KEY。")

llm = ChatOpenAI(
    model="deepseek-chat", 
    openai_api_key=API_KEY, 
    openai_api_base=BASE_URL,
    temperature=0.1
)

# ==========================================
# 1. 定义状态 (State)
# ==========================================
class SubTask(TypedDict):
    id: str
    site_type: str      # top, bridge, hollow
    structure_path: Optional[str]
    energy: Optional[float]
    status: str         # pending, modeled, calculated

class ResearchState(TypedDict):
    user_request: str
    surface_name: str
    adsorbate: str
    plan: List[SubTask]
    next_worker: str
    final_report: str
    logs: Annotated[list, operator.add]

# ==========================================
# 2. 智能体节点 (Agents)
# ==========================================

def supervisor_agent(state: ResearchState):
    """
    Supervisor: 使用 DeepSeek 进行规划
    """
    print("\n=== [Supervisor (DeepSeek)] 正在思考... ===")
    
    # --- 阶段 1: 初始规划 ---
    if not state.get("plan"):
        prompt = f"""
        你是一个计算化学研究主管。用户的需求是: "{state['user_request']}"
        
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
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            # 清理可能存在的 markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            data = json.loads(content)
            
            surface = data.get("surface", "Pt(111)")
            adsorbate = data.get("adsorbate", "O")
            sites = data.get("sites", ["top"])
            
            new_plan = []
            for site in sites:
                new_plan.append({
                    "id": f"task_{site}",
                    "site_type": site,
                    "structure_path": None,
                    "energy": None,
                    "status": "pending"
                })
            
            print(f">> DeepSeek 规划完成: {surface} + {adsorbate}, 位点: {sites}")
            
            # --- 人工确认环节 ---
            print("\n📋 [Supervisor] 拟定研究计划如下:")
            print(f"   - 表面模型: {surface}")
            print(f"   - 吸附分子: {adsorbate}")
            print(f"   - 待计算位点: {sites}")
            
            confirm = input("\n❓ 是否执行此计划? (输入 y 继续，n 取消): ").strip().lower()
            if confirm != 'y':
                print("🚫 任务已取消。")
                return {"next_worker": "FINISH", "logs": ["用户取消任务"]}
            
            print("✅ 计划已确认，正在分发给 Modeling Agent...\n")
            
            return {
                "surface_name": surface,
                "adsorbate": adsorbate,
                "plan": new_plan,
                "next_worker": "modeling_agent",
                "logs": [f"Supervisor: 规划了 {len(sites)} 个任务"]
            }
        except Exception as e:
            print(f"Supervisor 出错: {e}")
            return {"next_worker": "FINISH", "logs": [f"Error: {e}"]}

    # --- 阶段 2: 调度 ---
    plan = state["plan"]
    
    # 检查是否有任务需要建模
    if any(t["status"] == "pending" for t in plan):
        return {"next_worker": "modeling_agent"}
    
    # 检查是否有任务需要计算
    if any(t["status"] == "modeled" for t in plan):
        return {"next_worker": "calculation_agent"}
    
    # --- 阶段 3: 报告 ---
    # 只要没有 pending 或 modeled 的任务，就说明都处理完了（包括 failed）
    if not any(t["status"] in ["pending", "modeled"] for t in plan):
        # 让 LLM 写报告
        results_str = ""
        for t in plan:
            if t.get("energy") is not None:
                results_str += f"- 位点 {t['site_type']}: 能量 {t['energy']:.4f} eV\n"
            else:
                results_str += f"- 位点 {t['site_type']}: 计算失败\n"

        prompt = f"""
        所有计算已完成。请根据以下结果写一份简短的科学报告，指出哪个位点最稳定（能量最低）。
        
        任务: {state['user_request']}
        结果:
        {results_str}
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # --- 生成终端可点击的超链接 (OSC 8) ---
        # 这种格式可以在终端显示文字，点击打开文件，且不暴露路径
        report_content = response.content + "\n\n### 📂 结构文件 (按住 Ctrl 点击打开)\n"
        for t in plan:
            path = t.get("structure_path")
            if path and os.path.exists(path):
                # 构造 file:// URL
                abs_path = os.path.abspath(path).replace("\\", "/")
                file_url = f"file:///{abs_path}"
                filename = os.path.basename(path)
                
                # OSC 8 转义序列: \033]8;;URL\033\TEXT\033]8;;\033\
                link_text = f"📄 查看 {t['site_type']} 模型 ({filename})"
                hyperlink = f"\033]8;;{file_url}\033\\{link_text}\033]8;;\033\\"
                
                report_content += f"- {hyperlink}\n"

        return {
            "final_report": report_content,
            "next_worker": "FINISH"
        }
        
    return {"next_worker": "FINISH"}

def modeling_agent(state: ResearchState):
    """
    Modeling Agent: 调用 real_structure_builder
    """
    print("\n=== [Modeling Agent] 开始建模... ===")
    plan = state["plan"]
    updated_plan = []
    
    for task in plan:
        if task["status"] == "pending":
            # 调用真实工具
            path = real_structure_builder(
                state["surface_name"], 
                state["adsorbate"], 
                task["site_type"]
            )
            
            new_task = task.copy()
            if path:
                new_task["structure_path"] = path
                new_task["status"] = "modeled"
            else:
                new_task["status"] = "failed" # 标记失败
                
            updated_plan.append(new_task)
        else:
            updated_plan.append(task)
            
    return {"plan": updated_plan}

def calculation_agent(state: ResearchState):
    """
    Calculation Agent: 调用 real_energy_calculator
    """
    print("\n=== [Calculation Agent] 开始计算 (ASE/EMT)... ===")
    plan = state["plan"]
    updated_plan = []
    
    for task in plan:
        if task["status"] == "modeled":
            # 调用真实工具
            energy = real_energy_calculator(task["structure_path"])
            
            new_task = task.copy()
            if energy is not None:
                new_task["energy"] = energy
                new_task["status"] = "calculated"
            else:
                new_task["status"] = "failed"
                
            updated_plan.append(new_task)
        else:
            updated_plan.append(task)
            
    return {"plan": updated_plan}

# ==========================================
# 3. 构建图
# ==========================================
def router(state: ResearchState):
    nxt = state["next_worker"]
    if nxt == "modeling_agent": return "modeling"
    elif nxt == "calculation_agent": return "calculation"
    elif nxt == "FINISH": return "end"
    return "supervisor"

workflow = StateGraph(ResearchState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("modeling", modeling_agent)
workflow.add_node("calculation", calculation_agent)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    router,
    {"modeling": "modeling", "calculation": "calculation", "end": END}
)

workflow.add_edge("modeling", "supervisor")
workflow.add_edge("calculation", "supervisor")

app = workflow.compile()

# ==========================================
# 4. 运行入口
# ==========================================
if __name__ == "__main__":
    print("🚀 启动真实计算化学智能体 (Powered by DeepSeek & ASE)")
    
    # 检查 Key
    if not API_KEY or "在这里填入" in API_KEY:
        print("❌ 错误: 请打开代码文件，在第 15 行填入您的 DeepSeek API Key。")
        exit(1)

    user_input = input("\n👤 请输入研究指令 (默认: 研究Pt(111)上O原子的吸附): \n> ").strip()
    if not user_input:
        user_input = "研究Pt(111)上O原子的吸附"
        
    initial_state = {
        "user_request": user_input,
        "plan": [],
        "logs": [],
        "next_worker": "supervisor"
    }
    
    try:
        final_state = app.invoke(initial_state)
        print("\n" + "="*30)
        print("✅ 最终报告 (由 DeepSeek 生成)")
        print("="*30)
        print(final_state["final_report"])
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
