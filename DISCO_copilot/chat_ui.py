import streamlit as st
import os
import json
import time
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from real_tools import real_structure_builder, real_energy_calculator

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="DISCO-Pilot Chat",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS: Hide Streamlit default elements for a cleaner look
st.markdown("""
<style>
    /* Hide top Header */
    header {visibility: hidden;}
    /* Hide bottom Footer */
    footer {visibility: hidden;}
    
    /* Adjust chat message style (optional) */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar: Settings (Avatar & Key)
# ==========================================
with st.sidebar:
    st.title("⚙️ Settings")
    
    # 1. API Key
    env_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_key = st.text_input("DeepSeek API Key", value=env_key, type="password")
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 🖼️ Avatar Settings")
    
    # Load local avatars
    current_dir = os.path.dirname(os.path.abspath(__file__))
    user_img_path = os.path.join(current_dir, "user.png")
    agent_img_path = os.path.join(current_dir, "agent.png")

    # 2. User Avatar
    if os.path.exists(user_img_path):
        user_avatar = user_img_path
        st.sidebar.image(user_avatar, width=50, caption="User Avatar")
    else:
        user_avatar = "👤"
    
    # 3. AI Avatar
    if os.path.exists(agent_img_path):
        ai_avatar = agent_img_path
        st.sidebar.image(ai_avatar, width=50, caption="AI Avatar")
    else:
        ai_avatar = "🤖"

# ==========================================
# Logic Functions
# ==========================================
def get_llm():
    return ChatOpenAI(
        model="deepseek-chat", 
        openai_api_key=os.environ.get("DEEPSEEK_API_KEY"), 
        openai_api_base="https://api.deepseek.com",
        temperature=0.1
    )

def run_supervisor_planning(user_request):
    llm = get_llm()
    prompt = f"""
    You are a computational chemistry research supervisor. The user's request is: "{user_request}"
    
    Please extract the Surface and Adsorbate the user wants to study.
    Then, based on chemical knowledge, list the adsorption Sites to be tested on that surface.
    For fcc(111) surface, we focus on stable sites: top, fcc (hollow), hcp (hollow). 
    (Note: Bridge sites often relax to hollow sites, so we skip them for efficiency).
    
    Please return strictly in JSON format, do not include Markdown formatting markers, format as follows:
    {{
        "surface": "Pt(111)",
        "adsorbate": "O",
        "sites": ["top", "fcc", "hcp"]
    }}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
    return json.loads(content)

def generate_report(user_request, results):
    llm = get_llm()
    results_str = ""
    for res in results:
        if res.get("energy") is not None:
            results_str += f"- Site {res['site']}: Energy {res['energy']:.4f} eV\n"
        else:
            results_str += f"- Site {res['site']}: Calculation Failed\n"

    prompt = f"""
    All calculations are completed. Please write a short scientific report based on the following results, indicating which site is the most stable (lowest energy).
    Please use a professional academic tone. Do not use Markdown headers (#), just use paragraphs.
    
    Task: {user_request}
    Results:
    {results_str}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ==========================================
# Chat Main Logic
# ==========================================

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome Message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I am DISCO-Pilot, your computational chemistry assistant. What system would you like to study?\n\nExample: *Study the adsorption configuration of O atom on Pt(111) surface*"
    })

if "current_plan" not in st.session_state:
    st.session_state.current_plan = None # Store plan pending confirmation
if "user_request_cache" not in st.session_state:
    st.session_state.user_request_cache = ""

# 1. Display History Messages
for msg in st.session_state.messages:
    # Choose avatar based on role
    avatar = user_avatar if msg["role"] == "user" else ai_avatar
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 2. Handle User Input
if prompt := st.chat_input("Enter instructions here..."):
    
    # --- Display User Message ---
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- AI Processing Logic ---
    with st.chat_message("assistant", avatar=ai_avatar):
        
        # Case A: User is confirming the plan (entered y/yes/ok)
        if st.session_state.current_plan and prompt.lower() in ["y", "yes", "ok", "confirm", "agree"]:
            
            plan = st.session_state.current_plan
            user_req = st.session_state.user_request_cache
            
            # 1. Execution Process (Using st.status for collapsible display)
            results = []
            with st.status("🚀 Executing Workflow...", expanded=True) as status:
                
                # --- Phase 1: Modeling ---
                st.write("🗣️ **Supervisor** -> **Modeling Agent**: Build structures.")
                
                modeled_tasks = []
                for site in plan["sites"]:
                    st.code(f"real_structure_builder(surface='{plan['surface']}', adsorbate='{plan['adsorbate']}', site='{site}')", language="python")
                    path = real_structure_builder(plan["surface"], plan["adsorbate"], site)
                    
                    if path:
                        filename = os.path.basename(path)
                        # st.caption(f"✅ Created: `{filename}`")
                        modeled_tasks.append({"site": site, "path": path})
                    else:
                        st.error(f"❌ Failed: {site}")
                    time.sleep(0.5) 

                # --- Phase 2: Calculation ---
                st.write("🗣️ **Supervisor** -> **Calculation Agent**: Optimize & Calculate Energy.")
                
                for task in modeled_tasks:
                    site = task["site"]
                    path = task["path"]
                    filename = os.path.basename(path)
                    
                    st.code(f"real_energy_calculator(structure_path='.../{filename}')", language="python")
                    energy = real_energy_calculator(path)
                    
                    if energy is not None:
                        # st.caption(f"✅ Energy: **{energy:.4f} eV**")
                        results.append({"site": site, "energy": energy, "path": path})
                    else:
                        st.error(f"❌ Failed: {site}")
                        results.append({"site": site, "energy": None})
                    time.sleep(0.5)

                status.update(label="✅ Done", state="complete", expanded=False)
            
            # 2. Generate Report
            with st.spinner("Drafting analysis report..."):
                report_text = generate_report(user_req, results)
                
                # Construct Final Response
                final_response = f"### 📊 Research Report\n\n{report_text}\n\n**Data Summary:**"
                st.markdown(final_response)
                
                # Display a small table
                df = pd.DataFrame(results)
                if not df.empty:
                    # Custom HTML table for larger font size
                    st.markdown("#### 📉 Calculation Results")
                    rows_html = ""
                    for _, row in df.iterrows():
                        val = f"{row['energy']:.4f}" if row['energy'] is not None else "Failed"
                        rows_html += f"<tr style='border-bottom: 1px solid #eee;'><td style='padding:4px 12px;'>{row['site']}</td><td style='padding:4px 12px;'>{val} eV</td></tr>"
                    
                    st.markdown(f"""
                    <div style="font-size: 1.0rem; margin-bottom: 1rem;">
                        <table style="width:auto; min-width: 40%; border-collapse: collapse;">
                            <tr style="border-bottom: 2px solid #ccc; background-color: #f8f9fa;">
                                <th style="text-align:left; padding:4px 12px;">Site</th>
                                <th style="text-align:left; padding:4px 12px;">Energy</th>
                            </tr>
                            {rows_html}
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Log to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_response + "\n\n(Data table displayed)"
                })
            
            # Clear plan state, ready for next conversation
            st.session_state.current_plan = None
            st.session_state.user_request_cache = ""

        # Case B: User rejected the plan
        elif st.session_state.current_plan and prompt.lower() in ["n", "no", "cancel"]:
            response = "Okay, task cancelled. Please tell me your new research goal."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.current_plan = None

        # Case C: New Task Request (Supervisor Planning)
        else:
            if not api_key:
                st.error("Please set API Key in the sidebar first")
            else:
                with st.spinner("Supervisor is planning..."):
                    try:
                        plan = run_supervisor_planning(prompt)
                        
                        # Construct Response Content
                        response_md = f"""
Received. I have formulated the following research plan for you:

- **Surface Model**: `{plan['surface']}`
- **Adsorbate**: `{plan['adsorbate']}`
- **Sites to Calculate**: `{', '.join(plan['sites'])}`

If confirmed, please enter **"y"** or **"yes"** to start execution.
"""
                        st.markdown(response_md)
                        
                        # Log State
                        st.session_state.messages.append({"role": "assistant", "content": response_md})
                        st.session_state.current_plan = plan
                        st.session_state.user_request_cache = prompt
                        
                    except Exception as e:
                        st.error(f"Planning failed: {e}")
