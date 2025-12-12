import streamlit as st
import os
import shutil
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from http import HTTPStatus
import dashscope

# --- 页面基本设置 ---
st.set_page_config(page_title="AI 视频逆向提示词工具", layout="wide", page_icon="🎬")

st.title("🎬 AI 视频逆向提示词工具")
st.markdown("""
此工具可以将视频拆解为镜头，并反推 **Runway/Midjourney/Sora** 可用的提示词。
""")

# --- 侧边栏：设置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 尝试从云端机密里读取 Key
    if "ali_key" in st.secrets:
        api_key = st.secrets["ali_key"]
        st.success("已自动加载内置 API Key ✅")
    else:
        # 如果没配置机密（比如你在本地运行），还是允许手动输入
        api_key = st.text_input("请输入阿里 API Key (sk-...)", type="password")
    
    st.divider()
    
    threshold = st.slider("切镜灵敏度 (Threshold)", 10.0, 50.0, 27.0, help="数值越小切分越细")
    
    st.info("💡 提示：分析结果包含摄影参数和中英文 Prompt，可直接复制使用。")

# --- 核心功能函数 ---

def split_video(video_path, output_dir, threshold_val):
    """调用 PySceneDetect 切分视频"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold_val))
    
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    save_images(
        scene_list,
        video,
        num_images=1,
        output_dir=output_dir,
        image_name_template='Shot_$SCENE_NUMBER',
        image_extension='jpg'
    )
    return sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])

def analyze_image_advanced(image_path, api_key):
    """
    调用阿里 AI 进行深度分析
    构造了复杂的 Prompt 以获取 6 维数据 + 提示词
    """
    dashscope.api_key = api_key
    image_url = f"file://{os.path.abspath(image_path)}"
    
    # 🌟 核心修改：这是一个专业的逆向提示词指令
    system_prompt = """
    你是一位精通电影摄影和AI视频生成的专家。请仔细分析这张画面，并按照以下严格格式输出信息。
    不要输出任何多余的开场白，直接输出结果。
    
    请按以下结构分析：
    
    ### 1. 深度分析
    - **景别**：(如：极特写、中景、大远景...)
    - **运镜方式**：(根据画面模糊和构图推测，如：推镜头 Dolly In、由于是静态图请推测可能的运镜，如 缓慢平移 Pan Right...)
    - **环境描述**：(详细描述背景、地点、时间、天气)
    - **主体动作**：(人物或主体的具体行为、表情、姿态)
    - **光照描述**：(如：赛博朋克霓虹光、自然侧逆光、伦勃朗光、色温冷暖)
    - **镜头参数**：(估算风格，如：85mm人像镜头、f/1.8大光圈、移轴摄影效果、胶片颗粒感)

    ### 2. 综合描述
    (请将上述6点融合为一段通顺、极具画面感的文案，约100字)

    ### 3. AI 提示词生成 (Prompt)
    (请生成一段可以直接放入 Runway/Midjourney 的高质量提示词，包含主体、环境、风格、光照、镜头语言关键词。用逗号分隔)
    
    **中文 Prompt**:
    (在此处输出中文提示词)
    
    **English Prompt**:
    (Here output the English prompt, high quality, comma separated tags, photorealistic, 8k, cinematic lighting)
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": system_prompt}
            ]
        }
    ]
    
    try:
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-max', # 使用 Max 模型以获得最好的理解力
            messages=messages
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content[0]['text']
        else:
            return f"Error: {response.message}"
    except Exception as e:
        return f"系统错误: {e}"

def extract_prompts(full_text):
    """简单的辅助函数，尝试从长文本中提取出 Prompt 部分以便单独显示"""
    cn_prompt = ""
    en_prompt = ""
    
    lines = full_text.split('\n')
    for i, line in enumerate(lines):
        if "**中文 Prompt**" in line or "中文 Prompt" in line:
            # 尝试获取下一行
            if i + 1 < len(lines): cn_prompt = lines[i+1].strip()
        if "**English Prompt**" in line or "English Prompt" in line:
            if i + 1 < len(lines): en_prompt = lines[i+1].strip()
            
    return cn_prompt, en_prompt

# --- 主界面逻辑 ---

uploaded_file = st.file_uploader("📂 请上传视频文件 (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(video_path)

    # 🟢 就在这里！这行代码必须是完整的同一行
    if st.button("🚀 开始深度拆解与生成"):
        if not api_key:
            st.error("❌ 请先在左侧侧边栏输入 API Key！")
        else:
            output_folder = "web_output_images"
            
            with st.spinner('✂️ 正在智能切分镜头...'):
                images = split_video(video_path, output_folder, threshold)
            
            st.success(f"✅ 识别到 {len(images)} 个镜头，开始生成提示词...")
            
            progress_bar = st.progress(0)
            results_container = st.container()

            for i, img_name in enumerate(images):
                img_path = os.path.join(output_folder, img_name)
                
                # 1. AI 分析
                full_analysis = analyze_image_advanced(img_path, api_key)
                
                # 尝试提取 Prompt 以便单独放入代码框
                cn_prompt_clean, en_prompt_clean = extract_prompts(full_analysis)
                
                progress_bar.progress((i + 1) / len(images))

                # 2. 界面展示布局
                with results_container:
                    st.markdown(f"### 🎬 镜头 #{i+1}")
                    
                    col1, col2 = st.columns([1, 1.5]) 
                    
                    with col1:
                        st.image(img_path, use_column_width=True, caption=f"关键帧: {img_name}")
                    
                    with col2:
                        # 使用 expander 收纳详细分析
                        with st.expander("📊 查看 6 维深度分析 (点击展开)", expanded=True):
                            # 粗略分割，防止显示过多重复内容
                            display_text = full_analysis.split("### 3. AI 提示词生成")[0]
                            st.markdown(display_text)

                        # 专门的提示词区域
                        st.markdown("#### 📋 AI 提示词 (可直接复制)")
                        
                        st.caption("中文 Prompt:")
                        if cn_prompt_clean:
                            st.code(cn_prompt_clean, language="None")
                        else:
                            st.warning("自动提取失败，请查看上方分析")
