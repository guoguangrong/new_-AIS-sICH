# 导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置（必须在最前面）
st.set_page_config(
    page_title="急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器",
    layout="wide"
)

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('xgboost.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ xgboost.pkl 文件未找到，请确保该文件存在于项目目录中")
    st.stop()
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.stop()

# 读取 CSV 数据文件
try:
    test_dataset = pd.read_csv('data.csv')
except FileNotFoundError:
    st.error("❌ data.csv 文件未找到，请确保该文件存在于项目目录中")
    st.stop()
except Exception as e:
    st.error(f"❌ 数据文件加载失败: {e}")
    st.stop()

# 读取SHAP值数据
try:
    shap_df = pd.read_csv('shap_values.csv')
except FileNotFoundError:
    shap_df = None
except Exception as e:
    shap_df = None

# 按照指定顺序排列10个特征变量
feature_names = [
    "age",               # 年龄
    "nihss_admit",       # 入院NIHSS评分
    "sbp_baseline",      # 基线收缩压
    "opt",               # OPT (发病至穿刺时间)
    "adl_total",         # 基线自理能力评分
    "post_gastric_tube", # 术后留置胃管
    "agitation",         # 躁动
    "bnp_total",         # 基线BNP
    "aptt_total",        # 基线APTT
    "anc_total"          # 基线ANC
]

# 特征中文名称映射
feature_names_cn = {
    "age": "年龄",
    "nihss_admit": "入院NIHSS评分",
    "sbp_baseline": "基线收缩压",
    "opt": "发病至穿刺时间",
    "adl_total": "基线自理能力评分",
    "post_gastric_tube": "术后留置胃管",
    "agitation": "躁动情况",
    "bnp_total": "基线BNP",
    "aptt_total": "基线APTT",
    "anc_total": "基线ANC"
}

# 躁动情况映射
agitation_map = {0: "无躁动", 1: "轻度躁动", 2: "中度躁动", 3: "重度躁动"}

# 自定义CSS样式
st.markdown("""
<style>
    /* 输入框标签字体加大 */
    .stNumberInput label, .stSelectbox label {
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    /* 输入框数值字体加大 */
    .stNumberInput input {
        font-size: 18px !important;
    }
    
    /* 预测结果卡片样式 */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    
    .prediction-title {
        font-size: 24px;
        font-weight: bold;
        color: white;
        margin-bottom: 15px;
    }
    
    .prediction-prob {
        font-size: 48px;
        font-weight: bold;
        color: white;
        margin: 20px 0;
    }
    
    .prediction-level {
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }
    
    .prediction-advice {
        font-size: 16px;
        color: white;
        background: rgba(255,255,255,0.2);
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
    }
    
    /* 分隔线样式 */
    hr {
        margin: 20px 0;
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
        transition: all 0.3s;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 22px;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .metric-unit {
        font-size: 14px;
        color: #95a5a6;
        margin-left: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取风险评估结果")
st.markdown("---")

# 在高级设置中添加转换选项
with st.sidebar:
    st.markdown("### ⚙️ 设置")
    transform_adl = st.checkbox(
        "转换自理能力评分", 
        value=True,
        help="如果训练数据中高分表示自理能力差，勾选此项自动转换"
    )
    
    show_diagnosis = st.checkbox(
        "显示模型诊断信息", 
        value=False,
        help="勾选后显示特征重要性、趋势测试等诊断信息"
    )
    
    if transform_adl:
        st.info("💡 已启用adl_total转换")

# 创建左右两列布局
left_col, right_col = st.columns([1.2, 0.8])

with left_col:
    # 第一行指标
    col1, col2 = st.columns(2)
    
    with col1:
        age_num = st.number_input(
            "年龄 (岁)", 
            min_value=0.0, 
            max_value=120.0,
            value=89.0, 
            step=1.0, 
            format="%.0f"
        )
        
        nihss_admit_num = st.number_input(
            "入院NIHSS评分 (分)", 
            min_value=0.0, 
            max_value=42.0,
            value=23.0, 
            step=1.0, 
            format="%.0f"
        )
        
        sbp_baseline_num = st.number_input(
            "基线收缩压 (mmHg)", 
            min_value=0.0, 
            max_value=300.0,
            value=167.0, 
            step=1.0, 
            format="%.0f"
        )
        
        opt_num = st.number_input(
            "发病至穿刺时间 (分钟)", 
            min_value=0.0, 
            max_value=1440.0,
            value=300.0, 
            step=5.0, 
            format="%.0f"
        )
        
        adl_total_input = st.number_input(
            "基线自理能力评分 (分)", 
            min_value=0.0, 
            max_value=100.0,
            value=0.0, 
            step=1.0, 
            format="%.0f"
        )
        
        if transform_adl:
            adl_total_num = 100 - adl_total_input
        else:
            adl_total_num = adl_total_input
    
    with col2:
        post_gastric_tube = st.selectbox(
            "术后是否留置胃管",
            options=[0, 1],
            format_func=lambda x: "是" if x == 1 else "否"
        )
        
        agitation = st.selectbox(
            "术后躁动情况",
            options=[0, 1, 2, 3],
            format_func=lambda x: agitation_map[x]
        )
        
        bnp_total_num = st.number_input(
            "基线BNP (pg/mL)", 
            min_value=0.0, 
            value=100.0, 
            step=10.0, 
            format="%.0f"
        )
        
        aptt_total_num = st.number_input(
            "基线APTT (秒)", 
            min_value=0.0, 
            value=35.0, 
            step=1.0, 
            format="%.1f"
        )
        
        anc_total_num = st.number_input(
            "基线中性粒细胞计数 (×10^9/L)", 
            min_value=0.0, 
            value=5.0, 
            step=0.5, 
            format="%.1f"
        )
    
    # 预测按钮
    st.markdown("---")
    predict_btn = st.button("预测", type="primary", use_container_width=True)

with right_col:
    # 预测结果显示区域
    st.markdown("### 📊 预测结果")
    
    # 默认显示等待预测的占位符
    prediction_placeholder = st.empty()
    
    if predict_btn:
        # 按指定顺序组装输入值
        feature_values = [
            age_num,
            nihss_admit_num,
            sbp_baseline_num,
            opt_num,
            adl_total_num,
            post_gastric_tube,
            agitation,
            bnp_total_num,
            aptt_total_num,
            anc_total_num
        ]
        
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # 模型预测
        try:
            proba = model.predict_proba(input_df)[0]
            risk_prob = proba[1]
        except Exception as e:
            st.error(f"模型预测失败: {e}")
            st.stop()
        
        # 根据阈值划分风险等级
        if risk_prob < 0.30:
            pred_class = "低风险"
            advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于低风险。建议继续保持当前治疗方案，定期随访。"
            risk_class = "risk-low"
        elif risk_prob < 0.70:
            pred_class = "中风险"
            advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于中风险。建议密切观察，遵医嘱进行相关检查。"
            risk_class = "risk-medium"
        else:
            pred_class = "高风险"
            advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于高风险。建议立即就医，加强监测和预防措施。"
            risk_class = "risk-high"
        
        # 显示预测结果卡片
        prediction_placeholder.markdown(f"""
        <div class="prediction-card {risk_class}">
            <div class="prediction-title">📈 风险评估结果</div>
            <div class="prediction-prob">{risk_prob:.1%}</div>
            <div class="prediction-level">{pred_class}</div>
            <div class="prediction-advice">
                💡 {advice}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示风险阈值说明
        st.markdown("""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 15px; margin-top: 15px;">
            <div style="font-size: 14px; color: #6c757d; margin-bottom: 10px;">风险阈值说明</div>
            <div style="display: flex; justify-content: space-between;">
                <div><span style="color: #11998e;">●</span> 低风险: &lt;30%</div>
                <div><span style="color: #f5576c;">●</span> 中风险: 30%-70%</div>
                <div><span style="color: #eb3349;">●</span> 高风险: &gt;70%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 如果启用了转换，显示转换说明
        if transform_adl:
            st.caption(f"💡 注：自理能力评分已转换 ({adl_total_input:.0f} → {adl_total_num:.0f})")
        
        # 显示输入摘要（折叠）
        with st.expander("查看输入信息摘要"):
            opt_hours = opt_num / 60
            opt_display = f"{opt_num:.0f} 分钟 ({opt_hours:.1f} 小时)"
            
            input_summary = pd.DataFrame({
                "变量名称": ["年龄", "入院NIHSS评分", "基线收缩压", "发病至穿刺时间", "基线自理能力评分",
                            "术后留置胃管", "躁动情况", "基线BNP", "基线APTT", "基线ANC"],
                "输入值": [f"{age_num:.0f} 岁", 
                          f"{nihss_admit_num:.0f} 分",
                          f"{sbp_baseline_num:.0f} mmHg",
                          opt_display,
                          f"{adl_total_input:.0f} 分",
                          "是" if post_gastric_tube == 1 else "否",
                          agitation_map[agitation],
                          f"{bnp_total_num:.0f} pg/mL",
                          f"{aptt_total_num:.1f} 秒",
                          f"{anc_total_num:.1f} ×10^9/L"]
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)
        
        # ==================== 诊断功能（仅在开关打开时显示） ====================
        if show_diagnosis:
            st.markdown("---")
            st.markdown("### 🔧 模型诊断")
            
            # 检查训练数据中adl_total的分布
            if 'adl_total' in test_dataset.columns and 'diagnosis' in test_dataset.columns:
                high_risk = test_dataset[test_dataset['diagnosis'] == 1]['adl_total']
                low_risk = test_dataset[test_dataset['diagnosis'] == 0]['adl_total']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("高风险组平均adl_total", f"{high_risk.mean():.2f}")
                with col2:
                    st.metric("低风险组平均adl_total", f"{low_risk.mean():.2f}")
            
            # 测试不同adl_total值的影响
            st.markdown("#### adl_total影响测试")
            test_adl_values = [0, 20, 40, 60, 80, 100]
            test_results = []
            
            for adl in test_adl_values:
                test_input = input_df.copy()
                test_input['adl_total'] = adl
                prob = model.predict_proba(test_input.values)[0][1]
                test_results.append({
                    'adl_total值': adl,
                    '风险概率': f"{prob:.2%}"
                })
            
            st.dataframe(pd.DataFrame(test_results), use_container_width=True, hide_index=True)
            
            # 绘制趋势图
            fig, ax = plt.subplots(figsize=(8, 4))
            adl_list = [r['adl_total值'] for r in test_results]
            prob_list = [float(r['风险概率'].strip('%')) / 100 for r in test_results]
            
            ax.plot(adl_list, prob_list, marker='o', linewidth=2, markersize=6, color='steelblue')
            ax.set_xlabel('adl_total', fontsize=10)
            ax.set_ylabel('风险概率', fontsize=10)
            ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)
    
    else:
        # 未预测时的占位符
        prediction_placeholder.markdown("""
        <div style="background: #f8f9fa; border-radius: 20px; padding: 60px 30px; text-align: center; border: 2px dashed #dee2e6;">
            <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
            <div style="font-size: 18px; color: #6c757d;">填写左侧信息后点击"预测"按钮</div>
            <div style="font-size: 14px; color: #adb5bd; margin-top: 10px;">将在此处显示风险评估结果</div>
        </div>
        """, unsafe_allow_html=True)

# 添加使用说明
st.markdown("---")
st.caption("注：本预测结果仅供参考，不能替代专业医疗建议。如有疑问，请咨询专业医生。")