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
    st.success("✅ 成功加载模型")
except FileNotFoundError:
    st.error("❌ xgboost.pkl 文件未找到，请确保该文件存在于项目目录中")
    st.stop()
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.stop()

# 读取 CSV 数据文件
try:
    test_dataset = pd.read_csv('data.csv')
    st.success("✅ 成功加载 data.csv")
except FileNotFoundError:
    st.error("❌ data.csv 文件未找到，请确保该文件存在于项目目录中")
    st.stop()
except Exception as e:
    st.error(f"❌ 数据文件加载失败: {e}")
    st.stop()

# 读取SHAP值数据
try:
    shap_df = pd.read_csv('shap_values.csv')
    st.success("✅ 成功加载 shap_values.csv")
except FileNotFoundError:
    st.warning("⚠️ shap_values.csv 文件未找到，将使用简化版诊断")
    shap_df = None
except Exception as e:
    st.warning(f"⚠️ SHAP文件加载失败: {e}")
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

# 特征英文到中文的映射（用于图表）
feature_display_names = {
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

# 检查数据中是否包含所有需要的特征列
missing_features = [f for f in feature_names if f not in test_dataset.columns]
if missing_features:
    st.warning(f"⚠️ 数据文件中缺少以下特征列: {missing_features}")
    st.info("LIME解释功能可能无法正常工作，但预测功能仍然可用。")
else:
    st.info(f"✅ 数据包含所有 {len(feature_names)} 个特征列")

st.title("急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取风险评估结果")

# 在输入区域添加转换选项
with st.expander("⚙️ 高级设置"):
    transform_adl = st.checkbox(
        "转换自理能力评分（如果高分表示自理能力差）", 
        value=True,
        help="根据数据分析，训练数据中adl_total可能是反向编码（高分表示自理能力差）。勾选此项可自动转换。"
    )
    
    if transform_adl:
        st.info("💡 **已启用转换**：输入的自理能力评分将转换为：转换后 = 100 - 原始值")
        st.caption("例如：原始输入100分（完全自理）→ 转换后0分（表示自理能力差）")

# 输入组件 - 按照指定顺序排列
col1, col2 = st.columns(2)

with col1:
    age_num = st.number_input(
        "年龄 (岁)", 
        min_value=0.0, 
        max_value=120.0,
        value=70.0, 
        step=1.0, 
        format="%.2f",
        help="患者年龄"
    )
    
    nihss_admit_num = st.number_input(
        "入院NIHSS评分", 
        min_value=0.0, 
        max_value=42.0,
        value=0.0, 
        step=1.0, 
        format="%.2f",
        help="美国国立卫生研究院卒中量表评分，评估神经功能缺损程度（0-42分，分数越高病情越重）"
    )
    
    sbp_baseline_num = st.number_input(
        "基线收缩压 (mmHg)", 
        min_value=0.0, 
        max_value=300.0,
        value=167.0, 
        step=1.0, 
        format="%.2f",
        help="入院时测量的收缩压值"
    )
    
    opt_num = st.number_input(
        "发病至穿刺时间 (分钟)", 
        min_value=0.0, 
        max_value=1440.0,
        value=300.0, 
        step=5.0, 
        format="%.2f",
        help="从发病到血管穿刺的时间，单位：分钟"
    )
    
    adl_total_input = st.number_input(
        "基线自理能力评分", 
        min_value=0.0, 
        max_value=100.0,
        value=100.0, 
        step=1.0, 
        format="%.2f",
        help="日常生活能力评分（0-100分）。⚠️ 根据模型需要，高分可能表示自理能力差"
    )
    
    # 应用转换
    if transform_adl:
        adl_total_num = 100 - adl_total_input
        st.caption(f"⚠️ 已转换：原始输入 {adl_total_input:.0f} → 模型使用 {adl_total_num:.0f}")
    else:
        adl_total_num = adl_total_input

with col2:
    post_gastric_tube = st.selectbox(
        "术后是否留置胃管",
        options=[0, 1],
        format_func=lambda x: "是" if x == 1 else "否",
        help="术后是否需要留置胃管"
    )
    
    agitation = st.selectbox(
        "术后躁动情况",
        options=[0, 1, 2, 3],
        format_func=lambda x: agitation_map[x],
        help="术后患者躁动程度评估"
    )
    
    bnp_total_num = st.number_input(
        "基线BNP (pg/mL)", 
        min_value=0.0, 
        value=100.0, 
        step=10.0, 
        format="%.2f",
        help="脑钠肽水平"
    )
    
    aptt_total_num = st.number_input(
        "基线APTT (秒)", 
        min_value=0.0, 
        value=35.0, 
        step=1.0, 
        format="%.2f",
        help="活化部分凝血活酶时间"
    )
    
    anc_total_num = st.number_input(
        "基线中性粒细胞计数 (×10^9/L)", 
        min_value=0.0, 
        value=7.0, 
        step=0.5, 
        format="%.2f",
        help="中性粒细胞绝对值计数"
    )

# 预测按钮
if st.button("预测", type="primary"):
    # 按指定顺序组装输入值
    feature_values = [
        age_num,             # 年龄
        nihss_admit_num,     # 入院NIHSS评分
        sbp_baseline_num,    # 基线收缩压
        opt_num,             # OPT (发病至穿刺时间)
        adl_total_num,       # 基线自理能力评分（已转换）
        post_gastric_tube,   # 术后留置胃管
        agitation,           # 躁动
        bnp_total_num,       # 基线BNP
        aptt_total_num,      # 基线APTT
        anc_total_num        # 基线ANC
    ]
    
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # 模型预测
    try:
        proba = model.predict_proba(input_df)[0]
        risk_prob = proba[1]  # 高风险概率
    except Exception as e:
        st.error(f"模型预测失败: {e}")
        st.stop()
    
    # 根据阈值划分风险等级 (3:7 = 30% 和 70%)
    if risk_prob < 0.30:
        pred_class = "低风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于低风险。建议继续保持当前治疗方案，定期随访。"
        risk_color = "green"
        risk_level = "低"
    elif risk_prob < 0.70:
        pred_class = "中风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于中风险。建议密切观察，遵医嘱进行相关检查。"
        risk_color = "orange"
        risk_level = "中"
    else:
        pred_class = "高风险"
        advice = f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于高风险。建议立即就医，加强监测和预防措施。"
        risk_color = "red"
        risk_level = "高"
    
    # 显示预测结果
    st.markdown("---")
    st.subheader("📊 预测结果")
    
    # 风险等级卡片
    col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
    with col_result2:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {risk_color}20; border: 1px solid {risk_color};'>
            <h2 style='color: {risk_color};'>风险分类: {pred_class}</h2>
            <h3 style='color: {risk_color};'>风险概率: {risk_prob:.2%}</h3>
            <p style='color: {risk_color};'>风险等级: {risk_level}风险 (阈值: 30% / 70%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 健康建议
    st.subheader("💡 健康建议")
    st.info(advice)
    
    # 风险提示条
    st.subheader("📈 风险可视化")
    
    # 使用三列布局
    col1_metric, col2_metric, col3_metric = st.columns([2, 1, 1])
    with col1_metric:
        st.metric("风险概率", f"{risk_prob:.1%}", delta=None)
        progress_width = int(risk_prob * 100)
        st.markdown(f"""
        <div style="width: 100%; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
            <div style="width: {progress_width}%; background-color: {risk_color}; border-radius: 5px; padding: 3px 0; text-align: center; color: white;">
                {risk_prob:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2_metric:
        st.markdown("**阈值说明**")
    with col3_metric:
        st.markdown("🔵 低风险: <30%<br>🟡 中风险: 30%-70%<br>🔴 高风险: >70%", unsafe_allow_html=True)
    
    # 显示输入摘要
    with st.expander("查看输入信息摘要"):
        opt_hours = opt_num / 60
        opt_display = f"{opt_num:.0f} 分钟 ({opt_hours:.1f} 小时)"
        
        # 显示原始输入和转换后的值
        input_summary = pd.DataFrame({
            "变量名称": ["年龄", "入院NIHSS评分", "基线收缩压", "发病至穿刺时间", "基线自理能力评分",
                        "术后留置胃管", "躁动情况", "基线BNP", "基线APTT", "基线ANC"],
            "输入值": [f"{age_num} 岁", 
                      f"{nihss_admit_num} 分",
                      f"{sbp_baseline_num} mmHg",
                      opt_display,
                      f"{adl_total_input:.0f} 分" + (f" → 转换后: {adl_total_num:.0f} 分" if transform_adl else ""),
                      "是" if post_gastric_tube == 1 else "否",
                      agitation_map[agitation],
                      f"{bnp_total_num} pg/mL",
                      f"{aptt_total_num} 秒",
                      f"{anc_total_num} ×10^9/L"]
        })
        st.dataframe(input_summary, use_container_width=True, hide_index=True)
    
    # ==================== 诊断功能 ====================
    with st.expander("🔧 模型诊断（验证特征影响方向）"):
        st.markdown("### 📊 训练数据分析")
        
        # 检查训练数据中adl_total的分布
        if 'adl_total' in test_dataset.columns and 'diagnosis' in test_dataset.columns:
            high_risk = test_dataset[test_dataset['diagnosis'] == 1]['adl_total']
            low_risk = test_dataset[test_dataset['diagnosis'] == 0]['adl_total']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("高风险组平均adl_total", f"{high_risk.mean():.2f}")
                st.metric("高风险组中位数adl_total", f"{high_risk.median():.2f}")
            with col2:
                st.metric("低风险组平均adl_total", f"{low_risk.mean():.2f}")
                st.metric("低风险组中位数adl_total", f"{low_risk.median():.2f}")
            
            # 判断数据方向
            if high_risk.mean() > low_risk.mean():
                st.warning("⚠️ **训练数据显示**：高风险组平均adl_total > 低风险组")
                st.info("这证实了训练数据中adl_total是反向编码（高分表示自理能力差）")
                if transform_adl:
                    st.success("✅ 已启用adl_total转换，输入将自动转换")
                else:
                    st.warning("⚠️ 建议启用adl_total转换以获得正确结果")
            else:
                st.success("✅ 训练数据正确：高风险组平均adl_total < 低风险组")
                if transform_adl:
                    st.warning("⚠️ 训练数据正确，但您启用了转换。建议关闭转换选项")
        
        st.markdown("### 📊 特征重要性排名")
        if shap_df is not None:
            shap_df['feature_cn'] = shap_df['feature'].map(feature_display_names)
            importance_df = shap_df[['feature_cn', 'feature', 'mean_abs_shap']].sort_values('mean_abs_shap', ascending=False)
            importance_df.columns = ['特征名称', '原始特征名', '平均绝对SHAP值']
            st.dataframe(importance_df, use_container_width=True, hide_index=True)
            
            # 显示adl_total的SHAP值
            adl_shap = shap_df[shap_df['feature'] == 'adl_total']['mean_abs_shap'].values
            if len(adl_shap) > 0:
                st.info(f"**adl_total（自理能力评分）** 的平均绝对SHAP值为 **{adl_shap[0]:.4f}**")
        
        st.markdown("### 🧪 测试不同adl_total值的影响")
        st.markdown("测试当前输入下，改变adl_total对预测结果的影响：")
        
        test_adl_values = [0, 20, 40, 60, 80, 100]
        test_results = []
        
        for adl in test_adl_values:
            test_input = input_df.copy()
            test_input['adl_total'] = adl
            prob = model.predict_proba(test_input.values)[0][1]
            risk_level = '高风险' if prob >= 0.7 else '中风险' if prob >= 0.3 else '低风险'
            test_results.append({
                'adl_total值': adl,
                '预测风险概率': f"{prob:.2%}",
                '风险等级': risk_level
            })
        
        st.dataframe(pd.DataFrame(test_results), use_container_width=True, hide_index=True)
        
        # 绘制趋势图
        fig, ax = plt.subplots(figsize=(10, 6))
        adl_list = [r['adl_total值'] for r in test_results]
        prob_list = [float(r['预测风险概率'].strip('%')) / 100 for r in test_results]
        
        ax.plot(adl_list, prob_list, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('adl_total (自理能力评分)', fontsize=12)
        ax.set_ylabel('预测风险概率', fontsize=12)
        ax.set_title('adl_total与预测风险概率的关系', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='高风险阈值 (70%)')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='中风险阈值 (30%)')
        ax.legend()
        ax.set_ylim(0, 1)
        
        st.pyplot(fig)
        plt.close(fig)
        
        # 判断趋势
        if prob_list[0] > prob_list[-1]:
            st.success("✅ **趋势正确**：adl_total越高，风险越低")
            if transform_adl:
                st.info("注意：由于启用了转换，输入的高分（自理能力好）会被转换为低分（自理能力差），从而正确反映风险关系。")
        else:
            st.error("❌ **趋势错误**：adl_total越高，风险越高")
            st.warning("请检查adl_total转换设置是否正确")
    
    # LIME解释
    st.subheader("🔍 LIME特征贡献解释")
    st.markdown("下图展示了各特征对预测结果的贡献程度（红色=增加风险，蓝色=降低风险）")
    
    # 尝试生成LIME解释
    try:
        available_features = [f for f in feature_names if f in test_dataset.columns]
        
        if len(available_features) < len(feature_names):
            st.warning(f"⚠️ LIME解释功能需要完整的特征列")
            st.warning(f"缺少的特征: {set(feature_names) - set(available_features)}")
            st.info("预测功能正常，但特征解释图无法显示。")
        else:
            X_train_lime = test_dataset[feature_names].values
            
            lime_explainer = LimeTabularExplainer(
                training_data=X_train_lime,
                feature_names=feature_names,
                class_names=['低风险', '高风险'],
                mode='classification',
                discretize_continuous=True
            )
            
            lime_exp = lime_explainer.explain_instance(
                data_row=input_df.values.flatten(),
                predict_fn=model.predict_proba,
                num_features=10
            )
            
            lime_html = lime_exp.as_html(show_table=True)
            components.html(lime_html, height=600, scrolling=True)
            
    except Exception as e:
        st.warning(f"⚠️ LIME解释生成失败: {e}")
        st.info("预测结果仍然有效，LIME功能暂时不可用。")
    
    # 添加使用说明
    st.markdown("---")
    st.caption("注：本预测结果仅供参考，不能替代专业医疗建议。如有疑问，请咨询专业医生。")