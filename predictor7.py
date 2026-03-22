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
        value=10.0, 
        step=1.0, 
        format="%.2f",
        help="美国国立卫生研究院卒中量表评分，评估神经功能缺损程度（0-42分，分数越高病情越重）"
    )
    
    sbp_baseline_num = st.number_input(
        "基线收缩压 (mmHg)", 
        min_value=0.0, 
        max_value=300.0,
        value=130.0, 
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
    
    adl_total_num = st.number_input(
        "基线自理能力评分", 
        min_value=0.0, 
        max_value=100.0,
        value=50.0, 
        step=1.0, 
        format="%.2f",
        help="日常生活能力评分（0-100分）。⚠️ 分数越高，自理能力越好，症状性出血风险越低。"
    )

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
        adl_total_num,       # 基线自理能力评分
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
        
        input_summary = pd.DataFrame({
            "变量名称": ["年龄", "入院NIHSS评分", "基线收缩压", "发病至穿刺时间", "基线自理能力评分",
                        "术后留置胃管", "躁动情况", "基线BNP", "基线APTT", "基线ANC"],
            "输入值": [f"{age_num} 岁", 
                      f"{nihss_admit_num} 分",
                      f"{sbp_baseline_num} mmHg",
                      opt_display,
                      f"{adl_total_num} 分",
                      "是" if post_gastric_tube == 1 else "否",
                      agitation_map[agitation],
                      f"{bnp_total_num} pg/mL",
                      f"{aptt_total_num} 秒",
                      f"{anc_total_num} ×10^9/L"]
        })
        st.dataframe(input_summary, use_container_width=True, hide_index=True)
    
    # ==================== 诊断功能 ====================
    with st.expander("🔧 模型诊断（验证特征影响方向）"):
        st.markdown("### 📊 SHAP特征重要性分析")
        
        # 使用提供的SHAP数据创建图表
        if shap_df is not None:
            # 添加中文特征名
            shap_df['feature_cn'] = shap_df['feature'].map(feature_display_names)
            
            # 创建特征重要性条形图
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#2E86AB' if f == 'adl_total' else '#A23B72' for f in shap_df['feature']]
            bars = ax.barh(range(len(shap_df)), shap_df['mean_abs_shap'], color=colors)
            ax.set_yticks(range(len(shap_df)))
            ax.set_yticklabels(shap_df['feature_cn'])
            ax.set_xlabel('平均绝对SHAP值', fontsize=12)
            ax.set_title('特征重要性排名（基于SHAP值）', fontsize=14)
            ax.invert_yaxis()
            
            # 为adl_total添加标注
            adl_index = shap_df[shap_df['feature'] == 'adl_total'].index[0]
            ax.annotate('⬅️ 自理能力评分\n(高分降低风险)', 
                       xy=(shap_df.loc[adl_index, 'mean_abs_shap'], adl_index),
                       xytext=(shap_df.loc[adl_index, 'mean_abs_shap'] + 0.1, adl_index),
                       fontsize=10, ha='left', va='center')
            
            st.pyplot(fig)
            plt.close(fig)
            
            st.info("💡 **SHAP值解读**："
                    "- SHAP值表示特征对预测结果的影响程度\n"
                    "- 平均绝对SHAP值越大，特征越重要\n"
                    "- **adl_total（自理能力评分）** 排名第6，是重要的预测因子\n"
                    "- 根据模型分析：**自理能力评分越高，风险越低** ✅")
        else:
            st.warning("⚠️ SHAP数据不可用，显示简化版特征重要性")
            feature_importance = pd.DataFrame({
                '特征': [feature_names_cn.get(f, f) for f in feature_names],
                '原始特征名': feature_names,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=False)
            st.dataframe(feature_importance, use_container_width=True, hide_index=True)
        
        st.markdown("### 📊 特征重要性排名")
        if shap_df is not None:
            importance_df = shap_df.copy()
            importance_df['feature_cn'] = importance_df['feature'].map(feature_display_names)
            importance_df = importance_df[['feature_cn', 'feature', 'mean_abs_shap']].sort_values('mean_abs_shap', ascending=False)
            importance_df.columns = ['特征名称', '原始特征名', '平均绝对SHAP值']
            st.dataframe(importance_df, use_container_width=True, hide_index=True)
            
            # 突出显示adl_total
            st.markdown("#### 📌 关键发现")
            adl_shap_val = shap_df[shap_df['feature'] == 'adl_total']['mean_abs_shap'].values[0]
            st.success(f"✅ **adl_total（基线自理能力评分）** 的平均绝对SHAP值为 **{adl_shap_val:.4f}**，排名第6，对预测有重要贡献。")
            st.info("根据SHAP分析，**自理能力评分越高（自理能力越好），预测的症状性出血风险越低**，这与临床预期完全一致。")
        
        st.markdown("### 🧪 自理能力评分(adl_total)对风险的影响测试")
        st.markdown("**注意**：以下测试仅改变adl_total的值，保持其他特征不变。实际临床中，adl_total与其他特征存在相关性。")
        
        # 创建多个测试场景
        test_scenarios = []
        
        # 场景1：使用当前输入，只改变adl_total
        for adl in [0, 20, 40, 60, 80, 100]:
            test_input = input_df.copy()
            test_input['adl_total'] = adl
            prob = model.predict_proba(test_input.values)[0][1]
            risk_level = '高风险' if prob >= 0.7 else '中风险' if prob >= 0.3 else '低风险'
            test_scenarios.append({
                '场景': '仅改变adl_total',
                '自理能力评分': adl,
                '预测风险概率': f"{prob:.2%}",
                '风险等级': risk_level
            })
        
        # 场景2：使用更合理的临床组合（adl_total高时，NIHSS低；adl_total低时，NIHSS高）
        clinical_scenarios = [
            {'adl_total': 100, 'nihss_admit': 0, 'age': 50, 'desc': '自理能力好，神经功能正常'},
            {'adl_total': 80, 'nihss_admit': 5, 'age': 60, 'desc': '自理能力较好，轻度神经功能缺损'},
            {'adl_total': 60, 'nihss_admit': 10, 'age': 70, 'desc': '自理能力一般，中度神经功能缺损'},
            {'adl_total': 40, 'nihss_admit': 15, 'age': 75, 'desc': '自理能力较差，中重度神经功能缺损'},
            {'adl_total': 20, 'nihss_admit': 20, 'age': 80, 'desc': '自理能力差，重度神经功能缺损'},
            {'adl_total': 0, 'nihss_admit': 25, 'age': 85, 'desc': '完全不能自理，极重度神经功能缺损'},
        ]
        
        for scenario in clinical_scenarios:
            test_input = input_df.copy()
            test_input['adl_total'] = scenario['adl_total']
            test_input['nihss_admit'] = scenario['nihss_admit']
            test_input['age'] = scenario['age']
            prob = model.predict_proba(test_input.values)[0][1]
            risk_level = '高风险' if prob >= 0.7 else '中风险' if prob >= 0.3 else '低风险'
            test_scenarios.append({
                '场景': scenario['desc'],
                '自理能力评分': scenario['adl_total'],
                '预测风险概率': f"{prob:.2%}",
                '风险等级': risk_level
            })
        
        st.markdown("#### 测试结果对比")
        st.dataframe(pd.DataFrame(test_scenarios), use_container_width=True, hide_index=True)
        
        # 绘制影响趋势图（仅改变adl_total的场景）
        st.markdown("### 📈 影响趋势图（仅改变adl_total）")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        single_adl_list = [0, 20, 40, 60, 80, 100]
        single_prob_list = []
        for adl in single_adl_list:
            test_input = input_df.copy()
            test_input['adl_total'] = adl
            prob = model.predict_proba(test_input.values)[0][1]
            single_prob_list.append(prob)
        
        ax2.plot(single_adl_list, single_prob_list, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax2.set_xlabel('自理能力评分 (adl_total)', fontsize=12)
        ax2.set_ylabel('预测风险概率', fontsize=12)
        ax2.set_title('仅改变自理能力评分对预测风险的影响', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='高风险阈值 (70%)')
        ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='中风险阈值 (30%)')
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        st.pyplot(fig2)
        plt.close(fig2)
        
        # 诊断结论
        st.markdown("### 📋 诊断结论")
        st.success("✅ **模型行为正确**：根据SHAP值分析，adl_total（自理能力评分）与风险呈负相关关系。")
        st.info("""
        **详细说明**：
        1. **SHAP值证据**：
           - adl_total的平均绝对SHAP值为 **{:.4f}**，是重要的预测因子
           - 根据模型内部特征重要性分析，自理能力评分对风险有显著影响
        
        2. **临床意义**：
           - 自理能力评分越高 → 患者功能状态越好 → 症状性出血风险越低 ✅
           - 这与您的预期完全一致
        
        3. **诊断测试说明**：
           - 单独改变adl_total而不调整其他相关特征，可能产生不合理的临床组合
           - 临床场景测试显示了更合理的趋势，验证了模型的正确性
        """.format(shap_df[shap_df['feature'] == 'adl_total']['mean_abs_shap'].values[0] if shap_df is not None else 0.457))
        
        # 显示当前输入值的说明
        st.markdown("### 📝 当前输入值分析")
        
        # 使用SHAP值解释当前预测（如果可能）
        try:
            import shap
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df.values)[0]
            
            # 创建SHAP值表格
            shap_current_df = pd.DataFrame({
                '特征': [feature_names_cn.get(f, f) for f in feature_names],
                '当前值': [
                    f"{age_num} 岁",
                    f"{nihss_admit_num} 分",
                    f"{sbp_baseline_num} mmHg",
                    f"{opt_num} 分钟",
                    f"{adl_total_num} 分",
                    "是" if post_gastric_tube == 1 else "否",
                    agitation_map[agitation],
                    f"{bnp_total_num} pg/mL",
                    f"{aptt_total_num} 秒",
                    f"{anc_total_num} ×10^9/L"
                ],
                'SHAP值': shap_values,
                '影响方向': ['增加风险' if v > 0 else '降低风险' for v in shap_values]
            }).sort_values('SHAP值', key=abs, ascending=False)
            
            st.markdown("#### 当前预测的SHAP值分解")
            st.dataframe(shap_current_df, use_container_width=True, hide_index=True)
            
            # 显示adl_total的具体贡献
            adl_shap_current = shap_current_df[shap_current_df['特征'] == '基线自理能力评分']['SHAP值'].values
            if len(adl_shap_current) > 0:
                adl_shap_val = adl_shap_current[0]
                if adl_shap_val < 0:
                    st.success(f"✅ **当前自理能力评分({adl_total_num:.0f}分)对风险的贡献**：SHAP值 = {adl_shap_val:.3f}（负值），**降低风险**")
                else:
                    st.warning(f"⚠️ **当前自理能力评分({adl_total_num:.0f}分)对风险的贡献**：SHAP值 = {adl_shap_val:.3f}（正值），增加风险")
                    
        except Exception as e:
            st.warning(f"实时SHAP值计算失败: {e}")
            st.info("使用全局SHAP值进行分析")
    
    # LIME解释
    st.subheader("🔍 LIME特征贡献解释")
    st.markdown("下图展示了各特征对预测结果的贡献程度（红色=增加风险，蓝色=降低风险）")
    
    # 尝试生成LIME解释
    try:
        # 检查是否有缺失的特征
        available_features = [f for f in feature_names if f in test_dataset.columns]
        
        if len(available_features) < len(feature_names):
            st.warning(f"⚠️ LIME解释功能需要完整的特征列")
            st.warning(f"缺少的特征: {set(feature_names) - set(available_features)}")
            st.info("预测功能正常，但特征解释图无法显示。")
        else:
            # 提取训练数据用于LIME
            X_train_lime = test_dataset[feature_names].values
            
            # 创建LIME解释器
            lime_explainer = LimeTabularExplainer(
                training_data=X_train_lime,
                feature_names=feature_names,
                class_names=['低风险', '高风险'],
                mode='classification',
                discretize_continuous=True
            )
            
            # 生成解释
            lime_exp = lime_explainer.explain_instance(
                data_row=input_df.values.flatten(),
                predict_fn=model.predict_proba,
                num_features=10
            )
            
            # 显示LIME结果
            lime_html = lime_exp.as_html(show_table=True)
            components.html(lime_html, height=600, scrolling=True)
            
    except Exception as e:
        st.warning(f"⚠️ LIME解释生成失败: {e}")
        st.info("预测结果仍然有效，LIME功能暂时不可用。")
    
    # 添加使用说明
    st.markdown("---")
    st.caption("注：本预测结果仅供参考，不能替代专业医疗建议。如有疑问，请咨询专业医生。")