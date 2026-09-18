import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

#  页面配置
st.set_page_config(
    page_title="房地产数据分析驾驶舱",
    page_icon="🏠",
    layout='wide',
    initial_sidebar_state='expanded'
)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据生成（模拟）

@st.cache_data(ttl=3600)
def generate_house_data(n=2000):
    """生成模拟的二手房交易数据"""
    np.random.seed(2024)
    # 数值型特征
    area = np.random.normal(130,30,n).astype(int)
    # print(area)
    area = np.clip(area,60,250)
    bedrooms = np.random.choice([1,2,3,4,5],n,p=[0.05,0.30,0.40,0.20,0.05])
    bathrooms = np.random.choice([1,2,3],n,p=[0.30,0.55,0.15])

    age = np.random.exponential(10,n).astype(int)
    age = np.clip(age,0,40)
    distance_to_center = np.round(np.random.exponential(4,n) + 0.5,1)

    # 分类特征
    districts = np.random.choice(['东城','西城','南城','北城'],n,p=[0.30,0.25,0.20,0.25])
    floors = np.random.choice(['低','中','高'],n,p=[0.30,0.40,0.30])
    renovations = np.random.choice(['毛坯','简装','精装','豪装'],n,p=[0.10,0.30,0.40,0.20])

    # 系数映射
    districts_coef = {'东城':80,'西城':40,'南城':20,'北城':60}
    floors_coef = {'低':-15,'中':0,'高':20}
    reno_coef = {'毛坯':-30,'简装':-10,'精装':10,'豪装':40}

    district_factor = np.array([districts_coef[d] for d in districts])
    floor_factor = np.array([floors_coef[d] for d in floors])
    reno_factor = np.array([reno_coef[d] for d in renovations])

    # 生成价格 （线性组合 + 噪声）
    base_price = ( area * 1.5 + bedrooms * 10 + bathrooms * 8
      - age * 1.2 - distance_to_center * 5 +
      district_factor + floor_factor + reno_factor
      )
    noise = np.random.normal(0,0.12*base_price)
    price = np.round(base_price + noise,0).astype(int)
    price = np.clip(price,50,800)

    df = pd.DataFrame({
        'price':price,
        'area':area,
        'bedrooms':bedrooms,
        'age':age,
        'distance_to_center':distance_to_center,
        'floor':floors,
        'district':districts,
        'renovation':renovations,
    })
    return df

# 加载数据

with st.spinner('正在加载模拟数据'):
    df = generate_house_data()
st.sidebar.title('筛选控制')
# 城区多选
selected_districts = st.sidebar.multiselect('城区',
       options = sorted(df['district'].unique()),
       default=sorted(df['district'].unique())
)
# 装修
selected_renovations = st.sidebar.multiselect('装修程度',
    options=sorted(df['renovation'].unique()),
    default=sorted(df['renovation'].unique()),
)
# 楼层多选
selected_floors = st.sidebar.multiselect('装修程度',
    options=sorted(df['floor'].unique()),
    default=sorted(df['floor'].unique()),
)
# 价格范围
price_min,price_max = int(df['price'].min()),int(df['price'].max())
price_range = st.sidebar.slider("总价范围（万元）",
                  min_value=price_min,
                  max_value=price_max,
                  value=(price_min,price_max)
)
# 面积范围
area_min,area_max = int(df['area'].min()),int(df['area'].max())
area_range = st.sidebar.slider("面积范围（㎡）",
                  min_value=area_min,
                  max_value=area_max,
                  value=(area_min,area_max)
)
# 房龄范围
age_min,age_max = int(df['age'].min()),int(df['age'].max())
age_range = st.sidebar.slider("房龄范围（年）",
                  min_value=age_min,
                  max_value=age_max,
                  value=(age_min,age_max)
)
# 应用筛选
filtered_df = df[
    (df['district'].isin(selected_districts)) &
    (df['renovation'].isin(selected_renovations)) &
    (df['floor'].isin(selected_floors)) &
    (df['price'] >=price_range[0])  &
    (df['price'] <= price_range[1]) &
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1]) &
    (df['area'] >= area_range[0]) &
    (df['area'] <= area_range[1])
]

# KPI 指标卡片
st.title('房地产市场价格分析驾驶舱')

col1,col2,col3,col4 = st.columns(4)
with col1:
    st.metric('总房源数',f'{len(filtered_df)}套')

with col2:
    st.metric('平均价格', f'{filtered_df["price"].mean():.0f}万元')

with col3:
    st.metric('中位价格', f'{filtered_df["price"].median():.0f}万元')


with col4:
    most_expensive = filtered_df.groupby('district')['price'].mean().idxmax()
    st.metric("最贵的区域",most_expensive)
st.divider()

# 图表区域









