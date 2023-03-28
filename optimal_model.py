# -*- coding: utf-8 -*-
"""

@author: ASUS
"""

import xgboost
import shap
import json
import warnings
warnings.filterwarnings("ignore")
shap.initjs()
import pandas as pd
df = pd.read_excel('mydata.xlsx')
df.sort_values(by=['id', 'year'], ascending=[True, True], inplace=True, ignore_index=True)
df_new = df[~df.year.isin([2002, 2003, 2004, 2005])].reset_index().rename(columns={'index': 'original_index'})
for i in range(len(df_new)):
    df_new.HE.iloc[i] = df.HE.iloc[df_new.original_index.iloc[i]-4]
    df_new.EMSC.iloc[i] = df.EMSC.iloc[df_new.original_index.iloc[i]-4]  

X = df_new.iloc[:,6:]
y = df_new['rate']

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
 
# 参数集定义
param_grid = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8], #  基本学习器的深度： 基本学习器的数量： 要拟合的弱学习器数量，该值越大，模型越复杂，越容易过拟合树的最大深度，该值越大，模型越复杂，越容易拟合训练数据，越容易过拟合;树生长停止条件之一
            'n_estimators': [30, 50, 100, 300, 500, 1000,2000], # 基本学习器的数量： 要拟合的弱学习器数量，该值越大，模型越复杂，越容易过拟合
            'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.03, 0.05, 0.5],# 学习率：每个基模型的惩罚项，降低单个模型的影响，为了防止过拟合，该值越接近1越容易或拟合，越接近0精度越低
            "gamma":[0.0, 0.1, 0.2, 0.3, 0.4],#损失减少阈值： 在树的叶节点上进一步划分所需的最小损失减少，在模型训练过程中，只有损失下降的值超过该值，才会继续分裂节点，该值越小模型越复杂，越容易过拟合,树生长停止条件之一
            "reg_alpha":[0.0001,0.001, 0.01, 0.1, 1, 100],# L1正则化：L1正则化用于对叶子的个数进行惩罚,用于防止过拟合
            "reg_lambda":[0.0001,0.001, 0.01, 0.1, 1, 100],# L2正则化:L2正则化用于对叶子节点的得分进行惩罚，L1和L2正则化项共同惩罚树的复杂度,值越小模型的鲁棒性越高
            "min_child_weight": [2,3,4,5,6,7,8],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "subsample":[0.6, 0.7, 0.8, 0.9]}
# 随机搜索并打印最佳参数
gsearch1 = RandomizedSearchCV(XGBRegressor(), param_grid, cv=4)

gsearch1.fit(X, y)

print("best_score_:",gsearch1.best_params_,gsearch1.best_score_)

best_model = xgboost.XGBRegressor(**gsearch1.best_params_)
model = best_model.fit(X, y)
mean_squared_error(y,model.predict(X))

best_model = xgboost.XGBRegressor()
model = best_model.fit(X, y)
mean_squared_error(y,model.predict(X))

#使用SHAP解释模型预测
model = best_model.fit(X, y)
explainer = shap.Explainershap_values = explainer(X)
shap.summary_plot(shap_values, X)

# 使用SHAP解释模型预测
model = best_model.fit(X, y)
explainer = shap.Explainer(model)
shap_values = explainer(X)
shap_values
import ipywidgets as widgets
import pyecharts.options as opts
from pyecharts.charts import Pie

df_mean = df_new.groupby('id').mean().iloc[:,5:]

combination_hill = pd.DataFrame(df_new.groupby('id').sum().iloc[:,5:].loc[[2,3,7,8]].sum()/48).T
combination_hill

combination_lake = pd.DataFrame(df_new.groupby('id').sum().iloc[:,5:].loc[[1,4,5,6,9,10,11]].sum()/48).T
combination_lake

#选择的样本某年的解释：以环形图形式可视化

x_data = list(df_new)[6:]
x_data_mean = [i+" Mean" for i in x_data]
y_data = list(map(abs, combination_1_4_5_6.values.tolist()[0]))
y_data = [i for i in y_data]

data_pair = [list(z) for z in zip(x_data_mean, y_data)]
data_pair.sort(key=lambda x: x[1])
pie = (
    Pie(init_opts=opts.InitOpts())
    .add(
        series_name=f"Feature Mean of id=1, 4, 5, 6",
        data_pair=data_pair,
        rosetype="radius",
        radius=["50%", "70%"],
        label_opts=opts.LabelOpts(is_show=False, position="center"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"湖沼型流行区干预优先级可视化",
            pos_left="center",
            pos_top="20",
            title_textstyle_opts=opts.TextStyleOpts(color="#2a4d69"),
        ),
        legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
    )
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
        label_opts=opts.LabelOpts(),
    )
)
pie.render_notebook()

# 选择的样本某年的解释：以环形图形式可视化
x_data = list(df_new)[6:]
x_data_mean = [i+" Mean" for i in x_data]
y_data = list(map(abs, combination_2_3_7_8.values.tolist()[0]))
y_data = [i for i in y_data]

data_pair = [list(z) for z in zip(x_data_mean, y_data)]
data_pair.sort(key=lambda x: x[1])
pie = (
    Pie(init_opts=opts.InitOpts())
    .add(
        series_name=f"Feature Mean of id=2, 3, 7, 8",
        data_pair=data_pair,
        rosetype="radius",
        radius=["50%", "70%"],
        label_opts=opts.LabelOpts(is_show=False, position="center"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"山丘型流行区干预优先级可视化",
            pos_left="center",
            pos_top="20",
            title_textstyle_opts=opts.TextStyleOpts(color="#2a4d69"),
        ),
        legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
    )
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
        label_opts=opts.LabelOpts(),
    )
)
pie.render_notebook()


# 选择的样本某年的解释：以环形图形式可视化

x_data = list(df_new)[6:]
x_data_mean = [i+" Mean" for i in x_data]
y_data = list(map(abs, df_mean.loc[[sample_id_for_mean.value]].values.tolist()[0]))
y_data = [i for i in y_data]

data_pair = [list(z) for z in zip(x_data_mean, y_data)]
data_pair.sort(key=lambda x: x[1])
pie = (
    Pie(init_opts=opts.InitOpts())
    .add(
        series_name=f"Feature Mean of id={sample_id_for_mean.value}",
        data_pair=data_pair,
        rosetype="radius",
        radius=["60%", "80%"],
        label_opts=opts.LabelOpts(is_show=False, position="center"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            #title=f"id为{sample_id_for_mean.value}的特征均值可视化",
            pos_left="center",
            pos_top="20",
            title_textstyle_opts=opts.TextStyleOpts(color="#2a4d69"),
        ),
        legend_opts=opts.LegendOpts(type_="scroll", pos_left="90%", orient="vertical"),
    )
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
        label_opts=opts.LabelOpts(),
    )
)
pie.render_notebook()

# 所有样本的解释：以力图形式可视化
shap.force_plot(explainer.expected_value, shap_values.values, X)

from itertools import combinations
feature_list = list(df_new)[6:]

combinations_dict = {}
for n in range(3,10):
    combinations_dict[n] = []
    for i in combinations(feature_list, n):
        combinations_dict[n].append(list(i))
combinations_dict

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse_dict = {}
r2_dict = {}
rmse_dict = {}
mae_dict = {}
for n in combinations_dict.keys():
    mse_dict[n] = []
    r2_dict[n] = []
    rmse_dict[n] = []
    mae_dict[n] = []
    for i in combinations_dict[n]:
        X_compare = df_new[i]
        model_compare = xgboost.XGBRegressor()
        model_compare = model_compare.fit(X_compare, y)
        y_predict = model_compare.predict(X_compare)
        mse = mean_squared_error(y, y_predict)
        r2 = r2_score(y, y_predict)
        rmse = mean_squared_error(y, y_predict, squared=False)
        mae = mean_absolute_error(y, y_predict)
        mse_dict[n].append(mse)
        r2_dict[n].append(r2)
        rmse_dict[n].append(rmse)
        mae_dict[n].append(mae)
# mse_dict

min_mse_list = []
max_r2_list = []
for i in mse_dict.keys():
    min_mse = min(mse_dict[i])
    max_r2 = max(r2_dict[i])
    min_mse_combination_features = combinations_dict[i][mse_dict[i].index(min(mse_dict[i]))]
    max_r2_combination_features = combinations_dict[i][r2_dict[i].index(max(r2_dict[i]))]
    min_mse_list.append([min_mse, min_mse_combination_features])
    max_r2_list.append([max_r2, max_r2_combination_features])
    print("*"*8 + f"选取{i}个特征" + "*"*8)
    print(f"最小MSE为: {min_mse}\n特征组合为: {min_mse_combination_features}\n")
    print(f"最大R2为: {max_r2}\n特征组合为: {max_r2_combination_features}\n")


import math
def scaling(param_list, param_str):
    if param_str == 'MSE':
        result_list = [10000/(math.log(i)*(-1)) for i in param_list]
    elif param_str == 'R2':
        result_list = [abs(math.log(1-i)) * 10 for i in param_list]
    elif param_str == 'MAE':
        result_list = [100/(math.log(i)*(-1)) for i in param_list]
    elif param_str == 'RMSE':
        result_list = [100/(math.log(i)*(-1)) for i in param_list]
    return result_list
    
def draw_nested_pies(param_dict, param_str):
    inner_x_data4 = [str(i) for i in combinations_dict[4]]
    inner_y_data4 = scaling(param_dict[4], param_str)

    inner_data_pair4 = [list(z) for z in zip(inner_x_data4, inner_y_data4)]
    inner_data_pair4.sort(key=lambda x: x[1])

    inner_x_data5 = [str(i) for i in combinations_dict[5]]
    inner_y_data5 = scaling(param_dict[5], param_str)
    inner_data_pair5 = [list(z) for z in zip(inner_x_data5, inner_y_data5)]
    inner_data_pair5.sort(key=lambda x: x[1])

    inner_x_data6 = [str(i) for i in combinations_dict[6]]
    inner_y_data6 = scaling(param_dict[6], param_str)
    inner_data_pair6 = [list(z) for z in zip(inner_x_data6, inner_y_data6)]
    inner_data_pair6.sort(key=lambda x: x[1])

    inner_x_data7 = [str(i) for i in combinations_dict[7]]
    inner_y_data7 = scaling(param_dict[7], param_str)
    inner_data_pair7 = [list(z) for z in zip(inner_x_data7, inner_y_data7)]
    inner_data_pair7.sort(key=lambda x: x[1])

    inner_x_data8 = [str(i) for i in combinations_dict[8]]
    inner_y_data8 = scaling(param_dict[8], param_str)
    inner_data_pair8 = [list(z) for z in zip(inner_x_data8, inner_y_data8)]
    inner_data_pair8.sort(key=lambda x: x[1])

    inner_x_data9 = [str(i) for i in combinations_dict[9]]
    inner_y_data9 = scaling(param_dict[9], param_str)
    inner_data_pair9 = [list(z) for z in zip(inner_x_data9, inner_y_data9)]
    inner_data_pair9.sort(key=lambda x: x[1])

    outer_x_data = [str(i) for i in combinations_dict[3]]
    outer_y_data = scaling(param_dict[3], param_str)
    outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]
    outer_data_pair.sort(key=lambda x: x[1])
    if param_str == 'MSE':
        bar_max = 900
        bar_min = 700
    elif param_str == 'R2':
        bar_max = 88
        bar_min = 70
    elif param_str == 'MAE':
        bar_max = 18
        bar_min = 13
    elif param_str == 'RMSE':
        bar_max = 20
        bar_min = 13.9
    c = (
        Pie(init_opts=opts.InitOpts(
            width="900px",
            height="700px",)
           )
        .add(
            series_name=f"{param_str} —— 三三特征组",
            center=["50%", "50%"],
            rosetype="radius",
            radius=["60%", "70%"],
            data_pair=outer_data_pair,
        )

        .add(
            series_name=f"{param_str} —— 四四特征组",
            center=["50%", "50%"],
            rosetype="radius",
            data_pair=inner_data_pair4,
            radius=["50%", "60%"],
        )
        .add(
            series_name=f"{param_str} —— 五五特征组",
            center=["50%", "50%"],
            rosetype="radius",
            data_pair=inner_data_pair5,
            radius=["40%", "50%"],
        )
        .add(
            series_name=f"{param_str} —— 六六特征组",
            center=["50%", "50%"],
            rosetype="radius",
            data_pair=inner_data_pair6,
            radius=["30%", "40%"],
        )
        .add(
            series_name=f"{param_str} —— 七七特征组",
            center=["50%", "50%"],
            rosetype="radius",
            data_pair=inner_data_pair7,
            radius=["20%", "30%"],
        )
        .add(
            series_name=f"{param_str} —— 八八特征组",
            center=["50%", "50%"],
            rosetype="radius",
            data_pair=inner_data_pair8,
            radius=["10%", "20%"],
        )
        .add(
            series_name=f"{param_str} —— 九九特征组",
            center=["50%", "50%"],
            rosetype="radius",
            data_pair=inner_data_pair9,
            radius=["0%", "10%"],

        )

        .set_global_opts(
            title_opts=opts.TitleOpts(
               # title=f"{param_str}可视化",
                pos_left="45%",
                pos_top="40",
                title_textstyle_opts=opts.TextStyleOpts(color="#2a4d69"),
            ),
            legend_opts=opts.LegendOpts(is_show=False, type_="scroll", pos_left="60%", orient="vertical"),
            visualmap_opts=opts.VisualMapOpts(
                is_show=True,
                max_=bar_max,
                min_=bar_min,
                range_text=[param_str, ''],
                item_width=50,
                item_height=150,
                textstyle_opts=opts.TextStyleOpts(color="#FFF", font_size=18),
                pos_left="85%",
                pos_top="10%"
            ),
        )

        .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter="{a} <br/>{b}: {c}"
            ),
            label_opts=opts.LabelOpts(is_show=False)
        )
    )

    return c
#     c.render_notebook()

pie_r2 = draw_nested_pies(r2_dict, 'R2')
pie_r2.render_notebook()

r2_dict
r2_list = []
for i in r2_dict.keys():
    for j in range(len(r2_dict[i])):
        r2_list.append([r2_dict[i][j], combinations_dict[i][j]])
r2_list
y_axis = []
cost = []
label = []
for i in r2_list:
    temp_X = X[i[1]]
    cost.append(temp_X.mean().sum())
    y_axis.append(i[0])
    label.append(str(i[1]))
data_rmse = {'cost': cost,
             'r2': y_axis,
             'label': label}
df_r2 = pd.DataFrame(data_rmse)

pie_rmse = draw_nested_pies(rmse_dict, 'RMSE')
pie_rmse.render_notebook()

pie_mse = draw_nested_pies(mse_dict, 'MSE')
pie_mse.render_notebook()

pie_mae = draw_nested_pies(mae_dict, 'MAE')
pie_mae.render_notebook()

shap_RS = abs(shap_values.values[:, -1]).tolist()
shap_normal = np.average(abs(shap_values.values[:, :-1]), axis=1).tolist()
RS = df_new['RS'].tolist()
benefit = [i*j for i, j in zip(shap_RS, RS)]
loss = [i*j for i, j in zip(shap_normal, RS)]
RS_related = pd.DataFrame({'id': df_new['id'], 'year': df_new['year'], 'RS cost': RS, 'shap_RS': shap_RS, 
                            'shap_normal': shap_normal, 'benefit': benefit, 'loss': loss})

y_axis = []
cost = []
label = []
for i in max_r2_list:
    temp_X = X[i[1]]
    cost.append(temp_X.mean().sum())
    y_axis.append(i[0])
    label.append(str(i[1]))
    
data_draw = {'cost': cost,
             'R2': y_axis,
             'label': label}
df_draw = pd.DataFrame(data_draw)
df_draw

# library
import seaborn as sns
import pandas as pd
import numpy as np
data_draw = {'cost': cost,
             'R2': pre_y,
             'label': label}
