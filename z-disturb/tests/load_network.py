import pandapower as pp
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg（必须在导入pyplot之前设置）
import matplotlib.pyplot as plt
from pandapower.plotting import simple_plot
import os
net = pp.from_pickle(os.path.join(os.path.dirname(__file__), './z-disturb/tests/assets/model.p'))

# 创建网络图
simple_plot(net,
    plot_loads=True,        # 是否显示负载
    plot_sgens=True,        # 是否显示静态发电机
    plot_line_switches=True,  # 是否显示线路开关
    
            )

# 保存图片
plt.savefig('network_plot.png', dpi=300, bbox_inches='tight')
plt.close()
