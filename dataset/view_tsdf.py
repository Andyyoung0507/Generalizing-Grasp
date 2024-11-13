# 利用下述代码可以实现 tsdf 文件可视化（借用浏览器），程序中可以看出轮廓为 梨
import plotly.graph_objs as go
import numpy as np

def visualize_sdf_plotly(sdf, grid_points, level=0.0):
    x, y, z = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]
    values = sdf.flatten()

    # 生成三维等值面
    fig = go.Figure(data=go.Volume(
        x=x, y=y, z=z, value=values,
        isomin=level, isomax=level,
        opacity=0.1, # 表面透明度
        surface_count=1, # 等值面数目
        colorscale='Viridis'
    ))
    fig.update_layout(scene_aspectmode='cube')
    fig.show()

if __name__ == "__main__":
    # 加载SDF文件，实际上是一个鸭梨
    sdf_data = np.load("/home/axe/Downloads/datasets/GraspNet/models/015/grid_sampled_sdf.npz")
    sdf = sdf_data['sdf']
    grid_points = sdf_data['points'].reshape(-1, 3)
    
    # 使用Plotly可视化SDF
    visualize_sdf_plotly(sdf, grid_points)
