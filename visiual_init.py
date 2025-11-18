import open3d as o3d

# 读取点云数据
pcd = o3d.io.read_point_cloud("arrow.ply")

# 创建可视化工具
vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window(window_name='Open3D', visible=True)
vis.add_geometry(pcd)
vis.run()

# 获取所选点的信息
# points = vis.get_picked_points()
# vis.destroy_window()