# boundary_segmented_3dpointspoints
1. 通过点云法向量过滤噪点
2. 通过1得出的点云集合使用ransac提取平面
3. 依赖2得到的提取平面在x方向最远得100个点作为分界
