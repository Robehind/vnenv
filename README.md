# vnenv
ONGOING

先确认已经正确安装显卡驱动，以及python3.7版本

建议使用conda来安装pytorch 1.4.0，会自动安装相关依赖，包括cuda等，使用清华的源

再用pip安装requirements.txt中的所有其他依赖

当前main.py导入的是demo_arg.py中的参数，因此运行main.py即可测试demo。
需要使用gpu时修改demo_arg.py中的gpu_ids参数即可.

训练结束后会保存模型参数在上级目录的check_points文件夹中，修改demo_arg.py中的load_model_dir参数为模型参数路径，即可运行main_eval.py进行测试。

5.8 更新
新加入了一个可视化脚本
修改了计算SPL的bug，但是在431以外的房间，如果设置了8方向移动或者90度以上的旋转动作，spl的计算会不准确
因此在431以外的房间建议使用基础动作：前进，左转45度，右转45度，向上看45度，向下看45度
另外431房间是之前target-driven数据转换而来，因此只支持90度的旋转，不支持45度旋转，也不支持上下看
而且grid_size为0.5，但431以外的房间grid_size都为0.25，设置参数文件时要尤其注意

demo需要的数据文件见如下百度云。(5.8已更新)
链接：https://pan.baidu.com/s/1iSq-P5e8Guc5-k5H0RbdHg 
提取码：26h6 