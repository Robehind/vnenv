# vnenv
ONGOING

先确认已经正确安装显卡驱动，以及python3.7版本

建议使用conda来安装pytorch 1.4.0，会自动安装相关依赖，包括cuda等，使用清华的源

再用pip安装requirements.txt中的所有其他依赖

当前main.py导入的是demo_arg.py中的参数，因此运行main.py即可测试demo。
需要使用gpu时修改demo_arg.py中的gpu_ids参数即可.

训练结束后会保存模型参数在上级目录的check_points文件夹中，修改demo_arg.py中的load_model_dir参数为模型参数路径，即可运行main_eval.py进行测试。

demo需要的文件见百度云。
链接：https://pan.baidu.com/s/13S9un-5224hVRebNWIX-6w 
提取码：b7x5