# 这里记录遇到的问题

## 20261123

## 之前的

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torchvision 0.20.1+cu121 requires torch==2.5.1+cu121, but you have torch 2.15.0+cpu which is incompatible.
这个是大电脑上遇到的，笔记本没有；我没搞明白这块的环境

## 2142

(.venv) D:\mutimodel_demo>python scripts\enroll_face.py --name alice --camera 0 --num-samples 20
2026-11-23 23:01:59 | ERROR | enroll_face | 模型初始化失败: module 'onnxruntime' has no attribute 'get_available_providers'

## 2159

标题打错了！是multimodel...

## 2246

应该弄一个一键安装环境
