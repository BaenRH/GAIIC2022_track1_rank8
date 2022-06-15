# GAIIC2022_track1_rank8
[2022人工智能技术创新大赛-赛道1-电商关键属性匹配](https://www.heywhale.com/org/gaiic2022/competition/area/620b34c41f3cf500170bd6ca/content)  NEZHA-LSTM 方案
## 成绩
- 初赛 rank4
- 复赛 rank8
另外队友

[ZhongYupei](https://github.com/ZhongYupei/GAIIC2022_track1_rank8)

[JoanSF]()

## 文件说明

    |--- helper
        |--- build_optimizer.py                                   # 构建优化器等
        |--- coarse_unmatch.py                                    # 分离coarse data中图文不匹配数据   
        |--- FGM.py                                               # fgm
        |--- transform_coarse_to_fine.py                          # coarse转fine
        |--- transform_txt_to_json.py                             # txt文件转json
    |--- new-base
        |--- vocab.txt                                            # 新词表
    |--- output                                                   # 预测文件输出
    |--- save_finetune                                            # finetune保存路径
    |--- save_pretrain                                            # pretrain保存路径
    |--- AdaptiveLoss.py                                          # pretrain 损失
    |--- check_weight.py                                          # 计算模型参数量
    |--- datasets.py                                              # dataset
    |--- attr_to_attrvals.json                                    # 属性表                                       
    |--- model_nezha.py                                           # 模型
    |--- file_utils.py                                            # 模型相关函数
    |--- pretrain.py                                              # Pretrain
    |--- run_match.py                                             # Finetun
    |--- run_kfold.py                                             # k折
    |--- test.py                                                  # 测试
    |--- test_2model.py                                           # 测试（图文与其他12个属性分开预测）
    |--- test_kfold.py                                            # k折测试
