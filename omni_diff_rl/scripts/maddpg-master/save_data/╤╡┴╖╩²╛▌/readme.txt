失败作品
	policy_1和policy_2的神经网络动力学模型与默认策略的相同：没有+=，observation传入的是绝对位置
	policy_3和policy_4的神经网络动力学模型与默认策略的不相同：神经网络换成+=，observation传入的是相对位置

保存的rewards均为追逐者pursur的reward  
policy_25 对应模型为policy_5.1


policy_13----policy_6   加速度：4：2    绝对位置    训练时初始距离为（-4.5，4.5） ----成功模型
policy_12----policy_12   加速度：4：2.4  绝对位置绝对速度  训练时初始距离为（-5.21，5.21）
policy_11 (realitive)---policy_11 (realitive)(policy_5_success)   加速度：4：2  初始距离为随机（-4,4）采用相对位置,相对速度训练  符合预期效果
policy_11 ---policy_5_success   加速度：4：2  初始距离为随机（-4,4）采用相对位置训练   符合预期效果 ----成功模型
policy_10 (relative) ---policy_10 (relative)(policy_5)   加速度：4：2.4  初始距离为随机（-4,4）采用相对位置,相对速度训练   符合预期效果
policy_10  ---policy_5   加速度：4：2.4  初始距离为随机（-4,4）采用相对位置训练   符合预期效果  ----成功模型
policy_20----policy_20    加速度：4：2.4 
policy_21----policy_21    加速度：4：2.4 
policy_22----policy_22    加速度：4：2.4     符合预期
policy_25----policy_25（5.1）    加速度：4：2 相对位置训练，展示的时候用的都是绝对位置

以下为神经vs神经，试图找到刚好追不上的策略，以200step以上表示追不上，初始训练距离均为(-10,10)
policy_26----policy_26    加速度：4：3.75， 250steps
policy_27----policy_27    加速度：4：3.7，  208steps
policy_28----policy_28    加速度：4：3.6，  164steps

policy_30----policy_30    加速度：1：0.75   231steps
policy_31----policy_31    加速度：1：0.7    210steps
policy_32----policy_32    加速度：1：0.6    164steps

policy_33----policy_33    加速度：7:6.8     292steps
policy_34----policy_34    加速度：7:6.6     165steps
policy_35----policy_35    加速度：7:6.7     214steps

policy_36----policy_36    加速度：10:9      128steps
policy_37----policy_37    加速度：10:9.1    135steps

policy_38----policy_38    加速度：13：12.4，132steps
policy_39----policy_39    加速度：13：12.5，235steps
policy_40----policy_40    加速度：13：12.3，102steps、

policy_41----policy_41    加速度：9：8，    145steps
policy_42----policy_42    加速度：9：8.5，  249steps
policy_43----policy_43    加速度：9：8.4，  184steps
policy_44----policy_44    加速度：9:8.3     169steps

policy_50----policy_50    加速度：2：1.5， 
policy_51----policy_51    加速度：2：1.6，   
policy_52----policy_52    加速度：2：1.7，   205steps

policy_53----policy_53    加速度：6：5.5，   227steps
policy_54----policy_54    加速度：6：5.4

policy_57----policy_57    加速度：8：7.4
policy_58----policy_58    加速度：8：7.5
policy_59----policy_59    加速度：8：7.6    224steps

policy_60----policy_60    加速度：12：11.3 
policy_61----policy_61    加速度：12：11.4   400steps未追上 

policy_62----policy_62    加速度：3:2.5      138steps
policy_63----policy_63    加速度：3:2.6      212steps

policy_64----policy_64    加速度：5:4.4      117steps
policy_65----policy_65    加速度：5:4.7      203steps
policy_66----policy_66    加速度：5:4.6      163steps

policy_67----policy_67    加速度11：10.5     231steps
policy_68----policy_68    加速度11：10.4     400steps未追上
policy_69----policy_69    加速度11：10.3     196steps

policy_70----policy_70    加速度10：9.4      400steps未追上
policy_71----policy_71    加速度10：9.4      162steps

policy_72----policy_72    加速度14：13.4      
policy_73----policy_73    加速度14：13.5
policy_74----policy_74    加速度14：13.3     101steps
policy_75----policy_75    加速度14：13.3     141steps
policy_76----policy_76    加速度14：12.8     76steps

policy_80----policy_80    加速度：13：12.4 
policy_81----policy_81    加速度：13：11.5   400steps未追上 

policy_82----policy_82    加速度：15：14.4 
policy_83----policy_83    加速度：15：14.5   400steps未追上 
policy_84----policy_84    加速度：15：14.3   352steps
policy_85----policy_85    加速度：15：14     201steps
policy_86----policy_86    加速度：15：13.7     66steps




policy_continueTest 为连续情况的训练模型

2020.12.13
policy_888 4:2.4,连续情况，p网络最后一层激活函数为tanh，q网络为none，p网络加了0.5的dropout  相对位置训练

2020.12.14
policy_999 4:2,连续情况，p网络最后一层激活函数为tanh，q网络为none。 相对位置训练

2020.12.15
policy_222  4:3 连续情况，p网络最后一层激活函数为tanh，q网络为none。 相对位置训练，神经网络可追上














