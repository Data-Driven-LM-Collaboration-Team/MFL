# 本机I3 NF4：四层联合DPC距离单因素试验

9片45个配对位完成V3实际完整时段源图与新输出配对；新增9/9严格有效、9次LLM调用，0失败/重试，无FATAL。J125未使完整异常对象对应all提高，类别从A的4/6降为3/6，不采用、不扩展。

## 条件和唯一因素

社区unsloth/Qwen3-VL-8B-Instruct-bnb-4bit revision10d842936325905934ea24e64325d774693f166e，16文件7242509964字节，NF4 double quant/BF16 compute；RTX3080 20GB WSL、torch2.8.0+cu128/transformers5.19.0/bnb0.50.2/accelerate1.11.0。与官方服务器权重/平台分开，不混分。
只将DPC候选池内cosine距离由main改为main与三个对齐DeepStack单位归一化后距离的固定等权平均；main时间新颖度/候选池、首尾时间组、stride16空间底线、每组预算、DPC密度/delta/索引平局、merge半径2和main相似度阈值.7均不变。四层实际仍用同groups分别平均，不把距离拼接向量送模型。
全部视频帧仍编码；原I3原生局部图、Prompt/schema、采样seed、8192输出/32768context、图片和文字全部保留，无参考/目标框/旧答案输入。父A/C125/S/U36终态只读继承，本轮只新增J9调用。

## 配对指标

固定六片工程异常统计any/all/event/category，三正常单列系统误报；不是完整167新准确率，不与旧49%直接相减。同聊天非盲审，human_verified/训练资格false；固定189/167/3/19、原参考/manual不改。

|臂|any/6|all/6|event/6|category/6|异常片错绑/6|技术失败/9|正常系统误报/3|
|---|---:|---:|---:|---:|---:|---:|---:|
|A量化不压缩|2|0|3|4|6|0|3|
|C125主层距离均值|4|0|5|3|5|0|3|
|S125原中心向量|4|0|5|3|5|0|3|
|U125均匀原向量|2|0|3|4|4|0|3|
|J125四层联合距离均值|4|0|5|3|4|0|3|

any/event相对A有所增加、错绑由6片降至4片，但all仍0/6，且类别下降；不能称压缩改善完整对象定位。与同预算C125相比，J的any/all/event/category相同，错绑少1片，仍不满足采用。

## 全部研究成本和实际预算

|臂|实际视觉Token|实际LLM调用|输出Token|全部LLM Token|终态端到端总秒|峰值allocated/reserved MiB|
|---|---:|---:|---:|---:|---:|---:|
|A量化不压缩|67164|9|4534|94761|254.388|11231.03/11878.00|
|C125主层距离均值|17280|9|4125|44468|219.853|9105.06/9678.00|
|S125原中心向量|17280|9|5417|45760|281.461|9105.06/9678.00|
|U125均匀原向量|17280|9|3530|43873|191.323|9105.06/9678.00|
|J125四层联合距离均值|17280|9|5453|45796|293.160|9105.06/9678.00|

J实际视觉17280，对A67164减少74.27%；逐片与C125预算完全相同，实际中心变化9/9片、groups变化9/9片。不是口头声明换距离，实际路由/特征/距离哈希及组变化已留痕。
新增9调用共5453输出/45796全部LLM Token，99成本源文件逐Token连续/数量/CPU解码==RAW、INPUT_RECEIPT/压缩输入/GENERATION_START/RESPONSE/记录一致。J视觉准备16.197秒，生成266.004秒，完整终态elapsed合计293.160秒。
父36费用仅作同条件比较、不重复计入新增；工程CPU预检、加载、GPU门禁、等待另存，不伪称整次实验全部开销。A原生编码包含在generation，压缩路径另测准备，不能直接用两种generation值作同义速度比较；所有帧先编码，不能宣称视觉编码FLOPs降低74%。

## 新J逐片反例

1. 全窗肢体对抗与唯一黑衣施力者可对应，any/event通过；缺浅衣长裤施动者且条纹旁观者own异常，另虚构强取/失稳和走廊G06政策，all/category失败。原初始接触manual保留。
2. 唯一真实后段攀墙者可对应，时间提前另留；额外狗攻击、杆击、跌倒无图支持，all/category失败。不把类码或后段攀墙补给虚构早期攻击。
3. 赤膊施动者正确理解为破坏；货架/散物是受影响物，不因关联事件就自身own异常，额外正向绑定使all失败。重复G02不算两次成功。
4. 理解禁骑/G06，但泛指浅衣骑车不能在两名浅衣骑手间唯一定位；any/all失败，正文事件与类别独立通过。三条长椅人群own均false，不额外算错绑。
5. 红衣和唯一黑衣骑手正确；泛指白T背包白车不能覆盖两个以下装区分的真实白衣骑手，漏核心者，all失败。重复同一实际异常者不自动计无关对象错绑。黑衣方向以已有更正源事实为准，不改原参考。
6. 将F023-37喷液/泄漏当G04，源图无所称早期喷溅，漏后段真实火烟；不能借之后火情追认早期液体事件，ground受影响物也非自身异常。
7. 正常稳定拾取被当危险抛物，白红普通行人还被own异常；一臂一clip误报，不按对象数累加。
8. 正常可控跳转与普通路人被当跌倒/碰撞风险，未见失稳；人物提前入画错误单列。
9. 正常可控落地屈膝被当跌倒并虚构帮助站起，G06仅凭黄线/锻炼无危险交通支持；白衣ownfalse不另加错绑。
kind/role枚举本身不扣分；骑手/车复合描述若唯一可通过。真实事件理解、对象唯一对应、类别代码分开；正常非空events冻结派生ANOMALY照记，不事后修NORMAL。时间误差单留、不声称时间IoU。

## 当前工程证明

freeze099c0b31dd52706ce205db00ed12445c1ece917eae3cc84f9c56aeb055740215、35成员；39本地/39WSL CPU、779来源依赖、父36/402文件、16模型和9全部实际Tensor SHA与父完全一致。原非距离merge数学AST逐项一致。手算反例main不变/DeepStack改变导致中心21变26；同四层退化与非有限/错位拒绝真实RED→GREEN留痕。
新lease93f17fcb2fad475da8267d92f091984e，GPU-2cc76af1-0caf-68ad-f174-cb7b941a4541；252语言NF4，四加载错误空。短mixed131/最长真实I3 13388、四比例×4decode共32全词logits/rawBF16/cache/DeepStack/mRoPE/full-retention/dense-after通过，父dense逐值一致。工程peakallocated13034.70/reserved14078MiB。
proof.pt SHA041e7987c3a9c06a768b8bd9ca5bdee963b001335b5f96771408a94bed6bbbaa；proof JSON SHAcddc855ca5e4f4a50ba5dbd076909020e4b5388cee27485a0a68bfbfce1d88da。独立CPU子进程26707在05:44:15.775完成38archive/32步回读，CUDA未初始化；硬屏障成功05:44:16.833，首GENERATE05:44:21.241，其后每调用检查当前freeze/lease/GPU/证明/CPU进程匹配。不是研究后补审计。4decode不等于8192最大输出证明。
05:49:17研究结束，05:49:20队列退出释放租约，06:01实际/proc进程空，无FATAL。退出时worker visual_pending是历史时点，最终九片报告为当前评价事实。

## 决定与接续

J不通过all净增加与category不降护栏，无候选/扩展false。重复均值/选择/新颖度/四层距离在此9片未改善完整绑定；当前证据不足以继续无限扫描层权重或预算。下一步先完成独立官方B阶段分配复核；社区与官方保持控制和成本分开。小试完成不等于长期根Goal完成，原生Goal仍paused。
当前v002仅修正继承S/U父文件中相对父试验的new_call标志，v001原字节保留，所有裁决/输出/分母未改；最终比较中只有J9为新增。
机器入口：SMALL_TRIAL_DECISION.json、ADJUDICATION_INDEX.json及九v002、ACTUAL_SAME_BUDGET_AUDIT.json、FINAL_RESEARCH_COST_CPU_AUDIT.json、FINAL_INDEPENDENT_HASH_READBACK.json。原RAW/Token流/receipt在outputs，终态records。
状态只读：C:/Users/10495/anaconda3/python.exe -B validation/local_control.py status；成本CPU复核：/mnt/l/VAD/envs/qwen3vl8b_nf4_linux/bin/python -B evaluation_only/cost_reconciliation/audit_completed.py。
