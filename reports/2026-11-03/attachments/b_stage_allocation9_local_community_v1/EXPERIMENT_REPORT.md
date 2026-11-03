# 本机I3 NF4：B两阶段相同总预算分配试验

9片45个配对位已完成V3完整时段视觉审核，primary pending=0。新18终态中16严格有效、2严格失败；35次实际LLM调用，无研究重试，无FATAL。两个预算分配候选均未使完整对象对应all提高，不采用、不扩展。

## 条件与口径

同一community checkpoint unsloth/Qwen3-VL-8B-Instruct-bnb-4bit，revision10d842936325905934ea24e64325d774693f166e；本机WSL RTX3080 20GB，NF4 double quant/BF16。torch2.8.0+cu128/transformers5.19.0/bnb0.50.2/accelerate1.11.0。A/Bfull/B125各9父终态锁SHA继承，不重跑或重计新增成本；新两臂各9。不同官方权重/服务器结果不合并。

六片固定工程异常统计any/all/event/category，三正常单列系统误报；不是完整167的新准确率，也不与旧49%或旧L3历史分数直接相减。固定189/167/3/19及参考/manual不改，同聊天非盲审，human_verified/正式训练资格false。

## 结果

|臂|any/6|all/6|event/6|category/6|异常片错绑/6|技术失败/9|正常误报/3|
|---|---:|---:|---:|---:|---:|---:|---:|
|量化A不压缩|2|0|3|4|6|0|3|
|Bfull不压缩|4|2|5|5|3|0|3|
|B125双阶段|3|0|4|4|4|0|3|
|R25_D125|2|0|3|2|3|2|3|
|R125_D25|3|0|4|4|4|0|3|

R25_D125相对A的any/event未下降，但category从4降至2并新增两项技术失败；R125_D25的any/event有所增加，all仍0。正常误报均3/3，不能由个别片any成功宣称压缩改善了完整对象定位。

## 全部阶段视觉Token与成本

|臂|实际视觉Token|完整执行预算|实际LLM调用|输出Token|全部LLM Token|终态端到端总秒|研究峰值allocated/reserved MiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|量化A不压缩|67164|67164|9|4534|94761|254.388|11231.03/11878.00|
|Bfull不压缩|134328|134328|18|23604|208895|1096.818|12049.46/13688.00|
|B125双阶段|34560|34560|18|22864|107552|1049.775|9641.46/10552.00|
|R25_D125|38787|41611|17|24251|108736|1121.052|9589.03/10532.00|
|R125_D25|41611|41611|18|22933|114672|1062.381|9982.63/10902.00|

R125_D25两阶段全部实际视觉41611，比单次A67164减少38.05%。R25_D125完整执行应同为41611，但第五片名册严格失败导致决策未执行，实际少2824个视觉Token；不能把该失败省略算为有效效率提升。
所有视频帧仍先编码，原生局部图和文字全部保留。减少的是进入语言模型的视觉序列；两阶段重复视觉编码、名册和决策全部输出均入账，不宣称视觉编码FLOPs同比下降或整体变快。两新臂全部LLM分别仍高于A。

新增35调用共47184输出/223408全部LLM Token；18个名册和17个决策，第三片决策失败成本及第五片名册失败成本全部保留。CPU逐Token流start/数量、RAW、实际压缩输入/RESPONSE/记录/阶段ratio、CPU tokenizer逐流解码==RAW全部一致，383成本源文件。工程CPU预检、模型加载、GPU证明、资源等待另存，不计入研究调用或伪称整次实验总开销。父27控制费用只作同条件成本比较，不再次报新增。

基线A原生路径的视觉编码包含在generation时间，压缩B路径视觉准备与CPU输入准备分开记录；不能直接把两类generation_s当同义延迟。上表用终态完整elapsed_s。新R25 GPU准备20.942秒/CPU输入准备13.717秒，R125为22.451/14.888秒，详细逐阶段值见FINAL_RESEARCH_COST_CPU_AUDIT.json。

## 逐片新增臂裁决

|片|R25_D125|R125_D25|主要可核查事实|
|---|---|---|---|
|1 clip_01dcf7|1/0/1/1；错绑True|0/0/0/1；错绑True|全窗推搡与浅衣长裤施力者可对应；R25仍误标条纹旁人。R125仅互动/绳圈游戏未正确理解真实施力。初始接触manual不解。|
|2 clip_034025|1/0/1/0；错绑True|1/0/1/0；错绑True|唯一后段攀墙者可对应；额外狗攻击无依据，R25只G01错误代码，R125有额外G01。时间提前另记。|
|3 clip_087917|严格失败；无语义结果|1/0/1/1；错绑True|R25重复同一事件对象ID严格失败；R125赤膊施动者正确，但浅衣背景者也owntrue。|
|4 clip_0f383c|0/0/1/1；错绑False|0/0/1/1；错绑False|两真实浅衣骑手；通用白/浅衣同方向描述不能唯一分辨。正文骑行/G06独立正确，未标长椅者own。|
|5 clip_1075c0|严格失败；无语义结果|1/0/1/1；错绑False|R25名册窗外证据严格失败未启动决策；R125红衣/唯一深衣骑手可对应但漏两名白衣，下装区分未给。|
|6 clip_22cbff|0/0/0/0；错绑True|0/0/0/0；错绑True|两臂把F024-F037车辆/水花事故作为异常；F024原图无所称喷溅，漏后段真实火烟。不能借后段火追认早期事故。|
|7 clip_6c4f0e|系统误报；错绑|系统误报；错绑|两臂把稳定弯腰捡取当G04失稳风险，系统正常误报；白红人ownfalse不额外错绑。|
|8 clip_73998e|系统误报；错绑|系统误报；错绑|两臂把可控跳转/挥臂当G02危险抛掷，无离手物体或危险目标支持。|
|9 clip_c1e5da|系统误报；错绑|系统误报；错绑|两臂把可控落地屈膝/蹲身当G04事故/跌倒，无已确认失稳；不外推片外未来。|

异常位数字依次为any/all/event/category；正常不作空集合all成功。重复预测同一真实异常人不自动当额外无关者错绑，名册未选中的重复/幻觉也不自动加错绑；实际给普通人或不存在动作的own异常才扣。

第一片曾因R25仅通用互动而暂留pending；重读实际全窗决策中肯定的推搡表述、唯一浅衣深长裤e004以及F034原图后，按原V3最低any/event条件通过。该输出窗F000-F072包含已确认施力，与仅错误早期窗或仅可能构成/绳圈游戏的历史输出不同。没有修改RAW/规则或初始manual，旧preliminary原字节保留，由v001取代。

第五片全时段重看也发现来源说明关于唯一黑衣者进入方向的旧文字不准确；当前v001追加具体帧序列说明，原参考和旧来源文字保留。主体数/原指标未改，不用方向纠正额外增加成功数。

## 工程证明与限制

freeze806633c39a2a608a3efae45ce15472490cdffdf17a2d1abd8ef4e0fbb5e17f57，37成员；45本地/45WSL CPU、779来源依赖、父27控制、16权重/9全部I3和中性输入Tensor一致。34个父中性名册decision实际processor重建最长17273。新UUID租约/252NF4加载四错误项空；短mixed、最长I3 13388和真实B 17273四比例×4decode共48全词logits/rawBF16/cache/DeepStack/mRoPE/full-retention/dense-after通过，独立CPU57archive回读并与共享父dense逐值一致。

GPU proof.pt SHA8d0aed0397ee6ab3925e1daec12ecd0eaececf14dfb35ba8ef9e4441c392d185；GPU工程峰值allocated12387.71/reserved13746MiB。该4decode证明不等于8192最大输出或以后自己名册的上界。真实研究04:44:53完成，队列04:44:56退出释放租约，04:45:09实际进程空，无FATAL；原始终态、两失败与所有成本不修补。

## 决定与接续

两个新候选均不通过all净增加护栏；R25还不通过类别/技术失败护栏，扩展false，selected_candidate=null。该切片完成不等于长期根Goal完成，原生Goal仍paused。

独立官方复核只预注册于L:/VAD/experiments/qwen3vl8b_i3_b_stage_allocation9_server_v1_20261101，当前0运行代码/0freeze/0GPU/0研究。其官方A/Bfull18控制来自不同父根，必须新增官方B125控制9和R25/R125各9，总27新终态/最多54调用；先TDD/官方最长B输入fresh预检/新lease与48步门禁，不继承社区控制。

## 可复核入口

机器结果SMALL_TRIAL_DECISION.json，九份evaluation_only/paired_visual_adjudications/*.v001.json及ADJUDICATION_INDEX.json；全部原输出位于outputs和records。
成本脚本evaluation_only/cost_reconciliation/audit_completed.py，最终独立CPU流证据FINAL_RESEARCH_COST_CPU_AUDIT.json；验证FINAL_INDEPENDENT_HASH_READBACK.json后再以本报告为当前评价事实。worker退出时STATUS visual_evaluation=pending仅为历史写入时点。

运行状态命令（只读，不重复start）：
    C:/Users/10495/anaconda3/python.exe -B validation/local_control.py status
CPU成本复核（WSL，不使用GPU；只在需核验新增/最终终态时执行）：
    /mnt/l/VAD/envs/qwen3vl8b_nf4_linux/bin/python -B evaluation_only/cost_reconciliation/audit_completed.py
