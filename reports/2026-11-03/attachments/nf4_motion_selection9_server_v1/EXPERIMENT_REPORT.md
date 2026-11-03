# 官方在线NF4：同预算时间新颖度M125小试

9/9新strict-valid、9调用，官方A/C125/S/U36只读控制加M9，共九片45位已完成V3实际全时段配对、成本和绑定证据核验，primary pending0。M未提高all，错绑从官方A的5片增加到6片，不采用、不扩展。

| 方法 | any | all | event | category | 异常片错绑 | 正常系统误报 | 全部视觉Token | 全部LLM Token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I3_nf4_full | 3/6 | 0/6 | 4/6 | 4/6 | 5/6 | 3/3 | 67164 | 95653 |
| C125_video | 4/6 | 0/6 | 5/6 | 4/6 | 4/6 | 3/3 | 17280 | 43982 |
| S125_center | 4/6 | 0/6 | 5/6 | 3/6 | 5/6 | 3/3 | 17280 | 45139 |
| U125_uniform | 3/6 | 0/6 | 4/6 | 4/6 | 4/6 | 3/3 | 17280 | 44412 |
| M125_motion | 3/6 | 0/6 | 5/6 | 4/6 | 6/6 | 3/3 | 17280 | 45373 |

全五路技术失败0/9。M全部视觉17280，对官方A 67164减少74.27%。所有帧和原生局部图仍编码，图片/文字全保留，不减少编码器工作。实际九片每时间组精确预算、首尾、stride16底线、singleton原向量组和全部索引覆盖实核；九片M中心均不同于S/U。算法与已实现本机M源字节相同，但权重/量化来源/平台不同，所有分数与成本独立，不合并。

逐片新输出：首片事件正文能识别真实推搡/拉扯对抗，event独立通过，但列出的条纹者受推/黑衣与条纹者接触不能对应实际浅衣长裤对黑衣的施力，受害者也不自动异常，any/all未过；原初始接触manual保留，不借后段黑衣动作补当前窗口。第二识别攀墙但时间提前，额外打狗/狗攻击虚构。第三破坏者对应，却把受影响货架也own异常。第四两浅衣骑手不唯一，又把普通树下坐站人群own异常。第五七骑手ID反复复制深衣，真实四人。第六误把火烟作液体泄漏。三正常均把受控拾取/表演跳跃/蹲身或普通路人虚构为抛物、碰撞、拉拽或跌倒。每臂每clip只计一次正常误报。

V3实际事件理解、真正主体和唯一身份须共同成立；event/category独立。首片event的通过不认可其对象归属，正如真实倒地可以与错误倒地者指代分开。kind/角色格式本身不扣身份；受影响设施/受害者不自动自身异常。时间偏差另列，不冒充IoU。新九输出已全部实际读取，完整源页/必要原图复用同聊天真实工具返回按SHA核验，首片overview_02本轮重看不增加样本，不套本机M评分。

官方模型Qwen/Qwen3-VL-8B-Instruct revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b在线NF4 double quant/BF16 compute，252语言Linear，视觉/DeepStack/head/embed BF16；torch2.14.1+cu130/TF5.19/bnb0.50.2/accelerate1.15。freeze 933ce0ae59491fe23f41b8dd03200d8122d30772edd9834262d187ee61ecf68a，36成员。49本地/49远端CPU、来源依赖、15模型和9全部实际输入Tensor与官方父逐项一致通过。只有一次start，GPU4 UUID GPU-2f7e5fbd-73d6-3cfc-824d-af6fc077fb57、新leasef2c201d87b384d9b9b1a44943b86005b，初始36062/util17、3秒33502/util10，post-CUDA torch34726.3125/NVML36403均过24000，标准UUID规范一致。cap20000+4000及26000初始门槛不降，未重启旧FATAL。

真实加载四错误项全空、新M短mixed/最长13388四比例×4decode共32全词logits/rawBF16/cache/DeepStack/mRoPE/full-retention/dense-after及父官方dense逐值一致通过，38archive序列化CPU回读及另进程CPU审计均通过，后者CUDA未初始化。peakallocated13050.19/reserved14094MiB。4decode不证明8192最大输出。04:13:07推理结束、04:13:10释放lease，04:15:26实际进程空，无FATAL。

9新调用合计5030输出/45373总LLM Token，99成本源文件逐Token连续/数量/INPUT/RESPONSE/记录及CPU tokenizer解码==RAW核验。父36费用不重计，加载/资源等待/工程门禁另存。每臂准备/生成时间、显存与实际Token在预算审计；基线编码计时口径不同，未宣称等条件速度收益。

ADJUDICATION_INDEX SHA 28ece43a78860baee5253662dc4c972a67410383700183f8f6bd1c9f25e9434b；CPU成本审计SHA 5e7ff5e9bf11e1d2d998c6b90085c8175587c2fcf1c9b32c581066dad195ee45；GPU独立CPU审计SHA 1d9fbf6482f828aa327abdc117107639cbf5a5a5054df60627f5e974aae6e3cc；绑定预算审计SHA f5576fb76d4ca22a7025873c47d6e720f2426d5d8acda11fcf9c026e6ca1ff00。568绑定文件fresh回读一致。worker STATUS visual pending为退出时历史标记，本机此最终报告是当前评价事实。

固定189/167/3/19、原参考/manual不改，human_verified及正式训练资格false。同九片开发小试不外推167，负结果不完成长期根Goal。本机独立阶段预算分配实验正在推进，其结果与本版分开，不为占服务器卡重跑本版。
