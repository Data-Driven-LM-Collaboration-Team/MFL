# 本机社区 NF4 I3 五路：9片配对小试

UTC 2026-11-01 01:13。推理、逐Token/RAW成本对账及V3全时段配对视觉裁决已完成。45/45终态严格有效，54次实际LLM调用，无失败或语义重试；worker已退出并释放GPU租约。技术有效不等于视觉正确。

**三种均值Token压缩均未增加“全部主体对应正确”的片数，暂不采用或扩展。B自建名册改善了主体对应，但费用增加且未压缩视觉Token，不能称压缩成功。** 本结果仅限固定六个工程异常片和三个正常对照，不是固定167异常的整体正确率，也不是独立测试集泛化。

## 同条件与方法

本机WSL RTX3080 20GB，同一模型实例运行五路。社区预量化checkpoint为unsloth/Qwen3-VL-8B-Instruct-bnb-4bit，revision 10d842936325905934ea24e64325d774693f166e。16文件7242509964字节本轮再次实核。语言252个Linear为NF4 double quant/BF16计算；视觉、三个DeepStack、embedding/head保留BF16。加载四错误项为空。环境torch2.8.0+cu128 / Transformers5.19.0 / bitsandbytes0.50.2 / accelerate1.11.0。

freeze：2c25d5460209afa9fab5a6f6153b262790ce8ac8e678dea92fb579c680d16d0c，28成员未变。当前lease 37d9dcadab6243d2944e68b63b4702bc及GPU UUID GPU-2cc76af1-0caf-68ad-f174-cb7b941a4541匹配加载/门禁证据。新短mixed与最长实际I3输入13388、四比例各4decode共32步全词logits/raw BF16/cache/DeepStack/mRoPE/full-retention/dense-after校验通过，.pt经独立CPU回读。证明不覆盖8192最大输出或服务器24576填充门禁；研究阶段实际调用均在原限额内完成，未改输入/输出预算绕过问题。

A为本机量化I3完整视频/局部图不压缩。B先无政策自建中性名册，再按原规则选自己的实体ID。C50/C25/C125仅对视频main及三个DeepStack按原merge_plan均值压缩；全部视频帧仍先编码，原生局部图片和文字全部保留。原Prompt/schema、10FPS加末帧、采样/种子、8192输出/32768context均固定。未把参考或本评估文件送入模型。原I3九片Tensor SHA逐项一致，但本机社区权重与服务器官方在线NF4不同，分数及成本分开报告。

## 实际视觉结果

按冻结BINDING_PROTOCOL_V3.md：必须理解真实可见异常、关联真正自身异常主体，并能由外貌/动作/关系唯一指认。不要求bbox或九宫格；kind/角色枚举单独不决定失败。受害者、工具、背景不自动是自身异常主体。any允许至少一个正确对象同时存在额外错绑；all要求全部已确认主体且无额外错绑。大类代码独立；时间偏差和未确认细节另存，不测时间IoU。

| 方法 | 至少一个主体正确 /6 | 全部主体正确 /6 | 真实事件 /6 | 类别整体正确 /6 | 额外错绑片 /6 | 技术失败 /9 | 正常误报 /3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 不压缩 | 2 | 0 | 3 | 4 | 6 | 0 | 3 |
| B 名册再规则 | 4 | 2 | 5 | 5 | 3 | 0 | 3 |
| C50 均值压缩 | 3 | 0 | 4 | 4 | 5 | 0 | 3 |
| C25 均值压缩 | 4 | 0 | 5 | 3 | 4 | 0 | 3 |
| C125 均值压缩 | 4 | 0 | 5 | 3 | 5 | 0 | 3 |

B的两片all通过为店内破坏和玻璃楼前四骑手：前者不再把受影响货架/商品自身标异常；后者用红/黑衣及两白衣不同下装和骑车关系分别指认四组。车色/篮子和部分时间不准独立保留，all通过不等于完整描述全真。

三C提高了部分“至少一个”覆盖，但未解决全对象唯一分辨、重复人物、受影响物被标自身异常、火烟误作液体和正常动作误判。C25/C125额外类别错误也增加，不能只挑any宣称整体提升。

九片完整逐臂裁决在evaluation_only/paired_visual_adjudications/*.v001.json。旧第一片preliminary及B_RAW_SCOPE均原字节保留，但其primary pending已由最终v001取代，不能继续计为当前未裁决。所有原参考/manual保留；早段接触的局部性质、衣着/手部细节与时间偏差不强行消除。本聊天同一Codex查看，非独立人核。

三正常对照每臂系统标签均ANOMALY。特别是C125在普通拾取片文字否定异常、对象abnormal=false，语义无正向错绑，但冻结系统按非空events仍派生ANOMALY：如实记系统误报，不事后改NORMAL。正常片不纳入六异常any/all，也不以空异常集合造all成功。重复事件只记一个clip误报。

## Token与成本

| 方法 | 全部视觉Token总数 | 相对A减少 | 总LLM Token，含输出 | 实际LLM调用 | 生成秒数总计 | 端到端秒数总计 |
|---|---:|---:|---:|---:|---:|---:|
| A | 67164 | 0% | 94761 | 9 | 245.100 | 254.388 |
| B | 两阶段重复编码，未压缩 | 无 | 208895 | 18 | 1069.322 | 1096.818 |
| C50 | 38580 | 42.56% | 65774 | 9 | 208.557 | 229.089 |
| C25 | 24331 | 63.77% | 52081 | 9 | 228.874 | 250.358 |
| C125 | 17280 | 74.27% | 44468 | 9 | 198.851 | 219.853 |

原视频Token61596，加原生局部图片5568。C保留后视频分别33012/18763/11712，图片5568保持不变。实际累计视觉Token减少不是精确请求比例；首尾/空间底线及保留图片影响总压缩率。全部帧已进入视觉编码器，不能声称减少了视觉编码FLOPs。B两阶段各自完整图像输入及全部名册费用均计入LLM/耗时，不用其决策短输出掩盖额外调用。

全部54调用41081输出Token、465979总LLM Token；B额外九名册单列于成本快照。逐流start连续、Token数量、INPUT_RECEIPT/RESPONSE/RAW和记录聚合一致。成本不含CPU准备、工程门禁与排队等待，不能当完整项目费用。生成计时对A含编码，对C预编码单列于端到端及收据，因此跨方法运行时以端到端看；顺序单次运行不作独立重复性能基准。

## 采用判断与下一机制

三C的all均为0，未满足比A净增加；C25/C125类别整体还下降。B在本片集上的绑定改善是有效开发集观察，但视觉Token未省20%，总LLM成本约A的2.20倍。因此无方法同时通过本轮全部护栏，candidate=null、扩展false。不能把该9片结果推广为原189/167整体指标，历史49.7%/旧BF16同片结果只保留背景。

下一独立切片沿已经预注册的S125_center与U125_uniform：固定C125相同各时间组预算、首尾与空间底线，仅选择原中心向量而不平均main/DeepStack；均匀选择作同预算对照。先在本机同社区checkpoint独立版本实现/测试/冻结和fresh加载门禁，原本机A/C125各九终态只读锁SHA继承，最多新增18调用，不重跑五路、不改原参考。该切片区分均值混合损伤与删Token本身，尚不能事先断言会提升。

服务器S/U v3资源门禁FATAL保持原样，零模型加载/调用，不重启或降低门槛。服务器资源每30分钟实查；合规快照不等于CUDA后预算准入，需独立安全版本和fresh门禁才用服务器。官方权重本机复制是另一个CPU准备，不阻塞已授权社区模型研究。

## 证据入口

- evaluation_only/paired_visual_adjudications/ADJUDICATION_INDEX.json，9片45位、primary_pending=0。
- evaluation_only/paired_visual_adjudications/FRESH_HASH_AUDIT_20261101T011153Z.json，全部绑定输出、源图与协议fresh回读。
- evaluation_only/cost_reconciliation/TERMINAL_COST_AUDIT_20261101T010655Z.json，45终态54调用及完整阶段成本。
- validation/GPU_PROOF_INDEPENDENT_CPU_AUDIT.json，32步、38个archive项。
- validation/model_pin_observations/20261101T011314Z.json，16模型文件实核。
- HANDOFF_STATUS.json和MONITOR_STATUS.json，最新当前状态及历史worker退出标记分开。

复核命令：Windows Python运行validation/local_control.py status只读冻结源码、真实/proc、状态、GPU与终态；不要再次start/preflight。原生Goal get_goal仍paused，未伪造active或完成；当前授权小试已完成，不表示长期根Goal达标。human_verified=false，正式监督训练资格false。

