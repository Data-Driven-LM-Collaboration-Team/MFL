# 本机社区NF4 I3：名册与决策两阶段压缩9片报告

三档压缩都未通过预注册采用护栏，不扩展。B50在店内破坏片保留了全部主体对应，但视觉总量仍比单次A多且出现严格失败；B25/B125的全部主体正确均没有超过A。原Bfull的2/6全部主体对应在压缩后没有完整保留。

本轮新增27个不可变终态（24严格有效、3严格失败）、52个实际LLM调用；父A/Bfull各9终态只读继承，共9片45个配对位。全部新名册和决策均实际读取，与既有完整时段源图/必要原图工具返回按SHA配对；最终primary pending=0。技术有效不等于视觉正确。

## 绑定与正常误报

六个固定工程异常片作any/all/event/category分母，三个原正常片单独计系统误报。不是完整167/189正确率，也不与历史约49%或官方服务器checkpoint混算。

| 路线 | any/6 | all/6 | event/6 | category/6 | 错绑片/有效异常输出 | 严格失败/9 | 正常系统误报/3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| I3_nf4_full | 2 | 0 | 3 | 4 | 6/6 | 0 | 3 |
| B_roster_then_rules | 4 | 2 | 5 | 5 | 3/6 | 0 | 3 |
| B50_both_stages | 2 | 1 | 3 | 3 | 2/5 | 1 | 3 |
| B25_both_stages | 1 | 0 | 2 | 2 | 3/4 | 2 | 3 |
| B125_both_stages | 3 | 0 | 4 | 4 | 4/6 | 0 | 3 |

失败保留在端到端分母，false_binding=null不能静默当正确或正常。B50普通拾取片全部own=false，但非空events按冻结合同派生ANOMALY，因此系统误报与语义错绑分开；没有事后改NORMAL。正常片不以空异常集合造all成功。

第一片的历史preliminary原字节保留。最终按冻结V3“必须正确理解实际异常事件”的原条件核对完整RAW：B25仅写早段挥臂/转身、结尾“可能构成推”；B125写绳圈抛接/旋转、结尾“构成人际”，没有给出已确认的中后段伸击/反向施力。G01代码与衣着清单不能替模型补未叙述的施力关系，any/event未过。原早段接触性质/介入角色manual没有被强判；这是输出未满足既定正向匹配条件。

四骑手片B125至少正确对应红衣和真实骑车，但没有把两个白衣背包者按下装/关系分别识别，all失败；深衣重复ID不自动算无关者错绑，详细轨迹错误另留。两浅衣骑手片三档都缺少足以区分二人的描述。火烟片三档继续把火烟误作白液/水花/车事故，不能只靠类码认成功。

## 全阶段实际成本

| 路线 | 实际LLM调用 | 输出Token | 全部LLM Token | 实际全部视觉Token | 相对单次A视觉减少 | 生成秒合计 | 端到端秒合计 | 峰值allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I3_nf4_full | 9 | 4534 | 94761 | 67164 | 0.00% | 245.100 | 254.388 | 11231.03/11878.00 |
| B_roster_then_rules | 18 | 23604 | 208895 | 134328 | -100.00% | 1069.322 | 1096.818 | 12049.46/13688.00 |
| B50_both_stages | 17 | 20995 | 136000 | 71046 | -5.78% | 923.403 | 970.196 | 10679.56/11336.00 |
| B25_both_stages | 17 | 24323 | 114774 | 44753 | 33.37% | 1080.396 | 1128.131 | 9928.82/10640.00 |
| B125_both_stages | 18 | 22864 | 107552 | 34560 | 48.54% | 1000.639 | 1049.775 | 9641.46/10552.00 |

新三档总计68182输出/358326全部LLM Token。名册额外阶段分别B50 9次/68078LLM、B25 9次/57220LLM、B125 9次/48631LLM，均已计入。B50和B25各有一名册严格失败，未开始对应决策，不能虚造两次未发生的LLM调用。B25另有一次决策重复e001严格失败，其实际两阶段费用均保留。

B50/B25实际视觉总量含失败，分别71046/44753；假如全部9片都执行完整两阶段，预注册预算分别77160/48662。失败跳过决策带来的少用Token不是有效推理节省收益。B125九片都完成两阶段，实际34560，较A67164减少48.54%。所有帧仍进入视觉编码器，局部图和文字全保留；不能声称省去视觉编码。时间包含每阶段准备/压缩/生成；等待GPU、模型加载和工程门禁另列在运行记录，不计入研究调用表。父A/Bfull费用是其历史控制成本，未再次计入358326。

## 采用结论与证据边界

B50_both_stages：未通过 category_non_decrease, technical_failures_non_increase, whole_actual_visual_saving_at_least_20pct。
B25_both_stages：未通过 all_net_increase, any_non_decrease, event_non_decrease, category_non_decrease, technical_failures_non_increase。
B125_both_stages：未通过 all_net_increase。

冻结：d7b8b360e94201e172761416511e5a6e98c0950d9ef784192351ba24cc8efb76；本机unsloth/Qwen3-VL-8B-Instruct-bnb-4bit revision10d842936325905934ea24e64325d774693f166e，NF4 double quant/BF16 compute。全部帧/原生局部图、原I3 Prompt/schema、采样/种子、8192输出/32768context不变。新三臂共用同一模型实例；父A/Bfull来自同一固定checkpoint/量化/平台的前一实例，fresh dense全词逐值匹配证明留存。

38本地/WSL CPU与9输入/16模型文件/779来源依赖预检通过，当前freeze/lease的真实252语言量化加载、四项加载错误为空、三长度48步全词logits/cache/DeepStack/mRoPE/full-retention/dense-after与57archive CPU回读通过后才研究。4decode不等于8192最大输出证明。52实际调用已在独立CPU上逐流起点/数量与tokenizer解码==RAW、原/实际输入、视觉保留、RESPONSE/记录加总及全部SHA核验，CUDA未初始化。

最终逐片文件见evaluation_only/paired_visual_adjudications/*.v001.json；机器结论见evaluation_only/SMALL_TRIAL_DECISION.json；成本原始核验见evaluation_only/cost_reconciliation/TERMINAL_COST_AUDIT_20261101T030348Z.json。所有preliminary、失败RAW与历史index保留，已被最终v001取代的pending不继续计入当前状态。

所有结论来自同一Codex聊天的非盲开发小试，human_verified=false、正式训练资格false；原189/167/3/19、原参考及非主判manual不改。不宣称完整167改善、统计显著或泛化。

下一独立执行：服务器官方checkpoint S/U v5已另有fresh冻结/租约/加载/门禁，按其自己的官方A/C125控制报告，绝不与本机community分数或费用混合。本机下一机制只先登记新的可检验假设，未实现/未运行前不声称有效，也不自动扩展232。

复核：先运行validation/local_control.py status；本轮已结束，禁止再次start/preflight或覆盖终态。成本只读审计可运行evaluation_only/cost_reconciliation/audit_completed.py（WSL固定Python，CUDA禁用），每次保留独立快照，不重复推理。
