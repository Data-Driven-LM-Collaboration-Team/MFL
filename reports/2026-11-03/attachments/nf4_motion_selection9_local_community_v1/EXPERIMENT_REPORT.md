# 本机community NF4：同预算时间新颖度选择M125

新M125_motion九片均strict-valid、9实际调用，36只读父控制加新9位共45配对位已完成V3完整时段视觉审核，primary pending0。M相比本机量化不压缩A的any由2增至3、event由3增至4，all仍0；未通过采用护栏，不采用、不扩展。技术valid不是视觉准确率。

| 方法 | any | all | event | category | 异常片错绑 | 正常系统误报 | 全部视觉Token | 全部LLM Token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I3_nf4_full | 2/6 | 0/6 | 3/6 | 4/6 | 6/6 | 3/3 | 67164 | 94761 |
| C125_video | 4/6 | 0/6 | 5/6 | 3/6 | 5/6 | 3/3 | 17280 | 44468 |
| S125_center | 4/6 | 0/6 | 5/6 | 3/6 | 5/6 | 3/3 | 17280 | 45760 |
| U125_uniform | 2/6 | 0/6 | 3/6 | 4/6 | 4/6 | 3/3 | 17280 | 43873 |
| M125_motion | 3/6 | 0/6 | 4/6 | 4/6 | 5/6 | 3/3 | 17280 | 45089 |

全五路技术失败均0/9。M全部视觉17280，对单次A 67164减少74.27%；图片/文字全保留，全部视频帧及原生局部图仍编码，未节省编码器工作。九片实际每时间组预算、首尾全保留、stride16底线、原向量singleton组与去除索引完整性全部实核，且九片M中心索引均不同于S/U，未把同一选择重命名成新实验。main与三个DeepStack同索引不平均，32步全词GPU门禁与50项CPU测试支持工程实现。

逐片M判断：首片把灰浅衣短裤者写为早段推条纹者、黑衣只称被推却own异常，未对应实际浅衣长裤对黑衣及更后反向施力；原初始接触manual保留。第二真实攀墙至少对应但额外捏造打狗。第三破坏者正确却把受影响货架/商品也自身异常。第四两浅衣骑手不唯一。第五红/黑骑手至少正确但12个ID反复复制深衣，不能完整对应四人。第六把火烟写成白液泡沫，kind=fire_smoke不能补正文误解。三正常均将稳定拾取/受控跳跃蹲身或普通路人写成危险抛掷、失衡或抢夺；每臂每clip只计一次误报。

V3要求真实异常理解、真正主体及唯一身份同时成立；kind/角色枚举不单独扣分，受害/受影响设施不自动自身异常。event/category独立，错误动作不借另一时窗真实动作追认。正常仅报系统误报，不给空异常集合all成功。源全时段页/必要原图来自同聊天真实工具返回按SHA复用，本轮另重看首/第七/第九native图，不新增样本。新M输出全部逐项读取，未套父C/S/U裁决。非独立人核、原参考/manual及固定189/167/3/19不变。

实现先RED：M新颖度与DPC/U差异、预算/四层原向量、两组继承控制和实际worker/preflight/freeze路由，随后50本地及50WSL CPU通过；779来源依赖、父36/402文件、16模型与9全部实际I3 Tensor SHA fresh核验。freeze 222f30ba26eb6c4ba77b892e3916bd6ed7f747149f1a85912893214cdffa3f92，38成员。仅一次start，真实252语言NF4加载、四错误项空、当前lease下短mixed/最长真实13388四比例×4decode共32步全词logits/raw BF16/cache/DeepStack/mRoPE/全保留/dense-after及父本机dense逐值一致通过，38archive独立CPU回读且CUDA未初始化。peakallocated13034.70/reserved14078MiB。4decode不证明8192最大输出。

模型固定unsloth/Qwen3-VL-8B-Instruct-bnb-4bit revision10d842936325905934ea24e64325d774693f166e，16文件7242509964字节；NF4 double quant/BF16 compute，视觉/DeepStack/head/embed BF16，torch2.8.0+cu128/TF5.19/bnb0.50.2/accelerate1.11不改。资源初始18000/util20+3秒复核、CUDA后16384、cap14336+2048不变。8192输出/32768context/Prompt/schema/采样/seed/原生局部图不变，0语义重试。03:38:19研究结束、实际进程退出，无FATAL。

9新调用合计4746输出/45089总LLM Token，99成本源文件的逐Token连续/数量、INPUT/RESPONSE/记录、CPU tokenizer流解码==RAW全部通过。父36控制费用不重计，工程门禁/加载/等待另存。准备/生成时间与峰值显存在预算审计中，基线编码计时口径不同，未宣称等条件速度提升。

最终ADJUDICATION_INDEX SHA acafebcede86b8aa013f52052006ab33006ea8ce982eebf6e9cb11ff6cbeafce；CPU成本审计SHA 7b92412e61792dd1a22dff669bf67647e5137f1893498a7f61a56a1b3a59aae3；GPU独立CPU审计SHA 3db51306cfa6a884c0a4bae7138363a60b9449834e72bb45f23199796043c64a；绑定证据预算审计SHA 03dcbbf01844acb35e2f76a97129f781d5769b942525f2f30ef82b5f01fbd159。577绑定文件fresh回读一致。所有RAW/终态与父控制只读，worker visual pending是退出历史，不覆盖当前最终评价。

同一9片是开发诊断集，不能声称完整167提升或把官方服务器不同权重结果混分。human_verified与正式训练资格false，根Goal仍paused且小试不等于长期目标完成。
