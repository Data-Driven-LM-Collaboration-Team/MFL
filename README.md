
## 1. 实验目标

我们的最终目标不是单纯做检测，而是服务对象级异常检测和后续 token compression：

```text
Qwen3VL ViT tokens
-> object binding
-> object tracking / object track tokens
-> object-level anomaly vector
-> 保留异常对象 token，压缩低异常对象和背景 token
```

因此在前置阶段，召回率比精度更重要一些。漏掉异常对象会导致后续 anomaly vector 和 token compression 都无法恢复；误召回一些正常对象的代价主要是多保留一部分 token。

## 2. Object Binding 高召回结果

object binding 的任务是：

```text
输入：Qwen3VL ViT visual token grid
输出：哪些 token 属于可见 object
```

训练时使用 YOLO / tracking 的 object 标注作为监督；推理时只需要 Qwen3VL ViT tokens 和 token geometry。

### 2.1 当前推荐：spatial instance coverage recall

这组方法在高召回和可接受 precision 之间最均衡。

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.05 | 0.4717 | 0.9812 | 0.6371 |
| 0.10 | 0.5453 | 0.9745 | 0.6993 |
| 0.20 | 0.6243 | 0.9654 | 0.7583 |
| 0.30 | 0.6796 | 0.9569 | 0.7947 |
| 0.35 | 0.7038 | 0.9526 | 0.8095 |
| 0.40 | 0.7265 | 0.9470 | 0.8222 |
| 0.50 | 0.7723 | 0.9342 | 0.8456 |
| 0.60 | 0.8251 | 0.9122 | 0.8664 |
| 0.70 | 0.8871 | 0.8696 | 0.8783 |

如果我们更重视 recall，推荐使用：

```text
threshold = 0.30: Recall 0.9569, Precision 0.6796
threshold = 0.35: Recall 0.9526, Precision 0.7038
threshold = 0.40: Recall 0.9470, Precision 0.7265
```

### 2.2 更激进版本：soft edge / class restore

这组方法能把 recall 拉得更高，但低阈值下 precision 明显更差。

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.05 | 0.2053 | 0.9993 | 0.3407 |
| 0.10 | 0.3060 | 0.9967 | 0.4683 |
| 0.20 | 0.4575 | 0.9896 | 0.6257 |
| 0.30 | 0.5707 | 0.9780 | 0.7208 |
| 0.35 | 0.6202 | 0.9697 | 0.7566 |
| 0.40 | 0.6680 | 0.9598 | 0.7878 |
| 0.50 | 0.7629 | 0.9272 | 0.8370 |
| 0.60 | 0.8495 | 0.8730 | 0.8611 |

它适合做召回上限验证，但不适合直接作为 tracking 输入的默认配置，因为低阈值会把大量背景和邻近对象 token 一起选进来。

## 3. Anomaly Vector 高召回结果

anomaly vector 的任务是：

```text
输入：一个 object-event 的 target object tokens、context object tokens、background tokens 和运动/关系特征
输出：该 object 是否异常
```

当前高召回方法采用 recall-oriented 训练策略：提高异常样本权重，加强异常对象越过阈值的约束，适当放松正常对象约束。

| Threshold | Recall | Precision | FPR | F1 |
|---:|---:|---:|---:|---:|
| 0.05 | 0.9362 | 0.4251 | 0.1574 | 0.5847 |
| 0.10 | 0.9362 | 0.4571 | 0.1382 | 0.6143 |
| 0.15 | 0.9362 | 0.4718 | 0.1303 | 0.6275 |
| 0.20 | 0.9255 | 0.4847 | 0.1224 | 0.6362 |
| 0.30 | 0.9096 | 0.5015 | 0.1124 | 0.6465 |
| 0.40 | 0.8989 | 0.5137 | 0.1058 | 0.6538 |
| 0.50 | 0.8883 | 0.5285 | 0.0985 | 0.6627 |
| 0.60 | 0.8883 | 0.5370 | 0.0952 | 0.6693 |
| 0.75 | 0.8617 | 0.5510 | 0.0873 | 0.6722 |
| 0.90 | 0.8138 | 0.5977 | 0.0681 | 0.6892 |
| 0.95 | 0.7447 | 0.6542 | 0.0489 | 0.6965 |

可以看到，高召回 anomaly vector 的最高 recall 是 `0.9362`，但 precision 只有 `0.4251-0.4718`。如果使用固定阈值 `0.50`，recall 仍有 `0.8883`，precision 是 `0.5285`。

对于 token compression，比较合理的使用方式是：

```text
高异常分数对象：保留 token
中等分数对象：轻度 merge
低异常分数对象和背景：强 merge / prune
```

也就是说，它不一定直接作为最终报警器，而是更适合作为压缩策略的前置筛选器。

## 4. Tracking 阶段遇到的困难

虽然单帧 object binding 的 token recall 已经能做到 `0.95+`，但 binding 后 tracking 仍然明显掉点。当前最好 tracking 结果如下：

| 指标 | 数值 |
|---|---:|
| Detection Precision | 0.5970 |
| Detection Recall | 0.2657 |
| Detection F1 | 0.3677 |
| Mean IDF1 | 0.3603 |
| Track Purity | 0.8469 |

这个结果说明：tracking 的问题不是轨迹内部完全混乱。Track purity 有 `0.8469`，说明一旦连成 track，内部相对还算纯。真正问题是：

```text
object proposal 覆盖率低
有效 track 数量不够
很多对象在 proposal / association 阶段丢失
```

### 4.1 token recall 高不等于 proposal recall 高

低阈值可以选中很多 GT object tokens，但这些 token 可能是离散的、断裂的、混入背景的。把 token 变成 object proposal 时，需要经过连通域、bbox 合成、NMS 和 frame cap，这一步会丢掉很多对象。

### 4.2 低阈值会导致 proposal 爆炸

为了提高召回，我们试过更激进的 proposal 策略。结果是 proposal 和 track 数量暴涨，但 detection recall 没有明显提升：

| 方法 | Det Precision | Det Recall | Det F1 | Pred Tracks |
|---|---:|---:|---:|---:|
| 当前最好 tracking | 0.5970 | 0.2657 | 0.3677 | 4149 |
| seed 诊断版 | 0.3285 | 0.2574 | 0.2886 | 20109 |

这说明只是放低阈值或增加 seed，并不能自动提升 tracking recall，反而会产生大量碎片 track 和假对象。

### 4.3 多对象近距离场景中 token 容易粘连

在人群、车辆密集、骑车人接近行人的场景里，多个对象的响应区域会连成一片。连通域方法容易把多个对象合成一个 proposal，或者把一个对象拆成多个碎片。

### 4.4 Association ambiguity 很高

当前 tracking 里存在大量候选关联歧义：

```text
ambiguous proposal candidates ≈ 390k
ambiguous track candidates    ≈ 474k
```

这说明跨帧关联时，很多 proposal 的位置、外观和类别都太相似，简单的 IoU + appearance matching 很难稳定判断身份。

## 5. 当前结论

1. Object binding 的单帧 token recall 已经可以做到很高，推荐高召回工作点是 `threshold=0.30-0.40`。
2. Anomaly vector 的高召回版本在固定阈值 `0.50` 下 recall 可以达到 `0.8883`，低阈值最高可到 `0.9362`。
3. 当前最大瓶颈是 binding 后的 tracking：token-level recall 高，但 object proposal / track recall 低。
4. 后续不应只继续降低 binding 阈值，而应改进 token-to-instance proposal，例如使用 object query / slot-style decoder 替代简单连通域。

## 6. 下一步建议

短期建议做三组 tracking 对照：

```text
spatial_instance_coverage_recall:
  threshold = 0.30 / 0.35 / 0.40

soft_edge_classrestore:
  threshold = 0.35 / 0.40
```

评估时不要只看 token recall，而要看：

```text
Detection Recall
Detection Precision
Detection F1
Mean IDF1
Track Purity
Pred Track 数量
```

如果高召回阈值不能提升 downstream Det Recall，就说明真正瓶颈在 proposal 形成和跨帧身份关联，而不是 token score 本身。

## 7. 结果来源

Object binding:

```text
/mnt/data/mfl/token_compression/data/token_compression/20260613_data/results/qwen3vl_yolo_spatial_instance_coverage_recall_v13_20260707/metrics.json
/mnt/data/mfl/token_compression/data/token_compression/20260613_data/results/qwen3vl_yolo_spatial_soft_edge_classrestore_v16_20260707/metrics.json
```

Anomaly vector:

```text
/home/expand_disk/code_repository/mfl/token_compression/docs/reports/assets/20260630_anomaly_vector/v55/metrics.json
```

Tracking:

```text
/mnt/data/mfl/token_compression/data/token_compression/20260613_data/results/exp_20260708_v13_binding_tracking_gridcc_mutual_t070/binding_slot_tracking_metrics.json
/mnt/data/mfl/token_compression/data/token_compression/20260613_data/results/exp_20260708_v18_binding_tracking_gridcc_seeddiag_t068/binding_slot_tracking_metrics.json
```
