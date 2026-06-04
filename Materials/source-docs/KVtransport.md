3.11 KV 传输优化 
3.11.1  项目背景
vLLM现状：实现了 PD 分离场景下的 Mooncake Connector，主要用于 KV Cache 的传输。初版主要针对 DeepSeek（DS）模型开发，该模型的 attention 使用 MLA（Multi-Head Linear Attention）。DS 模型具有如下特性：
-  KV Cache 由 embedding 之后的 hidden states 经过下投影矩阵生成。 
-  在 TP（Tensor Parallel）域内，所有 TP 可以缓存全量 KV Cache。因此在 D 节点拉取 KV Cache 时，选择哪个 TP_RANK 都是可行的。 
随着 Qwen3 模型的适配工作开展，在 GQA、MHA、MQA 等场景下，TP 异构引入了新的问题：
1. TP 异构导致 KV Cache 不一致
  -  TP 域内的 KV Cache 不再统一，Decode 节点需要选择正确的 Prefill TP Rank 拉取 KV Cache，并在 D 节点的 local_block 内进行重排序。 
  -  在保证精度的前提下，还需控制性能，确保 TPOT 增加在可容忍范围内。 
2. PP（Pipeline Parallel）适配
  -  在拉取 KV Cache 时需考虑 P 节点的 PP stage，即使是 MLA，不同 PP stage 的 KV Cache 也不一样。 
  -  系统需根据 PP stage 拉取不同 stage 的 KV Cache，并放置在 D 节点对应地址。 
3.11.2 目标与约束
1.  在不影响性能的前提下，实现 PD 分离支持 GQA TP 异构的 KV Cache 传输。 
2.  支持 PP（流水线并行）场景下的 KV Cache 传输。 
3.  支持开启 MTP（Multi-Token Prefill）下的 PP 流水线。 
3.11.3 技术实现
3.11.3.1 Decode KV Cache 重排序
在所有 KV Cache 拉到 D 节点后，需要进行重排序。GQA 格式下，由于 P、D 节点 TP 异构，P 节点生成的 KV Cache 与 D 节点存储的数据排布不一致。例如 Qwen3 235B 模型，P 节点 TP16，num_heads=4，每个 TP 只能生成 tokens 的一个 head 的 KV Cache，D 节点无法直接 forward。
解决方案：
-  以 block 为粒度对 KV Cache 进行重排序，reshape 成符合 D 节点 TP size 的 KV Cache 排布。 
-  vLLM Connector 感知 TP 配置，并完成 KV Cache 的 split 和 cat。 
-  底层通信库 Mooncake TransferEngine 只负责数据传输，不感知 TP 配置。 
-  为避免额外大 buffer，引入逐 block 的 zero-copy 传输，并在传输完成后逐 block transpose，实现 cat 效果。 
[图片]

[图片]
性能优化：
-  初版逐 block 重排序在 4k KV Cache 上约 120ms，但由于每层 decode 间需要频繁 launch kernel，4k×1.5k benchmark 中 TPOT 增加 28ms。 
-  优化后使用 _npu_page_load 读取 KV Cache，并用 _reshape_and_cache 写回，每层只处理一次，仅需申请一层 KV Cache 大小 buffer，大幅减少 kernel launch 时间。 
-  性能对比： 
暂时无法在飞书文档外展示此内容
3.11.3.2 选择 P 节点的 Rank
-  Decode 节点需选择对应 TP rank 的 Prefill TP rank 拉取 KV Cache。 
-  PP 场景下，先按 PP stage 对 P 节点进行分组，再在每个 PP stage 内选取 TP rank 拉取 KV Cache。 
-  MLA 场景下支持随机选择 Prefill TP rank，以实现动态负载均衡。 
-  GQA 场景改造后，可适配上述功能。
[图片]
 
3.11.3.3 KV Cache 传输
-  根据源地址和目标地址列表调用 Mooncake TransferEngine 进行 KV Cache 拉取。 
-  GQA TP 切分下，D 节点每层可能需要多次拉取 KV Cache。 
-  引入地址偏移，确保 KV Cache 正确存放。 
-  PP 场景下，单个 P rank 拉取的 KV Cache 不是完整，需要根据 PP stage 和 get_pp_indices 计算 D 节点正确放置层数。 
[图片]
3.11.3.4 手动划分 PP Layer
-  MTP 下 KV Cache 多出 MTP 层，get_pp_indices 无法感知，导致部分 KV Cache 未传输，PP 层切分不均。 
-  引入 PPlayer 手动划分，通过 Connector 配置让 Mooncake Connector 感知层数切分，例如： 
export VLLM_PP_LAYER_PARTITION=33,28
"kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {
        "dp_size": 1,
        "tp_size": 8,
        "pp_size": 2,
        "pp_layer_partition": "33,28"
    },
    "decode": {
        "dp_size": 16,
        "tp_size": 1,
        "pp_size": 1
    }
}
3.11.4 开发中遇到的 BUG
1. Decode 重排序精度问题
  -  前半段数据正确，后半段乱套。原因：未明确何时开始重排序，何时发送结束信号。 
  -  解决方案：所有 KV Cache 拉取完成后才重排序，重排序完成后再通知 P 释放显存。 
2. Mooncake 不支持层数不对等的 KV Cache 传输
  -  初版 PP KV Cache 传输报错访存越界。 
  -  原因：Mooncake 版本根据首地址和本端注册的层数拉取，报错信息显示循环首地址。 
  -  后续 Mooncake 版本更新解决该问题。 
3.11.5 性能收益
- _reshape 替换 _copy：TPOT 从 120ms 降至 94ms，无异步调度时几乎无 TPOT 损耗。 
-  PP 适配后，DeepSeek PD 分离性能瓶颈由 P 节点转移至 D 节点，调整 PD 分离配比提升归一化 QPS： 
暂时无法在飞书文档外展示此内容
3.11.6 ACL graph 支持 Qwen235B PD 分离
在 Mooncake Connector 与 vLLM 场景下，forward 流程与 Mooncake cat 方法中均使用了 torch_npu._npu_reshape_and_cache 算子。问题表现为在 ACL Graph 场景下，调用 vLLM 的 connect 时偶发 cancel 异常。
两处算子在功能上等价，但分别存在图内（forward）和图外（Mooncake cat）调用。
详细的问题如下：
1.  forward 内部入图的 _npu_reshape_and_cache 与 Mooncake cat 中的 _npu_reshape_and_cache 互相影响。 
2.  具体表现： 
  -  两处算子分别替换成等效实现均能正常跑通，精度正常； 
  -  注释任意一个算子，也能正常运行（不管精度）； 
  -  同时保留两者时可能抛出异常或行为不稳定。 
3.  异常类型： 
  -  ACL Graph + --async-scheduling 时，vLLM connect 抛出 cancel 异常（无具体报错信息）。 
  -  关闭 --async-scheduling 后，异常定位到 forward 内 _npu_reshape_and_cache。 
暂时无法在飞书文档外展示此内容
4. 根因定位
PageAttention 算子更新时, 每次需要额外申请workspace, 该workspace申请在单算子流的pool上, 算子更新把这个workspace写入图中, 算子更新结束后, workspace被释放. 但是图重放时单算子流同步在进行kv cache的传输. 在单算子流视角下, 该workspace是可以复用的, 但是图的视角中, 需要使用该workspace进行 PageAttebtion 计算, 于是发生了内存踩踏. 之所以直接在device上创建tensor不可以, 用to.device就可以, 是因为to device时有额外隐式的同步操作, 巧合下避免了内存踩踏。
5. 解决方案
规避方案为在slot mapping生成后加入同步操作，改变流上算子排布逻辑及运行/下发逻辑。
3.11.7 transpose_kv_cache_by_block 融合算子
当前使用 _npu_reshape_and_cache + _npu_page_load 的方式来实现 GQA 下 kv cache 传输完成后的 layout 的转换，在这个过程中存在下面的问题
-  算子调用次数过多：目前每层需 launch 4 个算子，一个请求就要调用 376 次算子，
-  异步调度下无法掩盖：当前是依赖 decode 间 bubble 来掩盖算子调用的开销，未来异步调度优化 decode 间 bubble 后，可能会无法掩盖
这本质上是一个 transpose 的操作，只是因为没有合适的算子能一次性完成这个 transpose，因此可以考虑实现融合算子，一次完成所有层 KV Cache 的 layout 转换。 
3.11.7.1 transpose_kv_cache_by_block 算子设计
功能描述
对传入的 k_cache 和 v_cache TensorList, 每隔 block_len 个数据是一个 block，按照传入的 block_ids 指定的 block 进行 layout 转换，每个 block 内被包含 split_num 份数据，每一份表示 num_head/split_num 个 head 的 kv_cache, 在转换完成后会将这 split_num 份数据在 num_head 维度进行合并。
本算子将转换结果写回原内存上，且不额外使用workspace。
接口定义
transpose_kv_cache_by_block(k_cache, v_cache, block_ids, block_size, head_sum, head_dim, split_num, layer_num)
其中 k_cache，v_cache 为 TensorList，表示模型所有 layer 的 kv cache，List 长度为 layer_num
block_ids 为 List[int], 表示每一个 layer 的 kv cache 所在的位置，所有 layer 共享同一个 block_ids
[
    {
        "op": "TransposeKVCacheByBlock",
        "input_desc": [
            {
                "name": "KCache",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND"
                ],
                "type": [
                    "fp16",
                            "bf16"
                ]
            },
            {
                "name": "VCache",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND"
                ],
                "type": [
                    "fp16",
                            "bf16"
                ]
            },
            {
                "name": "blockIDs",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND"
                ],
                "type": [
                    "int32",
                    "int32"
                ]
            }
        ],
        "output_desc": [],
        "attr": [
            {
                "name": "blockSize",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "headNum",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "headDim",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "splitNum",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "layerNum",
                "param_type": "required",
                "type": "int"
            }
        ]
    }
]
算子执行过程设计
示例：num_head = 4，split_num=4
[图片]
示例：num_head = 4， split_num=2
[图片]
3.11.7.2 实现设计

    首先计算一共需要多少个block需要做layout转换。通过blockIDs的shape可以知道每个layer有多少个block需要layout转换:
$$block\_num = blockIDs.shape(0)$$
    则一共有：
$$total\_block\_num = layerNum * block\_num$$
    如果每次计算的数据量data_size_load_once单个core就可以完成读取，则使用模版一，否则使用模版二
$$data\_size\_load\_once = block\_size * head\_num * head\_dim * sizeof(T)$$
模版一
模版一的特点是一个core就可以完成一个block所有数据的读取，所以可以把整个block读入UB的时候就重排，然后直接写回原内存位置。整体方案如下图所示:
暂时无法在飞书文档外展示此内容
负载均衡
      首先确定每个核计算block的数量以及哪些block，以达到负载均衡的目的。
      其中每个core需要至少计算轮次：
$$round = total\_block\_num / vector\_core\_num$$
      有tailCoreNum个core需要多计算一轮：
$$tailCoreNum = total\_block\_num \% vector\_core\_num$$
    确定每个核的搬运次数、起始和结束位置举例如下图所示，每个核计算的block可能分布在不同的layer里，是为了负载均衡。
暂时无法在飞书文档外展示此内容
重排方案
      使用DataCopy指令，并设置以下几个主要搬运参数，是重排的关键参数
DataCopyParams repeatParams;
repeatParams.blockCount = blockSize_; // 一共搬运blockSize_次
repeatParams.blockLen = headNum_ / splitNum_ * headDim_ * sizeof(T) / dataBlockSize_; // dataBlockSize_是根据dtype得到的dataBlockSize_ = 32B / sizeof(T)，每次搬运的数据长度
repeatParams.srcStride = 0; // 源数据连续搬运
repeatParams.dstStride = (headNum_ * headDim_ - headNum_ / splitNum_ * headDim_) * sizeof(T) / dataBlockSize_; // 目的数据需要间隔存放

可以发现以上的参数一条指令完成repeatParams.blockCount * repeatParams.blockLen数据量的搬运，是总数据量的1/splitNum_，一次DataCopy完成的重排效果如下图所示：
暂时无法在飞书文档外展示此内容
所以需要splitNum_次DataCopy后完成所有数据的重排，每次需要偏移已经完成搬运的部分。
dstFactor_ = headNum_ / splitNum_ * headDim_;
srcFactor_ = blockSize_ * headNum_ / splitNum_ * headDim_;
for (uint32_t i = 0; i < splitNum_; ++i) {
    DataCopy(cacheLocal[i * dstFactor_],
             cacheGm[i * srcFactor_ + offsetBlock], repeatParams);
}
在完成在ub里的重排后，直接连续搬出即可。
模版二
    模版二是为了解决单个core没法读取一个block里所有数据的情况，让几个core读取一个block的数据。
协同计算核数
    几个core读取一个block的数据称作协同计算核数，为了充分利用算力不让有些core没有使用起来，协同计算核数目前取aiv_corenum的因子，比如48个core，协同计算核数可以是2、3、4、6、8、12、16、24、48。我们取能满足读取一个block的最小因子，如果最大因子也不能满足，则超出了该算子的计算能力，会拦截并报错。我们令这个因子的名称为blockSizeSplitNum，也就是说这个切分维度是在block_size这个维度上的。
负载均衡
    此方案的负载均衡公式如下：
    core分组：
      
  $$core\_num = vector\_core\_num / blockSizeSplitNum$$
        
     其中每个core分组需要至少计算轮次：
$$round = total\_block\_num / core\_num$$
      有tailCoreNum个core分组需要多计算一轮：
$$tailCoreNum = total\_block\_num \% core\_num$$
重排方案
与模版一差不多，主要是有以下差异。 
因为切分的是block_size，DataCopy的repeatParams参数中应修正:
blockSizePerTime_ = blockSize / blockSizeSplitNum;
repeatParams.blockCount = blockSizePerTime_;
在计算某个block时，会变成多核计算同一个block，此时搬入地址和搬出地址要做相应的偏移：
uint32_t blockSizeIndex = blockIdx_ % blockSizeSplitNum_;
srcOffset = blockSizeIndex * blockSizePerTime_ * headNumSplited_ * headDim_;
dstOffset = blockSizeIndex * blockSizePerTime_ * headNum_ * headDim_;
CopyIn(kCacheGm_, offsetBlock + srcOffset, repeatParams);
CopyOut(kCacheGm_, offsetBlock + dstOffset);
在搬出之前要做一下核间同步，保证所有核都完成读入以后才搬出。
3.11.7.3 性能实测
使用该融合算子后，reshape kv cache的时间开销可从UT中的7ms降低至0.24ms，在PD分离场景下，qwen3-235B的TTFT可降低约90~110ms。
[图片]

暂时无法在飞书文档外展示此内容