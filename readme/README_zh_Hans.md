## Dify Higress 模型插件

**Author:** higress
**Version:** 0.0.3
**Type:** model

### 介绍

该插件用于在 Dify 中通过访问 **Higress AI 网关的 Model API** 访问模型服务。Higress AI 网关支持代理丰富的模型供应商（云厂商/平台）或自建模型推理服务, 并提供面向 AI 流量的治理、观测、安全认证等能力, 在 Dify 侧通过使用该插件, 只需要配置网关路由地址、协议与鉴权方式，即可按需使用不同模型能力。

该插件既支持访问 **开源 Higress**，也支持商业化的 **阿里云云原生 AI 网关**。

### 开始使用

1. 在 Higress AI 网关控制台创建不同使用场景和协议的 Model API, 按需配置流量防护、消费者认证、观测与统计、插件等策略，以便对 AI 流量进行统一治理与运营。

![](./_assets/higress_model_api_zh.png)
![](./_assets/higress_model_api_detail_zh.png)

2. 在 Dify 中安装本插件后，进入 **设置 / 模型供应商（Model Provider）**，选择 **Higress**，点击 **添加模型**。
3. 按照插件配置项填写, 部分关键字段的说明如下:
   
   - **使用场景**: 期望访问的 Model API 路由使用场景, 支持文本生成、图片生成、文本排序、向量嵌入
   - **模型协议**: 期望访问的 Model API 路由协议, 例如OpenAI兼容、百炼原生协议等
   - **Higress AI 网关路由**: 期望访问的 AI 网关 Model API 路由地址, 格式为 `{http/https}://{网关域名}/{自定义前缀}`。
   - **消费者鉴权策略**: 根据对应 Model API 的消费者认证配置选择 **不开启 / API Key / AK/SK (HMAC)**，并补充对应的密钥信息。
   - **透传模型名称**: 当对应Model API 选择透传模型名称时, 该项需要填写, 例如`qwen-max`。

![](./_assets/configuration_zh.png)

4. 在 Dify 应用或知识库构建时, 选择上述步骤构建的Higress模型, 即可实现让 Dify 应用通过 Higress AI 网关代理访问模型。
