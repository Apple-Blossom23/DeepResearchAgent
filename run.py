import ReAct_Workflow, ReAct_Events, ReAct_Tools
from custom_dashscope_llm import customDashscopeLLM
from config import config


async def main():
    # 从控制台获取用户输入
    print("请输入问题/需求（支持JSON格式和普通文本）：")
    print("示例 JSON 格式：")
    print('{"input":"如何设计一个可插拔的深度研究代理?","metadata":{"project":"demo"},"attachments":[]}')
    print("或直接输入文本： 示例：如何设计一个可插拔的深度研究代理?\n")
    user_input = input("请输入：")

    # 使用统一配置
    llm = customDashscopeLLM(
        model_code=config.DEFAULT_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DETERMINISTIC_TEMPERATURE,
        top_p=config.DETERMINISTIC_TOP_P,
        context_window=config.DEFAULT_CONTEXT_WINDOW,
        max_tokens=config.DEFAULT_NUM_OUTPUT
    )

    conclusion_llm = customDashscopeLLM(
        model_code=config.CONCLUSION_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DETERMINISTIC_TEMPERATURE,
        top_p=config.DETERMINISTIC_TOP_P
    )

    filter_llm = customDashscopeLLM(
        model_code=config.FILTER_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DETERMINISTIC_TEMPERATURE,
        top_p=config.DETERMINISTIC_TOP_P,
        context_window=config.DEFAULT_CONTEXT_WINDOW,
        max_tokens=config.DEFAULT_NUM_OUTPUT
    )

    planning_judge_llm = customDashscopeLLM(
        model_code=config.PLANNING_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DETERMINISTIC_TEMPERATURE,
        top_p=config.DETERMINISTIC_TOP_P
    )

    planning_llm = customDashscopeLLM(
        model_code=config.PLANNING_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DEFAULT_TEMPERATURE,
        top_p=config.DEFAULT_TOP_P,
        context_window=config.DEFAULT_CONTEXT_WINDOW,
        max_tokens=config.DEFAULT_NUM_OUTPUT
    )

    plan_modify_llm = customDashscopeLLM(
        model_code=config.PLANNING_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DEFAULT_TEMPERATURE,
        top_p=config.DEFAULT_TOP_P,
        context_window=config.DEFAULT_CONTEXT_WINDOW,
        max_tokens=config.DEFAULT_NUM_OUTPUT
    )

    plan_update_llm = customDashscopeLLM(
        model_code=config.PLANNING_MODEL_NAME,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.DETERMINISTIC_TEMPERATURE,
        top_p=config.DETERMINISTIC_TOP_P,
        context_window=config.DEFAULT_CONTEXT_WINDOW,
        max_tokens=config.DEFAULT_NUM_OUTPUT
    )

    agent = ReAct_Workflow.ReActAgent(
        llm=llm,
        conclusion_llm=conclusion_llm,
        filter_llm=filter_llm,
        planning_llm=planning_llm,
        planning_jugde_llm=planning_judge_llm,
        plan_modify_llm=plan_modify_llm,
        plan_update_llm=plan_update_llm,
        tools=ReAct_Tools.tools,
        timeout=config.WORKFLOW_TIMEOUT,
        verbose=config.WORKFLOW_VERBOSE
    )

    ret = await agent.run(input=user_input)
    return ret['response']


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(main())
    print(result)
