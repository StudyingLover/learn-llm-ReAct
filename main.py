from enum import Enum
from typing import Dict, Optional
import os
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
)


def extract_json(text):
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        json_content = text[json_start:json_end].replace("\\_", "_")
        return json_content
    except Exception as e:
        return f"Error extracting JSON: {e}"


def calculate(what):
    return eval(what)


def average_dog_weight(name):
    if name in "Scottish Terrier":
        return "Scottish Terriers average 20 lbs"
    elif name in "Border Collie":
        return "a Border Collies average weight is 37 lbs"
    elif name in "玩具贵宾犬":
        return "玩具贵宾犬的平均体重为 7 磅"
    else:
        return "An average dog weights 50 lbs"


action_doc = """
calculate(what)
    此函数接受一个字符串参数 `what`，并使用 `eval` 函数执行该字符串表示的表达式。
    注意：使用 `eval` 函数可能会引起安全问题，因为它会执行参数中的任何代码。

average_dog_weight(name)
    此函数根据狗的品种名称 `name` 返回该品种狗的平均体重。
    - 如果 `name` 是 "Scottish Terrier"，则返回 "Scottish Terriers average 20 lbs"。
    - 如果 `name` 是 "Border Collie"，则返回 "a Border Collies average weight is 37 lbs"。
    - 如果 `name` 是 "玩具贵宾犬"，则返回 "玩具贵宾犬的平均体重为 7 磅"。
    - 如果 `name` 不是上述任何一种，则返回 "An average dog weights 50 lbs"。"""


class Agent:
    def __init__(self, system="你好"):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="deepseek-chat", messages=self.messages
        )
        return completion.choices[0].message.content


class Actions(str, Enum):
    calculate = "calculate"
    average_dog_weight = "average_dog_weight"


class LLMOutput(BaseModel):
    # 思考、行动、暂停、观察
    thinkint: str
    actions: Optional[Actions]
    parameter: Optional[Dict[str, object]]
    observation: str
    answer: str


print(LLMOutput.model_json_schema())

system_prompt = f"""
你是一个结构化数据的处理器,你精通json格式的数据,并且可以输出结构化的json数据。你可以根据给定的文字和json scheme,输出符合scheme的json数据。请注意,你的输出会直接被解析,如果格式不正确,会导致解析失败,你会被狠狠地批评的。
"""

react_prompt = f"""
你在一个思考、行动、暂停、观察的循环中运行。

你可用的操作是：

1. 计算：
例如计算：4 * 7 / 3
运行计算并返回数字 - 使用 Python，因此请确保在必要时使用浮点语法

2. 平均狗体重：
例如平均狗体重：牧羊犬
在给定品种的情况下返回狗的平均体重

下面是acions的文档：
{action_doc}

你的输出应该是一个json格式的字符串，输出需要遵循的json scheme如下：{LLMOutput.model_json_schema()}
其中
- thinkint 是你的思考
- actions 是你的行动
- parameter 是你需要执行行动的参数
- observation 是你的观察
- answer 是你的答案


当你认为自己已经找到了答案时，不输出actions,同时输出answer。
输出时，**直接返回json字符串**，不需要任何额外的输出，你的输出将被直接解析为json字符串，所以请确保你的输出是一个合法的json字符串。
不要在输出中包含任何额外的信息，只需返回json字符串。你的输出将被使用LLMOutput.model_validate_json建立一个LLMOutput对象
每次输出之进行一步操作，不要在一个输出中包含多个操作。
"""


def query(question, max_turns=5):
    i = 0
    bot = Agent(system_prompt)
    next_prompt = react_prompt + question
    while i < max_turns:
        i += 1
        result = extract_json(bot(next_prompt))
        print("*" * 80)
        print("Output: ", result)
        llm_output = LLMOutput.model_validate_json(result)
        if llm_output.actions:
            print("Actions:", llm_output.actions)
            # There is an action to run

            if llm_output.actions not in Actions.__members__.keys():
                raise Exception(
                    "Unknown action: {}: {}".format(
                        llm_output.actions, llm_output.parameter
                    )
                )
            print(" -- running {} {}".format(llm_output.actions, llm_output.parameter))
            observation = eval(Actions[llm_output.actions])(**(llm_output.parameter))
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return llm_output.answer


question = """我有两只狗，一只边境牧羊犬和一只苏格兰梗犬。
它们的总体重是多少"""
print(query(question))
