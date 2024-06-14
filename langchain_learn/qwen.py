
# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus
import dashscope
from langchain import llms

def call_with_messages():
    """
    https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-thousand-questions/?spm=a2c4g.11186623.0.0.703141e3Ck93Jo
    """
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '如何做炒西红柿鸡蛋？'}]

    response = dashscope.Generation.call(
        # dashscope.Generation.Models.qwen_turbo,
        model='qwen1.5-1.8b-chat',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


def embed_with_str():
    """
    https://help.aliyun.com/zh/dashscope/developer-reference/generic-text-vector/?spm=a2c4g.11186623.0.0.616b4c5eNxhr4f
    """
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v1,
        input='衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买')
    if resp.status_code == HTTPStatus.OK:
        print(resp)
    else:
        print(resp)


if __name__ == '__main__':
    call_with_messages()
    # embed_with_str()
    