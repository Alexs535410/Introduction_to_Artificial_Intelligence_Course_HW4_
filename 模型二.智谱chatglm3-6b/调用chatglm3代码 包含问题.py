from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

model_name = "/mnt/data/chatglm3-6b" # 本地路径
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,torch_dtype="auto").eval()

questions = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道",
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上「意思」分别是什么意思。"
]

for i, question in enumerate(questions, 1):
    print(f"\n{'='*50}\n问题 {i}: {question}\n{'='*50}")
    
    # 准备模型输入（每次独立处理，不保留历史）
    inputs = tokenizer(question, return_tensors="pt").input_ids.cuda()
    
    # 创建流式输出器
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print("AI回答: ", end="", flush=True)
    
    # 生成回复
    outputs = model.generate(
        inputs,
        streamer=streamer,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 添加分隔线
    print("\n" + "-"*50)