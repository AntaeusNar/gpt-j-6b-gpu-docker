from transformers import GPTJForCausalLM, AutoConfig, GPT2Tokenizer, AutoTokenizer
import transformers
import torch
from datetime import datetime

def format_timedelta(td):
    seconds = td.total_seconds()
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if seconds < 1:
        return "<1 sec"
    return '{} {} {} {}'.format(
            "" if int(days) == 0 else str(int(days)) + ' days',
            "" if int(hours) == 0 else str(int(hours)) + ' hours',
            "" if int(minutes) == 0 else str(int(minutes))  + ' mins',
            "" if int(seconds) == 0 else str(int(seconds))  + ' secs'
        )

    
t1 = datetime.now()
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
print('⌚ Model tokenizer created', format_timedelta(datetime.now()-t1))
        
t1 = datetime.now()
model = GPTJForCausalLM.from_pretrained('./gpt-j-6B')
print('⌚ Model loaded (.from_pretrained)', format_timedelta(datetime.now()-t1))

t1 = datetime.now()

model.half().cuda()

print('⌚ Model half().cuda()', format_timedelta(datetime.now()-t1))

t1 = datetime.now()
prompt = "Hello my name is Paul and"
input_ids = tokenizer.encode(str(prompt), return_tensors='pt').cuda()


output = model.generate(
    input_ids,
    do_sample=True,
    max_length=100,
    temperature=0.8,
    top_k=0,
    top_p=0.7,
    eos_token_id=198,
    min_length=1,
    repetition_penalty=0.5
)
print('⌚ Test response time', format_timedelta(datetime.now() - t1))
print('🤖 Test response', tokenizer.decode(output[0], skip_special_tokens=True))

def eval(input):
    t1 = datetime.now()

    input_ids = tokenizer.encode(str(input.text), return_tensors='pt').cuda()
    token_count = input_ids.size(dim=1)
    if token_count + input.generate_tokens_limit > 2048:
        raise Exception(f"This model can't generate more then 2048 tokens, you passed {token_count} "+
            f"input tokens and requested to generate {input.generate_tokens_limit} tokens") 
    output = model.generate(
        input_ids,
        do_sample=True,
        max_length=token_count + input.generate_tokens_limit,
        top_p=input.top_p,
        top_k=input.top_k,
        temperature=input.temperature,
        eos_token_id=input.eos_token_id,
        min_length=input.min_length,
        repetition_penalty=input.repetition_penalty
    )
    resp = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'⌚ Response time {format_timedelta(datetime.now() - t1)} in len: { len(input.text) } resp len { len(resp) }')
    return resp

