import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()
from language_dict import language_dict

# language_dict = {
#     'en':{'full_name':'English','yes':'yes','no':'no','prompt_suffix':' Please select your answer from ({},{}). And make sure that your answer should not contain any other word than given answers.'},
#     'zh':{'full_name':'Chinese','yes':'是','no':'否','prompt_suffix':' 请从以下答案中选择您的答案:({},{})。并确保您的答案不包含任何其他单词。'},
#     'ja':{'full_name':'Japanese','yes':'はい','no':'いいえ','prompt_suffix':' 以下の答えからお選びください:({},{})。そして、あなたの答えには他の言葉が含まれていないことを確認してください。'},
#     'ko':{'full_name':'Korean','yes':'예','no':'아니요','prompt_suffix':' 다음 답변 중에서 선택하십시오:({},{})。그리고 당신의 답변에는 다른 단어가 포함되지 않도록 주의하십시오.'},
#     'es':{'full_name':'Spanish','yes':'sí','no':'no','prompt_suffix':' Por favor, seleccione su respuesta de entre ({},{}). Y asegúrese de que su respuesta no contenga ninguna otra palabra.'},
#     'fr':{'full_name':'French','yes':'oui','no':'non','prompt_suffix':' Veuillez sélectionner votre réponse parmi ({},{}). Et assurez-vous que votre réponse ne contient aucun autre mot.'},
#     'de':{'full_name':'German','yes':'ja','no':'nein','prompt_suffix':' Bitte wählen Sie Ihre Antwort aus ({},{}). Und stellen Sie sicher, dass Ihre Antwort kein anderes Wort enthält.'},
#     'it':{'full_name':'Italian','yes':'sì','no':'no','prompt_suffix':' Si prega di selezionare la risposta da ({},{}). E assicurarsi che la risposta non contenga altre parole.'},
#     'pt':{'full_name':'Portuguese','yes':'sim','no':'não','prompt_suffix':' Por favor, selecione sua resposta de ({},{}). E certifique-se de que sua resposta não contenha nenhuma outra palavra.'},
#     'ru':{'full_name':'Russian','yes':'да','no':'нет','prompt_suffix':' Пожалуйста, выберите ваш ответ из ({},{}). И убедитесь, что ваш ответ не содержит никаких других слов.'},
#     'ar':{'full_name':'Arabic','yes':'نعم','no':'لا','prompt_suffix':' يرجى اختيار إجابتك من ({},{}). وتأكد من أن إجابتك لا تحتوي على أي كلمة أخرى.'},
#     'tr':{'full_name':'Turkish','yes':'evet','no':'hayır','prompt_suffix':' Lütfen cevabınızı ({},{}). seçin. Ve cevabınızın başka bir kelime içermediğinden emin olun.'},
#     'vi':{'full_name':'Vietnamese','yes':'có','no':'không','prompt_suffix':' Vui lòng chọn câu trả lời của bạn từ ({},{}). Và đảm bảo rằng câu trả lời của bạn không chứa bất kỳ từ nào khác.'},
#     'th':{'full_name':'Thai','yes':'ใช่','no':'ไม่','prompt_suffix':' โปรดเลือกคำตอบของคุณจาก ({},{}). และตรวจสอบให้แน่ใจว่าคำตอบของคุณไม่มีคำอื่นๆ.'},
#     'id':{'full_name':'Indonesian','yes':'ya','no':'tidak','prompt_suffix':' Silakan pilih jawaban Anda dari ({},{}). Dan pastikan bahwa jawaban Anda tidak mengandung kata lain.'},
#     'ms':{'full_name':'Malay','yes':'ya','no':'tidak','prompt_suffix':' Sila pilih jawapan anda dari ({},{}). Dan pastikan bahawa jawapan anda tidak mengandungi sebarang perkataan lain.'},
# }

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w",encoding="utf8")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        prompt_suffix = language_dict[args.language]['prompt_suffix'].format(language_dict[args.language]['yes'],language_dict[args.language]['no'])
        conv.append_message(conv.roles[0], qs + prompt_suffix)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}},ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
