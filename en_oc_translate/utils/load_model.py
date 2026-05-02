from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from configs.teacher_config import TeacherTrainingConfig
from configs.student_config import StudentTrainingConfig

def load_saved_model(model_name: str, saved_dir: str, config: TeacherTrainingConfig|StudentTrainingConfig):
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=config.bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(model, saved_dir)
    return model