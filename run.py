from torch import autocast
from diffusers import StableDiffusionPipeline
import yaml
import datetime
import os



def main():
    with open('conf.yaml') as f:
        conf_list = yaml.safe_load(f)
    now = datetime.datetime.now()
    result_dir = "./result/" + now.strftime("%Y-%m-%d-%H-%M-%S")

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to(device)
    for conf_id, conf in enumerate(conf_list):
        dir_name = os.path.join(result_dir, str(conf_id))
        os.makedirs(dir_name)
        prompt = conf["input"]
        batch_size = conf["batch_size"]
        with autocast("cuda"):
            samples = pipe([prompt] * batch_size, guidance_scale=7.5)["sample"]
        for sample_id, sample in enumerate(samples):
            file_name = os.path.join(dir_name, str(sample_id)+".png")
            sample.save(file_name)



if __name__ == "__main__":
    main()