import os.path

DIR_ROOT_DATALOADER = "C:/workspace/project/Cantonese_transformer/Data_loader/"
MODEL_SAVE_PATH = "C:/workspace/project/Cantonese_transformer/Model_repo/"

MODELS = [
    "hfl/chinese-roberta-wwm-ext"
]

CORPORA = [
    "openrice",
    "lihkg",
    "discusshk",
    "wiki",
    "hkgolden",
    "parallel",
]

DATA_SCRIPTS = [
    "corpusloader_openrice.py"
    "corpusloader_lihkg.py"
    "corpusloader_discusshk.py"
    "corpusloader_wiki.py"
    "corpusloader_hkgolden.py"
    "corpusloader_parallel.py"
]


class Config:
    def __init__(self):
        self.task = [
            "parallel"
        ]
        self.model_name = "hfl/chinese-roberta-wwm-ext"
        self.dataset_scripts = [f"corpusloader_{task}.py" for task in self.task]
        self.save_path_suffix = "_cl"

        self.contrastive = True
        self.r = 1
        # self.temperature = 0.05

        self.dataset_scripts = [os.path.join(DIR_ROOT_DATALOADER, script) for script in self.dataset_scripts]
        model_path = f"{self.model_name.split('/')[-1]}_{'-'.join(self.task)}"
        self.model_save_path = os.path.join(MODEL_SAVE_PATH, f"{model_path}_finetuned{self.save_path_suffix}")
        self.log_save_path = f"./log/{model_path}_mlm{self.save_path_suffix}"
        self.check_point = None

        self.chunk_size = 512
        self.mlm_probability = 0.15
        self.train_batch_size = 2  # batch size per device
        self.eval_batch_size = 2
        self.test_batch_size = 2

        # training
        self.learning_rate = 1e-4
        self.num_train_epochs = 100
        self.gradient_accumulation_steps = 2  # actual batch size = train_batch_size * num_of_device * this
        self.warmup_steps = 10000

        # evaluation
        self.eval_per_step = 30000  # model update step
        self.start_step = 1

        # save
        self.save_per_step = 30000

        self.mini_test = True
