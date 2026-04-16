import os
import argparse
from pipeline.pipeline_executor import InitiatePipeline
from global_variables import *
from huggingface_hub import HfApi
from qpsychometric import *


def main(base_dir, only_zero_shot=False):
    hf_api = HfApi()

    zero_shot_models = list(hf_api.list_models(filter="zero-shot-classification"))
    valid_pipeline_tags = {"zero-shot-classification", "text-classification"}

    if only_zero_shot:
        combined_models = zero_shot_models
    else:
        text_classification_models = list(hf_api.list_models(filter="text-classification"))
        combined_models = zero_shot_models + text_classification_models

    filtered_models = [m for m in combined_models if m.pipeline_tag in valid_pipeline_tags]
    unique_filtered_models = list({m.modelId: m for m in filtered_models}.values())

    all_qmnli_questionnaires = all_psychometrics['QMNLI'][['ASI', 'BIG5', 'CS', 'GAD7', 'PHQ9', 'SOC']].get_questions()

    InitiatePipeline(
        models_info=unique_filtered_models,
        questionnaires_questions_lists=all_qmnli_questionnaires,
        base_dir=base_dir,
        override_results=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch NLI evaluation of all HuggingFace zero-shot/text-classification models.")
    parser.add_argument('--base_dir', type=str, default='./model_logs',
                        help='Directory to save results (default: ./model_logs)')
    parser.add_argument('--only_zero_shot', action='store_true',
                        help='Restrict to zero-shot-classification models only')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace API token (optional, reduces rate limiting)')
    args = parser.parse_args()

    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token

    main(args.base_dir, args.only_zero_shot)
